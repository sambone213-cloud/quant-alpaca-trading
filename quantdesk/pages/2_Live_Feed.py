"""
Live Dashboard — Multi-stock auto-updating feed with SMC filter
Auto-refreshes every N seconds. No clicking required.
"""

import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
from collections import deque
from datetime import datetime

from feed_manager import FeedManager, ContractConfig, PriceTick, ContractState
from alpaca_client import AlpacaClient as PolymarketClient, StockMarket

# ─── Theme ───────────────────────────────────────────────────────────────────

DARK_BG  = "#0d0f14"
PANEL_BG = "#13161e"
BORDER   = "#1e2330"
ACCENT   = "#00d4ff"
ACCENT2  = "#7b2fff"
GREEN    = "#00ff88"
RED      = "#ff4466"
YELLOW   = "#ffd166"
TEXT     = "#e2e8f0"
MUTED    = "#64748b"
ORANGE   = "#ff9f43"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "grid.color": BORDER, "grid.linewidth": 0.5,
    "legend.facecolor": PANEL_BG, "legend.edgecolor": BORDER,
    "font.family": "monospace",
})

st.set_page_config(page_title="Live Dashboard", page_icon="📡", layout="wide")

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Space+Grotesk:wght@400;600;700&display=swap');
  html, body, [class*="css"] {{ background:{DARK_BG}; color:{TEXT}; font-family:'Space Grotesk',sans-serif; }}
  .stApp {{ background:{DARK_BG}; }}
  [data-testid="stSidebar"] {{ background:{PANEL_BG}; border-right:1px solid {BORDER}; }}
  [data-testid="stSidebar"] * {{ color:{TEXT} !important; }}
  [data-testid="stMetric"] {{ background:{PANEL_BG}; border:1px solid {BORDER}; border-radius:8px; padding:12px !important; }}
  [data-testid="stMetricLabel"] {{ color:{MUTED} !important; font-size:10px !important; letter-spacing:1px; text-transform:uppercase; }}
  [data-testid="stMetricValue"] {{ color:{ACCENT} !important; font-family:'JetBrains Mono',monospace !important; font-size:20px !important; }}
  .stButton>button {{ background:linear-gradient(135deg,{ACCENT2},{ACCENT}); color:#000 !important; border:none; border-radius:6px; font-weight:600; font-family:'JetBrains Mono',monospace; padding:8px 18px; transition:opacity 0.2s; }}
  .stButton>button:hover {{ opacity:0.85; }}
  .stTabs [data-baseweb="tab-list"] {{ background:{PANEL_BG}; border-bottom:1px solid {BORDER}; gap:0; }}
  .stTabs [data-baseweb="tab"] {{ background:transparent; color:{MUTED}; font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:1px; text-transform:uppercase; border-bottom:2px solid transparent; padding:10px 16px; }}
  .stTabs [aria-selected="true"] {{ color:{ACCENT} !important; border-bottom:2px solid {ACCENT} !important; }}
  .stInfo  {{ background:rgba(0,212,255,0.07) !important; border-left:3px solid {ACCENT} !important; }}
  .stSuccess {{ background:rgba(0,255,136,0.07) !important; border-left:3px solid {GREEN} !important; }}
  .stWarning {{ background:rgba(255,209,102,0.07) !important; border-left:3px solid {YELLOW} !important; }}
  .stError {{ background:rgba(255,68,102,0.07) !important; border-left:3px solid {RED} !important; }}
  h1 {{ font-family:'Space Grotesk',sans-serif !important; font-weight:700;
        background:linear-gradient(90deg,{ACCENT},{ACCENT2}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  h3 {{ font-family:'JetBrains Mono',monospace !important; color:{ACCENT} !important; font-size:11px !important; letter-spacing:2px; text-transform:uppercase; }}
  .live-dot {{ display:inline-block; width:7px; height:7px; background:{GREEN}; border-radius:50%;
               animation:pulse 1.5s infinite; margin-right:5px; }}
  .alert-dot {{ display:inline-block; width:7px; height:7px; background:{RED}; border-radius:50%;
                animation:pulse 0.8s infinite; margin-right:5px; }}
  @keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.2;}} }}
  hr {{ border-color:{BORDER}; margin:12px 0; }}
  .contract-card {{
    background:{PANEL_BG}; border:1px solid {BORDER}; border-radius:10px;
    padding:16px; margin-bottom:12px; transition:border-color 0.3s;
  }}
  .contract-card:hover {{ border-color:{ACCENT}; }}
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────

def _init():
    if "fm" not in st.session_state:
        st.session_state.fm = None
    if "feed_running" not in st.session_state:
        st.session_state.feed_running = False
    if "tracked" not in st.session_state:
        st.session_state.tracked = {}   # contract_id -> {label, token_id, config}
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "search_done" not in st.session_state:
        st.session_state.search_done = False
    if "selected_contract" not in st.session_state:
        st.session_state.selected_contract = None
    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 5
    if "alert_flash" not in st.session_state:
        st.session_state.alert_flash = []

_init()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def price_color(p: float) -> str:
    if p > 0.65: return GREEN
    if p < 0.35: return RED
    return YELLOW

def fmt_cents(p: float) -> str:
    """Format price — for stocks shows as $XX.XX."""
    if p is None:
        return "—"
    if p > 1.0:
        return f"${p:,.2f}"
    return f"{p*100:.2f}¢"

def fmt_time(ts: float) -> str:
    return datetime.utcfromtimestamp(ts).strftime("%H:%M:%S UTC")

def get_fm() -> FeedManager:
    N   = int(st.session_state.get("N_particles", 3000))
    pv  = float(st.session_state.get("process_vol", 0.02))
    an  = bool(st.session_state.get("adaptive_noise", True))

    # Detect if filter config has changed since the FeedManager was created
    cfg_key = (N, pv, an)
    if st.session_state.fm is None or st.session_state.get("_fm_cfg") != cfg_key:
        was_running = st.session_state.feed_running and st.session_state.fm is not None
        if was_running:
            st.session_state.fm.stop()

        st.session_state.fm = FeedManager(
            N_particles=N,
            process_vol=pv,
            adaptive_noise=an,
        )
        st.session_state["_fm_cfg"] = cfg_key

        # Re-add and restart any already-tracked contracts
        if was_running:
            for cid, info in st.session_state.tracked.items():
                config = ContractConfig(
                    id=cid, label=info["label"], token_id=info["token_id"],
                    poll_interval=st.session_state.refresh_interval,
                    bankroll=info.get("bankroll", 1000),
                    kelly_fraction=info.get("kelly_fraction", 0.25),
                    alert_threshold=info.get("alert_threshold", 0.05),
                    alert_above=info.get("alert_above"),
                    alert_below=info.get("alert_below"),
                )
                st.session_state.fm.add_contract(config, prior=info.get("prior", 0.5))
            st.session_state.fm.start()

    return st.session_state.fm


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="padding:12px 0 16px 0;">
      <div style="font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:3px; color:{MUTED};">LIVE DASHBOARD</div>
      <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:17px;
                  background:linear-gradient(90deg,{ACCENT},{ACCENT2}); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        POLYMARKET FEED
      </div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:{GREEN}; margin-top:2px;">✓ No API key required</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<div style='font-size:9px; letter-spacing:2px; color:{MUTED}; margin-bottom:6px;'>AUTO-REFRESH</div>", unsafe_allow_html=True)
    st.session_state.refresh_interval = st.slider("Interval (seconds)", 3, 30, 5, key="refresh_slider")

    st.divider()
    st.markdown(f"<div style='font-size:9px; letter-spacing:2px; color:{MUTED}; margin-bottom:6px;'>FILTER CONFIG</div>", unsafe_allow_html=True)
    st.session_state.N_particles   = st.select_slider("Particles", [1000, 2500, 3000, 5000], value=3000, key="np_slider")
    st.session_state.process_vol   = st.slider("Process Vol σ", 0.005, 0.10, 0.02, 0.005, key="pv_slider")
    st.session_state.adaptive_noise = st.toggle("Adaptive Noise", True, key="an_toggle")

    st.divider()
    st.markdown(f"<div style='font-size:9px; letter-spacing:2px; color:{MUTED}; margin-bottom:6px;'>KELLY CONFIG</div>", unsafe_allow_html=True)
    default_bankroll = st.number_input("Default Bankroll ($)", 100, 100000, 1000, step=100, key="bankroll_input")
    default_kelly    = st.slider("Kelly Fraction", 0.05, 1.0, 0.25, 0.05, key="kelly_input")

    st.divider()
    # Feed status
    n_tracked = len(st.session_state.tracked)
    fm = st.session_state.fm
    total_ticks = sum(
        fm.get_state(cid).tick_count
        for cid in st.session_state.tracked
        if fm and fm.get_state(cid)
    ) if fm else 0

    if st.session_state.feed_running:
        st.markdown(f"""
        <div>
          <div><span class='live-dot'></span><span style='font-family:JetBrains Mono,monospace; font-size:11px; color:{GREEN};'>LIVE</span></div>
          <div style='font-family:JetBrains Mono,monospace; font-size:10px; color:{MUTED}; margin-top:4px;'>{n_tracked} contract{"s" if n_tracked!=1 else ""} · {total_ticks:,} ticks</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='font-family:JetBrains Mono,monospace; font-size:11px; color:{MUTED};'>● IDLE · {n_tracked} contracts queued</div>", unsafe_allow_html=True)

    # Global start/stop
    if not st.session_state.feed_running:
        if st.button("▶  START ALL FEEDS", use_container_width=True, key="global_start"):
            fm = get_fm()
            for cid, info in st.session_state.tracked.items():
                if fm.get_state(cid) is None:
                    config = ContractConfig(
                        id=cid, label=info["label"], token_id=info["token_id"],
                        poll_interval=st.session_state.refresh_interval,
                        bankroll=info.get("bankroll", default_bankroll),
                        kelly_fraction=info.get("kelly_fraction", default_kelly),
                        alert_threshold=info.get("alert_threshold", 0.05),
                        alert_above=info.get("alert_above"),
                        alert_below=info.get("alert_below"),
                    )
                    fm.add_contract(config, prior=info.get("prior", 0.5))
            fm.start()
            st.session_state.feed_running = True
            st.rerun()
    else:
        if st.button("⏹  STOP ALL FEEDS", use_container_width=True, key="global_stop"):
            if st.session_state.fm:
                st.session_state.fm.stop()
            st.session_state.feed_running = False
            st.rerun()


# ─── Main Area ────────────────────────────────────────────────────────────────

st.title("📡 Live Dashboard")
st.markdown(f"<p style='color:{MUTED}; font-size:13px; margin-top:-12px;'>Multi-stock live feed · SMC particle filter · Kelly sizing · Auto-updating</p>", unsafe_allow_html=True)

tab_dash, tab_search, tab_detail, tab_alerts = st.tabs([
    "DASHBOARD", "ADD CONTRACT", "CONTRACT DETAIL", "ALERTS"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD (auto-updating grid)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_dash:
    if not st.session_state.tracked:
        st.markdown(f"""
        <div style="text-align:center; padding:80px 20px; color:{MUTED};">
          <div style="font-size:48px; margin-bottom:16px;">📡</div>
          <div style="font-family:'Space Grotesk',sans-serif; font-size:20px; color:{TEXT}; margin-bottom:8px;">No contracts tracked yet</div>
          <div style="font-size:14px;">Go to <b>Add Stock</b> to search and add stocks to track</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        fm = get_fm()

        # ── Global alert banner ──────────────────────────────────────────
        all_alerts = fm.alert_log if fm else []
        recent_alerts = [a for a in all_alerts if time.time() - a["timestamp"] < 300]
        if recent_alerts:
            latest = recent_alerts[-1]
            label_short = st.session_state.tracked.get(latest["contract_id"], {}).get("label", latest["contract_id"])[:40]
            st.error(f"🔔 **{label_short}** — {latest['message']}  ·  {fmt_time(latest['timestamp'])}")

        # ── Contract cards grid ──────────────────────────────────────────
        contracts = list(st.session_state.tracked.items())
        n_cols = min(len(contracts), 3)
        rows = [contracts[i:i+n_cols] for i in range(0, len(contracts), n_cols)]

        for row in rows:
            cols = st.columns(len(row))
            for col, (cid, info) in zip(cols, row):
                state = fm.get_state(cid) if fm else None
                tick  = state.latest_tick if state else None
                label = info["label"]

                with col:
                    price    = tick.price if tick else None
                    estimate = tick.filter_estimate if tick else None
                    kelly    = tick.kelly_bet if tick else None
                    pcolor   = price_color(price) if price else MUTED
                    ticks_n  = state.tick_count if state else 0
                    has_alert = state and state.last_alert and (time.time() - state.last_alert_time) < 300

                    # Card header
                    alert_html = f"<span class='alert-dot'></span>" if has_alert else ""
                    status_dot = f"<span class='live-dot'></span>" if st.session_state.feed_running else ""
                    st.markdown(f"""
                    <div class='contract-card' style="border-left:3px solid {pcolor};">
                      <div style="font-family:'JetBrains Mono',monospace; font-size:9px; color:{MUTED}; letter-spacing:1px; margin-bottom:4px;">
                        {status_dot}{alert_html}STOCK
                      </div>
                      <div style="font-size:13px; color:{TEXT}; font-weight:600; margin-bottom:10px; line-height:1.3;">
                        {label[:55]}{"…" if len(label)>55 else ""}
                      </div>
                      <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:10px;">
                        <div>
                          <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:{MUTED};">MARKET PRICE</div>
                          <div style="font-family:'JetBrains Mono',monospace; font-size:22px; color:{pcolor}; font-weight:600;">
                            {fmt_cents(price) if price else "—"}
                          </div>
                        </div>
                        <div>
                          <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:{MUTED};">FILTER EST.</div>
                          <div style="font-family:'JetBrains Mono',monospace; font-size:22px; color:{ACCENT}; font-weight:600;">
                            {fmt_cents(estimate) if estimate else "—"}
                          </div>
                        </div>
                      </div>
                      <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; font-family:'JetBrains Mono',monospace; font-size:10px;">
                        <div>
                          <div style="color:{MUTED};">KELLY BET</div>
                          {f'<div style="color:{GREEN};">${kelly:,.0f}</div>' if kelly and kelly > 0 else f'<div style="color:{MUTED};">—</div>'}
                        </div>
                        <div>
                          <div style="color:{MUTED};">TICKS</div>
                          <div style="color:{TEXT};">{ticks_n:,}</div>
                        </div>
                        <div>
                          <div style="color:{MUTED};">SPREAD</div>
                          <div style="color:{TEXT};">{fmt_cents(tick.spread) if tick and tick.spread else "—"}</div>
                        </div>
                      </div>
                      {f'<div style="margin-top:8px; padding:6px 8px; background:rgba(255,68,102,0.1); border-radius:4px; font-family:JetBrains Mono,monospace; font-size:10px; color:{RED};">{state.last_alert}</div>' if has_alert else ""}
                    </div>
                    """, unsafe_allow_html=True)

                    # Mini sparkline
                    if state and len(state.prices) > 3:
                        prices_list = state.prices[-60:]
                        ests_list   = state.estimates[-60:]
                        fig, ax = plt.subplots(figsize=(3.5, 1.2))
                        fig.patch.set_alpha(0)
                        ax.set_facecolor(PANEL_BG)
                        ax.plot(prices_list, color=MUTED, linewidth=0.8, alpha=0.5)
                        if ests_list:
                            ax.plot(ests_list, color=pcolor, linewidth=1.5)
                        ax.set_ylim(0, 1)
                        ax.axis("off")
                        fig.tight_layout(pad=0)
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

                    # Detail / Remove buttons
                    b1, b2 = st.columns(2)
                    if b1.button("Detail", key=f"detail_{cid[:8]}", use_container_width=True):
                        st.session_state.selected_contract = cid
                        st.rerun()
                    if b2.button("Remove", key=f"remove_{cid[:8]}", use_container_width=True):
                        st.session_state.tracked.pop(cid, None)
                        if fm:
                            fm.remove_contract(cid)
                        st.rerun()

        # ── Auto-refresh ─────────────────────────────────────────────────
        if st.session_state.feed_running and st.session_state.tracked:
            st.divider()
            interval = st.session_state.refresh_interval
            st.markdown(f"""
            <div style="text-align:center; font-family:'JetBrains Mono',monospace; font-size:10px; color:{MUTED};">
              <span class='live-dot'></span>AUTO-REFRESHING EVERY {interval}s
              · {datetime.now().strftime("%H:%M:%S")}
            </div>
            """, unsafe_allow_html=True)
            time.sleep(interval)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ADD CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.markdown("### Add Stocks")

    # ── Top Markets button + search box ──────────────────────────────────
    col_top, col_q, col_btn = st.columns([1.2, 4, 1])

    with col_top:
        st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
        if st.button("🔥 Top Stocks", use_container_width=True, key="top_markets"):
            st.session_state["search_q"] = ""
            st.session_state["_do_search"] = True

    query = col_q.text_input("", placeholder="Search ticker or company (e.g. AAPL, Tesla, energy ETF)…",
                              label_visibility="collapsed", key="search_q")
    with col_btn:
        st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
        if st.button("Search", use_container_width=True, key="do_search"):
            st.session_state["_do_search"] = True

    do_search = st.session_state.pop("_do_search", False)

    if do_search:
        label = f"'{query}'" if query else "top stocks by volume"
        with st.spinner(f"Fetching {label}…"):
            client = PolymarketClient()
            results = client.search_markets(
                query=query,
                limit=50,
                min_volume=0,
            )
            results.sort(key=lambda m: m.volume_24h, reverse=True)
            st.session_state.search_results = results
            st.session_state.search_done = True

    if st.session_state.search_done and not st.session_state.search_results:
        st.warning("No results found. Try a different ticker or company name.")

    if st.session_state.search_results:
        st.markdown(f"<div style='font-family:JetBrains Mono,monospace; font-size:9px; color:{MUTED}; margin:10px 0 6px; letter-spacing:2px;'>{len(st.session_state.search_results)} RESULTS · SORTED BY 24H VOLUME</div>", unsafe_allow_html=True)

        for m in st.session_state.search_results:
            already = m.condition_id in st.session_state.tracked
            pc = price_color(m.yes_price)

            col_info, col_act = st.columns([5, 1])
            with col_info:
                bar = int(m.yes_price * 100)
                st.markdown(f"""
                <div style="background:{PANEL_BG}; border:1px solid {BORDER}; border-left:3px solid {pc};
                            border-radius:8px; padding:10px 14px; margin-bottom:5px;">
                  <div style="font-size:13px; color:{TEXT}; font-weight:500; margin-bottom:5px;">{m.question[:85]}</div>
                  <div style="display:flex; gap:16px; flex-wrap:wrap; align-items:center;">
                    <span style="font-family:'JetBrains Mono',monospace; font-size:13px; color:{pc}; font-weight:600;">${m.price:,.2f}</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:11px; color:{MUTED};">spread ${m.spread:.2f}</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:11px; color:{MUTED};">vol {m.volume_24h:,.0f} shares</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:11px; color:{MUTED};">liq ${m.liquidity:,.0f}</span>
                  </div>
                  <div style="background:{BORDER}; border-radius:3px; height:3px; margin-top:7px;">
                    <div style="background:{pc}; width:{bar}%; height:100%; border-radius:3px; opacity:0.5;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with col_act:
                st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
                if already:
                    st.markdown(f"<div style='font-family:JetBrains Mono,monospace; font-size:10px; color:{GREEN}; text-align:center; margin-top:8px;'>✓ Tracking</div>", unsafe_allow_html=True)
                else:
                    if st.button("+ Add", key=f"add_{m.condition_id[:10]}", use_container_width=True):
                        st.session_state.tracked[m.condition_id] = {
                            "label": m.question,
                            "token_id": m.yes_token_id,
                            "bankroll": default_bankroll,
                            "kelly_fraction": default_kelly,
                            "alert_threshold": 0.05,
                            "alert_above": None,
                            "alert_below": None,
                            "prior": m.yes_price,
                        }
                        fm = get_fm()
                        if st.session_state.feed_running:
                            config = ContractConfig(
                                id=m.condition_id, label=m.question,
                                token_id=m.yes_token_id,
                                poll_interval=st.session_state.refresh_interval,
                                bankroll=default_bankroll,
                                kelly_fraction=default_kelly,
                                alert_threshold=0.05,
                            )
                            fm.add_contract(config, prior=m.yes_price)
                        st.success("Added! Go to Dashboard tab.")
                        st.rerun()

    # ── Manual entry (expanded by default now — primary workflow) ─────────
    st.divider()
    st.markdown("### Manual Entry — Enter Ticker")
    st.markdown(f"<p style='color:{MUTED}; font-size:12px; margin-top:-8px;'>Enter a ticker symbol directly. Use standard US stock tickers (e.g. AAPL, SPY, NVDA).</p>", unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    m_label = mc1.text_input("Label", placeholder="e.g. BTC above 100k", key="m_label")
    m_token = mc2.text_input("YES Token ID", placeholder="Numeric or 0x token ID", key="m_token")
    m_prior = mc3.slider("Prior estimate", 0.01, 0.99, 0.5, key="m_prior")
    if st.button("➕  Add Contract", key="add_manual", use_container_width=True):
        if m_token:
            st.session_state.tracked[m_token] = {
                "label": m_label or m_token[:30],
                "token_id": m_token,
                "bankroll": default_bankroll,
                "kelly_fraction": default_kelly,
                "alert_threshold": 0.05,
                "alert_above": None,
                "alert_below": None,
                "prior": m_prior,
            }
            fm = get_fm()
            if st.session_state.feed_running:
                config = ContractConfig(
                    id=m_token, label=m_label or m_token[:30],
                    token_id=m_token,
                    poll_interval=st.session_state.refresh_interval,
                    bankroll=default_bankroll,
                    kelly_fraction=default_kelly,
                )
                fm.add_contract(config, prior=m_prior)
            st.success("Added! Go to Dashboard tab.")
            st.rerun()
        else:
            st.error("Please enter a token ID.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONTRACT DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

with tab_detail:
    tracked_ids = list(st.session_state.tracked.keys())
    if not tracked_ids:
        st.info("Add contracts in the **Add Contract** tab first.")
    else:
        # Contract selector
        labels = {cid: st.session_state.tracked[cid]["label"][:60] for cid in tracked_ids}
        sel_default = st.session_state.selected_contract if st.session_state.selected_contract in tracked_ids else tracked_ids[0]
        sel_cid = st.selectbox(
            "Select contract",
            tracked_ids,
            format_func=lambda x: labels.get(x, x),
            index=tracked_ids.index(sel_default) if sel_default in tracked_ids else 0,
            key="detail_selector"
        )
        st.session_state.selected_contract = sel_cid
        info  = st.session_state.tracked.get(sel_cid, {})
        fm    = get_fm()
        state = fm.get_state(sel_cid)

        if state is None or not state.ticks:
            feed_running = st.session_state.feed_running
            token_id = info.get("token_id", "unknown")

            if not feed_running:
                st.warning("⏸ Feed is not running. Press **START ALL FEEDS** in the sidebar.")
            else:
                st.info(f"⏳ Waiting for first tick… Feed is running. This usually takes up to {int(st.session_state.refresh_interval) * 2}s.")

            # Try a live diagnostic fetch right now so the user doesn't have to wait
            with st.spinner("Testing connection to Alpaca for this stock…"):
                try:
                    from alpaca_client import AlpacaClient as PolymarketClient, StockMarket
                    _client = PolymarketClient()
                    _ob = _client.get_order_book(token_id)
                    _mid = _client.get_midpoint(token_id)

                    if _ob and _ob.best_bid > 0:
                        st.success(f"✅ Order book reachable — mid: {_ob.mid*100:.2f}¢  bid: {_ob.best_bid*100:.2f}¢  ask: {_ob.best_ask*100:.2f}¢")
                        st.info("Data is coming in — refresh in a few seconds or wait for auto-refresh.")
                    elif _mid:
                        st.success(f"✅ Midpoint reachable: {_mid*100:.2f}¢  (no order book depth)")
                        st.info("Data is coming in — refresh in a few seconds.")
                    else:
                        st.error(
                            f"❌ Could not fetch price for token ID:\n\n`{token_id}`\n\n"
                            "This usually means the token ID is wrong or the market has no active order book. "
                            "Try re-adding the contract from the **Add Contract** tab."
                        )
                except Exception as e:
                    st.error(f"❌ Connection error: {e}")
        else:
            tick    = state.latest_tick
            prices  = state.prices
            ests    = state.estimates
            lowers  = state.lowers
            uppers  = state.uppers
            ob      = state.order_book

            # ── KPI row ───────────────────────────────────────────────
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            pc = price_color(tick.price)
            k1.metric("Stock Price",      fmt_cents(tick.price))
            k2.metric("Filter Estimate",  fmt_cents(tick.filter_estimate) if tick.filter_estimate else "—")
            k3.metric("95% CI",           f"{fmt_cents(tick.filter_lower)}–{fmt_cents(tick.filter_upper)}" if tick.filter_lower else "—")
            k4.metric("Kelly Bet",        f"${tick.kelly_bet:,.0f}" if tick.kelly_bet and tick.kelly_bet > 0 else "$0")
            k5.metric("Bid-Ask Spread",   fmt_cents(tick.spread) if tick.spread else "—")
            k6.metric("Ticks",            f"{state.tick_count:,}")

            # ── Buy / Disregard Signal ────────────────────────────────
            st.divider()
            if tick.filter_estimate and tick.price:
                edge        = tick.filter_estimate - tick.price
                abs_edge    = abs(edge)
                ci_width    = (tick.filter_upper - tick.filter_lower) if tick.filter_lower and tick.filter_upper else 1.0
                kelly       = tick.kelly_bet or 0
                n_ticks     = state.tick_count
                imbalance   = tick.depth_imbalance or 0.5
                spread_val  = tick.spread or 0

                # Score each dimension
                edge_ok        = abs_edge >= (0.005 * tick.price) if tick.price > 1 else abs_edge >= 0.04  # 0.5% edge for stocks
                ci_ok          = ci_width <= (0.10 * tick.price) if tick.price > 1 else ci_width <= 0.25   # 10% of price
                kelly_ok       = kelly > 0
                ticks_ok       = n_ticks >= 10
                spread_ok      = spread_val <= (0.002 * tick.price) if tick.price > 1 else spread_val <= 0.04  # 0.2% spread
                imbalance_ok   = (edge > 0 and imbalance >= 0.50) or \
                                 (edge < 0 and imbalance <= 0.50)
                direction_yes  = edge > 0

                signals_passed = sum([edge_ok, ci_ok, kelly_ok, ticks_ok, spread_ok, imbalance_ok])
                total_signals  = 6

                if signals_passed >= 5 and edge_ok and kelly_ok:
                    verdict       = "BUY YES" if direction_yes else "BUY NO"
                    verdict_color = GREEN
                    verdict_icon  = "🟢"
                    verdict_bg    = "rgba(0,255,136,0.07)"
                    verdict_border = GREEN
                elif signals_passed >= 3 and edge_ok:
                    verdict       = "WEAK EDGE — USE CAUTION"
                    verdict_color = YELLOW
                    verdict_icon  = "🟡"
                    verdict_bg    = "rgba(255,209,102,0.07)"
                    verdict_border = YELLOW
                else:
                    verdict       = "DISREGARD"
                    verdict_color = RED
                    verdict_icon  = "🔴"
                    verdict_bg    = "rgba(255,68,102,0.07)"
                    verdict_border = RED

                def check_html(ok, label, detail):
                    color = GREEN if ok else RED
                    icon  = "✓" if ok else "✗"
                    return f"<div style='margin:3px 0;'><span style='color:{color}; font-family:JetBrains Mono,monospace;'>{icon}</span> <span style='color:{TEXT}; font-size:12px;'>{label}</span> <span style='color:{MUTED}; font-size:11px;'>{detail}</span></div>"

                checks_html = "".join([
                    check_html(edge_ok,       "Edge",              f"${edge:+.3f}  (need ≥0.5% of price)"),
                    check_html(ci_ok,         "Filter confidence", f"CI width ${ci_width:.3f}  (need ≤10% of price)"),
                    check_html(kelly_ok,      "Kelly bet",         f"${kelly:,.0f}  (need >$0)"),
                    check_html(ticks_ok,      "Data quality",      f"{n_ticks} ticks  (need ≥10)"),
                    check_html(spread_ok,     "Spread",            f"${spread_val:.3f}  (need ≤0.2% of price)"),
                    check_html(imbalance_ok,  "Order flow",        f"imbalance {imbalance:.2f}  ({'buy' if imbalance>0.5 else 'sell'} pressure)"),
                ])

                st.markdown(f"""
                <div style="background:{verdict_bg}; border:1px solid {verdict_border}; border-radius:10px; padding:16px 20px;">
                  <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
                    <div>
                      <span style="font-family:'JetBrains Mono',monospace; font-size:10px; color:{MUTED}; letter-spacing:2px;">SIGNAL</span><br>
                      <span style="font-family:'Space Grotesk',sans-serif; font-size:24px; font-weight:700; color:{verdict_color};">{verdict_icon} {verdict}</span>
                    </div>
                    <div style="text-align:right;">
                      <span style="font-family:'JetBrains Mono',monospace; font-size:10px; color:{MUTED}; letter-spacing:2px;">CHECKS PASSED</span><br>
                      <span style="font-family:'JetBrains Mono',monospace; font-size:28px; font-weight:700; color:{verdict_color};">{signals_passed}/{total_signals}</span>
                    </div>
                  </div>
                  {checks_html}
                  <div style="margin-top:10px; font-family:'JetBrains Mono',monospace; font-size:10px; color:{MUTED};">
                    ⚠ This is a mathematical signal only — not financial advice. Always consider event risk, liquidity, and position sizing.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Waiting for enough data to generate a signal…")

            # ── Alert config ──────────────────────────────────────────
            with st.expander("⚙  Alert Configuration", expanded=False):
                ac1, ac2, ac3 = st.columns(3)
                new_threshold = ac1.slider("Move Alert Threshold", 0.01, 0.20, info.get("alert_threshold", 0.05), 0.01, key=f"thr_{sel_cid[:8]}")
                new_above     = ac2.slider("Alert if price crosses ABOVE", 0.0, 1.0, info.get("alert_above") or 0.0, 0.01, key=f"ab_{sel_cid[:8]}")
                new_below     = ac3.slider("Alert if price crosses BELOW", 0.0, 1.0, info.get("alert_below") or 0.0, 0.01, key=f"bl_{sel_cid[:8]}")

                if st.button("Save Alert Config", key=f"save_alert_{sel_cid[:8]}"):
                    st.session_state.tracked[sel_cid]["alert_threshold"] = new_threshold
                    st.session_state.tracked[sel_cid]["alert_above"]     = new_above if new_above > 0 else None
                    st.session_state.tracked[sel_cid]["alert_below"]     = new_below if new_below > 0 else None
                    if state:
                        state.config.alert_threshold = new_threshold
                        state.config.alert_above     = new_above if new_above > 0 else None
                        state.config.alert_below     = new_below if new_below > 0 else None
                    st.success("Alert config saved.")

            # ── Charts ───────────────────────────────────────────────
            st.divider()
            chart_t, ob_t, kelly_t, hist_t = st.tabs(["PRICE & FILTER", "ORDER BOOK", "KELLY CURVE", "HISTORY"])

            with chart_t:
                if len(prices) >= 2:
                    t_ax = range(len(prices))
                    fig, ax = plt.subplots(figsize=(13, 5))
                    ax.plot(t_ax, prices, color=MUTED, alpha=0.4, linewidth=0.9, label="Stock Price")
                    if ests:
                        ax.plot(range(len(ests)), ests, color=ACCENT, linewidth=2.0, label="SMC Filter Estimate")
                    if lowers and uppers:
                        n = min(len(lowers), len(uppers))
                        ax.fill_between(range(n), lowers[:n], uppers[:n], color=ACCENT, alpha=0.12, label="95% Credible Interval")
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Probability")
                    ax.set_xlabel(f"Ticks (every {info.get('poll_interval', 5):.0f}s)")
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                    label_short = info["label"][:65]
                    ax.set_title(label_short, color=TEXT, fontsize=11)
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Waiting for more ticks…")

            with ob_t:
                if ob:
                    bids = ob.bids[:8]
                    asks = ob.asks[:8]
                    fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(12, 5))
                    if bids:
                        bp = [b[0] for b in bids]
                        bs = [b[1] for b in bids]
                        ax_b.barh(range(len(bp)), bs, color=GREEN, alpha=0.75)
                        ax_b.set_yticks(range(len(bp)))
                        ax_b.set_yticklabels([f"{p*100:.1f}¢" for p in bp], fontsize=9)
                        ax_b.set_title("BIDS", color=GREEN)
                        ax_b.invert_xaxis()
                        ax_b.grid(True, alpha=0.3, axis="x")
                        ax_b.set_xlabel("Size (USDC)")
                    if asks:
                        ap = [a[0] for a in asks]
                        as_ = [a[1] for a in asks]
                        ax_a.barh(range(len(ap)), as_, color=RED, alpha=0.75)
                        ax_a.set_yticks(range(len(ap)))
                        ax_a.set_yticklabels([f"{p*100:.1f}¢" for p in ap], fontsize=9)
                        ax_a.set_title("ASKS", color=RED)
                        ax_a.grid(True, alpha=0.3, axis="x")
                        ax_a.set_xlabel("Size (USDC)")
                    fig.suptitle("Order Book — YES Side", color=TEXT)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    imb = ob.depth_imbalance
                    sig = "🟢 Buy pressure" if imb > 0.55 else "🔴 Sell pressure" if imb < 0.45 else "⚪ Balanced"
                    st.markdown(f"<span style='font-family:JetBrains Mono,monospace; font-size:12px;'>Mid: <b>{ob.mid*100:.2f}¢</b> · Imbalance: <b>{imb:.3f}</b> · {sig}</span>", unsafe_allow_html=True)
                else:
                    st.info("Order book data not yet received.")

            with kelly_t:
                if tick.filter_estimate and tick.bid:
                    from simulators import kelly_fraction_sweep, compute_cvar
                    fracs, growths = kelly_fraction_sweep(tick.filter_estimate, tick.bid, n_points=100)
                    var, cvar = compute_cvar(tick.filter_estimate, tick.bid, info.get("bankroll", 1000), N=20000)

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(fracs, growths, color=ACCENT, linewidth=2)
                    ax.fill_between(fracs, 0, growths, where=(np.array(growths) > 0), color=GREEN, alpha=0.08)
                    ax.fill_between(fracs, growths, 0, where=(np.array(growths) < 0), color=RED, alpha=0.10)
                    ax.axhline(0, color=MUTED, linewidth=0.8)
                    full_k = info.get("kelly_fraction", 0.25)
                    ax.axvline(full_k, color=ACCENT2, linestyle="--", linewidth=1.5, label=f"Your fraction ({full_k:.0%})")
                    ax.set_xlabel("Kelly Fraction f")
                    ax.set_ylabel("Expected Log-Growth G(f)")
                    ax.set_title("Kelly Growth Curve", color=TEXT)
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Filter Edge", f"{(tick.filter_estimate - tick.bid)*100:+.2f}¢")
                    r2.metric(f"VaR (5%)", f"${var:,.2f}")
                    r3.metric(f"CVaR (5%)", f"${cvar:,.2f}")
                else:
                    st.info("Need at least one tick with bid price for Kelly curve.")

            with hist_t:
                history = fm.load_history(sel_cid, limit=2000)
                if history:
                    import pandas as pd
                    ts_v = [h[0] for h in history]
                    pr_v = [h[1] for h in history]
                    fe_v = [h[2] for h in history if h[2] is not None]
                    dt_s = [datetime.utcfromtimestamp(t).strftime("%H:%M:%S UTC") for t in ts_v]

                    # Warn if data is stale (newest record older than 10 minutes)
                    age_seconds = time.time() - ts_v[-1] if ts_v else 0
                    if age_seconds > 600:
                        age_h = int(age_seconds // 3600)
                        age_m = int((age_seconds % 3600) // 60)
                        st.warning(
                            f"⚠️ This history is from a previous session — newest record is "
                            f"**{age_h}h {age_m}m old**. Live ticks will appear here once "
                            f"the feed has been running for a few minutes.",
                            icon=None,
                        )
                    else:
                        st.info(f"Showing persisted DB history · {len(pr_v):,} records · newest: {dt_s[-1]}")

                    fig, ax = plt.subplots(figsize=(13, 4))
                    ax.plot(range(len(pr_v)), pr_v, color=ACCENT2, linewidth=1.2, alpha=0.8, label="Price")
                    if fe_v:
                        ax.plot(range(len(fe_v)), fe_v, color=ACCENT, linewidth=1.5, alpha=0.9, label="Filter")
                    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.8)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("YES Price")
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                    step = max(1, len(dt_s)//8)
                    ax.set_xticks(range(0, len(dt_s), step))
                    ax.set_xticklabels(dt_s[::step], rotation=30, fontsize=8)
                    st.pyplot(fig)
                    plt.close()

                    arr = np.array(pr_v)
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Records", f"{len(pr_v):,}")
                    s2.metric("Mean", fmt_cents(arr.mean()))
                    s3.metric("Std Dev", fmt_cents(arr.std()))
                    s4.metric("Range", fmt_cents(arr.max()-arr.min()))

                    df = pd.DataFrame({"timestamp": ts_v, "datetime": dt_s, "price": pr_v})
                    safe = info["label"][:25].replace(" ","_").replace("/","-")
                    st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(),
                                       file_name=f"polymarket_{safe}.csv", mime="text/csv")
                else:
                    st.info("No logged history yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_alerts:
    st.markdown("### Alert Log")
    fm = get_fm()
    all_alerts = fm.alert_log if fm else []

    if not all_alerts:
        st.markdown(f"""
        <div style="text-align:center; padding:40px 20px; color:{MUTED};">
          <div style="font-size:32px; margin-bottom:12px;">🔔</div>
          <div style="font-size:14px;">No alerts fired yet. Alerts trigger when:<br>
            · Price moves more than the threshold in a single tick<br>
            · Price crosses above/below a level you set
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for alert in reversed(all_alerts[-50:]):
            cid   = alert["contract_id"]
            label = st.session_state.tracked.get(cid, {}).get("label", cid)[:55]
            ts    = fmt_time(alert["timestamp"])
            msg   = alert["message"]
            age   = time.time() - alert["timestamp"]
            is_new = age < 300
            border_c = RED if is_new else BORDER

            st.markdown(f"""
            <div style="background:{PANEL_BG}; border:1px solid {border_c}; border-left:3px solid {border_c};
                        border-radius:8px; padding:10px 14px; margin-bottom:6px;">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                  <div style="font-size:13px; color:{TEXT}; margin-bottom:3px;">{msg}</div>
                  <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:{MUTED};">{label}</div>
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:{MUTED}; white-space:nowrap; margin-left:12px;">
                  {ts}{"  <span style='color:" + RED + ";'>NEW</span>" if is_new else ""}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # DB alerts
    st.divider()
    st.markdown("### Persisted Alerts (Database)")
    db_alerts = fm.load_alerts(limit=50) if fm else []
    if db_alerts:
        import pandas as pd
        df_alerts = pd.DataFrame(db_alerts)
        df_alerts["datetime"] = df_alerts["timestamp"].apply(fmt_time)
        df_alerts["contract"] = df_alerts["contract_id"].apply(
            lambda x: st.session_state.tracked.get(x, {}).get("label", x)[:40]
        )
        st.dataframe(
            df_alerts[["datetime", "contract", "message"]].rename(columns={"datetime": "Time", "contract": "Contract", "message": "Alert"}),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No persisted alerts yet.")


# ─── Footer ──────────────────────────────────────────────────────────────────

st.divider()
n_t = len(st.session_state.tracked)
st.markdown(f"""
<div style="text-align:center; font-family:'JetBrains Mono',monospace; font-size:10px; color:{MUTED};">
  QUANT PM STACK · POLYMARKET LIVE FEED · {n_t} CONTRACT{"S" if n_t!=1 else ""} TRACKED
  {f"· <span style='color:{GREEN};'>LIVE</span>" if st.session_state.feed_running else ""}
</div>
""", unsafe_allow_html=True)
