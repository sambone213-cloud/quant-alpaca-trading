"""
Match Simulation to Contract — calibrate ABM parameters to mirror a real Polymarket contract.
"""

import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timezone

from feed_manager import FeedManager
from market import PredictionMarketABM
from filters import PredictionMarketParticleFilter
from simulators import calculate_kelly_bet, kelly_fraction_sweep, compute_cvar

# ─── Theme ────────────────────────────────────────────────────────────────────

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

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "grid.color": BORDER, "grid.linewidth": 0.5,
    "legend.facecolor": PANEL_BG, "legend.edgecolor": BORDER,
    "font.family": "monospace",
})

st.set_page_config(page_title="Match Simulation", page_icon="🎯", layout="wide")
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Space+Grotesk:wght@400;600;700&display=swap');
  html, body, [class*="css"] {{ background:{DARK_BG}; color:{TEXT}; font-family:'Space Grotesk',sans-serif; }}
  .stApp {{ background:{DARK_BG}; }}
  [data-testid="stSidebar"] {{ background:{PANEL_BG}; border-right:1px solid {BORDER}; }}
  [data-testid="stSidebar"] * {{ color:{TEXT} !important; }}
  [data-testid="stMetric"] {{ background:{PANEL_BG}; border:1px solid {BORDER}; border-radius:8px; padding:12px !important; }}
  [data-testid="stMetricValue"] {{ color:{ACCENT} !important; font-family:'JetBrains Mono',monospace !important; }}
  .stButton>button {{ background:linear-gradient(135deg,{ACCENT2},{ACCENT}); color:#000 !important; border:none;
    border-radius:6px; font-weight:600; font-family:'JetBrains Mono',monospace; padding:8px 18px; }}
  .stTabs [data-baseweb="tab-list"] {{ background:{PANEL_BG}; border-bottom:1px solid {BORDER}; }}
  .stTabs [data-baseweb="tab"] {{ background:transparent; color:{MUTED}; font-family:'JetBrains Mono',monospace;
    font-size:10px; letter-spacing:1px; text-transform:uppercase; border-bottom:2px solid transparent; padding:10px 16px; }}
  .stTabs [aria-selected="true"] {{ color:{ACCENT} !important; border-bottom:2px solid {ACCENT} !important; }}
  h1 {{ font-family:'Space Grotesk',sans-serif !important; font-weight:700;
    background:linear-gradient(90deg,{ACCENT},{ACCENT2}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  h3 {{ font-family:'JetBrains Mono',monospace !important; color:{ACCENT} !important;
    font-size:11px !important; letter-spacing:2px; text-transform:uppercase; }}
  hr {{ border-color:{BORDER}; }}
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🎯 Match Simulation to Contract")
st.markdown(f"<p style='color:{MUTED}; font-size:13px; margin-top:-12px;'>Calibrate the ABM simulator to mirror a real Polymarket contract — then test your filter settings before going live.</p>", unsafe_allow_html=True)

with st.expander("📖  HOW TO USE THIS PAGE", expanded=False):
    st.markdown(f"""
<div style="font-size:13px; line-height:1.9; color:{TEXT};">

**The goal:** Make the simulator behave like the real contract you're tracking, so you can test your particle filter settings in a controlled environment where you *know* the true probability.

**Step 1 — Pull from live feed**
Select a contract you're already tracking. This page will read its current price, spread, volatility and imbalance directly and suggest matching simulator parameters.

**Step 2 — Tune the parameters**
The auto-suggested values are starting points. Adjust them based on what you know about the contract:
- High-news contract (election night, Fed decision)? → Increase **Jump Size** and **Hawkes α**
- Stable slow-moving contract? → Decrease **Process Volatility**, start in **Calm** regime
- Thin liquidity / wide spreads? → Decrease **Market Makers**, increase **Noise Agents**
- Strong informed trading (price moves fast toward truth)? → Increase **Informed Agents**

**Step 3 — Run and compare**
After running, compare the simulator's price chart to the contract's real price chart. If they look similar in terms of volatility and jumpiness, your filter settings are well-calibrated.

**Step 4 — Transfer settings**
Use the recommended filter settings shown at the bottom to configure the live feed sidebar (**Particles**, **Process Vol**, **Adaptive Noise**).

**What the simulation tells you:**
- Whether your filter can track this type of contract reliably
- What Kelly bet size is mathematically justified given the current edge
- Whether the edge is real or just noise (watch the credible interval width)

</div>
    """, unsafe_allow_html=True)


# ─── Step 1: Contract Source ──────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Step 1 — Contract Source")

source = st.radio("", ["Pull from live feed", "Enter manually"],
                  horizontal=True, label_visibility="collapsed")

contract_data = {}

if source == "Pull from live feed":
    fm = st.session_state.get("fm")
    tracked = st.session_state.get("tracked", {})

    if not tracked:
        st.warning("No contracts tracked yet. Add some in the Live Dashboard first, or use manual entry.")
    else:
        labels = {cid: info["label"][:70] for cid, info in tracked.items()}
        sel_cid = st.selectbox("Select tracked contract", list(labels.keys()),
                                format_func=lambda x: labels[x], key="sim_match_sel")
        info  = tracked.get(sel_cid, {})
        state = fm.get_state(sel_cid) if fm else None
        tick  = state.latest_tick if state and state.ticks else None

        if tick:
            price     = tick.price
            spread    = tick.spread or 0.02
            imbalance = tick.depth_imbalance or 0.5
            n_ticks   = state.tick_count

            # Estimate realized volatility from recent prices
            prices_arr = np.array(state.prices[-200:]) if len(state.prices) >= 5 else np.array([price])
            if len(prices_arr) >= 5:
                log_rets = np.diff(np.log(np.clip(prices_arr, 0.01, 0.99)))
                realized_vol = float(np.std(log_rets))
            else:
                realized_vol = 0.02

            # Estimate days remaining
            end_date_str = info.get("end_date", "")
            days_remaining = None
            if end_date_str:
                try:
                    end_dt = datetime.strptime(end_date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    days_remaining = max(1, (end_dt - datetime.now(timezone.utc)).days)
                except Exception:
                    pass

            st.markdown(f"""
            <div style="background:{PANEL_BG}; border:1px solid {BORDER}; border-radius:8px; padding:14px 18px; margin:10px 0;">
              <div style="font-size:13px; color:{TEXT}; font-weight:600; margin-bottom:8px;">{info['label'][:80]}</div>
              <div style="display:flex; gap:20px; flex-wrap:wrap; font-family:'JetBrains Mono',monospace; font-size:11px; color:{MUTED};">
                <span>Price: <b style="color:{ACCENT};">{price*100:.1f}¢</b></span>
                <span>Spread: <b style="color:{TEXT};">{spread*100:.1f}¢</b></span>
                <span>Imbalance: <b style="color:{TEXT};">{imbalance:.2f}</b></span>
                <span>Realized vol: <b style="color:{TEXT};">{realized_vol:.4f}</b></span>
                <span>Ticks: <b style="color:{TEXT};">{n_ticks}</b></span>
                {f'<span>Days left: <b style="color:{YELLOW};">{days_remaining}</b></span>' if days_remaining else ""}
              </div>
            </div>
            """, unsafe_allow_html=True)

            contract_data = {
                "price": price,
                "spread": spread,
                "imbalance": imbalance,
                "realized_vol": realized_vol,
                "days_remaining": days_remaining,
                "label": info["label"],
                "filter_estimate": tick.filter_estimate or price,
                "filter_lower": tick.filter_lower,
                "filter_upper": tick.filter_upper,
                "kelly_bet": tick.kelly_bet or 0,
                "bankroll": info.get("bankroll", 1000),
                "kelly_fraction": info.get("kelly_fraction", 0.25),
            }
        else:
            st.info("No ticks yet for this contract. Start the feed and wait for data, then come back.")

else:  # Manual entry
    mc1, mc2, mc3, mc4 = st.columns(4)
    contract_data = {
        "price":          mc1.slider("Current Market Price", 0.01, 0.99, 0.50, help="The YES price on Polymarket (0–1)"),
        "spread":         mc2.slider("Bid-Ask Spread", 0.001, 0.10, 0.02, 0.001, help="Ask minus bid. Wide = illiquid."),
        "imbalance":      mc3.slider("Order Book Imbalance", 0.0, 1.0, 0.5, 0.01, help=">0.5 = more bids than asks (buy pressure). <0.5 = sell pressure."),
        "realized_vol":   mc4.slider("Observed Price Volatility", 0.001, 0.05, 0.01, 0.001, help="Std dev of log returns per tick. Pull from contract history if unsure."),
        "days_remaining": st.number_input("Days until contract ends", 1, 365, 30, help="How many days until resolution."),
        "label":          "Manual Contract",
        "filter_estimate": None,
        "filter_lower": None,
        "filter_upper": None,
        "kelly_bet": 0,
        "bankroll": 1000,
        "kelly_fraction": 0.25,
    }


# ─── Step 2: Auto-Suggested Parameters ───────────────────────────────────────

if contract_data:
    st.markdown("---")
    st.markdown("### Step 2 — Suggested Simulation Parameters")
    st.markdown(f"<p style='color:{MUTED}; font-size:12px; margin-top:-6px;'>Auto-calibrated from contract data. Adjust as needed.</p>", unsafe_allow_html=True)

    price        = contract_data["price"]
    spread       = contract_data["spread"]
    imbalance    = contract_data["imbalance"]
    rv           = contract_data["realized_vol"]
    days_left    = contract_data.get("days_remaining") or 30

    # ── Suggest parameters based on contract characteristics ──────────
    # Regime: infer from spread and vol
    if rv > 0.025 or spread > 0.04:
        suggested_regime = "volatile"
    elif abs(imbalance - 0.5) > 0.15:
        suggested_regime = "trending"
    else:
        suggested_regime = "calm"

    # Agents: infer from spread (tight spread = more MMs, more informed)
    suggested_n_inf   = max(5,  min(30, int(15 - spread * 200)))
    suggested_n_noise = max(20, min(150, int(rv * 3000)))
    suggested_n_mm    = max(3,  min(15, int(10 - spread * 100)))

    # Filter: match process_vol to realized vol
    suggested_proc_vol = round(max(0.01, min(0.15, rv * 1.5)), 3)

    # Jump: higher when close to resolution date
    urgency = max(0.0, min(1.0, 1.0 - days_left / 90.0))
    suggested_jump    = round(0.03 + urgency * 0.07, 3)
    suggested_hawkes  = round(0.3 + urgency * 0.4, 2)

    suggested_steps   = min(5000, max(500, days_left * 20))

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(f"<div style='font-family:JetBrains Mono,monospace; font-size:9px; color:{MUTED}; letter-spacing:2px; margin-bottom:8px;'>MARKET</div>", unsafe_allow_html=True)
        true_p    = st.slider("Hidden Truth (your estimate)", 0.01, 0.99,
                               float(round(contract_data.get("filter_estimate") or price, 2)),
                               help="Set this to what YOU believe the true probability is. Use the filter estimate as a starting point.")
        n_steps   = st.number_input("Simulation Steps", 200, 10000, int(suggested_steps), step=100,
                                     help=f"Suggested based on {days_left} days remaining. More steps = more data.")
        init_reg  = st.selectbox("Initial Regime", ["calm", "volatile", "trending"],
                                  index=["calm", "volatile", "trending"].index(suggested_regime),
                                  help=f"Auto-suggested '{suggested_regime}' based on spread ({spread*100:.1f}¢) and volatility ({rv:.4f}).")

    with col_b:
        st.markdown(f"<div style='font-family:JetBrains Mono,monospace; font-size:9px; color:{MUTED}; letter-spacing:2px; margin-bottom:8px;'>AGENTS</div>", unsafe_allow_html=True)
        n_inf   = st.number_input("Informed Agents", 1, 100, suggested_n_inf,
                                   help=f"Suggested {suggested_n_inf} — tighter spreads imply more informed traders.")
        n_noise = st.number_input("Noise Agents", 5, 500, suggested_n_noise,
                                   help=f"Suggested {suggested_n_noise} — higher volatility implies more random trading.")
        n_mm    = st.number_input("Market Makers", 1, 20, suggested_n_mm,
                                   help=f"Suggested {suggested_n_mm} — based on spread width.")

    with col_c:
        st.markdown(f"<div style='font-family:JetBrains Mono,monospace; font-size:9px; color:{MUTED}; letter-spacing:2px; margin-bottom:8px;'>FILTER & DYNAMICS</div>", unsafe_allow_html=True)
        N_particles  = st.select_slider("Particles", [1000, 2500, 5000, 10000, 20000], value=5000,
                                         help="5000 is a good default. Increase if CI is too wide after many ticks.")
        process_vol  = st.slider("Process Volatility", 0.005, 0.15, suggested_proc_vol, 0.005,
                                  help=f"Suggested {suggested_proc_vol} — matched to contract's realized vol of {rv:.4f}.")
        jump_std     = st.slider("Jump Size σ", 0.01, 0.15, suggested_jump,
                                  help=f"Suggested {suggested_jump} — higher as resolution approaches ({days_left} days left).")
        hawkes_alpha = st.slider("Hawkes Excitation α", 0.0, 1.5, suggested_hawkes,
                                  help=f"Suggested {suggested_hawkes} — higher near resolution when news triggers order clusters.")

    # ── Show suggestion reasoning ──────────────────────────────────────
    with st.expander("💡 Why these values?", expanded=False):
        st.markdown(f"""
<div style="font-size:12px; line-height:1.8; color:{TEXT};">

| Parameter | Suggested | Reason |
|---|---|---|
| Initial Regime | **{suggested_regime}** | Spread {spread*100:.1f}¢ {'(wide → volatile)' if spread > 0.04 else '(tight → calm)'}, vol {rv:.4f} |
| Informed Agents | **{suggested_n_inf}** | Tight spreads → more informed flow |
| Noise Agents | **{suggested_n_noise}** | Vol {rv:.4f} → {'high' if rv > 0.02 else 'low'} random activity |
| Market Makers | **{suggested_n_mm}** | Spread {spread*100:.1f}¢ → {'few' if spread > 0.04 else 'many'} MMs |
| Process Vol | **{suggested_proc_vol}** | 1.5× realized vol ({rv:.4f}) |
| Jump Size | **{suggested_jump}** | {'High urgency' if urgency > 0.5 else 'Low urgency'} ({days_left} days left) |
| Hawkes α | **{suggested_hawkes}** | {'Near resolution — expect news clusters' if urgency > 0.5 else 'Far from resolution — moderate clustering'} |

</div>
        """, unsafe_allow_html=True)

    # ─── Step 3: Run Simulation ───────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### Step 3 — Run & Compare")

    run_sim = st.button("▶  RUN MATCHED SIMULATION", use_container_width=True, key="run_matched")

    if run_sim:
        market = PredictionMarketABM(
            true_prob=true_p,
            n_informed=n_inf,
            n_noise=n_noise,
            n_mm=n_mm,
            jump_std=jump_std,
            hawkes_alpha=hawkes_alpha,
            initial_regime=init_reg,
        )
        pf = PredictionMarketParticleFilter(
            N_particles=N_particles,
            prior_prob=float(price),
            process_vol=process_vol,
            adaptive_noise=True,
        )

        prices, estimates, lowers, uppers, regimes = [], [], [], [], []
        prog = st.progress(0, text="Running simulation…")
        for i in range(int(n_steps)):
            p = market.step()
            pf.update(p)
            prices.append(p)
            estimates.append(pf.estimate())
            lo, hi = pf.credible_interval()
            lowers.append(lo)
            uppers.append(hi)
            regimes.append(market.regime)
            if i % 300 == 0:
                prog.progress(i / n_steps, text=f"Step {i:,}/{int(n_steps):,}")
        prog.progress(1.0, text="Done.")

        final_est   = estimates[-1]
        filter_err  = abs(final_est - true_p)
        ci_width    = uppers[-1] - lowers[-1]
        rv_sim      = float(np.std(np.diff(np.log(np.clip(prices, 0.01, 0.99)))))

        # ── KPIs ──────────────────────────────────────────────────────
        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Filter Final Estimate", f"{final_est*100:.1f}¢",
                   delta=f"Truth: {true_p*100:.1f}¢")
        m2.metric("Filter Error", f"{filter_err:.4f}",
                   delta="↓ good" if filter_err < 0.05 else "↑ high")
        m3.metric("95% CI Width", f"{ci_width*100:.1f}¢",
                   delta="tight" if ci_width < 0.15 else "wide — need more ticks")
        m4.metric("Sim Realized Vol", f"{rv_sim:.4f}",
                   delta=f"Contract: {rv:.4f}")
        m5.metric("Regime Breakdown", f"{market.regime_breakdown().get('volatile', 0)*100:.0f}% vol")

        # ── Charts ────────────────────────────────────────────────────
        tab1, tab2 = st.tabs(["SIMULATION vs CONTRACT", "FILTER QUALITY"])

        with tab1:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            # Left: simulation
            ax = axes[0]
            t = range(len(prices))
            ax.plot(t, prices, color=MUTED, alpha=0.35, linewidth=0.8, label="Sim Price")
            ax.plot(t, estimates, color=ACCENT, linewidth=1.8, label="Filter Estimate")
            ax.fill_between(t, lowers, uppers, color=ACCENT, alpha=0.12, label="95% CI")
            ax.axhline(true_p, color=GREEN, linestyle="--", linewidth=1.2, label=f"Your Truth ({true_p:.2f})")
            ax.axhline(price, color=YELLOW, linestyle=":", linewidth=1, label=f"Real Price ({price:.2f})")
            ax.set_ylim(0, 1)
            ax.set_title("Simulation", color=TEXT)
            ax.set_ylabel("Probability")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Right: real contract history (if available)
            ax2 = axes[1]
            real_prices = contract_data.get("real_prices")
            real_ests   = contract_data.get("real_estimates")
            if real_prices and len(real_prices) > 2:
                ax2.plot(real_prices, color=MUTED, alpha=0.35, linewidth=0.8, label="Real Price")
                if real_ests:
                    ax2.plot(real_ests, color=ACCENT2, linewidth=1.8, label="Live Filter")
                ax2.set_ylim(0, 1)
                ax2.set_title("Real Contract (Live Feed)", color=TEXT)
                ax2.legend(fontsize=8)
            else:
                if "fm" in st.session_state and st.session_state.fm:
                    cids = list(st.session_state.get("tracked", {}).keys())
                    if cids:
                        _state = st.session_state.fm.get_state(cids[0])
                        if _state and len(_state.prices) > 2:
                            ax2.plot(_state.prices[-500:], color=MUTED, alpha=0.35, linewidth=0.8, label="Real Price")
                            ax2.plot(_state.estimates[-500:], color=ACCENT2, linewidth=1.8, label="Live Filter")
                            ax2.set_ylim(0, 1)
                            ax2.set_title("Real Contract (Live Feed)", color=TEXT)
                            ax2.legend(fontsize=8)
                        else:
                            ax2.text(0.5, 0.5, "No live data yet\nStart feed and return",
                                     ha="center", va="center", color=MUTED, transform=ax2.transAxes)
                    else:
                        ax2.text(0.5, 0.5, "No contracts tracked\nAdd one in Live Dashboard",
                                 ha="center", va="center", color=MUTED, transform=ax2.transAxes)
                else:
                    ax2.text(0.5, 0.5, "Live feed not running",
                             ha="center", va="center", color=MUTED, transform=ax2.transAxes)
                ax2.set_title("Real Contract (Live Feed)", color=TEXT)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylabel("Probability")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Vol comparison
            vol_match = abs(rv_sim - rv) / max(rv, 0.001)
            if vol_match < 0.3:
                st.success(f"✅ Volatility match is good — sim {rv_sim:.4f} vs contract {rv:.4f} ({vol_match*100:.0f}% diff). Parameters are well-calibrated.")
            elif vol_match < 0.7:
                st.warning(f"🟡 Volatility is roughly matched — sim {rv_sim:.4f} vs contract {rv:.4f} ({vol_match*100:.0f}% diff). Try adjusting Noise Agents or Jump Size.")
            else:
                st.error(f"❌ Volatility mismatch — sim {rv_sim:.4f} vs contract {rv:.4f} ({vol_match*100:.0f}% diff). Significantly adjust parameters and re-run.")

        with tab2:
            diag = pf.diagnostics
            ess_ratios  = [d.ess_ratio for d in diag]
            obs_noises  = [d.obs_noise for d in diag]
            resamples   = [i for i, d in enumerate(diag) if d.resampled]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
            ax1.plot(ess_ratios, color=GREEN, linewidth=1.2, label="ESS Ratio")
            ax1.axhline(0.5, color=RED, linestyle="--", linewidth=0.8, label="Resample threshold")
            for r in resamples[::max(1, len(resamples)//20)]:
                ax1.axvline(r, color=YELLOW, alpha=0.2, linewidth=0.5)
            ax1.set_ylabel("ESS / N")
            ax1.set_ylim(0, 1.05)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            ax2.plot(obs_noises, color=ACCENT2, linewidth=1.2, label="Adaptive Obs. Noise σ")
            ax2.set_ylabel("σ_obs")
            ax2.set_xlabel("Step")
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ─── Step 4: Recommended Live Feed Settings ───────────────────────
        st.divider()
        st.markdown("### Step 4 — Recommended Live Feed Settings")
        st.markdown(f"<p style='color:{MUTED}; font-size:12px; margin-top:-6px;'>Apply these in the Live Dashboard sidebar to match your filter to this contract type.</p>", unsafe_allow_html=True)

        # Adjust recommendations based on how the simulation performed
        rec_proc_vol  = process_vol if filter_err < 0.08 else round(process_vol * 1.2, 3)
        rec_particles = N_particles if ci_width < 0.20 else min(20000, N_particles * 2)
        rec_adaptive  = ci_width > 0.15 or rv > 0.02

        r1, r2, r3 = st.columns(3)
        r1.markdown(f"""
        <div style="background:{PANEL_BG}; border:1px solid {ACCENT}; border-radius:8px; padding:14px; text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace; font-size:9px; color:{MUTED}; letter-spacing:2px;">PARTICLES</div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:28px; color:{ACCENT}; font-weight:700;">{rec_particles:,}</div>
          <div style="font-size:11px; color:{MUTED};">Set in Live Feed sidebar</div>
        </div>
        """, unsafe_allow_html=True)
        r2.markdown(f"""
        <div style="background:{PANEL_BG}; border:1px solid {ACCENT}; border-radius:8px; padding:14px; text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace; font-size:9px; color:{MUTED}; letter-spacing:2px;">PROCESS VOL σ</div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:28px; color:{ACCENT}; font-weight:700;">{rec_proc_vol}</div>
          <div style="font-size:11px; color:{MUTED};">Set in Live Feed sidebar</div>
        </div>
        """, unsafe_allow_html=True)
        r3.markdown(f"""
        <div style="background:{PANEL_BG}; border:1px solid {'GREEN' if rec_adaptive else MUTED}; border-radius:8px; padding:14px; text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace; font-size:9px; color:{MUTED}; letter-spacing:2px;">ADAPTIVE NOISE</div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:28px; color:{GREEN if rec_adaptive else MUTED}; font-weight:700;">{'ON' if rec_adaptive else 'OFF'}</div>
          <div style="font-size:11px; color:{MUTED};">{'Recommended — volatile contract' if rec_adaptive else 'Optional — stable contract'}</div>
        </div>
        """, unsafe_allow_html=True)

        if filter_err > 0.10:
            st.warning(f"⚠ Filter error is high ({filter_err:.3f}). Try increasing **Process Volatility** or **Particles** and re-running.")
        elif filter_err < 0.03:
            st.success(f"✅ Filter converged well (error {filter_err:.3f}). These settings should work well on the live contract.")
        else:
            st.info(f"Filter error {filter_err:.3f} is acceptable. Monitor the live CI width — if it stays above 25¢ after 20+ ticks, increase particles.")
