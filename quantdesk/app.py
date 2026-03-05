import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import pandas as pd
from market import PredictionMarketABM
from filters import PredictionMarketParticleFilter
from simulators import (
    calculate_kelly_bet, kelly_fraction_sweep, compute_cvar,
    rare_event_IS, stratified_binary_mc,
    simulate_correlated_outcomes_gaussian,
    simulate_correlated_outcomes_t,
    simulate_correlated_outcomes_clayton,
    simulate_correlated_outcomes_gumbel,
    copula_comparison, stress_test_correlations,
    brier_score, log_score,
)

# ─────────────────────────────────────────────
# THEME CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Quant PM Stack",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG   = "#0d0f14"
PANEL_BG  = "#13161e"
BORDER    = "#1e2330"
ACCENT    = "#00d4ff"
ACCENT2   = "#7b2fff"
GREEN     = "#00ff88"
RED       = "#ff4466"
YELLOW    = "#ffd166"
TEXT      = "#e2e8f0"
MUTED     = "#64748b"

REGIME_COLORS = {"calm": "#00ff88", "volatile": "#ff4466", "trending": "#ffd166"}

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       BORDER,
    "grid.linewidth":   0.5,
    "legend.facecolor": PANEL_BG,
    "legend.edgecolor": BORDER,
    "font.family":      "monospace",
})

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {{
    background-color: {DARK_BG};
    color: {TEXT};
    font-family: 'Space Grotesk', sans-serif;
  }}

  .stApp {{ background-color: {DARK_BG}; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background: {PANEL_BG};
    border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

  /* Metrics */
  [data-testid="stMetric"] {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 16px !important;
  }}
  [data-testid="stMetricLabel"] {{ color: {MUTED} !important; font-size: 11px !important; letter-spacing: 1px; text-transform: uppercase; }}
  [data-testid="stMetricValue"] {{ color: {ACCENT} !important; font-family: 'JetBrains Mono', monospace !important; font-size: 22px !important; }}
  [data-testid="stMetricDelta"] {{ font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }}

  /* Buttons */
  .stButton > button {{
    background: linear-gradient(135deg, {ACCENT2}, {ACCENT});
    color: #000 !important;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
    padding: 10px 24px;
    transition: opacity 0.2s;
  }}
  .stButton > button:hover {{ opacity: 0.85; }}

  /* Inputs */
  .stSlider > div > div > div {{ background: {ACCENT} !important; }}
  .stNumberInput input, .stSelectbox select {{ 
    background: {PANEL_BG} !important; 
    color: {TEXT} !important; 
    border: 1px solid {BORDER} !important;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{ background: {PANEL_BG}; border-bottom: 1px solid {BORDER}; gap: 0; }}
  .stTabs [data-baseweb="tab"] {{ 
    background: transparent; 
    color: {MUTED}; 
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-bottom: 2px solid transparent;
    padding: 12px 20px;
  }}
  .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important; }}

  /* Info / Warning boxes */
  .stInfo {{ background: rgba(0,212,255,0.08) !important; border-left: 3px solid {ACCENT} !important; }}
  .stSuccess {{ background: rgba(0,255,136,0.08) !important; border-left: 3px solid {GREEN} !important; }}
  .stWarning {{ background: rgba(255,209,102,0.08) !important; border-left: 3px solid {YELLOW} !important; }}

  /* Progress bar */
  .stProgress > div > div {{ background: {ACCENT} !important; }}

  /* Expanders */
  .streamlit-expanderHeader {{
    background: {PANEL_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    color: {MUTED} !important;
  }}

  /* Headers */
  h1 {{ 
    font-family: 'Space Grotesk', sans-serif !important; 
    font-weight: 700;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
  }}
  h2 {{ font-family: 'Space Grotesk', sans-serif !important; color: {TEXT} !important; }}
  h3 {{ 
    font-family: 'JetBrains Mono', monospace !important; 
    color: {ACCENT} !important; 
    font-size: 13px !important;
    letter-spacing: 2px;
    text-transform: uppercase;
  }}

  /* DataFrames */
  [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}

  /* Radio buttons */
  .stRadio label {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; color: {MUTED}; }}
  .stRadio [data-baseweb="radio"] span {{ border-color: {ACCENT} !important; }}

  /* Divider */
  hr {{ border-color: {BORDER}; }}

  /* Tag/badge style */
  .regime-badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="padding: 16px 0 24px 0;">
      <div style="font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:3px; color:{MUTED}; margin-bottom:4px;">ANTHROPIC RESEARCH</div>
      <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:20px; 
                  background:linear-gradient(90deg,{ACCENT},{ACCENT2}); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        QUANT PM STACK
      </div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:{MUTED}; margin-top:2px;">v2.0 · INSTITUTIONAL</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:10px; letter-spacing:2px; color:{MUTED}; margin-bottom:8px;'>SELECT ENGINE</div>", unsafe_allow_html=True)

    engine = st.radio(
        "",
        [
            "🔴  ABM + Particle Filter",
            "🔵  Dependency Copulas",
            "🟡  Tail Risk Simulator",
            "🟢  Kelly & Risk Engine",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(f"<div style='font-size:10px; letter-spacing:2px; color:{MUTED}; margin-bottom:8px;'>SESSION</div>", unsafe_allow_html=True)
    if "run_count" not in st.session_state:
        st.session_state.run_count = 0
    if "run_history" not in st.session_state:
        st.session_state.run_history = []

    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Runs", st.session_state.run_count)
    if st.button("Clear History", use_container_width=True):
        st.session_state.run_history = []
        st.session_state.run_count = 0
        st.rerun()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def fmt_pct(v: float) -> str:
    return f"{v*100:.2f}%"

def sparkline_color(v: float, truth: float) -> str:
    return GREEN if abs(v - truth) < 0.05 else YELLOW if abs(v - truth) < 0.15 else RED


# ─────────────────────────────────────────────
# ENGINE 1 — ABM + PARTICLE FILTER
# ─────────────────────────────────────────────

if "ABM" in engine:
    st.title("ABM + Particle Filter")
    st.markdown(f"<p style='color:{MUTED}; font-size:13px; margin-top:-12px;'>Agent-Based Market Dynamics · Sequential Monte Carlo · Regime Detection</p>", unsafe_allow_html=True)

    # ── Plain-English Reference Guide ───────────────────────────────────
    with st.expander("📖  HOW TO READ THIS SIMULATOR", expanded=False):
        st.markdown(f"""
<div style="font-size:13px; line-height:1.9; color:{TEXT};">

**What is this?**
This simulator creates a fake prediction market and tests whether the particle filter can figure out the hidden true probability just by watching noisy prices — the same way it works on real Polymarket data.

---

#### 🎯 Market Parameters

**Hidden Truth Probability**
The answer the market is *actually* trying to find — e.g. 0.65 means there's a 65% true chance the event happens. The market doesn't know this. Only you do. Watch how quickly the filter converges to it.

**Simulation Steps**
How many trading rounds to run. Each step = one trade arriving. More steps = more data for the filter to work with. 2000 is a good default; use 5000+ to test filter convergence on hard problems.

**Initial Regime**
The market's starting "mood":
- 🟢 **Calm** — low volatility, small spreads, slow price moves. Filter converges quickly.
- 🔴 **Volatile** — wide spreads, jumps, erratic prices. Filter takes longer but adaptive noise helps.
- 🟡 **Trending** — prices drift in one direction. Tests filter's ability to track a moving target.

Regimes switch randomly over time using a Markov chain, so the market doesn't stay in one state forever.

---

#### 👥 Agent Types

**Informed Agents**
Traders who *know* something close to the true probability. They trade when the market price is far from the truth — buying when it's too low, selling when it's too high. More informed agents = prices converge to truth faster. Think: analysts, insiders, researchers.

**Noise Agents**
Traders who trade randomly with no useful information — they just add noise. More noise agents = messier prices, harder for the filter. Think: retail traders, bots, random hedgers.

**Market Makers**
Set the bid/ask spread and provide liquidity. They don't take directional bets — they just earn the spread. More market makers = tighter spreads, more stable prices. In volatile regimes they widen spreads to protect themselves.

*The ratio of informed to noise traders is the key variable. A market with 10 informed and 50 noise agents is realistic. 50 informed vs 5 noise = very efficient market where price quickly finds truth.*

---

#### 🔬 Filter Parameters

**Particles**
The number of simultaneous hypotheses the filter maintains about the true probability. Each "particle" is one guess. More particles = more accurate but slower.
- 1,000 — fast, rough estimate
- 5,000 — good balance (default)
- 20,000 — high precision, use for final analysis

**Process Volatility (σ_proc)**
How much the filter expects the *true probability itself* to change each step. Low values (0.01–0.03) assume the truth is stable — good for slow-moving events. High values (0.08–0.15) let the filter track fast-moving situations but make it noisier. Match this to how quickly you expect the real-world event to evolve.

**Adaptive Observation Noise**
When ON, the filter automatically increases its uncertainty during volatile market periods to avoid "filter shock" — where a sudden price jump causes it to overreact. Recommended to keep ON for real data.

---

#### ⚡ Jump & Clustering Parameters

**Jump Size σ**
The size of sudden price discontinuities — like a news event breaking mid-market. Bigger values = more dramatic jumps. At 0.05 you get realistic occasional shocks; at 0.15 you get crisis-level volatility.

**Hawkes Excitation α**
Controls order flow clustering — how much one trade triggers more trades. At 0.0 trades arrive independently. At 1.0+ you get bursts of rapid activity (like a news event causing a flurry of orders). Real markets have α around 0.4–0.7.

---

#### 📊 Reading the Output

**Filter Error** — how far the filter's final estimate is from the hidden truth. Lower is better. Compare to **Final Price Error** to see how much the filter improves on raw market prices.

**Informed P&L** — profit earned by informed agents. Positive = they successfully exploited mispricing. Large values mean the market was inefficient.

**Noise Agent Bleed** — losses from noise traders. Large negative = noisy market. In real markets this is the "cost" retail traders pay to informed traders.

**ESS Ratio** — Effective Sample Size. Drops when particles cluster (filter is certain) or collapse (filter is confused). Dips below 0.5 trigger a resample. Frequent resamples = high uncertainty.

**Particle Entropy** — diversity of the filter's hypotheses. 1.0 = completely uncertain. 0.0 = fully confident. Watch this drop as the filter converges toward the truth.

</div>
        """, unsafe_allow_html=True)

    # ── Parameters ──────────────────────────────
    with st.expander("⚙  SIMULATION PARAMETERS", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Market")
            true_p   = st.slider("Hidden Truth Probability", 0.01, 0.99, 0.65, key="abm_truth",
                                  help="The real answer the market is trying to discover. Only you know this. Watch how close the filter gets.")
            n_steps  = st.number_input("Simulation Steps", 200, 10000, 2000, step=100, key="abm_steps",
                                        help="Number of trading rounds. More steps = more data for the filter. Try 2000 to start.")
            init_reg = st.selectbox("Initial Regime", ["calm", "volatile", "trending"], key="abm_regime",
                                     help="Starting market condition. Calm = tight spreads. Volatile = jumpy prices. Trending = drifting prices. Regimes switch randomly over time.")
        with c2:
            st.markdown("### Agents")
            n_inf   = st.number_input("Informed Agents", 1, 100, 10, key="abm_inf",
                                       help="Traders who know the true probability and exploit mispricings. More = faster price discovery. Think analysts or insiders.")
            n_noise = st.number_input("Noise Agents", 10, 500, 50, key="abm_noise",
                                       help="Traders who trade randomly with no useful information. More = noisier prices, harder for the filter to extract signal.")
            n_mm    = st.number_input("Market Makers", 1, 20, 5, key="abm_mm",
                                       help="Provide liquidity by quoting bid/ask. More MMs = tighter spreads. They widen spreads in volatile regimes to protect themselves.")
        with c3:
            st.markdown("### Filter")
            N_particles = st.select_slider("Particles", [1000, 2500, 5000, 10000, 20000], value=5000, key="abm_particles",
                                            help="Number of simultaneous probability hypotheses. More = more accurate but slower. 5000 is a good balance.")
            process_vol = st.slider("Process Volatility", 0.01, 0.20, 0.05, key="abm_pvol",
                                     help="How much the filter expects the true probability to drift each step. Low (0.01–0.03) for stable events. High (0.08+) for fast-moving situations.")
            adaptive_on = st.toggle("Adaptive Observation Noise", True, key="abm_adaptive",
                                     help="Automatically inflates filter uncertainty during volatile periods to prevent overreaction to sudden price jumps. Keep ON for real data.")

        st.markdown("### Jump-Diffusion (Merton)")
        j1, j2 = st.columns(2)
        jump_std     = j1.slider("Jump Size σ", 0.01, 0.15, 0.05, key="abm_jstd",
                                  help="Size of sudden price discontinuities (news events). 0.05 = realistic shocks. 0.15 = crisis-level volatility.")
        hawkes_alpha = j2.slider("Hawkes Excitation α", 0.0, 1.5, 0.6, key="abm_halpha",
                                  help="How much one trade triggers more trades (order flow clustering). 0 = independent arrivals. 0.6 = realistic clustering. >1 = explosive bursts.")

    run_abm = st.button("▶  RUN SIMULATION", key="run_abm", use_container_width=True)

    if run_abm:
        st.session_state.run_count += 1

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
            prior_prob=0.50,
            process_vol=process_vol,
            adaptive_noise=adaptive_on,
        )

        prices, estimates, lowers, uppers = [], [], [], []
        intensities, regimes = [], []

        progress = st.progress(0, text="Initializing simulation...")
        for i in range(int(n_steps)):
            p = market.step()
            pf.update(p)
            est = pf.estimate()
            lo, hi = pf.credible_interval()
            prices.append(p)
            estimates.append(est)
            lowers.append(lo)
            uppers.append(hi)
            intensities.append(market.hawkes_intensity)
            regimes.append(market.regime)
            if i % 200 == 0:
                progress.progress(i / n_steps, text=f"Step {i:,} / {int(n_steps):,} · Regime: {market.regime.upper()}")
        progress.progress(1.0, text="Complete.")

        # ── KPIs ──────────────────────────────────
        st.divider()
        k1, k2, k3, k4, k5 = st.columns(5)
        final_err = abs(prices[-1] - true_p)
        filter_err = abs(estimates[-1] - true_p)
        rvol = market.realized_volatility()
        rb = market.regime_breakdown()

        k1.metric("Final Price Error", f"{final_err:.4f}", delta=f"{'↓' if final_err < 0.05 else '↑'} target <0.05")
        k2.metric("Filter Error", f"{filter_err:.4f}", delta=f"vs price {final_err-filter_err:+.4f}")
        k3.metric("Informed P&L", f"${market.informed_pnl:.2f}")
        k4.metric("Noise Agent Bleed", f"${market.noise_pnl:.2f}")
        k5.metric("Realized Vol (ann.)", f"{rvol:.3f}")

        # ── Main chart tabs ───────────────────────
        st.divider()
        tab1, tab2, tab3, tab4 = st.tabs(["PRICE & FILTER", "REGIME MAP", "HAWKES INTENSITY", "FILTER DIAGNOSTICS"])

        with tab1:
            fig, ax = plt.subplots(figsize=(14, 5))
            t = range(len(prices))
            ax.plot(t, prices, color=MUTED, alpha=0.35, linewidth=0.8, label="Order Book Price")
            ax.plot(t, estimates, color=ACCENT, linewidth=1.8, label=r"SMC Filter $\mathbb{E}[x_t|y_{1:t}]$")
            ax.fill_between(t, lowers, uppers, color=ACCENT, alpha=0.12, label="95% Credible Interval")
            ax.axhline(true_p, color=GREEN, linestyle="--", linewidth=1.2, label=f"Truth ({true_p:.2f})")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_xlabel("Step")
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with tab2:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), gridspec_kw={"height_ratios": [4, 1]})
            ax1.plot(t, prices, color=MUTED, alpha=0.3, linewidth=0.6)
            ax1.plot(t, estimates, color=ACCENT, linewidth=1.5)
            ax1.axhline(true_p, color=GREEN, linestyle="--", linewidth=1)
            ax1.set_ylim(0, 1)
            ax1.set_ylabel("Probability")
            ax1.grid(True, alpha=0.3)

            # Regime background coloring
            regime_arr = np.array(regimes)
            for regime_name, color in REGIME_COLORS.items():
                mask = regime_arr == regime_name
                changes = np.where(np.diff(mask.astype(int)))[0]
                starts = [0] + list(changes[mask[changes + 1]])
                ends = list(changes[~mask[changes + 1]]) + [len(regimes) - 1]
                for s, e in zip(starts, ends):
                    ax2.axvspan(s, e, color=color, alpha=0.6)

            ax2.set_yticks([])
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Regime", fontsize=9)
            patches = [mpatches.Patch(color=c, label=r.upper()) for r, c in REGIME_COLORS.items()]
            ax2.legend(handles=patches, loc="upper right", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Regime breakdown
            rc1, rc2, rc3 = st.columns(3)
            for col, (rname, rcolor) in zip([rc1, rc2, rc3], REGIME_COLORS.items()):
                pct = rb.get(rname, 0)
                col.metric(
                    f"{rname.upper()} %",
                    f"{pct*100:.1f}%",
                )

        with tab3:
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(intensities, color=ACCENT2, linewidth=1.2, alpha=0.9)
            ax.axhline(market.hawkes_mu, color=YELLOW, linestyle="--", linewidth=1, label=f"Baseline μ={market.hawkes_mu:.2f}")
            ax.fill_between(range(len(intensities)), market.hawkes_mu, intensities, alpha=0.2, color=ACCENT2)
            ax.set_ylabel("Hawkes Intensity λ(t)")
            ax.set_xlabel("Step")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            st.info("Spikes indicate clusters of informed trading activity — self-exciting order flow consistent with news arrival models.")

        with tab4:
            diag = pf.diagnostics
            ess_ratios = [d.ess_ratio for d in diag]
            obs_noises = [d.obs_noise for d in diag]
            resamples  = [i for i, d in enumerate(diag) if d.resampled]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
            ax1.plot(ess_ratios, color=GREEN, linewidth=1.2, label="ESS Ratio")
            ax1.axhline(0.5, color=RED, linestyle="--", linewidth=0.8, label="Resample threshold")
            for r in resamples[::max(1, len(resamples)//30)]:
                ax1.axvline(r, color=YELLOW, alpha=0.3, linewidth=0.5)
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

            entropy = pf.particle_entropy()
            d1, d2 = st.columns(2)
            d1.metric("Particle Entropy (diversity)", f"{entropy:.3f}", delta="1.0 = max diversity")
            d2.metric("Total Resamples", len(resamples))


# ─────────────────────────────────────────────
# ENGINE 2 — DEPENDENCY COPULAS
# ─────────────────────────────────────────────

elif "Copula" in engine:
    st.title("Dependency Copulas")
    st.markdown(f"<p style='color:{MUTED}; font-size:13px; margin-top:-12px;'>Extreme Co-Movement · Tail Dependence · Multi-Copula Comparison · Stress Testing</p>", unsafe_allow_html=True)

    with st.expander("⚙  CONTRACT SETUP", expanded=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### Parameters")
            nu_param = st.slider("Student-t DoF (ν)", 1, 20, 4, key="cop_nu")
            N_sim = st.select_slider("Paths", [100_000, 250_000, 500_000, 1_000_000], value=500_000, key="cop_n")
            gumbel_theta = st.slider("Gumbel θ", 1.01, 5.0, 2.0, key="cop_gtheta")
            clayton_theta = st.slider("Clayton θ", 0.1, 5.0, 2.0, key="cop_ctheta")

        with c2:
            st.markdown("### Contract Probabilities (5-State Sweep)")
            states = ["PA", "MI", "WI", "GA", "AZ"]
            probs_input = []
            prob_cols = st.columns(5)
            defaults = [0.52, 0.53, 0.51, 0.48, 0.50]
            for i, (col, state, default) in enumerate(zip(prob_cols, states, defaults)):
                p = col.slider(state, 0.01, 0.99, default, key=f"cop_p{i}")
                probs_input.append(p)

    st.markdown("### Correlation Matrix")
    corr_cols = st.columns([2, 1])
    with corr_cols[0]:
        base_corr = np.array([
            [1.0, 0.7, 0.7, 0.4, 0.3],
            [0.7, 1.0, 0.8, 0.3, 0.3],
            [0.7, 0.8, 1.0, 0.3, 0.3],
            [0.4, 0.3, 0.3, 1.0, 0.5],
            [0.3, 0.3, 0.3, 0.5, 1.0],
        ])
        fig, ax = plt.subplots(figsize=(5, 4))
        mask = np.zeros_like(base_corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = False
        sns.heatmap(base_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    xticklabels=states, yticklabels=states,
                    linewidths=0.5, linecolor=BORDER,
                    cbar_kws={"shrink": 0.8},
                    ax=ax, annot_kws={"size": 11})
        ax.set_title("Base Correlation Matrix", fontsize=11, color=TEXT, pad=10)
        st.pyplot(fig)
        plt.close()

    run_cop = st.button("▶  SIMULATE ALL COPULAS", key="run_cop", use_container_width=True)

    if run_cop:
        st.session_state.run_count += 1
        with st.spinner("Running copula simulations..."):
            results = copula_comparison(probs_input, base_corr, nu=nu_param, N=N_sim)
            clay_sim = simulate_correlated_outcomes_clayton(probs_input, theta=clayton_theta, N=N_sim)
            gumb_sim = simulate_correlated_outcomes_gumbel(probs_input, theta=gumbel_theta, N=N_sim)
            results["clayton_custom"] = float(clay_sim.all(axis=1).mean())
            results["gumbel_custom"]  = float(gumb_sim.all(axis=1).mean())

        st.divider()
        st.markdown("### Joint Sweep Probabilities")

        labels = ["Independent", "Gaussian", f"Student-t (ν={nu_param})", f"Clayton (θ={clayton_theta})", f"Gumbel (θ={gumbel_theta})"]
        values = [
            results["independent"],
            results["gaussian"],
            results["student_t"],
            results["clayton_custom"],
            results["gumbel_custom"],
        ]
        colors_bar = [MUTED, ACCENT, RED, YELLOW, ACCENT2]

        m_cols = st.columns(5)
        for col, label, val, color in zip(m_cols, labels, values, colors_bar):
            baseline = results["independent"]
            mult = val / baseline if baseline > 0 else 1.0
            col.markdown(f"""
            <div style="background:{PANEL_BG}; border:1px solid {color}; border-radius:8px; padding:12px; text-align:center;">
              <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:{MUTED}; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px;">{label}</div>
              <div style="font-family:'JetBrains Mono',monospace; font-size:22px; color:{color}; font-weight:600;">{val:.4f}</div>
              <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:{MUTED}; margin-top:4px;">{mult:.1f}× independent</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        tab_c1, tab_c2, tab_c3 = st.tabs(["COPULA COMPARISON", "MARGINAL DISTRIBUTIONS", "CORRELATION STRESS TEST"])

        with tab_c1:
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(labels, values, color=colors_bar, alpha=0.85, height=0.6)
            ax.axvline(results["independent"], color=MUTED, linestyle="--", linewidth=1, label="Independent baseline")
            ax.set_xlabel("Joint Sweep Probability")
            ax.set_title("Copula Comparison — P(All States Win)", color=TEXT)
            for bar, val in zip(bars, values):
                ax.text(val + 0.0001, bar.get_y() + bar.get_height()/2,
                        f"{val:.5f}", va="center", ha="left", fontsize=9, color=TEXT)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="x")
            st.pyplot(fig)
            plt.close()

            key_insight = results["student_t"] / max(results["gaussian"], 1e-9)
            st.success(f"Student-t copula identifies extreme sweeps as **{key_insight:.1f}×** more likely than Gaussian. Fat tails matter.")

        with tab_c2:
            gauss_sim = simulate_correlated_outcomes_gaussian(probs_input, base_corr, 50_000)
            t_sim     = simulate_correlated_outcomes_t(probs_input, base_corr, nu_param, 50_000)
            wins_gauss = gauss_sim.sum(axis=1)
            wins_t     = t_sim.sum(axis=1)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            bins = np.arange(-0.5, 6.5, 1)
            ax1.hist(wins_gauss, bins=bins, color=ACCENT, alpha=0.75, edgecolor=BORDER, density=True)
            ax1.set_title("Gaussian Copula — States Won", color=TEXT)
            ax1.set_xlabel("# States Won")
            ax1.set_ylabel("Density")
            ax1.grid(True, alpha=0.3)

            ax2.hist(wins_t, bins=bins, color=RED, alpha=0.75, edgecolor=BORDER, density=True)
            ax2.set_title(f"Student-t (ν={nu_param}) — States Won", color=TEXT)
            ax2.set_xlabel("# States Won")
            ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab_c3:
            stress_results = stress_test_correlations(probs_input, base_corr, N=100_000)
            if stress_results:
                stress_df = pd.DataFrame(stress_results)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(stress_df["stress_level"], stress_df["sweep_prob"],
                        color=ACCENT, linewidth=2, marker="o", markersize=6)
                ax.fill_between(stress_df["stress_level"], 0, stress_df["sweep_prob"], alpha=0.15, color=ACCENT)
                ax.axvline(1.0, color=YELLOW, linestyle="--", linewidth=1, label="Base correlation")
                ax.set_xlabel("Correlation Stress Multiplier")
                ax.set_ylabel("Sweep Probability")
                ax.set_title("Correlation Stress Test — Gaussian Copula", color=TEXT)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                st.warning("Higher correlation → exponentially higher joint tail risk. This is the key systematic risk in multi-state prediction market portfolios.")


# ─────────────────────────────────────────────
# ENGINE 3 — TAIL RISK
# ─────────────────────────────────────────────

elif "Tail" in engine:
    st.title("Tail Risk Simulator")
    st.markdown(f"<p style='color:{MUTED}; font-size:13px; margin-top:-12px;'>Importance Sampling · Variance Reduction · Deep OTM Binary Contracts</p>", unsafe_allow_html=True)

    with st.expander("⚙  CONTRACT PARAMETERS", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            k_crash = c1.slider("Crash Threshold", 0.05, 0.40, 0.20, 0.01, key="tail_k")
            sigma   = c1.slider("Underlying Volatility σ", 0.05, 0.50, 0.15, key="tail_sigma")
        with c2:
            T_days  = c2.slider("Horizon (days)", 1, 30, 5, key="tail_T")
            S0      = c2.number_input("Current Level S₀", 1000, 10000, 5000, step=100, key="tail_s0")
        with c3:
            N_paths = c3.select_slider("Paths", [10_000, 50_000, 100_000, 500_000], value=100_000, key="tail_N")
            J_strat = c3.slider("Stratification Strata J", 5, 50, 10, key="tail_J")

    run_tail = st.button("▶  EXECUTE RADON-NIKODYM MEASURE CHANGE", key="run_tail", use_container_width=True)

    if run_tail:
        st.session_state.run_count += 1
        T = T_days / 252.0
        K = S0 * (1 - k_crash)

        with st.spinner("Running importance sampling..."):
            p_IS, se_IS         = rare_event_IS(S0=S0, K_crash=k_crash, sigma=sigma, T=T, N_paths=N_paths)
            p_strat, se_strat   = stratified_binary_mc(S0=S0, K=K, sigma=sigma, T=T, J=J_strat, N_total=N_paths)

        # Crude MC for comparison (fewer paths as it's variance-inefficient for rare events)
        Z = np.random.standard_normal(min(N_paths, 50_000))
        S_T_crude = S0 * np.exp((-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        crude_payoffs = (S_T_crude < K).astype(float)
        p_crude = float(crude_payoffs.mean())
        se_crude = float(crude_payoffs.std() / np.sqrt(len(crude_payoffs)))

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Crude MC", f"{p_crude:.6f}", delta=f"±{se_crude:.6f} SE")
        m2.metric("Stratified MC", f"{p_strat:.6f}", delta=f"±{se_strat:.6f} SE")
        m3.metric("Importance Sampling", f"{p_IS:.6f}", delta=f"±{se_IS:.6f} SE")

        # Variance reduction ratio
        vr_is    = (se_crude ** 2) / max(se_IS ** 2, 1e-20)
        vr_strat = (se_crude ** 2) / max(se_strat ** 2, 1e-20)

        st.divider()
        v1, v2 = st.columns(2)
        v1.metric("IS Variance Reduction", f"{vr_is:.0f}×", delta="vs Crude MC")
        v2.metric("Stratified VR", f"{vr_strat:.0f}×", delta="vs Crude MC")

        # Path distribution chart
        tab_t1, tab_t2 = st.tabs(["PATH DISTRIBUTION", "TILTED MEASURE"])

        with tab_t1:
            fig, ax = plt.subplots(figsize=(12, 4))
            crashes = S_T_crude[S_T_crude < K]
            non_crashes = S_T_crude[S_T_crude >= K]
            ax.hist(non_crashes, bins=80, color=ACCENT, alpha=0.6, label="Non-crash paths", density=True)
            ax.hist(crashes, bins=max(5, len(crashes)//5), color=RED, alpha=0.9, label=f"Crash paths (< {K:.0f})", density=True)
            ax.axvline(K, color=YELLOW, linestyle="--", linewidth=1.5, label=f"K = {K:.0f}")
            ax.set_xlabel(f"S_T")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("Crude MC — Path Distribution", color=TEXT)
            st.pyplot(fig)
            plt.close()
            st.info("Crude MC almost never samples the crash region. Importance sampling guarantees coverage by tilting the drift toward the threshold.")

        with tab_t2:
            # Show the tilted vs original measure
            mu_orig = -0.5 * sigma**2
            log_threshold = np.log(K / S0)
            mu_tilt = log_threshold / T
            x = np.linspace(-0.4, 0.2, 300)
            pdf_orig = np.exp(-0.5*((x - mu_orig*T)/(sigma*np.sqrt(T)))**2)
            pdf_tilt = np.exp(-0.5*((x - mu_tilt*T)/(sigma*np.sqrt(T)))**2)
            pdf_orig /= pdf_orig.max()
            pdf_tilt /= pdf_tilt.max()

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x, pdf_orig, color=ACCENT, linewidth=2, label="Original measure P")
            ax.plot(x, pdf_tilt, color=RED, linewidth=2, label="Tilted measure Q")
            ax.axvline(log_threshold, color=YELLOW, linestyle="--", label=f"log(K/S₀) = {log_threshold:.3f}")
            ax.fill_between(x, 0, pdf_tilt, where=(x < log_threshold), color=RED, alpha=0.2, label="IS coverage zone")
            ax.set_xlabel("Log Return")
            ax.set_ylabel("Relative Density")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("Radon-Nikodym Measure Change", color=TEXT)
            st.pyplot(fig)
            plt.close()


# ─────────────────────────────────────────────
# ENGINE 4 — KELLY & RISK ENGINE
# ─────────────────────────────────────────────

elif "Kelly" in engine:
    st.title("Kelly & Risk Engine")
    st.markdown(f"<p style='color:{MUTED}; font-size:13px; margin-top:-12px;'>Fractional Kelly · CVaR · Position Sizing · Expected Log-Growth</p>", unsafe_allow_html=True)

    with st.expander("⚙  PARAMETERS", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Edge")
            filter_prob  = c1.slider("Filter Probability Estimate", 0.01, 0.99, 0.65, key="kelly_p",
                                      help="Your best estimate of the true probability. In the live feed this comes from the particle filter. Your edge = this minus the market price.")
            market_price = c1.slider("Current Market Price", 0.01, 0.99, 0.52, key="kelly_mp",
                                      help="What the market charges for a YES share. If your estimate is higher than this, you have a positive edge and should bet YES.")
        with c2:
            st.markdown("### Sizing")
            bankroll     = c2.number_input("Bankroll ($)", 100, 1_000_000, 10_000, step=500, key="kelly_br",
                                            help="Total capital available. Kelly sizes bets as a % of this number.")
            kelly_frac   = c2.slider("Kelly Fraction", 0.05, 1.0, 0.25, 0.05, key="kelly_f",
                                      help="Fraction of the mathematically optimal bet to actually place. 1.0 = full Kelly (aggressive, high variance). 0.25 = quarter Kelly (standard, much safer). Never go above 0.5 in practice.")
        with c3:
            st.markdown("### Risk")
            cvar_alpha   = c3.slider("CVaR Confidence Level", 0.01, 0.10, 0.05, 0.01, key="kelly_cvar",
                                      help="Tail threshold for risk metrics. 0.05 = look at the worst 5% of outcomes. CVaR tells you the average loss in those worst cases.")
            N_mc_kelly   = c3.select_slider("CVaR Paths", [10_000, 50_000, 100_000], value=50_000, key="kelly_n",
                                             help="Monte Carlo sample size for risk estimates. More = more accurate but slower. 50k is a good balance.")

    run_kelly = st.button("▶  COMPUTE POSITION", key="run_kelly", use_container_width=True)

    if run_kelly:
        bet = calculate_kelly_bet(filter_prob, market_price, bankroll, kelly_frac)
        edge = filter_prob - market_price
        b = (1 - market_price) / market_price
        full_kelly_f = max(0, (b * filter_prob - (1 - filter_prob)) / b)
        var, cvar = compute_cvar(filter_prob, market_price, bankroll, cvar_alpha, N_mc_kelly)

        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Edge", f"{edge:+.4f}", delta="Long" if edge > 0 else "Short / No Bet")
        m2.metric("Full Kelly f*", f"{full_kelly_f:.4f}")
        m3.metric(f"Bet Size ({kelly_frac:.0%} Kelly)", f"${bet:,.2f}")
        m4.metric(f"VaR ({cvar_alpha:.0%})", f"${var:,.2f}")
        m5.metric(f"CVaR ({cvar_alpha:.0%})", f"${cvar:,.2f}")

        tab_k1, tab_k2, tab_k3 = st.tabs(["KELLY CURVE", "P&L DISTRIBUTION", "SENSITIVITY"])

        with tab_k1:
            fracs, growths = kelly_fraction_sweep(filter_prob, market_price, n_points=100)
            peak_idx = np.argmax(growths)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(fracs, growths, color=ACCENT, linewidth=2)
            ax.fill_between(fracs, 0, growths, where=(growths > 0), color=GREEN, alpha=0.1, label="Positive growth")
            ax.fill_between(fracs, growths, 0, where=(growths < 0), color=RED, alpha=0.15, label="Ruin zone")
            ax.axvline(fracs[peak_idx], color=GREEN, linestyle="--", linewidth=1.5,
                       label=f"Full Kelly = {fracs[peak_idx]:.3f}")
            ax.axvline(full_kelly_f * kelly_frac, color=ACCENT2, linestyle="--", linewidth=1.5,
                       label=f"Fractional ({kelly_frac:.0%}) = {full_kelly_f*kelly_frac:.3f}")
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.set_xlabel("Kelly Fraction f")
            ax.set_ylabel("Expected Log-Growth G(f)")
            ax.set_title("Kelly Growth Curve", color=TEXT)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            st.info("Overbetting past full Kelly has lower expected growth than underbetting. The curve is asymmetric — right side decays to −∞ at f=1.")

        with tab_k2:
            if bet > 0:
                outcomes = np.random.binomial(1, filter_prob, N_mc_kelly)
                pnl = np.where(outcomes == 1, bet * b, -bet)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(pnl[pnl > 0], bins=3, color=GREEN, alpha=0.7, label="Win", density=True)
                ax.hist(pnl[pnl < 0], bins=3, color=RED, alpha=0.7, label="Loss", density=True)
                ax.axvline(-var, color=YELLOW, linestyle="--", label=f"VaR ({cvar_alpha:.0%}): ${var:,.2f}")
                ax.axvline(-cvar, color=RED, linestyle="--", label=f"CVaR ({cvar_alpha:.0%}): ${cvar:,.2f}")
                ax.set_xlabel("P&L ($)")
                ax.set_ylabel("Density")
                ax.set_title("P&L Distribution — Single Bet", color=TEXT)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("No positive edge detected — bet size is $0. Filter probability must exceed market price for a long position.")

        with tab_k3:
            # Sensitivity: vary filter_prob ±0.15
            prob_range = np.linspace(max(0.02, filter_prob - 0.2), min(0.98, filter_prob + 0.2), 40)
            bets_range = [calculate_kelly_bet(p, market_price, bankroll, kelly_frac) for p in prob_range]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(prob_range, bets_range, color=ACCENT, linewidth=2)
            ax.axvline(filter_prob, color=GREEN, linestyle="--", label=f"Current estimate: {filter_prob:.2f}")
            ax.axvline(market_price, color=RED, linestyle="--", label=f"Market price: {market_price:.2f}")
            ax.fill_between(prob_range, 0, bets_range, alpha=0.1, color=ACCENT)
            ax.set_xlabel("Filter Probability Estimate")
            ax.set_ylabel(f"Bet Size (${bankroll:,} bankroll, {kelly_frac:.0%} Kelly)")
            ax.set_title("Position Size Sensitivity", color=TEXT)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            st.info("Sensitivity analysis shows how bet sizing responds to estimation error. The steeper this curve, the more important accurate probability estimation is.")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.divider()
st.markdown(f"""
<div style="text-align:center; padding:16px 0; font-family:'JetBrains Mono',monospace; font-size:10px; color:{MUTED}; letter-spacing:1px;">
  QUANT PM STACK v2.0 · SMC FILTER · HAWKES · MERTON JUMP · MULTI-COPULA · KELLY-CVaR
  · <span style="color:{ACCENT}">SESSION RUNS: {st.session_state.run_count}</span>
</div>
""", unsafe_allow_html=True)
