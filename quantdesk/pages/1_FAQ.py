import streamlit as st

DARK_BG  = "#0d0f14"
PANEL_BG = "#13161e"
BORDER   = "#1e2330"
ACCENT   = "#00d4ff"
ACCENT2  = "#7b2fff"
GREEN    = "#00ff88"
MUTED    = "#64748b"
TEXT     = "#e2e8f0"

st.set_page_config(page_title="Math Reference", layout="wide")

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Space+Grotesk:wght@400;600;700&display=swap');
  html, body, [class*="css"] {{ background: {DARK_BG}; color: {TEXT}; font-family: 'Space Grotesk', sans-serif; }}
  .stApp {{ background: {DARK_BG}; }}
  h1 {{ font-family: 'Space Grotesk', sans-serif !important; font-weight: 700;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT2}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  h2 {{ color: {ACCENT} !important; font-size: 14px !important; font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 2px; text-transform: uppercase; border-bottom: 1px solid {BORDER}; padding-bottom: 8px; margin-top: 32px; }}
  p, li {{ font-size: 14px; line-height: 1.8; color: {TEXT}; }}
  .stInfo {{ background: rgba(0,212,255,0.08) !important; border-left: 3px solid {ACCENT} !important; }}
</style>
""", unsafe_allow_html=True)

st.title("📐 Mathematical Reference")
st.markdown(f"<p style='color:{MUTED}; margin-top:-12px;'>Complete derivations and intuitions for every model in the stack.</p>", unsafe_allow_html=True)

# ── Section 1: Particle Filter ──────────────
st.header("1 · Sequential Monte Carlo (Particle Filter)")

st.write("""
The filter maintains **N particles** — each a hypothesis about the true event probability — and updates them 
as new market prices arrive via Bayes' rule applied in a Monte Carlo framework.
""")

st.latex(r"""
\text{State model (propagate):} \quad x_t = x_{t-1} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_{\text{proc}}^2)
""")

st.latex(r"""
\text{Observation likelihood:} \quad p(y_t | x_t) = \exp\!\left(-\frac{(y_t - \sigma(x_t))^2}{2\sigma_{\text{obs}}^2}\right)
""")

st.latex(r"""
\text{Weight update:} \quad w_t^{(i)} \propto w_{t-1}^{(i)} \cdot p(y_t | x_t^{(i)})
""")

st.latex(r"""
\text{Posterior estimate:} \quad \hat{p}_t = \sum_{i=1}^N w_t^{(i)} \cdot \sigma(x_t^{(i)})
""")

st.write("""
Note: particles live in **logit space** to enforce the (0,1) constraint on probabilities. The sigmoid function σ 
maps back to probability space. When the **Effective Sample Size** (ESS) drops below N/2, we resample and apply 
kernel smoothing (MCMC jitter) to prevent particle collapse.
""")

st.latex(r"\text{ESS} = \frac{1}{\sum_{i=1}^N (w_t^{(i)})^2}")

st.info("Adaptive Noise: The filter estimates local market volatility from a rolling window of observed prices and automatically inflates σ_obs during volatile regimes to prevent filter shock.")

# ── Section 2: Kyle's Lambda ──────────────
st.header("2 · Market Microstructure — Kyle's Lambda")

st.latex(r"\Delta P = \text{Trade Size} \cdot \lambda, \qquad \lambda = \frac{\sigma_v}{2\sigma_u}")

st.write("""
**λ** is the price impact per unit of order flow. **σ_v** is the fundamental uncertainty (distance 
of price from true value) and **σ_u** is noise trader liquidity. In volatile regimes, λ is scaled up 
to reflect wider spreads and lower depth.
""")

# ── Section 3: Jump Diffusion ──────────────
st.header("3 · Merton Jump-Diffusion")

st.latex(r"""
dS = \mu S\, dt + \sigma S\, dW_t + S\, dJ_t
""")

st.latex(r"""
dJ_t = \begin{cases} J \sim \mathcal{N}(\mu_J, \sigma_J^2) & \text{with prob } \lambda_{\text{jump}} \cdot dt \\ 0 & \text{otherwise} \end{cases}
""")

st.write("""
Prediction markets exhibit **discontinuous price jumps** when news breaks (election results, regulatory decisions). 
Pure Brownian motion cannot capture this. Jump frequency λ_jump is regime-dependent — higher in volatile regimes.
""")

# ── Section 4: Regime Switching ──────────────
st.header("4 · Markov Regime Switching")

st.latex(r"""
P(\text{regime}_{t+1} | \text{regime}_t) = \mathbf{Q} = 
\begin{pmatrix} 0.97 & 0.02 & 0.01 \\ 0.05 & 0.92 & 0.03 \\ 0.03 & 0.02 & 0.95 \end{pmatrix}
""")

st.write("Rows/columns: **[calm, volatile, trending]**. High diagonal entries reflect persistence. Each regime scales Kyle's λ and jump probability differently.")

# ── Section 5: Hawkes Process ──────────────
st.header("5 · Hawkes Self-Exciting Process")

st.latex(r"""
\lambda(t) = \mu + \alpha \sum_{t_i < t} e^{-\beta(t - t_i)}
""")

st.write("""
Order arrivals cluster in time: each trade increases the probability of subsequent trades exponentially 
decaying back to baseline μ. **α** controls excitation strength, **β** controls decay speed.
This captures the empirically observed **order flow clustering** in real markets.
""")

# ── Section 6: Copulas ──────────────
st.header("6 · Dependency Copulas")

st.write("Sklar's theorem guarantees that any joint distribution can be decomposed into marginals + a copula C:")

st.latex(r"F(x_1, \ldots, x_d) = C(F_1(x_1), \ldots, F_d(x_d))")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Gaussian Copula")
    st.latex(r"C^{Ga}(\mathbf{u}) = \Phi_\Sigma(\Phi^{-1}(u_1), \ldots, \Phi^{-1}(u_d))")
    st.write("No tail dependence — joint extremes are asymptotically independent. Underestimates crash risk.")

    st.subheader("Clayton Copula")
    st.latex(r"C^{Cl}(u_1, u_2) = (u_1^{-\theta} + u_2^{-\theta} - 1)^{-1/\theta}")
    st.write("Strong **lower tail** dependence — captures joint market crashes.")

with col2:
    st.subheader("Student-t Copula")
    st.latex(r"C^t(\mathbf{u}; \nu) = t_{\Sigma,\nu}(t_\nu^{-1}(u_1), \ldots, t_\nu^{-1}(u_d))")
    st.write("Symmetric fat-tail dependence — both joint crashes and joint booms are more likely.")

    st.subheader("Gumbel Copula")
    st.latex(r"C^{Gu}(u_1, u_2) = \exp\!\left(-\left[(-\ln u_1)^\theta + (-\ln u_2)^\theta\right]^{1/\theta}\right)")
    st.write("Strong **upper tail** dependence — captures joint market rallies.")

# ── Section 7: Importance Sampling ──────────────
st.header("7 · Importance Sampling (Radon-Nikodym)")

st.latex(r"""
\hat{p}^{IS} = \frac{1}{N}\sum_{i=1}^N \mathbf{1}_{S_T^{(i)} < K} \cdot \frac{dP}{dQ}(S_T^{(i)})
""")

st.latex(r"""
\frac{dP}{dQ} = \exp\!\left(\frac{(\mu_Q - \mu_P)\log S_T}{(\sigma\sqrt{T})^2} - \frac{\mu_Q^2 - \mu_P^2}{2\sigma^2}\right)
""")

st.write("""
By tilting the drift **μ_Q** toward the crash threshold, nearly all simulated paths become relevant. 
The Radon-Nikodym derivative reweights them back to the original measure P, giving an unbiased estimator 
with variance reduction of 100–10,000× vs crude Monte Carlo.
""")

# ── Section 8: Kelly ──────────────
st.header("8 · Fractional Kelly Criterion")

st.latex(r"""
f^* = \frac{bp - q}{b}, \qquad b = \frac{1 - \text{market price}}{\text{market price}}, \quad q = 1-p
""")

st.latex(r"""
G(f) = p \ln(1 + bf) + q \ln(1 - f)
""")

st.write("""
**f*** is the growth-maximizing bet fraction. In practice, **fractional Kelly** (e.g. 25% of f*) is used to 
account for estimation error in p — overbetting has catastrophic downside while underbetting merely reduces growth.
""")

st.info("CVaR (Conditional Value-at-Risk / Expected Shortfall) is computed as the expected loss conditional on being in the worst α% of outcomes. It is a coherent risk measure unlike VaR.")

st.latex(r"\text{CVaR}_\alpha = \mathbb{E}[-X \mid X \leq \text{VaR}_\alpha]")
