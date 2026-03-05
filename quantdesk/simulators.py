import numpy as np
from scipy.stats import norm, t as t_dist
from scipy.optimize import brentq
from typing import Tuple, List, Optional


# ============================================================
# Calibration
# ============================================================

def brier_score(predictions: List[float], outcomes: List[float]) -> float:
    """Proper scoring rule for probability calibration."""
    return float(np.mean((np.array(predictions) - np.array(outcomes)) ** 2))


def log_score(predictions: List[float], outcomes: List[float], eps: float = 1e-9) -> float:
    """Log scoring rule — more sensitive to extreme mispricings than Brier."""
    p = np.clip(predictions, eps, 1 - eps)
    o = np.array(outcomes)
    return float(np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


# ============================================================
# Binary Contract Monte Carlo
# ============================================================

def simulate_binary_contract(
    S0: float, K: float, mu: float, sigma: float, T: float, N_paths: int = 100_000
) -> Tuple[float, float]:
    """Crude Monte Carlo for binary contract (GBM)."""
    Z = np.random.standard_normal(N_paths)
    S_T = S0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = (S_T > K).astype(float)
    p_hat = payoffs.mean()
    se = np.sqrt(p_hat * (1 - p_hat) / N_paths)
    return float(p_hat), float(se)


def rare_event_IS(
    S0: float, K_crash: float, sigma: float, T: float, N_paths: int = 100_000
) -> Tuple[float, float]:
    """
    Importance Sampling (Radon-Nikodym measure change) for deep OTM crash contracts.
    Tilts the drift toward the threshold for massive variance reduction.
    """
    K = S0 * (1 - K_crash)
    mu_original = -0.5 * sigma ** 2
    log_threshold = np.log(K / S0)
    mu_tilt = log_threshold / T

    Z = np.random.standard_normal(N_paths)
    log_returns_tilted = mu_tilt * T + sigma * np.sqrt(T) * Z
    S_T_tilted = S0 * np.exp(log_returns_tilted)

    log_LR = (
        -0.5 * ((log_returns_tilted - mu_original * T) / (sigma * np.sqrt(T))) ** 2
        + 0.5 * ((log_returns_tilted - mu_tilt * T) / (sigma * np.sqrt(T))) ** 2
    )
    LR = np.exp(log_LR)

    payoffs = (S_T_tilted < K).astype(float)
    is_estimates = payoffs * LR
    p_IS = float(is_estimates.mean())
    se_IS = float(is_estimates.std() / np.sqrt(N_paths))
    return p_IS, se_IS


def stratified_binary_mc(
    S0: float, K: float, sigma: float, T: float, J: int = 10, N_total: int = 100_000
) -> Tuple[float, float]:
    """Stratified MC — guaranteed coverage of all probability strata."""
    n_per_stratum = N_total // J
    estimates = []
    for j in range(J):
        U = np.random.uniform(j / J, (j + 1) / J, n_per_stratum)
        Z = norm.ppf(U)
        S_T = S0 * np.exp((-0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        estimates.append((S_T > K).mean())
    p = float(np.mean(estimates))
    se = float(np.std(estimates) / np.sqrt(J))
    return p, se


# ============================================================
# Copulas
# ============================================================

def simulate_correlated_outcomes_gaussian(
    probs: List[float], corr_matrix: np.ndarray, N: int = 100_000
) -> np.ndarray:
    """Gaussian copula — thin tails, symmetric."""
    d = len(probs)
    L = np.linalg.cholesky(corr_matrix)
    Z = np.random.standard_normal((N, d))
    X = Z @ L.T
    U = norm.cdf(X)
    return (U < np.array(probs)).astype(int)


def simulate_correlated_outcomes_t(
    probs: List[float], corr_matrix: np.ndarray, nu: int = 4, N: int = 100_000
) -> np.ndarray:
    """Student-t copula — symmetric fat tails (joint crash risk)."""
    d = len(probs)
    L = np.linalg.cholesky(corr_matrix)
    Z = np.random.standard_normal((N, d))
    X = Z @ L.T
    S = np.random.chisquare(nu, N) / nu
    T = X / np.sqrt(S[:, None])
    U = t_dist.cdf(T, nu)
    return (U < np.array(probs)).astype(int)


def simulate_correlated_outcomes_clayton(
    probs: List[float], theta: float = 2.0, N: int = 100_000
) -> np.ndarray:
    """Clayton copula — strong lower tail dependence (joint failures)."""
    d = len(probs)
    V = np.random.gamma(1 / theta, 1, N)
    E = np.random.exponential(1, (N, d))
    U = (1 + E / V[:, None]) ** (-1 / theta)
    return (U < np.array(probs)).astype(int)


def simulate_correlated_outcomes_gumbel(
    probs: List[float], theta: float = 2.0, N: int = 100_000
) -> np.ndarray:
    """
    Gumbel copula — strong upper tail dependence.
    Uses the Marshall-Olkin algorithm for simulation.
    """
    d = len(probs)
    # Stable distribution via Chambers-Mallows-Stuck method
    alpha = 1.0 / theta
    U_stable = np.random.uniform(0, np.pi, N)
    E = np.random.exponential(1, N)
    S = (np.sin(alpha * U_stable) / (np.sin(U_stable) ** (1 / alpha))) * (
        np.sin((1 - alpha) * U_stable) / E
    ) ** ((1 - alpha) / alpha)

    E_mat = np.random.exponential(1, (N, d))
    U = np.exp(-(E_mat / S[:, None]) ** (1.0 / theta))
    return (U < np.array(probs)).astype(int)


def simulate_correlated_outcomes_frank(
    probs: List[float], theta: float = 5.0, N: int = 100_000
) -> np.ndarray:
    """Frank copula — symmetric, can capture negative dependence."""
    d = len(probs)
    results = np.zeros((N, d), dtype=int)

    # Bivariate Frank via conditional inversion for each pair
    # For d > 2 we use pair-wise approximation
    U = np.random.uniform(0, 1, (N, d))
    if abs(theta) < 1e-6:
        return (U < np.array(probs)).astype(int)

    # Apply Frank dependence to first two dimensions, others independent
    t = np.random.uniform(0, 1, N)
    v = np.random.uniform(0, 1, N)
    den = 1 - np.exp(-theta) - (1 - np.exp(-theta * t)) * (1 - v)
    u2 = -np.log(1 - (1 - np.exp(-theta)) * v / den) / theta
    U[:, 0] = t
    U[:, 1] = np.clip(u2, 0, 1)

    return (U < np.array(probs)).astype(int)


def copula_comparison(
    probs: List[float],
    corr_matrix: np.ndarray,
    nu: int = 4,
    N: int = 500_000,
) -> dict:
    """Run all copulas and return sweep probabilities for comparison."""
    gauss = simulate_correlated_outcomes_gaussian(probs, corr_matrix, N)
    t_cop = simulate_correlated_outcomes_t(probs, corr_matrix, nu, N)
    clay = simulate_correlated_outcomes_clayton(probs, theta=2.0, N=N)
    gumb = simulate_correlated_outcomes_gumbel(probs, theta=2.0, N=N)
    indep = float(np.prod(probs))

    return {
        "independent": indep,
        "gaussian": float(gauss.all(axis=1).mean()),
        "student_t": float(t_cop.all(axis=1).mean()),
        "clayton": float(clay.all(axis=1).mean()),
        "gumbel": float(gumb.all(axis=1).mean()),
    }


# ============================================================
# Risk / Position Sizing
# ============================================================

def calculate_kelly_bet(
    filtered_prob: float,
    market_price: float,
    bankroll: float,
    fraction: float = 0.25,
) -> float:
    """Fractional Kelly criterion for binary prediction markets."""
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1 - market_price) / market_price
    p = filtered_prob
    q = 1 - p
    kelly_f = (b * p - q) / b
    return float(max(0.0, bankroll * kelly_f * fraction))


def kelly_growth_rate(p: float, market_price: float, fraction: float = 1.0) -> float:
    """Expected log-growth rate at a given Kelly fraction."""
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1 - market_price) / market_price
    f = fraction * ((b * p - (1 - p)) / b)
    f = max(0, min(f, 0.99))
    if f == 0:
        return 0.0
    return float(p * np.log(1 + b * f) + (1 - p) * np.log(1 - f))


def compute_cvar(
    filtered_prob: float,
    market_price: float,
    bankroll: float,
    alpha: float = 0.05,
    N: int = 50_000,
) -> Tuple[float, float]:
    """
    Monte Carlo CVaR (Expected Shortfall) for a Kelly-sized position.
    Returns (VaR at alpha, CVaR at alpha).
    """
    bet = calculate_kelly_bet(filtered_prob, market_price, bankroll)
    if bet == 0:
        return 0.0, 0.0

    outcomes = np.random.binomial(1, filtered_prob, N)
    pnl = np.where(outcomes == 1, bet * (1 - market_price) / market_price, -bet)
    sorted_pnl = np.sort(pnl)
    var_idx = int(alpha * N)
    var = float(-sorted_pnl[var_idx])
    cvar = float(-sorted_pnl[:var_idx].mean())
    return var, cvar


def kelly_fraction_sweep(
    p: float, market_price: float, n_points: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (fractions, growth_rates) for plotting Kelly curve."""
    fractions = np.linspace(0, 1, n_points)
    growth = np.array([kelly_growth_rate(p, market_price, f) for f in fractions])
    return fractions, growth


# ============================================================
# Correlation Stress Testing
# ============================================================

def stress_test_correlations(
    probs: List[float],
    base_corr: np.ndarray,
    stress_levels: List[float] = None,
    N: int = 200_000,
) -> List[dict]:
    """
    Sweep correlation strength from base to max and compare sweep probs.
    stress_levels: list of multipliers on the off-diagonal of corr_matrix
    """
    if stress_levels is None:
        stress_levels = [0.5, 0.75, 1.0, 1.25, 1.5]

    d = len(probs)
    results = []
    for level in stress_levels:
        stressed = base_corr.copy()
        for i in range(d):
            for j in range(d):
                if i != j:
                    stressed[i, j] = np.clip(base_corr[i, j] * level, -0.99, 0.99)
        # Ensure PSD
        eigvals = np.linalg.eigvals(stressed)
        if np.all(eigvals > 0):
            sim = simulate_correlated_outcomes_gaussian(probs, stressed, N)
            results.append({
                "stress_level": level,
                "sweep_prob": float(sim.all(axis=1).mean()),
                "corr_matrix": stressed.tolist(),
            })
    return results
