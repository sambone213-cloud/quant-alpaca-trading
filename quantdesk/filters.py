import numpy as np
from scipy.special import expit, logit
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class FilterDiagnostics:
    step: int
    estimate: float
    lower: float
    upper: float
    ess: float
    ess_ratio: float
    resampled: bool
    obs_noise: float  # adaptive


class PredictionMarketParticleFilter:
    """
    Enhanced Sequential Monte Carlo filter.

    Enhancements over v1:
    - Adaptive observation noise (tracks market volatility)
    - Kernel smoothing (MCMC jitter) to prevent particle collapse
    - Diagnostics history for UI rendering
    - Entropy-based diversity metric
    """

    def __init__(
        self,
        N_particles: int = 5000,
        prior_prob: float = 0.5,
        process_vol: float = 0.05,
        obs_noise: float = 0.03,
        adaptive_noise: bool = True,
        kernel_bandwidth: float = 0.01,
    ):
        self.N = N_particles
        self.process_vol = process_vol
        self.obs_noise = obs_noise
        self.adaptive_noise = adaptive_noise
        self.kernel_bandwidth = kernel_bandwidth

        logit_prior = logit(prior_prob)
        self.logit_particles = logit_prior + np.random.normal(0, 0.5, N_particles)
        self.weights = np.ones(N_particles) / N_particles

        self.history: List[float] = []
        self.diagnostics: List[FilterDiagnostics] = []
        self._recent_obs: List[float] = []
        self._step = 0

    def _adapt_noise(self, observed: float):
        """Increase obs_noise when market is volatile to prevent filter shock."""
        self._recent_obs.append(observed)
        if len(self._recent_obs) > 50:
            self._recent_obs.pop(0)
        if self.adaptive_noise and len(self._recent_obs) >= 10:
            local_vol = float(np.std(self._recent_obs))
            self.obs_noise = max(0.01, min(0.15, local_vol * 2.0))

    def _kernel_smooth(self):
        """MCMC move step: add small jitter to prevent particle collapse."""
        jitter = np.random.normal(0, self.kernel_bandwidth, self.N)
        self.logit_particles += jitter

    def update(self, observed_price: float):
        self._adapt_noise(observed_price)

        # 1. Propagate
        noise = np.random.normal(0, self.process_vol, self.N)
        self.logit_particles += noise

        # 2. Probability space
        prob_particles = expit(self.logit_particles)

        # 3. Reweight (log-space for numerical stability)
        log_likelihood = -0.5 * ((observed_price - prob_particles) / self.obs_noise) ** 2
        log_weights = np.log(self.weights + 1e-300) + log_likelihood
        log_weights -= log_weights.max()
        self.weights = np.exp(log_weights)
        self.weights /= self.weights.sum()

        # 4. ESS check and resample
        ess = 1.0 / np.sum(self.weights ** 2)
        ess_ratio = ess / self.N
        resampled = False
        if ess_ratio < 0.5:
            self._systematic_resample()
            self._kernel_smooth()
            resampled = True

        est = self.estimate()
        low, high = self.credible_interval()
        self.history.append(est)

        self.diagnostics.append(FilterDiagnostics(
            step=self._step,
            estimate=est,
            lower=low,
            upper=high,
            ess=ess,
            ess_ratio=ess_ratio,
            resampled=resampled,
            obs_noise=self.obs_noise,
        ))
        self._step += 1

    def _systematic_resample(self):
        cumsum = np.cumsum(self.weights)
        u = (np.arange(self.N) + np.random.uniform()) / self.N
        indices = np.searchsorted(cumsum, u)
        self.logit_particles = self.logit_particles[indices]
        self.weights = np.ones(self.N) / self.N

    def estimate(self) -> float:
        probs = expit(self.logit_particles)
        return float(np.average(probs, weights=self.weights))

    def credible_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        probs = expit(self.logit_particles)
        sorted_idx = np.argsort(probs)
        sorted_probs = probs[sorted_idx]
        sorted_weights = self.weights[sorted_idx]
        cumw = np.cumsum(sorted_weights)
        lower = float(sorted_probs[np.searchsorted(cumw, alpha / 2)])
        upper = float(sorted_probs[np.searchsorted(cumw, 1 - alpha / 2)])
        return lower, upper

    def particle_entropy(self) -> float:
        """Diversity metric: high entropy = well-spread particles."""
        probs = expit(self.logit_particles)
        hist, _ = np.histogram(probs, bins=20, range=(0, 1), density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log(hist + 1e-10)) / np.log(20))

    def reset(self, prior_prob: float = 0.5):
        logit_prior = logit(prior_prob)
        self.logit_particles = logit_prior + np.random.normal(0, 0.5, self.N)
        self.weights = np.ones(self.N) / self.N
        self.history.clear()
        self.diagnostics.clear()
        self._recent_obs.clear()
        self._step = 0
