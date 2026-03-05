import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class OrderBookSnapshot:
    timestamp: int
    price: float
    bid: float
    ask: float
    spread: float
    volume: float
    informed_pnl: float
    noise_pnl: float
    regime: str
    intensity: float


class PredictionMarketABM:
    """
    Enhanced Agent-Based Model of a prediction market order book.
    
    Enhancements over v1:
    - Merton Jump-Diffusion: discontinuous price moves on news events
    - Markov Regime Switching: calm / volatile / trending states
    - Hawkes Process: self-exciting order arrival clustering
    - Full order book snapshots and analytics
    """

    REGIMES = {
        "calm":     {"vol_scale": 0.7,  "jump_prob": 0.005},
        "volatile": {"vol_scale": 1.8,  "jump_prob": 0.04},
        "trending": {"vol_scale": 1.2,  "jump_prob": 0.015},
    }

    TRANSITION = np.array([
        [0.97, 0.02, 0.01],
        [0.05, 0.92, 0.03],
        [0.03, 0.02, 0.95],
    ])
    REGIME_KEYS = ["calm", "volatile", "trending"]

    def __init__(
        self,
        true_prob: float,
        n_informed: int = 10,
        n_noise: int = 50,
        n_mm: int = 5,
        jump_mean: float = 0.0,
        jump_std: float = 0.05,
        hawkes_mu: float = 0.3,
        hawkes_alpha: float = 0.6,
        hawkes_beta: float = 1.0,
        initial_regime: str = "calm",
    ):
        self.true_prob = true_prob
        self.price = 0.50
        self.price_history: List[float] = [self.price]
        self.best_bid = 0.49
        self.best_ask = 0.51

        self.n_informed = n_informed
        self.n_noise = n_noise
        self.n_mm = n_mm

        self.jump_mean = jump_mean
        self.jump_std = jump_std

        self.hawkes_mu = hawkes_mu
        self.hawkes_alpha = hawkes_alpha
        self.hawkes_beta = hawkes_beta
        self.hawkes_intensity = hawkes_mu

        self.regime = initial_regime
        self.regime_history: List[str] = [initial_regime]

        self.volume = 0.0
        self.informed_pnl = 0.0
        self.noise_pnl = 0.0
        self.snapshots: List[OrderBookSnapshot] = []
        self.step_count = 0

    def _update_hawkes(self):
        dt = 1.0
        decay = np.exp(-self.hawkes_beta * dt)
        self.hawkes_intensity = (
            self.hawkes_mu + (self.hawkes_intensity - self.hawkes_mu) * decay
        )

    def _trigger_hawkes_event(self):
        self.hawkes_intensity += self.hawkes_alpha

    def _maybe_switch_regime(self):
        idx = self.REGIME_KEYS.index(self.regime)
        probs = self.TRANSITION[idx]
        self.regime = np.random.choice(self.REGIME_KEYS, p=probs)

    def _kyle_lambda(self) -> float:
        sigma_v = abs(self.true_prob - self.price) + 0.05
        sigma_u = 0.1 * np.sqrt(self.n_noise)
        base = sigma_v / (2 * sigma_u)
        return base * self.REGIMES[self.regime]["vol_scale"]

    def _maybe_jump(self):
        if np.random.random() < self.REGIMES[self.regime]["jump_prob"]:
            jump = np.random.normal(self.jump_mean, self.jump_std)
            self.price = np.clip(self.price + jump, 0.01, 0.99)

    def _informed_trade(self):
        signal = self.true_prob + np.random.normal(0, 0.02)
        lam = self._kyle_lambda()
        if signal > self.best_ask + 0.01:
            size = min(0.1, abs(signal - self.price) * 2)
            self.price += size * lam
            self.volume += size
            self.informed_pnl += (self.true_prob - self.best_ask) * size
            self._trigger_hawkes_event()
        elif signal < self.best_bid - 0.01:
            size = min(0.1, abs(self.price - signal) * 2)
            self.price -= size * lam
            self.volume += size
            self.informed_pnl += (self.best_bid - self.true_prob) * size
            self._trigger_hawkes_event()
        self.price = np.clip(self.price, 0.01, 0.99)
        self._update_book()

    def _noise_trade(self):
        direction = np.random.choice([-1, 1])
        size = np.random.exponential(0.02)
        self.price += direction * size * self._kyle_lambda()
        self.price = np.clip(self.price, 0.01, 0.99)
        self.volume += size
        self.noise_pnl -= abs(self.price - self.true_prob) * size * 0.5
        self._update_book()

    def _mm_update(self):
        spread = max(0.02, 0.05 * (1 - self.volume / 100))
        if self.regime == "volatile":
            spread *= 1.8
        self.best_bid = self.price - spread / 2
        self.best_ask = self.price + spread / 2

    def _update_book(self):
        spread = self.best_ask - self.best_bid
        self.best_bid = self.price - spread / 2
        self.best_ask = self.price + spread / 2

    def step(self) -> float:
        self._update_hawkes()
        self._maybe_switch_regime()
        self._maybe_jump()

        total = self.n_informed + self.n_noise + self.n_mm
        effective_rate = min(self.hawkes_intensity / self.hawkes_mu, 3.0)
        r = np.random.random()

        if r < (self.n_informed / total) * effective_rate * 0.3:
            self._informed_trade()
        elif r < (self.n_informed + self.n_noise) / total:
            self._noise_trade()
        else:
            self._mm_update()

        self.price_history.append(self.price)
        self.regime_history.append(self.regime)
        self.step_count += 1

        self.snapshots.append(OrderBookSnapshot(
            timestamp=self.step_count,
            price=self.price,
            bid=self.best_bid,
            ask=self.best_ask,
            spread=self.best_ask - self.best_bid,
            volume=self.volume,
            informed_pnl=self.informed_pnl,
            noise_pnl=self.noise_pnl,
            regime=self.regime,
            intensity=self.hawkes_intensity,
        ))
        return self.price

    def run(self, n_steps: int = 1000) -> np.ndarray:
        for _ in range(n_steps):
            self.step()
        return np.array(self.price_history)

    def regime_breakdown(self) -> dict:
        from collections import Counter
        counts = Counter(self.regime_history)
        total = len(self.regime_history)
        return {k: v / total for k, v in counts.items()}

    def realized_volatility(self, window: int = 100) -> float:
        if len(self.price_history) < 2:
            return 0.0
        prices = np.array(self.price_history[-window:])
        log_rets = np.diff(np.log(np.clip(prices, 1e-6, 1 - 1e-6)))
        return float(np.std(log_rets) * np.sqrt(252))
