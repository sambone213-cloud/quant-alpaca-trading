"""
FeedManager v2 — Multi-contract live dashboard engine.

Each tracked contract gets its own:
  - Polling thread
  - PredictionMarketParticleFilter instance
  - Price alert monitoring (move threshold + level crossings)
  - Kelly bet sizing on every tick
  - SQLite persistence
"""

import sqlite3
import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

from polymarket_client import PolymarketClient, PolymarketOrderBook


@dataclass
class ContractConfig:
    id: str
    label: str
    token_id: str = ""
    poll_interval: float = 5.0
    bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    alert_threshold: float = 0.05
    alert_above: Optional[float] = None
    alert_below: Optional[float] = None


@dataclass
class PriceTick:
    contract_id: str
    price: float
    timestamp: float = field(default_factory=time.time)
    bid: Optional[float] = None
    ask: Optional[float] = None
    depth_imbalance: Optional[float] = None
    spread: Optional[float] = None
    filter_estimate: Optional[float] = None
    filter_lower: Optional[float] = None
    filter_upper: Optional[float] = None
    kelly_bet: Optional[float] = None
    alert: Optional[str] = None


@dataclass
class ContractState:
    config: ContractConfig
    ticks: Deque[PriceTick]
    order_book: Optional[PolymarketOrderBook] = None
    tick_count: int = 0
    last_alert: Optional[str] = None
    last_alert_time: float = 0.0

    @property
    def latest_tick(self) -> Optional[PriceTick]:
        return self.ticks[-1] if self.ticks else None

    @property
    def prices(self) -> List[float]:
        return [t.price for t in self.ticks]

    @property
    def estimates(self) -> List[float]:
        return [t.filter_estimate for t in self.ticks if t.filter_estimate is not None]

    @property
    def lowers(self) -> List[float]:
        return [t.filter_lower for t in self.ticks if t.filter_lower is not None]

    @property
    def uppers(self) -> List[float]:
        return [t.filter_upper for t in self.ticks if t.filter_upper is not None]


class FeedManager:
    DB_PATH = "price_history.db"

    def __init__(
        self,
        db_path: Optional[str] = None,
        on_tick: Optional[Callable[[PriceTick], None]] = None,
        on_alert: Optional[Callable[[str, str], None]] = None,
        N_particles: int = 3000,
        process_vol: float = 0.02,
        adaptive_noise: bool = True,
    ):
        self.client = PolymarketClient()
        self._states: Dict[str, ContractState] = {}
        self._filters: Dict[str, object] = {}
        self._running = False
        self._threads: List[threading.Thread] = []
        self._on_tick = on_tick
        self._on_alert = on_alert
        self._lock = threading.RLock()
        self._db_path = db_path or self.DB_PATH
        self._N_particles = N_particles
        self._process_vol = process_vol
        self._adaptive_noise = adaptive_noise
        self._alert_log: List[dict] = []
        self._init_db()

    def _make_filter(self, prior: float = 0.5):
        from filters import PredictionMarketParticleFilter
        return PredictionMarketParticleFilter(
            N_particles=self._N_particles,
            prior_prob=prior,
            process_vol=self._process_vol,
            adaptive_noise=self._adaptive_noise,
        )

    # ── DB ────────────────────────────────────────────────────────────────

    def _init_db(self):
        db_dir = os.path.dirname(os.path.abspath(self._db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_ticks (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id     TEXT NOT NULL,
                    price           REAL NOT NULL,
                    filter_estimate REAL,
                    bid             REAL,
                    ask             REAL,
                    imbalance       REAL,
                    kelly_bet       REAL,
                    alert           TEXT,
                    timestamp       REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id TEXT NOT NULL,
                    message     TEXT NOT NULL,
                    timestamp   REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cid_ts ON price_ticks(contract_id, timestamp)")
            conn.commit()

    def _log_tick(self, tick: PriceTick):
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT INTO price_ticks
                       (contract_id, price, filter_estimate, bid, ask, imbalance, kelly_bet, alert, timestamp)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (tick.contract_id, tick.price, tick.filter_estimate,
                     tick.bid, tick.ask, tick.depth_imbalance,
                     tick.kelly_bet, tick.alert, tick.timestamp)
                )
                conn.commit()
        except Exception:
            pass

    def _log_alert(self, contract_id: str, message: str):
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO alerts (contract_id, message, timestamp) VALUES (?,?,?)",
                    (contract_id, message, time.time())
                )
                conn.commit()
        except Exception:
            pass

    def load_history(self, contract_id: str, limit: int = 5000) -> List[Tuple]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    """SELECT timestamp, price, filter_estimate, bid, ask, imbalance, kelly_bet
                       FROM price_ticks WHERE contract_id=?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (contract_id, limit)
                ).fetchall()
            return list(reversed(rows))
        except Exception:
            return []

    def load_alerts(self, contract_id: Optional[str] = None, limit: int = 100) -> List[dict]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                if contract_id:
                    rows = conn.execute(
                        "SELECT contract_id, message, timestamp FROM alerts WHERE contract_id=? ORDER BY timestamp DESC LIMIT ?",
                        (contract_id, limit)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT contract_id, message, timestamp FROM alerts ORDER BY timestamp DESC LIMIT ?",
                        (limit,)
                    ).fetchall()
            return [{"contract_id": r[0], "message": r[1], "timestamp": r[2]} for r in rows]
        except Exception:
            return []

    def db_stats(self) -> dict:
        try:
            with sqlite3.connect(self._db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM price_ticks").fetchone()[0]
                contracts = conn.execute("SELECT DISTINCT contract_id FROM price_ticks").fetchall()
                alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            size = os.path.getsize(self._db_path) / 1024
            return {"total_ticks": total, "contracts": [c[0] for c in contracts], "size_kb": size, "total_alerts": alerts}
        except Exception:
            return {"total_ticks": 0, "contracts": [], "size_kb": 0, "total_alerts": 0}

    # ── Contract Management ───────────────────────────────────────────────

    def add_contract(self, config: ContractConfig, prior: float = 0.5):
        with self._lock:
            self._states[config.id] = ContractState(config=config, ticks=deque(maxlen=10_000))
            self._filters[config.id] = self._make_filter(prior)
        if self._running:
            t = threading.Thread(target=self._poll_contract, args=(config,), daemon=True)
            t.start()
            self._threads.append(t)

    def remove_contract(self, contract_id: str):
        with self._lock:
            self._states.pop(contract_id, None)
            self._filters.pop(contract_id, None)

    def list_contracts(self) -> List[ContractConfig]:
        with self._lock:
            return [s.config for s in self._states.values()]

    def get_state(self, contract_id: str) -> Optional[ContractState]:
        with self._lock:
            return self._states.get(contract_id)

    def get_all_states(self) -> Dict[str, ContractState]:
        with self._lock:
            return dict(self._states)

    def reset_filter(self, contract_id: str, prior: float = 0.5):
        with self._lock:
            self._filters[contract_id] = self._make_filter(prior)
            if contract_id in self._states:
                self._states[contract_id].ticks.clear()
                self._states[contract_id].tick_count = 0

    # ── Alert Detection ───────────────────────────────────────────────────

    def _check_alerts(self, state: ContractState, tick: PriceTick) -> Optional[str]:
        config = state.config
        prev = state.latest_tick
        alert_msg = None

        if prev and config.alert_threshold:
            move = abs(tick.price - prev.price)
            if move >= config.alert_threshold:
                d = "▲" if tick.price > prev.price else "▼"
                alert_msg = f"{d} Move {move*100:+.1f}¢  ({prev.price*100:.1f}¢ → {tick.price*100:.1f}¢)"

        if config.alert_above and prev and prev.price < config.alert_above <= tick.price:
            alert_msg = f"🔔 Crossed ABOVE {config.alert_above*100:.0f}¢ → {tick.price*100:.1f}¢"
        if config.alert_below and prev and prev.price > config.alert_below >= tick.price:
            alert_msg = f"🔔 Crossed BELOW {config.alert_below*100:.0f}¢ → {tick.price*100:.1f}¢"

        if alert_msg and (time.time() - state.last_alert_time) > 60:
            state.last_alert = alert_msg
            state.last_alert_time = time.time()
            self._alert_log.append({"contract_id": tick.contract_id, "message": alert_msg, "timestamp": tick.timestamp})
            self._log_alert(tick.contract_id, alert_msg)
            if self._on_alert:
                try:
                    self._on_alert(tick.contract_id, alert_msg)
                except Exception:
                    pass
            return alert_msg
        return None

    # ── Tick Processing ───────────────────────────────────────────────────

    def _process_tick(self, config: ContractConfig, raw_price: float,
                      bid=None, ask=None, imbalance=None, spread=None, ob=None):
        from simulators import calculate_kelly_bet

        with self._lock:
            state = self._states.get(config.id)
            pf    = self._filters.get(config.id)
            if state is None or pf is None:
                return

        pf.update(raw_price)
        est = pf.estimate()
        lo, hi = pf.credible_interval()
        market_p = bid if bid else raw_price
        kelly = calculate_kelly_bet(est, market_p, config.bankroll, config.kelly_fraction)

        tick = PriceTick(
            contract_id=config.id,
            price=raw_price,
            bid=bid, ask=ask,
            depth_imbalance=imbalance,
            spread=spread,
            filter_estimate=est,
            filter_lower=lo,
            filter_upper=hi,
            kelly_bet=kelly,
        )

        with self._lock:
            tick.alert = self._check_alerts(state, tick)
            state.ticks.append(tick)
            state.tick_count += 1
            if ob is not None:
                state.order_book = ob

        self._log_tick(tick)
        if self._on_tick:
            try:
                self._on_tick(tick)
            except Exception:
                pass

    # ── Polling Loop ──────────────────────────────────────────────────────

    def _poll_contract(self, config: ContractConfig):
        token_id = config.token_id or config.id
        while self._running and config.id in self._states:
            try:
                ob = self.client.get_order_book(token_id)
                if ob and 0 < ob.best_bid < ob.best_ask < 1:
                    self._process_tick(config, ob.mid,
                                       bid=ob.best_bid, ask=ob.best_ask,
                                       imbalance=ob.depth_imbalance,
                                       spread=ob.best_ask - ob.best_bid, ob=ob)
                else:
                    mid = self.client.get_midpoint(token_id)
                    if mid and 0 < mid < 1:
                        self._process_tick(config, mid)
            except Exception:
                pass
            time.sleep(config.poll_interval)

    # ── Start / Stop ──────────────────────────────────────────────────────

    def start(self):
        self._running = True
        with self._lock:
            configs = [s.config for s in self._states.values()]
        for config in configs:
            t = threading.Thread(target=self._poll_contract, args=(config,), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._running = False
        self._threads.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    def search(self, query: str = "", limit: int = 20) -> list:
        return self.client.search_markets(query=query, limit=limit, min_volume=100)

    def get_api_history(self, token_id: str) -> List[Tuple]:
        return self.client.get_price_history(token_id)

    @property
    def alert_log(self) -> List[dict]:
        return list(self._alert_log[-200:])
