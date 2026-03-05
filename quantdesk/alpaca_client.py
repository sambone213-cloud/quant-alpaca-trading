"""
Alpaca Stock Data Client — drop-in replacement for polymarket_client.py
Uses Alpaca Markets API for real-time quotes, order book, and price history.

Free account: https://alpaca.markets → Paper Trading → get API keys
Data API docs: https://docs.alpaca.markets/reference/stockquotes-1

Price format: raw USD (e.g. 185.42)
Symbol: standard ticker string (e.g. "AAPL", "SPY", "TSLA")

Set environment variables or pass keys directly:
    ALPACA_API_KEY
    ALPACA_API_SECRET

Free tier gives: real-time IEX quotes, 15-min delayed SIP, full historical bars
"""

import os
import time
import threading
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import requests


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class StockMarket:
    """Mirrors PolymarketMarket — represents a tracked stock."""
    symbol: str                 # e.g. "AAPL"
    name: str                   # e.g. "Apple Inc."
    price: float                # latest trade/mid price
    bid: float
    ask: float
    volume_24h: float           # shares traded today
    avg_volume: float           # 30-day average daily volume
    market_cap: float           # USD
    sector: str = ""
    active: bool = True

    # Compatibility shims so feed_manager/filters don't need changes
    @property
    def condition_id(self) -> str:
        return self.symbol

    @property
    def yes_token_id(self) -> str:
        return self.symbol

    @property
    def yes_price(self) -> float:
        return self.mid

    @property
    def no_price(self) -> float:
        return 0.0

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.price

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)

    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid if self.mid > 0 else 0.0

    @property
    def question(self) -> str:
        return f"{self.symbol} — {self.name}"

    @property
    def liquidity(self) -> float:
        return self.volume_24h * self.price

    @property
    def end_date(self) -> str:
        return ""  # stocks don't expire


@dataclass
class StockOrderBook:
    """Mirrors PolymarketOrderBook — L2 order book for a stock."""
    symbol: str
    bids: List[Tuple[float, float]]   # [(price, size_shares), ...]
    asks: List[Tuple[float, float]]
    timestamp: float = field(default_factory=time.time)

    # Compatibility shims
    @property
    def token_id(self) -> str:
        return self.symbol

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask

    @property
    def spread(self) -> float:
        return max(0.0, self.best_ask - self.best_bid)

    @property
    def depth_imbalance(self) -> float:
        """0.0 = all asks (sell pressure), 1.0 = all bids (buy pressure), 0.5 = balanced."""
        bid_depth = sum(p * s for p, s in self.bids[:5])
        ask_depth = sum(p * s for p, s in self.asks[:5])
        total = bid_depth + ask_depth
        return bid_depth / total if total > 0 else 0.5


# ─── Client ───────────────────────────────────────────────────────────────────

class AlpacaClient:
    """
    Alpaca Markets data client.
    Mirrors PolymarketClient interface so the rest of the stack works unchanged.
    """

    DATA_URL   = "https://data.alpaca.markets"
    BROKER_URL = "https://api.alpaca.markets"

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        timeout: int = 10,
        paper: bool = True,
    ):
        # Try Streamlit secrets first, then environment variables
        try:
            import streamlit as st
            self.api_key    = api_key    or st.secrets.get("ALPACA_API_KEY", "")    or os.getenv("ALPACA_API_KEY", "")
            self.api_secret = api_secret or st.secrets.get("ALPACA_API_SECRET", "") or os.getenv("ALPACA_API_SECRET", "")
        except Exception:
            self.api_key    = api_key    or os.getenv("ALPACA_API_KEY", "")
            self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET", "")

        self.timeout = timeout
        self.paper   = paper

        self._headers = {
            "APCA-API-KEY-ID":     self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Accept":              "application/json",
        }

    def _has_keys(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def _data_get(self, path: str, params: dict = None) -> dict:
        url = self.DATA_URL + path
        r = requests.get(url, headers=self._headers, params=params or {}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _broker_get(self, path: str, params: dict = None) -> dict:
        url = self.BROKER_URL + path
        r = requests.get(url, headers=self._headers, params=params or {}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ── Search / Discovery ────────────────────────────────────────────────

    def search_markets(
        self,
        query: str = "",
        limit: int = 25,
        min_volume: float = 0,
        asset_class: str = "us_equity",
        debug: bool = False,
    ) -> List[StockMarket]:
        """
        Search for stocks. Without API keys returns a curated default list.
        With keys, searches Alpaca asset universe and fetches live quotes.
        """
        if not self._has_keys():
            return self._default_watchlist(query, limit)

        try:
            # Get matching assets
            params = {"status": "active", "asset_class": asset_class}
            if query:
                params["q"] = query.upper()
            assets_data = self._broker_get("/v2/assets", params)

            # Filter to tradeable assets with symbols matching query
            q_upper = query.upper()
            assets = [
                a for a in assets_data
                if a.get("tradable") and a.get("status") == "active"
                and (not query or q_upper in a.get("symbol", "") or q_upper in a.get("name", "").upper())
            ][:limit]

            if not assets:
                return []

            # Fetch snapshots (quote + daily bar) for all symbols at once
            symbols = [a["symbol"] for a in assets]
            markets = self._symbols_to_markets(symbols, assets)
            markets = [m for m in markets if m.volume_24h >= min_volume]
            markets.sort(key=lambda m: m.volume_24h, reverse=True)

            if debug:
                print(f"[DEBUG] search_markets: {len(assets)} assets → {len(markets)} parsed")
            return markets

        except Exception as e:
            if debug:
                print(f"[DEBUG] search_markets error: {e}")
            return self._default_watchlist(query, limit)

    def _symbols_to_markets(self, symbols: List[str], asset_meta: list = None) -> List[StockMarket]:
        """Fetch snapshots for a list of symbols and return StockMarket objects."""
        if not symbols:
            return []
        meta_map = {a["symbol"]: a for a in (asset_meta or [])}
        try:
            data = self._data_get(
                "/v2/stocks/snapshots",
                params={"symbols": ",".join(symbols), "feed": "iex"}
            )
            markets = []
            for sym, snap in data.items():
                try:
                    quote      = snap.get("latestQuote", {})
                    trade      = snap.get("latestTrade", {})
                    daily_bar  = snap.get("dailyBar", {})
                    prev_bar   = snap.get("prevDailyBar", {})
                    meta       = meta_map.get(sym, {})

                    bid   = float(quote.get("bp", 0) or 0)
                    ask   = float(quote.get("ap", 0) or 0)
                    price = float(trade.get("p", 0) or 0) or ((bid + ask) / 2 if bid and ask else 0)
                    vol   = float(daily_bar.get("v", 0) or 0)

                    if price <= 0:
                        continue

                    markets.append(StockMarket(
                        symbol=sym,
                        name=meta.get("name", sym),
                        price=price,
                        bid=bid,
                        ask=ask,
                        volume_24h=vol,
                        avg_volume=vol,  # could fetch 30d avg separately
                        market_cap=0.0,
                        sector=meta.get("exchange", ""),
                        active=True,
                    ))
                except Exception:
                    continue
            return markets
        except Exception as e:
            print(f"[DEBUG] snapshot fetch error: {e}")
            return []

    def _default_watchlist(self, query: str = "", limit: int = 25) -> List[StockMarket]:
        """
        Fallback list when no API keys are set.
        Returns popular tickers filtered by query.
        """
        defaults = [
            ("SPY",  "S&P 500 ETF",            580.0,  100_000_000),
            ("QQQ",  "Nasdaq 100 ETF",          490.0,   60_000_000),
            ("AAPL", "Apple Inc.",              215.0,   55_000_000),
            ("MSFT", "Microsoft Corp.",         415.0,   22_000_000),
            ("NVDA", "NVIDIA Corp.",            875.0,   45_000_000),
            ("TSLA", "Tesla Inc.",              175.0,   90_000_000),
            ("AMZN", "Amazon.com Inc.",         185.0,   35_000_000),
            ("GOOGL","Alphabet Inc.",           170.0,   25_000_000),
            ("META", "Meta Platforms",          520.0,   18_000_000),
            ("BRK.B","Berkshire Hathaway B",    365.0,    5_000_000),
            ("JPM",  "JPMorgan Chase",          205.0,    8_000_000),
            ("V",    "Visa Inc.",               280.0,    6_000_000),
            ("XOM",  "Exxon Mobil",             115.0,   15_000_000),
            ("GLD",  "Gold ETF",                235.0,    8_000_000),
            ("TLT",  "20yr Treasury ETF",        95.0,   20_000_000),
            ("IWM",  "Russell 2000 ETF",        210.0,   30_000_000),
            ("VIX",  "Volatility Index",         18.0,        0),
            ("AMD",  "Advanced Micro Devices",  165.0,   45_000_000),
            ("PLTR", "Palantir Technologies",    25.0,  120_000_000),
            ("COIN", "Coinbase Global",         245.0,   12_000_000),
        ]
        q = query.upper()
        filtered = [
            d for d in defaults
            if not q or q in d[0] or q in d[1].upper()
        ][:limit]

        return [
            StockMarket(
                symbol=sym, name=name,
                price=price, bid=price * 0.9995, ask=price * 1.0005,
                volume_24h=vol, avg_volume=vol,
                market_cap=0.0, sector="", active=True,
            )
            for sym, name, price, vol in filtered
        ]

    # ── Live Quote & Order Book ───────────────────────────────────────────

    def get_order_book(self, symbol: str) -> Optional[StockOrderBook]:
        """Fetch L2 order book for a symbol."""
        if not self._has_keys():
            return self._mock_order_book(symbol)
        try:
            data = self._data_get(f"/v2/stocks/{symbol}/quotes/latest",
                                   params={"feed": "iex"})
            quote = data.get("quote", {})
            bid   = float(quote.get("bp", 0) or 0)
            ask   = float(quote.get("ap", 0) or 0)
            bsz   = float(quote.get("bs", 1) or 1)
            asz   = float(quote.get("as", 1) or 1)

            if bid <= 0 and ask <= 0:
                return None

            # IEX gives us NBBO — build synthetic L2 with small depth
            bids = [(bid, bsz), (bid * 0.999, bsz * 2), (bid * 0.998, bsz * 3)]
            asks = [(ask, asz), (ask * 1.001, asz * 2), (ask * 1.002, asz * 3)]

            return StockOrderBook(symbol=symbol, bids=bids, asks=asks)
        except Exception as e:
            print(f"[order_book] {symbol}: {e}")
            return None

    def get_midpoint(self, symbol: str) -> Optional[float]:
        """Returns current mid price."""
        ob = self.get_order_book(symbol)
        if ob:
            return ob.mid
        # Fallback: latest trade
        try:
            data  = self._data_get(f"/v2/stocks/{symbol}/trades/latest",
                                    params={"feed": "iex"})
            price = data.get("trade", {}).get("p")
            return float(price) if price else None
        except Exception:
            return None

    def poll_price(self, symbol: str) -> Optional[float]:
        """Returns latest mid price. Called by feed_manager polling loop."""
        return self.get_midpoint(symbol)

    def _mock_order_book(self, symbol: str, price: float = None) -> StockOrderBook:
        """Returns a synthetic order book when no API keys are set."""
        prices = {s: p for s, n, p, v in [
            ("SPY", "SPY ETF", 580.0, 0), ("QQQ", "QQQ ETF", 490.0, 0),
            ("AAPL", "Apple", 215.0, 0), ("MSFT", "Microsoft", 415.0, 0),
            ("NVDA", "NVIDIA", 875.0, 0), ("TSLA", "Tesla", 175.0, 0),
        ]}
        p = price or prices.get(symbol, 100.0)
        spread = p * 0.0005
        bid, ask = p - spread, p + spread
        return StockOrderBook(
            symbol=symbol,
            bids=[(bid, 100), (bid * 0.999, 250), (bid * 0.998, 500)],
            asks=[(ask, 100), (ask * 1.001, 250), (ask * 1.002, 500)],
        )

    # ── Price History ─────────────────────────────────────────────────────

    def get_price_history(
        self,
        symbol: str,
        timeframe: str = "1Min",
        limit: int = 500,
    ) -> List[Tuple[int, float]]:
        """
        Returns [(timestamp_unix, close_price), ...] sorted oldest→newest.
        timeframe options: "1Min", "5Min", "15Min", "1Hour", "1Day"
        """
        if not self._has_keys():
            return self._mock_price_history(symbol, limit)
        try:
            data = self._data_get(
                f"/v2/stocks/{symbol}/bars",
                params={"timeframe": timeframe, "limit": limit,
                        "feed": "iex", "sort": "asc"}
            )
            bars = data.get("bars", [])
            return [
                (int(time.mktime(time.strptime(b["t"][:19], "%Y-%m-%dT%H:%M:%S"))),
                 float(b["c"]))
                for b in bars if "t" in b and "c" in b
            ]
        except Exception as e:
            print(f"[price_history] {symbol}: {e}")
            return []

    def _mock_price_history(self, symbol: str, limit: int = 200) -> List[Tuple[int, float]]:
        """Synthetic random walk for demo when no keys set."""
        import numpy as np
        base_prices = {"SPY": 580, "QQQ": 490, "AAPL": 215, "MSFT": 415,
                       "NVDA": 875, "TSLA": 175, "AMZN": 185}
        p0 = base_prices.get(symbol, 100.0)
        np.random.seed(hash(symbol) % 2**31)
        returns = np.random.normal(0, 0.002, limit)
        prices  = p0 * np.exp(np.cumsum(returns))
        now     = int(time.time())
        step    = 60  # 1 minute bars
        return [(now - (limit - i) * step, float(prices[i])) for i in range(limit)]

    # ── Multiple Symbols ──────────────────────────────────────────────────

    def get_multiple_order_books(self, symbols: List[str]) -> List[Optional[StockOrderBook]]:
        return [self.get_order_book(s) for s in symbols]

    # ── Asset Info ────────────────────────────────────────────────────────

    def get_asset_info(self, symbol: str) -> dict:
        """Returns basic asset metadata."""
        if not self._has_keys():
            return {"symbol": symbol, "name": symbol, "exchange": ""}
        try:
            return self._broker_get(f"/v2/assets/{symbol}")
        except Exception:
            return {"symbol": symbol, "name": symbol, "exchange": ""}


# ─── Compatibility aliases ────────────────────────────────────────────────────
# So any code that imports PolymarketClient or PolymarketMarket still works

PolymarketClient   = AlpacaClient
PolymarketMarket   = StockMarket
PolymarketOrderBook = StockOrderBook

