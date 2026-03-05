# Quant PM Stack

A Polymarket live trading dashboard with particle filter probability estimation, Kelly bet sizing, and agent-based market simulation.

## Features
- 📡 **Live Feed** — real-time Polymarket order book tracking with SMC particle filter
- 🎯 **Match Simulation** — calibrate simulator to mirror real contracts
- 🧮 **Kelly & Risk Engine** — fractional Kelly sizing with CVaR
- 🔵 **Dependency Copulas** — multi-contract correlation analysis
- 📐 **Math Reference** — full derivations for every model

## Setup

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### File structure
```
app.py                  ← main simulator (ABM, Copulas, Tail Risk, Kelly)
pages/
  1_FAQ.py              ← math reference
  2_Live_Feed.py        ← live Polymarket dashboard
  3_Sim_Match.py        ← match simulation to contract
feed_manager.py         ← multi-contract polling engine
filters.py              ← particle filter (SMC)
market.py               ← agent-based market model
polymarket_client.py    ← Polymarket API client
simulators.py           ← Monte Carlo, copulas, Kelly
requirements.txt
.streamlit/config.toml
```

## No API key required
All Polymarket data is public. No authentication needed.
