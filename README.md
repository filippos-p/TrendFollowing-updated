Crypto Trading Dashboard
========================

Quick Start:
  cd "C:\Users\Los\Downloads\Trend with Live data"
  python data_fetcher.py          # fetch/update price data from Bybit
  streamlit run dashboard.py      # launch live trading dashboard

Project Structure:
  config.py        - All configuration: paths, strategy params, scalars, symbols
  strategy.py      - Signal generation (EWMAC, BOLMOM, BREAKOUT) + single-symbol backtest
  portfolio.py     - Portfolio management, DB helpers, position sizing, trade execution
  dashboard.py     - Streamlit UI (4 tabs: positions, overview, trade log, system status)
  data_fetcher.py  - Bybit REST API data fetcher with SQLite storage

  _archive/        - Old files kept for reference (not used by the active system)
