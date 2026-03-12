"""
Centralized configuration for the crypto trading system.
All hardcoded values, paths, and strategy parameters live here.
"""

from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "crypto_data.db"
LOG_PATH = BASE_DIR / "data_collection.log"
CHECKPOINT_PATH = BASE_DIR / "checkpoints.json"

# --- Strategy Parameters ---
TARGET_VOLATILITY = 0.50
INITIAL_CAPITAL = 1000.0
SIGNAL_CLIP = 20

# --- Signal Weights ---
EWMAC_WEIGHTS = [0.08, 0.21, 0.21, 0.08, 0.21, 0.21]
BOLMOM_WEIGHTS = [1 / 6] * 6
BREAKOUT_WEIGHTS = [0.08, 0.21, 0.21, 0.08, 0.21, 0.21]

# Signal family multipliers
EWMAC_MULTIPLIER = 1.30
BOLMOM_MULTIPLIER = 1.30
BREAKOUT_MULTIPLIER = 1.70
COMBINED_MULTIPLIER = 1.05

# Signal family blend weights (must sum to ~1)
FAMILY_WEIGHTS = [0.3333, 0.3333, 0.3333]

# --- Forecast Scalars ---
EWMAC_SCALARS = {
    (2, 8): 0.785,
    (4, 16): 0.545,
    (8, 32): 0.365,
    (16, 64): 0.245,
    (32, 128): 0.168,
    (64, 256): 0.128,
}

BOLMOM_SCALARS = {
    8: 62.01 / 21,
    16: 55.33 / 21,
    32: 52.11 / 21,
    64: 50.41 / 21,
    128: 49.16 / 21,
    256: 48.41 / 21,
}

BREAKOUT_SCALARS = {
    10: 0.714 / 22,
    20: 0.791 / 26,
    40: 0.817 / 28,
    80: 0.837 / 28,
    160: 0.841 / 28,
    320: 0.834 / 26,
}

# --- Lookback Windows ---
EWMAC_COMBINATIONS = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]
BOLMOM_WINDOWS = [8, 16, 32, 64, 128, 256]
BREAKOUT_WINDOWS = [10, 20, 40, 80, 160, 320]

# --- Data Fetcher ---
DEFAULT_SYMBOLS = [
    'AVAXUSDT', 'BNBUSDT', 'SOLUSDT', 'BTCUSDT', 'DOGEUSDT',
    'ETHUSDT', 'HYPEUSDT', 'ADAUSDT', 'LINKUSDT', 'TRXUSDT',
    'SUIUSDT', 'BCHUSDT', 'XLMUSDT', 'XRPUSDT', 'TONUSDT',
]

DEFAULT_START_DATES = {
    'BTCUSDT': '2020-03-25',
    'ETHUSDT': '2020-10-21',
    'default': '2021-01-01',
}

# --- Dashboard Defaults ---
DASHBOARD_INITIAL_CAPITAL = 3040
DASHBOARD_TARGET_VOLATILITY = 0.70
DASHBOARD_DEFAULT_BUFFER = 0.10
POSITION_DIVISOR = 10
