"""
Portfolio management: live position tracking, trade execution, PnL,
mark-to-market, and multi-symbol dynamic backtest.
"""

import sqlite3
import time
import logging
from contextlib import contextmanager
from datetime import datetime, date

import numpy as np
import pandas as pd
from numba import njit

import requests

import config as cfg

log = logging.getLogger(__name__)

BYBIT_TICKERS_URL = "https://api.bybit.com/v5/market/tickers"


# ========================================================================
# Database helpers
# ========================================================================

@contextmanager
def db_connection(db_path=None, retries=3):
    """Yield an optimised SQLite connection with WAL mode and retry logic."""
    path = str(db_path or cfg.DB_PATH)
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(path, timeout=30 * (attempt + 1), check_same_thread=False)
            conn.execute('PRAGMA journal_mode = WAL')
            conn.execute('PRAGMA synchronous = NORMAL')
            conn.execute('PRAGMA cache_size = -524288')
            conn.execute('PRAGMA temp_store = MEMORY')
            conn.execute('PRAGMA mmap_size = 536870912')
            conn.execute('PRAGMA busy_timeout = 15000')
            conn.execute('PRAGMA optimize')
            try:
                yield conn
            finally:
                conn.close()
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                raise


def _migrate_schema(conn):
    """Detect old id-based PKs and recreate tables with correct PKs.

    The original code used ``id INTEGER PRIMARY KEY AUTOINCREMENT`` on
    ``current_positions`` and ``daily_portfolio_log``.  That makes
    ``INSERT OR REPLACE`` key on ``id`` (always new) instead of
    ``symbol`` / ``date``, causing duplicate rows on every write.

    This migration preserves existing data in ``trade_execution_log``
    (which legitimately uses an autoincrement id) and
    ``strategy_signals`` / ``price_data`` (untouched).
    """
    # -- current_positions: PK should be symbol, not id --
    cols = conn.execute("PRAGMA table_info(current_positions)").fetchall()
    if cols:
        pk_col = next((c[1] for c in cols if c[5]), None)
        if pk_col == 'id':
            log.info("Migrating current_positions: id PK -> symbol PK")
            rows = conn.execute(
                "SELECT symbol, position_usd, position_coins, entry_price, last_updated "
                "FROM current_positions"
            ).fetchall()
            conn.execute("DROP TABLE current_positions")
            conn.execute("""
                CREATE TABLE current_positions (
                    symbol TEXT PRIMARY KEY,
                    position_usd REAL DEFAULT 0,
                    position_coins REAL DEFAULT 0,
                    entry_price REAL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Re-insert keeping only the latest row per symbol
            seen = {}
            for r in rows:
                seen[r[0]] = r  # last wins
            for r in seen.values():
                conn.execute(
                    "INSERT OR REPLACE INTO current_positions VALUES (?,?,?,?,?)", r)

    # -- daily_portfolio_log: PK should be date, not id --
    cols = conn.execute("PRAGMA table_info(daily_portfolio_log)").fetchall()
    if cols:
        pk_col = next((c[1] for c in cols if c[5]), None)
        if pk_col == 'id':
            log.info("Migrating daily_portfolio_log: id PK -> date PK")
            # Keep only the latest entry per date
            rows = conn.execute(
                "SELECT date, total_capital, available_cash, total_exposure, "
                "unrealized_pnl, realized_pnl, daily_pnl, cumulative_pnl, "
                "num_positions, portfolio_volatility, created_at "
                "FROM daily_portfolio_log ORDER BY date ASC, ROWID ASC"
            ).fetchall()
            conn.execute("DROP TABLE daily_portfolio_log")
            conn.execute("""
                CREATE TABLE daily_portfolio_log (
                    date DATE PRIMARY KEY,
                    total_capital REAL NOT NULL,
                    available_cash REAL NOT NULL,
                    total_exposure REAL NOT NULL,
                    unrealized_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    daily_pnl REAL DEFAULT 0,
                    cumulative_pnl REAL DEFAULT 0,
                    num_positions INTEGER DEFAULT 0,
                    portfolio_volatility REAL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            seen = {}
            for r in rows:
                seen[r[0]] = r  # last per date wins
            for r in seen.values():
                conn.execute(
                    "INSERT OR REPLACE INTO daily_portfolio_log VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?)", r)

    conn.commit()


def ensure_tables():
    """Create all portfolio-tracking tables if they don't exist, and
    migrate old schemas that used ``id`` as PK."""
    with db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS current_positions (
                symbol TEXT PRIMARY KEY,
                position_usd REAL DEFAULT 0,
                position_coins REAL DEFAULT 0,
                entry_price REAL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_portfolio_log (
                date DATE PRIMARY KEY,
                total_capital REAL NOT NULL,
                available_cash REAL NOT NULL,
                total_exposure REAL NOT NULL,
                unrealized_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                daily_pnl REAL DEFAULT 0,
                cumulative_pnl REAL DEFAULT 0,
                num_positions INTEGER DEFAULT 0,
                portfolio_volatility REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_execution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                amount_usd REAL NOT NULL,
                amount_coins REAL NOT NULL,
                price REAL NOT NULL,
                reason TEXT,
                position_before_usd REAL,
                position_after_usd REAL,
                capital_before REAL,
                capital_after REAL,
                realized_pnl REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS buffer_settings (
                symbol TEXT PRIMARY KEY,
                buffer_pct REAL DEFAULT 0.10,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                category TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                close REAL,
                volume REAL DEFAULT 0,
                combined_signal REAL DEFAULT 0,
                ewmac_combined REAL DEFAULT 0,
                bolmom_combined REAL DEFAULT 0,
                breakout_combined REAL DEFAULT 0,
                vol_forecast REAL DEFAULT 0,
                ewmac_2_8 REAL DEFAULT 0, ewmac_4_16 REAL DEFAULT 0,
                ewmac_8_32 REAL DEFAULT 0, ewmac_16_64 REAL DEFAULT 0,
                ewmac_32_128 REAL DEFAULT 0, ewmac_64_256 REAL DEFAULT 0,
                bolmom_8 REAL DEFAULT 0, bolmom_16 REAL DEFAULT 0,
                bolmom_32 REAL DEFAULT 0, bolmom_64 REAL DEFAULT 0,
                bolmom_128 REAL DEFAULT 0, bolmom_256 REAL DEFAULT 0,
                breakout_10 REAL DEFAULT 0, breakout_20 REAL DEFAULT 0,
                breakout_40 REAL DEFAULT 0, breakout_80 REAL DEFAULT 0,
                breakout_160 REAL DEFAULT 0, breakout_320 REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, interval, category, timestamp)
            )
        """)
        conn.commit()
        # Migrate old schemas (id PK -> symbol/date PK)
        _migrate_schema(conn)


def optimize_indexes():
    """Create dashboard-optimised indexes and ANALYZE tables."""
    idx_stmts = [
        "CREATE INDEX IF NOT EXISTS idx_price_optimized ON price_data(interval, category, symbol, timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_price_latest ON price_data(symbol, interval, category, timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_signals_optimized ON strategy_signals(interval, category, symbol, timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_signals_latest ON strategy_signals(symbol, timestamp DESC) WHERE interval='1d' AND category='linear'",
        "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON current_positions(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_portfolio_date ON daily_portfolio_log(date DESC)",
        "CREATE INDEX IF NOT EXISTS idx_trades_date ON trade_execution_log(timestamp DESC)",
    ]
    with db_connection() as conn:
        for sql in idx_stmts:
            try:
                conn.execute(sql)
            except Exception:
                pass
        for tbl in ('price_data', 'strategy_signals', 'current_positions',
                    'daily_portfolio_log', 'trade_execution_log'):
            try:
                conn.execute(f"ANALYZE {tbl}")
            except Exception:
                pass
        conn.commit()


# ========================================================================
# Live price fetch (Bybit API — no DB, no auth required)
# ========================================================================

def fetch_live_prices(symbols, category='linear', save_to_db=True):
    """Fetch current last-traded prices from Bybit and optionally save to DB.

    Hits Bybit's ``/v5/market/tickers`` (no auth, public endpoint).
    When ``save_to_db=True`` the fetched prices are written into
    ``price_data`` as today's 1d candle (INSERT OR REPLACE) so the
    rest of the dashboard sees them immediately.

    Returns dict {symbol: price}.
    Falls back to DB prices on any network error.
    """
    result = {}
    try:
        resp = requests.get(
            BYBIT_TICKERS_URL,
            params={'category': category},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get('retCode') != 0:
            log.warning("Bybit tickers API error: %s", data.get('retMsg'))
            return _fallback_prices(symbols)
        sym_set = set(symbols)
        for item in data.get('result', {}).get('list', []):
            sym = item.get('symbol', '')
            if sym in sym_set:
                price = float(item.get('lastPrice', 0))
                if price > 0:
                    result[sym] = price
    except Exception as e:
        log.warning("Live price fetch failed: %s — falling back to DB", e)
        return _fallback_prices(symbols)

    # Fill any missing symbols from DB
    missing = [s for s in symbols if s not in result]
    if missing:
        fb = _fallback_prices(missing)
        result.update(fb)

    # Persist live prices into price_data so the rest of the dashboard
    # picks them up without a full data update.
    if save_to_db and result:
        _save_live_prices(result, category)

    return result


def _save_live_prices(prices_dict, category='linear'):
    """Write live ticker prices into price_data as today's 1d row.

    Uses INSERT OR REPLACE keyed on (symbol, timestamp, interval, category)
    so it updates today's candle close without creating duplicates.
    """
    now_str = datetime.now().strftime('%Y-%m-%d 00:00:00')
    rows = []
    for sym, price in prices_dict.items():
        # We only know the last price — set OHLC all to it (volume 0).
        # If a proper candle already exists for today it will be replaced
        # only with the close updated; but since we don't have real OHLCV
        # from the ticker, we read the existing row first and only update close.
        rows.append((sym, now_str, price, price, price, price, 0, '1d', category))

    try:
        with db_connection() as conn:
            # Update close on existing rows, insert new ones
            for row in rows:
                existing = conn.execute(
                    "SELECT open, high, low, volume FROM price_data "
                    "WHERE symbol=? AND timestamp=? AND interval=? AND category=?",
                    (row[0], row[1], '1d', category),
                ).fetchone()
                if existing:
                    # Keep original OHLV, only update close to live price
                    conn.execute(
                        "UPDATE price_data SET close=? "
                        "WHERE symbol=? AND timestamp=? AND interval=? AND category=?",
                        (row[5], row[0], row[1], '1d', category),
                    )
                else:
                    conn.execute(
                        "INSERT OR REPLACE INTO price_data "
                        "(symbol,timestamp,open,high,low,close,volume,interval,category) "
                        "VALUES (?,?,?,?,?,?,?,?,?)", row,
                    )
            conn.commit()
    except Exception as e:
        log.warning("Failed to save live prices to DB: %s", e)


def _fallback_prices(symbols):
    """Get last known prices from the DB as fallback."""
    prices = get_latest_prices(list(symbols), lookback_days=1)
    return {r['symbol']: r['close'] for _, r in prices.iterrows()} if not prices.empty else {}


def fast_update_prices(symbols, category='linear'):
    """Fast daily price update using the tickers endpoint (1 API call for all symbols).

    Instead of making N separate kline requests, this pulls OHLCV from
    Bybit's ``/v5/market/tickers`` which returns data for every perpetual
    in a single response.  The fields map to today's candle:

        prevPrice24h -> open, highPrice24h -> high,
        lowPrice24h  -> low,  lastPrice    -> close,
        volume24h    -> volume

    Returns dict {symbol: price} on success, empty dict on failure.
    """
    result = {}
    try:
        resp = requests.get(
            BYBIT_TICKERS_URL,
            params={'category': category},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get('retCode') != 0:
            log.warning("Bybit tickers API error: %s", data.get('retMsg'))
            return {}

        sym_set = set(symbols)
        now_str = datetime.now().strftime('%Y-%m-%d 00:00:00')
        rows = []
        for item in data.get('result', {}).get('list', []):
            sym = item.get('symbol', '')
            if sym not in sym_set:
                continue
            try:
                o = float(item.get('prevPrice24h', 0))
                h = float(item.get('highPrice24h', 0))
                lo = float(item.get('lowPrice24h', 0))
                c = float(item.get('lastPrice', 0))
                v = float(item.get('volume24h', 0))
            except (ValueError, TypeError):
                continue
            if c <= 0:
                continue
            result[sym] = c
            rows.append((sym, now_str, o, h, lo, c, v, '1d', category))

        if rows:
            with db_connection() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO price_data "
                    "(symbol,timestamp,open,high,low,close,volume,interval,category) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
            log.info("Fast-updated %d symbols via tickers endpoint", len(rows))

    except Exception as e:
        log.warning("fast_update_prices failed: %s", e)
    return result


BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"


def backfill_prices(symbols, days=30, category='linear', progress_cb=None):
    """Fetch the last *days* of daily klines for each symbol and fill gaps.

    Uses Bybit ``/v5/market/kline`` (interval=D, limit=days).  Each symbol
    requires one API call so this is N calls total, but it guarantees there
    are no missing daily candles in the DB.

    Parameters
    ----------
    symbols : list[str]
    days : int  – how many calendar days to backfill (default 30)
    category : str
    progress_cb : callable(float) – optional 0→1 progress callback

    Returns
    -------
    dict  {symbol: rows_inserted}
    """
    from datetime import timedelta

    end_ms = int(datetime.now().timestamp() * 1000)
    start_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    result = {}

    for idx, sym in enumerate(symbols):
        try:
            resp = requests.get(
                BYBIT_KLINE_URL,
                params={
                    'category': category,
                    'symbol': sym,
                    'interval': 'D',
                    'start': start_ms,
                    'end': end_ms,
                    'limit': min(days + 1, 1000),
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get('retCode') != 0:
                log.warning("Kline API error for %s: %s", sym, data.get('retMsg'))
                result[sym] = 0
                continue

            klines = data.get('result', {}).get('list', [])
            if not klines:
                result[sym] = 0
                continue

            rows = []
            for k in klines:
                try:
                    ts = datetime.utcfromtimestamp(float(k[0]) / 1000).strftime('%Y-%m-%d 00:00:00')
                    o, h, lo, c, v = (float(k[1]), float(k[2]),
                                      float(k[3]), float(k[4]), float(k[5]))
                except (ValueError, TypeError, IndexError):
                    continue
                rows.append((sym, ts, o, h, lo, c, v, '1d', category))

            if rows:
                with db_connection() as conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO price_data "
                        "(symbol,timestamp,open,high,low,close,volume,interval,category) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        rows,
                    )
                    conn.commit()
            result[sym] = len(rows)
            log.info("Backfilled %d daily candles for %s", len(rows), sym)

        except Exception as e:
            log.warning("backfill_prices failed for %s: %s", sym, e)
            result[sym] = 0

        if progress_cb:
            progress_cb((idx + 1) / len(symbols))

        # Small delay to avoid rate-limiting
        time.sleep(0.15)

    return result


# ========================================================================
# Signal computation — fast incremental path
# ========================================================================

# Maximum lookback needed by any signal rule (breakout_320 + some margin)
_SIGNAL_LOOKBACK = 400


def compute_signals_fast(symbols, strategy_module, progress_cb=None):
    """Compute signals reading only the last N rows per symbol.

    Functionally identical to ``compute_and_save_signals`` but avoids
    reading full price history (~2000 rows) when only the tail is needed
    for signal calculation.  Saves only the portion that changed.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _compute_one(symbol):
        try:
            with db_connection() as conn:
                # Count total rows to know if we can use the fast path
                total = conn.execute(
                    "SELECT COUNT(*) FROM price_data "
                    "WHERE symbol=? AND interval='1d' AND category='linear'",
                    (symbol,),
                ).fetchone()[0]
                if total < 100:
                    return symbol, None, None, f"Insufficient data ({total} days)"

                # Read only the tail needed for signal computation
                read_rows = min(total, _SIGNAL_LOOKBACK)
                df = pd.read_sql_query(
                    "SELECT timestamp, open, high, low, close, volume "
                    "FROM price_data WHERE symbol=? AND interval='1d' AND category='linear' "
                    "ORDER BY timestamp DESC LIMIT ?",
                    conn, params=(symbol, read_rows),
                )
            # Reverse so oldest-first
            df = df.iloc[::-1].reset_index(drop=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_out = strategy_module.add_signals(df)
            # Only save the last 5 rows (today + a few days buffer)
            tail = df_out.tail(5)
            return symbol, df_out, tail, "OK"
        except Exception as e:
            return symbol, None, None, str(e)

    workers = min(4, len(symbols))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_compute_one, symbols))

    ok, err, msgs = 0, 0, []
    for i, (sym, df_full, df_tail, msg) in enumerate(results):
        if progress_cb:
            progress_cb((i + 1) / len(results))
        if df_tail is not None:
            _save_signals_tail(sym, df_tail)
            ok += 1
            msgs.append(f"OK  {sym}")
        else:
            err += 1
            msgs.append(f"ERR {sym}: {msg}")
    return ok, err, msgs


def _save_signals_tail(symbol, df):
    """Upsert only the last few signal rows (fast incremental save)."""
    now_str = datetime.now().isoformat()
    valid = df.dropna(subset=['timestamp', 'close']).copy()
    if valid.empty:
        return

    signal_cols = [
        'volume', 'combined_signal', 'ewmac_combined', 'bolmom_combined',
        'breakout_combined', 'vol_forecast',
        'ewmac_2_8', 'ewmac_4_16', 'ewmac_8_32', 'ewmac_16_64', 'ewmac_32_128', 'ewmac_64_256',
        'bolmom_8', 'bolmom_16', 'bolmom_32', 'bolmom_64', 'bolmom_128', 'bolmom_256',
        'breakout_10', 'breakout_20', 'breakout_40', 'breakout_80',
        'breakout_160', 'breakout_320',
    ]
    for col in signal_cols:
        if col not in valid.columns:
            valid[col] = 0.0
    valid[signal_cols] = valid[signal_cols].fillna(0.0)

    timestamps = valid['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').values
    closes = valid['close'].values
    arrays = [valid[c].values for c in signal_cols]

    rows = [
        (symbol, '1d', 'linear', timestamps[i], float(closes[i]),
         *[float(a[i]) for a in arrays], now_str)
        for i in range(len(valid))
    ]

    with db_connection() as conn:
        # Delete only the dates we're replacing, not the entire history
        ts_list = list(timestamps)
        ph = ','.join('?' * len(ts_list))
        conn.execute(
            f"DELETE FROM strategy_signals "
            f"WHERE symbol=? AND interval='1d' AND category='linear' "
            f"AND timestamp IN ({ph})",
            [symbol] + ts_list,
        )
        conn.executemany(
            "INSERT INTO strategy_signals "
            "(symbol,interval,category,timestamp,close,volume,combined_signal,"
            "ewmac_combined,bolmom_combined,breakout_combined,vol_forecast,"
            "ewmac_2_8,ewmac_4_16,ewmac_8_32,ewmac_16_64,ewmac_32_128,ewmac_64_256,"
            "bolmom_8,bolmom_16,bolmom_32,bolmom_64,bolmom_128,bolmom_256,"
            "breakout_10,breakout_20,breakout_40,breakout_80,breakout_160,breakout_320,"
            "created_at) VALUES (" + ",".join(["?"] * 30) + ")",
            rows,
        )
        conn.commit()


# ========================================================================
# Data query helpers (used by the dashboard)
# ========================================================================

def get_trading_universe():
    """Return list of symbols that have price data."""
    with db_connection() as conn:
        df = pd.read_sql_query(
            "SELECT DISTINCT symbol FROM price_data "
            "WHERE interval='1d' AND category='linear' ORDER BY symbol",
            conn,
        )
    return df['symbol'].tolist()


def get_latest_prices(symbols, lookback_days=1):
    """Get the most recent close price for each symbol."""
    if not symbols:
        return pd.DataFrame()
    ph = ','.join('?' * len(symbols))
    query = f"""
        WITH ranked AS (
            SELECT symbol, timestamp, close, volume,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
            FROM price_data
            WHERE symbol IN ({ph}) AND interval='1d' AND category='linear'
        )
        SELECT symbol, timestamp, close, volume FROM ranked WHERE rn <= {lookback_days}
        ORDER BY symbol, timestamp DESC
    """
    with db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=symbols)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_latest_signals(symbols):
    """Get latest combined_signal per symbol, plus freshly computed volatility."""
    if not symbols:
        return pd.DataFrame()
    ph = ','.join('?' * len(symbols))
    query = f"""
        WITH latest AS (
            SELECT symbol, combined_signal, timestamp,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
            FROM strategy_signals
            WHERE symbol IN ({ph}) AND interval='1d' AND category='linear'
              AND combined_signal IS NOT NULL
        )
        SELECT symbol, combined_signal, timestamp FROM latest WHERE rn = 1
    """
    with db_connection() as conn:
        try:
            signals_df = pd.read_sql_query(query, conn, params=symbols)
        except Exception:
            signals_df = pd.DataFrame({
                'symbol': symbols,
                'combined_signal': [0.0] * len(symbols),
                'timestamp': [datetime.now()] * len(symbols),
            })

    vol_df = _compute_volatilities(symbols)
    result = pd.merge(signals_df, vol_df, on='symbol', how='left')
    result['volatility'] = result['volatility'].fillna(0.02)
    return result


def _compute_volatilities(symbols):
    """Compute realised daily volatility for each symbol from last 35 days.

    Uses a single SQL query with window functions instead of N separate queries.
    """
    if not symbols:
        return pd.DataFrame(columns=['symbol', 'volatility'])
    ph = ','.join('?' * len(symbols))
    query = f"""
        WITH ranked AS (
            SELECT symbol, close,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
            FROM price_data
            WHERE symbol IN ({ph}) AND interval='1d' AND category='linear'
        )
        SELECT symbol, close FROM ranked WHERE rn <= 35
        ORDER BY symbol, rn DESC
    """
    with db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=list(symbols))

    if df.empty:
        return pd.DataFrame({'symbol': symbols, 'volatility': [0.02] * len(symbols)})

    records = []
    for sym in symbols:
        sdf = df[df['symbol'] == sym]
        if len(sdf) < 10:
            records.append({'symbol': sym, 'volatility': 0.02})
            continue
        prices = sdf['close'].values.astype(np.float64)
        rets = np.diff(prices) / prices[:-1]
        rets = rets[np.isfinite(rets)]
        vol = float(np.std(rets, ddof=1)) if len(rets) >= 5 else 0.02
        records.append({'symbol': sym, 'volatility': vol})
    return pd.DataFrame(records)


# ========================================================================
# Signal computation + batch save
# ========================================================================

def compute_and_save_signals(symbols, strategy_module, progress_cb=None):
    """Compute signals for *symbols* and write them to the DB in batch."""
    from concurrent.futures import ThreadPoolExecutor

    def _compute_one(symbol):
        try:
            with db_connection() as conn:
                df = pd.read_sql_query(
                    "SELECT timestamp, open, high, low, close, volume "
                    "FROM price_data WHERE symbol=? AND interval='1d' AND category='linear' "
                    "ORDER BY timestamp",
                    conn, params=(symbol,),
                )
            if len(df) < 100:
                return symbol, None, f"Insufficient data ({len(df)} days)"
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_out = strategy_module.add_signals(df)
            return symbol, df_out, "OK"
        except Exception as e:
            return symbol, None, str(e)

    workers = min(4, len(symbols))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_compute_one, symbols))

    ok, err, msgs = 0, 0, []
    for i, (sym, df_sig, msg) in enumerate(results):
        if progress_cb:
            progress_cb((i + 1) / len(results))
        if df_sig is not None:
            _save_signals_batch(sym, df_sig)
            ok += 1
            msgs.append(f"OK  {sym}")
        else:
            err += 1
            msgs.append(f"ERR {sym}: {msg}")
    return ok, err, msgs


def _save_signals_batch(symbol, df):
    """Delete old signals for *symbol* and insert the new batch.

    Uses vectorised column access instead of iterrows() for speed.
    """
    now_str = datetime.now().isoformat()
    valid = df.dropna(subset=['timestamp', 'close']).copy()
    if valid.empty:
        return

    # Vectorised: build all columns as arrays, then zip into rows
    signal_cols = [
        'volume', 'combined_signal', 'ewmac_combined', 'bolmom_combined',
        'breakout_combined', 'vol_forecast',
        'ewmac_2_8', 'ewmac_4_16', 'ewmac_8_32', 'ewmac_16_64', 'ewmac_32_128', 'ewmac_64_256',
        'bolmom_8', 'bolmom_16', 'bolmom_32', 'bolmom_64', 'bolmom_128', 'bolmom_256',
        'breakout_10', 'breakout_20', 'breakout_40', 'breakout_80',
        'breakout_160', 'breakout_320',
    ]
    # Fill missing columns with 0, NaN with 0
    for col in signal_cols:
        if col not in valid.columns:
            valid[col] = 0.0
    valid[signal_cols] = valid[signal_cols].fillna(0.0)

    timestamps = valid['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').values
    closes = valid['close'].values
    arrays = [valid[c].values for c in signal_cols]

    rows = [
        (symbol, '1d', 'linear', timestamps[i], float(closes[i]),
         *[float(a[i]) for a in arrays], now_str)
        for i in range(len(valid))
    ]

    with db_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute(
            "DELETE FROM strategy_signals WHERE symbol=? AND interval='1d' AND category='linear'",
            (symbol,),
        )
        conn.executemany(
            "INSERT INTO strategy_signals "
            "(symbol,interval,category,timestamp,close,volume,combined_signal,"
            "ewmac_combined,bolmom_combined,breakout_combined,vol_forecast,"
            "ewmac_2_8,ewmac_4_16,ewmac_8_32,ewmac_16_64,ewmac_32_128,ewmac_64_256,"
            "bolmom_8,bolmom_16,bolmom_32,bolmom_64,bolmom_128,bolmom_256,"
            "breakout_10,breakout_20,breakout_40,breakout_80,breakout_160,breakout_320,"
            "created_at) VALUES (" + ",".join(["?"] * 30) + ")",
            rows,
        )
        conn.execute('COMMIT')


# ========================================================================
# Live Portfolio Manager
# ========================================================================

class PortfolioManager:
    """Manages live positions, trade execution, and mark-to-market."""

    def __init__(self, initial_capital=cfg.DASHBOARD_INITIAL_CAPITAL,
                 target_volatility=cfg.DASHBOARD_TARGET_VOLATILITY,
                 default_buffer_pct=cfg.DASHBOARD_DEFAULT_BUFFER):
        self.initial_capital = initial_capital
        self.target_volatility = target_volatility
        self.default_buffer_pct = default_buffer_pct
        ensure_tables()

    # ---- capital bookkeeping ----

    def get_portfolio_capital(self):
        """Return (total_capital, available_cash, realized_pnl, cumulative_pnl).

        Capital is always derived from ``initial_capital`` (the slider value)
        plus cumulative realized PnL from the trade log.  This means changing
        the initial capital slider immediately recalculates everything.
        """
        cum_rpnl = 0.0
        exposure = 0.0
        try:
            with db_connection() as conn:
                cum_rpnl = conn.execute(
                    "SELECT COALESCE(SUM(realized_pnl),0) FROM trade_execution_log"
                ).fetchone()[0]
                exposure = conn.execute(
                    "SELECT COALESCE(SUM(ABS(position_usd)),0) FROM current_positions "
                    "WHERE ABS(position_coins) > 1e-12"
                ).fetchone()[0]
        except Exception:
            pass

        total_capital = float(self.initial_capital) + cum_rpnl
        available_cash = total_capital - exposure
        return total_capital, available_cash, cum_rpnl, cum_rpnl

    # ---- positions ----

    def get_current_positions(self):
        with db_connection() as conn:
            return pd.read_sql_query(
                "SELECT symbol, position_usd, position_coins, entry_price, last_updated "
                "FROM current_positions WHERE position_coins != 0 "
                "ORDER BY ABS(position_usd) DESC", conn,
            )

    def get_buffer_setting(self, symbol):
        try:
            with db_connection() as conn:
                r = pd.read_sql_query(
                    "SELECT buffer_pct FROM buffer_settings WHERE symbol=?",
                    conn, params=(symbol,),
                )
            if not r.empty:
                return r['buffer_pct'].iloc[0]
        except Exception:
            pass
        return self.default_buffer_pct

    # ---- unrealised PnL ----

    def calc_unrealized_pnl(self, positions_df, latest_prices):
        """Compute total unrealised PnL across all open positions."""
        total = 0.0
        if positions_df.empty:
            return total
        for _, pos in positions_df.iterrows():
            sym = pos['symbol']
            if sym not in latest_prices:
                continue
            cur_price = latest_prices[sym]
            entry = pos['entry_price'] if pos['entry_price'] and pos['entry_price'] > 0 else cur_price
            coins = pos['position_coins']
            if coins == 0 or entry <= 0:
                continue
            if coins < 0:
                total += abs(coins) * (entry - cur_price)
            else:
                total += coins * (cur_price - entry)
        return total

    # ---- position calculation ----

    def calculate_positions(self, signals_data, current_positions_df):
        """
        Compute target positions following Robert Carver's framework.

        Per-symbol sizing (equal-weight allocation):
            block_value       = daily_vol × sqrt(365) × price
            alloc_capital     = theoretical_capital / N_symbols
            cash_vol_target   = target_volatility × alloc_capital
            vol_scalar        = cash_vol_target / block_value
            position_coins    = signal × vol_scalar / 10

        The /10 divides by the long-term average absolute forecast (Carver convention).

        Sizing base = total_capital + unrealized PnL (theoretical capital) so
        targets stay stable as individual trades are executed.

        Returns (positions_list, theoretical_capital, trading_capital, unrealized_pnl).
        """
        total_capital, available_cash, _, _ = self.get_portfolio_capital()
        price_map = {d['symbol']: d['price'] for d in signals_data}
        unrealized_pnl = self.calc_unrealized_pnl(current_positions_df, price_map)
        theoretical_capital = total_capital + unrealized_pnl

        valid = [d for d in signals_data if d['volatility'] > 0 and d['signal'] != 0]
        if not valid:
            return [], theoretical_capital, total_capital, unrealized_pnl

        n_symbols = len(valid)
        alloc_per_symbol = theoretical_capital / n_symbols  # equal weight

        positions = []
        for d in valid:
            row = current_positions_df[current_positions_df['symbol'] == d['symbol']]
            cur_usd = float(row.iloc[0]['position_usd']) if not row.empty else 0.0
            cur_coins = float(row.iloc[0]['position_coins']) if not row.empty else 0.0
            entry = float(row.iloc[0]['entry_price']) if not row.empty else d['price']

            # Carver position sizing
            block_value = d['volatility'] * np.sqrt(365.0) * d['price']
            if block_value <= 0:
                continue
            cash_vol_target = self.target_volatility * alloc_per_symbol
            vol_scalar = cash_vol_target / block_value
            # position = signal × vol_scalar / 10 (average absolute forecast)
            raw_coins = (d['signal'] * vol_scalar) / cfg.POSITION_DIVISOR
            # Cap at allocated capital
            max_coins = alloc_per_symbol / d['price']
            ach_coins = np.sign(raw_coins) * min(abs(raw_coins), max_coins)
            ach_usd = ach_coins * d['price']

            # Buffer logic
            buf = self.get_buffer_setting(d['symbol'])
            delta_usd = ach_usd - cur_usd

            if cur_usd == 0 and cur_coins == 0:
                # No existing position — open new
                if abs(ach_usd) < 1:
                    action, trade_usd = "HOLD", 0
                elif ach_usd > 0:
                    action = "BUY"
                    trade_usd = abs(ach_usd)
                else:
                    action = "SELL"
                    trade_usd = abs(ach_usd)
            else:
                # Existing position — check buffer
                bs = abs(cur_usd) * buf
                if abs(delta_usd) <= bs:
                    action, trade_usd = "HOLD", 0
                elif delta_usd > 0:
                    action = "BUY"
                    trade_usd = abs(delta_usd)
                else:
                    action = "SELL"
                    trade_usd = abs(delta_usd)

            positions.append({
                'symbol': d['symbol'], 'signal': d['signal'],
                'price': d['price'], 'volatility': d['volatility'],
                'alloc_weight': 1.0 / n_symbols,
                'current_position_usd': cur_usd,
                'current_position_coins': cur_coins,
                'entry_price': entry,
                'achievable_position_usd': ach_usd,
                'achievable_position_coins': ach_coins,
                'ideal_position_usd': alloc_per_symbol,
                'action': action,
                'trade_amount_usd': trade_usd,
                'trade_amount_coins': trade_usd / d['price'] if trade_usd > 0 else 0,
                'buffer_lower': cur_usd - abs(cur_usd) * buf,
                'buffer_upper': cur_usd + abs(cur_usd) * buf,
                'buffer_pct': buf,
            })
        return positions, theoretical_capital, total_capital, unrealized_pnl

    # ---- trade execution ----

    def execute_trade(self, symbol, action, amount_usd, amount_coins, price, reason="manual"):
        """Execute a BUY or SELL, updating positions, cash, and logs.

        Handles four cases correctly:
        1. Opening a new position (long or short) from zero
        2. Adding to an existing position (same direction)
        3. Partially closing a position
        4. Flipping a position (long→short or short→long)
        """
        with db_connection() as conn:
            # current position
            cur = conn.execute(
                "SELECT position_usd, position_coins, entry_price "
                "FROM current_positions WHERE symbol=?", (symbol,),
            ).fetchone()
            old_usd = float(cur[0]) if cur else 0.0
            old_coins = float(cur[1]) if cur else 0.0
            old_entry = float(cur[2] or price) if cur else price

            total_capital, available_cash, _, cum_pnl = self.get_portfolio_capital()

            # new position
            if action == "BUY":
                new_coins = old_coins + amount_coins
            elif action == "SELL":
                new_coins = old_coins - amount_coins
            else:
                return False
            new_usd = new_coins * price

            # realised PnL
            realized_pnl = 0.0
            if old_coins != 0:
                sign_changed = (old_coins > 0) != (new_coins > 0) and new_coins != 0
                if sign_changed:
                    # Position flip: close ALL old coins, then open new direction
                    closed_coins = abs(old_coins)
                elif abs(new_coins) < abs(old_coins):
                    # Partial or full close (same direction)
                    closed_coins = abs(old_coins) - abs(new_coins)
                else:
                    closed_coins = 0
                if closed_coins > 0:
                    if old_coins > 0:
                        realized_pnl = closed_coins * (price - old_entry)
                    else:
                        realized_pnl = closed_coins * (old_entry - price)

            # cash: releasing old exposure and taking on new exposure
            old_exposure = abs(old_coins * price)  # old position at current price
            new_exposure = abs(new_coins * price)
            new_cash = available_cash + old_exposure - new_exposure + realized_pnl

            # entry price
            if abs(new_coins) < 1e-12:
                new_entry = 0.0
            elif old_coins == 0:
                # Fresh open
                new_entry = price
            elif (old_coins > 0) != (new_coins > 0):
                # Flipped direction — new entry is current price
                new_entry = price
            elif abs(new_coins) > abs(old_coins):
                # Adding to position — weighted average
                added = abs(new_coins) - abs(old_coins)
                new_entry = (abs(old_coins) * old_entry + added * price) / abs(new_coins)
            else:
                # Partial close — keep old entry
                new_entry = old_entry

            # other positions' exposure (at their stored USD, will be updated by M2M)
            other_exp = conn.execute(
                "SELECT COALESCE(SUM(ABS(position_usd)),0) "
                "FROM current_positions WHERE symbol!=?", (symbol,),
            ).fetchone()[0]
            new_total_capital = new_cash + other_exp + abs(new_usd)

            # update DB
            if abs(new_coins) < 1e-12:
                conn.execute("DELETE FROM current_positions WHERE symbol=?", (symbol,))
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO current_positions "
                    "(symbol,position_usd,position_coins,entry_price,last_updated) "
                    "VALUES (?,?,?,?,?)",
                    (symbol, new_usd, new_coins, new_entry, datetime.now()),
                )

            conn.execute(
                "INSERT INTO trade_execution_log "
                "(timestamp,symbol,action,amount_usd,amount_coins,price,reason,"
                "position_before_usd,position_after_usd,capital_before,capital_after,realized_pnl) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(), symbol, action, amount_usd, amount_coins, price, reason,
                 old_usd, new_usd, total_capital, new_total_capital, realized_pnl),
            )

            # daily portfolio log — one row per day, always replaced
            today = date.today()
            cum_rpnl = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl),0) FROM trade_execution_log"
            ).fetchone()[0]
            daily_rpnl = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl),0) FROM trade_execution_log "
                "WHERE DATE(timestamp)=?", (today,),
            ).fetchone()[0]

            n_pos = conn.execute(
                "SELECT COUNT(*) FROM current_positions WHERE ABS(position_coins) > 1e-12"
            ).fetchone()[0]
            total_exposure = conn.execute(
                "SELECT COALESCE(SUM(ABS(position_usd)),0) FROM current_positions "
                "WHERE ABS(position_coins) > 1e-12"
            ).fetchone()[0]

            # Capital = initial + cumulative realized PnL (matches get_portfolio_capital)
            log_total_capital = float(self.initial_capital) + cum_rpnl
            log_cash = log_total_capital - total_exposure

            conn.execute(
                "INSERT OR REPLACE INTO daily_portfolio_log "
                "(date,total_capital,available_cash,total_exposure,"
                "unrealized_pnl,realized_pnl,cumulative_pnl,num_positions,created_at) "
                "VALUES (?,?,?,?,0,?,?,?,?)",
                (today, log_total_capital, log_cash, total_exposure,
                 daily_rpnl, cum_rpnl, n_pos, datetime.now()),
            )
            conn.commit()
        return True

    # ---- mark-to-market ----

    def mark_to_market(self, latest_prices_dict):
        """Update positions to current market prices and refresh daily log.

        This does two things:
        1. Updates ``position_usd`` in ``current_positions`` to reflect
           current market prices (so exposure numbers are accurate).
        2. Writes/replaces today's ``daily_portfolio_log`` entry with
           current unrealised PnL and correct cumulative figures.
        """
        with db_connection() as conn:
            positions = pd.read_sql_query(
                "SELECT symbol, position_usd, position_coins, entry_price "
                "FROM current_positions WHERE ABS(position_coins) > 1e-12", conn,
            )

            total_capital, available_cash, _, cum_pnl = self.get_portfolio_capital()

            if positions.empty:
                # No positions — just log cash status
                conn.execute(
                    "INSERT OR REPLACE INTO daily_portfolio_log "
                    "(date,total_capital,available_cash,total_exposure,"
                    "unrealized_pnl,realized_pnl,cumulative_pnl,num_positions,created_at) "
                    "VALUES (?,?,?,0,0,0,?,0,?)",
                    (date.today(), total_capital, available_cash, cum_pnl, datetime.now()),
                )
                conn.commit()
                return True

            # 1. Update position_usd to current market value
            exposure = 0.0
            unrealized = 0.0
            for _, pos in positions.iterrows():
                sym = pos['symbol']
                coins = pos['position_coins']
                entry = pos['entry_price'] if pos['entry_price'] and pos['entry_price'] > 0 else 0
                cur_price = latest_prices_dict.get(sym)
                if cur_price is None or cur_price <= 0:
                    exposure += abs(pos['position_usd'])
                    continue

                # Update stored USD to current market value
                new_usd = coins * cur_price
                conn.execute(
                    "UPDATE current_positions SET position_usd=?, last_updated=? "
                    "WHERE symbol=?", (new_usd, datetime.now(), sym),
                )
                exposure += abs(new_usd)

                # Unrealised PnL
                if entry > 0:
                    if coins > 0:
                        unrealized += coins * (cur_price - entry)
                    else:
                        unrealized += abs(coins) * (entry - cur_price)

            # 2. Get cumulative realized PnL from trade log (source of truth)
            cum_rpnl = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl),0) FROM trade_execution_log"
            ).fetchone()[0]
            today_rpnl = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl),0) FROM trade_execution_log "
                "WHERE DATE(timestamp)=?", (date.today(),),
            ).fetchone()[0]

            conn.execute(
                "INSERT OR REPLACE INTO daily_portfolio_log "
                "(date,total_capital,available_cash,total_exposure,unrealized_pnl,"
                "realized_pnl,daily_pnl,cumulative_pnl,num_positions,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (date.today(), total_capital, available_cash, exposure,
                 unrealized, today_rpnl, unrealized, cum_rpnl,
                 len(positions), datetime.now()),
            )
            conn.commit()
        return True

    # ---- universe management ----

    def handle_universe_change(self, new_symbols, old_symbols):
        added = set(new_symbols) - set(old_symbols)
        removed = set(old_symbols) - set(new_symbols)
        result = {'added': list(added), 'removed': list(removed),
                  'rebalance_needed': False, 'message': ''}
        if not added and not removed:
            result['message'] = 'No changes'
            return result
        if removed:
            positions = self.get_current_positions()
            in_removed = positions[positions['symbol'].isin(removed)]
            if not in_removed.empty:
                result['message'] += f"Positions in removed symbols: {list(removed)}. "
        if added:
            result['message'] += f"Added {len(added)} symbols. "
            if not self.get_current_positions().empty:
                result['rebalance_needed'] = True
        return result

    # ---- emergency / fix helpers ----

    def fix_cumulative_pnl(self):
        with db_connection() as conn:
            entries = pd.read_sql_query(
                "SELECT date, realized_pnl FROM daily_portfolio_log ORDER BY date ASC", conn,
            )
            if entries.empty:
                return True
            running = 0.0
            for _, row in entries.iterrows():
                running += row['realized_pnl']
                conn.execute(
                    "UPDATE daily_portfolio_log SET cumulative_pnl=? WHERE date=?",
                    (running, row['date']),
                )
            conn.commit()
        return True

    def emergency_revert(self):
        with db_connection() as conn:
            conn.execute("DELETE FROM daily_portfolio_log")
            trades = pd.read_sql_query(
                "SELECT DATE(timestamp) AS d, SUM(realized_pnl) AS rpnl "
                "FROM trade_execution_log GROUP BY DATE(timestamp) ORDER BY d ASC", conn,
            )
            if trades.empty:
                conn.commit()
                return True
            running = 0.0
            for _, t in trades.iterrows():
                running += t['rpnl']
                exp_q = pd.read_sql_query(
                    "SELECT COALESCE(SUM(ABS(position_usd)),0) AS e "
                    "FROM current_positions WHERE position_coins!=0", conn,
                )
                exp = exp_q['e'].iloc[0] or 0
                cash = self.initial_capital + running - exp
                conn.execute(
                    "INSERT OR REPLACE INTO daily_portfolio_log "
                    "(date,total_capital,available_cash,total_exposure,"
                    "realized_pnl,cumulative_pnl,num_positions) VALUES (?,?,?,?,?,?,1)",
                    (t['d'], cash + exp, cash, exp, t['rpnl'], running),
                )
            conn.commit()
        return True


# ========================================================================
# Dynamic multi-symbol portfolio backtest
# ========================================================================

@njit
def _dynamic_portfolio_numba(closes, vols, signals, capital, target_vol, divisor):
    n_sym, n_days = closes.shape
    capitals = np.zeros(n_days, dtype=np.float64)
    capitals[0] = capital
    allocs = np.zeros((n_sym, n_days), dtype=np.float64)
    pos = np.zeros((n_sym, n_days), dtype=np.float64)
    pnls = np.zeros((n_sym, n_days), dtype=np.float64)
    strengths = np.zeros((n_sym, n_days), dtype=np.float64)

    for day in range(1, n_days - 1):
        avail = capitals[day - 1]
        total_str = 0.0
        for s in range(n_sym):
            sig = signals[s, day - 1]
            vol = vols[s, day - 1]
            if not np.isnan(sig) and not np.isnan(vol) and vol > 0:
                ss = abs(sig)
                strengths[s, day] = ss
                total_str += ss
        if total_str > 0:
            for s in range(n_sym):
                sig = signals[s, day - 1]
                ss = strengths[s, day]
                if ss > 0 and not np.isnan(sig):
                    w = ss / total_str
                    ac = avail * w
                    allocs[s, day] = ac
                    cp = closes[s, day]
                    vol = vols[s, day - 1]
                    if not np.isnan(cp) and vol > 0 and ac > 0:
                        bv = vol * cp * np.sqrt(365.0)
                        if bv > 0:
                            vs = target_vol * ac / bv
                            raw = (sig * vs) / divisor
                            mp = ac / cp
                            pos[s, day] = np.sign(raw) * min(abs(raw), mp)
                            tp = closes[s, day + 1]
                            if not np.isnan(tp):
                                pnls[s, day] = pos[s, day] * (tp - cp)
        daily_pnl = 0.0
        for s in range(n_sym):
            daily_pnl += pnls[s, day]
        capitals[day] = max(1.0, capitals[day - 1] + daily_pnl)

    if n_days > 1:
        capitals[n_days - 1] = capitals[n_days - 2]

    return capitals, allocs, pos, pnls, strengths


def run_dynamic_portfolio(symbols=None, interval='1d', category='linear',
                          total_capital=5000, target_volatility=0.50,
                          position_divisor=cfg.POSITION_DIVISOR, start_date=None):
    """
    Load signal data from the DB, align across symbols, and run
    the numba-accelerated multi-symbol backtest.
    Returns (results_df, symbols_list) or None.
    """
    with db_connection() as conn:
        if symbols is None:
            syms_df = pd.read_sql_query(
                "SELECT DISTINCT symbol FROM strategy_signals "
                "WHERE interval=? AND category=? ORDER BY symbol",
                conn, params=(interval, category),
            )
            symbols = syms_df['symbol'].tolist()

        all_data = {}
        for sym in symbols:
            df = pd.read_sql_query(
                "SELECT timestamp, close, combined_signal "
                "FROM strategy_signals WHERE symbol=? AND interval=? AND category=? "
                "ORDER BY timestamp",
                conn, params=(sym, interval, category),
            )
            if df.empty:
                continue
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].ewm(span=36).std().fillna(
                df['returns'].ewm(span=36).std().mean()
            )
            all_data[sym] = df

    if not all_data:
        return None

    # align
    all_dates = sorted({d for df in all_data.values() for d in df['timestamp']})
    if start_date:
        sd = pd.to_datetime(start_date)
        all_dates = [d for d in all_dates if pd.to_datetime(d) >= sd]
    if not all_dates:
        return None

    sym_list = list(all_data.keys())
    ns, nd = len(sym_list), len(all_dates)
    closes = np.full((ns, nd), np.nan)
    volatilities = np.full((ns, nd), np.nan)
    sigs = np.full((ns, nd), np.nan)

    for i, sym in enumerate(sym_list):
        idx = all_data[sym].set_index('timestamp')
        for j, d in enumerate(all_dates):
            if d in idx.index:
                r = idx.loc[d]
                closes[i, j] = r['close']
                volatilities[i, j] = r['volatility']
                sigs[i, j] = r['combined_signal']

    caps, allocs, pos, pnls, strengths = _dynamic_portfolio_numba(
        closes, volatilities, sigs, total_capital, target_volatility, position_divisor,
    )

    results = pd.DataFrame({
        'timestamp': all_dates,
        'portfolio_capital': caps,
        'total_daily_pnl': np.sum(pnls, axis=0),
    })
    for i, sym in enumerate(sym_list):
        results[f'{sym}_allocation'] = allocs[i]
        results[f'{sym}_position'] = pos[i]
        results[f'{sym}_pnl'] = pnls[i]

    return results, sym_list
