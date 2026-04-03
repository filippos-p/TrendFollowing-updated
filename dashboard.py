"""
Streamlit live trading dashboard.
Run with:  streamlit run dashboard.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import time as _time
import streamlit as st
import streamlit.components.v1 as _components
import logging
import warnings

warnings.filterwarnings('ignore')
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

import config as cfg
import strategy
import portfolio as pf

# ========================================================================
# Page config
# ========================================================================
try:
    st.set_page_config(page_title="Live Trading Dashboard", page_icon="chart_with_upwards_trend",
                       layout="wide", initial_sidebar_state="expanded")
except Exception:
    pass

if not Path(str(cfg.DB_PATH)).exists():
    st.error("Database not found. Run data_fetcher.py first.")
    st.stop()

# ========================================================================
# Session state & manager
# ========================================================================

if 'pm' not in st.session_state:
    st.session_state.pm = pf.PortfolioManager()
if 'prev_symbols' not in st.session_state:
    st.session_state.prev_symbols = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'auto_refresh_interval' not in st.session_state:
    st.session_state.auto_refresh_interval = 60
if 'last_auto_refresh' not in st.session_state:
    st.session_state.last_auto_refresh = 0.0

pm: pf.PortfolioManager = st.session_state.pm

# ========================================================================
# Theme system
# ========================================================================

_THEMES = {
    'dark': {
        'bg': '#0f1118',
        'paper': '#0f1118',
        'surface': '#161b26',
        'surface2': '#1c2231',
        'grid': '#1e2536',
        'border': '#2a3345',
        'font': '#e0e0e0',
        'font_bright': '#ffffff',
        'muted': '#7a8599',
        'muted2': '#5c6a80',
        'label': '#8a95a8',
        'divider': '#2a3345',
        'midline': '#2a3345',
        'gauge_inner': '#1a2030',
        'bar_bg': '#1a2030',
        'spike': '#5c6a80',
        'annotation': '#7a8599',
        'accent': '#3b82f6',
        'positive': '#10b981',
        'negative': '#ef4444',
        'plotly_template': 'plotly_dark',
        'st_bg': '#0f1118',
        'st_secondary_bg': '#161b26',
        'st_text': '#e0e0e0',
    },
    'light': {
        'bg': '#ffffff',
        'paper': '#ffffff',
        'surface': '#f8f9fc',
        'surface2': '#f0f2f6',
        'grid': '#e5e7eb',
        'border': '#d1d5db',
        'font': '#1f2937',
        'font_bright': '#111827',
        'muted': '#6b7280',
        'muted2': '#9ca3af',
        'label': '#4b5563',
        'divider': '#e5e7eb',
        'midline': '#d1d5db',
        'gauge_inner': '#f3f4f6',
        'bar_bg': '#f3f4f6',
        'spike': '#9ca3af',
        'annotation': '#6b7280',
        'accent': '#3b82f6',
        'positive': '#059669',
        'negative': '#dc2626',
        'plotly_template': 'plotly_white',
        'st_bg': '#ffffff',
        'st_secondary_bg': '#f8f9fc',
        'st_text': '#1f2937',
    },
}

T = _THEMES[st.session_state.theme]

# Inject comprehensive professional CSS
_css = f"""
<style>
    /* ---- Base ---- */
    .stApp {{ background: {T['st_bg']}; color: {T['st_text']}; }}
    .stApp header {{ background: {T['st_bg']}; }}
    section[data-testid="stSidebar"] {{ background: {T['st_secondary_bg']}; border-right: 1px solid {T['border']}; }}

    /* ---- Typography ---- */
    h1 {{ font-size: 1.4rem !important; font-weight: 700 !important; letter-spacing: -0.02em; }}
    h2, [data-testid="stSubheader"] {{ font-size: 1.05rem !important; font-weight: 600 !important;
         color: {T['muted']} !important; text-transform: uppercase; letter-spacing: 0.06em; }}

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {{
        background: {T['surface']};
        border: 1px solid {T['border']};
        border-radius: 6px;
        padding: 12px 16px;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {T['muted']} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace !important;
        color: {T['font_bright']} !important;
    }}
    [data-testid="stMetricDelta"] {{
        font-family: 'SF Mono', 'Fira Code', monospace !important;
        font-size: 0.75rem !important;
    }}

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: {T['surface']};
        border-radius: 6px;
        padding: 3px;
        border: 1px solid {T['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 4px;
        padding: 8px 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        color: {T['muted']};
    }}
    .stTabs [aria-selected="true"] {{
        background: {T['accent']} !important;
        color: white !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
    .stTabs [data-baseweb="tab-border"] {{ display: none; }}

    /* ---- Buttons ---- */
    .stButton > button {{
        border: 1px solid {T['border']};
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        border-color: {T['accent']};
        color: {T['accent']};
    }}
    .stButton > button[kind="primary"] {{
        background: {T['accent']} !important;
        border-color: {T['accent']} !important;
    }}

    /* ---- Expanders ---- */
    [data-testid="stExpander"] {{
        background: {T['surface']};
        border: 1px solid {T['border']};
        border-radius: 6px;
    }}

    /* ---- Dataframes ---- */
    [data-testid="stDataFrame"] {{
        border: 1px solid {T['border']};
        border-radius: 6px;
    }}

    /* ---- Selectbox / inputs ---- */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {{
        border-color: {T['border']} !important;
        border-radius: 4px !important;
    }}

    /* ---- Divider ---- */
    hr {{ border-color: {T['border']} !important; opacity: 0.5; }}

    /* ---- Sidebar refinement ---- */
    section[data-testid="stSidebar"] [data-testid="stMetric"] {{
        padding: 8px 10px;
    }}
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{
        font-size: 1rem !important;
    }}

    /* ---- Number styling ---- */
    .mono {{ font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace; }}
</style>
"""
st.markdown(_css, unsafe_allow_html=True)

st.markdown(f'''<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:-8px;">
<span style="font-size:1.4rem;font-weight:700;color:{T['font_bright']};letter-spacing:-0.02em;">Trading Dashboard</span>
<span style="font-size:0.7rem;color:{T['muted2']};letter-spacing:0.05em;text-transform:uppercase;">Live</span>
</div>''', unsafe_allow_html=True)


# ========================================================================
# Cached data loaders
# ========================================================================

@st.cache_data(ttl=300)
def _universe():
    return pf.get_trading_universe()


@st.cache_data(ttl=300)
def _prices(symbols):
    return pf.get_latest_prices(list(symbols), lookback_days=1)


@st.cache_data(ttl=300)
def _signals(symbols):
    return pf.get_latest_signals(list(symbols))


@st.cache_data(ttl=600)
def _ohlc_history(symbol, start_date=None, end_date=None):
    """Fetch daily OHLC data for a symbol within an optional date range."""
    with pf.db_connection() as conn:
        q = ("SELECT timestamp, open, high, low, close, volume FROM price_data "
             "WHERE symbol=? AND interval='1d' AND category='linear'")
        params = [symbol]
        if start_date:
            q += " AND date(timestamp) >= ?"
            params.append(str(start_date))
        if end_date:
            q += " AND date(timestamp) <= ?"
            params.append(str(end_date))
        q += " ORDER BY timestamp"
        df = pd.read_sql_query(q, conn, params=params)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_data(ttl=600)
def _signal_history(symbol, start_date=None, end_date=None):
    """Fetch full signal history for a symbol."""
    with pf.db_connection() as conn:
        q = ("SELECT timestamp, combined_signal, ewmac_combined, bolmom_combined, "
             "breakout_combined, vol_forecast, close FROM strategy_signals "
             "WHERE symbol=? AND interval='1d' AND category='linear'")
        params = [symbol]
        if start_date:
            q += " AND date(timestamp) >= ?"
            params.append(str(start_date))
        if end_date:
            q += " AND date(timestamp) <= ?"
            params.append(str(end_date))
        q += " ORDER BY timestamp"
        df = pd.read_sql_query(q, conn, params=params)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_data(ttl=600)
def _latest_full_signals(symbol):
    """Get the most recent row of all individual signals for a symbol."""
    with pf.db_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM strategy_signals "
            "WHERE symbol=? AND interval='1d' AND category='linear' "
            "AND combined_signal IS NOT NULL "
            "ORDER BY timestamp DESC LIMIT 1",
            conn, params=[symbol],
        )
    return df


def _load_positions(selected):
    """Load prices, signals, current positions and compute target allocations."""
    prices = _prices(tuple(selected))
    signals = _signals(tuple(selected))
    cur_pos = pm.get_current_positions()
    if prices.empty or signals.empty:
        return None

    sig_list = []
    for sym in selected:
        sp = prices[prices['symbol'] == sym]
        ss = signals[signals['symbol'] == sym]
        if not sp.empty and not ss.empty:
            sig_list.append({
                'symbol': sym,
                'price': sp.iloc[0]['close'],
                'signal': ss.iloc[0]['combined_signal'],
                'volatility': ss.iloc[0]['volatility'],
            })
    if not sig_list:
        return None

    positions, theo, trading, unrealized = pm.calculate_positions(sig_list, cur_pos)
    return positions, theo, trading, unrealized, sig_list, cur_pos


def _do_mtm():
    """Mark-to-market: fetch LIVE prices from Bybit, save to DB, update positions."""
    if not selected_symbols:
        return
    live = pf.fetch_live_prices(selected_symbols)  # hits Bybit API + saves to DB
    if live:
        pm.mark_to_market(live)
        _prices.clear()
        _ohlc_history.clear()
        _signal_history.clear()
        _latest_full_signals.clear()


# ========================================================================
# Sidebar
# ========================================================================

# --- Theme ---
_theme_dark = st.sidebar.toggle(
    "Dark mode", value=(st.session_state.theme == 'dark'), key="theme_toggle")
if _theme_dark != (st.session_state.theme == 'dark'):
    st.session_state.theme = 'dark' if _theme_dark else 'light'
    st.rerun()

st.sidebar.markdown(f'<p style="font-size:0.72rem;color:{T["muted"]};text-transform:uppercase;'
                    f'letter-spacing:0.08em;margin:8px 0 4px 0;font-weight:600;">Universe</p>',
                    unsafe_allow_html=True)
trading_symbols = _universe()

if trading_symbols:
    mode = st.sidebar.radio("Selection mode", ["Custom", "All"],
                            horizontal=True, label_visibility="collapsed")
    if mode == "Custom":
        selected_symbols = st.sidebar.multiselect(
            "Symbols", trading_symbols,
            default=trading_symbols[:min(10, len(trading_symbols))],
        )
    else:
        selected_symbols = trading_symbols
    if not selected_symbols:
        st.sidebar.warning("No symbols selected")
        selected_symbols = []
else:
    selected_symbols = []
    st.sidebar.warning("No symbols in universe")


st.sidebar.divider()
st.sidebar.markdown(f'<p style="font-size:0.72rem;color:{T["muted"]};text-transform:uppercase;'
                    f'letter-spacing:0.08em;margin:0 0 6px 0;font-weight:600;">Refresh</p>',
                    unsafe_allow_html=True)
BYBIT_INTERVALS = ['1d', '4h', '1h', '15m', '5m']
fetch_interval = st.sidebar.selectbox(
    "Timeframe", BYBIT_INTERVALS, index=0, key="fetch_tf",
)


def _update_data():
    """Fetch latest prices. Returns (ok, msg, live_prices_dict)."""
    try:
        live_prices = {}
        if fetch_interval == '1d':
            # Fast path: single API call via tickers endpoint
            live_prices = pf.fast_update_prices(selected_symbols, category='linear')
            if not live_prices:
                return False, "No prices returned from Bybit", {}
        else:
            # Full path for intraday timeframes (4h, 1h, etc.)
            from data_fetcher import DataConfig, CryptoDataCollector
            c = CryptoDataCollector(DataConfig(
                db_path=str(cfg.DB_PATH), max_workers=7,
                request_interval=0.2, backup_enabled=False,
            ))
            c.update_price_data(selected_symbols, category='linear',
                                intervals=[fetch_interval])
        _universe.clear()
        _prices.clear()
        _ohlc_history.clear()
        _signal_history.clear()
        _latest_full_signals.clear()
        return True, f"Updated {len(selected_symbols)} symbols ({fetch_interval})", live_prices
    except Exception as e:
        return False, str(e), {}


if st.sidebar.button("Refresh All", key="btn_refresh_all", type="primary", use_container_width=True):
    if selected_symbols:
        with st.spinner("Updating data & signals..."):
            ok, msg, live = _update_data()
            if ok:
                bar = st.progress(0)
                sok, serr, smsgs = pf.compute_signals_fast(
                    selected_symbols, strategy, progress_cb=bar.progress,
                )
                st.success(f"Data + {sok} signals updated")
                if live:
                    pm.mark_to_market(live)
                else:
                    _do_mtm()
                _signals.clear()
                _signal_history.clear()
                _latest_full_signals.clear()
                st.rerun()
            else:
                st.error(msg)

with st.sidebar.expander("Advanced", expanded=False):
    _adv1, _adv2 = st.columns(2)
    with _adv1:
        if st.button("Data Only", key="btn_data", use_container_width=True):
            if selected_symbols:
                with st.spinner("Fetching..."):
                    ok, msg, live = _update_data()
                    (st.success if ok else st.error)(msg)
                    if ok:
                        if live:
                            pm.mark_to_market(live)
                        else:
                            _do_mtm()
                        st.rerun()
    with _adv2:
        if st.button("Signals Only", key="btn_sig", use_container_width=True):
            if selected_symbols:
                with st.spinner("Computing..."):
                    bar = st.progress(0)
                    ok, err, msgs = pf.compute_signals_fast(
                        selected_symbols, strategy, progress_cb=bar.progress,
                    )
                    st.success(f"Computed {ok} symbols")
                    if err:
                        st.warning(f"{err} errors")
                    _signals.clear()
                    _signal_history.clear()
                    _latest_full_signals.clear()
                    st.rerun()

_ar_c1, _ar_c2 = st.sidebar.columns([1, 2])
with _ar_c1:
    _ar_on = st.toggle("Auto", value=st.session_state.auto_refresh, key="_ar_toggle")
    if _ar_on != st.session_state.auto_refresh:
        st.session_state.auto_refresh = _ar_on
        st.session_state.last_auto_refresh = _time.time()
        st.rerun()
with _ar_c2:
    if st.session_state.auto_refresh:
        _ar_secs = st.select_slider(
            "every", options=[30, 60, 120, 300, 600],
            format_func=lambda x: f"{x//60}m" if x >= 60 else f"{x}s",
            value=st.session_state.auto_refresh_interval, key="_ar_interval",
            label_visibility="collapsed",
        )
        if _ar_secs != st.session_state.auto_refresh_interval:
            st.session_state.auto_refresh_interval = _ar_secs

if st.session_state.auto_refresh:
    _elapsed = _time.time() - st.session_state.last_auto_refresh
    _remaining = max(0, st.session_state.auto_refresh_interval - _elapsed)
    st.sidebar.caption(f"Next refresh in {int(_remaining)}s")

# Universe change check
uc = pm.handle_universe_change(trading_symbols, st.session_state.prev_symbols)
if uc.get('message') and uc['message'] != 'No changes':
    st.sidebar.info(uc['message'])
st.session_state.prev_symbols = trading_symbols.copy()

st.sidebar.divider()

# --- Portfolio ---
st.sidebar.markdown(f'<p style="font-size:0.72rem;color:{T["muted"]};text-transform:uppercase;'
                    f'letter-spacing:0.08em;margin:0 0 6px 0;font-weight:600;">Portfolio</p>',
                    unsafe_allow_html=True)
cap_input = st.sidebar.number_input("Initial Capital ($)", 10, 10_000_000, cfg.DASHBOARD_INITIAL_CAPITAL, 100)
tv = st.sidebar.slider("Target Volatility", 0.10, 1.00, cfg.DASHBOARD_TARGET_VOLATILITY, 0.05)
buf = st.sidebar.slider("Default Buffer Zone", 0.05, 0.50, cfg.DASHBOARD_DEFAULT_BUFFER, 0.01)
pm.initial_capital = cap_input
pm.target_volatility = tv
pm.default_buffer_pct = buf

# --- Admin ---
with st.sidebar.expander("Admin", expanded=False):
    if st.button("Optimize Database", key="btn_optimize"):
        with st.spinner("Optimizing..."):
            pf.optimize_indexes()
            st.success("Database optimized.")

# ========================================================================
# Tabs
# ========================================================================

# ========================================================================
# Top-of-page summary cards
# ========================================================================
if selected_symbols:
    _sum_prices = _prices(tuple(selected_symbols))
    _sum_signals = _signals(tuple(selected_symbols))
    _sum_pos = pm.get_current_positions()
    _sum_cap = pm.get_portfolio_capital()

    _portfolio_val = _sum_cap[0]
    _open_count = len(_sum_pos) if not _sum_pos.empty else 0

    # Daily P&L from portfolio log
    _daily_pnl = 0.0
    try:
        with pf.db_connection() as _sc:
            _last2 = pd.read_sql_query(
                "SELECT total_capital FROM daily_portfolio_log ORDER BY date DESC LIMIT 2", _sc)
        if len(_last2) >= 2:
            _daily_pnl = _last2.iloc[0]['total_capital'] - _last2.iloc[1]['total_capital']
    except Exception:
        pass

    # Strongest signal
    _strongest_sym, _strongest_val = "N/A", 0.0
    if not _sum_signals.empty:
        _abs_sig = _sum_signals.copy()
        _abs_sig['_abs'] = _abs_sig['combined_signal'].abs()
        _top = _abs_sig.sort_values('_abs', ascending=False).iloc[0]
        _strongest_sym = _top['symbol']
        _strongest_val = _top['combined_signal']

    _sc1, _sc2, _sc3, _sc4 = st.columns(4)
    _sc1.metric("Portfolio Value", f"${_portfolio_val:,.0f}")
    _sc2.metric("Daily P&L", f"${_daily_pnl:,.0f}",
                delta=f"{'+'if _daily_pnl>=0 else ''}{_daily_pnl:.0f}")
    _sc3.metric("Open Positions", _open_count)
    _sc4.metric("Strongest Signal", f"{_strongest_sym}",
                delta=f"{_strongest_val:+.1f}")

    st.markdown("")

tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Symbol Analysis", "Position Management", "Portfolio Overview", "Trade Log", "System"
])

# ------ TAB 0: Symbol Analysis ------
with tab0:
    if not selected_symbols:
        st.info("Select symbols in the sidebar to begin.")
    else:
        # --- Symbol picker + date range on one row ---
        _sa_top1, _sa_top2, _sa_top3, _sa_top4 = st.columns([2, 2, 2, 3])
        with _sa_top1:
            sa_sym = st.selectbox("Symbol", selected_symbols, key="sa_sym")
        with _sa_top4:
            sa_quick = st.selectbox(
                "Quick range", ["Last 40 days", "Last 6 months", "Last 1 year",
                                "Last 2 years", "All time", "Custom"],
                index=0, key="sa_quick",
            )
        today = datetime.now().date()
        if sa_quick == "Custom":
            with _sa_top2:
                sa_start = st.date_input("From", today - timedelta(days=365), key="sa_from")
            with _sa_top3:
                sa_end = st.date_input("To", today, key="sa_to")
        else:
            range_map = {
                "Last 40 days": 40, "Last 6 months": 180,
                "Last 1 year": 365, "Last 2 years": 730, "All time": None,
            }
            days = range_map.get(sa_quick)
            sa_start = (today - timedelta(days=days)) if days else None
            sa_end = today

        # --- Load data ---
        ohlc = _ohlc_history(sa_sym, sa_start, sa_end)
        sig_hist = _signal_history(sa_sym, sa_start, sa_end)
        latest_sig = _latest_full_signals(sa_sym)

        if ohlc.empty:
            st.warning(f"No price data for {sa_sym}.")
        else:
            # ============================================================
            # LEFT: unified subplot (price+forecasts+position)
            # RIGHT: gauge + rating + signals
            # ============================================================
            _dark_bg = T['bg']
            _grid = T['grid']

            # Zone color function used by gauge + signals
            def _zone_color(v):
                if v <= -10: return '#8b1a1a'
                elif v <= -5: return '#d32f2f'
                elif v <= 5: return '#00acc1'
                elif v <= 10: return '#43a047'
                else: return '#1b5e20'

            # Compute position % data
            _has_pos = (not sig_hist.empty
                        and 'vol_forecast' in sig_hist.columns
                        and 'close' in sig_hist.columns)
            if _has_pos:
                _sh = sig_hist.copy()
                _sh['block_value'] = _sh['vol_forecast'] * _sh['close'] * np.sqrt(365)
                _sh['pos_pct'] = np.clip(np.where(
                    _sh['block_value'] > 0,
                    (_sh['combined_signal'] * pm.target_volatility * _sh['close'])
                    / (_sh['block_value'] * cfg.POSITION_DIVISOR) * 100,
                    0,
                ), -100, 100)

            _n_rows = 1 + (1 if not sig_hist.empty else 0) + (1 if _has_pos else 0)
            _row_heights = [0.50]
            _titles = [sa_sym]
            if not sig_hist.empty:
                _row_heights.append(0.30)
                _titles.append('Forecasts')
            if _has_pos:
                _row_heights.append(0.20)
                _titles.append('Relative Position Size')

            col_charts, col_panel = st.columns([5, 2])

            with col_charts:
                fig = make_subplots(
                    rows=_n_rows, cols=1, shared_xaxes=True,
                    vertical_spacing=0.04,
                    row_heights=_row_heights,
                    subplot_titles=_titles,
                )

                # Row 1: Candlestick
                fig.add_trace(go.Candlestick(
                    x=ohlc['timestamp'], open=ohlc['open'], high=ohlc['high'],
                    low=ohlc['low'], close=ohlc['close'],
                    increasing_line_color=T['positive'], decreasing_line_color=T['negative'],
                    increasing_fillcolor=T['positive'], decreasing_fillcolor=T['negative'],
                    name='Price', showlegend=False,
                ), row=1, col=1)

                # Helper scatter on price subplot — Candlestick traces don't
                # trigger the unified hover crosshair, so this thin line ensures
                # the vertical hover line appears when hovering the price panel
                fig.add_trace(go.Scatter(
                    x=ohlc['timestamp'], y=ohlc['close'],
                    mode='lines', line=dict(width=0.5, color='rgba(255,255,255,0.04)'),
                    showlegend=False, name='_price_hover',
                    hovertemplate='%{x|%d %b %y}  $%{y:,.4f}<extra></extra>',
                ), row=1, col=1)

                # Row 2: Forecasts
                _cur_row = 2
                if not sig_hist.empty:
                    fig.add_trace(go.Scatter(
                        x=sig_hist['timestamp'], y=sig_hist['combined_signal'],
                        mode='lines', name='Combined',
                        line=dict(color='#42a5f5', width=2.2),
                    ), row=_cur_row, col=1)
                    for _col, _name, _clr in [
                        ('ewmac_combined', 'EWMAC', '#ab47bc'),
                        ('bolmom_combined', 'BOLMOM', '#ff7043'),
                        ('breakout_combined', 'Breakout', '#66bb6a'),
                    ]:
                        if _col in sig_hist.columns:
                            fig.add_trace(go.Scatter(
                                x=sig_hist['timestamp'], y=sig_hist[_col],
                                mode='lines', name=_name,
                                line=dict(color=_clr, width=1.3, dash='dot'),
                                opacity=0.8,
                            ), row=_cur_row, col=1)
                    fig.add_hline(y=0, line_dash="dot", line_color=T['divider'],
                                  line_width=1, row=_cur_row, col=1)
                    fig.update_yaxes(
                        title_text='Signal', range=[-22, 22],
                        gridcolor=_grid, row=_cur_row, col=1,
                    )
                    _cur_row += 1

                # Row 3: Relative Position Size
                if _has_pos:
                    _pos_colors = np.where(_sh['pos_pct'] >= 0, T['positive'], T['negative'])
                    fig.add_trace(go.Bar(
                        x=_sh['timestamp'], y=_sh['pos_pct'],
                        marker_color=_pos_colors.tolist(),
                        name='Position %', showlegend=False,
                        hovertemplate='%{y:+.1f}%<extra></extra>',
                    ), row=_cur_row, col=1)
                    # Helper scatter to ensure hover crosshair triggers on bar subplot
                    fig.add_trace(go.Scatter(
                        x=_sh['timestamp'], y=_sh['pos_pct'],
                        mode='lines', line=dict(width=0.5, color='rgba(255,255,255,0.04)'),
                        showlegend=False, name='_alloc_hover',
                        hovertemplate='%{y:+.1f}%<extra></extra>',
                    ), row=_cur_row, col=1)
                    fig.add_hline(y=0, line_color=T['divider'], line_width=1,
                                  row=_cur_row, col=1)
                    fig.update_yaxes(
                        title_text='% Alloc', ticksuffix='%',
                        gridcolor=_grid, row=_cur_row, col=1,
                    )

                # Apply to all x-axes: date format, grid (spikes off — JS crosshair handles it)
                for _r in range(1, _n_rows + 1):
                    fig.update_xaxes(
                        gridcolor=_grid, tickformat='%d %b %y',
                        showticklabels=True, showspikes=False,
                        row=_r, col=1,
                    )

                fig.update_yaxes(title_text='Price', gridcolor=_grid, row=1, col=1)
                fig.update_layout(
                    height=700, margin=dict(l=0, r=0, t=26, b=0),
                    plot_bgcolor=_dark_bg, paper_bgcolor=_dark_bg,
                    font=dict(color=T['font'], size=11),
                    legend=dict(orientation='h', y=-0.03, x=0.5, xanchor='center',
                                font=dict(size=10)),
                    xaxis_rangeslider_visible=False,
                    hovermode='x',
                    bargap=0.05,
                )
                fig.update_annotations(font_size=12, font_color=T['annotation'])

                # Render via HTML component with JS hover handler for
                # full-height crosshair across all subplots
                _fig_json = fig.to_json()
                _spike_clr = T['spike']
                _bg_clr = T['bg']
                _font_clr = T['font']
                _chart_js = (
                    "(function(){"
                    "var el=document.getElementById('cc');"
                    "var fig=" + _fig_json + ";"
                    "Plotly.newPlot(el,fig.data,fig.layout,"
                    "{responsive:true,displayModeBar:true,"
                    "modeBarButtonsToRemove:['lasso2d','select2d']}).then(function(){"
                    # Fullscreen button in modebar
                    "var mb=el.querySelector('.modebar');"
                    "if(mb){"
                    "var grp=document.createElement('div');"
                    "grp.className='modebar-group';"
                    "var btn=document.createElement('a');"
                    "btn.className='modebar-btn';"
                    "btn.setAttribute('data-title','Fullscreen');"
                    "btn.innerHTML='<svg viewBox=\"0 0 1024 1024\" width=\"1em\" height=\"1em\">"
                    "<path d=\"M128 384V128h256V64H64v320h64zm0 256v256h256v-64H128V640H64zm768-256V128H640V64h320v320h-64zm0 256v256H640v-64h256V640h64z\" fill=\"currentColor\"/>"
                    "</svg>';"
                    "btn.style.cursor='pointer';"
                    "btn.onclick=function(){"
                    "var wrapper=el.closest('.stHtml')||el.parentElement;"
                    "if(document.fullscreenElement)document.exitFullscreen();"
                    "else wrapper.requestFullscreen();"
                    "};"
                    "grp.appendChild(btn);"
                    "mb.appendChild(grp);}"
                    # Crosshair shape setup
                    "var bs=JSON.parse(JSON.stringify(el.layout.shapes||[]));"
                    "var idx=bs.length;"
                    "bs.push({type:'line',x0:0,x1:0,y0:0,y1:1,"
                    "yref:'paper',xref:'x',"
                    "line:{color:'" + _spike_clr + "',width:1,dash:'dot'},"
                    "visible:false});"
                    "Plotly.relayout(el,{shapes:bs});"
                    # Throttled hover handler using requestAnimationFrame
                    "var last=null,raf=0;"
                    "el.on('plotly_hover',function(d){"
                    "if(!d.points.length)return;"
                    "var x=d.points[0].x;if(x===last)return;last=x;"
                    "if(raf)return;raf=requestAnimationFrame(function(){"
                    "raf=0;var u={};"
                    "u['shapes['+idx+'].x0']=last;"
                    "u['shapes['+idx+'].x1']=last;"
                    "u['shapes['+idx+'].visible']=true;"
                    "Plotly.relayout(el,u);});});"
                    "el.on('plotly_unhover',function(){"
                    "last=null;if(raf){cancelAnimationFrame(raf);raf=0;}"
                    "var u={};u['shapes['+idx+'].visible']=false;"
                    "Plotly.relayout(el,u);});"
                    "});})();"
                )
                _plotly_sri = "sha384-TAqBiqItCr14J//ULLD26bSQ8Z6uPnlisSwkvWaqP8SCSiDkgR8jNknuAv8uxSOT"
                _chart_html = (
                    '<div id="cc" style="width:100%;background:' + _bg_clr + ';">'
                    '</div>'
                    '<style>'
                    '#cc:-webkit-full-screen,#cc:fullscreen{background:' + _bg_clr + ';}'
                    '.modebar-btn svg{fill:' + _font_clr + ';}'
                    '</style>'
                    '<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"'
                    ' integrity="' + _plotly_sri + '"'
                    ' crossorigin="anonymous"></script>'
                    '<script>' + _chart_js + '</script>'
                )
                _components.html(_chart_html, height=720, scrolling=False)

            with col_panel:
                # --- Forecast gauge ---
                if not latest_sig.empty:
                    combined_val = float(latest_sig.iloc[0].get('combined_signal', 0) or 0)
                else:
                    combined_val = 0.0

                st.markdown(
                    f'<p style="font-size:13px;color:{T["muted"]};margin:0 0 2px 0;font-weight:600;">Forecast</p>',
                    unsafe_allow_html=True)

                _val_color = _zone_color(combined_val)
                _rounded_val = round(combined_val, 1)

                # --- Custom polar gauge: outer thin gradient + inner thick fill ---
                # Signal → angle: -20 → 180°, 0 → 90°, +20 → 0°
                def _s2d(v):
                    return 180.0 - (v + 20.0) / 40.0 * 180.0

                # Arc polygon: traces outer edge then inner edge
                def _arc(r_in, r_out, d_start, d_end, n=40):
                    a1 = np.linspace(np.radians(d_start), np.radians(d_end), n)
                    a2 = np.linspace(np.radians(d_end), np.radians(d_start), n)
                    return (np.concatenate([np.full(n, r_out), np.full(n, r_in)]).tolist(),
                            np.degrees(np.concatenate([a1, a2])).tolist())

                # Radii — outer band width=0.08, inner band width=0.40 → 5:1
                _RO_OUT, _RO_IN = 1.0, 0.92   # outer ring
                _RI_OUT, _RI_IN = 0.89, 0.49   # inner ring (tiny gap between)

                fig_gauge = go.Figure()

                # 1. Outer gradient ring — 5 color-zone arcs
                _zone_ranges = [(-20, -10), (-10, -5), (-5, 5), (5, 10), (10, 20)]
                _zone_colors = ['#8b1a1a', '#d32f2f', '#00acc1', '#43a047', '#1b5e20']
                for (_zlo, _zhi), _zc in zip(_zone_ranges, _zone_colors):
                    _d_lo, _d_hi = _s2d(_zhi), _s2d(_zlo)
                    _r, _th = _arc(_RO_IN, _RO_OUT, _d_lo, _d_hi, n=20)
                    fig_gauge.add_trace(go.Scatterpolar(
                        r=_r, theta=_th, fill='toself',
                        fillcolor=_zc,
                        line=dict(width=0, color='rgba(0,0,0,0)'),
                        showlegend=False, hoverinfo='skip',
                    ))

                # 2. Inner ring background (dark)
                _r, _th = _arc(_RI_IN, _RI_OUT, 0, 180, n=60)
                fig_gauge.add_trace(go.Scatterpolar(
                    r=_r, theta=_th, fill='toself',
                    fillcolor=T['gauge_inner'],
                    line=dict(width=0, color='rgba(0,0,0,0)'),
                    showlegend=False, hoverinfo='skip',
                ))

                # 3. Inner ring fill — from opposite extreme to value
                if _rounded_val >= 0:
                    _df_s, _df_e = _s2d(_rounded_val), 180.0   # -20 end to value
                else:
                    _df_s, _df_e = 0.0, _s2d(_rounded_val)     # +20 end to value
                if _df_e > _df_s + 0.5:
                    _r, _th = _arc(_RI_IN, _RI_OUT, _df_s, _df_e, n=60)
                    fig_gauge.add_trace(go.Scatterpolar(
                        r=_r, theta=_th, fill='toself',
                        fillcolor=_val_color,
                        line=dict(width=0, color='rgba(0,0,0,0)'),
                        showlegend=False, hoverinfo='skip',
                    ))

                # 4. White threshold line at current value
                _th_deg = _s2d(_rounded_val)
                fig_gauge.add_trace(go.Scatterpolar(
                    r=[_RI_IN, _RO_OUT], theta=[_th_deg, _th_deg],
                    mode='lines', line=dict(color='white', width=2),
                    showlegend=False, hoverinfo='skip',
                ))

                # 5. Tick labels outside outer ring
                for _v in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
                    fig_gauge.add_trace(go.Scatterpolar(
                        r=[_RO_OUT + 0.10], theta=[_s2d(_v)],
                        mode='text', text=[str(int(_v))],
                        textfont=dict(size=9, color=T['muted']),
                        showlegend=False, hoverinfo='skip',
                    ))

                # 6. Value in center
                fig_gauge.add_annotation(
                    text=f"<b>{_rounded_val:+.1f}</b>",
                    x=0.5, y=0.08, xref='paper', yref='paper',
                    font=dict(size=28, color=_val_color, family='Arial'),
                    showarrow=False,
                )

                fig_gauge.update_layout(
                    polar=dict(
                        sector=[0, 180],
                        radialaxis=dict(visible=False, range=[0, 1.25]),
                        angularaxis=dict(visible=False, direction='counterclockwise', rotation=0),
                        bgcolor='rgba(0,0,0,0)',
                    ),
                    height=210,
                    margin=dict(l=8, r=8, t=8, b=30),
                    paper_bgcolor=T['paper'],
                    showlegend=False,
                )
                st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart")

                # --- Buy Rating ---
                if combined_val <= -10:
                    rating_text, rating_color = "Strong Sell", "#8b1a1a"
                elif combined_val <= -5:
                    rating_text, rating_color = "Sell", "#d32f2f"
                elif combined_val <= 5:
                    rating_text, rating_color = "Neutral", "#00acc1"
                elif combined_val <= 10:
                    rating_text, rating_color = "Buy", "#43a047"
                else:
                    rating_text, rating_color = "Strong Buy", "#1b5e20"

                st.markdown(
                    f"""<div style="text-align:center;padding:2px 0 10px 0;">
                    <span style="font-size:11px;color:{T['muted2']};">Buy Rating</span><br>
                    <span style="font-size:20px;font-weight:700;color:white;
                    background:{rating_color};padding:5px 16px;border-radius:5px;
                    display:inline-block;margin-top:3px;">{rating_text}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # --- Symbol info: price, 24h change, volume ---
                # Fetch last 2 days to compute real 24h change
                _sa_prices_2d = pf.get_latest_prices([sa_sym], lookback_days=2)
                _sa_rows = _sa_prices_2d[_sa_prices_2d['symbol'] == sa_sym].sort_values('timestamp', ascending=False) if not _sa_prices_2d.empty else pd.DataFrame()
                if not _sa_rows.empty:
                    _sa_close = float(_sa_rows.iloc[0]['close'])
                    _sa_prev = float(_sa_rows.iloc[1]['close']) if len(_sa_rows) >= 2 else _sa_close
                    _sa_vol = float(_sa_rows.iloc[0].get('volume', 0))
                    _sa_chg = ((_sa_close / _sa_prev - 1) * 100) if _sa_prev else 0
                    _chg_color = T['positive'] if _sa_chg >= 0 else T['negative']
                    st.markdown(f'''
                    <div style="margin:4px 0 8px 0;padding:10px 12px;border-radius:4px;
                                background:{T['surface']};border:1px solid {T['border']};">
                      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                        <span style="font-size:9px;color:{T['muted2']};text-transform:uppercase;letter-spacing:0.08em;">Price</span>
                        <span style="font-size:14px;color:{T['font_bright']};font-weight:700;font-family:monospace;">${_sa_close:,.6g}</span>
                      </div>
                      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                        <span style="font-size:9px;color:{T['muted2']};text-transform:uppercase;letter-spacing:0.08em;">24h Change</span>
                        <span style="font-size:14px;color:{_chg_color};font-weight:700;font-family:monospace;">{_sa_chg:+.2f}%</span>
                      </div>
                      <div style="display:flex;justify-content:space-between;">
                        <span style="font-size:9px;color:{T['muted2']};text-transform:uppercase;letter-spacing:0.08em;">Volume</span>
                        <span style="font-size:14px;color:{T['font_bright']};font-weight:700;font-family:monospace;">{_sa_vol:,.0f}</span>
                      </div>
                    </div>''', unsafe_allow_html=True)

                # --- Signals (center-aligned bars) ---
                st.markdown(
                    f'<p style="font-size:13px;color:{T["muted"]};margin:8px 0 4px 0;font-weight:600;">Signals</p>',
                    unsafe_allow_html=True,
                )
                if not latest_sig.empty:
                    _row = latest_sig.iloc[0]
                    _sig_items = [
                        ('EWMAC', 'ewmac_combined'),
                        ('BOLMOM', 'bolmom_combined'),
                        ('Breakout', 'breakout_combined'),
                        ('Combined', 'combined_signal'),
                    ]

                    def _smooth_color(v):
                        """Smooth gradient: dark red(-20) -> red(-10) -> cyan(0) -> green(+10) -> dark green(+20)."""
                        t = max(0.0, min(1.0, (v + 20) / 40))  # 0..1
                        # 5-stop gradient with smooth interpolation
                        stops = [
                            (0.00, (139, 26, 26)),    # dark red  at -20
                            (0.25, (211, 47, 47)),    # red       at -10
                            (0.50, (0, 172, 193)),     # cyan      at  0
                            (0.75, (67, 160, 71)),     # green     at +10
                            (1.00, (27, 94, 32)),      # dark green at +20
                        ]
                        for i in range(len(stops) - 1):
                            t0, c0 = stops[i]
                            t1, c1 = stops[i + 1]
                            if t <= t1:
                                f = (t - t0) / (t1 - t0) if t1 > t0 else 0
                                r = int(c0[0] + (c1[0] - c0[0]) * f)
                                g = int(c0[1] + (c1[1] - c0[1]) * f)
                                b = int(c0[2] + (c1[2] - c0[2]) * f)
                                return f'rgb({r},{g},{b})'
                        return f'rgb({stops[-1][1][0]},{stops[-1][1][1]},{stops[-1][1][2]})'

                    _sig_html = ''
                    for _label, _col_name in _sig_items:
                        _val = float(_row.get(_col_name, 0) or 0)
                        _c = _smooth_color(_val)
                        # Center-aligned: 0 = 50%, bar extends left or right from center
                        _bar_pct = abs(_val) / 20 * 50  # max 50% of total width
                        if _val >= 0:
                            _bar_left = 50
                        else:
                            _bar_left = 50 - _bar_pct
                        _sig_html += f'''
                        <div style="display:flex;align-items:center;margin:3px 0;">
                          <span style="width:62px;font-size:11px;color:{T['label']};flex-shrink:0;">{_label}</span>
                          <div style="flex:1;height:14px;background:{T['bar_bg']};border-radius:2px;overflow:hidden;margin:0 6px;position:relative;">
                            <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:{T['midline']};"></div>
                            <div style="position:absolute;left:{_bar_left:.1f}%;top:0;width:{_bar_pct:.1f}%;height:100%;background:{_c};border-radius:2px;"></div>
                          </div>
                          <span style="width:38px;font-size:11px;color:{_c};text-align:right;font-weight:600;">{_val:+.1f}</span>
                        </div>'''
                    st.markdown(_sig_html, unsafe_allow_html=True)
                else:
                    st.caption("No signal data.")

# ------ TAB 1: Positions ------
with tab1:
    if selected_symbols:
        with st.spinner("Loading..."):
            result = _load_positions(selected_symbols)
        if result is None:
            st.error("No data for selected symbols")
        else:
            positions, theo_cap, trading_cap, unrealized, sig_list, cur_pos = result
            if positions:
                pos_df = pd.DataFrame(positions)

                # Summary metrics bar
                exposure = cur_pos['position_usd'].abs().sum() if not cur_pos.empty else 0
                cash = trading_cap - exposure
                util = exposure / trading_cap * 100 if trading_cap else 0
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Trading Capital", f"${trading_cap:,.0f}")
                c2.metric("Exposure", f"${exposure:,.0f}")
                c3.metric("Cash", f"${cash:,.0f}")
                c4.metric("Unrealized PnL", f"${unrealized:,.0f}")
                c5.metric("Utilization", f"{util:.1f}%")

                # ---- Position cards ----
                needs_action = pos_df[pos_df['action'] != 'HOLD']
                _n_action = len(needs_action)

                # Execute all bar
                if _n_action:
                    _ea1, _ea2, _ea3 = st.columns([2, 2, 3])
                    with _ea1:
                        ea_reason = st.selectbox("Reason for all", ["rebalance", "signal_change", "buffer_breach", "manual"],
                                                 key="ea_reason")
                    with _ea2:
                        if st.button(f"Execute All ({_n_action} trades)", key="exec_all", type="primary"):
                            ok_count, fail_count = 0, 0
                            for _, p in needs_action.iterrows():
                                ok = pm.execute_trade(
                                    p['symbol'], p['action'],
                                    abs(p['trade_amount_usd']), abs(p['trade_amount_coins']),
                                    p['price'], ea_reason)
                                if ok:
                                    ok_count += 1
                                else:
                                    fail_count += 1
                            if ok_count:
                                st.success(f"Executed {ok_count} trades")
                            if fail_count:
                                st.error(f"{fail_count} trades failed")
                            if ok_count:
                                _prices.clear()
                                _do_mtm()
                                st.rerun()

                # Render position cards in a 2-column grid
                _sorted_pos = pos_df.sort_values('trade_amount_usd', key=abs, ascending=False)
                _card_cols = st.columns(2)
                for _ci, (_, p) in enumerate(_sorted_pos.iterrows()):
                    _action = p['action']
                    _cur_usd = p['current_position_usd']
                    _tgt_usd = p['achievable_position_usd']
                    _tgt_coins = p['achievable_position_coins']
                    _trade_usd = p['trade_amount_usd']
                    _sig = p['signal']
                    _vol = p['volatility'] * 100
                    _price = p['price']
                    _buf_lo = p['buffer_lower']
                    _buf_hi = p['buffer_upper']

                    # Side bar color
                    if _action == 'BUY':
                        _act_color = T['positive']
                        _act_bg = T['positive'] + '14'
                    elif _action == 'SELL':
                        _act_color = T['negative']
                        _act_bg = T['negative'] + '14'
                    else:
                        _act_color = T['border']
                        _act_bg = T['surface']

                    # Progress bar: current vs target allocation
                    _max_alloc = max(abs(_cur_usd), abs(_tgt_usd), 1)
                    _cur_pct = abs(_cur_usd) / _max_alloc * 100
                    _tgt_pct = abs(_tgt_usd) / _max_alloc * 100

                    # Side label
                    if _tgt_usd > 0:
                        _side_label = 'LONG'
                        _side_color = T['positive']
                    elif _tgt_usd < 0:
                        _side_label = 'SHORT'
                        _side_color = T['negative']
                    else:
                        _side_label = 'FLAT'
                        _side_color = T['muted2']

                    _buf_str = (f"${_buf_lo:.0f} – ${_buf_hi:.0f}"
                                if _cur_usd != 0 else "N/A (new)")

                    _card_html = f'''
                    <div style="border-left:3px solid {_act_color};background:{T['surface']};
                                border:1px solid {T['border']};border-left:3px solid {_act_color};
                                border-radius:4px;padding:12px 14px;margin-bottom:10px;">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                        <div style="display:flex;align-items:center;gap:8px;">
                          <span style="font-size:14px;font-weight:700;color:{T['font_bright']};
                                font-family:'SF Mono','Fira Code',monospace;">{p['symbol']}</span>
                          <span style="font-size:9px;font-weight:700;color:{_side_color};
                                letter-spacing:0.08em;">{_side_label}</span>
                        </div>
                        <span style="font-size:10px;font-weight:700;color:{_act_color};
                              border:1px solid {_act_color};
                              padding:2px 8px;border-radius:3px;letter-spacing:0.08em;">{_action}</span>
                      </div>
                      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px 12px;
                                  font-size:11px;margin-bottom:8px;">
                        <div><span style="color:{T['muted2']};font-size:9px;text-transform:uppercase;letter-spacing:0.06em;">Signal</span><br>
                             <span style="color:{T['font_bright']};font-family:monospace;font-weight:600;">{_sig:+.3f}</span></div>
                        <div><span style="color:{T['muted2']};font-size:9px;text-transform:uppercase;letter-spacing:0.06em;">Volatility</span><br>
                             <span style="color:{T['font_bright']};font-family:monospace;font-weight:600;">{_vol:.2f}%</span></div>
                        <div><span style="color:{T['muted2']};font-size:9px;text-transform:uppercase;letter-spacing:0.06em;">Price</span><br>
                             <span style="color:{T['font_bright']};font-family:monospace;font-weight:600;">${_price:,.4f}</span></div>
                      </div>
                      <div style="display:flex;justify-content:space-between;font-size:11px;color:{T['muted']};margin-bottom:4px;">
                        <span style="font-family:monospace;">${_cur_usd:,.0f}</span>
                        <span style="color:{T['muted2']};">&rarr;</span>
                        <span style="font-family:monospace;color:{T['font_bright']};">${_tgt_usd:,.0f}</span>
                        <span style="font-family:monospace;font-size:10px;color:{T['muted2']};">({_tgt_coins:+.4f} coins)</span>
                      </div>
                      <div style="position:relative;height:4px;background:{T['surface2']};border-radius:2px;overflow:hidden;margin:4px 0 6px 0;">
                        <div style="position:absolute;height:100%;width:{_cur_pct:.1f}%;background:{T['muted2']};
                                    border-radius:2px;opacity:0.4;"></div>
                        <div style="position:absolute;height:100%;width:{_tgt_pct:.1f}%;background:{_act_color};
                                    border-radius:2px;"></div>
                      </div>
                      <div style="display:flex;justify-content:space-between;font-size:10px;">
                        <span style="color:{T['muted2']};">Buffer: {_buf_str}</span>
                        <span style="font-weight:700;color:{_act_color};font-family:monospace;">
                          ${abs(_trade_usd):,.0f}</span>
                      </div>
                    </div>'''

                    with _card_cols[_ci % 2]:
                        st.markdown(_card_html, unsafe_allow_html=True)
                        # Individual execute button for non-HOLD actions
                        if _action != 'HOLD':
                            _xc1, _xc2 = st.columns([3, 1])
                            with _xc1:
                                _xreason = st.selectbox("Reason", ["buffer_breach", "signal_change", "rebalance", "manual"],
                                                        key=f"r_{p['symbol']}", label_visibility="collapsed")
                            with _xc2:
                                if st.button("Execute", key=f"x_{p['symbol']}", use_container_width=True):
                                    ok = pm.execute_trade(p['symbol'], p['action'],
                                                          abs(p['trade_amount_usd']), abs(p['trade_amount_coins']),
                                                          p['price'], _xreason)
                                    if ok:
                                        st.success(f"{p['action']} executed for {p['symbol']}")
                                        _prices.clear()
                                        st.rerun()
                                    else:
                                        st.error(f"Failed for {p['symbol']}")

                # Full data table in expander
                with st.expander("Full Position Data"):
                    disp = pos_df.copy()
                    disp['Signal'] = disp['signal'].round(4)
                    disp['Vol%'] = (disp['volatility'] * 100).round(3)
                    disp['Price'] = disp['price'].round(4)
                    disp['Current$'] = disp['current_position_usd'].round(2)
                    disp['Target$'] = disp['achievable_position_usd'].round(2)
                    disp['TargetCoins'] = disp['achievable_position_coins'].round(4)
                    disp['Buffer'] = disp.apply(
                        lambda r: "N/A (new)" if r['current_position_usd'] == 0
                        else f"${r['buffer_lower']:.0f} - ${r['buffer_upper']:.0f}", axis=1)
                    disp['Action'] = disp['action']
                    disp['Trade$'] = disp['trade_amount_usd'].round(2)

                    def _color_action(v):
                        if v == 'BUY': return 'background-color: lightgreen'
                        if v == 'SELL': return 'background-color: lightcoral'
                        return 'background-color: lightgray'

                    st.dataframe(
                        disp[['symbol', 'Signal', 'Vol%', 'Price', 'Current$',
                              'Target$', 'TargetCoins', 'Buffer', 'Action', 'Trade$'
                              ]].style.map(_color_action, subset=['Action']),
                        use_container_width=True, height=500,
                    )

                # Manual entry
                with st.expander("Manual Trade Entry"):
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    msym = mc1.selectbox("Symbol", selected_symbols, key="m_sym")
                    mact = mc2.selectbox("Action", ["BUY", "SELL"], key="m_act")
                    musd = mc3.number_input("Amount ($)", 0.0, step=0.001, format="%.5f", key="m_usd")
                    _manual_prices = _prices(tuple(selected_symbols))
                    _mp_row = _manual_prices[_manual_prices['symbol'] == msym] if not _manual_prices.empty else pd.DataFrame()
                    _default_price = float(_mp_row.iloc[0]['close']) if not _mp_row.empty else 0.00001
                    mprc = mc4.number_input("Price", 0.00001, step=0.00001, format="%.6f", key="m_prc",
                                            value=_default_price)
                    if st.button("Log Manual Trade"):
                        if musd > 0 and mprc > 0:
                            ok = pm.execute_trade(msym, mact, musd, musd / mprc, mprc, "manual")
                            if ok:
                                st.success(f"Manual {mact} logged for {msym}")
                                st.rerun()
                        else:
                            st.error("Enter valid amount and price")
            else:
                st.info("No valid positions generated")

# ------ TAB 2: Portfolio Overview ------
with tab2:
    # --- Row 1: Exposure metrics ---
    cur_pos = pm.get_current_positions()
    prices_df = _prices(tuple(selected_symbols)) if selected_symbols else pd.DataFrame()
    price_map = {}
    if not prices_df.empty:
        for _, r in prices_df.iterrows():
            price_map[r['symbol']] = r['close']

    rows = []
    total_unr = 0
    total_long = 0
    total_short = 0
    if not cur_pos.empty:
        for _, p in cur_pos.iterrows():
            sym = p['symbol']
            cprice = price_map.get(sym, p['entry_price'] or 0)
            entry = p['entry_price'] if p['entry_price'] and p['entry_price'] > 0 else cprice
            coins = p['position_coins']
            market_val = coins * cprice
            notional = abs(market_val)
            if coins > 0:
                total_long += notional
                upnl = coins * (cprice - entry)
                pct = (cprice / entry - 1) * 100 if entry else 0
                side = "LONG"
            elif coins < 0:
                total_short += notional
                upnl = abs(coins) * (entry - cprice)
                pct = -(cprice / entry - 1) * 100 if entry else 0
                side = "SHORT"
            else:
                continue
            total_unr += upnl
            rows.append({
                'symbol': sym, 'side': side,
                'coins': coins, 'entry': entry,
                'current': cprice, 'notional': notional,
                'market_val': market_val,
                'unrealized': upnl, 'pnl%': pct,
            })

    if rows:
        edf = pd.DataFrame(rows)
        net_exposure = total_long - total_short
        gross_exposure = total_long + total_short

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Net Exposure", f"${net_exposure:,.2f}",
                   delta="Long" if net_exposure > 0 else ("Short" if net_exposure < 0 else "Flat"))
        mc2.metric("Gross Exposure", f"${gross_exposure:,.2f}")
        mc3.metric("Long", f"${total_long:,.2f}")
        mc4.metric("Short", f"${total_short:,.2f}")
        mc5.metric("Unrealized PnL", f"${total_unr:,.2f}")

    # --- Row 2: History chart + Donut side by side ---
    _ov_left, _ov_right = st.columns([3, 2])
    with _ov_left:
        try:
            with pf.db_connection() as conn:
                hist = pd.read_sql_query(
                    "SELECT date, total_capital, available_cash, total_exposure, "
                    "unrealized_pnl, realized_pnl, cumulative_pnl, num_positions "
                    "FROM daily_portfolio_log ORDER BY date DESC LIMIT 30", conn,
                )
            if not hist.empty:
                hist = hist.sort_values('date').reset_index(drop=True)
                st.markdown(f'<p style="font-size:14px;font-weight:600;color:{T["font"]};margin:0 0 4px 0;">Portfolio History (30d)</p>',
                            unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist['date'], y=hist['total_capital'],
                                         mode='lines+markers', name='Trading Capital',
                                         line=dict(color='#42a5f5', width=2), marker=dict(size=4)))
                if 'unrealized_pnl' in hist.columns:
                    fig.add_trace(go.Scatter(
                        x=hist['date'], y=hist['total_capital'] + hist['unrealized_pnl'],
                        mode='lines+markers', name='Total Value',
                        line=dict(color='#66bb6a', width=2), marker=dict(size=4)))
                fig.add_trace(go.Scatter(x=hist['date'], y=hist['cumulative_pnl'],
                                         mode='lines+markers', name='Cumulative PnL',
                                         line=dict(color='#ff7043', width=2), marker=dict(size=4)))
                fig.update_layout(
                    height=340, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(tickformat="%d %b", gridcolor=T['grid']),
                    yaxis=dict(gridcolor=T['grid']),
                    plot_bgcolor=T['bg'], paper_bgcolor=T['paper'],
                    font=dict(color=T['font'], size=11),
                    legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center', font=dict(size=10)),
                )
                st.plotly_chart(fig, use_container_width=True, key="hist_chart")
            else:
                st.info("No portfolio history yet. Execute trades or run M2M.")
        except Exception as e:
            st.error(f"Error loading history: {e}")

    with _ov_right:
        if rows:
            st.markdown(f'<p style="font-size:14px;font-weight:600;color:{T["font"]};margin:0 0 4px 0;">Capital Allocation</p>',
                        unsafe_allow_html=True)
            _donut_df = edf[['symbol', 'notional']].copy()
            _alloc_cap = pm.get_portfolio_capital()
            _cash_val = max(0, _alloc_cap[1])
            if _cash_val > 0:
                _donut_df = pd.concat([_donut_df, pd.DataFrame([{'symbol': 'Cash', 'notional': _cash_val}])],
                                      ignore_index=True)
            _donut_colors = ['#42a5f5', '#ab47bc', '#ff7043', '#66bb6a', '#26a69a',
                             '#ef5350', '#ffa726', '#8d6e63', '#78909c', '#ec407a',
                             '#29b6f6', '#9ccc65', '#d4e157', '#ffca28', '#bdbdbd']
            fig_donut = go.Figure(go.Pie(
                labels=_donut_df['symbol'], values=_donut_df['notional'],
                hole=0.55, marker=dict(colors=_donut_colors[:len(_donut_df)]),
                textinfo='label+percent', textfont=dict(size=11),
                hovertemplate='%{label}: $%{value:,.0f}<extra></extra>',
            ))
            fig_donut.update_layout(
                height=340, margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor=T['bg'], paper_bgcolor=T['paper'],
                font=dict(color=T['font']), showlegend=False,
            )
            st.plotly_chart(fig_donut, use_container_width=True, key="alloc_donut")
        else:
            st.info("No open positions for allocation chart.")

    # --- Row 3: Signal heatmap full width ---
    _hm_signals = _signals(tuple(selected_symbols)) if selected_symbols else pd.DataFrame()
    if not _hm_signals.empty:
        st.markdown(f'<p style="font-size:14px;font-weight:600;color:{T["font"]};margin:8px 0 4px 0;">Signal Heatmap</p>',
                    unsafe_allow_html=True)
        _hm_cols = ['combined_signal']
        _extra = ['ewmac_combined', 'bolmom_combined', 'breakout_combined']
        _hm_cols += [c for c in _extra if c in _hm_signals.columns]
        _hm_data = _hm_signals[['symbol'] + _hm_cols].set_index('symbol')
        _hm_data.columns = [c.replace('_combined', '').replace('_signal', '').upper()
                            for c in _hm_data.columns]
        _hm_vals = _hm_data.values.astype(float)

        fig_hm = go.Figure(go.Heatmap(
            z=_hm_vals, x=_hm_data.columns.tolist(),
            y=_hm_data.index.tolist(),
            colorscale=[[0, '#8b1a1a'], [0.25, '#d32f2f'], [0.5, '#00acc1'],
                        [0.75, '#43a047'], [1, '#1b5e20']],
            zmid=0, zmin=-20, zmax=20,
            text=np.round(_hm_vals, 1), texttemplate='%{text}',
            textfont=dict(size=11),
            hovertemplate='%{y} | %{x}: %{z:.1f}<extra></extra>',
            colorbar=dict(title='Signal', tickvals=[-20, -10, 0, 10, 20]),
        ))
        fig_hm.update_layout(
            height=max(200, len(_hm_data) * 28 + 40),
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor=T['bg'], paper_bgcolor=T['paper'],
            font=dict(color=T['font'], size=11),
            xaxis=dict(side='top'),
        )
        st.plotly_chart(fig_hm, use_container_width=True, key="sig_heatmap")

    # --- Row 4: Exposure table + history data ---
    if rows:
        with st.expander("Exposure Breakdown"):
            def _cpnl(v):
                if v > 0: return 'color: green'
                if v < 0: return 'color: red'
                return ''
            def _cside(v):
                if v == 'LONG': return 'color: green'
                if v == 'SHORT': return 'color: red'
                return ''
            st.dataframe(
                edf[['symbol', 'side', 'coins', 'entry', 'current', 'notional',
                     'unrealized', 'pnl%']].round(4).style
                    .map(_cpnl, subset=['unrealized', 'pnl%'])
                    .map(_cside, subset=['side']),
                use_container_width=True,
            )

    try:
        if not hist.empty:
            with st.expander("Portfolio History Data"):
                hist_disp = hist.copy()
                hist_disp['date'] = pd.to_datetime(hist_disp['date']).dt.strftime('%Y-%m-%d')
                st.dataframe(hist_disp.round(2), use_container_width=True)
    except NameError:
        pass

# ------ TAB 3: Trade Log ------
with tab3:
    try:
        with pf.db_connection() as conn:
            c1, c2, c3 = st.columns(3)
            fsym = c1.selectbox("Symbol", ["All"] + (selected_symbols or []), key="tl_sym")
            fact = c2.selectbox("Action", ["All", "BUY", "SELL"], key="tl_act")
            flim = c3.selectbox("Show Last", [25, 50, 100, "All"], key="tl_lim")

            q = ("SELECT timestamp,symbol,action,amount_usd,amount_coins,price,"
                 "reason,position_before_usd,position_after_usd,realized_pnl "
                 "FROM trade_execution_log WHERE 1=1")
            params = []
            if fsym != "All":
                q += " AND symbol=?"
                params.append(fsym)
            if fact != "All":
                q += " AND action=?"
                params.append(fact)
            q += " ORDER BY timestamp DESC"
            if flim != "All":
                q += f" LIMIT {flim}"

            trades = pd.read_sql_query(q, conn, params=params)

        if not trades.empty:
            _n_buys = len(trades[trades['action'] == 'BUY'])
            _n_sells = len(trades[trades['action'] == 'SELL'])
            _total_rpnl = trades['realized_pnl'].sum()
            _wins = len(trades[trades['realized_pnl'] > 0])
            _losses = len(trades[trades['realized_pnl'] < 0])
            _wr = (_wins / (_wins + _losses) * 100) if (_wins + _losses) > 0 else 0

            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Total Trades", len(trades))
            sc2.metric("Buys / Sells", f"{_n_buys} / {_n_sells}")
            sc3.metric("Win Rate", f"{_wr:.0f}%", delta=f"{_wins}W / {_losses}L")
            sc4.metric("Realized PnL", f"${_total_rpnl:,.2f}")
            sc5.metric("Avg P&L/Trade",
                        f"${_total_rpnl / len(trades):,.2f}" if len(trades) else "$0")

            # --- Realized P&L chart first (hero element) ---
            # --- Realized P&L per symbol chart ---
            _rpnl_trades = trades[trades['realized_pnl'].notna() & (trades['realized_pnl'] != 0)].copy()
            if not _rpnl_trades.empty:
                st.subheader("Realized P&L per Symbol")
                # Sort chronologically for cumulative calculation
                _rpnl_trades = _rpnl_trades.sort_values('timestamp')
                _rpnl_trades['timestamp'] = pd.to_datetime(_rpnl_trades['timestamp'])

                fig_rpnl = go.Figure()
                _rpnl_syms = _rpnl_trades['symbol'].unique()
                _rpnl_colors = ['#42a5f5', '#ab47bc', '#ff7043', '#66bb6a', '#26a69a',
                                '#ef5350', '#ffa726', '#8d6e63', '#78909c', '#ec407a',
                                '#29b6f6', '#9ccc65', '#d4e157', '#ffca28']
                for _i, _sym in enumerate(_rpnl_syms):
                    _sym_trades = _rpnl_trades[_rpnl_trades['symbol'] == _sym].copy()
                    _sym_trades['cum_pnl'] = _sym_trades['realized_pnl'].cumsum()
                    fig_rpnl.add_trace(go.Scatter(
                        x=_sym_trades['timestamp'], y=_sym_trades['cum_pnl'],
                        mode='lines+markers', name=_sym,
                        line=dict(color=_rpnl_colors[_i % len(_rpnl_colors)], width=2),
                        marker=dict(size=5),
                        hovertemplate='%{x}<br>%{y:$,.2f}<extra>%{fullData.name}</extra>',
                    ))

                # Also add total cumulative
                _rpnl_total = _rpnl_trades.copy()
                _rpnl_total['cum_pnl'] = _rpnl_total['realized_pnl'].cumsum()
                fig_rpnl.add_trace(go.Scatter(
                    x=_rpnl_total['timestamp'], y=_rpnl_total['cum_pnl'],
                    mode='lines', name='Total',
                    line=dict(color='white', width=2.5, dash='dot'),
                    hovertemplate='%{x}<br>%{y:$,.2f}<extra>Total</extra>',
                ))

                fig_rpnl.add_hline(y=0, line_dash="dot", line_color=T['divider'], line_width=1)
                fig_rpnl.update_layout(
                    height=350, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor=T['bg'], paper_bgcolor=T['paper'],
                    font=dict(color=T['font']),
                    xaxis=dict(gridcolor=T['grid'], tickformat='%d %b %y'),
                    yaxis=dict(gridcolor=T['grid'], title='Cumulative P&L ($)'),
                    legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
                    hovermode='x unified',
                )
                st.plotly_chart(fig_rpnl, use_container_width=True, key="rpnl_chart")

                # Per-symbol summary table
                _rpnl_summary = _rpnl_trades.groupby('symbol')['realized_pnl'].agg(
                    ['sum', 'count']).rename(columns={'sum': 'Total P&L', 'count': 'Trades'})
                _rpnl_summary = _rpnl_summary.sort_values('Total P&L', ascending=False)
                _rpnl_summary['Total P&L'] = _rpnl_summary['Total P&L'].round(2)
                st.dataframe(_rpnl_summary, use_container_width=True)

            # --- Trade timeline ---
            st.markdown(f'<p style="font-size:14px;font-weight:600;color:{T["font"]};margin:12px 0 6px 0;">Recent Trades</p>',
                        unsafe_allow_html=True)
            _tl_html = f'''<div style="border:1px solid {T['border']};border-radius:4px;overflow:hidden;">
            <div style="display:grid;grid-template-columns:140px 90px 42px 85px 85px 70px 1fr;
                        padding:6px 12px;font-size:9px;text-transform:uppercase;letter-spacing:0.08em;
                        color:{T['muted2']};border-bottom:1px solid {T['border']};background:{T['surface']};">
              <span>Time</span><span>Symbol</span><span>Side</span><span>Amount</span>
              <span>Price</span><span>Reason</span><span style="text-align:right;">P&L</span>
            </div>'''
            for _ti, (_, _tr) in enumerate(trades.head(25).iterrows()):
                _tr_act = _tr['action']
                _tr_clr = T['positive'] if _tr_act == 'BUY' else T['negative']
                _tr_rpnl = _tr.get('realized_pnl', 0) or 0
                _pnl_clr = T['positive'] if _tr_rpnl >= 0 else T['negative']
                _pnl_str = f'<span style="color:{_pnl_clr};font-weight:600;">${_tr_rpnl:+,.2f}</span>' if _tr_rpnl != 0 else '<span style="color:{T["muted2"]};">—</span>'
                _tr_ts = str(_tr['timestamp'])[:19]
                _row_bg = T['surface'] if _ti % 2 == 0 else 'transparent'
                _tl_html += f'''
                <div style="display:grid;grid-template-columns:140px 90px 42px 85px 85px 70px 1fr;
                            align-items:center;padding:5px 12px;font-size:12px;
                            background:{_row_bg};border-bottom:1px solid {T['border']}22;">
                  <span style="font-size:10px;color:{T['muted']};font-family:monospace;">{_tr_ts}</span>
                  <span style="font-weight:700;color:{T['font_bright']};font-family:monospace;font-size:11px;">{_tr['symbol']}</span>
                  <span style="font-size:10px;font-weight:700;color:{_tr_clr};">{_tr_act}</span>
                  <span style="font-family:monospace;color:{T['font']};">${_tr['amount_usd']:,.2f}</span>
                  <span style="font-family:monospace;color:{T['muted']};">${_tr['price']:,.4f}</span>
                  <span style="font-size:10px;color:{T['muted2']};">{_tr.get('reason','')}</span>
                  <span style="text-align:right;font-family:monospace;">{_pnl_str}</span>
                </div>'''
            _tl_html += '</div>'
            st.markdown(_tl_html, unsafe_allow_html=True)

            if len(trades) > 25:
                st.caption(f"Showing 25 of {len(trades)} trades")

            # Full table + export in expander
            with st.expander("Full Trade Data"):
                st.dataframe(trades.round(4), use_container_width=True, height=500)
                csv = trades.to_csv(index=False)
                st.download_button("Export CSV", csv,
                                   f"trades_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
        else:
            st.info("No trades match filters")
    except Exception as e:
        st.error(f"Error: {e}")

# ------ TAB 4: System Status ------
with tab4:
    st.markdown(f'<p style="font-size:14px;font-weight:600;color:{T["font"]};margin:0 0 8px 0;">Diagnostics</p>',
                unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        cap = pm.get_portfolio_capital()
        _diag_html = f'''
        <div style="background:{T['surface']};border:1px solid {T['border']};border-radius:6px;padding:14px;">
          <p style="font-size:11px;font-weight:600;color:{T['muted']};text-transform:uppercase;letter-spacing:0.06em;margin:0 0 10px 0;">Configuration</p>
          <div style="font-size:12px;font-family:monospace;line-height:1.8;color:{T['font']};">
            Capital: ${pm.initial_capital:,}<br>
            Target Vol: {pm.target_volatility:.1%}<br>
            Buffer: {pm.default_buffer_pct:.1%}<br>
            <span style="border-top:1px solid {T['border']};display:block;margin:6px 0;"></span>
            Current Capital: <span style="color:{T['font_bright']};font-weight:600;">${cap[0]:,.0f}</span><br>
            Cash: <span style="color:{T['font_bright']};font-weight:600;">${cap[1]:,.0f}</span><br>
            Realized PnL: <span style="color:{T['font_bright']};font-weight:600;">${cap[2]:,.0f}</span>
          </div>
        </div>'''
        st.markdown(_diag_html, unsafe_allow_html=True)

    with dc2:
        try:
            with pf.db_connection() as conn:
                cur = conn.cursor()
                _db_rows = []
                for tbl in ('current_positions', 'daily_portfolio_log',
                            'trade_execution_log', 'buffer_settings',
                            'price_data', 'strategy_signals'):
                    try:
                        cnt = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                        _db_rows.append(f'{tbl}: <span style="color:{T["font_bright"]};font-weight:600;">{cnt:,}</span>')
                    except Exception:
                        _db_rows.append(f'{tbl}: <span style="color:{T["negative"]};">not found</span>')
                r = cur.execute(
                    "SELECT MAX(timestamp) FROM price_data "
                    "WHERE interval='1d' AND category='linear'"
                ).fetchone()
                r2 = cur.execute(
                    "SELECT MAX(timestamp),MAX(created_at) FROM strategy_signals "
                    "WHERE interval='1d' AND category='linear'"
                ).fetchone()
            _freshness = f'Latest price: <span style="color:{T["font_bright"]};font-weight:600;">{r[0] if r else "N/A"}</span>'
            if r2 and r2[0]:
                _freshness += f'<br>Latest signal: <span style="color:{T["font_bright"]};font-weight:600;">{r2[0]}</span>'
                _freshness += f'<br>Computed: <span style="color:{T["font_bright"]};font-weight:600;">{r2[1]}</span>'
            _db_html = f'''
            <div style="background:{T['surface']};border:1px solid {T['border']};border-radius:6px;padding:14px;">
              <p style="font-size:11px;font-weight:600;color:{T['muted']};text-transform:uppercase;letter-spacing:0.06em;margin:0 0 10px 0;">Database</p>
              <div style="font-size:12px;font-family:monospace;line-height:1.8;color:{T['font']};">
                {"<br>".join(_db_rows)}
                <span style="border-top:1px solid {T['border']};display:block;margin:6px 0;"></span>
                {_freshness}
              </div>
            </div>'''
            st.markdown(_db_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(str(e))

    # Data quality
    if selected_symbols:
        st.markdown(f'<p style="font-size:14px;font-weight:600;color:{T["font"]};margin:16px 0 8px 0;">Data Quality</p>',
                    unsafe_allow_html=True)
        qc1, qc2 = st.columns(2)
        with qc1:
            st.markdown("**Price Data**")
            try:
                with pf.db_connection() as conn:
                    for sym in selected_symbols[:5]:
                        r = conn.execute(
                            "SELECT COUNT(*),MIN(timestamp),MAX(timestamp) FROM price_data "
                            "WHERE symbol=? AND interval='1d' AND category='linear'", (sym,)
                        ).fetchone()
                        if r and r[0]:
                            st.text(f"{sym}: {r[0]} days ({r[1]} to {r[2]})")
                        else:
                            st.text(f"{sym}: no data")
            except Exception as e:
                st.error(str(e))
        with qc2:
            st.markdown("**Signal Quality**")
            try:
                with pf.db_connection() as conn:
                    for sym in selected_symbols[:5]:
                        r = conn.execute(
                            "SELECT COUNT(*),AVG(ABS(combined_signal)) "
                            "FROM strategy_signals WHERE symbol=? AND interval='1d' "
                            "AND category='linear' AND combined_signal IS NOT NULL", (sym,)
                        ).fetchone()
                        if r and r[0]:
                            avg = r[1] or 0
                            st.text(f"{sym}: {r[0]} sigs, avg={avg:.1f}")
                        else:
                            st.text(f"{sym}: no signals")
            except Exception as e:
                st.error(str(e))

    # Buffer management
    st.markdown(f'<p style="font-size:14px;font-weight:600;color:{T["font"]};margin:16px 0 8px 0;">Buffer Management</p>',
                unsafe_allow_html=True)
    if selected_symbols:
        bc1, bc2 = st.columns(2)
        with bc1:
            bsym = st.selectbox("Symbol", selected_symbols, key="buf_sym")
            bval = st.slider(f"Buffer for {bsym}", 0.05, 0.50,
                              pm.get_buffer_setting(bsym), 0.01, key="buf_val")
            if st.button("Update Buffer"):
                with pf.db_connection() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO buffer_settings (symbol,buffer_pct,last_updated) "
                        "VALUES (?,?,?)", (bsym, bval, datetime.now()))
                    conn.commit()
                st.success(f"Buffer updated for {bsym}")
                st.rerun()
            bulk = st.slider("Bulk buffer", 0.05, 0.50, 0.10, 0.01, key="bulk_buf")
            if st.button("Apply to All"):
                with pf.db_connection() as conn:
                    for s in selected_symbols:
                        conn.execute(
                            "INSERT OR REPLACE INTO buffer_settings (symbol,buffer_pct,last_updated) "
                            "VALUES (?,?,?)", (s, bulk, datetime.now()))
                    conn.commit()
                st.success(f"Buffer={bulk:.1%} for {len(selected_symbols)} symbols")
                st.rerun()
        with bc2:
            st.markdown("**Current Buffers**")
            try:
                with pf.db_connection() as conn:
                    ph = ','.join('?' * len(selected_symbols))
                    bdf = pd.read_sql_query(
                        f"SELECT symbol, buffer_pct, last_updated FROM buffer_settings "
                        f"WHERE symbol IN ({ph}) ORDER BY symbol",
                        conn, params=selected_symbols,
                    )
                if not bdf.empty:
                    bdf['Buffer%'] = (bdf['buffer_pct'] * 100).round(1)
                    st.dataframe(bdf[['symbol', 'Buffer%']], use_container_width=True, hide_index=True)
                else:
                    st.info("Using defaults")
            except Exception:
                pass

    # Maintenance
    st.markdown(f'<p style="font-size:14px;font-weight:600;color:{T["font"]};margin:16px 0 8px 0;">Maintenance</p>',
                unsafe_allow_html=True)
    _mt1, _mt2 = st.columns(2)
    with _mt1:
        if st.button("Emergency Revert Portfolio", key="btn_revert", use_container_width=True):
            if pm.emergency_revert():
                st.success("Reverted from trade log")
                st.rerun()
            else:
                st.warning("Nothing to revert")
    with _mt2:
        if st.button("Fix Cumulative PnL", key="btn_fix_pnl", use_container_width=True):
            if pm.fix_cumulative_pnl():
                st.success("Fixed")
                st.rerun()
            else:
                st.info("Already correct")

    # Danger Zone
    with st.expander("Danger Zone", expanded=False):
        st.markdown(f'<p style="color:{T["negative"]};font-size:12px;font-weight:600;">'
                    f'This permanently deletes ALL position data, trade logs, and portfolio history.</p>',
                    unsafe_allow_html=True)
        confirm = st.checkbox("I understand this cannot be undone", key="reset_confirm")
        if confirm and st.button("RESET PORTFOLIO", key="reset_btn", type="primary"):
            try:
                with pf.db_connection() as conn:
                    conn.execute('BEGIN')
                    for tbl in ('current_positions', 'daily_portfolio_log',
                                'trade_execution_log', 'buffer_settings'):
                        try:
                            conn.execute(f"DELETE FROM {tbl}")
                        except Exception:
                            pass
                    conn.execute('COMMIT')
                st.success("Portfolio reset complete")
                st.rerun()
            except Exception as e:
                st.error(str(e))

st.markdown(f'<div style="border-top:1px solid {T["border"]};margin-top:24px;padding-top:8px;">'
            f'<span style="font-size:10px;color:{T["muted2"]};letter-spacing:0.05em;">'
            f'SIGNAL-WEIGHTED ALLOCATION WITH BUFFER ZONES</span></div>',
            unsafe_allow_html=True)

# ========================================================================
# Auto-refresh loop
# ========================================================================
if st.session_state.auto_refresh and selected_symbols:
    _elapsed = _time.time() - st.session_state.last_auto_refresh
    _remaining = st.session_state.auto_refresh_interval - _elapsed
    if _remaining <= 0:
        st.session_state.last_auto_refresh = _time.time()
        ok, msg, live = _update_data()
        if ok and live:
            pm.mark_to_market(live)
        st.rerun()
    else:
        # Sleep up to 5s then recheck (avoids blocking UI too long)
        _time.sleep(min(_remaining, 5))
        st.rerun()
