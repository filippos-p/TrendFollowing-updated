"""
Streamlit live trading dashboard.
Run with:  streamlit run dashboard.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
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

st.title("Live Trading Dashboard")
st.caption("Position management with buffer zones, signal-weighted allocation, and mark-to-market PnL")

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

pm: pf.PortfolioManager = st.session_state.pm


# ========================================================================
# Cached data loaders
# ========================================================================

@st.cache_data(ttl=300)
def _universe():
    return pf.get_trading_universe()


@st.cache_data(ttl=60)
def _prices(symbols):
    return pf.get_latest_prices(list(symbols), lookback_days=1)


@st.cache_data(ttl=60)
def _signals(symbols):
    return pf.get_latest_signals(list(symbols))


@st.cache_data(ttl=120)
def _ohlc_history(symbol, start_date=None, end_date=None):
    """Fetch daily OHLC data for a symbol within an optional date range."""
    with pf.db_connection() as conn:
        q = ("SELECT timestamp, open, high, low, close, volume FROM price_data "
             "WHERE symbol=? AND interval='1d' AND category='linear'")
        params = [symbol]
        if start_date:
            q += " AND timestamp >= ?"
            params.append(str(start_date))
        if end_date:
            q += " AND timestamp <= ?"
            params.append(str(end_date))
        q += " ORDER BY timestamp"
        df = pd.read_sql_query(q, conn, params=params)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_data(ttl=120)
def _signal_history(symbol, start_date=None, end_date=None):
    """Fetch full signal history for a symbol."""
    with pf.db_connection() as conn:
        q = ("SELECT timestamp, combined_signal, ewmac_combined, bolmom_combined, "
             "breakout_combined, vol_forecast, close FROM strategy_signals "
             "WHERE symbol=? AND interval='1d' AND category='linear'")
        params = [symbol]
        if start_date:
            q += " AND timestamp >= ?"
            params.append(str(start_date))
        if end_date:
            q += " AND timestamp <= ?"
            params.append(str(end_date))
        q += " ORDER BY timestamp"
        df = pd.read_sql_query(q, conn, params=params)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_data(ttl=120)
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
        _prices.clear()  # invalidate cached DB prices so UI shows fresh data


# ========================================================================
# Sidebar
# ========================================================================

st.sidebar.title("Trading Controls")

if st.sidebar.button("Optimize Database (run once)"):
    with st.spinner("Optimizing..."):
        pf.optimize_indexes()
        st.success("Database optimized.")

st.sidebar.subheader("Data Management")
trading_symbols = _universe()

if trading_symbols:
    mode = st.sidebar.radio("Symbol selection", ["Custom", "All"])
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


BYBIT_INTERVALS = ['1d', '4h', '1h', '15m', '5m']
fetch_interval = st.sidebar.selectbox(
    "Fetch timeframe", BYBIT_INTERVALS, index=0, key="fetch_tf",
    help="Default: 1d. Other timeframes stored separately in DB for future strategies.",
)


def _update_data():
    try:
        from data_fetcher import DataConfig, CryptoDataCollector
        c = CryptoDataCollector(DataConfig(
            db_path=str(cfg.DB_PATH), max_workers=7,
            request_interval=0.2, backup_enabled=False,
        ))
        c.update_price_data(selected_symbols, category='linear',
                            intervals=[fetch_interval])
        _universe.clear()
        _prices.clear()
        _signals.clear()
        return True, f"Updated {len(selected_symbols)} symbols ({fetch_interval})"
    except Exception as e:
        return False, str(e)


c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Update Data", key="btn_data"):
        if selected_symbols:
            with st.spinner("Fetching..."):
                ok, msg = _update_data()
                (st.success if ok else st.error)(msg)
                if ok:
                    _do_mtm()
                    st.rerun()
with c2:
    if st.button("Calc Signals", key="btn_sig"):
        if selected_symbols:
            with st.spinner("Computing..."):
                bar = st.progress(0)
                ok, err, msgs = pf.compute_and_save_signals(
                    selected_symbols, strategy, progress_cb=bar.progress,
                )
                st.success(f"Computed {ok} symbols")
                if err:
                    st.warning(f"{err} errors")
                with st.expander("Details"):
                    for m in msgs:
                        st.text(m)
                _signals.clear()
                _prices.clear()
                st.rerun()

if st.sidebar.button("Update Data + Signals", key="btn_both"):
    if selected_symbols:
        with st.spinner("Updating data & signals..."):
            ok, msg = _update_data()
            if ok:
                bar = st.progress(0)
                sok, serr, smsgs = pf.compute_and_save_signals(
                    selected_symbols, strategy, progress_cb=bar.progress,
                )
                st.success(f"Data + {sok} signals updated")
                _do_mtm()
                st.rerun()
            else:
                st.error(msg)


# Universe change check
uc = pm.handle_universe_change(trading_symbols, st.session_state.prev_symbols)
if uc.get('message') and uc['message'] != 'No changes':
    st.sidebar.info(uc['message'])
st.session_state.prev_symbols = trading_symbols.copy()

# Portfolio settings
st.sidebar.subheader("Portfolio Settings")
cap_input = st.sidebar.number_input("Initial Capital ($)", 10, 10_000_000, cfg.DASHBOARD_INITIAL_CAPITAL, 100)
tv = st.sidebar.slider("Target Volatility", 0.10, 1.00, cfg.DASHBOARD_TARGET_VOLATILITY, 0.05)
buf = st.sidebar.slider("Default Buffer Zone", 0.05, 0.50, cfg.DASHBOARD_DEFAULT_BUFFER, 0.01)
pm.initial_capital = cap_input
pm.target_volatility = tv
pm.default_buffer_pct = buf

# ========================================================================
# Tabs
# ========================================================================

tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Symbol Analysis", "Position Management", "Portfolio Overview", "Trade Log", "System Status"
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
                "Quick range", ["Last 6 months", "Last 1 year",
                                "Last 2 years", "All time", "Custom"],
                index=1, key="sa_quick",
            )
        today = datetime.now().date()
        if sa_quick == "Custom":
            with _sa_top2:
                sa_start = st.date_input("From", today - timedelta(days=365), key="sa_from")
            with _sa_top3:
                sa_end = st.date_input("To", today, key="sa_to")
        else:
            range_map = {
                "Last 6 months": 180, "Last 1 year": 365,
                "Last 2 years": 730, "All time": None,
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
            # LEFT: Stacked candlestick + forecasts  |  RIGHT: gauge + rating + signals
            # ============================================================
            col_charts, col_panel = st.columns([5, 2])

            with col_charts:
                # --- Build combined figure: candlestick on top, forecasts below ---
                from plotly.subplots import make_subplots
                fig_combo = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.6, 0.4],
                    subplot_titles=[sa_sym, 'Forecasts'],
                )

                # Candlestick
                fig_combo.add_trace(go.Candlestick(
                    x=ohlc['timestamp'], open=ohlc['open'], high=ohlc['high'],
                    low=ohlc['low'], close=ohlc['close'],
                    increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                    increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350',
                    name='Price', showlegend=False,
                ), row=1, col=1)

                # Forecasts
                if not sig_hist.empty:
                    fig_combo.add_trace(go.Scatter(
                        x=sig_hist['timestamp'], y=sig_hist['combined_signal'],
                        mode='lines', name='Combined',
                        line=dict(color='white', width=2),
                    ), row=2, col=1)
                    for _col, _name, _clr in [
                        ('ewmac_combined', 'EWMAC', '#ab47bc'),
                        ('bolmom_combined', 'BOLMOM', '#ff7043'),
                        ('breakout_combined', 'Breakout', '#66bb6a'),
                    ]:
                        if _col in sig_hist.columns:
                            fig_combo.add_trace(go.Scatter(
                                x=sig_hist['timestamp'], y=sig_hist[_col],
                                mode='lines', name=_name,
                                line=dict(color=_clr, width=1.2),
                                opacity=0.75,
                            ), row=2, col=1)
                    fig_combo.add_hline(y=0, line_dash="dot", line_color="#555",
                                        line_width=1, row=2, col=1)

                _dark_bg = '#181924'
                _grid = '#2a2a3e'
                fig_combo.update_layout(
                    height=560, margin=dict(l=0, r=0, t=28, b=0),
                    plot_bgcolor=_dark_bg, paper_bgcolor=_dark_bg,
                    font=dict(color='#c0c0c0', size=11),
                    legend=dict(orientation='h', y=-0.04, x=0.5, xanchor='center',
                                font=dict(size=10)),
                    xaxis_rangeslider_visible=False,
                    xaxis2=dict(gridcolor=_grid, tickformat='%b %Y'),
                    yaxis=dict(gridcolor=_grid, title='Price'),
                    yaxis2=dict(gridcolor=_grid, title='Signal', range=[-22, 22]),
                    hovermode='x unified',
                )
                fig_combo.update_xaxes(gridcolor=_grid, row=1, col=1)
                fig_combo.update_annotations(font_size=12, font_color='#999')
                st.plotly_chart(fig_combo, use_container_width=True, key="combo_chart")

            with col_panel:
                # --- Forecast gauge (matching frcst.png) ---
                if not latest_sig.empty:
                    combined_val = float(latest_sig.iloc[0].get('combined_signal', 0) or 0)
                else:
                    combined_val = 0.0

                st.markdown(
                    '<p style="font-size:13px;color:#999;margin:0 0 2px 0;font-weight:600;">Forecast</p>',
                    unsafe_allow_html=True)

                # Thin outer arc = gradient steps, wide inner arc = fill to value
                # Build color steps for thin outer ring
                _gauge_steps = []
                _n_seg = 40
                for _i in range(_n_seg):
                    _lo = -20 + _i * (40 / _n_seg)
                    _hi = _lo + (40 / _n_seg)
                    _t = _i / (_n_seg - 1)  # 0..1
                    # dark red -> red -> gray -> green -> dark green
                    if _t < 0.25:
                        _r = int(140 + (200 - 140) * (_t / 0.25))
                        _g = int(20 + (50 - 20) * (_t / 0.25))
                        _b = 30
                    elif _t < 0.5:
                        _f = (_t - 0.25) / 0.25
                        _r = int(200 - (200 - 100) * _f)
                        _g = int(50 + (100 - 50) * _f)
                        _b = int(30 + (80 - 30) * _f)
                    elif _t < 0.75:
                        _f = (_t - 0.5) / 0.25
                        _r = int(100 - (100 - 50) * _f)
                        _g = int(100 + (180 - 100) * _f)
                        _b = int(80 - (80 - 40) * _f)
                    else:
                        _f = (_t - 0.75) / 0.25
                        _r = int(50 - (50 - 20) * _f)
                        _g = int(180 - (180 - 130) * _f)
                        _b = int(40 - (40 - 20) * _f)
                    _gauge_steps.append(dict(range=[_lo, _hi], color=f'rgb({_r},{_g},{_b})'))

                # Color for the filled bar (matches the value's position on the gradient)
                _vt = max(0, min(1, (combined_val + 20) / 40))
                if _vt < 0.25:
                    _br, _bg_c, _bb = int(140 + 60 * (_vt / 0.25)), int(20 + 30 * (_vt / 0.25)), 30
                elif _vt < 0.5:
                    _f = (_vt - 0.25) / 0.25
                    _br, _bg_c, _bb = int(200 - 100 * _f), int(50 + 50 * _f), int(30 + 50 * _f)
                elif _vt < 0.75:
                    _f = (_vt - 0.5) / 0.25
                    _br, _bg_c, _bb = int(100 - 50 * _f), int(100 + 80 * _f), int(80 - 40 * _f)
                else:
                    _f = (_vt - 0.75) / 0.25
                    _br, _bg_c, _bb = int(50 - 30 * _f), int(180 - 50 * _f), int(40 - 20 * _f)
                _bar_color = f'rgb({_br},{_bg_c},{_bb})'

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=combined_val,
                    number=dict(font=dict(size=38, color='white'), valueformat='+.1f'),
                    gauge=dict(
                        axis=dict(
                            range=[-20, 20],
                            tickwidth=1, tickcolor='#666',
                            tickvals=[-20, -15, -10, -5, 0, 5, 10, 15, 20],
                            tickfont=dict(size=9, color='#888'),
                            ticklen=6,
                        ),
                        bar=dict(color=_bar_color, thickness=0.65),
                        bgcolor='#252530',
                        borderwidth=0,
                        steps=_gauge_steps,
                        threshold=dict(
                            line=dict(color='white', width=2),
                            thickness=0.82, value=combined_val,
                        ),
                    ),
                ))
                fig_gauge.update_layout(
                    height=180, margin=dict(l=18, r=18, t=24, b=0),
                    paper_bgcolor='#181924', font_color='#c0c0c0',
                )
                st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart")

                # --- Buy Rating ---
                if combined_val <= -10:
                    rating_text, rating_color = "Strong Sell", "#c62828"
                elif combined_val <= -5:
                    rating_text, rating_color = "Sell", "#e53935"
                elif combined_val <= 5:
                    rating_text, rating_color = "Neutral", "#546e7a"
                elif combined_val <= 10:
                    rating_text, rating_color = "Buy", "#2e7d32"
                else:
                    rating_text, rating_color = "Strong Buy", "#1b5e20"

                st.markdown(
                    f"""<div style="text-align:center;padding:2px 0 10px 0;">
                    <span style="font-size:11px;color:#888;">Buy Rating</span><br>
                    <span style="font-size:20px;font-weight:700;color:white;
                    background:{rating_color};padding:5px 16px;border-radius:5px;
                    display:inline-block;margin-top:3px;">{rating_text}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # --- Signals (only combined families + overall) ---
                st.markdown(
                    '<p style="font-size:13px;color:#999;margin:8px 0 4px 0;font-weight:600;">Signals</p>',
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

                    def _signal_bar_color(v):
                        """Map signal value [-20,+20] to dark red -> dark green."""
                        t = max(0.0, min(1.0, (v + 20) / 40))
                        # Multi-stop gradient: dark red -> red -> dim -> green -> dark green
                        if t < 0.25:
                            r, g, b = 140 + int(60 * t / 0.25), 25, 25
                        elif t < 0.45:
                            f = (t - 0.25) / 0.20
                            r, g, b = int(200 - 80 * f), int(25 + 55 * f), int(25 + 35 * f)
                        elif t < 0.55:
                            f = (t - 0.45) / 0.10
                            r, g, b = int(120 - 40 * f), int(80 + 10 * f), int(60 + 10 * f)
                        elif t < 0.75:
                            f = (t - 0.55) / 0.20
                            r, g, b = int(80 - 50 * f), int(90 + 80 * f), int(70 - 35 * f)
                        else:
                            f = (t - 0.75) / 0.25
                            r, g, b = int(30 - 10 * f), int(170 - 40 * f), int(35 - 15 * f)
                        return f'rgb({r},{g},{b})'

                    _sig_html = ''
                    for _label, _col_name in _sig_items:
                        _val = float(_row.get(_col_name, 0) or 0)
                        _pct = max(0, min(100, (_val + 20) / 40 * 100))
                        _c = _signal_bar_color(_val)
                        _sig_html += f'''
                        <div style="display:flex;align-items:center;margin:3px 0;">
                          <span style="width:62px;font-size:11px;color:#bbb;flex-shrink:0;">{_label}</span>
                          <div style="flex:1;height:14px;background:#252530;border-radius:2px;overflow:hidden;margin:0 6px;">
                            <div style="width:{_pct:.1f}%;height:100%;background:linear-gradient(90deg,#8b1a1a,{_c});border-radius:2px;"></div>
                          </div>
                          <span style="width:36px;font-size:11px;color:white;text-align:right;">{_val:+.1f}</span>
                        </div>'''
                    st.markdown(_sig_html, unsafe_allow_html=True)
                else:
                    st.caption("No signal data.")

            # ============================================================
            # Relative Position Size
            # ============================================================
            if not sig_hist.empty and 'vol_forecast' in sig_hist.columns and 'close' in sig_hist.columns:
                st.markdown("#### Relative Position Size")
                st.caption(
                    "Position as % of per-symbol capital allocation. "
                    "+100% = fully long, -100% = fully short."
                )
                _sh = sig_hist.copy()
                # block_value = vol * price * sqrt(365)
                _sh['block_value'] = _sh['vol_forecast'] * _sh['close'] * np.sqrt(365)
                # vol_scalar = target_vol * alloc_capital / block_value
                # position_coins = signal * vol_scalar / 10
                # position_usd = position_coins * price = signal * target_vol * alloc / (block_value * 10) * price
                # position_pct = position_usd / alloc * 100 = signal * target_vol * price / (block_value * 10) * 100
                _sh['pos_pct'] = np.where(
                    _sh['block_value'] > 0,
                    (_sh['combined_signal'] * pm.target_volatility * _sh['close'])
                    / (_sh['block_value'] * cfg.POSITION_DIVISOR) * 100,
                    0,
                )

                fig_pos = go.Figure()
                _pos_colors = np.where(_sh['pos_pct'] >= 0, '#26a69a', '#ef5350')
                fig_pos.add_trace(go.Bar(
                    x=_sh['timestamp'], y=_sh['pos_pct'],
                    marker_color=_pos_colors.tolist(),
                    name='Position %',
                    hovertemplate='%{x|%Y-%m-%d}<br>Position: %{y:+.1f}%<extra></extra>',
                ))
                fig_pos.add_hline(y=0, line_color="#555", line_width=1)
                fig_pos.update_layout(
                    height=200, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor='#181924', paper_bgcolor='#181924',
                    font=dict(color='#c0c0c0', size=11),
                    xaxis=dict(gridcolor=_grid, tickformat='%b %Y'),
                    yaxis=dict(gridcolor=_grid, title='% of Allocation',
                               ticksuffix='%'),
                    bargap=0.05, showlegend=False,
                )
                st.plotly_chart(fig_pos, use_container_width=True, key="pos_size_chart")

# ------ TAB 1: Positions ------
with tab1:
    st.subheader("Position Management")
    if selected_symbols:
        with st.spinner("Loading..."):
            result = _load_positions(selected_symbols)
        if result is None:
            st.error("No data for selected symbols")
        else:
            positions, theo_cap, trading_cap, unrealized, sig_list, cur_pos = result
            if positions:
                pos_df = pd.DataFrame(positions)

                # Summary
                st.subheader("Portfolio Summary")
                exposure = cur_pos['position_usd'].abs().sum() if not cur_pos.empty else 0
                cash = trading_cap - exposure
                util = exposure / trading_cap * 100 if trading_cap else 0
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Trading Capital", f"${trading_cap:,.0f}")
                c2.metric("Exposure", f"${exposure:,.0f}")
                c3.metric("Cash", f"${cash:,.0f}")
                c4.metric("Unrealized PnL", f"${unrealized:,.0f}")
                c5.metric("Utilization", f"{util:.1f}%")

                # Position table
                st.subheader("Positions")
                disp = pos_df.copy()
                disp['Signal'] = disp['signal'].round(4)
                disp['Vol%'] = (disp['volatility'] * 100).round(3)
                disp['Price'] = disp['price'].round(4)
                disp['Current$'] = disp['current_position_usd'].round(2)
                disp['Target$'] = disp['achievable_position_usd'].round(2)
                disp['TargetCoins'] = disp['achievable_position_coins'].round(4)
                disp['Buffer'] = disp.apply(
                    lambda r: f"${r['buffer_lower']:.0f} - ${r['buffer_upper']:.0f}", axis=1)
                disp['Action'] = disp['action']
                disp['Trade$'] = disp['trade_amount_usd'].round(2)

                def _color_action(v):
                    if v == 'BUY': return 'background-color: lightgreen'
                    if v == 'SELL': return 'background-color: lightcoral'
                    return 'background-color: lightgray'

                st.dataframe(
                    disp[['symbol', 'Signal', 'Vol%', 'Price', 'Current$',
                          'Target$', 'TargetCoins', 'Buffer', 'Action', 'Trade$'
                          ]].style.applymap(_color_action, subset=['Action']),
                    use_container_width=True, height=500,
                )

                # Trade execution
                st.subheader("Execute Trades")
                needs_action = pos_df[pos_df['action'] != 'HOLD']
                if not needs_action.empty:
                    # Execute All button
                    ea_reason = st.selectbox("Reason for all", ["rebalance", "signal_change", "buffer_breach", "manual"],
                                             key="ea_reason")
                    if st.button("Execute All Trades", key="exec_all", type="primary"):
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
                            _do_mtm()
                            st.rerun()

                    st.divider()
                    # Individual trades
                    for _, p in needs_action.iterrows():
                        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
                        c1.write(f"**{p['symbol']}** | Cur: ${p['current_position_usd']:.0f} -> Tgt: ${p['achievable_position_usd']:.0f}")
                        c2.write(f"**{p['action']}** ${p['trade_amount_usd']:.0f} ({p['trade_amount_coins']:.4f} coins)")
                        reason = c3.selectbox("Reason", ["buffer_breach", "signal_change", "rebalance", "manual"],
                                              key=f"r_{p['symbol']}")
                        if c4.button("Execute", key=f"x_{p['symbol']}"):
                            ok = pm.execute_trade(p['symbol'], p['action'],
                                                  abs(p['trade_amount_usd']), abs(p['trade_amount_coins']),
                                                  p['price'], reason)
                            if ok:
                                st.success(f"{p['action']} executed for {p['symbol']}")
                                st.rerun()
                            else:
                                st.error(f"Failed for {p['symbol']}")
                        st.divider()
                else:
                    st.info("All positions within buffer zones.")

                # Manual entry
                with st.expander("Manual Trade Entry"):
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    msym = mc1.selectbox("Symbol", selected_symbols, key="m_sym")
                    mact = mc2.selectbox("Action", ["BUY", "SELL"], key="m_act")
                    musd = mc3.number_input("Amount ($)", 0.0, step=0.001, format="%.5f", key="m_usd")
                    mprc = mc4.number_input("Price", 0.00001, step=0.00001, format="%.6f", key="m_prc")
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
    st.subheader("Portfolio Overview")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Mark-to-Market"):
            _do_mtm()
            st.success("M2M updated")
            st.rerun()
    with c2:
        if st.button("Rebalancing Check"):
            if selected_symbols:
                result = _load_positions(selected_symbols)
                if result:
                    pos_list = result[0]
                    actions = [p for p in pos_list if p['action'] != 'HOLD']
                    if actions:
                        st.warning(f"{len(actions)} positions need rebalancing")
                        st.dataframe(pd.DataFrame(actions)[['symbol', 'current_position_usd',
                                                             'achievable_position_usd', 'action', 'trade_amount_usd']])
                    else:
                        st.success("No rebalancing needed")
    with c3:
        cap_info = pm.get_portfolio_capital()
        st.info(f"Capital: ${cap_info[0]:,.0f} | Cash: ${cap_info[1]:,.0f}")

    # History chart
    try:
        with pf.db_connection() as conn:
            hist = pd.read_sql_query(
                "SELECT date, total_capital, available_cash, total_exposure, "
                "unrealized_pnl, realized_pnl, cumulative_pnl, num_positions "
                "FROM daily_portfolio_log ORDER BY date DESC LIMIT 30", conn,
            )
        if not hist.empty:
            st.subheader("Portfolio History (30d)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist['date'], y=hist['total_capital'],
                                     mode='lines+markers', name='Trading Capital'))
            if 'unrealized_pnl' in hist.columns:
                fig.add_trace(go.Scatter(
                    x=hist['date'], y=hist['total_capital'] + hist['unrealized_pnl'],
                    mode='lines+markers', name='Total Value'))
            fig.add_trace(go.Scatter(x=hist['date'], y=hist['cumulative_pnl'],
                                     mode='lines+markers', name='Cumulative PnL'))
            fig.update_layout(
                height=400, xaxis_title="Date", yaxis_title="Value ($)",
                xaxis=dict(tickformat="%Y-%m-%d", dtick="D1"),
            )
            st.plotly_chart(fig, use_container_width=True)
            hist_disp = hist.copy()
            hist_disp['date'] = pd.to_datetime(hist_disp['date']).dt.strftime('%Y-%m-%d')
            st.dataframe(hist_disp.round(2), use_container_width=True)
        else:
            st.info("No portfolio history yet. Execute trades or run M2M.")
    except Exception as e:
        st.error(f"Error loading history: {e}")

    # Current positions detail
    cur_pos = pm.get_current_positions()
    if not cur_pos.empty:
        st.subheader("Exposure Breakdown")
        prices_df = _prices(tuple(selected_symbols)) if selected_symbols else pd.DataFrame()
        price_map = {}
        if not prices_df.empty:
            for _, r in prices_df.iterrows():
                price_map[r['symbol']] = r['close']

        rows = []
        total_unr = 0
        total_long = 0
        total_short = 0
        for _, p in cur_pos.iterrows():
            sym = p['symbol']
            cprice = price_map.get(sym, p['entry_price'] or 0)
            entry = p['entry_price'] if p['entry_price'] and p['entry_price'] > 0 else cprice
            coins = p['position_coins']
            market_val = coins * cprice  # signed: positive=long, negative=short
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
                    .applymap(_cpnl, subset=['unrealized', 'pnl%'])
                    .applymap(_cside, subset=['side']),
                use_container_width=True,
            )

# ------ TAB 3: Trade Log ------
with tab3:
    st.subheader("Trade Log")
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
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total Trades", len(trades))
            sc2.metric("Buys", len(trades[trades['action'] == 'BUY']))
            sc3.metric("Sells", len(trades[trades['action'] == 'SELL']))
            sc4.metric("Realized PnL", f"${trades['realized_pnl'].sum():.2f}")
            st.dataframe(trades.round(4), use_container_width=True, height=500)

            if st.button("Export CSV"):
                csv = trades.to_csv(index=False)
                st.download_button("Download", csv,
                                   f"trades_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
        else:
            st.info("No trades match filters")
    except Exception as e:
        st.error(f"Error: {e}")

# ------ TAB 4: System Status ------
with tab4:
    st.subheader("System Status")

    # Emergency tools
    ec1, ec2 = st.columns(2)
    with ec1:
        if st.button("Emergency Revert Portfolio"):
            if pm.emergency_revert():
                st.success("Reverted from trade log")
                st.rerun()
    with ec2:
        if st.button("Fix Cumulative PnL"):
            if pm.fix_cumulative_pnl():
                st.success("Fixed")
                st.rerun()

    # Diagnostics
    st.subheader("Diagnostics")
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("**Settings**")
        st.text(f"Capital: ${pm.initial_capital:,}")
        st.text(f"Target Vol: {pm.target_volatility:.1%}")
        st.text(f"Buffer: {pm.default_buffer_pct:.1%}")
        cap = pm.get_portfolio_capital()
        st.text(f"Current Capital: ${cap[0]:,.0f}")
        st.text(f"Cash: ${cap[1]:,.0f}")
        st.text(f"Realized PnL: ${cap[2]:,.0f}")

    with dc2:
        st.markdown("**Database**")
        try:
            with pf.db_connection() as conn:
                cur = conn.cursor()
                for tbl in ('current_positions', 'daily_portfolio_log',
                            'trade_execution_log', 'buffer_settings',
                            'price_data', 'strategy_signals'):
                    try:
                        cnt = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                        st.text(f"{tbl}: {cnt:,}")
                    except Exception:
                        st.text(f"{tbl}: not found")
                # freshness
                r = cur.execute(
                    "SELECT MAX(timestamp) FROM price_data "
                    "WHERE interval='1d' AND category='linear'"
                ).fetchone()
                st.text(f"Latest price: {r[0] if r else 'N/A'}")
                r2 = cur.execute(
                    "SELECT MAX(timestamp),MAX(created_at) FROM strategy_signals "
                    "WHERE interval='1d' AND category='linear'"
                ).fetchone()
                if r2 and r2[0]:
                    st.text(f"Latest signal: {r2[0]}")
                    st.text(f"Signals computed: {r2[1]}")
        except Exception as e:
            st.error(str(e))

    # Data quality
    if selected_symbols:
        st.subheader("Data Quality")
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
    st.subheader("Buffer Settings")
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

    # Portfolio reset
    st.subheader("Portfolio Reset")
    st.warning("This deletes ALL position data, trade logs, and portfolio history.")
    confirm = st.checkbox("I understand", key="reset_confirm")
    if confirm and st.button("RESET PORTFOLIO", key="reset_btn"):
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

st.markdown("---")
st.caption("Trading Dashboard | Signal-weighted allocation with buffer zones")
