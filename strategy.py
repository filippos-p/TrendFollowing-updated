"""
Signal generation and single-symbol backtesting engine.
Produces EWMAC, BOLMOM, and BREAKOUT signals, then combines them.
"""

import numpy as np
import pandas as pd
from numba import njit

import config as cfg


# ---------------------------------------------------------------------------
# Technical signal calculators
# ---------------------------------------------------------------------------

def _ewmac(df, short_window, long_window, vol_lookback=30, cap=cfg.SIGNAL_CLIP):
    scalar = cfg.EWMAC_SCALARS.get((short_window, long_window), 1.0)
    price_diff_vol = (df['close'] - df['close'].shift(1)).ewm(span=vol_lookback).std()
    ema_short = df['close'].ewm(span=short_window, adjust=False).mean() / price_diff_vol
    ema_long = df['close'].ewm(span=long_window, adjust=False).mean() / price_diff_vol
    raw = ema_short - ema_long
    returns_vol = df['returns'].ewm(span=vol_lookback).std()
    signal = (scalar * raw / returns_vol).fillna(0).clip(-cap, cap)
    return signal


def _bolmom(df, lookback, cap=cfg.SIGNAL_CLIP):
    scalar = cfg.BOLMOM_SCALARS.get(lookback, 1.0)
    rolling_mean = df['close'].rolling(window=lookback).mean()
    rolling_std = df['close'].rolling(window=lookback).std()
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    mid = (upper + lower) / 2
    band_width = upper - lower
    returns_vol = df['returns'].ewm(span=30).std()
    signal = (scalar * (df['close'] - mid) / band_width) / returns_vol
    return signal.fillna(0).clip(-cap, cap)


def _breakout(df, lookback, cap=cfg.SIGNAL_CLIP):
    scalar = cfg.BREAKOUT_SCALARS.get(lookback, 1.0)
    rolling_max = df['close'].rolling(window=lookback).max()
    rolling_min = df['close'].rolling(window=lookback).min()
    midpoint = (rolling_max + rolling_min) / 2
    price_range = (rolling_max - rolling_min).replace(0, np.nan)
    normalized = (df['close'] - midpoint) / price_range
    returns_vol = df['returns'].ewm(span=30).std()
    signal = (normalized * 40 * scalar) / returns_vol
    return signal.fillna(0).clip(-cap, cap)


# ---------------------------------------------------------------------------
# Weighted combination with dynamic normalization
# ---------------------------------------------------------------------------

def _combine_weighted(signals, weights, multiplier, cap=cfg.SIGNAL_CLIP):
    """Combine a list of Series using weights with NaN-aware normalization."""
    df_signals = pd.concat(signals, axis=1)
    valid_weight_sum = df_signals.notna().mul(weights, axis=1).sum(axis=1)
    weighted_sum = df_signals.sum(axis=1, skipna=True)
    combined = weighted_sum / valid_weight_sum
    return (multiplier * combined).clip(-cap, cap)


# ---------------------------------------------------------------------------
# Main signal pipeline
# ---------------------------------------------------------------------------

def add_signals(df):
    """
    Compute all signals and add them as columns to *df*.
    Returns the augmented DataFrame.
    """
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['vol_forecast'] = df['returns'].ewm(span=30).std()

    # --- EWMAC ---
    ewmac_signals = []
    for (short, long), w in zip(cfg.EWMAC_COMBINATIONS, cfg.EWMAC_WEIGHTS):
        sig = _ewmac(df, short, long) * w
        df[f'ewmac_{short}_{long}'] = sig
        ewmac_signals.append(sig)
    df['ewmac_combined'] = _combine_weighted(
        ewmac_signals, cfg.EWMAC_WEIGHTS, cfg.EWMAC_MULTIPLIER
    )

    # --- BOLMOM ---
    bolmom_signals = []
    for win, w in zip(cfg.BOLMOM_WINDOWS, cfg.BOLMOM_WEIGHTS):
        sig = _bolmom(df, win) * w
        df[f'bolmom_{win}'] = sig
        bolmom_signals.append(sig)
    df['bolmom_combined'] = _combine_weighted(
        bolmom_signals, cfg.BOLMOM_WEIGHTS, cfg.BOLMOM_MULTIPLIER
    )

    # --- BREAKOUT ---
    breakout_signals = []
    for win, w in zip(cfg.BREAKOUT_WINDOWS, cfg.BREAKOUT_WEIGHTS):
        sig = _breakout(df, win) * w
        df[f'breakout_{win}'] = sig
        breakout_signals.append(sig)
    df['breakout_combined'] = _combine_weighted(
        breakout_signals, cfg.BREAKOUT_WEIGHTS, cfg.BREAKOUT_MULTIPLIER
    )

    # --- Final blend ---
    components = [
        df['ewmac_combined'] * cfg.FAMILY_WEIGHTS[0],
        df['bolmom_combined'] * cfg.FAMILY_WEIGHTS[1],
        df['breakout_combined'] * cfg.FAMILY_WEIGHTS[2],
    ]
    blend = pd.concat(components, axis=1)
    valid_w = blend.notna().mul(cfg.FAMILY_WEIGHTS, axis=1).sum(axis=1)
    df['combined_signal'] = (
        (cfg.COMBINED_MULTIPLIER * blend.sum(axis=1, skipna=True) / valid_w)
        .clip(-cfg.SIGNAL_CLIP, cfg.SIGNAL_CLIP)
    )
    return df


# ---------------------------------------------------------------------------
# Numba-accelerated single-symbol backtest
# ---------------------------------------------------------------------------

@njit
def backtest_positions(closes, volatilities, signals, target_volatility):
    """
    Realistic backtest: signal[i] -> position[i+1] -> PnL from close[i+1] to close[i+2].
    Returns (capitals, position_sizes, daily_pnls).
    """
    n = len(closes)
    capitals = np.zeros(n, dtype=np.float64)
    positions = np.zeros(n, dtype=np.float64)
    pnls = np.zeros(n, dtype=np.float64)
    capitals[0] = 1000.0

    for i in range(1, n - 1):
        block_value = volatilities[i - 1] * closes[i - 1] * np.sqrt(365.0)
        if block_value > 0 and capitals[i - 1] > 1.0:
            vol_target = target_volatility * capitals[i - 1]
            vol_scalar = vol_target / block_value
            raw = (signals[i - 1] * vol_scalar) / 10.0
            max_pos = abs(capitals[i - 1] / closes[i])
            positions[i] = np.sign(raw) * min(abs(raw), max_pos)
            pnls[i] = positions[i] * (closes[i + 1] - closes[i])
            capitals[i] = max(1.0, capitals[i - 1] + pnls[i])
        else:
            capitals[i] = capitals[i - 1]

    # Last day: position but no PnL
    if n > 1:
        j = n - 1
        bv = volatilities[j - 1] * closes[j - 1] * np.sqrt(365.0)
        if bv > 0 and capitals[j - 1] > 1.0:
            vs = (target_volatility * capitals[j - 1]) / bv
            raw = (signals[j - 1] * vs) / 10.0
            mp = abs(capitals[j - 1] / closes[j])
            positions[j] = np.sign(raw) * min(abs(raw), mp)
        capitals[j] = capitals[j - 1]

    return capitals, positions, pnls


def apply_strategy(df, target_volatility=cfg.TARGET_VOLATILITY):
    """Add signals and run backtest on a single-symbol DataFrame."""
    df = add_signals(df)
    closes = df['close'].values.astype(np.float64)
    vols = df['vol_forecast'].fillna(df['vol_forecast'].mean()).values.astype(np.float64)
    sigs = df['combined_signal'].fillna(0).values.astype(np.float64)

    capitals, pos, pnls = backtest_positions(closes, vols, sigs, target_volatility)
    df['total_capital'] = capitals
    df['position_size'] = pos
    df['daily_pnl'] = pnls
    df['daily_returns'] = df['daily_pnl'] / df['total_capital'].shift(1)
    return df


# ---------------------------------------------------------------------------
# Performance metrics (standalone use)
# ---------------------------------------------------------------------------

def performance_metrics(df):
    """Return a dict of key performance metrics from a backtested DataFrame."""
    rets = df['daily_returns'].dropna()
    total_ret = (df['total_capital'].iloc[-1] / df['total_capital'].iloc[0]) - 1
    ann_vol = rets.std() * np.sqrt(365)
    sharpe = (rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else 0
    cum = (1 + rets).cumprod()
    max_dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    return {
        'Total Return': f"{total_ret:.2%}",
        'Annualized Vol': f"{ann_vol:.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_dd:.2%}",
        'Final Capital': f"${df['total_capital'].iloc[-1]:,.0f}",
    }
