# Crypto Trend Following — Research & Live Trading Dashboard

Momentum research on cryptocurrency markets, signal construction from Robert Carver's systematic trading framework, and a live Streamlit dashboard for portfolio management with volatility-targeted position sizing.

![Dashboard 1](disppic1.PNG)
![Dashboard 2](disppic2.PNG)
![Dashboard 3](disppic3.PNG)
![Dashboard 4](disppic4.PNG)

We also created a new tab, where you can see the combined and individual forecasts as a timeseries for each coin in the trading universe. Along with that there is a relative (to the allocation weight — reminder that we equal weight all the coins from our universe) position size as a bar chart, where you can see the calculated position for each coin. This helps visualize what kind of position (both in size and in direction) we take and how price moves after our position. Or you can flip it and see what kind of positions we take because of the current price. After all we are just trend following and our position just scales based on the product of the (price constructed) signal times our volatility target exposure relative to the recent volatility.

![Symbol Analysis](newtab.png)

---

## Theoretical Framework

### Data

Daily data from 2018 onwards, sourced from Bybit's API. The universe consists of the top cryptocurrencies by market capitalization (excluding stablecoins and unavailable pairs). We use USDT perpetual futures — the most liquid trading pairs.

Trend works better on lower-volatility assets, and smaller-cap coins tend to be more volatile due to thinner order books. For that reason the top-decile volatility coins are excluded from the portfolio (more on this in [Research Findings](#research-findings)).

### Signal Construction

Three momentum-based trading rules, each producing continuous (not binary) forecasts — stronger signals yield larger position sizes:

**EWMAC (Exponential Weighted Moving Average Crossover)**

Taken from Robert Carver's *Systematic Trading*. We compute exponential moving averages at two speeds and take their difference, normalized by rolling volatility. Lookback windows follow a geometric progression (2, 4, 8, 16, 32, 64, 128, 256) — this guarantees that any pair of adjacent lookbacks has the same pairwise correlation, approximately `1/sqrt(a)` where `a` is the ratio between them.

We can verify the correlation symmetry in practice:
![adjacent_lookbacks](https://github.com/user-attachments/assets/0742ea85-2fe4-4b6c-aa51-4039c25f0cd6)

Six EWMAC variations are used (pairs of fast/slow lookbacks): `2_8, 4_16, 8_32, 16_64, 32_128, 64_256`.

**Forecast Scaling**

All signals are scaled so the mean absolute forecast equals 10. A forecast of +10 is an average buy, -10 an average sell. Forecasts are capped at ±20 (double the average) to prevent extreme position sizes.

Scalars are computed per variation by averaging the absolute forecasts across the full universe, then dividing 10 by that average.

Distribution of average forecasts across symbols and EWMAC variations:
![ewmac_vars_distr](https://github.com/user-attachments/assets/97e4f453-90a8-44af-bc61-ac6e05f7f45c)
![ewmac_vars_distr_2](https://github.com/user-attachments/assets/16c08dbd-5d4d-4970-8232-4b27f352cbbd)

The scaling applies uniformly across assets.

**Weighting via Handcrafting**

Since we don't know which lookback works best (and don't want to overfit by picking the backtested winner), we weight variations by their correlation structure using Carver's handcrafting method:
![handcrafting](https://github.com/user-attachments/assets/cd30cb67-3187-48ef-b88b-24838a688c07)

Variations are split into two groups (short: `2_8, 4_16, 8_32` and long: `16_64, 32_128, 64_256`) with weights `42%, 16%, 42%` within each group, halved across groups.

We adjust weights to reduce turnover — the fastest pair (`2_8`) switches forecast far more frequently, which incurs trading costs:
![dnld](https://github.com/user-attachments/assets/3c02c31b-4591-41fc-9952-c816ff849eeb)

Visualized for BTC/USDT:
![side_2_side_turnover](https://github.com/user-attachments/assets/605a8de8-e2c6-4739-a1e7-44713cf9e863)

The two fastest variations are highly correlated anyway:
![corr](https://github.com/user-attachments/assets/eb2e7262-ac6c-4332-8e18-1517057b34de)

So we shift weight from `2_8` to `4_16` (and symmetrically from `16_64` to `32_128`).

The combined EWMAC signal is then rescaled (multiplier ~1.30) so the combined average absolute forecast remains 10:
![comb_sig_theor_code](https://github.com/user-attachments/assets/3fcf6eb6-79e6-4e71-be64-40c6023293c7)

Resulting distribution:
![combined_ewmac_distr](https://github.com/user-attachments/assets/2932bd43-58c4-43f2-a4f2-f617697a392b)

**BOLMOM (Bollinger Band Momentum)**

Measures distance from a rolling mean ± 2σ band (adapted from [@ScottPh77711570](https://twitter.com/ScottPh77711570)). Equal-weighted across all lookback variations.

**Breakout**

Rolling min/max channel — calculates normalized distance from the channel midpoint. Same weight structure as EWMAC.

For BOLMOM and Breakout, the mean forecast distributes more naturally around 10 than EWMAC, likely due to the volatility and min/max components in their construction:
![image](https://github.com/user-attachments/assets/5a7ced6e-76b7-437b-b3b5-8a25b31607db)

### Research Findings

Performance is measured via Information Coefficient (IC) — the rank correlation between forecast and subsequent returns. Framework for what constitutes "good" IC:
![ic_perf_bench](https://github.com/user-attachments/assets/13602072-68ab-4c94-af93-582e6feb8dcf)

Trading costs assumed at 0.10% (2x Binance taker fee of 0.05%, to account for daily turnover scenarios).

Combined signal IC:
![image](https://github.com/user-attachments/assets/2be2ecc3-ad11-45f5-88e4-afcd2653aa52)

**Volatility decile analysis** confirms trend works better on lower-volatility coins. After excluding the top volatility decile:
![image](https://github.com/user-attachments/assets/5bb56b56-ca5c-400d-bf85-0b4d3a3e414f)

**Volatility-adjusting** all three signals (not just EWMAC) further improves IC:

Before:
![signals_by_decile](https://github.com/user-attachments/assets/4a9f654e-aa9a-41c6-abb9-f7d7320b606a)

After:
![signals_vol_stand_by_decile](https://github.com/user-attachments/assets/b041b37d-54fc-4514-9e8d-8a899f2eb391)

Combined signal improvement (top decile excluded):
![image](https://github.com/user-attachments/assets/a75c4584-7e71-495c-84b3-972e768bba67)

**Signal decay** — predictive power by forward horizon:
![image](https://github.com/user-attachments/assets/5f6ff7e2-a56d-48b8-882e-c699a13d4a2e)

Signals are strongest within the first week and remain robust up to two weeks out.

IC before and after volatility adjustment:
![image](https://github.com/user-attachments/assets/a1d79f69-b001-4bcd-83c1-ee25f3f336c6)

**Conclusion**: Trend following produces a weak but robust edge. Performance improves significantly when adjusting for asset volatility, both in signal construction and universe selection.

### Position Sizing

With a daily forecast scaled across the universe, positions are sized using a volatility target:

1. **Block value** = daily volatility forecast (annualized) × close price — captures the expected daily cash volatility of the asset
2. **Annual cash vol target** = target volatility × trading capital (capital updates with realized PnL)
3. **Volatility scalar** = annual cash vol target / block value
4. **Position** = `(signal × vol_scalar) / 10`

The annual vol target is a long-term average — realized vol will differ because:
- Our vol forecast is naive (rolling 30-day EWM std)
- Position size depends on forecast strength, not just vol
- Portfolio-level vol is lower than the sum of individual vols due to imperfect correlation:

![variance_calc](https://github.com/user-attachments/assets/53b49597-34bc-421c-90e4-f97ed4b66ed2)

The variance sums only for uncorrelated assets. For correlated ones, we get the grand sum of the covariance matrix — practically always lower than the naive sum.

---

## Live Trading Dashboard

### Dashboard Tabs

| Tab | What it does |
|-----|-------------|
| **Symbol Analysis** | Candlestick charts, forecast gauge (±20 scale), signal bars (EWMAC/BOLMOM/Breakout/Combined), relative position size — all with synchronized zoom and crosshair |
| **Position Management** | Target positions, buffer zones, trade execution (individual or batch) |
| **Portfolio Overview** | M2M, exposure breakdown (net/gross/long/short), portfolio history chart |
| **Trade Log** | Filterable trade history with CSV export |
| **System Status** | Diagnostics, data quality, buffer management, portfolio reset |

### Key Features

- **Live data** from Bybit REST API (no API key needed for public market data)
- **Volatility-targeted sizing** per Carver's framework with real-time capital tracking
- **Buffer zones** to reduce turnover — trades only execute when positions breach configurable thresholds
- **Mark-to-market** with live ticker prices, correct handling of flips (long→short), partial closes, and fresh opens
- **Multi-timeframe storage**: 1d, 4h, 1h, 15m, 5m data stored separately in SQLite (WAL mode) for future strategy expansion

### Quick Start

```bash
pip install streamlit pandas numpy numba plotly requests

# Fetch price data from Bybit
python data_fetcher.py

# Launch dashboard
streamlit run dashboard.py
```

### Project Structure

```
config.py        — Strategy params, forecast scalars, symbol universe
strategy.py      — Signal generation (EWMAC, BOLMOM, BREAKOUT) + Numba backtest engine
portfolio.py     — Position sizing, trade execution, M2M, SQLite helpers
dashboard.py     — Streamlit UI (5 tabs)
data_fetcher.py  — Bybit REST API data fetcher with parallel downloads
```

---

## Future Work

- **Auto-refresh** — Scheduled data updates and M2M at configurable intervals
- **Multi-timeframe strategies** — Leverage the 4h/1h/15m data already stored for intraday signal generation
- **Cloud deployment** — Move from SQLite to a cloud DB for persistent hosted access
- **Instrument weights** — Configurable per-symbol allocation beyond equal-weight
- **Portfolio-level risk** — Correlation-aware position sizing and portfolio volatility tracking
