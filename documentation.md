# US Macro Rotation Module Documentation

## Overview
The US Macro Rotation module implements a monthly, macro-aware asset rotation strategy using a pooled machine learning model and multi-horizon momentum features. It builds a broad, diversified universe across equities, bonds, real assets, cash proxies, and crypto, then predicts forward returns for each asset and allocates to the top candidates while enforcing trend and risk constraints.

## Key Characteristics
- Monthly rebalance cadence (month start, after market open).
- Multi-horizon momentum features and regime features.
- Pooled Gradient Boosting model for cross-asset prediction.
- Trend filter based on a 147-day SMA.
- Optional Top-N selection.
- Cash proxy used both as a floor threshold and fallback allocation.
- Bitcoin allocation capped by a configurable maximum weight.

## Requirements
- QuantConnect Lean runtime (`QCAlgorithm`, `AlgorithmImports`).
- `pandas` for feature engineering.
- `scikit-learn` for `GradientBoostingRegressor`.

## Universe Definition
The algorithm defines a fixed universe of assets grouped by macro-sensitive categories:

- Equities (cyclical): `VDE`, `VIS`, `VAW`, `VNQ`, `VFH`
- Equities (growth): `VGT`, `VOX`, `VCR`
- Equities (defensive): `VPU`, `VDC`, `VHT`
- Real assets: `GLD`, `DBC`
- Bonds (short): `VGSH`, `VCSH`
- Bonds (intermediate): `VGIT`, `VCIT`
- Bonds (long): `VGLT`, `VCLT`
- Cash proxy: `SHV`
- USD proxy: `UUP`
- Crypto: `BTCUSD` (Bitfinex)

All securities are set to zero fee using `ConstantFeeModel(0)`.

## Data Configuration
- Resolution: `Daily` for all assets.
- Normalization: `TOTAL_RETURN` for history requests.
- Lookback: 504 trading days (approx. 2 years).

## Scheduling
Rebalance runs monthly:
- Date rule: month start, based on the first symbol.
- Time rule: 1 minute after market open.

Warm-up period is set to `self._lookback` to ensure full feature coverage.

## Parameters
The algorithm uses the following parameters and defaults:

- `max_bitcoin_weight` (float, default `0.25`): Upper cap on BTC allocation.
- `_lookback` (int, default `504`): Historical lookback window.
- `_ma_period` (int, default `147`): SMA period used in trend filter.
- `_top_n` (int, default `4`): Number of assets selected in Top-N mode.
- `_use_top_n` (bool, default `True`): Enables Top-N selection.
- `_reallocate_filtered_to_cash` (bool, default `False`): Controls how rejected slots are handled.

## User-Defined Allocation Modes
The following settings determine whether the strategy behaves more like a momentum portfolio or a general allocator:

- Momentum-tilted portfolio:
  - `_use_top_n = True`
  - `_top_n` set to a small number (e.g., 3-6)
  - Effect: Concentrates in the strongest predicted assets, increasing performance dispersion and drawdown sensitivity.
- Broad allocator portfolio:
  - `_use_top_n = False`
  - Effect: Includes all assets that clear the prediction floor (and SMA filter), producing a broader, more diversified allocation.

In allocator mode, the strategy is still driven by momentum signals, but the capital is spread across a wider set of eligible assets rather than focusing on only the top-ranked names. This tends to lower concentration risk at the cost of potentially weaker upside capture.

## High-Level Flow
At each rebalance:
1. Load total return price history for all symbols.
2. Build momentum, volatility, and regime features.
3. Train a pooled ML model on recent data and predict forward returns.
4. Apply a cash floor threshold based on the cash proxy prediction.
5. Apply Top-N filtering and SMA trend filtering.
6. Construct final weights with optional cash reallocation.
7. Enforce a max BTC weight and rebalance.

## Feature Engineering
The strategy generates features per asset and date, including:

### Asset Momentum
- `mom_21`, `mom_63`, `mom_126`, `mom_189`, `mom_252`: Percent change over each lookback.
- `mom_avg`: Average of the above momentum windows.

### Cash-Relative Momentum
- `mom_vs_cash_XX`: Asset momentum minus cash proxy momentum for each lookback.
- `mom_vs_cash_avg`: Average of cash-relative momentum windows.
- `asset_beats_cash`: Binary flag indicating if `mom_vs_cash_avg > 0`.

### Trend and Distance to SMA
- `sma_147`: 147-day moving average.
- `trend_up`: Binary flag for price above SMA.
- `sma_dist`: Normalized distance from SMA.

### Volatility and Risk
- `rv_126`: 126-day realized volatility (annualized).
- `high_vol`, `low_vol`: Flags based on rolling 75th/25th percentiles.
- `mom_vol_adj`: Momentum adjusted by volatility.

### Regime and Category Features
- Category momentum for each asset group (average across assets).
- Relative regime spreads:
  - `cyclical_vs_defensive`
  - `cyclical_vs_growth`
  - `growth_vs_defensive`
  - `bond_curve_slope`

### Credit Spread Regime
- Uses VCIT minus VGIT momentum to estimate credit spread behavior.
- Features: `credit_spread_21`, `credit_spread_63`, `credit_spread_126`, and `credit_spread_mom`.

## Model Training and Prediction
- Model: `GradientBoostingRegressor`.
- Training set: last 504 rows per symbol from the pooled dataset.
- Target: 21-day forward return (`fwd_ret`).
- Prediction: one value per symbol (latest feature row only).

If insufficient data is available, predictions return an empty series and the rebalance exits.

## Filtering and Eligibility
1. **Prediction Floor**: Any prediction must exceed the cash proxy prediction. The cash prediction is also floored at 0.0.
2. **Top-N (optional)**: The best `_top_n` predictions are selected to define candidate slots.
3. **Trend Filter**: Candidates must have price above the 147-day SMA.

Filter results are stored in `FilterResult`:
- `passed`: symbols passing all filters.
- `rejected`: candidates rejected by SMA.
- `candidates`: candidates after Top-N and floor filter.

## Portfolio Construction
- Each candidate represents an equal slot.
- If `passed` is empty, 100% is allocated to the cash proxy.
- If `reallocate_filtered_to_cash` is `True`, rejected slots go to cash.
- If `False`, rejected weight is redistributed proportionally to passed assets.
- BTC weight is capped at `max_bitcoin_weight`, with excess redistributed.

### Risk Profile of Cash Reallocation
- `reallocate_filtered_to_cash = True` (defensive tilt):
  - Increases exposure to the cash proxy when assets fail the SMA filter.
  - Typically lowers volatility and drawdowns in weak or choppy regimes.
  - May reduce upside capture during sharp recoveries if fewer assets pass filters.
- `reallocate_filtered_to_cash = False` (risk-on tilt):
  - Keeps the portfolio fully invested by concentrating rejected weight into passed assets.
  - Typically increases volatility and drawdown potential due to higher concentration.
  - May improve upside capture when trend signals are strong and persistent.

## Execution Model
- Positions not in target weights are fully liquidated.
- Target holdings are set with a 0.999 multiplier to avoid margin issues.
- Debug messages log final holdings after each rebalance.

## Operational Notes
- The cash proxy (`SHV`) acts as both a prediction baseline and a safe asset.
- All history requests use total return data to include dividends.
- The pooled model assumes features are comparable across assets.

## Potential Extensions
- Add transaction cost and slippage modeling.
- Introduce risk parity or volatility targeting for position sizing.
- Incorporate macroeconomic data feeds as additional features.
- Use cross-validation or time-series split for more robust model evaluation.
- Add confidence thresholds or uncertainty-aware allocation.

## Limitations and Risks
- The pooled model may be sensitive to regime shifts or structural breaks.
- Feature generation depends on complete price history across symbols.
- The SMA filter can lag in fast-moving regimes.
- Predictions use historical relationships without explicit stationarity checks.
- Cryptocurrency exposure adds non-trivial tail risk.

## Troubleshooting
- If the algorithm stays in cash, check:
  - The cash proxy prediction relative to other assets.
  - SMA filter rejecting all candidates.
  - Insufficient historical data.
- If predictions are empty:
  - Verify `history` returns and data normalization settings.
  - Confirm lookback is long enough for feature windows.

## Reproducibility
- `random_state=1` is set for the Gradient Boosting model to ensure deterministic training behavior.

## Example Usage
This module is intended to be run directly in QuantConnect Lean:

1. Open `S&P Sectors/main.py` in a Lean project.
2. Configure parameter `max_bitcoin_weight` if desired.
3. Run a backtest over the default date range.
4. Inspect monthly allocations and debug output to validate behavior.
