# US Macro Rotation

Monthly, macro-aware asset rotation strategy using pooled machine learning and
multi-horizon momentum features. The algorithm predicts forward returns across
equities, bonds, real assets, cash, and crypto, then allocates to the strongest
candidates while enforcing trend and risk constraints.

## Highlights
- Monthly rebalance at month start, after market open.
- Pooled `GradientBoostingRegressor` model trained on cross-asset features.
- Multi-horizon momentum, volatility, and regime features.
- 147-day SMA trend filter and cash-proxy prediction floor.
- Optional Top-N selection with configurable cash reallocation.
- Bitcoin allocation capped by parameter.

## Universe
Grouped macro-sensitive assets:
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

All securities use `ConstantFeeModel(0)`.

## How It Works
At each monthly rebalance:
1. Load total-return price history (lookback: 504 trading days).
2. Build momentum, volatility, trend, and regime features.
3. Train a pooled ML model and predict 21-day forward returns.
4. Apply a cash-proxy prediction floor and optional Top-N filter.
5. Apply a 147-day SMA trend filter.
6. Build final weights with optional cash reallocation.
7. Enforce max BTC weight and rebalance.

## Feature Summary
- Momentum: 21/63/126/189/252-day percent changes, plus averages.
- Cash-relative momentum and `asset_beats_cash` flag.
- Trend/SMA distance and 147-day SMA filter.
- Realized volatility (126d), high/low vol flags, volatility-adjusted momentum.
- Regime signals: category momentum and spreads (e.g., cyclical vs defensive).
- Credit spread regime from VCIT minus VGIT momentum.

## Parameters
- `max_bitcoin_weight` (float, default `0.25`): cap on BTC allocation.
- `_lookback` (int, default `504`): historical window.
- `_ma_period` (int, default `147`): SMA period.
- `_top_n` (int, default `4`): Top-N candidates.
- `_use_top_n` (bool, default `True`): enable Top-N mode.
- `_reallocate_filtered_to_cash` (bool, default `False`):
  - `True`: rejected slots go to cash proxy.
  - `False`: rejected weight redistributes across passed assets.

## Requirements
- QuantConnect Lean (`QCAlgorithm`, `AlgorithmImports`)
- `pandas`
- `scikit-learn`

## Usage
Open `main.py` in a Lean project and run a backtest. Adjust
`max_bitcoin_weight` if desired. The algorithm logs final monthly holdings.

## Notes and Risks
- The pooled model assumes feature comparability across assets.
- SMA filter can lag in fast-moving regimes.
- Crypto exposure adds tail risk.
- Regime shifts can reduce model effectiveness.
# US-Macro-Rotation
