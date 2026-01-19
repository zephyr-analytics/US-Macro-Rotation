# region imports
from AlgorithmImports import *
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from typing import Dict, Sequence
# endregion


class FilterResult:
    """
    Container for filter output at a rebalance step.

    Parameters
    ----------
    passed : Sequence[Symbol]
        Symbols that cleared both the prediction floor and SMA filter.
    rejected : Sequence[Symbol]
        Symbols that met the prediction criteria but failed the SMA filter.
    candidates : Sequence[Symbol]
        Symbols considered after the prediction floor and Top-N filters.
    """

    def __init__(
        self,
        passed: Sequence[Symbol],
        rejected: Sequence[Symbol],
        candidates: Sequence[Symbol]
    ) -> None:
        self.passed = passed
        self.rejected = rejected
        self.candidates = candidates

class UsMacroRotation(QCAlgorithm):
    """
    U.S. macro-driven asset rotation algorithm with machine-learning-based
    return prediction, momentum filtering, and regime-aware allocation.
    """

    def initialize(self) -> None:
        """
        Initialize the algorithm configuration, universe, models, and schedules.

        Sets start date, capital, universe symbols (equities, bonds, crypto, cash),
        machine learning models, warmup period, and monthly rebalance schedule.

        Returns
        -------
        None
        """
        #----Backtest and account settings
        self.set_start_date(2012, 1, 1)
        self.settings.daily_precise_end_time = False
        self.settings.seed_initial_prices = True
        self.set_cash(1_000_000)

        #----User defined settings
        self._reallocate_filtered_to_cash = False
        self._use_top_n = True
        self._top_n = 4

        # ---- Define asset universe
        self._bitcoin = self.add_crypto(
            "BTCUSD",
            market=Market.BITFINEX,
            leverage=1.0
        ).symbol

        self._equities_cyclical = [
            self.add_equity("VDE", Resolution.Daily).symbol,
            self.add_equity("VIS", Resolution.Daily).symbol,
            self.add_equity("VAW", Resolution.Daily).symbol,
            self.add_equity("VNQ", Resolution.Daily).symbol,
            self.add_equity("VFH", Resolution.Daily).symbol,
        ]

        self._equities_growth = [
            self.add_equity("VGT", Resolution.Daily).symbol,
            self.add_equity("VOX", Resolution.Daily).symbol,
            self.add_equity("VCR", Resolution.Daily).symbol,
        ]

        self._equities_defensive = [
            self.add_equity("VPU", Resolution.Daily).symbol,
            self.add_equity("VDC", Resolution.Daily).symbol,
            self.add_equity("VHT", Resolution.Daily).symbol,
        ]

        self._real_assets = [
            self.add_equity("GLD", Resolution.Daily).symbol,
            self.add_equity("DBC", Resolution.Daily).symbol,
        ]

        self._bonds_short = [
            self.add_equity("VGSH", Resolution.Daily).symbol,
            self.add_equity("VCSH", Resolution.Daily).symbol,
        ]

        self._bonds_intermediate = [
            self.add_equity("VGIT", Resolution.Daily).symbol,
            self.add_equity("VCIT", Resolution.Daily).symbol,
        ]

        self._bonds_long = [
            self.add_equity("VGLT", Resolution.Daily).symbol,
            self.add_equity("VCLT", Resolution.Daily).symbol,
        ]

        self._cash_proxy = self.add_equity("SHV", Resolution.Daily).symbol
        self._usd = self.add_equity("UUP", Resolution.Daily).symbol

        self._symbols = (
            self._equities_cyclical
            + self._equities_growth
            + self._equities_defensive
            + self._real_assets
            + self._bonds_short
            + self._bonds_intermediate
            + self._bonds_long
            + [self._cash_proxy, self._bitcoin, self._usd]
        )

        for symbol in self._symbols:
            self.securities[symbol].fee_model = ConstantFeeModel(0)

        self._model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            subsample=0.7,
            loss="huber",
            random_state=1
        )

        self._max_bitcoin_weight = float(self.get_parameter("max_bitcoin_weight", 0.25))
        self._lookback = 504
        self._ma_period = 147

        self.schedule.on(
            self.date_rules.month_start(self._symbols[0]),
            self.time_rules.after_market_open(self._symbols[0], 1),
            self._rebalance
        )

        self.set_warm_up(self._lookback)


    def on_warmup_finished(self) -> None:
        """
        Trigger an initial rebalance once the warmup period is complete.

        Returns
        -------
        None
        """
        self._rebalance()


    # ========== MAIN REBALANCE ==========
    def _rebalance(self) -> None:
        """
        Execute the monthly rebalance routine.

        Fetches historical price data, computes momentum features,
        predicts forward returns, filters eligible assets, constructs
        final portfolio weights, and places trades.

        Returns
        -------
        None
        """
        if self.is_warming_up:
            return

        price_data = self._get_price_data()
        if price_data.empty:
            return

        features_df = self._compute_momentum_features(price_data)

        prediction_by_symbol = self._predict_returns(features_df, price_data)
        if self._cash_proxy not in prediction_by_symbol:
            return

        cash_threshold = prediction_by_symbol[self._cash_proxy]
        floor_threshold = max(0.0, cash_threshold)

        filt = self._filter_assets(prediction_by_symbol, floor_threshold, price_data)

        if not filt.passed:
            self.set_holdings(self._cash_proxy, 1.0)
            return

        final_weights = self._build_final_weights(filt)
        self._rebalance_sell_then_buy(final_weights)


    # ========== HELPERS ==========
    def _get_price_data(self) -> pd.DataFrame:
        """
        Retrieve historical total-return adjusted price data.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by date with symbols as columns containing
            closing prices.
        """
        history = self.history(
            self._symbols,
            self._lookback,
            Resolution.DAILY,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN
        )
        return history["close"].unstack(0).dropna()


    def _compute_momentum_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling multi-dimensional momentum features per asset and date.

        Parameters
        ----------
        price_data : pandas.DataFrame
            DataFrame indexed by date with symbols as columns containing prices.

        Returns
        -------
        pd.DataFrame
            Indexed by (date, symbol) with feature columns.
        """
        lookbacks = [21, 63, 126, 189, 252]
        records = []

        credit = price_data.get(self._bonds_intermediate[1])
        treasury = price_data.get(self._bonds_intermediate[0])

        spread_df = None
        if credit is not None and treasury is not None:
            spread_df = pd.DataFrame({
                "credit_spread_21":  (credit / credit.shift(21)  - 1.0) - (treasury / treasury.shift(21)  - 1.0),
                "credit_spread_63":  (credit / credit.shift(63)  - 1.0) - (treasury / treasury.shift(63)  - 1.0),
                "credit_spread_126": (credit / credit.shift(126) - 1.0) - (treasury / treasury.shift(126) - 1.0),
            })

            spread_df["credit_spread_mom"] = (
                spread_df["credit_spread_21"] +
                spread_df["credit_spread_63"] +
                spread_df["credit_spread_126"]
            ) / 3.0

        category_map = {
            "eq_cyclical": self._equities_cyclical,
            "eq_growth": self._equities_growth,
            "eq_defensive": self._equities_defensive,
            "real_assets": self._real_assets,
            "bond_short": self._bonds_short,
            "bond_intermediate": self._bonds_intermediate,
            "bond_long": self._bonds_long,
        }

        def _category_momentum(symbols):
            series_list = []

            for s in symbols:
                px = price_data.get(s)
                if px is None:
                    continue

                mom = pd.DataFrame(index=px.index)
                for lb in lookbacks:
                    mom[f"mom_{lb}"] = px / px.shift(lb) - 1.0

                mom["mom_avg"] = mom[[f"mom_{lb}" for lb in lookbacks]].mean(axis=1)
                series_list.append(mom["mom_avg"])

            if not series_list:
                return None

            return pd.concat(series_list, axis=1).mean(axis=1)

        category_df = pd.DataFrame(index=price_data.index)

        for name, symbols in category_map.items():
            mom = _category_momentum(symbols)
            if mom is not None:
                category_df[f"{name}_mom"] = mom

        category_df["cyclical_vs_defensive"] = (
            category_df.get("eq_cyclical_mom", 0.0) -
            category_df.get("eq_defensive_mom", 0.0)
        )

        category_df["cyclical_vs_growth"] = (
            category_df.get("eq_cyclical_mom", 0.0) -
            category_df.get("eq_growth_mom", 0.0)
        )

        category_df["growth_vs_defensive"] = (
            category_df.get("eq_growth_mom", 0.0) -
            category_df.get("eq_defensive_mom", 0.0)
        )

        category_df["bond_curve_slope"] = (
            category_df.get("bond_long_mom", 0.0) -
            category_df.get("bond_short_mom", 0.0)
        )

        cash_px = price_data.get(self._cash_proxy)
        cash_mom = {}

        if cash_px is not None:
            for lb in lookbacks:
                cash_mom[lb] = cash_px / cash_px.shift(lb) - 1.0

        for symbol in self._symbols:
            series = price_data.get(symbol)
            if series is None or len(series) < max(lookbacks) + 50:
                continue

            df = pd.DataFrame({"price": series})

            for lb in lookbacks:
                df[f"mom_{lb}"] = df["price"] / df["price"].shift(lb) - 1.0

            df["mom_avg"] = (
                df["mom_21"] +
                df["mom_63"] +
                df["mom_126"] +
                df["mom_189"] +
                df["mom_252"]
            ) / 5.0

            if cash_mom:
                for lb in lookbacks:
                    df[f"mom_vs_cash_{lb}"] = df[f"mom_{lb}"] - cash_mom[lb]

                df["mom_vs_cash_avg"] = (
                    df["mom_vs_cash_21"] +
                    df["mom_vs_cash_63"] +
                    df["mom_vs_cash_126"] +
                    df["mom_vs_cash_189"] +
                    df["mom_vs_cash_252"]
                ) / 5.0


            df["asset_beats_cash"] = (df["mom_vs_cash_avg"] > 0).astype(int)

            df["sma_147"] = df["price"].rolling(147).mean()
            df["trend_up"] = (df["price"] > df["sma_147"]).astype(int)
            df["sma_dist"] = (df["price"] - df["sma_147"]) / df["sma_147"]

            df["log_ret"] = np.log(df["price"] / df["price"].shift(1))
            df["rv_126"] = df["log_ret"].rolling(126).std() * np.sqrt(252)
            rv_q = df["rv_126"].rolling(252)
            df["high_vol"] = (df["rv_126"] > rv_q.quantile(0.75)).astype(int)
            df["low_vol"]  = (df["rv_126"] < rv_q.quantile(0.25)).astype(int)
            df["mom_vol_adj"] = df["mom_avg"] / df["rv_126"]

            if spread_df is not None:
                df = df.join(spread_df, how="left")

            df = df.join(category_df, how="left")

            df = df.dropna(subset=[
                "mom_21", "mom_63", "mom_126", "mom_189", "mom_252",
                "mom_avg", "sma_147"
            ])

            for dt, row in df.iterrows():
                records.append({
                    "date": dt,
                    "symbol": symbol,

                    # asset momentum
                    "mom_21": row["mom_21"],
                    "mom_63": row["mom_63"],
                    "mom_126": row["mom_126"],
                    "mom_189": row["mom_189"],
                    "mom_252": row["mom_252"],
                    "mom_avg": row["mom_avg"],

                    # trend / risk
                    "trend_up": row["trend_up"],
                    "sma_dist": row["sma_dist"],
                    "rv_126": row["rv_126"],
                    "high_vol": row["high_vol"],
                    "low_vol": row["low_vol"],
                    "mom_vol_adj": row["mom_vol_adj"],
                    "asset_beats_cash": row["asset_beats_cash"],

                    # credit regime
                    "credit_spread_mom": row.get("credit_spread_mom", 0.0),

                    # category momentum
                    "eq_cyclical_mom": row.get("eq_cyclical_mom", 0.0),
                    "eq_growth_mom": row.get("eq_growth_mom", 0.0),
                    "eq_defensive_mom": row.get("eq_defensive_mom", 0.0),
                    "real_assets_mom": row.get("real_assets_mom", 0.0),
                    "bond_short_mom": row.get("bond_short_mom", 0.0),
                    "bond_intermediate_mom": row.get("bond_intermediate_mom", 0.0),
                    "bond_long_mom": row.get("bond_long_mom", 0.0),

                    # relative regime
                    "cyclical_vs_defensive": row.get("cyclical_vs_defensive", 0.0),
                    "growth_vs_defensive": row.get("growth_vs_defensive", 0.0),
                    "cyclical_vs_growth": row.get("cyclical_vs_growth", 0.0),
                    "bond_curve_slope": row.get("bond_curve_slope", 0.0),
                })

        if not records:
            return pd.DataFrame()

        features = pd.DataFrame.from_records(records)
        return features.set_index(["date", "symbol"]).sort_index()


    def _predict_returns(
        self,
        features_df: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> pd.Series:
        """
        Predict forward 21-day returns using pooled multi-asset features.

        Parameters
        ----------
        features_df : pandas.DataFrame
            Multi-indexed by (date, symbol) with feature columns.
        price_data : pandas.DataFrame
            DataFrame indexed by date with symbols as columns containing prices.

        Returns
        -------
        pd.Series
            One predicted forward return per asset (symbol).
        """
        horizon = 21
        window = 504

        fwd_ret = (
            price_data
            .pct_change(horizon)
            .shift(-horizon)
            .stack()
        )

        fwd_ret.index = fwd_ret.index.set_names(["date", "symbol"])
        fwd_ret = fwd_ret.rename("fwd_ret")

        fwd_ret = fwd_ret.to_frame("fwd_ret")

        df = pd.concat([features_df, fwd_ret], axis=1, join="inner").dropna()

        if df.empty:
            return pd.Series(dtype=float)

        df["fwd_ret_risk_adj"] = df["fwd_ret"] / df["rv_126"]

        df["fwd_ret_risk_adj"] = df["fwd_ret_risk_adj"].clip(-5.0, 5.0)

        train_df = df.groupby(level=1).tail(window)

        feature_cols = [c for c in train_df.columns if c not in ["fwd_ret", "fwd_ret_risk_adj"]]

        if len(train_df) < 500:
            return pd.Series(dtype=float)

        X = train_df[feature_cols].values
        y = train_df["fwd_ret"].values

        self._model.fit(X, y)

        preds = {}

        for symbol in features_df.index.get_level_values(1).unique():
            sym_df = features_df.xs(symbol, level=1)

            if sym_df.empty:
                continue

            latest_row = sym_df.tail(1)[feature_cols]
            X_latest = latest_row.values
            preds[symbol] = float(self._model.predict(X_latest)[0])

        return pd.Series(preds)


    def _filter_assets(
        self,
        preds: pd.Series,
        floor: float,
        price_data: pd.DataFrame
    ) -> FilterResult:
        """
        Apply prediction floor, optional Top-N, and SMA trend filters.

        Parameters
        ----------
        preds : pandas.Series
            Predicted forward returns indexed by symbol.
        floor : float
            Minimum prediction threshold derived from the cash proxy.
        price_data : pandas.DataFrame
            DataFrame indexed by date with symbols as columns containing prices.

        Returns
        -------
        FilterResult
            Result object with passed, rejected, and candidate symbols.
        """
        eligible = preds[preds > floor].drop(
            labels=[self._cash_proxy],
            errors="ignore"
        )

        if eligible.empty:
            return FilterResult([], [], [])

        if self._use_top_n:
            eligible = eligible.nlargest(self._top_n)

        candidates = list(eligible.index)

        passed, rejected = [], []

        for symbol in candidates:
            series = price_data.get(symbol)
            if series is None or len(series) < self._ma_period:
                rejected.append(symbol)
                continue

            price = series.iloc[-1]
            sma = series.iloc[-self._ma_period:].mean()

            if price > sma:
                passed.append(symbol)
            else:
                rejected.append(symbol)

        return FilterResult(passed, rejected, candidates)


    def _build_final_weights(self, filt: FilterResult) -> Dict[Symbol, float]:
        """
        Construct final portfolio weights based on filter results.

        Parameters
        ----------
        filt : FilterResult
            Output from the prediction and SMA filtering stage.

        Returns
        -------
        dict[Symbol, float]
            Target weights keyed by symbol.
        """

        passed = filt.passed
        rejected = filt.rejected
        candidates = filt.candidates

        if not passed:
            return {self._cash_proxy: 1.0}

        weights = {}

        if self._use_top_n:
            total_slots = self._top_n
        else:
            total_slots = len(candidates)

        if total_slots == 0:
            return {self._cash_proxy: 1.0}

        slot_w = 1.0 / total_slots

        for s in passed:
            weights[s] = slot_w

        rejected_slots = len(rejected)
        rejected_weight = slot_w * rejected_slots

        if self._reallocate_filtered_to_cash:
            weights[self._cash_proxy] = rejected_weight
        else:
            add = rejected_weight / len(passed)
            for s in passed:
                weights[s] += add
            weights[self._cash_proxy] = 0.0

        if self._bitcoin in weights:
            btc_weight = weights[self._bitcoin]

            if btc_weight > self._max_bitcoin_weight:
                excess = btc_weight - self._max_bitcoin_weight
                weights[self._bitcoin] = self._max_bitcoin_weight

                others = [
                    s for s in weights
                    if s not in (self._bitcoin, self._cash_proxy)
                ]

                if others:
                    total_other = sum(weights[s] for s in others)
                    for s in others:
                        weights[s] += (weights[s] / total_other) * excess
                else:
                    weights[self._cash_proxy] += excess

        return weights


    def _rebalance_sell_then_buy(self, final_weights: Dict[Symbol, float]) -> None:
        """
        Liquidate positions not in the target set, then buy target weights.

        Parameters
        ----------
        final_weights : dict[Symbol, float]
            Target weights keyed by symbol.
        """
        for kv in self.portfolio:
            sym = kv.key
            holding = kv.value

            if holding.invested and final_weights.get(sym, 0.0) == 0.0:
                self.liquidate(sym)

        for sym, target_w in final_weights.items():
            if target_w > 0:
                self.set_holdings(sym, target_w * 0.999)

        self.debug("Final monthly holdings:")
        for sym, w in final_weights.items():
            self.debug(f"  {sym.Value}: {w:.2%}")
