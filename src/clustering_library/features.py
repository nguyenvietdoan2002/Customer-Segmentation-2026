# -*- coding: utf-8 -*-
"""Customer-level feature engineering.

20 behavioural candidates across 5 families → QuantileTransformer(normal)
→ greedy correlation filter in model space (|r| > 0.85) → StandardScaler
→ ±3σ clip → equal-total-weight feature families.

The post-transform filter is deliberate: monotone quantile mapping preserves
ranks but can turn weak raw-Pearson relationships into near-duplicate model
coordinates. Each retained feature is subsequently multiplied by
``1 / sqrt(n_family_features)`` so every behavioural family has equal total
squared weight in Euclidean distance.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

logger = logging.getLogger(__name__)


def _gini(values: np.ndarray) -> float:
    """Gini coefficient for a 1-D non-negative array; 0 for a single value."""
    vals = np.sort(values)
    n = len(vals)
    if n <= 1 or vals.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * vals).sum() - (n + 1) * vals.sum()) / (n * vals.sum()))


class FeatureEngineer:
    """Behavioural customer feature engineering for K-Means segmentation.

    Workflow::

        fe = FeatureEngineer(
            data_path="data/processed/cleaned_uk_data.csv",
            raw_data_path="data/raw/online_retail.csv",
        )
        fe.load_data()
        fe.create_customer_features()   # 20 candidates → model-space filter
        fe.transform_features()         # QuantileTransformer(normal)
        fe.scale_features()             # StandardScaler + ±3σ clip
        fe.save_features()              # CSV + model artifacts
    """

    CANDIDATES: List[str] = [
        # Family 1 — Purchase Rhythm
        "Recency",
        "InterPurchaseCV",
        "ActiveSpanDays",
        "MonthlyOrderRate",
        # Family 2 — Spending Shape
        "AOV",
        "SpendGini",
        "PriceCV",
        # Family 3 — Basket Behavior
        "NewSKURate",
        "RepeatSKUFraction",
        "BasketSizeCV",
        # Family 4 — Volume & Bulk
        "BurstIndex",
        "BulkLineRate",
        "QuantityCV",
        "MedianBasketQty",
        # Family 5 — Product Concentration
        "SKU_HHI",
        "MaxSpendShare",
        # Family 6 — Customer Lifecycle
        "ReturnRate",
        "ReturnValueRate",
        "SpendAcceleration",
        # Family 7 — Seasonality
        "QuarterConcentration",
    ]

    BULK_QTY_THRESHOLD: int = 12

    FEATURE_FAMILIES: Dict[str, List[str]] = {
        "Purchase Rhythm": [
            "Recency", "InterPurchaseCV", "ActiveSpanDays", "MonthlyOrderRate",
        ],
        "Spending Shape": [
            "AOV", "SpendGini", "PriceCV", "MaxSpendShare", "SpendAcceleration",
        ],
        "Basket Behaviour": [
            "NewSKURate", "RepeatSKUFraction", "BasketSizeCV",
        ],
        "Volume & Bulk": [
            "BurstIndex", "BulkLineRate", "QuantityCV", "MedianBasketQty", "SKU_HHI",
        ],
        "Customer Lifecycle": [
            "ReturnRate", "ReturnValueRate", "QuarterConcentration",
        ],
    }

    # Protect one pre-specified, interpretable anchor per central business axis.
    # ReturnValueRate is intentionally not protected: in model space it is nearly
    # collinear with ReturnRate and would double-weight cancellation behaviour.
    FORCE_KEEP: List[str] = [
        "Recency", "MonthlyOrderRate", "ReturnRate",
        "QuarterConcentration", "SKU_HHI",
    ]

    def __init__(
        self,
        data_path: str,
        raw_data_path: Optional[str] = None,
        corr_threshold: float = 0.85,
        force_keep: Optional[List[str]] = None,
    ) -> None:
        self.data_path = data_path
        self.raw_data_path = raw_data_path
        self.corr_threshold = corr_threshold
        self.force_keep: List[str] = self.FORCE_KEEP if force_keep is None else force_keep
        self.df: Optional[pd.DataFrame] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.customer_feature_candidates: Optional[pd.DataFrame] = None
        self.customer_features: Optional[pd.DataFrame] = None
        self.customer_features_transformed: Optional[pd.DataFrame] = None
        self.customer_features_scaled_unweighted: Optional[pd.DataFrame] = None
        self.customer_features_scaled: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None
        self.qt: Optional[QuantileTransformer] = None
        self.feature_customer: List[str] = []
        self.family_weights: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"])
        logger.info("Cleaned dataset shape: %s", self.df.shape)

        if self.raw_data_path and os.path.exists(self.raw_data_path):
            self.raw_df = pd.read_csv(
                self.raw_data_path, encoding="ISO-8859-1", low_memory=False
            )
            self.raw_df["InvoiceDate"] = pd.to_datetime(self.raw_df["InvoiceDate"])
            logger.info("Raw dataset shape: %s", self.raw_df.shape)
        elif self.raw_data_path:
            logger.warning("raw_data_path not found (%s). ReturnRate will be 0.", self.raw_data_path)
        return self.df

    # ------------------------------------------------------------------ #
    # Feature families                                                     #
    # ------------------------------------------------------------------ #

    def _compute_purchase_rhythm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recency, InterPurchaseCV, ActiveSpanDays, MonthlyOrderRate."""
        snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        date_bounds = df.groupby("CustomerID")["InvoiceDate"].agg(["min", "max"])
        recency = (snapshot - date_bounds["max"]).dt.days.rename("Recency")
        active_span = (date_bounds["max"] - date_bounds["min"]).dt.days.rename("ActiveSpanDays")

        inv_dates = (
            df.groupby(["CustomerID", "InvoiceNo"])["InvoiceDate"]
            .min()
            .reset_index(name="InvoiceDate")
            .sort_values(["CustomerID", "InvoiceDate"])
        )
        inv_dates["gap"] = inv_dates.groupby("CustomerID")["InvoiceDate"].diff().dt.days
        gap_agg = inv_dates.groupby("CustomerID")["gap"].agg(["mean", "std"])
        inter_cv = (gap_agg["std"] / gap_agg["mean"]).fillna(0.0).rename("InterPurchaseCV")

        count_inv = df.groupby("CustomerID")["InvoiceNo"].nunique()
        monthly_rate = (count_inv / (active_span / 30).clip(lower=1)).rename("MonthlyOrderRate")

        return pd.concat([recency, inter_cv, active_span, monthly_rate], axis=1)

    def _compute_spending_shape(self, df: pd.DataFrame) -> pd.DataFrame:
        """AOV, SpendGini, PriceCV."""
        cust_agg = df.groupby("CustomerID").agg(
            total_spend=("TotalPrice", "sum"),
            n_invoices=("InvoiceNo", "nunique"),
        )
        aov = (cust_agg["total_spend"] / cust_agg["n_invoices"]).rename("AOV")

        inv_spend = (
            df.groupby(["CustomerID", "InvoiceNo"])["TotalPrice"]
            .sum()
            .reset_index(name="InvoiceTotalPrice")
        )
        spend_gini = (
            inv_spend.groupby("CustomerID")["InvoiceTotalPrice"]
            .apply(lambda x: _gini(x.values))
            .rename("SpendGini")
        )

        price_agg = df.groupby("CustomerID")["UnitPrice"].agg(["mean", "std"])
        price_cv = (price_agg["std"] / price_agg["mean"]).fillna(0.0).rename("PriceCV")

        return pd.concat([aov, spend_gini, price_cv], axis=1)

    def _compute_basket_behavior(self, df: pd.DataFrame) -> pd.DataFrame:
        """NewSKURate, RepeatSKUFraction, BasketSizeCV."""
        inv_sku = (
            df.groupby(["CustomerID", "InvoiceNo"])
            .agg(date=("InvoiceDate", "min"), skus=("StockCode", lambda x: frozenset(x)))
            .reset_index()
            .sort_values(["CustomerID", "date"])
        )

        def _new_sku_rate(group: pd.DataFrame) -> float:
            seen: set = set()
            rates = []
            for skus in group["skus"]:
                new = skus - seen
                rates.append(len(new) / len(skus))
                seen.update(skus)
            return float(np.mean(rates)) if rates else 1.0

        new_sku_rate = (
            inv_sku.groupby("CustomerID", group_keys=False)
            .apply(_new_sku_rate, include_groups=False)
            .rename("NewSKURate")
        )

        sku_inv_count = df.groupby(["CustomerID", "StockCode"])["InvoiceNo"].nunique()
        repeat_frac = (
            (sku_inv_count > 1).groupby(level="CustomerID").mean().rename("RepeatSKUFraction")
        )

        basket_size = (
            df.groupby(["CustomerID", "InvoiceNo"])["StockCode"]
            .nunique()
            .reset_index(name="n_skus")
        )
        bs_agg = basket_size.groupby("CustomerID")["n_skus"].agg(["mean", "std"])
        basket_cv = (bs_agg["std"] / bs_agg["mean"]).fillna(0.0).rename("BasketSizeCV")

        return pd.concat([new_sku_rate, repeat_frac, basket_cv], axis=1)

    def _compute_volume_bulk(self, df: pd.DataFrame) -> pd.DataFrame:
        """BurstIndex, BulkLineRate, QuantityCV, MedianBasketQty."""
        inv_spend = (
            df.groupby(["CustomerID", "InvoiceNo"])["TotalPrice"]
            .sum()
            .reset_index(name="InvoiceTotalPrice")
        )
        inv_agg = inv_spend.groupby("CustomerID")["InvoiceTotalPrice"].agg(["max", "mean"])
        burst_index = (inv_agg["max"] / inv_agg["mean"]).rename("BurstIndex")

        df_bulk = df.assign(is_bulk=(df["Quantity"] >= self.BULK_QTY_THRESHOLD).astype(float))
        bulk_rate = df_bulk.groupby("CustomerID")["is_bulk"].mean().rename("BulkLineRate")

        qty_agg = df.groupby("CustomerID")["Quantity"].agg(["mean", "std"])
        qty_cv = (qty_agg["std"] / qty_agg["mean"]).fillna(0.0).rename("QuantityCV")

        basket_qty = (
            df.groupby(["CustomerID", "InvoiceNo"])["Quantity"]
            .sum()
            .reset_index(name="basket_qty")
        )
        median_basket = basket_qty.groupby("CustomerID")["basket_qty"].median().rename("MedianBasketQty")

        return pd.concat([burst_index, bulk_rate, qty_cv, median_basket], axis=1)

    def _compute_concentration(self, df: pd.DataFrame) -> pd.DataFrame:
        """SKU_HHI, MaxSpendShare."""
        sku_spend = df.groupby(["CustomerID", "StockCode"])["TotalPrice"].sum()
        cust_total = sku_spend.groupby(level="CustomerID").transform("sum")
        sku_share = sku_spend / cust_total
        sku_hhi = (sku_share ** 2).groupby(level="CustomerID").sum().rename("SKU_HHI")

        inv_spend = (
            df.groupby(["CustomerID", "InvoiceNo"])["TotalPrice"]
            .sum()
            .reset_index(name="inv_total")
        )
        inv_spend["cust_total"] = inv_spend.groupby("CustomerID")["inv_total"].transform("sum")
        max_spend_share = (
            (inv_spend["inv_total"] / inv_spend["cust_total"])
            .groupby(inv_spend["CustomerID"])
            .max()
            .rename("MaxSpendShare")
        )

        return pd.concat([sku_hhi, max_spend_share], axis=1)

    def _compute_lifecycle(
        self, df: pd.DataFrame, customer_ids: pd.Index
    ) -> pd.DataFrame:
        """ReturnRate (count) + ReturnValueRate (value) from raw cancelled invoices.

        Both require raw_data_path; customers absent from raw data or with no
        cancellations get 0.0.
        """
        if self.raw_df is not None:
            raw_uk = self.raw_df[
                (self.raw_df["Country"] == "United Kingdom")
                & (self.raw_df["CustomerID"].notna())
            ].copy()
            raw_uk["CustomerID"] = raw_uk["CustomerID"].astype(float).astype(int)
            raw_uk["InvoiceNo"] = raw_uk["InvoiceNo"].astype(str)
            raw_uk["is_cancelled"] = raw_uk["InvoiceNo"].str.startswith("C")
            raw_uk["AbsValue"] = (raw_uk["Quantity"] * raw_uk["UnitPrice"]).abs()

            inv_flags = (
                raw_uk.groupby(["CustomerID", "InvoiceNo"])["is_cancelled"]
                .first()
                .reset_index()
            )
            agg_count = inv_flags.groupby("CustomerID")["is_cancelled"].agg(
                n_cancelled="sum", n_total="count"
            )
            return_rate = (agg_count["n_cancelled"] / agg_count["n_total"]).rename("ReturnRate")
            return_rate = return_rate.reindex(customer_ids).fillna(0.0)

            cust_cancel_val = (
                raw_uk[raw_uk["is_cancelled"]]
                .groupby("CustomerID")["AbsValue"]
                .sum()
            )
            cust_total_val = raw_uk.groupby("CustomerID")["AbsValue"].sum()
            return_value_rate = (cust_cancel_val / cust_total_val).rename("ReturnValueRate")
            return_value_rate = return_value_rate.reindex(customer_ids).fillna(0.0)
        else:
            logger.warning("raw_data_path not set — ReturnRate and ReturnValueRate = 0.")
            return_rate = pd.Series(0.0, index=customer_ids, name="ReturnRate")
            return_value_rate = pd.Series(0.0, index=customer_ids, name="ReturnValueRate")

        return pd.concat([return_rate, return_value_rate], axis=1)

    def _compute_spend_acceleration(
        self, df: pd.DataFrame, customer_ids: pd.Index
    ) -> pd.Series:
        """Spending trajectory: (last-3-month avg) / (first-3-month avg) - 1, clipped to [-3, 3].

        Customers with fewer than 3 distinct months of history get 0.0.
        """
        monthly = (
            df.assign(YearMonth=df["InvoiceDate"].dt.to_period("M"))
            .groupby(["CustomerID", "YearMonth"])["TotalPrice"]
            .sum()
            .reset_index(name="MonthlySpend")
        )

        def _accel(group: pd.DataFrame) -> float:
            spend = group.sort_values("YearMonth")["MonthlySpend"].values
            if len(spend) < 3:
                return 0.0
            first_avg = spend[:3].mean()
            last_avg = spend[-3:].mean()
            raw = (last_avg - first_avg) / (abs(first_avg) + 1e-9)
            return float(np.clip(raw, -3.0, 3.0))

        accel = (
            monthly.groupby("CustomerID")
            .apply(_accel, include_groups=False)
            .rename("SpendAcceleration")
        )
        return accel.reindex(customer_ids).fillna(0.0)

    def _compute_quarter_concentration(
        self, df: pd.DataFrame, customer_ids: pd.Index
    ) -> pd.Series:
        """Max quarter spend fraction — captures seasonal concentration.

        Values in [0.25, 1.0]: 0.25 = uniform across quarters, 1.0 = all spend
        in one quarter. Customers absent from df get 0.25 (no concentration).
        """
        q_df = df.copy()
        q_df["Quarter"] = q_df["InvoiceDate"].dt.quarter

        cust_total = q_df.groupby("CustomerID")["TotalPrice"].sum()
        cust_qmax = (
            q_df.groupby(["CustomerID", "Quarter"])["TotalPrice"]
            .sum()
            .groupby("CustomerID")
            .max()
        )

        conc = (cust_qmax / cust_total.clip(lower=1e-9)).rename("QuarterConcentration")
        conc = conc.clip(0.25, 1.0)
        return conc.reindex(customer_ids).fillna(0.25)

    # ------------------------------------------------------------------ #
    # Correlation filter                                                   #
    # ------------------------------------------------------------------ #

    def _filter_by_correlation(self, df: pd.DataFrame) -> List[str]:
        """Greedy pairwise correlation filter: drop the higher mean-|r| feature
        from each pair exceeding the threshold. Features in FORCE_KEEP are never
        eligible for removal.
        """
        cols = list(df.columns)
        protected = set(self.force_keep) & set(cols)

        zero_var = [c for c in cols if df[c].std() == 0 and c not in protected]
        if zero_var:
            logger.info("Dropped %d constant feature(s): %s", len(zero_var), zero_var)
            cols = [c for c in cols if c not in zero_var]

        if protected:
            logger.info("Force-keeping: %s", sorted(protected))

        logger.info("Model-space correlation filter — starting with %d candidates:", len(cols))
        while True:
            corr = df[cols].corr().abs()
            arr = corr.to_numpy().copy()
            np.fill_diagonal(arr, 0.0)
            arr = np.nan_to_num(arr, nan=0.0)

            violating_mask = (arr > self.corr_threshold).any(axis=1)
            droppable_violating = [
                c for c, v in zip(cols, violating_mask) if v and c not in protected
            ]
            if not droppable_violating:
                if violating_mask.any():
                    logger.info("Note: remaining violations are between protected features — stopping.")
                break

            mean_corr = pd.Series(arr.mean(axis=0), index=cols)
            drop = mean_corr[droppable_violating].idxmax()
            drop_i = cols.index(drop)
            partner = cols[int(arr[drop_i].argmax())]
            logger.info(
                "Dropped '%s' (|r|=%.3f with '%s', mean|r|=%.3f)",
                drop, arr[drop_i, cols.index(partner)], partner, mean_corr[drop],
            )
            cols.remove(drop)

        logger.info("Kept %d features after filtering.", len(cols))
        return cols

    # ------------------------------------------------------------------ #
    # Main pipeline                                                        #
    # ------------------------------------------------------------------ #

    def create_customer_features(self) -> pd.DataFrame:
        """Compute candidates and select non-redundant model-space features."""
        df = self.df

        logger.info("[1/8] Purchase Rhythm...")
        rhythm = self._compute_purchase_rhythm(df)
        logger.info("[2/8] Spending Shape...")
        spending = self._compute_spending_shape(df)
        logger.info("[3/8] Basket Behavior (sequential SKU tracking — may take ~30 s)...")
        basket = self._compute_basket_behavior(df)
        logger.info("[4/8] Volume & Bulk...")
        volume = self._compute_volume_bulk(df)
        logger.info("[5/8] Product Concentration...")
        concentration = self._compute_concentration(df)

        base = (
            rhythm.join(spending, how="outer")
            .join(basket, how="outer")
            .join(volume, how="outer")
            .join(concentration, how="outer")
        )

        logger.info("[6/8] Customer Lifecycle (ReturnRate + ReturnValueRate)...")
        lifecycle = self._compute_lifecycle(df, base.index)
        logger.info("[7/8] Spend Acceleration...")
        accel = self._compute_spend_acceleration(df, base.index)
        logger.info("[8/8] Quarter Concentration...")
        qconc = self._compute_quarter_concentration(df, base.index)

        all_candidates = (
            base.join(lifecycle, how="outer")
            .join(accel, how="outer")
            .join(qconc, how="outer")
        )[self.CANDIDATES]

        n_missing = all_candidates.isnull().sum().sum()
        if n_missing > 0:
            logger.warning("%d NaN values — filling with column medians.", n_missing)
            all_candidates = all_candidates.fillna(all_candidates.median())

        logger.info("Computed %d candidates for %d customers", len(self.CANDIDATES), len(all_candidates))
        self.customer_feature_candidates = all_candidates.copy()

        provisional_qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(500, len(all_candidates)),
            random_state=42,
        )
        provisional = provisional_qt.fit_transform(all_candidates)
        provisional = StandardScaler().fit_transform(provisional)
        provisional_df = pd.DataFrame(
            np.clip(provisional, -3.0, 3.0),
            columns=all_candidates.columns,
            index=all_candidates.index,
        )

        self.feature_customer = self._filter_by_correlation(provisional_df)
        self.customer_features = all_candidates[self.feature_customer].reset_index()
        logger.info("Final feature set (%d features):", len(self.feature_customer))
        for f in self.feature_customer:
            logger.info("    %s", f)
        return self.customer_features

    def transform_features(self) -> pd.DataFrame:
        """Apply QuantileTransformer to map features to a normal distribution."""
        indexed = self.customer_features.set_index("CustomerID")

        self.qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=500,
            random_state=42,
        )
        transformed = self.qt.fit_transform(indexed.values)

        self.customer_features_transformed = pd.DataFrame(
            transformed,
            columns=self.feature_customer,
            index=indexed.index,
        )
        logger.info("QuantileTransformer(normal) applied.")
        return self.customer_features_transformed

    def _build_family_weights(self) -> Dict[str, float]:
        """Give every retained family equal total squared Euclidean weight."""
        feature_to_family = {
            feature: family
            for family, features in self.FEATURE_FAMILIES.items()
            for feature in features
        }
        counts = pd.Series(
            [feature_to_family[feature] for feature in self.feature_customer]
        ).value_counts()
        return {
            feature: float(1.0 / np.sqrt(counts[feature_to_family[feature]]))
            for feature in self.feature_customer
        }

    def scale_features(self) -> pd.DataFrame:
        """Standardise, clip, then balance total weight across feature families."""
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(self.customer_features_transformed)

        scaled_df = pd.DataFrame(
            scaled,
            columns=self.feature_customer,
            index=self.customer_features_transformed.index,
        )
        clipped = scaled_df.clip(lower=-3.0, upper=3.0)
        self.customer_features_scaled_unweighted = clipped
        self.family_weights = self._build_family_weights()
        weighted = clipped.mul(pd.Series(self.family_weights), axis=1)
        self.customer_features_scaled = weighted
        logger.info("Feature scaling complete (±3σ clip + family balancing applied).")
        return weighted

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_pipeline(self, output_dir: str = "../models") -> None:
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Run scale_features() first.")
        if self.qt is None:
            raise RuntimeError("QuantileTransformer not fitted. Run transform_features() first.")

        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        joblib.dump(self.qt,     f"{output_dir}/quantile_transformer.pkl")

        metadata: Dict = {
            "bulk_qty_threshold": self.BULK_QTY_THRESHOLD,
            "feature_order":      self.feature_customer,
            "n_candidates":       len(self.CANDIDATES),
            "corr_threshold":     self.corr_threshold,
            "correlation_space": "quantile_transformed_scaled_clipped",
            "post_scale_clip":    3.0,
            "family_weights":     self.family_weights,
            "feature_families":   self.FEATURE_FAMILIES,
        }
        with open(f"{output_dir}/pipeline_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Pipeline artifacts saved to: %s/", output_dir)

    def save_features(self, output_dir: str = "../data/processed") -> None:
        os.makedirs(output_dir, exist_ok=True)

        self.customer_feature_candidates.to_csv(
            f"{output_dir}/customer_feature_candidates.csv"
        )
        self.customer_features.set_index("CustomerID").to_csv(
            f"{output_dir}/customer_features.csv"
        )
        self.customer_features_transformed.to_csv(
            f"{output_dir}/customer_features_transformed.csv"
        )
        self.customer_features_scaled_unweighted.to_csv(
            f"{output_dir}/customer_features_scaled_unweighted.csv"
        )
        self.customer_features_scaled.to_csv(
            f"{output_dir}/customer_features_scaled.csv"
        )
        logger.info("All features saved to: %s", output_dir)

        models_dir = os.path.join(os.path.dirname(output_dir), "models")
        self.save_pipeline(models_dir)
