# -*- coding: utf-8 -*-
"""RFM computation utility."""

import pandas as pd


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction data to customer-level R/F/M metrics.

    Args:
        df: Cleaned transactions with InvoiceDate, CustomerID, InvoiceNo, TotalPrice.

    Returns:
        Customer-level DataFrame: Recency (days), Frequency (invoices), Monetary.
    """
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    return (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )
