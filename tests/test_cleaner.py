# -*- coding: utf-8 -*-
"""Smoke tests for DataCleaner."""

import pytest

from clustering_library import DataCleaner


def test_load_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        DataCleaner("/definitely/not/a/real/path.csv").load_data()


def test_clean_data_applies_all_rules(raw_csv):
    cleaner = DataCleaner(str(raw_csv))
    cleaner.load_data()
    df = cleaner.clean_data()

    assert not df.empty
    # No cancelled invoices survive.
    assert not df["InvoiceNo"].astype(str).str.startswith("C").any()
    # UK-only.
    assert (df["Country"] == "United Kingdom").all()
    # No missing CustomerID.
    assert df["CustomerID"].notna().all()
    # Valid quantity and price only.
    assert (df["Quantity"] > 0).all()
    assert (df["UnitPrice"] > 0).all()
    # TotalPrice is derived correctly.
    assert "TotalPrice" in df.columns
    assert (df["TotalPrice"] == df["Quantity"] * df["UnitPrice"]).all()


def test_create_time_features(raw_csv):
    cleaner = DataCleaner(str(raw_csv))
    cleaner.load_data()
    cleaner.clean_data()
    cleaner.create_time_features()

    df = cleaner.df_uk
    assert df["DayOfWeek"].between(0, 6).all()
    assert df["HourOfDay"].between(0, 23).all()
