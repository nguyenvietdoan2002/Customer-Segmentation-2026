# -*- coding: utf-8 -*-
"""Data loading and cleaning for retail transaction datasets."""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    A class for cleaning and preprocessing retail transaction data.

    This class handles data loading, cleaning operations, and basic exploratory
    data analysis for online retail datasets.
    """

    def __init__(self, data_path: str) -> None:
        """
        Initialize the DataCleaner with data path.

        Args:
            data_path: Path to the raw data file
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.df_uk: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """
        Load and display basic information about the dataset.

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        dtype = dict(
            InvoiceNo=np.object_,
            StockCode=np.object_,
            Description=np.object_,
            Quantity=np.int64,
            UnitPrice=np.float64,
            CustomerID=np.float64,  # keep as float so NaN is preserved
            Country=np.object_,
        )

        self.df = pd.read_csv(
            self.data_path,
            encoding="ISO-8859-1",
            parse_dates=["InvoiceDate"],
            dtype=dtype,
        )

        # Convert CustomerID: preserve NaN as NaN (not "nan" string)
        self.df["CustomerID"] = self.df["CustomerID"].apply(
            lambda x: str(int(x)).zfill(6) if pd.notna(x) else np.nan
        )

        logger.info("Dataset shape: %s", self.df.shape)
        logger.info("Number of records: %s", f"{len(self.df):,}")

        return self.df

    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset by removing invalid records and focusing on UK customers.

        Returns:
            pd.DataFrame: Cleaned UK dataset
        """
        # Add TotalPrice column
        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]

        # Remove cancelled invoices (InvoiceNo starts with 'C')
        self.df = self.df[~self.df["InvoiceNo"].astype(str).str.startswith("C")]

        # Focus on UK customers only
        self.df_uk = self.df[self.df["Country"] == "United Kingdom"].copy()

        # Drop records with missing CustomerID (true NaN, not string)
        self.df_uk = self.df_uk.dropna(subset=["CustomerID"])

        # Remove records with invalid quantity or price
        self.df_uk = self.df_uk[
            (self.df_uk["Quantity"] > 0) & (self.df_uk["UnitPrice"] > 0)
        ]

        return self.df_uk

    def create_time_features(self) -> None:
        """Create time-based features for analysis."""
        self.df_uk["DayOfWeek"] = self.df_uk["InvoiceDate"].dt.dayofweek
        self.df_uk["HourOfDay"] = self.df_uk["InvoiceDate"].dt.hour

    def save_cleaned_data(self, output_dir: str = "../data/processed") -> None:
        """
        Save cleaned data to specified directory.

        Args:
            output_dir: Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        self.df_uk.to_csv(f"{output_dir}/cleaned_uk_data.csv", index=False)
        logger.info("Cleaned data saved: %s/cleaned_uk_data.csv", output_dir)
