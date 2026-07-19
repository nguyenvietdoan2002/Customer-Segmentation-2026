# -*- coding: utf-8 -*-
"""EDA visualizations for customer segmentation."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visual_style import (
    BLUE as ACCENT_BLUE,
    CLUSTER_COLORS,
    SEQUENTIAL_CMAP as CMAP_SEQ,
)


class DataVisualizer:
    """Visualization methods for EDA in the customer segmentation pipeline."""

    def __init__(self) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette(CLUSTER_COLORS)

    def plot_revenue_over_time(self, df: pd.DataFrame) -> None:
        """
        Plot daily and monthly revenue patterns.

        Args:
            df: Dataframe with InvoiceDate and TotalPrice columns
        """
        plt.figure(figsize=(12, 5))
        daily_revenue = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()
        daily_revenue.plot(color=ACCENT_BLUE)
        plt.title("Daily Revenue")
        plt.xlabel("Date")
        plt.ylabel("Revenue (GBP)")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 5))
        monthly_revenue = df.groupby(pd.Grouper(key="InvoiceDate", freq="ME"))["TotalPrice"].sum()
        monthly_revenue.plot(kind="bar", color=ACCENT_BLUE)
        plt.title("Monthly Revenue")
        plt.xlabel("Month")
        plt.ylabel("Revenue (GBP)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_time_patterns(self, df: pd.DataFrame) -> None:
        """
        Plot purchase patterns by day and hour.

        Args:
            df: Dataframe with time features
        """
        plt.figure(figsize=(12, 5))
        day_hour_counts = df.groupby(["DayOfWeek", "HourOfDay"]).size().unstack(fill_value=0)
        sns.heatmap(day_hour_counts, cmap=CMAP_SEQ)
        plt.title("Purchase Activity by Day and Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week (0=Monday, 6=Sunday)")
        plt.tight_layout()
        plt.show()

    def plot_product_analysis(self, df: pd.DataFrame, top_n: int = 10) -> None:
        """
        Plot top products by quantity and revenue.

        Args:
            df: Transaction dataframe
            top_n: Number of top products to show
        """
        plt.figure(figsize=(12, 5))
        top_products = (
            df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_products.values, y=top_products.index, color=ACCENT_BLUE)
        plt.title(f"Top {top_n} Products by Units Sold")
        plt.xlabel("Units Sold")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 5))
        top_revenue_products = (
            df.groupby("Description")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_revenue_products.values, y=top_revenue_products.index, color=ACCENT_BLUE)
        plt.title(f"Top {top_n} Products by Revenue")
        plt.xlabel("Revenue (GBP)")
        plt.tight_layout()
        plt.show()

    def plot_customer_distribution(self, df: pd.DataFrame) -> None:
        """
        Plot customer behavior distributions.

        Args:
            df: Transaction dataframe
        """
        plt.figure(figsize=(10, 5))
        transactions_per_customer = df.groupby("CustomerID")["InvoiceNo"].nunique()
        sns.histplot(transactions_per_customer, bins=30, kde=True, color=ACCENT_BLUE)
        plt.title("Distribution of Transactions per Customer")
        plt.xlabel("Number of Transactions")
        plt.ylabel("Number of Customers")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        spend_per_customer = df.groupby("CustomerID")["TotalPrice"].sum()
        spend_filter = spend_per_customer < spend_per_customer.quantile(0.99)
        sns.histplot(spend_per_customer[spend_filter], bins=30, kde=True, color=ACCENT_BLUE)
        plt.title("Distribution of Total Spend per Customer")
        plt.xlabel("Total Spend (GBP)")
        plt.ylabel("Number of Customers")
        plt.tight_layout()
        plt.show()

    def plot_rfm_analysis(self, rfm_data: pd.DataFrame) -> None:
        """
        Plot RFM analysis visualizations.

        Args:
            rfm_data: RFM dataframe
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        sns.histplot(rfm_data["Recency"], bins=30, kde=True, ax=axes[0], color=ACCENT_BLUE)
        axes[0].set_title("Recency Distribution (Days Since Last Purchase)")
        axes[0].set_xlabel("Days")

        sns.histplot(rfm_data["Frequency"], bins=30, kde=True, ax=axes[1], color=ACCENT_BLUE)
        axes[1].set_title("Frequency Distribution (Number of Transactions)")
        axes[1].set_xlabel("Number of Transactions")

        monetary_filter = rfm_data["Monetary"] < rfm_data["Monetary"].quantile(0.99)
        sns.histplot(rfm_data.loc[monetary_filter, "Monetary"], bins=30, kde=True, ax=axes[2], color=ACCENT_BLUE)
        axes[2].set_title("Monetary Distribution (Total Spend)")
        axes[2].set_xlabel("Total Spend (GBP)")

        plt.tight_layout()
        plt.show()
