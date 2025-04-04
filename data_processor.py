import pandas as pd
import numpy as np
from datetime import timedelta
import os
from typing import List, Optional, Tuple
import config  # Import the configuration file


def calculate_driver_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates additional metrics useful for driver analysis based on transaction data.

    Requires 'CustomerID', 'InvoiceDate', and 'StockCode' columns in the input DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing customer transaction data.
                           Must include 'CustomerID', 'InvoiceDate', 'StockCode'.

    Returns:
        pd.DataFrame: DataFrame indexed by CustomerID, containing:
            - 'time_to_second_purchase' (float): Days between first and second purchase.
                                                 Filled with -1 if only one purchase.
            - 'category_count' (int): Number of unique StockCodes purchased by the customer.
                                      Filled with 0 if no purchases or StockCode missing.
    """
    if df.empty or "CustomerID" not in df.columns or "InvoiceDate" not in df.columns:
        print(
            "Warning: Input DataFrame is empty or missing required columns for driver metrics."
        )
        return pd.DataFrame(
            columns=["CustomerID", "time_to_second_purchase", "category_count"]
        ).set_index("CustomerID")

    # Ensure CustomerID is string type for consistent joining
    df = df.copy()  # Avoid modifying original DataFrame
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])  # Ensure datetime

    # Create a DataFrame with all unique customer IDs
    all_customers = pd.DataFrame({"CustomerID": df["CustomerID"].unique()})

    # --- Time to Second Purchase ---
    # Sort data by customer and date
    df_sorted = df.sort_values(["CustomerID", "InvoiceDate"])

    # Get first purchase date for each customer
    first_purchase = (
        df_sorted.groupby("CustomerID")["InvoiceDate"].first().reset_index()
    )
    first_purchase.rename(columns={"InvoiceDate": "first_purchase"}, inplace=True)

    # For each customer, get all their purchase dates
    purchase_dates = (
        df_sorted.groupby("CustomerID")["InvoiceDate"].apply(list).reset_index()
    )

    # Function to get second purchase date if it exists
    def get_second_purchase(dates: List[pd.Timestamp]) -> Optional[pd.Timestamp]:
        """Returns the second timestamp in a list, or None if fewer than two."""
        return dates[1] if len(dates) > 1 else None

    # Apply function to get second purchase date
    purchase_dates["second_purchase"] = purchase_dates["InvoiceDate"].apply(
        get_second_purchase
    )
    purchase_dates = purchase_dates[["CustomerID", "second_purchase"]]

    # Merge first and second purchase dates
    time_metrics = all_customers.merge(first_purchase, on="CustomerID", how="left")
    time_metrics = time_metrics.merge(purchase_dates, on="CustomerID", how="left")

    # Calculate time to second purchase (in days)
    time_metrics["time_to_second_purchase"] = np.nan
    mask = (
        ~time_metrics["second_purchase"].isna() & ~time_metrics["first_purchase"].isna()
    )
    time_metrics.loc[mask, "time_to_second_purchase"] = (
        time_metrics.loc[mask, "second_purchase"]
        - time_metrics.loc[mask, "first_purchase"]
    ).dt.days

    # --- Category Count ---
    # Count unique StockCodes per customer
    if "StockCode" in df.columns:
        category_counts = df.groupby("CustomerID")["StockCode"].nunique().reset_index()
        category_counts.rename(columns={"StockCode": "category_count"}, inplace=True)
    else:
        # Fallback if StockCode is not available
        print("Warning: 'StockCode' column not found. Category count will be 0.")
        category_counts = pd.DataFrame(
            {"CustomerID": df["CustomerID"].unique(), "category_count": 0}
        )

    # Merge all metrics
    driver_metrics = all_customers.merge(
        time_metrics[["CustomerID", "time_to_second_purchase"]],
        on="CustomerID",
        how="left",
    )
    driver_metrics = driver_metrics.merge(category_counts, on="CustomerID", how="left")

    # Fill missing values and set index
    driver_metrics["time_to_second_purchase"] = (
        driver_metrics["time_to_second_purchase"].fillna(-1).astype(float)
    )  # Use float for consistency, -1 indicates no second purchase
    driver_metrics["category_count"] = (
        driver_metrics["category_count"].fillna(0).astype(int)
    )
    driver_metrics.set_index("CustomerID", inplace=True)  # Set index after calculations

    print(
        f"Calculated driver metrics (time_to_second, category_count) for {len(driver_metrics)} customers."
    )
    return driver_metrics


# Use config for the default file path
def load_and_clean_data(
    file_path: str = config.DATA_FILE_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Loads the Online Retail dataset, performs initial cleaning, and calculates driver metrics.

    Args:
        file_path (str): The path to the Excel data file. Defaults to the path specified in config.py.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The cleaned transaction DataFrame suitable for RFM/CLV analysis.
            - pd.DataFrame: DataFrame with driver metrics per customer (indexed by CustomerID).
            - pd.Timestamp: The snapshot date (last invoice date + 1 day).

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")

    # Load the dataset
    data = pd.read_excel(file_path)
    print(f"Original dataset shape: {data.shape}")
    print(f"Missing values before cleaning:\n{data.isnull().sum()}")

    # --- Basic data cleaning ---
    # Convert InvoiceDate to datetime
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

    # Create TotalPrice column
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

    # 1. Remove rows with missing CustomerID
    data.dropna(subset=["CustomerID"], inplace=True)
    print(f"Shape after removing missing CustomerID: {data.shape}")

    # 2. Convert CustomerID to integer then string
    # Ensure CustomerID is treated as a discrete identifier
    data["CustomerID"] = data["CustomerID"].astype(int).astype(str)

    # 3. Filter out returns/cancellations (negative quantity)
    # Also filter out zero quantity if it represents non-sales
    data_clean = data[data["Quantity"] > 0].copy()
    print(f"Shape after removing non-positive quantity rows: {data_clean.shape}")

    # 4. Filter out zero unit prices (potentially bad data or free items)
    data_clean = data_clean[data_clean["UnitPrice"] > 0].copy()
    print(f"Shape after removing zero unit price rows: {data_clean.shape}")

    # 5. Check and optionally remove duplicates
    initial_rows = data_clean.shape[0]
    data_clean.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - data_clean.shape[0]} duplicate rows.")
    print(f"Clean dataset shape: {data_clean.shape}")

    # 6. Define snapshot date (for recency calculation)
    snapshot_date: pd.Timestamp = data_clean["InvoiceDate"].max() + timedelta(days=1)
    print(f"Analysis snapshot date: {snapshot_date}")

    # --- Calculate Driver Metrics ---
    print("\nCalculating driver metrics...")
    # Pass only necessary columns to avoid unnecessary data transfer
    # Ensure StockCode exists before passing
    required_driver_cols = ["CustomerID", "InvoiceDate"]
    if "StockCode" in data_clean.columns:
        required_driver_cols.append("StockCode")
    driver_metrics_df = calculate_driver_metrics(
        data_clean[required_driver_cols].copy()
    )

    print(f"\nFinal cleaned data shape for RFM/CLV: {data_clean.shape}")
    print(f"Missing values after cleaning:\n{data_clean.isnull().sum()}")

    # Return the main cleaned data AND the driver metrics
    return data_clean, driver_metrics_df, snapshot_date


if __name__ == "__main__":
    try:
        print("Running data processor...")
        # Use config path when running directly
        cleaned_data, driver_metrics, analysis_date = load_and_clean_data(
            config.DATA_FILE_PATH
        )
        print("\nData loading, cleaning, and driver metric calculation finished.")
        print(f"Snapshot date: {analysis_date}")
        print("\nCleaned Data Head:")
        print(cleaned_data.head())
        print("\nDriver Metrics Head:")
        print(driver_metrics.head())
        print("\nDriver Metrics Info:")
        driver_metrics.info()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
