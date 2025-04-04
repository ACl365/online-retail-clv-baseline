import pandas as pd
from data_processor import load_and_clean_data  # Assuming this returns correct types
from typing import Tuple  # Import Tuple if load_and_clean_data returns it
import config  # Import the configuration file


def calculate_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Calculates Recency, Frequency, Monetary (RFM) metrics, scores (1-5 quintiles, based on config),
    and assigns customer segments based on these scores.

    Args:
        df (pd.DataFrame): The cleaned transaction DataFrame. Must contain
                           'CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalPrice'.
        snapshot_date (pd.Timestamp): The reference date for recency calculation
                                      (typically the day after the last transaction date).

    Returns:
        pd.DataFrame: DataFrame indexed by CustomerID, containing RFM metrics,
                      scores (R_Score, F_Score, M_Score), combined RFM_Score (string),
                      and customer Segment (string).

    Raises:
        ValueError: If the input DataFrame is missing required columns.
    """
    required_cols = ["CustomerID", "InvoiceDate", "InvoiceNo", "TotalPrice"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain {required_cols} columns.")

    # Ensure CustomerID is string type
    df = df.copy()
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Aggregate data per customer
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),  # Number of unique invoices
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )

    # Ensure Monetary is positive for scoring robustness (prevents issues with qcut)
    rfm = rfm[rfm["Monetary"] > 0].copy()
    if rfm.empty:
        print(
            "Warning: No customers with positive Monetary value found after aggregation."
        )
        # Return an empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "CustomerID",
                "Recency",
                "Frequency",
                "Monetary",
                "R_Score",
                "F_Score",
                "M_Score",
                "RFM_Score",
                "Segment",
            ]
        ).set_index("CustomerID")

    print(f"Calculated base RFM metrics for {len(rfm)} customers.")
    print("RFM Metrics Summary:")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe())

    # Create RFM quintile scores (1-5, with 5 being best) using config.RFM_QUANTILES
    q = config.RFM_QUANTILES
    r_labels = list(
        range(q, 0, -1)
    )  # Labels for Recency (higher score is better) -> [5, 4, 3, 2, 1] for q=5
    fm_labels = list(
        range(1, q + 1)
    )  # Labels for Frequency/Monetary (higher score is better) -> [1, 2, 3, 4, 5] for q=5

    # For Recency, lower values are better (more recent) -> higher score
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"], q=q, labels=r_labels, duplicates="drop"
    ).astype(int)
    # For Frequency and Monetary, higher values are better -> higher score
    # Use rank(method='first') to handle ties consistently before qcut
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), q=q, labels=fm_labels, duplicates="drop"
    ).astype(int)
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"), q=q, labels=fm_labels, duplicates="drop"
    ).astype(int)

    # Calculate RFM Combined Score (as string for easy lookup if needed)
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    # Define customer segments based on RFM score patterns
    def assign_segment(row: pd.Series) -> str:
        """Assigns a customer segment based on R, F, M scores."""
        # Use q for dynamic thresholding if needed, or keep fixed logic for now
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        high_score = q - 1  # e.g., 4 for q=5
        mid_score = q // 2 + 1  # e.g., 3 for q=5
        low_score = 2  # e.g., 2 for q=5

        # Segmentation logic (adjust thresholds based on q if desired, or keep fixed)
        if r >= high_score and f >= high_score and m >= high_score:  # Top scores
            return "Champions"
        elif (
            r >= mid_score and f >= mid_score and m >= mid_score
        ):  # Generally good scores
            return "Loyal Customers"
        elif r >= high_score and f >= 1 and f <= low_score:  # Recent, but low frequency
            return "New Customers"
        elif (
            r >= mid_score and r <= high_score and f >= mid_score and f <= high_score
        ):  # Good frequency, recent enough
            return "Potential Loyalists"
        elif (
            r >= low_score and r <= mid_score and f >= low_score and m >= low_score
        ):  # Was good, but recency slipping
            return "At Risk"
        elif (
            r <= low_score and f >= high_score and m >= high_score
        ):  # High value/freq, but inactive
            return "Can't Lose Them"
        elif r <= low_score and f <= low_score:  # Low scores across R and F
            return "Hibernating/Lost"
        else:
            # Catch-all for other combinations
            return "Needs Attention"

    rfm["Segment"] = rfm.apply(assign_segment, axis=1)

    print("\nCustomer segmentation complete.")
    print("Segment Distribution:")
    print(
        rfm["Segment"].value_counts(normalise=True).map("{:.1%}".format)
    )  # Format as percentage

    # Set CustomerID as index for easier merging later
    rfm.set_index("CustomerID", inplace=True)

    return rfm[
        [
            "Recency",
            "Frequency",
            "Monetary",
            "R_Score",
            "F_Score",
            "M_Score",
            "RFM_Score",
            "Segment",
        ]
    ]


if __name__ == "__main__":
    try:
        print("--- Starting RFM Analysis ---")
        # 1. Load and clean data (uses default path from config via data_processor)
        print("Loading and cleaning data...")
        cleaned_data, _, snapshot_date = load_and_clean_data()
        print("Data loaded and cleaned.")

        # 2. Calculate RFM
        print("\nCalculating RFM metrics and segments...")
        rfm_results = calculate_rfm(cleaned_data, snapshot_date)  # Index is CustomerID
        print("\nRFM analysis finished successfully.")

        # 3. Display results
        print("\nRFM Results Head:")
        print(rfm_results.head())

        print("\nSegment Characteristics (Mean Values):")
        print(
            rfm_results.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
            .mean()
            .round(2)
        )

        print("\nSegment Counts:")
        print(rfm_results["Segment"].value_counts())

        print("\n--- RFM Analysis Complete ---")

    except FileNotFoundError as e:
        # Update error message to refer to config path
        print(
            f"Error: {e}. Make sure '{config.DATA_FILE_NAME}' is in the directory '{config.BASE_DIR}'."
        )
    except ValueError as e:
        print(f"Data Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during RFM analysis: {e}")
        import traceback

        traceback.print_exc()
