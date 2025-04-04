import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from data_processor import (
    load_and_clean_data,
)  # Assuming this returns the correct types now
from rfm_analyzer import calculate_rfm  # Assuming this returns the correct types now
import warnings
from typing import Optional
import config  # Import the configuration file

warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # Lifetimes can generate FutureWarnings


def prepare_lifetimes_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the summary DataFrame required by the lifetimes library.

    Args:
        df (pd.DataFrame): The cleaned transaction DataFrame. Must contain
                           'CustomerID', 'InvoiceDate', and 'TotalPrice' columns.

    Returns:
        pd.DataFrame: Summary DataFrame indexed by CustomerID, with columns:
                      'frequency', 'recency', 'T', and 'monetary_value'.

    Raises:
        ValueError: If the input DataFrame is missing required columns.
    """
    required_cols = ["CustomerID", "InvoiceDate", "TotalPrice"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain {required_cols} columns.")

    # Ensure CustomerID is string type for consistency
    df = df.copy()
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Create the summary data using lifetimes utility function
    # Use the maximum date from the data as the observation period end
    observation_period_end = df["InvoiceDate"].max()
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="TotalPrice",
        observation_period_end=observation_period_end,
        freq=config.LIFETIMES_FREQ,  # Use frequency unit from config
    )
    print(f"Prepared lifetimes summary data for {len(summary)} customers.")
    print("Lifetimes Summary Head:")
    print(summary.head())
    return summary


def fit_bgnbd_model(summary_df: pd.DataFrame) -> BetaGeoFitter:
    """
    Fits the Beta Geometric/Negative Binomial Distribution (BG/NBD) model
    to predict customer purchase frequency and probability of being alive.

    Args:
        summary_df (pd.DataFrame): Lifetimes summary data (output of prepare_lifetimes_summary).
                                   Must contain 'frequency', 'recency', and 'T'.

    Returns:
        BetaGeoFitter: The fitted BG/NBD model object.
    """
    # BG/NBD model requires frequency > 0 for fitting (customers who made repeat purchases)
    summary_filtered = summary_df[summary_df["frequency"] > 0].copy()
    print(
        f"Fitting BG/NBD model on {len(summary_filtered)} customers with frequency > 0."
    )

    # Use penalizer from config
    bgf = BetaGeoFitter(penalizer_coef=config.BGNBD_PENALIZER)
    bgf.fit(
        summary_filtered["frequency"],
        summary_filtered["recency"],
        summary_filtered["T"],
    )

    print("BG/NBD model fitted successfully.")
    print("Model Parameters (r, alpha, a, b):")
    print(bgf.summary)
    return bgf


def fit_gamma_gamma_model(summary_df: pd.DataFrame) -> Optional[GammaGammaFitter]:
    """
    Fits the Gamma-Gamma model to predict the average transaction value for customers.
    Checks model assumptions (correlation between frequency and monetary value).

    Args:
        summary_df (pd.DataFrame): Lifetimes summary data. Must contain 'frequency'
                                   and 'monetary_value'.

    Returns:
        Optional[GammaGammaFitter]: The fitted Gamma-Gamma model object, or None if
                                    fitting conditions aren't met or assumptions are
                                    severely violated (currently proceeds with warning).
    """
    # Gamma-Gamma model requires frequency > 0 and monetary_value > 0
    summary_filtered = summary_df[
        (summary_df["frequency"] > 0) & (summary_df["monetary_value"] > 0)
    ].copy()
    if summary_filtered.empty:
        print(
            "Warning: No customers found with frequency > 0 and monetary_value > 0. Cannot fit Gamma-Gamma model."
        )
        return None
    print(
        f"Fitting Gamma-Gamma model on {len(summary_filtered)} customers with frequency > 0 and monetary_value > 0."
    )

    # Check Gamma-Gamma assumption: correlation between frequency and monetary value should be low
    correlation = summary_filtered[["frequency", "monetary_value"]].corr().iloc[0, 1]
    print(f"Correlation between frequency and monetary value: {correlation:.4f}")
    # Use correlation threshold from config
    if abs(correlation) > config.GAMMA_GAMMA_CORR_THRESHOLD:
        print(
            f"Warning: Correlation ({correlation:.4f}) exceeds threshold ({config.GAMMA_GAMMA_CORR_THRESHOLD})."
        )
        print("Gamma-Gamma model assumptions may be violated. Proceeding with caution.")
        # Consider returning None or raising an error in stricter implementations.

    # Use penalizer from config
    ggf = GammaGammaFitter(penalizer_coef=config.GAMMA_GAMMA_PENALIZER)
    ggf.fit(summary_filtered["frequency"], summary_filtered["monetary_value"])

    print("Gamma-Gamma model fitted successfully.")
    print("Model Parameters (p, q, v):")
    print(ggf.summary)
    return ggf


def predict_clv(
    summary_df: pd.DataFrame,
    bgf_model: BetaGeoFitter,
    ggf_model: Optional[GammaGammaFitter],
    time_months: int = config.CLV_PREDICTION_MONTHS,  # Default from config
    freq: str = config.LIFETIMES_FREQ,  # Default from config
    discount_rate: float = config.MONTHLY_DISCOUNT_RATE,  # Default from config
) -> pd.DataFrame:
    """
    Predicts Customer Lifetime Value (CLV) using fitted BG/NBD and Gamma-Gamma models.

    Args:
        summary_df (pd.DataFrame): Lifetimes summary data (indexed by CustomerID).
        bgf_model (BetaGeoFitter): Fitted BG/NBD model.
        ggf_model (Optional[GammaGammaFitter]): Fitted Gamma-Gamma model. If None, CLV is set to 0.
        time_months (int): Prediction horizon in months. Defaults to value in config.py.
        freq (str): Frequency unit used in summary_data_from_transaction_data. Defaults to value in config.py.
        discount_rate (float): Monthly discount rate for CLV calculation. Defaults to value in config.py.

    Returns:
        pd.DataFrame: Original summary DataFrame with an added 'predicted_clv' column.
                      CLV is 0 if ggf_model is None or for customers where prediction fails.
    """
    summary_with_clv = summary_df.copy()

    if ggf_model is None:
        print("Gamma-Gamma model not available. Setting predicted_clv to 0.")
        summary_with_clv["predicted_clv"] = 0.0
        return summary_with_clv

    # Calculate CLV using the lifetimes library function
    # This function internally handles customers with frequency=0 or monetary_value=0
    # by assigning them a CLV of 0.
    summary_with_clv["predicted_clv"] = ggf_model.customer_lifetime_value(
        bgf_model,  # The model to use to predict the number of future transactions
        summary_with_clv["frequency"],
        summary_with_clv["recency"],
        summary_with_clv["T"],
        summary_with_clv["monetary_value"],
        time=time_months,  # months
        freq=freq,  # frequency expressed in days
        discount_rate=discount_rate,  # monthly discount rate
    )

    # Handle potential negative CLV predictions (can happen with low monetary value/high penalty)
    # and ensure CLV is non-negative. Also fill any NaNs that might arise.
    summary_with_clv["predicted_clv"] = (
        summary_with_clv["predicted_clv"].clip(lower=0).fillna(0)
    )

    print(f"Predicted CLV for {time_months} months.")
    print("CLV Prediction Summary:")
    print(summary_with_clv["predicted_clv"].describe())

    return summary_with_clv


if __name__ == "__main__":
    try:
        print("--- Starting CLV Prediction ---")
        # 1. Load and clean data (using config path via data_processor default)
        print("Loading and cleaning data...")
        cleaned_data, _, snapshot_date = load_and_clean_data()
        print("Data loaded and cleaned.")

        # 2. Prepare Lifetimes summary data
        print("\nPreparing data for Lifetimes models...")
        lifetimes_summary = prepare_lifetimes_summary(
            cleaned_data
        )  # Index is CustomerID

        # 3. Fit BG/NBD model
        print("\nFitting BG/NBD model...")
        bgf = fit_bgnbd_model(lifetimes_summary)

        # 4. Fit Gamma-Gamma model
        print("\nFitting Gamma-Gamma model...")
        ggf = fit_gamma_gamma_model(lifetimes_summary)

        # 5. Predict CLV (using defaults from config via predict_clv function)
        if ggf:  # Proceed only if Gamma-Gamma model was successfully fitted
            print("\nPredicting CLV...")
            clv_predictions_summary = predict_clv(
                lifetimes_summary, bgf, ggf
            )  # Uses defaults from config
            print("\nCLV prediction finished successfully.")

            # 6. Optional: Merge with RFM segments for analysis
            print("\nCalculating RFM segments...")
            # Assuming calculate_rfm returns df indexed by CustomerID
            rfm_results = calculate_rfm(cleaned_data, snapshot_date)

            # Merge CLV predictions with RFM results (both should be indexed by CustomerID)
            final_results = rfm_results.merge(
                clv_predictions_summary[["predicted_clv"]],
                left_index=True,  # Merge on CustomerID index
                right_index=True,
                how="left",
            )
            # Fill CLV for customers who might be in RFM but not CLV (e.g., freq=0)
            final_results["predicted_clv"].fillna(0, inplace=True)

            print("\nFinal Results Head (RFM + CLV):")
            print(final_results.head())

            print(
                f"\nAverage Predicted {config.CLV_PREDICTION_MONTHS}-Month CLV by Segment:"
            )
            print(
                final_results.groupby("Segment")["predicted_clv"]
                .mean()
                .round(2)
                .sort_values(ascending=False)
            )

            print(
                f"\nTotal Predicted {config.CLV_PREDICTION_MONTHS}-Month CLV by Segment:"
            )
            print(
                final_results.groupby("Segment")["predicted_clv"]
                .sum()
                .round(2)
                .sort_values(ascending=False)
            )

            total_clv = final_results["predicted_clv"].sum()
            print(
                f"\nTotal Predicted {config.CLV_PREDICTION_MONTHS}-Month CLV for all segments: £{total_clv:,.2f}"
            )

        else:
            print("\nCLV prediction skipped due to Gamma-Gamma model issues.")

        print("\n--- CLV Prediction Complete ---")

    except FileNotFoundError as e:
        # Refer to config file path in error message
        print(
            f"Error: {e}. Make sure '{config.DATA_FILE_NAME}' is in the directory '{config.BASE_DIR}'."
        )
    except ValueError as e:
        print(f"Data Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CLV prediction: {e}")
        import traceback

        traceback.print_exc()
