# app.py
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import warnings
import re  # For markdown parsing

# Import functions from our modules
from data_processor import load_and_clean_data
from rfm_analyzer import calculate_rfm
from clv_predictor import (
    prepare_lifetimes_summary,
    fit_bgnbd_model,
    fit_gamma_gamma_model,
    predict_clv,
)
import config  # Import the configuration file

warnings.filterwarnings("ignore", category=FutureWarning)
# All potential chained assignments have been fixed using .loc

print("--- Initialising Strategic CLV Dashboard Application ---")

# --- 1. Configuration & Caching (Now using config.py) ---
# Cache directory creation is handled in config.py


# --- 2. Data Loading & Processing (with Caching & Driver Metrics) ---
def get_processed_data():
    """Loads and processes data, using cache if available."""
    rfm_data = pd.DataFrame()
    clv_data = pd.DataFrame()  # This will hold combined data
    snapshot_date = None

    # Use cache paths from config
    if os.path.exists(config.CLV_CACHE_PATH) and os.path.exists(
        config.SNAPSHOT_CACHE_PATH
    ):
        print("Loading processed data from cache...")
        try:
            clv_data = pd.read_pickle(config.CLV_CACHE_PATH)  # Load the combined data
            with open(config.SNAPSHOT_CACHE_PATH, "r") as f:
                snapshot_date = pd.to_datetime(f.read().strip())

            if (
                not clv_data.empty
                and snapshot_date
                and "CustomerID" in clv_data.columns
                and "predicted_clv" in clv_data.columns
                and "Segment" in clv_data.columns
            ):
                print("Cached data loaded successfully.")
                clv_data["CustomerID"] = clv_data["CustomerID"].astype(str)
                # Extract RFM data part if needed separately elsewhere, but mostly use combined clv_data
                rfm_cols = [
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
                # Ensure all expected RFM columns exist before extracting
                rfm_cols_present = [
                    col for col in rfm_cols if col in clv_data.columns
                ]
                rfm_data = clv_data[rfm_cols_present].copy()
                return rfm_data, clv_data, snapshot_date
            else:
                print("Cache files invalid or incomplete. Reprocessing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Reprocessing...")

    print("Processing data from scratch...")
    try:
        print("Loading and cleaning raw data & calculating drivers...")
        # Use data file path from config
        cleaned_data, driver_metrics_df, snapshot_date = load_and_clean_data(
            config.DATA_FILE_PATH
        )
        # Ensure CustomerID is string in driver metrics (index is already CustomerID)
        driver_metrics_df.index = driver_metrics_df.index.astype(str)

        print("\nCalculating RFM segments...")
        rfm_data = calculate_rfm(cleaned_data, snapshot_date)  # Index is CustomerID
        rfm_data.index = rfm_data.index.astype(str)  # Ensure index is string

        print("\nPreparing data for Lifetimes models...")
        # Ensure CustomerID is string before passing to lifetimes
        cleaned_data["CustomerID"] = cleaned_data["CustomerID"].astype(str)
        lifetimes_summary = prepare_lifetimes_summary(cleaned_data)  # Index is CustomerID
        lifetimes_summary.index = lifetimes_summary.index.astype(
            str
        )  # Ensure index is string

        print("\nFitting BG/NBD model...")
        bgf = fit_bgnbd_model(lifetimes_summary)

        print("\nFitting Gamma-Gamma model...")
        ggf = fit_gamma_gamma_model(lifetimes_summary)

        # Combine base summary data for CLV calculation
        clv_base = lifetimes_summary.copy()

        if ggf and bgf:
            print("\nPredicting CLV and Probability Alive...")
            # Use prediction months from config
            clv_base = predict_clv(
                clv_base,
                bgf,
                ggf,
                time_months=config.CLV_PREDICTION_MONTHS,
                freq=config.LIFETIMES_FREQ,
                discount_rate=config.MONTHLY_DISCOUNT_RATE,
            )
            clv_base["prob_alive"] = bgf.conditional_probability_alive(
                clv_base["frequency"], clv_base["recency"], clv_base["T"]
            )
            # Keep the average monetary value used by GammaGamma
            clv_base.rename(
                columns={"monetary_value": "avg_transaction_value"}, inplace=True
            )
        else:
            print("CLV/Prob Alive prediction skipped. Setting defaults.")
            clv_base["predicted_clv"] = 0.0
            clv_base["prob_alive"] = 0.0
            clv_base["avg_transaction_value"] = (
                0.0  # Add column even if prediction failed
            )

        # --- Combine ALL data: RFM + CLV Predictions + Driver Metrics ---
        # Merge using CustomerID index
        clv_data = rfm_data.merge(
            clv_base[["predicted_clv", "prob_alive", "avg_transaction_value"]],
            left_index=True,
            right_index=True,
            how="left",
        )
        clv_data = clv_data.merge(
            driver_metrics_df,  # Already indexed by CustomerID (string)
            left_index=True,
            right_index=True,
            how="left",
        )
        clv_data.reset_index(inplace=True)  # Reset index to get CustomerID as column

        # Fill NAs resulting from merges or filtered-out customers using .loc to avoid chained assignment
        clv_data.loc[:, "predicted_clv"] = clv_data["predicted_clv"].fillna(0.0)
        clv_data.loc[:, "prob_alive"] = clv_data["prob_alive"].fillna(0.0)
        clv_data.loc[:, "avg_transaction_value"] = clv_data["avg_transaction_value"].fillna(0.0)
        # Handle missing driver metrics (e.g., for customers with 0/1 purchase) using .loc
        # Use -1 to indicate N/A for time_to_second_purchase
        clv_data.loc[:, "time_to_second_purchase"] = clv_data["time_to_second_purchase"].fillna(-1)
        clv_data.loc[:, "category_count"] = clv_data["category_count"].fillna(0)

        print("Combined RFM, CLV, and Driver metrics.")

        # Cache the combined results using paths from config
        print("Caching processed data...")
        clv_data.to_pickle(config.CLV_CACHE_PATH)
        with open(config.SNAPSHOT_CACHE_PATH, "w") as f:
            f.write(str(snapshot_date))
        print("Data cached successfully.")

        # Extract RFM part for compatibility if needed
        # Ensure original rfm_data columns exist before selecting
        original_rfm_cols = [
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
        rfm_cols_to_extract = [
            col for col in original_rfm_cols if col in clv_data.columns
        ]
        rfm_data_extracted = clv_data[rfm_cols_to_extract].copy()

        return rfm_data_extracted, clv_data, snapshot_date  # Return extracted RFM data

    except FileNotFoundError as e:
        print(f"FATAL ERROR: Data file '{config.DATA_FILE_PATH}' not found. {e}")
        return pd.DataFrame(), pd.DataFrame(), None
    except Exception as e:
        print(f"FATAL ERROR during data processing: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame(), None


# --- Load data globally ---
rfm_data, clv_data, snapshot_date = get_processed_data()


# --- Helper Function to Load and Parse Markdown ---
def load_markdown(file_path):
    """Loads markdown content from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading markdown file {file_path}: {e}")
        return "# Error Loading Content\n\nThe methodology content could not be loaded."


def parse_markdown_sections(markdown_content):
    """Parses markdown content into sections based on H2 headers (##)."""
    sections = {}
    # Regex to find H2 headers (lines starting with '## ')
    header_pattern = r"^## (.*)"
    # Split the content by H2 headers, keeping the headers
    parts = re.split(header_pattern, markdown_content, flags=re.MULTILINE)

    current_header = "Introduction"  # Default for content before first H2
    current_content = parts[0].strip() if parts else ""

    i = 1
    while i < len(parts):
        # Store the previous section
        sections[current_header] = current_content.strip()
        # Get the new header and content
        current_header = parts[i].strip()
        current_content = (
            parts[i + 1].strip() if (i + 1) < len(parts) else ""
        )
        i += 2

    # Add the last section
    sections[current_header] = current_content.strip()

    # Filter out empty sections (sometimes happens with splitting)
    return {
        header: content
        for header, content in sections.items()
        if content or header == "Introduction"
    }


# Load and parse markdown content using path from config
methodology_content_md = load_markdown(config.WRITE_UP_FILE_PATH)
methodology_sections = parse_markdown_sections(methodology_content_md)

# --- 3. Visualization Helper Functions ---
# Define all the visualization functions needed for the dashboard


def create_segment_treemap(data_df: pd.DataFrame):
    """
    Creates a treemap visualisation of segment contribution to total CLV.

    Args:
        data_df (pd.DataFrame): DataFrame containing customer data. Must include
                                'Segment' and 'predicted_clv' columns.

    Returns:
        go.Figure: A Plotly treemap figure.
    """
    if (
        data_df.empty
        or "Segment" not in data_df.columns
        or "predicted_clv" not in data_df.columns
    ):
        return go.Figure(layout={"title": "No data available for treemap"})

    # Aggregate CLV by segment
    segment_clv = data_df.groupby("Segment")["predicted_clv"].sum().reset_index()
    # Handle potential division by zero if total CLV is 0
    total_clv_sum = segment_clv["predicted_clv"].sum()
    segment_clv["percent"] = (
        (segment_clv["predicted_clv"] / total_clv_sum * 100)
        if total_clv_sum > 0
        else 0
    )

    # Sort by CLV value descending
    segment_clv = segment_clv.sort_values("predicted_clv", ascending=False)

    # Create treemap (Original Version)
    fig = px.treemap(
        segment_clv,
        path=["Segment"],
        values="predicted_clv",
        color="percent",
        color_continuous_scale="Viridis",
        hover_data={"predicted_clv": ":,.2f", "percent": ":.2f%"}, # Original hover_data attempt
        custom_data=["Segment"],
    )

    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),  # Set all margins to 0
        paper_bgcolor='rgba(0,0,0,0)', # Set background transparent
        coloraxis_showscale=False,
        template=config.PLOTLY_TEMPLATE,
    )

    return fig


def generate_executive_summary_text(rfm_data: pd.DataFrame, clv_data: pd.DataFrame):
    """
    Generates dynamic text insights for the executive summary tab.

    Calculates key metrics like total CLV, top segment contribution, at-risk value,
    and impact of early engagement and category diversity.

    Args:
        rfm_data (pd.DataFrame): DataFrame with RFM data (potentially unused if clv_data has all info).
        clv_data (pd.DataFrame): DataFrame with combined RFM, CLV, and driver metrics.
                                 Must include 'Segment', 'predicted_clv', 'time_to_second_purchase',
                                 'category_count', 'Frequency'.

    Returns:
        list: A list of Dash HTML components representing the summary text.
    """
    if clv_data.empty:
        return "No data available for analysis."

    # Calculate key metrics
    # total_customers = len(clv_data["CustomerID"].unique())  # F841: Unused
    total_clv = clv_data["predicted_clv"].sum()

    # Top segment by CLV
    segment_clv = (
        clv_data.groupby("Segment")["predicted_clv"]
        .sum()
        .sort_values(ascending=False)
    )
    top_segment = segment_clv.index[0] if not segment_clv.empty else "N/A"
    top_segment_pct = (
        (segment_clv.iloc[0] / total_clv * 100)
        if not segment_clv.empty and total_clv > 0
        else 0
    )

    # At risk value
    at_risk_mask = clv_data["Segment"] == "At Risk"
    at_risk_value = clv_data.loc[at_risk_mask, "predicted_clv"].sum()
    at_risk_pct = (at_risk_value / total_clv * 100) if total_clv > 0 else 0

    # Early engagement impact (using time_to_second_purchase)
    # Use bins from config for consistency
    early_cutoff = (
        config.TIME_TO_SECOND_PURCHASE_BINS[1]
        if len(config.TIME_TO_SECOND_PURCHASE_BINS) > 1
        else 30
    )
    early_mask = (clv_data["time_to_second_purchase"] <= early_cutoff) & (
        clv_data["time_to_second_purchase"] >= 0
    )
    late_mask = clv_data["time_to_second_purchase"] > early_cutoff

    early_avg_clv = (
        clv_data.loc[early_mask, "predicted_clv"].mean() if early_mask.any() else 0
    )
    late_avg_clv = (
        clv_data.loc[late_mask, "predicted_clv"].mean() if late_mask.any() else 0
    )

    early_impact = (
        ((early_avg_clv - late_avg_clv) / late_avg_clv * 100) if late_avg_clv > 0 else 0
    )

    # Category diversity impact
    # Define diversity threshold (e.g., based on config or analysis)
    diversity_threshold = 3  # Example threshold
    diverse_mask = clv_data["category_count"] >= diversity_threshold
    limited_mask = clv_data["category_count"] < diversity_threshold

    diverse_avg_clv = (
        clv_data.loc[diverse_mask, "predicted_clv"].mean() if diverse_mask.any() else 0
    )
    limited_avg_clv = (
        clv_data.loc[limited_mask, "predicted_clv"].mean() if limited_mask.any() else 0
    )

    diversity_impact = (
        ((diverse_avg_clv - limited_avg_clv) / limited_avg_clv * 100)
        if limited_avg_clv > 0
        else 0
    )

    # Calculate additional metrics for enhanced insights
    # Average CLV for At Risk customers
    avg_at_risk_clv = at_risk_value / clv_data[at_risk_mask].shape[0] if clv_data[at_risk_mask].shape[0] > 0 else 0
    
    # Potential uplift from targeted campaigns
    # Assuming a conservative 15% success rate for at-risk reactivation
    potential_recovery_rate = 0.15
    potential_recovery_value = at_risk_value * potential_recovery_rate
    
    # Second purchase conversion rate
    new_customers = clv_data[clv_data["Segment"] == "New Customers"].shape[0]
    repeat_customers = clv_data[clv_data["Frequency"] > 1].shape[0]
    second_purchase_rate = repeat_customers / (new_customers + repeat_customers) * 100 if (new_customers + repeat_customers) > 0 else 0
    
    # Generate the enhanced summary text with more quantified insights
    summary = [
        html.H5("Key Value Insights"),
        html.P(
            [
                f"The '{top_segment}' segment contributes ",
                html.Strong(f"{top_segment_pct:.2f}%"),
                f" of total predicted {config.CLV_PREDICTION_MONTHS}-Month CLV, "
                "representing our highest-value customer group.",
            ]
        ),
        html.P(
            [
                f"'At Risk' customers represent ",
                html.Strong(f"£{at_risk_value:,.2f}"),
                f" ({at_risk_pct:.2f}%) in potential CLV that could be lost "
                "without intervention.",
            ]
        ),
        html.Hr(className="my-2"),
        html.H5("Strategic Opportunities"),
        html.P(
            [
                f"Customers making a second purchase within {early_cutoff} days show ",
                html.Strong(f"{early_impact:+.2f}%"),
                " higher average CLV than those who take longer.",
            ]
        ),
        html.P(
            [
                f"Customers purchasing from {diversity_threshold}+ categories have ",
                html.Strong(f"{diversity_impact:+.2f}%"),
                " higher average CLV than those with limited category exposure.",
            ]
        ),
        html.Hr(className="my-2"),
        html.H5("Actionable Recommendations"),
        html.P(
            [
                f"Targeting 'At Risk' customers (avg. predicted CLV £{avg_at_risk_clv:.2f}) with uplift-modelled campaigns could recover ",
                html.Strong(f"£{potential_recovery_value:,.2f}"),
                f" of potential value, based on a conservative {potential_recovery_rate*100:.0f}% success rate."
            ]
        ),
        html.P(
            [
                f"Currently, only {second_purchase_rate:.1f}% of new customers make a second purchase. Accelerating second purchases for 'New Customers' could boost their segment CLV by an estimated ",
                html.Strong(f"{early_impact:.1f}%"),
                " based on our driver analysis."
            ]
        ),
        html.P(
            [
                html.Em(
                    "Note: This analysis is based on historical data. Modern "
                    "AI-driven approaches would enable more granular, "
                    "real-time insights and personalised intervention strategies."
                )
            ]
        ),
    ]

    return summary


def create_segment_distribution_chart(data_df: pd.DataFrame, clicked_segment: str = None):
    """
    Creates a pie chart showing the distribution of customer segments.

    Args:
        data_df (pd.DataFrame): DataFrame containing customer data. Must include 'Segment'.
        clicked_segment (str, optional): The segment name to highlight (pull out) in the pie chart. Defaults to None.

    Returns:
        go.Figure: A Plotly pie chart figure.
    """
    if data_df.empty or "Segment" not in data_df.columns:
        return go.Figure(
            layout={"title": "No data available for segment distribution"}
        )

    # Count customers by segment
    segment_counts = data_df["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]
    total_count = segment_counts["Count"].sum()
    segment_counts["Percent"] = (
        (segment_counts["Count"] / total_count * 100) if total_count > 0 else 0
    )

    # Sort by count descending
    segment_counts = segment_counts.sort_values("Count", ascending=False)

    # Create pie chart
    fig = px.pie(
        segment_counts,
        names="Segment",
        values="Count",
        hover_data={"Percent": ":.2f%"},
        labels={"Count": "Number of Customers"},
        custom_data=["Segment"],  # Add segment name for callbacks
    )

    # Highlight clicked segment if provided
    if clicked_segment:
        fig.update_traces(
            pull=[
                0.1 if segment == clicked_segment else 0
                for segment in segment_counts["Segment"]
            ]
        )

    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        template=config.PLOTLY_TEMPLATE,  # Use template from config
    )

    return fig


def create_rfm_scatter_plot(data_df: pd.DataFrame, clicked_segment: str = None):
    """
    Creates a scatter plot of Recency vs Frequency, coloured by Monetary value.

    Highlights a specific segment if `clicked_segment` is provided.

    Args:
        data_df (pd.DataFrame): DataFrame containing customer data. Must include 'Recency',
                                'Frequency', 'Monetary', 'Segment', 'CustomerID', 'predicted_clv'.
        clicked_segment (str, optional): The segment name to highlight. Defaults to None.

    Returns:
        go.Figure: A Plotly scatter plot figure.
    """
    if data_df.empty or not all(
        col in data_df.columns
        for col in [
            "Recency",
            "Frequency",
            "Monetary",
            "Segment",
            "CustomerID",
            "predicted_clv",
        ]
    ):
        return go.Figure(
            layout={"title": "No data available for RFM scatter plot"}
        )

    # Create a copy to avoid modifying the original
    plot_df = data_df.copy()

    # Highlight clicked segment if provided
    if clicked_segment:
        plot_df["opacity"] = plot_df["Segment"].apply(
            lambda x: 1.0 if x == clicked_segment else 0.3
        )
        plot_df["size"] = plot_df["Segment"].apply(
            lambda x: 10 if x == clicked_segment else 6
        )
    else:
        plot_df["opacity"] = 1.0
        plot_df["size"] = 8

    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x="Recency",
        y="Frequency",
        color="Segment",
        size="Monetary",
        hover_name="CustomerID",
        hover_data={
            "Recency": True,
            "Frequency": True,
            "Monetary": ":,.2f",
            "predicted_clv": ":,.2f",
            "CustomerID": False,
            "opacity": False,
            "size": False,
            "Segment": True
        },
        opacity=plot_df["opacity"],
        size_max=15,
    )

    fig.update_layout(
        xaxis_title="Recency (days)",
        yaxis_title="Frequency (# of purchases)",
        legend_title="Segment",
        margin=dict(t=30, l=10, r=10, b=10),
        template=config.PLOTLY_TEMPLATE,  # Use template from config
    )

    return fig


def create_segment_profile_radar(data_df):
    """Creates a radar chart comparing key metrics across segments."""
    required_cols = [
        "Segment",
        "R_Score",
        "F_Score",
        "M_Score",
        "predicted_clv",
        "prob_alive",
    ]
    if data_df.empty or not all(col in data_df.columns for col in required_cols):
        return go.Figure(
            layout={"title": "No data available for segment profile radar"}
        )

    # Calculate average metrics by segment
    segment_profiles = data_df.groupby("Segment").agg({
        'R_Score': 'mean',
        'F_Score': 'mean',
        'M_Score': 'mean',
        'predicted_clv': 'mean',
        'prob_alive': 'mean'
    }).reset_index()

    # Normalize CLV for radar chart (scale to 1-5 like RFM scores)
    max_clv = segment_profiles["predicted_clv"].max()
    segment_profiles["CLV_Score"] = (
        (1 + 4 * (segment_profiles["predicted_clv"] / max_clv)) if max_clv > 0 else 1
    )

    # Normalize prob_alive for radar chart (scale to 1-5)
    segment_profiles["Active_Score"] = 1 + 4 * segment_profiles["prob_alive"]

    # Create radar chart
    fig = go.Figure()

    categories = ["Recency (R)", "Frequency (F)", "Monetary (M)", "CLV", "Prob. Active"]

    for i, segment in enumerate(segment_profiles["Segment"]):
        fig.add_trace(
            go.Scatterpolar(
                r=[
                    segment_profiles.loc[i, "R_Score"],
                    segment_profiles.loc[i, "F_Score"],
                    segment_profiles.loc[i, 'M_Score'],
                    segment_profiles.loc[i, "CLV_Score"],
                    segment_profiles.loc[i, "Active_Score"]
                ],
                theta=categories,
                fill="toself",
                name=segment,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        margin=dict(t=30, l=10, r=10, b=10),
        template=config.PLOTLY_TEMPLATE,  # Use template from config
    )

    return fig


def create_clv_distribution_plot(data_df: pd.DataFrame):
    """
    Creates a histogram showing the distribution of predicted CLV.

    Args:
        data_df (pd.DataFrame): DataFrame containing customer data. Must include 'predicted_clv'.

    Returns:
        go.Figure: A Plotly histogram figure.
    """
    if data_df.empty or "predicted_clv" not in data_df.columns:
        return go.Figure(
            layout={"title": "No data available for CLV distribution"}
        )

    # Filter to customers with positive CLV for better visualisation
    plot_df = data_df[data_df["predicted_clv"] > 0].copy()

    if plot_df.empty:
        return go.Figure(layout={"title": "No positive CLV data available"})

    # Create histogram
    fig = px.histogram(
        plot_df,
        x="predicted_clv",
        color="Segment",
        marginal='box',
        nbins=50,
        opacity=0.7,
    )

    fig.update_layout(
        xaxis_title=f"Predicted {config.CLV_PREDICTION_MONTHS}-Month CLV (£)",
        yaxis_title="Number of Customers",
        legend_title="Segment",
        margin=dict(t=30, l=10, r=10, b=10),
        template=config.PLOTLY_TEMPLATE,  # Use template from config
    )

    return fig


def create_segment_clv_boxplot(data_df):
    """Creates a box plot comparing CLV distributions across segments."""
    if data_df.empty or not all(
        col in data_df.columns for col in ["Segment", "predicted_clv"]
    ):
        return go.Figure(
            layout={"title": "No data available for segment CLV boxplot"}
        )

    # Filter to customers with positive CLV for better visualisation
    plot_df = data_df[data_df["predicted_clv"] > 0].copy()

    if plot_df.empty:
        return go.Figure(layout={"title": "No positive CLV data available"})

    # Create box plot
    fig = px.box(
        plot_df, x="Segment", y="predicted_clv", color="Segment", points="outliers"
    )

    fig.update_layout(
        xaxis_title="Customer Segment",
        yaxis_title=f"Predicted {config.CLV_PREDICTION_MONTHS}-Month CLV (£)",
        showlegend=False,
        margin=dict(t=30, l=10, r=10, b=10),
        template=config.PLOTLY_TEMPLATE,  # Use template from config
    )

    return fig


def create_prob_active_plot(data_df):
    """Creates a visualisation of probability of being active by segment."""
    if data_df.empty or not all(
        col in data_df.columns for col in ["Segment", "prob_alive"]
    ):
        return go.Figure(
            layout={"title": "No data available for probability active plot"}
        )

    # Calculate average probability by segment
    segment_probs = data_df.groupby("Segment")["prob_alive"].mean().reset_index()
    segment_probs = segment_probs.sort_values("prob_alive", ascending=False)

    # Create bar chart
    fig = px.bar(
        segment_probs, x="Segment", y="prob_alive", color="Segment", text_auto=".2%",
        hover_data={'prob_alive': ':.2%'} # Add explicit hover formatting
    )

    fig.update_layout(
        xaxis_title="Customer Segment",
        yaxis_title="Avg. Probability of Being Active",
        yaxis=dict(tickformat=".2%"), # Match axis format to label/hover
        showlegend=False,
        margin=dict(t=30, l=10, r=10, b=10),
        template=config.PLOTLY_TEMPLATE,  # Use template from config
    )

    return fig


def plot_driver_comparison(
    data_df: pd.DataFrame, driver_col: str, clv_col: str = "predicted_clv", title_prefix: str = "", bins: list = None
):
    """
    Generic function to create a box plot comparing CLV based on a driver column.

    Handles both categorical drivers and continuous drivers (by binning).

    Args:
        data_df (pd.DataFrame): DataFrame containing customer data. Must include the
                                specified `clv_col` and `driver_col`.
        driver_col (str): The column name of the driver metric to analyse.
        clv_col (str, optional): The column name for the CLV metric. Defaults to "predicted_clv".
        title_prefix (str, optional): Prefix for the plot title. Defaults to "".
        bins (list, optional): List of bin edges for continuous drivers. If provided,
                               the driver column will be binned before plotting.
                               Special handling for 'time_to_second_purchase' is included.
                               Defaults to None (treat as categorical or unbinned).

    Returns:
        go.Figure: A Plotly box plot figure.
    """
    if (
        data_df.empty
        or driver_col not in data_df.columns
        or clv_col not in data_df.columns
    ):
        return go.Figure(layout={"title": f"Missing data for {driver_col} analysis"})

    plot_data = data_df[data_df[clv_col] > 0].copy() # Focus on positive CLV

    if plot_data.empty:
        return go.Figure(layout={"title": f"No positive CLV data for {driver_col}"})

    x_axis_label = driver_col.replace('_', ' ').title()
    y_axis_label = clv_col.replace("_", " ").title() + " (£)"

    # Handle binning for continuous drivers if requested
    if bins:
        try:
            # Create bins, handle -1 for 'time_to_second_purchase' N/A
            if driver_col == "time_to_second_purchase":
                # Define specific bins for time to second purchase, treating -1 separately
                bin_labels = [f'{bins[i]}-{bins[i+1]-1} days' for i in range(len(bins)-1)] + [f'>= {bins[-1]} days', 'First Purchase Only']
                # Ensure cut_bins covers the max value correctly
                max_val = plot_data[driver_col][plot_data[driver_col] >= 0].max()
                cut_bins = (
                    bins + [max_val + 1]
                    if not pd.isna(max_val)
                    else bins + [bins[-1] + 1]
                )  # Add upper bound safely

                plot_data[f'{driver_col}_binned'] = pd.cut(
                    plot_data[driver_col][
                        plot_data[driver_col] >= 0
                    ],  # Only bin non-negative values
                    bins=cut_bins,
                    labels=bin_labels[:-1],
                    right=False,
                    include_lowest=True,
                )
                # Assign the 'First Purchase Only' label to original -1 values using .loc
                plot_data[f'{driver_col}_binned'] = plot_data[f'{driver_col}_binned'].cat.add_categories(['First Purchase Only'])
                plot_data.loc[plot_data[driver_col] == -1, f"{driver_col}_binned"] = (
                    "First Purchase Only"
                )
                x_col = f'{driver_col}_binned'
                x_axis_label = "Time to Second Purchase"  # Override label
            else:
                # General binning for other continuous variables
                max_val = plot_data[driver_col].max()
                cut_bins = (
                    bins + [max_val + 1]
                    if not pd.isna(max_val)
                    else bins + [bins[-1] + 1]
                )
                bin_labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-1)] + [f'>= {bins[-1]}']
                plot_data[f"{driver_col}_binned"] = pd.cut(
                    plot_data[driver_col],
                    bins=cut_bins,
                    labels=bin_labels,
                    right=False,
                    include_lowest=True,
                )
                x_col = f'{driver_col}_binned'

        except Exception as e:
            print(f"Error binning {driver_col}: {e}. Using raw values.")
            x_col = driver_col  # Fallback to raw values if binning fails
    else:
        x_col = driver_col  # Use the raw driver column if no bins provided

    # Convert binned column to string type for Plotly if it exists
    if "_binned" in x_col and x_col in plot_data.columns:
        plot_data[x_col] = plot_data[x_col].astype(str)

    fig = px.box(
        plot_data,
        x=x_col,
        y=clv_col,
        points=False,
        labels={clv_col: y_axis_label, x_col: x_axis_label})
    fig.update_layout(
        title_x=0.5,
        margin=dict(t=30, l=10, r=10, b=10),
        template=config.PLOTLY_TEMPLATE,  # Use template from config
    )
    return fig


# --- 4. Initialize Dash App with Bootstrap ---
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
)  # Add Font Awesome if using icons
app.title = "Strategic Customer Value & Growth Engine"
server = app.server  # For deployment


# --- 5. Define App Layout ---
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1(
                    "CLV Analysis & Growth Engine: From Historical Foundation to Modern AI",
                    className="app-title",
                )
            )
        ),  # Apply CSS class
        dbc.Row(
            dbc.Col(
                html.P(
                    "Demonstrating Core CLV Principles (2010 Data) & Charting the Course for AI-Driven Strategy (2025+)",
                    className="text-muted text-center ps-2 mb-4",
                )
            ) # Close dbc.Col
        ),
        dbc.Tabs(
            id="app-tabs",
            active_tab="tab-summary",
            children=[
                dbc.Tab(label="Executive Summary", tab_id="tab-summary"),
                dbc.Tab(label="Segmentation Deep Dive", tab_id="tab-segmentation"),
                dbc.Tab(label="Predictive Insights", tab_id="tab-prediction"),
                dbc.Tab(label="Key CLV Drivers", tab_id="tab-drivers"),  # NEW TAB
                dbc.Tab(label="Strategic Action Centre", tab_id="tab-strategy"),
                dbc.Tab(label="Data & Methodology", tab_id="tab-methodology"),
            ],
        ),
        # Wrap content area in a spinner for initial load indication
        dbc.Row(
            dbc.Col(
                dbc.Spinner(
                    id="tab-content",
                    color="primary",
                    spinner_style={"width": "3rem", "height": "3rem"},
                ),
                className="p-4 border border-top-0 bg-white",
            )
        ),  # Content area
    ],
    fluid=True,
)

# --- 6. Define Callbacks ---


# Callback to render tab content
@callback(Output("tab-content", "children"), Input("app-tabs", "active_tab"))
def render_tab_content(active_tab):
    if clv_data.empty:  # Check combined data now
        return dbc.Alert(
            "Error Loading Data: Could not load or process required data. Check logs.",
            color="danger",
        )

    # --- EXECUTIVE SUMMARY TAB ---
    if active_tab == "tab-summary":
        total_clv = clv_data["predicted_clv"].sum()
        # Calculate average only for those predicted > 0
        avg_clv_pred_positive = (
            clv_data[clv_data["predicted_clv"] > 0]["predicted_clv"].mean()
            if (clv_data["predicted_clv"] > 0).any()
            else 0
        )
        active_customers = len(
            clv_data["CustomerID"].unique()
        )  # Use unique CustomerIDs
        prob_active_avg = (
            clv_data[clv_data["prob_alive"] > 0]["prob_alive"].mean()
            if (clv_data["prob_alive"] > 0).any()
            else 0
        )

        kpi_card_1 = dbc.Card(
            [
                dbc.CardHeader(
                    f"Total Predicted {config.CLV_PREDICTION_MONTHS}M CLV",
                    className="kpi-card-header",
                ),
                dbc.CardBody(f"£{total_clv:,.2f}", className="kpi-card-body"),
            ]
        )
        kpi_card_2 = dbc.Card(
            [
                dbc.CardHeader("Avg. CLV (Predicted > 0)", className="kpi-card-header"),
                dbc.CardBody(
                    f"£{avg_clv_pred_positive:,.2f}", className="kpi-card-body"
                ),
            ]
        )
        kpi_card_3 = dbc.Card(
            [
                dbc.CardHeader("Total Customers Analysed", className="kpi-card-header"),
                dbc.CardBody(f"{active_customers:,}", className="kpi-card-body"),
            ]
        )
        kpi_card_4 = dbc.Card(
            [
                dbc.CardHeader("Avg. Probability Active", className="kpi-card-header"),
                dbc.CardBody(f"{prob_active_avg:.2%}", className="kpi-card-body"),
            ]
        )

        # Reuse layout from previous refinement
        return html.Div(
            [
                html.H3("Executive Summary: The CLV Landscape", className="mb-4"),
                dbc.Row(
                    [
                        dbc.Col(kpi_card_1, md=3),
                        dbc.Col(kpi_card_2, md=3),
                        dbc.Col(kpi_card_3, md=3),
                        dbc.Col(kpi_card_4, md=3),
                    ],
                    className="mb-4",
                ),
                dbc.Row( # Add g-0 to remove gutters and explicitly name children
                    className="g-0 mb-4", # Keyword argument
                    children=[ # Explicitly name children argument
                        dbc.Col(
                            [
                                html.H4("Segment Value Contribution", className="mb-0"), # Remove bottom margin
                                # Wrap Graph in a Div with no padding/margin
                                html.Div(
                                    dcc.Graph(
                                        id="summary-treemap",
                                        figure=create_segment_treemap(clv_data),
                                        style={'height': '100%', 'width': '100%'}
                                    ),
                                    style={'padding': 0, 'margin': 0}
                                )
                            ],
                            md=7,
                            className="p-0" # Keep column padding removal
                        ),
                        dbc.Col(
                            [
                                html.H4("Key Insights"),
                                # Ensure combined data is passed
                                dbc.Card(
                                    dbc.CardBody(
                                        id="dynamic-summary-text",
                                        children=generate_executive_summary_text(
                                            clv_data, clv_data
                                        ),
                                    ),
                                    className="h-100",
                                ),
                            ],
                            md=5,
                        ),
                    ],
                    # className="mb-4", # Removed duplicate className
                ),
                dbc.Alert(
                    [
                        html.H5(
                            [
                                "Modernisation Lens ", # Corrected spelling
                                html.I(className="fas fa-lightbulb ms-1"),
                            ],
                            className="alert-heading",
                        ),  # Changed icon
                        html.P(
                            [
                                "This dashboard visualises a foundational CLV analysis using ",
                                html.Strong("historical 2010-2011 data"),
                                ". The insights derived (e.g., segment value, basic drivers) are illustrative of core principles. ",
                                html.Strong(
                                    "However, a modern (2025+) AI-powered approach"
                                ),
                                " would leverage richer data streams (web behaviour, multi-channel interactions, unstructured text) and advanced techniques (Deep Learning, Uplift Modelling, NLP) for significantly more accurate predictions, causal intervention measurement, and hyper-personalised, automated strategies. ",
                                "See the ",
                                html.Strong("Methodology Tab"),
                                " for a detailed comparison.",
                            ],
                            className="mb-0",
                        ),
                    ],
                    color="info",
                    className="mt-4",
                ),  # Added margin top
            ]
        )

    # --- SEGMENTATION TAB ---
    elif active_tab == "tab-segmentation":
        # Reuse layout, ensure unique IDs if plots are reused elsewhere
        return html.Div(
            [
                html.H3("Segmentation Deep Dive", className="mb-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="segment-dist-pie-segmentation",
                                figure=create_segment_distribution_chart(clv_data),
                            ),
                            md=5,
                        ),  # Changed ID
                        dbc.Col(
                            dcc.Graph(
                                id="rfm-scatter-segmentation",
                                figure=create_rfm_scatter_plot(clv_data),
                            ),
                            md=7,
                        ),  # Changed ID
                    ],
                    className="mb-4 align-items-center",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="segment-radar-segmentation",
                                figure=create_segment_profile_radar(clv_data),
                            ),
                            ), # Close dcc.Graph
                    ], # Close Row children list [ from 1006
                    className="mb-4" # Removed trailing comma
                ),
                dbc.Row(
                    [
                        dbc.Col( # Start Col for Table
                            [ # Start main children list for Col
                                # Removed extra inner list wrapper
                                html.H4("Detailed Segment Data"),
                                html.P(
                                    "Click segment in Treemap (Summary Tab) or Pie Chart (above) to filter.",
                                    className="text-muted small",
                                ), # Close html.P
                                # DataTable is now a direct sibling in the Col's children list
                                dash_table.DataTable(
                                    id="segment-data-table",  # ID used for filtering
                                    columns=[
                                        {"name": "Customer ID", "id": "CustomerID"},
                                        {"name": "Segment", "id": "Segment"},
                                        {"name": "Recency", "id": "Recency"},
                                        {"name": "Frequency", "id": "Frequency"},
                                        {
                                            "name": "Monetary (£)",
                                            "id": "Monetary",
                                            "type": "numeric",
                                            "format": dash_table.Format.Format(
                                                precision=2,
                                                scheme=dash_table.Format.Scheme.fixed,
                                            )
                                        },
                                        {"name": "R", "id": "R_Score"},
                                        {"name": "F", "id": "F_Score"},
                                        {"name": "M", "id": "M_Score"},
                                        {
                                            "name": f"Predicted {config.CLV_PREDICTION_MONTHS}M CLV (£)",
                                            "id": "predicted_clv",
                                            "type": "numeric",
                                            "format": dash_table.Format.Format(
                                                precision=2,
                                                scheme=dash_table.Format.Scheme.fixed,
                                            )
                                        },
                                        {
                                            "name": "Prob. Active",
                                            "id": "prob_alive",
                                            "type": "numeric",
                                            "format": dash_table.Format.Format(
                                                precision=2,
                                                scheme=dash_table.Format.Scheme.percentage,
                                            ) # Close Format()
                                        },
                                        {"name": "Categories", "id": "category_count"},
                                        {
                                            "name": "Days to 2nd Purch.",
                                            "id": "time_to_second_purchase",
                                        },
                                    ],
                                    # Initial data load, will be updated by callbacks
                                    data=clv_data.to_dict("records"),
                                    filter_action="native",
                                    sort_action="native",
                                    sort_mode="multi",
                                    page_action="native",
                                    page_current=0,
                                    page_size=15,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"textAlign": "left", "padding": "5px"},
                                    style_header={"fontWeight": "bold"},
                                    style_data_conditional=[
                                        {
                                            "if": {"row_index": "odd"},
                                            "backgroundColor": "rgb(248, 248, 248)",
                                        }
                                    ],
                                ), # Close DataTable
                            ], # Close main children list for Col
                            md=12,
                        )
                    ],
                    className="mb-4",
                ),
            ] # Close list of children for html.Div (12 spaces)
        ) # Closing Div for Segmentation Tab

    # --- PREDICTION TAB ---
    elif active_tab == "tab-prediction": # Start Prediction Tab elif
        return html.Div(
            [
                html.H3("Predictive Insights & Future Value", className="mb-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="clv-distribution-pred",
                                figure=create_clv_distribution_plot(clv_data),
                            ),
                            md=6,
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id="clv-boxplot-pred",
                                figure=create_segment_clv_boxplot(clv_data),
                            ),
                            md=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="prob-active-plot-pred",
                                figure=create_prob_active_plot(clv_data),
                            ),
                            md=12,
                        ),
                    ],
                    className="mb-4",
                ),
                # Add Model Validation Section
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Model Validation & Limitations"),
                                    dbc.CardBody(
                                        [
                                            html.P(
                                                "Validating predictive models is crucial to understand their performance and reliability. "
                                                "Typically, this involves training the model on one period (calibration) and testing its predictions "
                                                "against actual outcomes in a subsequent period (holdout)."
                                            ),
                                            html.P(
                                                [
                                                    html.Strong(
                                                        "Limitation with this Dataset: "
                                                    ), # Close html.Strong
                                                    # Text content directly in the list now
                                                    "The available 2010-2011 data provides a limited timeframe for robust temporal validation. "
                                                    "A standard calibration/holdout split (e.g., train on 9 months, test on 3 months) is possible but may not fully represent longer-term predictive accuracy.",
                                                ] # Close list for second P's content
                                            ), # Close second html.P
                                            html.P(
                                                "Common validation metrics for CLV models include comparing predicted vs. actual purchases (e.g., using Mean Squared Error - MSE, Mean Absolute Error - MAE) "
                                                "and assessing the correlation between predicted and actual monetary value in the holdout period."
                                            ),
                                            html.P(
                                                [
                                                    "While not a substitute for holdout validation, examining the model fit parameters (like those printed during the BG/NBD fitting process in the logs) can offer some indication of model convergence and plausibility. ",
                                                    html.Em(
                                                        "However, true performance assessment requires testing on unseen data."
                                                    )
                                               ] # Close list for fourth P's content
                                           ), # Close fourth html.P
                                       ] # Close CardBody children list [ from 1143
                                   ), # Close CardBody ) from 1142
                               ], # Close Card children list [ from 1140
                           ), # Close Card ) from 1139
                           md=12, # Add md=12 to Col
                       ), # Close Col ) from 1138
                   ], # Close Row children list [ from 1137
                   className="mb-4", # Add className to Row
               ), # Close Row ) from 1136
           ], # Close main tab children list [ from 1102
       ) # Close main tab html.Div ) from 1101

    # --- KEY DRIVERS TAB ---
    elif active_tab == "tab-drivers":
        # Use bins from config
        tts_bins = config.TIME_TO_SECOND_PURCHASE_BINS
        # Define category bins here or in config
        cat_bins = [0, 1, 2, 3, 5]  # Example: 0, 1, 2, 3-4, 5+

        return html.Div(
            [
                html.H3("Key CLV Drivers Analysis", className="mb-4"),
                html.P(
                    f"Comparing average predicted {config.CLV_PREDICTION_MONTHS}-Month CLV across different behavioural groups. Boxes show median and interquartile range.",
                    className="text-muted",
                ),
                dbc.Row(
                    [
                        dbc.Col( # First Driver Column
                            html.Div(
                                [
                                    html.H5("CLV by Time to Second Purchase"),
                                    dcc.Graph(
                                        figure=plot_driver_comparison(
                                            clv_data,
                                            "time_to_second_purchase",
                                            bins=tts_bins,
                                        )
                                    ), # Close dcc.Graph
                                    html.P( # Start Strategic Implication P
                                        [
                                            html.Strong("Strategic Implication: "),
                                            "Faster second purchases correlate strongly with higher long-term value. ",
                                            html.Em(
                                                "Focus on accelerating the journey for 'New Customers' and 'Potential Loyalists' through targeted onboarding and incentives."
                                            )
                                        ],
                                        className="small text-muted mt-2",
                                    ), # Close Strategic Implication P
                                ],
                                className="driver-plot-container mb-3",
                            ), # Close html.Div
                            md=6 # Set column width
                        ), # Close First Driver Column
                        dbc.Col( # Second Driver Column
                            html.Div(
                                [
                                    html.H5("CLV by Number of Categories Purchased"),
                                    dcc.Graph(
                                        figure=plot_driver_comparison(
                                            clv_data, "category_count", bins=cat_bins
                                        )
                                    ), # Close dcc.Graph
                                    html.P( # Start Strategic Implication P
                                        [
                                            html.Strong("Strategic Implication: "),
                                            "Customers engaging with a wider range of product categories tend to be more valuable. ",
                                            html.Em(
                                                "Encourage category exploration through cross-selling, bundling, and personalised recommendations, especially for segments with low category counts."
                                            ),
                                        ],
                                        className="small text-muted mt-2",
                                    ), # Close Strategic Implication P
                                ],
                                className="driver-plot-container mb-3",
                            ), # Close html.Div
                            md=6 # Set column width
                        ), # Close Second Driver Column
                    ]
                ),
                # Add more driver plots here (e.g., acquisition month if calculated, return behavior if calculated)
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div( # Div for Note
                                [
                                    html.H5("Note on Drivers"),
                                    html.P( # Start Note P
                                        "This analysis uses simple binning and compares CLV distributions. Causality is not implied. More sophisticated feature engineering and modelling (e.g., regression, ML) would be needed for precise driver quantification.",
                                        className="small text-muted",
                                    ), # Close Note P
                                ],
                                className="mt-3",
                            ),
                            md=12,
                        )
                    ]
                ),
            ] # Close main list for drivers tab
        ) # Close main html.Div for drivers tab

    # --- STRATEGY TAB ---
    elif active_tab == "tab-strategy":
        # Reuse strategy mapping layout from previous refinement
        strategies = {  # Updated with modern / AI-driven concepts
            "Champions": [
                "AI-Enhanced VIP Loyalty (Dynamic Tiers/Benefits)",
                "Predictive Churn Prevention Offers",
                "Exclusive Co-creation Invites",
                "Proactive Premier Support",
            ],
            "Loyal Customers": [
                "Personalised Next-Best-Offer (ML)",
                "Automated Cross-Sell/Up-sell Journeys",
                "Loyalty Point Multipliers (Targeted)",
            ],
            "Potential Loyalists": [
                "Adaptive Onboarding (AI-Personalised Content/Timing)",
                "Incentivised Category Exploration (ML-Recommended)",
                "Early Engagement Nudges",
            ],
            "New Customers": [
                "Optimised Welcome Offer (A/B Tested)",
                "Predictive 'Second Purchase' Nudge",
                "Value Proposition Reinforcement",
            ],
            "At Risk": [
                "Uplift-Modelled Reactivation (Target 'Persuadables')",
                "Personalised Win-Back Offer (Content/Channel)",
                "Churn Driver Feedback Loop",
            ],
            "Can't Lose Them": [
                "High-Value Causal Uplift Offers",
                "Dedicated Retention Specialist Outreach",
                "Analyse & Address Churn Reason",
            ],
            "Hibernating/Lost": [
                "Low-Cost Re-engagement (Content-Focused)",
                "Final High-Impact Uplift Offer",
                "Sunset Policy",
            ],
            # 'Needs Attention' is less common; often merged or handled by specific risk flags
        }
        strategy_items = []
        for segment, actions in strategies.items():
            strategy_items.append(
                dbc.ListGroupItem(
                    [html.H6(segment, className="mb-1"), html.Small(", ".join(actions))]
                )
            )

        return html.Div(
            [
                html.H3("Strategic Action Centre & Simulation", className="mb-4"),
                dbc.Row(
                dbc.Row( # Added missing dbc.Row(
                    [
                        # Column 1: Strategy Mapping
                        dbc.Col(
                            [
                                html.H4("Segment Strategy Mapping"),
                                html.P(
                                    "Key strategies tailored to segment characteristics. (Ideally, show estimated CLV impact per strategy).",
                                    className="text-muted small mb-2",
                                ),
                                dbc.ListGroup(
                                    strategy_items,
                                    flush=True,
                                    className="mb-4 border rounded",
                                    style={"maxHeight": "550px", "overflowY": "auto"},
                                ),  # Adjusted height slightly
                            ],
                            md=5,
                        ), # Added comma to separate columns
                        # Column 2: What-If Simulator
                        dbc.Col(
                            [
                                html.H4("What-If Scenario Planner"),
                                dbc.Card(
                                    [
                                        dbc.CardBody( # Start CardBody for Simulator
                                            [ # Start list for CardBody children
                                                html.P(
                                                    "Simulate the potential impact of interventions targeting key drivers:",
                                                    className="card-title",
                                                ), # Close html.P title
                                                html.P(
                                                    [
                                                        html.Em(
                                                            "Example: Improving 'Time to Second Purchase' for 'New Customers' could boost their CLV. Increasing 'Category Count' for 'Potential Loyalists' might increase their value."
                                                        )
                                                    ],
                                                    className="small text-muted mb-3",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Target 'At Risk': Increase Reactivation (+% Prob. Active)",
                                                            className="slider-label",
                                                        ),
                                                        # Use default from config
                                                        dcc.Slider(
                                                            id="slider-reactivation",
                                                            min=0,
                                                            max=50,
                                                            step=5,
                                                            value=config.DEFAULT_REACTIVATION_RATE_INCREASE,
                                                            marks={
                                                                i: f"{i}%"
                                                                for i in range(
                                                                    0, 51, 10
                                                                )
                                                            },
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Reduce 'Champion' Churn (-% Gap to 100% Prob. Active)",
                                                            className="slider-label",
                                                        ),
                                                        # Use default from config
                                                        dcc.Slider(
                                                            id="slider-churn-reduction",
                                                            min=0,
                                                            max=50,
                                                            step=5,
                                                            value=config.DEFAULT_CHURN_REDUCTION_RATE,
                                                            marks={
                                                                i: f"{i}%"
                                                                for i in range(
                                                                    0, 51, 10
                                                                )
                                                            },
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Improve 'New' -> 'Potential Loyalist' Conv. (+% CLV Boost)",
                                                            className="slider-label",
                                                        ),
                                                        # Use default from config
                                                        dcc.Slider(
                                                            id="slider-new-conversion",
                                                            min=0,
                                                            max=50,
                                                            step=5,
                                                            value=config.DEFAULT_NEW_CONVERSION_RATE_INCREASE,
                                                            marks={
                                                                i: f"{i}%"
                                                                for i in range(
                                                                    0, 51, 10
                                                                )
                                                            },
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Increase Avg. Value for Retained/Reactivated (+%)",
                                                            className="slider-label",
                                                        ),
                                                        # Use default from config
                                                        dcc.Slider(
                                                            id="slider-value-boost",
                                                            min=0,
                                                            max=20,
                                                            step=1,
                                                            value=config.DEFAULT_VALUE_BOOST_PCT,
                                                            marks={
                                                                i: f"{i}%"
                                                                for i in range(0, 21, 5)
                                                            },
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                dbc.Button(
                                                    [
                                                        "Run Simulation ",
                                                        html.I(
                                                            className="fas fa-play ms-1"
                                                        ),
                                                    ],
                                                    id="simulate-button",
                                                    color="primary",
                                                    n_clicks=0,
                                                    className="mt-2 w-100",
                                                ),
                                                dbc.Alert(
                                                    [
                                                        html.Strong(
                                                            "Simulation Assumptions: "
                                                        ),
                                                        html.Small(
                                                            "This is a simplified model. It scales CLV based on adjusted 'Probability Active' and 'Average Transaction Value' multipliers derived from sliders. It does not re-run underlying purchase/value models. 'New Cust.' boost directly scales their CLV."
                                                        ),
                                                    ],
                                                    color="warning",
                                                    className="mt-3 small",
                                                ),
                                                # Add note about advanced simulation
                                                dbc.Alert(
                                                    [
                                                        html.Strong(
                                                            "Causal Inference Note (Uplift Modelling): "
                                                        ),
                                                        html.Small(
                                                            [
                                                                "This 'What-If' simulator shows correlational impacts based on scaling existing metrics. ",
                                                                html.Strong(
                                                                    "It does NOT measure true causal effects."
                                                                ),
                                                                " A modern approach uses ",
                                                                html.Strong(
                                                                    "Uplift Modelling"
                                                                ),
                                                                " (via A/B tests or Causal ML) to estimate the ",
                                                                html.Em(
                                                                    "incremental impact"
                                                                ),
                                                                " of an intervention (e.g., an offer) – identifying customers who act *only because* of the intervention. This is crucial for maximising ROI by targeting 'persuadables' and avoiding wasted incentives on 'sure things' or 'lost causes'.",
                                                            ]
                                                        ),
                                                    ],
                                                    color="warning",
                                                    className="mt-3 small",
                                                ),  # Changed color to warning
                                            ] # Close list for CardBody children
                                        ), # Close CardBody for Simulator
                                    ], # Close list for Card children
                                ), # Close Card for Simulator
                            # Removed premature closing elements ], )
                                html.Div(
                                    [
                                        html.H5("Simulated Impact"),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        id="sim-kpi-clv",
                                                        children=[
                                                            dbc.CardHeader(
                                                                "Simulated Total CLV",
                                                                className="kpi-card-header",
                                                            ),
                                                            dbc.CardBody(
                                                                "-",
                                                                className="kpi-card-body",
                                                            ),
                                                        ],
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        id="sim-kpi-change",
                                                        children=[
                                                            dbc.CardHeader(
                                                                "CLV Change (%)",
                                                                className="kpi-card-header",
                                                            ),
                                                            dbc.CardBody(
                                                                "-",
                                                                className="kpi-card-body",
                                                            ),
                                                        ],
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        # Add spinner for simulation treemap
                                        dbc.Spinner(dcc.Graph(id="simulated-treemap")),
                                    ]
                                ), # Close html.Div for results
                            ], # Close list for Col children [ from 1345
                            md=7, # Set Col width
                        ) # Close Col for Simulator ) from 1345 - Removed comma
                    ], # Close main Row children list - Removed comma
                    className="mb-4" # Keyword argument
                ) # Close inner Row
            ) # Close outer Row started on 1324
            ] # Close main list for strategy tab (Indentation fixed)
        ) # Close main html.Div for strategy tab (Indentation fixed)

    # --- METHODOLOGY TAB ---
    elif active_tab == "tab-methodology":
        # 1. Updated Introductory Context
        intro_alert = dbc.Alert(
            [
                html.H4(
                    [
                        "Methodology: Foundational vs. Modern AI ",
                        html.I(className="fas fa-cogs ms-1"),
                    ],
                    className="alert-heading",
                ),
                html.P(
                    [
                        "This project utilises ",
                        html.Strong("historical 2010-2011 transaction data"),
                        " to demonstrate foundational CLV concepts using techniques appropriate for that era's data limitations (RFM, BG/NBD, Gamma-Gamma). These methods provide interpretable baseline insights into customer purchasing patterns and value segmentation.",
                    ]
                ),
                html.P(
                    [
                        html.Strong("Crucially, this serves as a benchmark."),
                        " A modern (2025+) CLV engine operates in a vastly different landscape. It ingests diverse, real-time data streams (behavioural, multi-channel, unstructured text) and employs sophisticated AI/ML models (Deep Learning, Gradient Boosting, Causal ML, NLP) for:",
                        html.Ul(
                            [
                                html.Li("Superior predictive accuracy."),
                                html.Li("Hyper-personalised customer experiences."),
                                html.Li(
                                    "Causal understanding of intervention impacts (Uplift)."
                                ),
                                html.Li(
                                    "Automated, optimised decision-making across the customer lifecycle."
                                ),
                            ],
                            className="mt-2",
                        ),
                    ]
                ),
                html.P(
                    "The sections below detail both the foundational methods applied here and outline the components of a modern, AI-driven approach."
                ),
            ],
            color="info",
            className="mb-4",
        )

        # 2. Add Conceptual Visualization Card
        conceptual_viz_card = dbc.Card(
            [
                dbc.CardHeader("Conceptual View: Historical vs. Modern CLV Engine"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                # Column 1: Historical Approach
                                dbc.Col(
                                    [
                                        html.H6(
                                            "Historical (This Project - 2010 Data)"
                                        ),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Input: Limited Transactions (R, F, M)"
                                                ),
                                                html.Li(
                                                    "Models: Probabilistic (BG/NBD, G-G)"
                                                ),
                                                html.Li(
                                                    "Insights: Segment trends, Basic CLV"
                                                ),
                                                html.Li(
                                                    "Actions: Manual strategy, Simple sim"
                                                ),
                                            ],
                                            className="list-unstyled small",
                                        ),
                                    ],
                                    md=6,
                                    className="border-end",
                                ),
                                # Column 2: Modern Approach
                                dbc.Col(
                                    [
                                        html.H6("Modern (2025 AI-Driven)"),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Input: Rich Streams (Behavioural, Multi-channel, Text)"
                                                ),
                                                html.Li(
                                                    "Models: AI/ML (DL, XGB, Uplift, NLP)"
                                                ),
                                                html.Li(
                                                    "Insights: Hyper-personalised predictions, Causal impact"
                                                ),
                                                html.Li(
                                                    "Actions: Real-time automation, Optimised journeys"
                                                ),
                                            ],
                                            className="list-unstyled small",
                                        ),
                                    ],
                                    md=6,
                                ),
                            ]
                        )
                    ]
                ),
            ],
            className="mb-4",
        )

        # 3. Render parsed markdown sections from the write-up file
        markdown_cards = []
        if not methodology_sections:
            markdown_cards.append(
                dbc.Alert(
                    f"Could not parse Methodology content from {config.WRITE_UP_FILE_PATH}",
                    color="danger",
                )
            )
        else:
            for header, content in methodology_sections.items():
                # Skip appendix or tailor which sections to show
                if "appendix" in header.lower():
                    continue
                # Ensure the 'Modern Approach' section is included if present in the markdown
                markdown_cards.append(
                    dbc.Card(
                        [
                            dbc.CardHeader(header),
                            dbc.CardBody(
                                dcc.Markdown(content, dangerously_allow_html=False)
                            ),
                        ],
                        className="mb-3 methodology-card",
                    )  # Add class for CSS styling
                )

        return html.Div(
            [
                html.H3("Data Foundation & Methodology", className="mb-4"),
                intro_alert,
                conceptual_viz_card,
                html.H4("Detailed Methodology (from Write-up)", className="mb-3"),
                *markdown_cards,  # Unpack the list of card components
            ]
        )
    else:
        return html.P("Select a tab")


# --- Additional Callbacks for Interactivity ---


# Callback for Treemap -> Table Filter
@callback(
    Output("segment-data-table", "data"),
    Output(
        "rfm-scatter-segmentation", "figure", allow_duplicate=True
    ),  # Also filter scatter on this tab
    Output(
        "segment-dist-pie-segmentation", "figure", allow_duplicate=True
    ),  # Highlight pie on this tab
    Input("summary-treemap", "clickData"),  # Listen to treemap on summary tab
    prevent_initial_call=True,
)
def update_table_on_treemap_click(clickData):
    triggered_id = ctx.triggered_id
    if not triggered_id or not clickData:
        print("No treemap click detected for table update.")
        scatter_fig = create_rfm_scatter_plot(clv_data)
        pie_fig = create_segment_distribution_chart(clv_data)
        return clv_data.to_dict("records"), scatter_fig, pie_fig

    point_data = clickData["points"][0]
    # Check if root is clicked (currentPath is '/') or 'All Segments' label
    if (
        point_data.get("currentPath") == "/"
        or point_data.get("label") == "All Segments"
    ):
        print("Treemap root or 'All Segments' clicked, resetting table filter.")
        scatter_fig = create_rfm_scatter_plot(clv_data)
        pie_fig = create_segment_distribution_chart(clv_data)
        return clv_data.to_dict("records"), scatter_fig, pie_fig

    elif "label" in point_data:
        clicked_segment = point_data["label"]
        print(f"Treemap clicked. Filtering table for segment: {clicked_segment}")
        filtered_df = clv_data[clv_data["Segment"] == clicked_segment]
        scatter_fig = create_rfm_scatter_plot(clv_data, clicked_segment=clicked_segment)
        pie_fig = create_segment_distribution_chart(
            clv_data, clicked_segment=clicked_segment
        )
        return filtered_df.to_dict("records"), scatter_fig, pie_fig

    # Fallback if clickData structure is unexpected
    print("Unexpected treemap click data structure.")
    scatter_fig = create_rfm_scatter_plot(clv_data)
    pie_fig = create_segment_distribution_chart(clv_data)
    return clv_data.to_dict("records"), scatter_fig, pie_fig


# Callback for Segmentation Pie -> Table Filter (Similar logic, different source)
@callback(
    Output("segment-data-table", "data", allow_duplicate=True),
    Output("rfm-scatter-segmentation", "figure", allow_duplicate=True),
    Output("segment-dist-pie-segmentation", "figure", allow_duplicate=True),
    Input(
        "segment-dist-pie-segmentation", "clickData"
    ),  # Listen to pie on segmentation tab
    prevent_initial_call=True,
)
def update_table_on_pie_click(clickData):
    triggered_id = ctx.triggered_id
    if not triggered_id or not clickData or not clickData["points"]:
        print("No valid pie click data for table update.")
        scatter_fig = create_rfm_scatter_plot(clv_data)
        pie_fig = create_segment_distribution_chart(clv_data)
        return clv_data.to_dict("records"), scatter_fig, pie_fig

    if "label" in clickData["points"][0]:
        clicked_segment = clickData["points"][0]["label"]
        print(
            f"Segmentation Pie clicked. Filtering table for segment: {clicked_segment}"
        )
        filtered_df = clv_data[clv_data["Segment"] == clicked_segment]
        # Update scatter and pie highlight
        scatter_fig = create_rfm_scatter_plot(clv_data, clicked_segment=clicked_segment)
        pie_fig = create_segment_distribution_chart(
            clv_data, clicked_segment=clicked_segment
        )
        return filtered_df.to_dict("records"), scatter_fig, pie_fig
    else:
        # No label found, reset to full view
        print("No label found in pie click data.")
        scatter_fig = create_rfm_scatter_plot(clv_data)
        pie_fig = create_segment_distribution_chart(clv_data)
        return clv_data.to_dict("records"), scatter_fig, pie_fig


# --- Refined Simulation Callback ---
@callback(
    Output("sim-kpi-clv", "children"),
    Output("sim-kpi-change", "children"),
    Output("simulated-treemap", "figure"),
    Input("simulate-button", "n_clicks"),
    State("slider-reactivation", "value"),
    State("slider-churn-reduction", "value"),
    State("slider-new-conversion", "value"),
    State("slider-value-boost", "value"),  # Add state for value boost slider
    State("app-tabs", "active_tab"),  # Ensure simulation only runs when tab is active
    prevent_initial_call=True,
)
def run_refined_what_if_simulation(
    n_clicks, react_pct, churn_red_pct, new_conv_pct, val_boost_pct, active_tab
):
    """Recalculates CLV based on refined intervention assumptions."""
    # Check if the simulation should run based on button click and active tab
    button_clicked = ctx.triggered_id == "simulate-button"
    # Only run if button clicked AND the strategy tab is active
    if not button_clicked or active_tab != "tab-strategy":
        # Return placeholder content if not triggered correctly
        return (
            [
                dbc.CardHeader("Simulated Total CLV", className="kpi-card-header"),
                dbc.CardBody("-", className="kpi-card-body"),
            ],
            [
                dbc.CardHeader("CLV Change (%)", className="kpi-card-header"),
                dbc.CardBody("-", className="kpi-card-body"),
            ],
            go.Figure(
                layout={
                    "title": "Run simulation to see impact",
                    "template": config.PLOTLY_TEMPLATE,
                }
            ),
        )

    print("Running Refined Simulation...")
    # Create a copy of the data to avoid modifying the original
    sim_clv_data = clv_data.copy()  # Work on a copy

    # Convert slider percentage values to decimal rates for calculations
    reactivation_rate = react_pct / 100.0
    churn_reduction_rate = churn_red_pct / 100.0
    new_conversion_clv_boost = new_conv_pct / 100.0
    value_boost_rate = val_boost_pct / 100.0

    # --- Apply Refined Simulation Logic ---

    # 1. Adjust 'prob_alive' for At Risk customers - increase their probability of being active
    at_risk_mask = sim_clv_data["Segment"] == "At Risk"
    # Apply reactivation rate to increase prob_alive, but cap at 95% to maintain realism
    sim_clv_data.loc[at_risk_mask, "prob_alive"] = np.minimum(
        sim_clv_data.loc[at_risk_mask, "prob_alive"] * (1 + reactivation_rate), 0.95
    )  # Cap at 95%

    # Adjust 'prob_alive' for Champions - reduce their churn probability
    champion_mask = sim_clv_data["Segment"] == "Champions"
    current_prob_alive_champ = sim_clv_data.loc[champion_mask, "prob_alive"]
    # Calculate potential increase based on the gap between current prob_alive and 100%
    potential_increase_champ = (
        1 - current_prob_alive_champ
    ) * churn_reduction_rate  # Reduce the gap to 100%
    # Apply the increase but cap at 100% (fully alive)
    sim_clv_data.loc[champion_mask, "prob_alive"] = np.minimum(
        current_prob_alive_champ + potential_increase_champ, 1.0
    )  # Cap at 100%

    # 2. Adjust 'avg_transaction_value' for targeted segments
    # Boost transaction value for both Champions and At Risk customers
    value_boost_mask = champion_mask | at_risk_mask  # Combine masks using OR operator
    # Apply percentage boost to average transaction value for these segments
    sim_clv_data.loc[value_boost_mask, "avg_transaction_value"] = sim_clv_data.loc[
        value_boost_mask, "avg_transaction_value"
    ] * (1 + value_boost_rate)
    # --- Recalculate Expected CLV based on modified prob_alive AND avg_transaction_value ---
    # The simulation uses a simplified scaling approach rather than re-running the full probabilistic models
    
    # Calculate scaling factors for both probability of being alive and average transaction value
    # Small epsilon value prevents division by zero for any customers with 0 values
    epsilon = 1e-9
    # Factor representing the relative change in probability of being alive
    prob_alive_factor = sim_clv_data["prob_alive"] / (clv_data["prob_alive"] + epsilon)
    # Factor representing the relative change in average transaction value
    avg_value_factor = sim_clv_data["avg_transaction_value"] / (
        clv_data["avg_transaction_value"] + epsilon
    )

    # Apply both scaling factors to the original predicted CLV to get simulated CLV
    # This approximation assumes CLV is proportional to prob_alive * avg_transaction_value
    sim_clv_data["simulated_clv"] = (
        clv_data["predicted_clv"] * prob_alive_factor * avg_value_factor
    )

    # 3. Apply direct CLV boost for 'New Customers' (representing faster progression through lifecycle)
    new_cust_mask = sim_clv_data["Segment"] == "New Customers"
    # Apply additional percentage boost to the already scaled CLV value for new customers
    sim_clv_data.loc[new_cust_mask, "simulated_clv"] = sim_clv_data.loc[
        new_cust_mask, "simulated_clv"
    ] * (1 + new_conversion_clv_boost)

    # Ensure all simulated CLV values are non-negative (prevent any negative values from calculations)
    sim_clv_data["simulated_clv"] = sim_clv_data["simulated_clv"].clip(
        lower=0
    )  # Ensure non-negative
    # --- Calculate KPIs & Update Figure ---
    # Calculate total CLV values and percentage change for KPI display
    original_total_clv = clv_data["predicted_clv"].sum()
    simulated_total_clv = sim_clv_data["simulated_clv"].sum()
    # Calculate percentage change, using epsilon to prevent division by zero
    clv_change_pct = (
        (simulated_total_clv - original_total_clv) / (original_total_clv + epsilon)
    ) * 100

    # Create updated treemap visualization using the simulated CLV values
    # Create a copy with renamed column to reuse the existing treemap function
    sim_treemap_df = sim_clv_data[["Segment", "simulated_clv"]].copy()
    # Rename column to match what the treemap function expects
    sim_treemap_df.rename(columns={"simulated_clv": "predicted_clv"}, inplace=True)
    # Generate the treemap using the existing function
    sim_treemap_fig = create_segment_treemap(
        sim_treemap_df
    )  # Pass the df with renamed column
    # Add a title to clarify this is the simulated result
    sim_treemap_fig.update_layout(title="Simulated Segment Contribution to Total CLV")
    
    # Add an alert to clarify the simulation limitations and assumptions
    simulation_alert = dbc.Alert(
        [
            html.H5("Simulation Assumptions and Limitations", className="alert-heading"),
            html.P(
                "This simulation uses a simplified scaling approach rather than re-running the full "
                "underlying probabilistic models. It applies percentage changes to key drivers (probability of "
                "being active and average transaction value) and scales CLV proportionally."
            ),
            html.P(
                "Important limitations to note:",
                className="mb-0"
            ),
            html.Ul([
                html.Li("The simulation assumes linear relationships between drivers and CLV, which may not fully capture complex interactions."),
                html.Li("It does not account for potential changes in purchase frequency patterns beyond what's reflected in probability of being active."),
                html.Li("Results should be interpreted as directional insights only, not precise forecasts."),
                html.Li("A true causal model would require controlled experiments or more sophisticated causal inference techniques.")
            ])
        ],
        color="info",
        className="mt-3"
    )
    
    # Add the alert to the treemap figure's layout
    sim_treemap_fig.update_layout(
        annotations=[
            dict(
                x=0.5,
                y=-0.15,
                xref="paper",
                yref="paper",
                text="Note: This simulation provides directional insights only and is not a full re-run of underlying models.",
                showarrow=False,
                font=dict(size=10, color="gray"),
                align="center"
            )
        ]
    )

    # Format KPI outputs for display in the dashboard
    # First KPI: Total simulated CLV value
    sim_kpi_clv_out = [
        dbc.CardHeader("Simulated Total CLV", className="kpi-card-header"),
        dbc.CardBody(f"£{simulated_total_clv:,.2f}", className="kpi-card-body"),
    ]
    # Second KPI: Percentage change in CLV with color coding
    sim_kpi_change_out = [
        dbc.CardHeader("CLV Change (%)", className="kpi-card-header"),
        dbc.CardBody(
            f"{clv_change_pct:+.2f}%",  # Use + sign for positive values
            # Apply color coding based on the direction and magnitude of change
            className=f"kpi-card-body {'text-success' if clv_change_pct > 0.1 else 'text-danger' if clv_change_pct < -0.1 else ''}", # Note: 'colour' attribute not standard for CardBody, using text-success/danger CSS classes
        ),
    ]  # Add color based on change

    return sim_kpi_clv_out, sim_kpi_change_out, sim_treemap_fig


# --- 7. Run the App ---
if __name__ == "__main__":
    print("--- Starting Dash Development Server (for local testing) ---")
    # Use debug=False for local testing closer to production behaviour if desired
    # Use host='0.0.0.0' to access from other devices on your network
    app.run(debug=False, port=8050)
    print("--- Dash Development Server Stopped ---")
