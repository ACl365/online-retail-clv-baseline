# config.py
"""
Configuration settings for the CLV Analysis Dashboard project.
Centralises filenames, model parameters, and dashboard defaults.
"""

import os

# --- File Paths ---
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # Gets the directory where config.py is located
DATA_FILE_NAME = "Online Retail.xlsx"
WRITE_UP_FILE_NAME = "clv-write_up_v1.md"
DATA_FILE_PATH = os.path.join(BASE_DIR, DATA_FILE_NAME)
WRITE_UP_FILE_PATH = os.path.join(BASE_DIR, WRITE_UP_FILE_NAME)

# --- Caching ---
CACHE_DIR_NAME = "cache"
CACHE_DIR_PATH = os.path.join(BASE_DIR, CACHE_DIR_NAME)
RFM_CACHE_FILE_NAME = "rfm_data.pkl"  # Consider renaming if it stores more now
CLV_CACHE_FILE_NAME = "clv_data.pkl"  # Stores combined RFM+CLV+Drivers
SNAPSHOT_CACHE_FILE_NAME = "snapshot_date.txt"
RFM_CACHE_PATH = os.path.join(CACHE_DIR_PATH, RFM_CACHE_FILE_NAME)
CLV_CACHE_PATH = os.path.join(CACHE_DIR_PATH, CLV_CACHE_FILE_NAME)
SNAPSHOT_CACHE_PATH = os.path.join(CACHE_DIR_PATH, SNAPSHOT_CACHE_FILE_NAME)

# --- Lifetimes Model Parameters ---
# Penalizer coefficients help prevent overfitting. 0.01 is a common default.
# Optimal values could be found via cross-validation on a larger/different dataset.
BGNBD_PENALIZER = 0.01
GAMMA_GAMMA_PENALIZER = 0.01
# Threshold for checking Gamma-Gamma correlation assumption (Frequency vs Monetary Value)
GAMMA_GAMMA_CORR_THRESHOLD = 0.1
# Prediction horizon for CLV calculation
CLV_PREDICTION_MONTHS = 12
# Monthly discount rate (e.g., 10% annual rate ~ 0.0083 monthly)
# (1 + annual_rate)^(1/12) - 1. For 10%, (1.1)^(1/12) - 1 = 0.00797
# Using 0.0083 as a common approximation.
MONTHLY_DISCOUNT_RATE = 0.0083
# Frequency unit used in lifetimes summary
LIFETIMES_FREQ = "D"  # Daily

# --- RFM Analysis Parameters ---
RFM_QUANTILES = 5

# --- Dashboard & Simulation Defaults ---
# Default values for the "What-If" simulation sliders (%)
DEFAULT_REACTIVATION_RATE_INCREASE = 10  # % increase in prob_alive for 'At Risk'
DEFAULT_CHURN_REDUCTION_RATE = (
    15  # % decrease in churn (not directly used in current simple sim, conceptual)
)
DEFAULT_NEW_CONVERSION_RATE_INCREASE = 20  # % CLV boost for 'New Customers'
DEFAULT_VALUE_BOOST_PCT = (
    5  # % increase in avg transaction value for 'Champions' & 'At Risk'
)

# --- Driver Analysis ---
# Bins for 'time_to_second_purchase' visualisation
# Adjust these based on data distribution if needed
TIME_TO_SECOND_PURCHASE_BINS = [0, 30, 60, 90, 180, 365]

# --- Plotting ---
PLOTLY_TEMPLATE = "plotly_white"  # e.g., 'plotly', 'plotly_white', 'ggplot2'

# --- Other ---
# Ensure cache directory exists
if not os.path.exists(CACHE_DIR_PATH):
    os.makedirs(CACHE_DIR_PATH)
    print(f"Created cache directory: {CACHE_DIR_PATH}")
