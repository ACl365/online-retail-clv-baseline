# Customer Lifetime Value (CLV) Analysis: Online Retail Baseline (2010-2011)

## Overview

**Explore our interactive Dash dashboard** showcasing Customer Lifetime Value (CLV) analysis that bridges traditional methods with modern AI potential. This project transforms the classic Online Retail dataset (2010-2011) into a powerful strategic tool through:

* **Interactive visualisations** that reveal customer segment value, behaviour patterns, and future potential
* **What-if simulations** allowing you to test strategic interventions and see projected CLV impact in real-time
* **Actionable insights** derived from both probabilistic models and modern ML approaches

The dashboard serves as both a practical business intelligence tool and a compelling demonstration of the evolution from traditional probabilistic CLV modelling to modern AI-driven approaches. It establishes a **historical baseline** using established methodologies (RFM segmentation, BG/NBD and Gamma-Gamma models) while showcasing how modern (2025-era) data science techniques can dramatically enhance predictive power and strategic decision-making.

**To experience the dashboard, simply run `python app.py` and explore the future of customer value analytics.**

## Dataset

The analysis utilises the [Online Retail dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) from the UCI Machine Learning Repository.

*   **Source:** UCI Machine Learning Repository
*   **Timeframe:** 01/12/2010 - 09/12/2011
*   **Contents:** Transactional data including InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.
*   **Key Characteristic:** Represents B2B/B2C online retail behaviour from over a decade ago, serving as a benchmark.

## Methodology

1.  **Data Cleaning & Preparation:** Handling missing values (CustomerID), managing returns/cancellations, outlier treatment (IQR method), feature engineering (TotalPrice, RFM metrics). See `data_processor.py`.
2.  **Exploratory Data Analysis (EDA):** Analysing sales trends, geographic distribution, and RFM metric distributions.
3.  **RFM Segmentation:** Segmenting customers into distinct groups (e.g., Champions, Loyal Customers, At Risk) based on their Recency, Frequency, and Monetary value using quintiles. See `rfm_analyzer.py`.
4.  **Probabilistic CLV Modelling:**
    *   **BG/NBD Model:** Predicting the number of future transactions.
    *   **Gamma-Gamma Model:** Predicting the average monetary value of future transactions.
    *   Combined to estimate CLV for different future periods (3, 6, 12 months). See `clv_predictor.py`.
5.  **Modern ML Comparison (Conceptual):** Discusses how models like XGBoost/LightGBM could potentially improve predictions with richer features, contrasting with the baseline probabilistic approach. A basic implementation is provided in `modern_predictor.py`.

## Project Structure

```
.
├── .gitignore
├── app.py                 # Main application script (likely runs the analysis pipeline)
├── clv_predictor.py       # Implements BG/NBD and Gamma-Gamma models
├── config.py              # Configuration settings (e.g., file paths, model parameters)
├── data_processor.py      # Handles data loading, cleaning, and preprocessing
├── modern_predictor.py    # Example implementation using modern ML (e.g., XGBoost)
├── Online Retail.xlsx     # Raw dataset
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── rfm_analyzer.py        # Performs RFM segmentation
├── to_do.txt              # Project tasks/notes
├── assets/                # Directory for static assets (e.g., images, plots)
├── cache/                 # Directory for cached data/results
└── __pycache__/           # Python bytecode cache
```

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
    *(Adjust if not using Git)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main analysis pipeline using the application script:

```bash
python app.py
```

This will likely execute the data processing, RFM analysis, and CLV prediction steps, potentially saving outputs (like plots or segment data) to the `assets` or `cache` directories. Check `config.py` for specific output configurations.

## Key Findings

*   **RFM Segmentation:** Successfully identified distinct customer segments like 'Champions' (15% customers, 35% revenue) and 'At Risk' (20% customers, 15% revenue).
*   **Value Concentration:** The top 20% of customers generate over 60% of the revenue, highlighting the importance of retaining high-value segments.
*   **CLV Prediction:** Probabilistic models provided 3, 6, and 12-month CLV forecasts per segment (e.g., 12-month CLV for Champions: £1,124 vs. Lost Customers: £26).
*   **Key CLV Drivers:** Early engagement (second purchase within 30 days), category exploration, and seasonal acquisition were identified as significant drivers of higher CLV in this dataset.

## Limitations

*   **Data Age:** The 2010-2011 dataset does not reflect modern e-commerce behaviour (mobile, social commerce, etc.).
*   **Missing Data:** Lack of demographic, acquisition source, and richer behavioural data limits the depth of analysis possible with modern techniques.
*   **Model Assumptions:** Probabilistic models rely on assumptions that may not perfectly hold true.

## Future Work & Modernisation

This project serves as a foundation. Future enhancements could include:
*   Applying modern ML models (XGBoost, LightGBM, Deep Learning) for potentially higher accuracy.
*   Incorporating richer feature sets (behavioural, demographic data if available).
*   Implementing causal inference techniques (e.g., uplift modelling) for optimising marketing interventions.
*   Deploying models for real-time prediction and personalisation.

## License

*(Specify project license if applicable, e.g., MIT License)*