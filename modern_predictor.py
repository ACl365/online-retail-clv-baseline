# modern_predictor.py
# Placeholder for modern ML model implementation (XGBoost/LightGBM)
# based on RFM features and potentially engineered features.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import joblib  # Or pickle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --- Configuration ---
MODEL_TYPE = "xgboost"  # 'xgboost' or 'lightgbm'

# Target variable definition and justification
# We use the probabilistic CLV from the Gamma-Gamma model as our target variable.
# This allows us to:
# 1. Compare modern ML approaches directly with traditional probabilistic methods
# 2. Leverage the domain knowledge encoded in the probabilistic model
# 3. Potentially improve predictions by incorporating additional features beyond RFM
TARGET_VARIABLE = "predicted_clv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_OUTPUT_PATH = "cache/modern_model.joblib"
FEATURES_PATH = "cache/rfm_results.pkl"  # Assuming RFM results are the base features
CLV_RESULTS_PATH = (
    "cache/clv_results.pkl"  # To potentially use Gamma-Gamma CLV as target
)
# Assuming data_processor.py saves cleaned transactions here
CLEANED_TRANSACTIONS_PATH = "cache/cleaned_transactions.pkl"


# --- Feature Engineering ---
def engineer_features(rfm_df, transactions_df):
    """
    Add more sophisticated features beyond basic RFM using transaction data.
    
    This function creates advanced features that capture customer behavior patterns
    beyond the standard RFM metrics, potentially improving predictive power.
    """
    print("Starting feature engineering...")

    # Ensure InvoiceDate is datetime
    transactions_df['InvoiceDate'] = pd.to_datetime(transactions_df['InvoiceDate'])

    # Calculate time between purchases
    transactions_df = transactions_df.sort_values(['CustomerID', 'InvoiceDate'])
    transactions_df['TimeDiff'] = transactions_df.groupby('CustomerID')['InvoiceDate'].diff().dt.days
    # Fill NaN for first purchase, maybe with a large value or mean/median? Using 0 for now.
    transactions_df['TimeDiff'].fillna(0, inplace=True)

    # Get first and last purchase dates for each customer
    customer_purchase_dates = transactions_df.groupby('CustomerID')['InvoiceDate'].agg(['min', 'max'])
    customer_purchase_dates.columns = ['FirstPurchase', 'LastPurchase']
    
    # Calculate days since first purchase and customer tenure
    max_date = transactions_df['InvoiceDate'].max()
    customer_purchase_dates['DaysSinceFirstPurchase'] = (max_date - customer_purchase_dates['FirstPurchase']).dt.days
    customer_purchase_dates['CustomerTenure'] = (customer_purchase_dates['LastPurchase'] - customer_purchase_dates['FirstPurchase']).dt.days
    
    # FEATURE 1: Calculate purchase frequency in last 3 months (90 days)
    three_months_ago = max_date - pd.Timedelta(days=90)
    recent_transactions = transactions_df[transactions_df['InvoiceDate'] >= three_months_ago]
    purchase_count_last_3months = recent_transactions.groupby('CustomerID')['InvoiceDate'].nunique().reset_index()
    purchase_count_last_3months.columns = ['CustomerID', 'PurchaseCountLast3Months']

    # Aggregate features at customer level
    customer_features = transactions_df.groupby('CustomerID').agg(
        MeanTimeBetweenPurchases=('TimeDiff', 'mean'),
        StdTimeBetweenPurchases=('TimeDiff', 'std'),
        AvgOrderQuantity=('Quantity', 'mean'),
        TotalQuantity=('Quantity', 'sum'),
        UniqueProducts=('StockCode', 'nunique')
    )
    
    # FEATURE 2: Calculate order value standard deviation (purchase amount volatility)
    # First calculate total price per invoice
    invoice_totals = transactions_df.groupby(['CustomerID', 'InvoiceNo'])['TotalPrice'].sum().reset_index()
    # Then calculate standard deviation of invoice totals per customer
    order_value_std = invoice_totals.groupby('CustomerID')['TotalPrice'].std().reset_index()
    order_value_std.columns = ['CustomerID', 'AvgOrderValueStdDev']
    
    # FEATURE 3: Calculate category diversity index
    # Count distinct categories per customer and normalize by total purchases
    # This measures how diverse a customer's purchasing behavior is
    if 'StockCode' in transactions_df.columns:
        total_purchases = transactions_df.groupby('CustomerID').size()
        unique_categories = transactions_df.groupby('CustomerID')['StockCode'].nunique()
        category_diversity = (unique_categories / total_purchases).reset_index()
        category_diversity.columns = ['CustomerID', 'CategoryDiversityIndex']
        # Cap at 1.0 for any calculation issues
        category_diversity['CategoryDiversityIndex'] = category_diversity['CategoryDiversityIndex'].clip(upper=1.0)
    else:
        # Create empty DataFrame if StockCode not available
        category_diversity = pd.DataFrame(columns=['CustomerID', 'CategoryDiversityIndex'])

    # Fill NaN for StdDev if customer has only one purchase (std is NaN)
    customer_features['StdTimeBetweenPurchases'].fillna(0, inplace=True)

    # Merge all feature DataFrames with RFM data
    rfm_df_engineered = rfm_df.join(customer_features, how='left')
    rfm_df_engineered = rfm_df_engineered.merge(customer_purchase_dates, left_index=True, right_index=True, how='left')
    rfm_df_engineered = rfm_df_engineered.merge(purchase_count_last_3months, left_index=True, right_on='CustomerID', how='left')
    rfm_df_engineered = rfm_df_engineered.merge(order_value_std, left_index=True, right_on='CustomerID', how='left')
    rfm_df_engineered = rfm_df_engineered.merge(category_diversity, left_index=True, right_on='CustomerID', how='left')
    
    # Set index back to CustomerID if it was reset during merges
    if rfm_df_engineered.index.name != 'CustomerID':
        # Drop duplicate CustomerID column if it exists
        if 'CustomerID' in rfm_df_engineered.columns:
            rfm_df_engineered = rfm_df_engineered.set_index('CustomerID')

    # Use scikit-learn's SimpleImputer for more robust NaN handling
    from sklearn.impute import SimpleImputer
    
    # Identify numeric columns for imputation
    numeric_cols = rfm_df_engineered.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Create and fit the imputer
    imputer = SimpleImputer(strategy='median')
    
    # Apply imputation to numeric columns only
    if numeric_cols:
        rfm_df_engineered[numeric_cols] = imputer.fit_transform(rfm_df_engineered[numeric_cols])
    
    print(f"Feature engineering complete. Created {len(rfm_df_engineered.columns)} features.")
    return rfm_df_engineered


# --- Data Loading and Preparation ---
def load_and_prepare_data():
    """
    Loads RFM features, transaction data, and potentially the target variable.
    Performs feature engineering and prepares data for modeling.
    """
    print("Loading data...")
    try:
        rfm_results = pd.read_pickle(FEATURES_PATH)
        clv_results = pd.read_pickle(CLV_RESULTS_PATH)
        # Load cleaned transaction data for feature engineering
        transactions = pd.read_pickle(CLEANED_TRANSACTIONS_PATH)
        print(f"Loaded {len(transactions)} transactions.")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Ensure prior steps (RFM, CLV, data processing) have run.")
        print(f"Specifically check for: {FEATURES_PATH}, {CLV_RESULTS_PATH}, {CLEANED_TRANSACTIONS_PATH}")
        return None, None, None # Return None for features as well

    # --- Feature Engineering Step ---
    # Pass both RFM results and transaction data
    rfm_engineered = engineer_features(rfm_results, transactions)

    # --- Target Variable Definition ---
    if TARGET_VARIABLE in clv_results.columns:
        data = rfm_engineered.join(clv_results[TARGET_VARIABLE], how="inner")
        print(f"Using '{TARGET_VARIABLE}' from CLV results as target.")
        if data.empty:
             print("Error: Join resulted in an empty DataFrame. Check indices of RFM and CLV results.")
             return None, None, None
    else:
        print(f"Target variable '{TARGET_VARIABLE}' not found in CLV results.")
        print("Halting: Cannot proceed without a defined target variable.")
        return None, None, None # Halt if target is missing

    # Define features (X) and target (y)
    base_features = ["Recency", "Frequency", "MonetaryValue"]
    engineered_features_list = [
        'MeanTimeBetweenPurchases', 'StdTimeBetweenPurchases',
        'AvgOrderQuantity', 'TotalQuantity', 'UniqueProducts'
    ]
    # Ensure engineered features actually exist in the dataframe after join/fillna
    final_feature_cols = base_features + [
        feat for feat in engineered_features_list if feat in data.columns
    ]

    # Check for missing features
    missing_features = [f for f in final_feature_cols if f not in data.columns]
    if missing_features:
        print(f"Warning: The following features are missing from the data: {missing_features}")
        final_feature_cols = [f for f in final_feature_cols if f in data.columns]

    if not final_feature_cols:
        print("Error: No features available for training.")
        return None, None, None

    X = data[final_feature_cols]
    y = data[TARGET_VARIABLE]

    # Handle potential infinities or large values if necessary
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Option 1: Fill NaNs resulting from inf replacement or other issues
    if X.isnull().values.any():
        print("Warning: NaNs found in features. Filling with median.")
        X = X.fillna(X.median())
    # Option 2: Drop rows with NaNs (might lose data)
    # combined = pd.concat([X, y], axis=1).dropna()
    # X = combined[final_feature_cols]
    # y = combined[TARGET_VARIABLE]


    print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Features used: {final_feature_cols}")
    return X, y, final_feature_cols # Return feature list for evaluation


# --- Model Training ---
def train_model(X_train, y_train):
    """
    Trains the specified ML model with reasonable default hyperparameters.
    
    Note on hyperparameter tuning:
    The hyperparameters used here (n_estimators, max_depth, learning_rate, etc.)
    are reasonable defaults that should work well for many problems. For optimal
    performance, these parameters should be tuned using techniques like:
    
    1. GridSearchCV - Exhaustive search over specified parameter values
    2. RandomizedSearchCV - Random search over parameter distributions
    3. Bayesian optimization (e.g., using libraries like Optuna or Hyperopt)
    
    A typical tuning process would involve:
    - Defining parameter search spaces
    - Using k-fold cross-validation to evaluate parameter combinations
    - Selecting the best parameters based on validation metrics
    - Refitting the model with the optimal parameters on the full training set
    """
    print(f"Training {MODEL_TYPE} model...")
    if MODEL_TYPE == "xgboost":
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,  # Could be tuned in range [50, 500]
            learning_rate=0.1,  # Could be tuned in range [0.01, 0.3]
            max_depth=5,        # Could be tuned in range [3, 10]
            subsample=0.8,      # Could be tuned in range [0.6, 1.0]
            colsample_bytree=0.8, # Could be tuned in range [0.6, 1.0]
            random_state=RANDOM_STATE,
            n_jobs=-1,
            early_stopping_rounds=10
        )
        # Use eval_set for early stopping
        # Need a validation set split from training data for this
        # For simplicity here, we'll skip the eval_set, but it's recommended
        # model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        model.fit(X_train, y_train, verbose=False)

    elif MODEL_TYPE == "lightgbm":
        model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=100,    # Could be tuned in range [50, 500]
            learning_rate=0.1,   # Could be tuned in range [0.01, 0.3]
            num_leaves=31,       # Could be tuned in range [20, 100]
            max_depth=-1,        # -1 means no limit
            min_child_samples=20, # Could be tuned in range [10, 50]
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train) # Add callbacks=[lgb.early_stopping(10)] if using validation set
    else:
        raise ValueError("Unsupported MODEL_TYPE specified in config.")

    print("Model training complete.")
    return model


# --- Evaluation ---
def evaluate_model(model, X_test, y_test, feature_names, y_train=None):
    """Evaluates the trained model on the test set and compares to baseline."""
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"Evaluation Metrics ({MODEL_TYPE}):")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    # Calculate baseline metrics (predicting mean CLV)
    # If y_train is not provided, use y_test mean as a fallback
    baseline_mean = y_train.mean() if y_train is not None else y_test.mean()
    y_pred_baseline = np.full_like(y_test, baseline_mean)
    baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
    baseline_rmse = mean_squared_error(y_test, y_pred_baseline, squared=False)
    
    # Compare against baseline
    print("\nComparison against baseline (predicting mean CLV):")
    print(f"  Baseline MAE: {baseline_mae:.4f}")
    print(f"  Baseline RMSE: {baseline_rmse:.4f}")
    print(f"  MAE Improvement: {(baseline_mae - mae) / baseline_mae * 100:.2f}%")
    print(f"  RMSE Improvement: {(baseline_rmse - rmse) / baseline_rmse * 100:.2f}%")
    
    # Compare against probabilistic model (conceptual)
    print("\nNote: For a full comparison with the probabilistic model (BG/NBD + Gamma-Gamma),")
    print("      we would need to evaluate both models on the same holdout period.")
    print("      The probabilistic model typically achieves RMSE of ~15-20% of mean CLV.")

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        try:
            # Use the passed feature_names list which corresponds to X_test columns
            feature_importance = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            print("\nFeature Importances:")
            print(feature_importance.head(10))
            
            # Print insights about top features
            print("\nInsights from top features:")
            top_features = feature_importance.head(3)['feature'].tolist()
            for feature in top_features:
                print(f"  - {feature} is a strong predictor of future customer value")
                
        except Exception as e:
            print(f"Could not retrieve/display feature importances: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Modern Predictor Script ---")
    X, y, feature_cols = load_and_prepare_data() # Get feature names back

    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(
            f"Train/Test split: {X_train.shape[0]} train, {X_test.shape[0]} test samples."
        )

        # Ensure feature names are passed correctly if using DataFrames
        model = train_model(X_train, y_train)
        # Pass feature names used in training to evaluation, along with y_train for baseline comparison
        evaluate_model(model, X_test, y_test, feature_cols, y_train)

        # Save the trained model
        print(f"Saving model to {MODEL_OUTPUT_PATH}...")
        joblib.dump(model, MODEL_OUTPUT_PATH)
        print("Model saved.")

        print("\n--- Modern Predictor Script Finished ---")
    else:
        print("Halting script due to data loading/preparation issues.")
