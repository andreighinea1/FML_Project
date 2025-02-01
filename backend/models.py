import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Suppress warnings globally
warnings.filterwarnings("ignore")


def objective(trial, model_name, X_train, y_train, X_test, y_test):
    """
    Objective function for hyperparameter tuning with Optuna.
    Optimizes based only on `sp500_next_1` (1-day ahead prediction).
    """
    # Use only the first prediction target (sp500_next_1)
    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]

    if model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 750, 1000),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 6, 12),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-4, 0.1),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-4, 0.1),
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 750, 1000),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "gamma": trial.suggest_loguniform("gamma", 1e-4, 1.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-4, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-4, 1.0),
            "enable_categorical": True,
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)
    else:
        raise ValueError("Unsupported model!")

    # Train model (suppress output)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Compute MSE only for 1-day ahead prediction
    mse = mean_squared_error(y_test, y_pred)

    return mse  # Optuna minimizes this value


def prepare_data(training_df, days_to_predict):
    """Creates shifted target columns and prepares train/test splits."""
    df = training_df.copy()

    # Step 1: Create future target columns (shifting by respective days)
    next_day_columns = []
    for day in days_to_predict:
        next_day_col = f"sp500_next_{day}"
        df[next_day_col] = df["sp500"].shift(-day)
        next_day_columns.append(next_day_col)

    # Drop rows with NaN values
    df = df.dropna()

    # Step 2: Define features (X) and targets (y)
    X = df.drop(columns=next_day_columns)
    y = df[next_day_columns]

    # Step 3: Perform time-based train-test split
    split_index = int(0.8 * len(X))  # 80% for training
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test


def train_model(model_name, X_train, y_train, X_test, **kwargs):
    """Trains a model for each target column if multi-output, else trains a single model."""

    # Model mapping
    model_classes = {
        "LinearRegression": LinearRegression,
        "LightGBM": lgb.LGBMRegressor,
        "XGBoost": xgb.XGBRegressor,
    }

    if model_name not in model_classes:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name == "XGBoost":
        kwargs["enable_categorical"] = True

    # Handle multi-output case
    multi_output = len(y_train.shape) > 1 and y_train.shape[1] > 1
    models, y_pred_list = [], []

    target_columns = y_train.columns if multi_output else [y_train.name]

    for target in target_columns:
        print(f"\nTraining {model_name} for target: {target}...\n")

        # Initialize model
        model = model_classes[model_name](**kwargs)

        # Train model
        model.fit(X_train, y_train[target])

        # Predict
        y_pred = model.predict(X_test)

        models.append(model)
        y_pred_list.append(y_pred)

    # Stack predictions for multiple outputs
    y_pred = np.column_stack(y_pred_list) if multi_output else y_pred_list[0]

    # Returns a list of models if multi-output, single model otherwise
    return models, y_pred


def evaluate_model(model_name, y_test, y_pred, days_to_predict):
    """Calculates MSE and R² for each prediction horizon and returns results."""
    results = {}
    for i, day in enumerate(days_to_predict):
        # Align predictions by shifting
        y_pred_aligned = (
            pd.Series(y_pred[:, i], index=y_test.index).shift(-day).dropna()
        )
        y_test_aligned = y_test.iloc[:-day, i]  # Align ground truth

        # Compute metrics
        mse = mean_squared_error(y_test_aligned, y_pred_aligned)
        r2 = r2_score(y_test_aligned, y_pred_aligned)

        results[day] = {"MSE": mse, "R²": r2}
        print(f"{model_name} - Day {day}: MSE={mse:.4f}, R²={r2:.4f}")

    return results


def plot_predictions(y_test, y_pred, days_to_predict, results, model_name):
    """Plots actual vs predicted values for each prediction horizon."""
    for i, day in enumerate(days_to_predict):
        y_pred_aligned = (
            pd.Series(y_pred[:, i], index=y_test.index).shift(-day).dropna()
        )

        # Extract stored metrics
        mse = results[day]["MSE"]
        r2 = results[day]["R²"]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.iloc[:, i].values, label=f"Actual Day {day}", alpha=0.8)
        plt.plot(
            y_pred_aligned.values,
            label=f"Predicted Day {day}",
            linestyle="--",
            alpha=0.8,
        )

        # Add metrics text box
        textstr = f"MSE: {mse:.4f}\nR²: {r2:.4f}"
        plt.text(
            0.05,
            0.85,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
        )

        plt.title(f"{model_name} - Actual vs Predicted for Day {day}")
        plt.xlabel("Test Samples")
        plt.ylabel("S&P 500 Index")
        plt.legend()
        plt.grid()
        plt.show()
