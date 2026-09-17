"""
Train Random Forest Regressor for TCO Material Recommendation System.
This script loads the demonstration dataset, preprocesses it, trains a
RandomForestRegressor, evaluates it, saves the model and scaler, and
generates basic plots for feature importance and actual vs predicted.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tco_materials.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Feature columns (all numerical properties)
FEATURE_COLS = [
    "Electrical_Conductivity",
    "Optical_Transparency",
    "Thermal_Stability",
    "Chemical_Stability",
    "Mercury_Corrosion_Resistance",
    "Adhesion_Strength",
    "Durability",
    "Cost",
]
TARGET_COL = "Suitability_Score"


def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load CSV and perform basic cleaning."""
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())

    # Handle missing values if any (none expected in demo data)
    if df.isnull().any().any():
        # For numerical columns fill with median; material name should not be missing
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        print("Filled missing numerical values with median.")

    # Ensure Material is string
    df["Material"] = df["Material"].astype(str)
    return df


def preprocess(df: pd.DataFrame):
    """Separate features/target, scale features, split train/test."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Scale numerical features (important for distance-based ranking later)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLS, index=X.index)

    # Reproducible split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )
    return X, y, X_scaled, X_train, X_test, y_train, y_test, scaler


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    """Train RandomForestRegressor with reasonable defaults."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,          # small dataset → shallow trees
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, X_scaled, y):
    """Compute and print metrics; return predictions."""
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X_scaled)

    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("\n=== Model Evaluation (Test Set) ===")
    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    # Full-set metrics for reference (small data)
    r2_all = r2_score(y, y_pred_all)
    print(f"\nFull-set R² (for reference on tiny demo set): {r2_all:.4f}")
    return y_pred_test, y_pred_all, r2, mae, rmse


def plot_feature_importance(model, feature_names, save_path: str):
    """Bar chart of Random Forest feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], color="steelblue")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.ylabel("Feature Importance")
    plt.title("Random Forest Feature Importance – TCO Suitability")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to {save_path}")


def plot_actual_vs_predicted(y_true, y_pred, save_path: str):
    """Scatter of actual vs predicted suitability scores."""
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, c="darkorange", edgecolors="k", s=80)
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    plt.plot(lims, lims, "k--", lw=2, label="Ideal")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual Suitability Score")
    plt.ylabel("Predicted Suitability Score")
    plt.title("Actual vs Predicted Suitability (Full Demo Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Actual vs Predicted plot saved to {save_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading demonstration/synthetic dataset...")
    df = load_and_clean_data(DATA_PATH)

    print("\nPreprocessing...")
    X, y, X_scaled, X_train, X_test, y_train, y_test, scaler = preprocess(df)

    print("Training RandomForestRegressor...")
    model = train_random_forest(X_train, y_train)

    print("Evaluating...")
    y_pred_test, y_pred_all, r2, mae, rmse = evaluate(
        model, X_test, y_test, X_scaled, y
    )

    # Save model & scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

    # Plots
    fi_path = os.path.join(MODEL_DIR, "feature_importance.png")
    avp_path = os.path.join(MODEL_DIR, "actual_vs_predicted.png")
    plot_feature_importance(model, FEATURE_COLS, fi_path)
    plot_actual_vs_predicted(y, y_pred_all, avp_path)

    # Show feature importances numerically
    print("\nFeature Importances:")
    for name, imp in sorted(
        zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {name:30s}: {imp:.4f}")

    print("\nTraining complete. Demo dataset is synthetic – replace with literature data when available.")


if __name__ == "__main__":
    main()
