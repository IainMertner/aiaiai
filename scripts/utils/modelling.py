import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from scripts.utils.model_config import XGB_PARAMS

# Define output columns for predictions
OUTPUT_COLS = [
    "manager",
    "season",
    "gw",
    "total_points",
    "target_remaining_points",
    "pred_remaining_points",
    "residuals",
    "final_points",
    "pred_final_points",
    "gw_rank",
    "pred_rank"
]

# Train an XGBoost model
def train_xgb(df, feature_cols):
    X = df[feature_cols]
    y = df["target_remaining_points"]

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)

    return model

# Evaluate the model and return predictions, MAE, and R2 score
def evaluate(model, df, feature_cols):
    X = df[feature_cols]
    y = df["target_remaining_points"]

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    return preds, mae, r2

# Postprocess predictions to compute final points, ranks, and win probabilities
def postprocess_predictions(df, preds):
    out = df.copy()
    # Add predictions to the output DataFrame
    out["pred_remaining_points"] = preds
    out["residuals"] = out["target_remaining_points"] - out["pred_remaining_points"]
    out["pred_final_points"] = out["total_points"] + out["pred_remaining_points"]
    # Compute ranks based on actual and predicted final points
    out["pred_rank"] = (
        out.groupby(["season", "gw"])["pred_final_points"]
        .rank(ascending=False, method="min")
    )
    # Return only the relevant output columns, sorted appropriately
    return (
        out[OUTPUT_COLS]
        .sort_values(["season", "gw", "pred_final_points"],
                     ascending=[True, True, False])
    )