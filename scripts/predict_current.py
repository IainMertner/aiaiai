import pandas as pd
import xgboost as xgb

from scripts.simulate_win_probs import simulate_win_probs

# Define output columns for predictions
OUTPUT_COLS = [
    "manager",
    "season",
    "gw",
    "total_points",
    "pred_remaining_points",
    "gw_rank",
    "win_probs"
]

# Predict current season using the trained model
def predict_current(feature_cols, tau):
    model = xgb.XGBRegressor()
    model.load_model("output/final_model.json")

    df = pd.read_csv("output/features.csv")

    current_seasons = (
        df.groupby("season")["gw"]
        .max()
        .loc[lambda s: s < 38]
        .index
        .tolist()
    )

    df = df[df["season"].isin(current_seasons)]

    preds = model.predict(df[feature_cols])
    df["pred_remaining_points"] = preds

    df = simulate_win_probs(df)
    df = (
        df[OUTPUT_COLS]
        .sort_values(["season", "gw", "win_probs"], 
                     ascending=[True, True, False])
    )
    
    df.to_csv(
        f"output/predictions/predictions_current_season.csv",
        index=False,
    )