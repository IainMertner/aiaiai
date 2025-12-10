import pandas as pd
import numpy as np
import xgboost as xgb

def predict_current(feature_cols):
    # load model
    model = xgb.XGBRegressor()
    model.load_model("output/final_model.json")
    # load features
    df = pd.read_csv("output/features.csv")
    # get feature columns
    # identify current season
    current_season = (
        df.groupby("season")["gw"]
        .max()
        .loc[lambda s: s < 38]
        .index
        .tolist()
    )
    df = df[df["season"].isin(current_season)]
    # predict remaining points
    preds = model.predict(df[feature_cols])
    df["pred_remaining_points"] = preds
    # predicted final points
    df["pred_final_points"] = df["total_points"] + df["pred_remaining_points"]
    # predicted rank
    df["pred_rank"]= (
        df.groupby(["season", "gw"])["pred_final_points"]
        .rank(ascending=False, method="min")
    )
    # win probability
    df["win_prob"] = (
        df.groupby(["season", "gw"])["pred_final_points"]
        .transform(lambda x: np.exp(x/50 - (x/50).max()) / np.exp(x/50 - (x/50).max()).sum())
    )
    # only save most important columns
    cols_to_keep = ["manager", "season", "gw", "total_points", "pred_remaining_points", "final_points", "pred_final_points", "gw_rank", "pred_rank", "win_prob"]
    df = df[cols_to_keep].sort_values(["season", "gw", "pred_final_points"], ascending=[True, True, False])
    df.to_csv(f"output/predictions/predictions_current_season.csv", index=False)