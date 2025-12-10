import pandas as pd
import numpy as np
import xgboost as xgb
from train_model import get_feature_columns

def predict_current():
    # load model
    model = xgb.XGBRegressor()
    model.load_model("output/final_model.json")
    # load features
    df = pd.read_csv("output/features.csv")
    # get feature columns
    feature_cols = get_feature_columns(df)
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
    df.to_csv(f"output/predictions_current_season.csv", index=False)

def main():
    predict_current()

if __name__ == "__main__":
    main()