import pandas as pd
import numpy as np
import xgboost as xgb
from train_model import get_feature_columns

def predict_winner():
    # load model
    model = xgb.XGBClassifier()
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
    # compute logits
    logits = model.predict(df[feature_cols], output_margin=True)
    df["logit"] = logits
    # only keep essential rows
    cols_to_keep = ["manager", "season", "gw", "total_points", "logit"]
    df = df[cols_to_keep]
    # compute probabilities
    df["win_prob"] = (
        df.groupby("gw")["logit"]
        .transform(lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum())
    )
    df = df.sort_values(["gw", "win_prob"], ascending=[True, False])
    df.to_csv("output/predictions_current_season.csv")

def main():
    predict_winner()

if __name__ == "__main__":
    main()