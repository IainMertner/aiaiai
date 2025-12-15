import pandas as pd

from scripts.utils.modelling import train_xgb

# Train final model on all completed seasons and save it
def train_model(feature_cols):
    df = pd.read_csv("output/features.csv")

    completed_seasons = (
        df.groupby("season")["gw"]
        .max()
        .loc[lambda s: s == 38]
        .index
        .tolist()
    )

    train_df = df[df["season"].isin(completed_seasons)]

    model = train_xgb(train_df, feature_cols)
    model.save_model("output/final_model.json")