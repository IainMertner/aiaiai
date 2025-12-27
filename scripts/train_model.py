import pandas as pd
from pathlib import Path

from scripts.utils.modelling import train_xgb
from scripts.utils.resource_path import resource_path

# Train final model on all completed seasons and save it
def train_model(feature_cols):
    df = pd.read_csv(resource_path("output/features.csv"))

    completed_seasons = (
        df.groupby("season")["gw"]
        .max()
        .loc[lambda s: s == 38]
        .index
        .tolist()
    )

    train_df = df[df["season"].isin(completed_seasons)]

    ensemble = train_xgb(train_df, feature_cols)
    path = Path(resource_path("output/models"))
    path.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(ensemble.models):
        model.save_model(resource_path(path/f"model_{i}.json"))

    return ensemble