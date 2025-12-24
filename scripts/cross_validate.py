import pandas as pd

from scripts.utils.modelling import train_xgb, evaluate, postprocess_predictions
from scripts.utils.resource_path import resource_path

# Cross-validate the model across completed seasons
def cross_validate(feature_cols):
    df = pd.read_csv(resource_path("output/features.csv"))

    completed_seasons = (
        df.groupby("season")["gw"]
        .max()
        .loc[lambda s: s == 38]
        .index
        .tolist()
    )

    results = []

    # Cross-validation loop
    for val_season in completed_seasons:
        # Prepare training and validation data
        train_df = df[
            (df["season"] != val_season) &
            (df["season"].isin(completed_seasons))
        ]
        val_df = df[df["season"] == val_season]
        # Train and evaluate model
        model = train_xgb(train_df, feature_cols)
        preds, mae, r2 = evaluate(model, val_df, feature_cols)
        # Postprocess and save validation predictions
        val_out = postprocess_predictions(val_df, preds)
        val_out.to_csv(
            resource_path(f"output/predictions/val_predictions_season_{val_season}.csv"),
            index=False,
        )

        results.append((val_season, mae, r2))

    print(results)
