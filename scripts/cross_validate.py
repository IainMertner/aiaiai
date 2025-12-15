import pandas as pd

from scripts.utils.modelling import train_xgb, evaluate, postprocess_predictions


def cross_validate(feature_cols, tau):
    df = pd.read_csv("output/features.csv")

    completed_seasons = (
        df.groupby("season")["gw"]
        .max()
        .loc[lambda s: s == 38]
        .index
        .tolist()
    )

    results = []

    for val_season in completed_seasons:
        train_df = df[
            (df["season"] != val_season) &
            (df["season"].isin(completed_seasons))
        ]
        val_df = df[df["season"] == val_season]

        model = train_xgb(train_df, feature_cols)
        preds, mae, r2 = evaluate(model, val_df, feature_cols)

        val_out = postprocess_predictions(val_df, preds, tau)
        val_out.to_csv(
            f"output/predictions/val_predictions_season_{val_season}.csv",
            index=False,
        )

        results.append((val_season, mae, r2))

    print(results)
