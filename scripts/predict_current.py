import pandas as pd
import xgboost as xgb

from scripts.utils.modelling import postprocess_predictions

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

    out = postprocess_predictions(df, preds, tau)
    out.to_csv(
        "output/predictions/predictions_current_season.csv",
        index=False,
    )