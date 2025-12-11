import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from scripts.utils.bayesian_shrinkage import apply_bayesian_shrinkage

def train_model(feature_cols):
    # load features
    df = pd.read_csv("output/features.csv")
    # identify completed seasons
    completed_seasons = (
        df.groupby("season")["gw"]
        .max()
        .loc[lambda s: s == 38]
        .index
        .tolist()
    )

    ## train final model on all completed seasons
    train_df = df[df["season"].isin(completed_seasons)]
    X_train = train_df[feature_cols]
    y_train = train_df["target_remaining_points"]
    ## final model
    final_model = XGBRegressor(
        n_estimators = 500,
        learning_rate = 0.03,
        max_depth = 5,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = "reg:squarederror"
    )
    # fit model
    final_model.fit(X_train, y_train)
    final_model.save_model("output/final_model.json")