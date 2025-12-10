import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

## get feature columns (all numeric columns except labels and identifiers)
def get_feature_columns(df):
    drop_cols = [
        "final_points",
        "target_remaining_points",
        "season",
        "manager"
    ]
    feature_cols = [col for col in df.columns if col not in drop_cols and np.issubdtype(df[col].dtype, np.number)]
    
    return feature_cols

### train one fold (one train/validation combo)
def train_one_fold(train_df, val_df, feature_cols, val_season):
    # train/val split
    X_train = train_df[feature_cols]
    y_train = train_df["target_remaining_points"]
    X_val = val_df[feature_cols]
    y_val = val_df["target_remaining_points"]

    ## XGBoost model
    model = XGBRegressor(
        n_estimators = 500,
        learning_rate = 0.03,
        max_depth = 5,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = "reg:squarederror",
        eval_metric = "rmse"
    )
    # fit model
    model.fit(X_train, y_train)
    # predict final points
    preds = model.predict(X_val)
    # eval metrics
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    ## save predictions
    val_out = val_df.copy()
    val_out["pred_remaining_points"] = preds
    # predicted final points
    val_out["pred_final_points"] = val_out["total_points"] + val_out["pred_remaining_points"]
    # predicted rank
    val_out["pred_rank"]= (
        val_out.groupby(["season", "gw"])["pred_final_points"]
        .rank(ascending=False, method="min")
    )
    # win probability
    val_out["win_prob"] = (
        val_out.groupby(["season", "gw"])["pred_final_points"]
        .transform(lambda x: np.exp(x/50 - (x/50).max()) / np.exp(x/50 - (x/50).max()).sum())
    )
    # only save most important columns
    cols_to_keep = ["manager", "season", "gw", "total_points", "pred_remaining_points", "final_points", "pred_final_points", "gw_rank", "pred_rank", "win_prob"]
    val_out = val_out[cols_to_keep].sort_values(["season", "gw", "pred_final_points"], ascending=[True, True, False])
    val_out.to_csv(f"output/val_predictions_season_{val_season}.csv", index=False)

    return model, mae, r2

def main():
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
    # get feature columns
    feature_cols = get_feature_columns(df)
    print(feature_cols)

    results = []
    ## season-level cross-validation
    for val_season in completed_seasons:
        # split into training and validation sets
        train_seasons = [season for season in completed_seasons if season != val_season]
        train_df = df[df["season"].isin(train_seasons)]
        val_df = df[df["season"] == val_season]
        # train model
        model, mae, r2 = train_one_fold(train_df, val_df, feature_cols, val_season)
        # store results
        results.append((val_season, mae, r2))
    
    print(results)

    ## train final model on all completed seasons
    train_df = df[df["season"].isin(completed_seasons)]
    X_train = train_df[feature_cols]
    y_train = train_df["final_points"]
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

if __name__ == "__main__":
    main()