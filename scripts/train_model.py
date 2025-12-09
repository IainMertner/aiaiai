import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

## get feature columns (all numeric columns except labels and identifiers)
def get_feature_columns(df):
    drop_cols = [
        "winner",
        "num_wins",
        "season",
        "manager"
    ]
    feature_cols = [col for col in df.columns if col not in drop_cols and np.issubdtype(df[col].dtype, np.number)]
    
    return feature_cols

### train one fold (one train/validation split)
def train_one_fold(train_df, val_df, feature_cols, val_season):
    X_train = train_df[feature_cols]
    y_train = train_df["winner"]
    X_val = val_df[feature_cols]
    y_val = val_df["winner"]

    # class imbalance (many more non-winners than winners)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg/pos if pos > 0 else 1.0

    ## XGBoost model
    model = XGBClassifier(
        n_estimators = 300,
        learning_rate = 0.05,
        max_depth = 4,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = "binary:logistic",
        scale_pos_weight = scale,
        eval_metric = "logloss"
    )
    # fit model
    model.fit(X_train, y_train)
    # predictions
    preds_proba = model.predict_proba(X_val)[:, 1]
    preds = (preds_proba > 0.5).astype(int)
    logits = model.predict(X_val, output_margin=True)
    # evaluation metrics
    auc = roc_auc_score(y_val, preds_proba)
    acc = accuracy_score(y_val, preds)
    ## save predictions
    val_out = val_df.copy()
    val_out["logit"] = logits
    cols_to_keep = ["manager", "season", "gw", "total_points", "winner", "logit"]
    val_out = val_out[cols_to_keep]
    # probabilities
    val_out["softmax_prob"] = (
        val_out.groupby(["season", "gw"])["logit"]
        .transform(lambda x: np.exp(x) / np.exp(x).sum())
    )
    val_out = val_out.sort_values(["season", "gw", "softmax_prob"], ascending=[True, True, False])
    save_path = f"output/val_predictions_season_{val_season}.csv"
    val_out.to_csv(save_path, index=False)
    

    return model, auc, acc

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

    results = []
    ## season-level cross-validation
    for val_season in completed_seasons:
        # split into training and validation sets
        train_seasons = [season for season in completed_seasons if season != val_season]
        train_df = df[df["season"].isin(train_seasons)]
        val_df = df[df["season"] == val_season]
        # train model
        model, auc, acc = train_one_fold(train_df, val_df, feature_cols, val_season)
        # store results
        results.append((val_season, auc, acc))
    
    avg_auc = np.mean([r[1] for r in results])
    avg_acc = np.mean([r[2] for r in results])
    print("avg_auc:", avg_auc)
    print("avg_acc:", avg_acc)

if __name__ == "__main__":
    main()