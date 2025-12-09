import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

## get feature columns (all numeric columns except labels and identifiers)
def get_feature_columns(df):
    drop_cols = [
        "winner",
        "season",
        "manager"
    ]
    feature_cols = [col for col in df.columns if col not in drop_cols and np.issubdtype(df[col].dtype, np.number)]

### train one fold (one train/validation split)
def train_one_fold(train_df, val_df, feature_cols):
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
    # evaluation metrics
    auc = roc_auc_score(y_val, preds_proba)
    acc = accuracy_score(y_val, preds)

    return model, auc, acc

def main():
    # load features
    features_df = pd.read_csv("output/features.csv")
    # identify completed seasons
    completed_seasons = sorted(features_df["season"].unique())
    print(completed_seasons)

if __name__ == "__main__":
    main()