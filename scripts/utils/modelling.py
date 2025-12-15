import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from scripts.utils.bayesian_shrinkage import apply_bayesian_shrinkage
from scripts.utils.model_config import XGB_PARAMS


OUTPUT_COLS = [
    "manager",
    "season",
    "gw",
    "total_points",
    "pred_remaining_points",
    "final_points",
    "pred_final_points",
    "gw_rank",
    "pred_rank",
    "win_prob_bayes",
]


def train_xgb(df, feature_cols):
    X = df[feature_cols]
    y = df["target_remaining_points"]

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)

    return model


def evaluate(model, df, feature_cols):
    X = df[feature_cols]
    y = df["target_remaining_points"]

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    return preds, mae, r2


def postprocess_predictions(df, preds, tau):
    out = df.copy()

    out["pred_remaining_points"] = preds
    out["pred_final_points"] = out["total_points"] + out["pred_remaining_points"]

    out["pred_rank"] = (
        out.groupby(["season", "gw"])["pred_final_points"]
        .rank(ascending=False, method="min")
    )

    out["win_prob"] = (
        out.groupby(["season", "gw"])["pred_final_points"]
        .transform(
            lambda x: np.exp(x / tau - (x / tau).max())
            / np.exp(x / tau - (x / tau).max()).sum()
        )
    )

    out = apply_bayesian_shrinkage(out)

    return (
        out[OUTPUT_COLS]
        .sort_values(["season", "gw", "pred_final_points"],
                     ascending=[True, True, False])
    )