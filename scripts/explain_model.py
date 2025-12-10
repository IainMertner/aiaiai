import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from train_model import get_feature_columns

def explain_model():
    # load model
    model = xgb.XGBRegressor()
    model.load_model("output/final_model.json")
    # load data
    df = pd.read_csv("output/features.csv")
    feature_cols = get_feature_columns(df)
    # identify current season
    current_season = (
        df.groupby("season")["gw"]
        .max()
        .loc[lambda s: s < 38]
        .index
        .tolist()
    )
    df = df[df["season"].isin(current_season)]
    X = df[feature_cols]

    # xgboost feature importance
    plt.figure(figsize=(10,8))
    xgb.plot_importance(model, importance_type="gain", max_num_features=15, height=0.5)
    plt.tight_layout()
    plt.savefig("output/feature_importance.png")
    plt.close()

def main():
    explain_model()

if __name__ == "__main__":
    main()