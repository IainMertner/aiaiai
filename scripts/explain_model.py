import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import shap

def explain_model(feature_cols):
    # load model
    model = xgb.XGBRegressor()
    model.load_model("output/final_model.json")
    # load data
    df = pd.read_csv("output/features.csv")
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
    plt.savefig("output/plots/feature_importance.png")
    plt.close()

    # shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("output/plots/feature_importance_shap.png")
    plt.close()