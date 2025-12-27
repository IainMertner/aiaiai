import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import shap

from scripts.utils.resource_path import resource_path

def explain_model(feature_cols):
    # load model
    model = xgb.XGBRegressor()
    model.load_model(resource_path("output/models/model_0.json"))
    # load data
    df = pd.read_csv(resource_path("output/features.csv"))
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
    plt.savefig(resource_path("output/plots/feature_importance.png"))
    plt.close()

    # shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(resource_path("output/plots/feature_importance_shap.png"))
    plt.close()
    # individual row shap valuesidxs = [102, 162, 163, 164]
    '''
    idxs = [102, 162, 87, 164]
    titles = [
        "iain – GW13",
        "nielsj – GW13",
        "george – GW13",
        "nielsj – GW15"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, idx, title in zip(axes.flatten(), idxs, titles):
        plt.sca(ax)  # tell matplotlib "draw into this axis"
        shap.plots.waterfall(
            shap_values[idx],
            max_display=10,
            show=False
        )
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
    '''