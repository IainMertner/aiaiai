import numpy as np
import pandas as pd

from scripts.utils.resource_path import resource_path

def mad_sigma(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def estimate_sigma():
    df = pd.concat([
        pd.read_csv(resource_path("output/predictions/val_predictions_season_2022.csv")),
        pd.read_csv(resource_path("output/predictions/val_predictions_season_2023.csv")),
        pd.read_csv(resource_path("output/predictions/val_predictions_season_2024.csv")),
    ], ignore_index=True)

    bins = [1, 7, 13, 19, 25, 31, 35, 39]
    labels = [
        "1-6", "7-12", "13-18",
        "19-24", "25-30", "31-34", "35-38"
    ]
    df["gw_bin"] = pd.cut(
        df["gw"],
        bins=bins,
        labels=labels,
        right=False
    )

    sigma_by_bin = (
        df
        .groupby("gw_bin")["residuals"]
        .apply(mad_sigma)
        .reset_index(name="sigma")
    )
    
    order = labels
    sigma_by_bin["gw_bin"] = pd.Categorical(
        sigma_by_bin["gw_bin"],
        categories=order,
        ordered=True
    )

    sigma_by_bin = sigma_by_bin.sort_values("gw_bin").reset_index(drop=True)

    sigma_by_bin["sigma_smoothed"] = (
        sigma_by_bin["sigma"]
        .rolling(window=3, center=True, min_periods=2)
        .mean()
    )

    sigma_by_bin.to_csv(resource_path("output/sigma_by_gw_bin.csv"), index=False)