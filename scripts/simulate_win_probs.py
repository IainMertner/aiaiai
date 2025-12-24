import pandas as pd
import numpy as np
from scipy.stats import t

def get_gw_sigma(gw):
    sigma_df = pd.read_csv("output/sigma_by_gw_bin.csv")

    if gw < 7:
        sigma = sigma_df.loc[sigma_df["gw_bin"] == "1-6", "sigma_smoothed"].iloc[0]
    elif gw < 13:
        sigma = sigma_df.loc[sigma_df["gw_bin"] == "7-12", "sigma_smoothed"].iloc[0]
    elif gw < 19:
        sigma = sigma_df.loc[sigma_df["gw_bin"] == "13-18", "sigma_smoothed"].iloc[0]
    elif gw < 25:
        sigma = sigma_df.loc[sigma_df["gw_bin"] == "19-24", "sigma_smoothed"].iloc[0]
    elif gw < 31:
        sigma = sigma_df.loc[sigma_df["gw_bin"] == "25-30", "sigma_smoothed"].iloc[0]
    elif gw < 35:
        sigma = sigma_df.loc[sigma_df["gw_bin"] == "31-34", "sigma_smoothed"].iloc[0]
    else:
        sigma = sigma_df.loc[sigma_df["gw_bin"] == "35-38", "sigma_smoothed"].iloc[0]

    return sigma

def simulate_win_probs(df, n_sims):
    for gw, gdf in df.groupby("gw"):
        sigma = get_gw_sigma(gw)

        managers = gdf["manager"].values
        points = gdf["total_points"].values
        mu = gdf["pred_remaining_points"].values

        idx = gdf.index.values

        n_managers = len(managers)

        remaining = mu[:, None] + sigma * np.random.normal(size=(n_managers, n_sims))
        final_points = points[:, None] + remaining
        winners = np.argmax(final_points, axis=0)
        wins = np.bincount(winners, minlength=n_managers)

        df.loc[idx, "win_probs"] = wins / n_sims

        print("Gameweek", gw, "simulated")
    
    return df