import pandas as pd
import numpy as np

### build features
def build_features(df, managers_df):
    # sort
    df = df.sort_values(["season", "manager", "gw"])

    # cumulative points
    df["total_points"] = (
        df.groupby(["season", "manager"])["gw_points"]
        .cumsum()
    )
    '''
    # gameweek rank
    df["gw_rank"] = (
        df.groupby(["season","gw"])["total_points"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    '''
    '''
    # points behind first
    df["points_behind_first"] = (
        df.groupby(["season", "gw"])["total_points"]
        .transform(lambda x: x.max() - x)
    )
    '''
    '''
    # deviation from gameweek mean
    df["gw_mean"] = (
        df.groupby(["season", "gw"])["gw_points"]
        .transform("mean")
    )
    '''
    '''
    df["gw_dev"] = df["gw_points"] - df["gw_mean"]
    '''
    # rolling averages and standard deviations
    df["avg_last3"] = (
        df.groupby(["season", "manager"])["gw_points"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0,1], drop=True)
    )
    df["avg_last5"] = (
        df.groupby(["season", "manager"])["gw_points"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=[0,1], drop=True)
    )
    df["std_last5"] = (
        df.groupby(["season", "manager"])["gw_points"]
        .rolling(5, min_periods=1)
        .std()
        .reset_index(level=[0,1], drop=True)
    )
    df["std_last5"] = df["std_last5"].fillna(0)
    '''
    # merge manager-level data
    df = df.merge(managers_df, on="manager", how="left")
    '''

    ### labels (did they win the season)
    completed_seasons = (
        df.groupby("season")["gw"]
        .max()
        .loc[lambda s: s == 38]
        .index
        .tolist()
    )
    final_points = (
        df[df["season"].isin(completed_seasons)]
        .groupby(["season", "manager"])["total_points"]
        .max()
        .rename("final_points")
        .reset_index()
    )
    df = df.merge(
        final_points,
        on=["season", "manager"],
        how="left",
    )

    return df

def main():
    points_df = pd.read_csv("raw/points.csv")
    managers_df = pd.read_csv("raw/managers.csv")
    
    features = build_features(points_df, managers_df)
    features.to_csv("output/features.csv", index=False)

if __name__ == "__main__":
    main()