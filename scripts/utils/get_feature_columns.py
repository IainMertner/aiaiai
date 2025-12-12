import numpy as np

## get feature columns (all numeric columns except labels and identifiers)
def get_feature_columns(df):
    cols = [
        "remaining_gws",
        "avg_points",
        "avg_std",
        "ewm_l"
    ]
    feature_cols = [col for col in df.columns if col in cols]
    return feature_cols