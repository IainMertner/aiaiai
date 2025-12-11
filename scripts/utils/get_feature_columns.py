import numpy as np

## get feature columns (all numeric columns except labels and identifiers)
def get_feature_columns(df):
    cols = [
        "gw",
        "avg_points",
        "avg_std",
        "ewm"
    ]
    feature_cols = [col for col in df.columns if col in cols]
    return feature_cols