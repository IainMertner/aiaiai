## get feature columns (all numeric columns except labels and identifiers)
def get_feature_columns(df):
    cols = [
        "remaining_gws",
        "avg_points",
        "cv",
        "ewm_l"
    ]
    feature_cols = [col for col in cols if col in df.columns]
    return feature_cols