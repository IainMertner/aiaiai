import numpy as np

## get feature columns (all numeric columns except labels and identifiers)
def get_feature_columns(df):
    drop_cols = [
        "final_points",
        "target_remaining_points",
        "gw_rank",
        "total_points",
        "season",
        "manager"
    ]
    feature_cols = [col for col in df.columns if col not in drop_cols and np.issubdtype(df[col].dtype, np.number)]
    print(feature_cols)
    return feature_cols