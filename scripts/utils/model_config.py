XGB_PARAMS = {
    "n_jobs": 1,
    "n_estimators": 300,
    "learning_rate": 0.08,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "monotone_constraints": "(1, 1, 0, 0)"
}