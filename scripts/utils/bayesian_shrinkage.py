def apply_bayesian_shrinkage(df):
    df = df.copy()

    # number of managers per gameweek (needed for the uniform prior)
    df["n_managers"] = df.groupby(["season", "gw"])["manager"].transform("count")

    # α(gw)
    df["alpha"] = df["remaining_gws"] / 38
    df["alpha"] = df["alpha"].clip(lower=0, upper=1)

    # uniform prior = 1 / number of managers
    df["prior"] = 1 / df["n_managers"]

    # shrink probabilities:
    # p_final = α * prior + (1 - α) * model_prob
    df["win_prob_bayes"] = (
        df["alpha"] * df["prior"]
        + (1 - df["alpha"]) * df["win_prob"]
    )

    return df