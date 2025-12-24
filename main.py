import pandas as pd

from scripts.prepare_features import prepare_features
from scripts.utils.get_feature_columns import get_feature_columns
from scripts.cross_validate import cross_validate
from scripts.train_model import train_model
from scripts.predict_current import predict_current
from scripts.explain_model import explain_model
from scripts.plot_probs import plot_probs
from scripts.estimate_sigma import mad_sigma, estimate_sigma
from scripts.simulate_win_probs import get_gw_sigma, simulate_win_probs

def main():
    ### prepare features
    points_df = pd.read_csv("raw/points.csv")
    managers_df = pd.read_csv("raw/managers.csv")
    
    features = prepare_features(points_df, managers_df)
    features.to_csv("output/features.csv", index=False)
    feature_cols = get_feature_columns(features)

    ### cross-validate
    cross_validate(feature_cols)

    ### train model
    train_model(feature_cols)

    ### estimate sigma
    estimate_sigma()

    ### predict current season
    n_sims = 100000
    predict_current(feature_cols, n_sims)

    ### explain model
    explain_model(feature_cols)

    ### plot win probabilities
    plot_probs()

if __name__ == "__main__":
    main()