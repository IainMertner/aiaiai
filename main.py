import pandas as pd

from scripts.prepare_features import prepare_features
from scripts.utils.get_feature_columns import get_feature_columns
from scripts.cross_validate import cross_validate
from scripts.train_model import train_model
from scripts.predict_current import predict_current
from scripts.explain_model import explain_model
from scripts.plot_probs import plot_probs

def main():
    ### prepare features
    points_df = pd.read_csv("raw/points.csv")
    managers_df = pd.read_csv("raw/managers.csv")
    
    features = prepare_features(points_df, managers_df)
    features.to_csv("output/features.csv", index=False)
    feature_cols = get_feature_columns(features)

    tau = 75

    ### cross-validate
    cross_validate(feature_cols, tau)

    ### train model
    train_model(feature_cols)

    ### predict current season
    predict_current(feature_cols, tau)

    ### explain model
    explain_model(feature_cols)

    ### plot win probabilities
    plot_probs()

if __name__ == "__main__":
    main()