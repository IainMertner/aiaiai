import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def plot_probs():
    df = pd.read_csv("output/predictions/predictions_current_season.csv")

    sns.set_theme(style="whitegrid")

    sns.lineplot(
        data=df,
        x="gw",
        y="win_prob",
        hue="manager",
        palette="tab20",
        linewidth=2.5,
        marker="o",
        markersize=6
    )

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    plt.title("FPL Title Race: Win Probability Over Time", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Gameweek", fontsize=12)
    plt.ylabel("Win Probability", fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, title="Manager")
    plt.tight_layout()
    plt.savefig("output/plots/win_probs_current_season.png")
    plt.close()