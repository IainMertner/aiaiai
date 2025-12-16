import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def plot_probs():
    df = pd.read_csv("output/predictions/predictions_current_season.csv")

    last_gw = df["gw"].max()

    df_reordered = pd.concat([
        df[df["gw"] == last_gw],   # seaborn sees these first → determines hue order
        df[df["gw"] != last_gw]    # rest of the data
    ], ignore_index=True)
        
    managers = sorted(df["manager"].unique())

    palette = sns.color_palette("tab20", len(managers))
    colour_map = dict(zip(managers, palette))

    sns.set_theme(style="whitegrid")

    sns.lineplot(
        data=df_reordered,
        x="gw",
        y="win_probs",
        hue="manager",
        linewidth=2.5,
        marker="o",
        palette=colour_map,
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