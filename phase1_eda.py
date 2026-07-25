"""
Phase 1: Exploratory Data Analysis
Spotify Music Genre Classification Project

Loads the dataset, inspects it, and produces:
- Class balance chart
- Feature distribution plots
- Correlation heatmap
- Radar chart per genre (hero visual)
- t-SNE cluster plot colored by genre
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/songs_normalize.csv"
PLOTS_DIR  = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── colour palette (one per genre) ─────────────────────────────────────────────
GENRE_COLORS = [
    "#1DB954", "#E91429", "#509BF5", "#FF6437",
    "#B49BC8", "#F037A5", "#C8A951", "#148A08",
    "#2D46B9", "#F3727F", "#988BBF", "#FFD700",
    "#00C4B4", "#FF8C00", "#A45BC4",
]

AUDIO_FEATURES = [
    "danceability", "energy", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence",
]

# ── helpers ────────────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. "
            "Please download songs_normalize.csv from Kaggle and place it in the data/ folder."
        )
    df = pd.read_csv(DATA_PATH)
    return df


def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  saved -> {path}")
    plt.close(fig)


def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  "#191414",
        "axes.facecolor":    "#191414",
        "axes.edgecolor":    "#3d3d3d",
        "axes.labelcolor":   "#ffffff",
        "text.color":        "#ffffff",
        "xtick.color":       "#b3b3b3",
        "ytick.color":       "#b3b3b3",
        "grid.color":        "#3d3d3d",
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    14,
        "axes.labelsize":    11,
    })

# ── 1. basic inspection ────────────────────────────────────────────────────────

def inspect_data(df):
    print("\n--- Dataset Info ---")
    print(f"Shape      : {df.shape}")
    print(f"Columns    : {list(df.columns)}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nGenre distribution:\n{df['genre'].value_counts()}")

# ── 2. class balance ───────────────────────────────────────────────────────────

def plot_class_balance(df):
    set_dark_style()
    counts = df["genre"].value_counts()
    genres = counts.index.tolist()
    colors = GENRE_COLORS[:len(genres)]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(genres, counts.values, color=colors, width=0.6, zorder=2)
    ax.set_title("Number of Songs per Genre", fontweight="bold")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=9, color="#b3b3b3")
    fig.tight_layout()
    save(fig, "class_balance.png")

# ── 3. feature distributions ───────────────────────────────────────────────────

def plot_distributions(df):
    set_dark_style()
    num_features = AUDIO_FEATURES + ["tempo", "loudness", "popularity"]
    cols = 3
    rows = int(np.ceil(len(num_features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(num_features):
        if feat not in df.columns:
            axes[i].set_visible(False)
            continue
        sns.histplot(df[feat], kde=True, ax=axes[i],
                     color="#1DB954", fill=True, alpha=0.5, linewidth=1.5)
        axes[i].set_title(feat.replace("_", " ").title())
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Audio Feature Distributions", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "feature_distributions.png")

# ── 4. correlation heatmap ─────────────────────────────────────────────────────

def plot_correlation_heatmap(df):
    set_dark_style()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(10, 145, as_cmap=True)   # red to green
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.5, linecolor="#3d3d3d",
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=15, pad=15)
    fig.tight_layout()
    save(fig, "correlation_heatmap.png")

# ── 5. radar chart (hero visual) ───────────────────────────────────────────────

def plot_radar_chart(df):
    set_dark_style()
    genres  = df["genre"].unique().tolist()
    n_feats = len(AUDIO_FEATURES)
    angles  = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles += angles[:1]                   # close the loop

    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor("#191414")

    # one subplot per genre in a grid
    cols = 4
    rows = int(np.ceil(len(genres) / cols))

    for idx, genre in enumerate(genres):
        ax = fig.add_subplot(rows, cols, idx + 1, polar=True)
        ax.set_facecolor("#191414")

        genre_data = df[df["genre"] == genre][AUDIO_FEATURES].mean().tolist()
        genre_data += genre_data[:1]

        color = GENRE_COLORS[idx % len(GENRE_COLORS)]
        ax.plot(angles, genre_data, color=color, linewidth=2)
        ax.fill(angles, genre_data, color=color, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [f.replace("_", "\n") for f in AUDIO_FEATURES],
            size=7, color="#b3b3b3"
        )
        ax.set_yticklabels([])
        ax.set_ylim(0, 1)
        ax.set_title(genre, size=10, fontweight="bold", color=color, pad=10)
        ax.tick_params(colors="#3d3d3d")
        ax.spines["polar"].set_color("#3d3d3d")
        ax.grid(color="#3d3d3d", linewidth=0.5)

    fig.suptitle("Audio Feature Radar Chart by Genre", fontsize=16, fontweight="bold",
                 color="#ffffff", y=1.01)
    fig.tight_layout()
    save(fig, "radar_chart.png")

# ── 6. t-SNE cluster plot ──────────────────────────────────────────────────────

def plot_tsne(df):
    set_dark_style()
    features = AUDIO_FEATURES + ["tempo", "loudness", "popularity",
                                  "danceability", "energy"]
    features = [f for f in features if f in df.columns]
    features = list(set(features))   # deduplicate

    sub = df[features + ["genre"]].dropna().copy()
    if len(sub) > 5000:
        sub = sub.sample(5000, random_state=42)

    X = StandardScaler().fit_transform(sub[features])
    tsne = TSNE(n_components=2, random_state=42, perplexity=40,
                n_iter=1000, verbose=0)
    coords = tsne.fit_transform(X)

    genres = sub["genre"].unique().tolist()
    color_map = {g: GENRE_COLORS[i % len(GENRE_COLORS)] for i, g in enumerate(genres)}

    fig, ax = plt.subplots(figsize=(12, 8))
    for genre in genres:
        mask = sub["genre"].values == genre
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color_map[genre], s=8, alpha=0.7, label=genre, rasterized=True
        )

    ax.set_title("t-SNE: Audio Features Colored by Genre", fontsize=15, fontweight="bold")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(
        bbox_to_anchor=(1.01, 1), loc="upper left",
        framealpha=0.15, edgecolor="#3d3d3d",
        labelcolor="#ffffff", fontsize=8
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()
    save(fig, "tsne_plot.png")

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 1: Exploratory Data Analysis ===")
    df = load_data()
    inspect_data(df)

    print("\nGenerating plots...")
    plot_class_balance(df)
    plot_distributions(df)
    plot_correlation_heatmap(df)
    plot_radar_chart(df)
    print("Running t-SNE (this may take a minute)...")
    plot_tsne(df)

    print("\nPhase 1 complete. All plots saved to outputs/plots/")


if __name__ == "__main__":
    main()
