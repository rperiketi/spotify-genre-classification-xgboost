"""
Phase 3: Model Training and Comparison
Spotify Music Genre Classification Project

Models trained:
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- K-Nearest Neighbors

Outputs:
- Classification reports in console
- Confusion matrix plots per model
- Accuracy vs F1 comparison chart
- Best model saved via joblib
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PKL   = "outputs/processed_data.pkl"
META_PKL   = "outputs/preprocessing_meta.pkl"
MODELS_DIR = "outputs/models"
PLOTS_DIR  = "outputs/plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

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
        "axes.titlesize":    13,
        "axes.labelsize":    11,
    })


def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved -> {path}")


def get_feature_columns(df):
    drop_cols = {"genre", "genre_encoded", "artist", "song", "year", "explicit"}
    return [c for c in df.columns if c not in drop_cols]


def load_data():
    if not os.path.exists(DATA_PKL):
        raise FileNotFoundError(
            "Processed data not found. Please run phase2_features.py first."
        )
    df   = pd.read_pickle(DATA_PKL)
    meta = joblib.load(META_PKL)
    return df, meta

# ── model definitions ──────────────────────────────────────────────────────────

def get_models(num_classes):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=6, use_label_encoder=False,
            eval_metric="mlogloss", random_state=42,
            n_jobs=-1, num_class=num_classes,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=11, metric="euclidean", n_jobs=-1
        ),
    }

# ── training and evaluation ────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test, genre_names):
    num_classes = len(np.unique(y_train))
    models = get_models(num_classes)
    results = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average="macro", zero_division=0)

        results[name] = {
            "model":  model,
            "preds":  preds,
            "acc":    acc,
            "f1":     f1,
        }
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Macro F1 : {f1:.4f}")
        # only report on labels that actually appear in the test set
        test_labels = sorted(np.unique(np.concatenate([y_test, preds])))
        test_genre_names = [genre_names[i] for i in test_labels if i < len(genre_names)]
        print(f"\n{classification_report(y_test, preds, labels=test_labels, target_names=test_genre_names, zero_division=0)}")

    return results

# ── confusion matrix plot ──────────────────────────────────────────────────────

def plot_confusion_matrices(results, y_test, genre_names):
    set_dark_style()
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    cmap = sns.light_palette("#1DB954", as_cmap=True)

    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["preds"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        present_labels = sorted(np.unique(np.concatenate([y_test, res["preds"]])))
        tick_names = [genre_names[i] if i < len(genre_names) else str(i)
                      for i in present_labels]

        sns.heatmap(
            cm_norm, ax=axes[idx],
            cmap=cmap, annot=len(present_labels) <= 15, fmt=".2f",
            linewidths=0.3, linecolor="#3d3d3d",
            xticklabels=tick_names, yticklabels=tick_names,
            cbar_kws={"shrink": 0.7},
            annot_kws={"size": 6},
        )
        acc = res["acc"]
        axes[idx].set_title(f"{name}  |  Acc {acc:.3f}", fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
        axes[idx].tick_params(axis="x", rotation=45, labelsize=6)
        axes[idx].tick_params(axis="y", rotation=0,  labelsize=6)

    fig.suptitle("Confusion Matrices (Normalised)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save(fig, "confusion_matrices.png")

# ── accuracy / F1 comparison chart ─────────────────────────────────────────────

def plot_model_comparison(results):
    set_dark_style()
    names  = list(results.keys())
    accs   = [results[n]["acc"] for n in names]
    f1s    = [results[n]["f1"]  for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#1DB954", zorder=2)
    bars2 = ax.bar(x + width / 2, f1s,  width, label="Macro F1", color="#509BF5", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Accuracy vs Macro F1", fontweight="bold")
    ax.legend(framealpha=0.15, edgecolor="#3d3d3d")
    ax.grid(axis="y", alpha=0.3, zorder=1)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save(fig, "model_comparison.png")

# ── save best model ────────────────────────────────────────────────────────────

def save_best_model(results, feature_cols, genre_names, meta):
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name}  (Macro F1 = {results[best_name]['f1']:.4f})")

    payload = {
        "model":           best_model,
        "model_name":      best_name,
        "feature_cols":    feature_cols,
        "genre_names":     genre_names,
        "scaler":          meta["scaler"],
        "label_encoders":  meta["label_encoders"],
        "genre_encoder":   meta["genre_encoder"],
        "scaled_features": meta["scaled_features"],
    }
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(payload, path)
    print(f"  Saved -> {path}")

    # also save comparison summary
    summary = {name: {"accuracy": res["acc"], "macro_f1": res["f1"]}
               for name, res in results.items()}
    summary_df = pd.DataFrame(summary).T
    summary_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    summary_df.to_csv(summary_path)
    print(f"  Saved -> {summary_path}")

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 3: Model Training and Comparison ===")

    df, meta = load_data()
    genre_encoder = meta["genre_encoder"]
    genre_names   = list(genre_encoder.classes_)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["genre_encoded"].values

    print(f"\nFeatures used : {len(feature_cols)}")
    print(f"Samples       : {len(X)}")
    print(f"Classes       : {genre_names}")

    # remove classes that have fewer than 2 samples (stratified split requires >= 2)
    from collections import Counter
    class_counts = Counter(y)
    valid_mask = np.array([class_counts[label] >= 2 for label in y])
    X = X[valid_mask]
    y = y[valid_mask]
    remaining_classes = sorted(np.unique(y))
    genre_names = [genre_encoder.classes_[i] for i in remaining_classes]
    # remap y to 0-indexed so models work cleanly
    y_remap = np.searchsorted(remaining_classes, y)
    y = y_remap
    print(f"After filtering rare classes: {len(X)} samples, {len(genre_names)} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    results = train_and_evaluate(X_train, X_test, y_train, y_test, genre_names)

    print("\nGenerating plots...")
    plot_confusion_matrices(results, y_test, genre_names)
    plot_model_comparison(results)

    save_best_model(results, feature_cols, genre_names, meta)

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
