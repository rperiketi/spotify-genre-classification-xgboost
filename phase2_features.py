"""
Phase 2: Feature Engineering
Spotify Music Genre Classification Project

Steps:
- Load raw CSV
- Drop nulls and duplicates
- Normalize audio features (StandardScaler)
- Encode categoricals (key, mode, time_signature)
- Create derived features
- Run feature importance with ExtraTreesClassifier
- Save processed dataframe to outputs/processed_data.pkl
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings("ignore")

DATA_PATH    = "data/songs_normalize.csv"
OUTPUT_DIR   = "outputs"
PLOTS_DIR    = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "popularity", "duration_ms",
]

CATEGORICAL_FEATURES = ["key", "mode", "time_signature"]

# ── helpers ────────────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. "
            "Please download songs_normalize.csv from Kaggle and place it in data/."
        )
    df = pd.read_csv(DATA_PATH)
    return df


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

# ── cleaning ───────────────────────────────────────────────────────────────────

def clean_data(df):
    before = len(df)
    df = df.dropna().copy()
    df = df.drop_duplicates().copy()
    after = len(df)
    print(f"  Rows before cleaning: {before}")
    print(f"  Rows after cleaning : {after}")
    return df

# ── encoding ───────────────────────────────────────────────────────────────────

def encode_categoricals(df):
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  Encoded '{col}' -> {len(le.classes_)} unique values")
    return df, label_encoders

# ── normalization ──────────────────────────────────────────────────────────────

def normalize_features(df):
    available = [f for f in NUMERIC_FEATURES if f in df.columns]
    scaler = StandardScaler()
    df[available] = scaler.fit_transform(df[available])
    print(f"  Scaled {len(available)} numeric features")
    return df, scaler, available

# ── derived features ───────────────────────────────────────────────────────────

def create_derived_features(df):
    # These are computed on already-scaled values, so still meaningful as products
    if "energy" in df.columns and "loudness" in df.columns:
        df["energy_loudness"] = df["energy"] * df["loudness"]
        print("  Created derived feature: energy_loudness")

    if "valence" in df.columns and "tempo" in df.columns:
        df["valence_tempo"] = df["valence"] * df["tempo"]
        print("  Created derived feature: valence_tempo")

    return df

# ── genre encoding ─────────────────────────────────────────────────────────────

def simplify_genres(df):
    """
    The raw genre column contains compound multi-genre strings like
    'hip hop, pop, R&B'. We extract the primary genre (first listed)
    to keep classification tractable and interpretable.
    """
    def extract_primary(raw):
        if not isinstance(raw, str):
            return "unknown"
        # strip set() artifacts
        raw = raw.replace("set()", "").strip(", ").strip()
        if not raw:
            return "unknown"
        primary = raw.split(",")[0].strip().lower()
        # clean up common naming variants
        mapping = {
            "dance/electronic": "electronic",
            "folk/acoustic": "folk",
            "r&b": "r&b",
            "world/traditional": "world",
        }
        for key, val in mapping.items():
            if key in primary:
                return val
        return primary

    df["genre"] = df["genre"].apply(extract_primary)
    print(f"  Primary genre distribution:\n{df['genre'].value_counts()}\n")
    return df


def encode_target(df):
    le = LabelEncoder()
    df["genre_encoded"] = le.fit_transform(df["genre"])
    print(f"  Target genres ({len(le.classes_)}): {list(le.classes_)}")
    return df, le

# ── feature importance ─────────────────────────────────────────────────────────

def plot_feature_importance(df, target_col="genre_encoded"):
    set_dark_style()

    drop_cols = ["genre", "genre_encoded", "artist", "song", "year", "explicit"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df[target_col].values

    clf = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    importance = pd.Series(clf.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_cols) * 0.4)))
    colors = ["#1DB954" if v > importance.median() else "#509BF5"
              for v in importance.values]
    ax.barh(importance.index, importance.values, color=colors, height=0.65)
    ax.set_title("Feature Importance (ExtraTrees)", fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)

    path = os.path.join(PLOTS_DIR, "feature_importance.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved -> {path}")

    return importance

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 2: Feature Engineering ===")

    df = load_data()

    print("\nStep 1: Cleaning...")
    df = clean_data(df)

    print("\nStep 2: Encoding categoricals...")
    df, label_encoders = encode_categoricals(df)

    print("\nStep 3: Normalizing numeric features...")
    df, scaler, scaled_features = normalize_features(df)

    print("\nStep 4: Creating derived features...")
    df = create_derived_features(df)

    print("\nStep 4b: Simplifying genre labels to primary genre...")
    df = simplify_genres(df)

    print("\nStep 5: Encoding target (genre)...")
    df, genre_encoder = encode_target(df)

    print("\nStep 6: Computing feature importance...")
    importance = plot_feature_importance(df)
    print(f"\n  Top 5 features:\n{importance.sort_values(ascending=False).head(5)}")

    # save processed data and artifacts
    out_pkl  = os.path.join(OUTPUT_DIR, "processed_data.pkl")
    out_meta = os.path.join(OUTPUT_DIR, "preprocessing_meta.pkl")

    df.to_pickle(out_pkl)
    joblib.dump(
        {
            "scaler":          scaler,
            "label_encoders":  label_encoders,
            "genre_encoder":   genre_encoder,
            "scaled_features": scaled_features,
        },
        out_meta,
    )

    print(f"\nSaved processed data  -> {out_pkl}")
    print(f"Saved preprocessing   -> {out_meta}")
    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
