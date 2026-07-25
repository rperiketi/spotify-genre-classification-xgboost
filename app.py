"""
Phase 4: Streamlit Dashboard
Spotify Music Genre Classification Project

Spotify-themed dark UI with:
- Sidebar sliders for all audio features
- Real-time genre prediction + confidence chart
- Dynamic radar chart vs genre average
- Nearest songs from dataset
- EDA and model comparison tabs
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── spotify dark theme CSS ─────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #121212;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #282828;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    .main-header {
        background: linear-gradient(135deg, #1DB954 0%, #0f7a35 60%, #121212 100%);
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }

    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: rgba(255,255,255,0.75);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    .genre-badge {
        display: inline-block;
        background: #1DB954;
        color: #000000;
        font-weight: 700;
        font-size: 1.6rem;
        padding: 0.5rem 1.4rem;
        border-radius: 50px;
        letter-spacing: 0.5px;
    }

    .metric-card {
        background: #181818;
        border: 1px solid #282828;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }

    .metric-card .label {
        font-size: 0.78rem;
        color: #B3B3B3;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }

    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1DB954;
    }

    .song-card {
        background: #181818;
        border: 1px solid #282828;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
        transition: background 0.2s;
    }

    .song-card:hover {
        background: #282828;
    }

    .song-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #FFFFFF;
    }

    .song-meta {
        font-size: 0.8rem;
        color: #B3B3B3;
        margin-top: 2px;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 1.2rem 0 0.7rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #1DB954;
        display: inline-block;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #000000;
        border-bottom: 1px solid #282828;
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        color: #B3B3B3;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        border-radius: 0;
        background: transparent;
    }

    .stTabs [aria-selected="true"] {
        color: #FFFFFF;
        border-bottom: 3px solid #1DB954;
        background: transparent;
    }

    .stSlider > div > div > div > div {
        background: #1DB954 !important;
    }

    .stButton > button {
        background: #1DB954;
        color: #000000;
        font-weight: 700;
        border: none;
        border-radius: 50px;
        padding: 0.55rem 1.5rem;
        font-size: 0.9rem;
        width: 100%;
        transition: transform 0.15s, background 0.15s;
    }

    .stButton > button:hover {
        background: #1ed760;
        transform: scale(1.02);
    }

    div[data-testid="stMarkdownContainer"] p {
        color: #B3B3B3;
    }

    [data-testid="stMetricValue"] {
        color: #1DB954 !important;
    }

    hr {
        border-color: #282828;
    }

    .sidebar-label {
        font-size: 0.7rem;
        color: #B3B3B3;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── constants ──────────────────────────────────────────────────────────────────

AUDIO_FEATURES = [
    "danceability", "energy", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
]

GENRE_COLORS = {
    "pop":            "#1DB954",
    "rock":           "#E91429",
    "hip-hop":        "#509BF5",
    "country":        "#FF6437",
    "jazz":           "#B49BC8",
    "r&b":            "#F037A5",
    "latin":          "#C8A951",
    "folk":           "#148A08",
    "blues":          "#2D46B9",
    "reggaeton":      "#F3727F",
    "edm":            "#988BBF",
    "set1":           "#FFD700",
    "Dark Trap":      "#00C4B4",
    "Rap":            "#FF8C00",
    "Underground Rap":"#A45BC4",
}
DEFAULT_COLOR = "#1DB954"

MODEL_PATH = "outputs/models/best_model.pkl"
DATA_PATH  = "data/songs_normalize.csv"

# ── data and model loading ─────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_raw_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_processed_data():
    path = "outputs/processed_data.pkl"
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)

# ── colour helper ──────────────────────────────────────────────────────────────

def genre_color(genre_name):
    if genre_name is None:
        return DEFAULT_COLOR
    lower = genre_name.lower()
    for key, col in GENRE_COLORS.items():
        if key in lower:
            return col
    return DEFAULT_COLOR

# ── radar chart ────────────────────────────────────────────────────────────────

def build_radar_chart(user_values, genre_avg_values, genre_name):
    features_display = [f.replace("_", " ").title() for f in AUDIO_FEATURES]
    features_closed  = features_display + [features_display[0]]

    user_closed  = user_values  + [user_values[0]]
    genre_closed = genre_avg_values + [genre_avg_values[0]]
    color = genre_color(genre_name)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=genre_closed, theta=features_closed,
        fill="toself", fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}",
        line=dict(color=color, width=2),
        name=f"{genre_name} average",
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_closed, theta=features_closed,
        fill="toself", fillcolor="rgba(255,255,255,0.08)",
        line=dict(color="#FFFFFF", width=2, dash="dot"),
        name="Your input",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#181818",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#3d3d3d", color="#B3B3B3"),
            angularaxis=dict(gridcolor="#3d3d3d", color="#B3B3B3"),
        ),
        showlegend=True,
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        font=dict(color="#FFFFFF", family="Inter"),
        legend=dict(bgcolor="#181818", bordercolor="#282828", borderwidth=1),
        margin=dict(t=30, b=30, l=30, r=30),
        height=380,
    )
    return fig

# ── confidence bar chart ───────────────────────────────────────────────────────

def build_confidence_chart(genre_names, probabilities):
    sorted_pairs = sorted(zip(probabilities, genre_names), reverse=True)
    probs, names = zip(*sorted_pairs)

    colors = [genre_color(n) for n in names]

    fig = go.Figure(go.Bar(
        x=list(probs),
        y=list(names),
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(color="#B3B3B3", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="#121212",
        plot_bgcolor="#181818",
        font=dict(color="#FFFFFF", family="Inter"),
        xaxis=dict(range=[0, 1.05], gridcolor="#282828", showgrid=True, color="#B3B3B3"),
        yaxis=dict(gridcolor="#282828", color="#FFFFFF"),
        margin=dict(t=10, b=10, l=10, r=60),
        height=320,
        showlegend=False,
    )
    return fig

# ── nearest songs ──────────────────────────────────────────────────────────────

def find_nearest_songs(user_vector, raw_df, processed_df, feature_cols, genre_name, n=5):
    if raw_df is None or processed_df is None:
        return []

    available = [f for f in feature_cols if f in processed_df.columns]
    X = processed_df[available].values
    user_arr = np.array(user_vector).reshape(1, -1)

    # trim or pad user vector to match processed columns
    user_trimmed = np.zeros((1, len(available)))
    for i, col in enumerate(available):
        # map slider raw feature names to processed column names
        if col in AUDIO_FEATURES or col in ["tempo", "loudness", "popularity", "duration_ms"]:
            pass  # already in correct position
        user_trimmed[0, i] = user_arr[0, i] if i < user_arr.shape[1] else 0

    sims = cosine_similarity(user_arr[:, :X.shape[1]], X)[0]
    top_indices = np.argsort(sims)[::-1][:n * 3]  # get extras to filter by genre

    songs = []
    # first try same genre
    for idx in top_indices:
        if idx >= len(raw_df):
            continue
        row = raw_df.iloc[idx]
        if str(row.get("genre", "")).lower() == genre_name.lower():
            songs.append({
                "song":   row.get("song", "Unknown"),
                "artist": row.get("artist", "Unknown"),
                "genre":  row.get("genre", ""),
                "year":   int(row.get("year", 0)) if row.get("year", 0) else "N/A",
                "similarity": float(sims[idx]),
            })
        if len(songs) >= n:
            break

    # fill with any genre if needed
    if len(songs) < n:
        for idx in top_indices:
            if idx >= len(raw_df):
                continue
            row = raw_df.iloc[idx]
            entry = {
                "song":   row.get("song", "Unknown"),
                "artist": row.get("artist", "Unknown"),
                "genre":  row.get("genre", ""),
                "year":   int(row.get("year", 0)) if row.get("year", 0) else "N/A",
                "similarity": float(sims[idx]),
            }
            if entry not in songs:
                songs.append(entry)
            if len(songs) >= n:
                break

    return songs[:n]

# ── sidebar ────────────────────────────────────────────────────────────────────

def build_sidebar():
    st.sidebar.markdown(
        "<div style='padding: 1rem 0 0.5rem 0;'>"
        "<span style='font-size:1.3rem; font-weight:700; color:#1DB954;'>Genre Classifier</span><br>"
        "<span style='font-size:0.78rem; color:#B3B3B3;'>Adjust audio features below</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    st.sidebar.markdown("<div class='sidebar-label'>Audio Features</div>", unsafe_allow_html=True)

    params = {}
    params["danceability"]      = st.sidebar.slider("Danceability",      0.0, 1.0, 0.65, 0.01)
    params["energy"]            = st.sidebar.slider("Energy",            0.0, 1.0, 0.70, 0.01)
    params["speechiness"]       = st.sidebar.slider("Speechiness",       0.0, 1.0, 0.08, 0.01)
    params["acousticness"]      = st.sidebar.slider("Acousticness",      0.0, 1.0, 0.20, 0.01)
    params["instrumentalness"]  = st.sidebar.slider("Instrumentalness",  0.0, 1.0, 0.02, 0.01)
    params["liveness"]          = st.sidebar.slider("Liveness",          0.0, 1.0, 0.15, 0.01)
    params["valence"]           = st.sidebar.slider("Valence",           0.0, 1.0, 0.55, 0.01)

    st.sidebar.markdown("---")
    st.sidebar.markdown("<div class='sidebar-label'>Rhythm and Scale</div>", unsafe_allow_html=True)

    params["tempo"]      = st.sidebar.slider("Tempo (BPM)",    40.0, 220.0, 120.0, 1.0)
    params["loudness"]   = st.sidebar.slider("Loudness (dB)", -60.0,   0.0,  -8.0, 0.5)
    params["popularity"] = st.sidebar.slider("Popularity",      0,     100,    60,   1)
    params["duration_ms"]= st.sidebar.slider("Duration (ms)", 30000, 600000, 210000, 1000)

    st.sidebar.markdown("---")
    st.sidebar.markdown("<div class='sidebar-label'>Musical Properties</div>", unsafe_allow_html=True)

    params["key"]            = st.sidebar.selectbox("Key",           list(range(12)), index=5)
    params["mode"]           = st.sidebar.selectbox("Mode",          [0, 1], index=1,
                                                    format_func=lambda x: "Minor" if x == 0 else "Major")
    params["time_signature"] = st.sidebar.selectbox("Time Signature",[3, 4, 5, 6, 7], index=1)

    return params

# ── prediction ─────────────────────────────────────────────────────────────────

def predict(params, payload):
    model         = payload["model"]
    feature_cols  = payload["feature_cols"]
    genre_encoder = payload["genre_encoder"]
    label_encoders= payload["label_encoders"]
    scaler        = payload["scaler"]
    scaled_features = payload["scaled_features"]

    # build a row matching the feature columns
    row = {}
    for col in feature_cols:
        if col in params:
            row[col] = params[col]
        else:
            row[col] = 0.0

    # encode categoricals
    for col, le in label_encoders.items():
        if col in row:
            val = str(row[col])
            if val in le.classes_:
                row[col] = int(le.transform([val])[0])
            else:
                row[col] = 0

    # build vector
    x = np.array([[row.get(c, 0.0) for c in feature_cols]], dtype=float)

    # scale numeric features
    available_scaled = [f for f in scaled_features if f in feature_cols]
    col_indices = [feature_cols.index(f) for f in available_scaled]
    # create a dummy full-feature array for scaler
    dummy = np.zeros((1, len(scaled_features)))
    for i, f in enumerate(scaled_features):
        if f in row:
            dummy[0, i] = row[f]
    dummy_scaled = scaler.transform(dummy)
    for i, idx in enumerate(col_indices):
        if i < dummy_scaled.shape[1]:
            x[0, idx] = dummy_scaled[0, i]

    # derived features
    e_idx = feature_cols.index("energy")   if "energy"   in feature_cols else -1
    l_idx = feature_cols.index("loudness") if "loudness" in feature_cols else -1
    v_idx = feature_cols.index("valence")  if "valence"  in feature_cols else -1
    t_idx = feature_cols.index("tempo")    if "tempo"    in feature_cols else -1
    el_idx = feature_cols.index("energy_loudness") if "energy_loudness" in feature_cols else -1
    vt_idx = feature_cols.index("valence_tempo")   if "valence_tempo"   in feature_cols else -1
    if el_idx >= 0 and e_idx >= 0 and l_idx >= 0:
        x[0, el_idx] = x[0, e_idx] * x[0, l_idx]
    if vt_idx >= 0 and v_idx >= 0 and t_idx >= 0:
        x[0, vt_idx] = x[0, v_idx] * x[0, t_idx]

    pred_idx  = model.predict(x)[0]
    genre_name = genre_encoder.inverse_transform([pred_idx])[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]

    return genre_name, proba, genre_encoder.classes_

# ── compute genre average for radar ───────────────────────────────────────────

def compute_genre_averages(raw_df):
    if raw_df is None:
        return {}
    avgs = {}
    for genre in raw_df["genre"].unique():
        sub = raw_df[raw_df["genre"] == genre]
        avgs[genre] = [
            float(sub[f].mean()) if f in sub.columns else 0.5
            for f in AUDIO_FEATURES
        ]
    return avgs

# ── EDA tab ────────────────────────────────────────────────────────────────────

def render_eda_tab():
    st.markdown("<div class='section-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)

    images = [
        ("Class Balance",          "outputs/plots/class_balance.png"),
        ("Feature Distributions",  "outputs/plots/feature_distributions.png"),
        ("Correlation Heatmap",    "outputs/plots/correlation_heatmap.png"),
        ("Radar Chart by Genre",   "outputs/plots/radar_chart.png"),
        ("t-SNE Cluster Plot",     "outputs/plots/tsne_plot.png"),
        ("Feature Importance",     "outputs/plots/feature_importance.png"),
    ]

    for title, path in images:
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path, use_container_width=True)
            st.markdown("---")
        else:
            st.warning(f"Plot not found: {path}  (run phase1_eda.py first)")

# ── model comparison tab ───────────────────────────────────────────────────────

def render_model_tab():
    st.markdown("<div class='section-title'>Model Comparison</div>", unsafe_allow_html=True)

    csv_path = "outputs/models/model_comparison.csv"
    img_path = "outputs/plots/model_comparison.png"
    cm_path  = "outputs/plots/confusion_matrices.png"

    if os.path.exists(csv_path):
        df_cmp = pd.read_csv(csv_path, index_col=0)
        df_cmp.columns = ["Accuracy", "Macro F1"]
        df_cmp = df_cmp.sort_values("Macro F1", ascending=False)
        df_cmp["Accuracy"] = df_cmp["Accuracy"].map("{:.3f}".format)
        df_cmp["Macro F1"] = df_cmp["Macro F1"].map("{:.3f}".format)
        st.dataframe(df_cmp)
    else:
        st.info("Run phase3_models.py to generate model comparison results.")

    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)

    if os.path.exists(cm_path):
        st.markdown("**Confusion Matrices**")
        st.image(cm_path, use_container_width=True)

# ── main app ───────────────────────────────────────────────────────────────────

def main():
    params = build_sidebar()

    # header
    st.markdown("""
    <div class='main-header'>
        <h1>Music Genre Classifier</h1>
        <p>Tune the audio features in the sidebar and see which genre your track most likely belongs to.</p>
    </div>
    """, unsafe_allow_html=True)

    payload   = load_model()
    raw_df    = load_raw_data()
    proc_df   = load_processed_data()
    genre_avgs = compute_genre_averages(raw_df)

    tab_predict, tab_eda, tab_models = st.tabs(
        ["Predict Genre", "Data Analysis", "Model Comparison"]
    )

    # ── predict tab ───────────────────────────────────────────────────────────
    with tab_predict:
        if payload is None:
            st.warning(
                "No trained model found. Please run phase1_eda.py, "
                "phase2_features.py, and phase3_models.py first. "
                "Then place songs_normalize.csv in the data/ folder."
            )
            st.stop()

        genre_name, proba, all_genres = predict(params, payload)
        color = genre_color(genre_name)

        # top prediction
        st.markdown("<div class='section-title'>Prediction</div>", unsafe_allow_html=True)
        col_badge, col_confidence = st.columns([1, 2])

        with col_badge:
            confidence = float(max(proba)) if proba is not None else 0.0
            st.markdown(f"""
            <div class='metric-card' style='border-color:{color};'>
                <div class='label'>Predicted Genre</div>
                <div style='margin:0.8rem 0;'>
                    <span class='genre-badge' style='background:{color};'>{genre_name}</span>
                </div>
                <div class='label'>Confidence</div>
                <div class='value' style='color:{color};'>{confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_confidence:
            if proba is not None:
                fig_conf = build_confidence_chart(list(all_genres), list(proba))
                st.plotly_chart(fig_conf, use_container_width=True)

        st.markdown("---")

        # radar chart
        st.markdown("<div class='section-title'>Feature Profile vs Genre Average</div>", unsafe_allow_html=True)
        user_audio_values = [params[f] for f in AUDIO_FEATURES]
        genre_avg_values  = genre_avgs.get(genre_name, [0.5] * len(AUDIO_FEATURES))
        fig_radar = build_radar_chart(user_audio_values, genre_avg_values, genre_name)
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")

        # nearest songs
        st.markdown("<div class='section-title'>Nearest Real Songs</div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#B3B3B3; font-size:0.85rem;'>Songs from the dataset closest to your input features</p>",
            unsafe_allow_html=True
        )

        if payload and proc_df is not None and raw_df is not None:
            feature_cols = payload["feature_cols"]
            user_vector  = [params.get(f, 0.0) for f in feature_cols]
            nearest      = find_nearest_songs(
                user_vector, raw_df, proc_df, feature_cols, genre_name, n=5
            )
            for song in nearest:
                gc = genre_color(song["genre"])
                st.markdown(f"""
                <div class='song-card'>
                    <div class='song-title'>{song['song']}</div>
                    <div class='song-meta'>
                        {song['artist']} &nbsp;|&nbsp;
                        <span style='color:{gc};'>{song['genre']}</span> &nbsp;|&nbsp;
                        {song['year']} &nbsp;|&nbsp;
                        Similarity: {song['similarity']:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run the pipeline scripts first to enable nearest-song lookup.")

    # ── eda and model tabs ─────────────────────────────────────────────────────
    with tab_eda:
        render_eda_tab()

    with tab_models:
        render_model_tab()


if __name__ == "__main__":
    main()
