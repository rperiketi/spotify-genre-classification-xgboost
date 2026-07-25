# Music Genre Classification

An end-to-end machine learning pipeline that classifies Spotify songs into genres using audio features, deployed as an interactive Streamlit dashboard with a Spotify-inspired dark theme.

---

## Project Structure

```
genre_classification/
├── data/
│   └── songs_normalize.csv    <-- place dataset here
├── outputs/
│   ├── plots/                 <-- auto-generated EDA plots
│   └── models/                <-- trained model artifacts
├── phase1_eda.py              <-- EDA and visualizations
├── phase2_features.py         <-- Feature engineering
├── phase3_models.py           <-- Model training and comparison
├── app.py                     <-- Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Download the Dataset

- Go to https://www.kaggle.com/code/varunsaikanuri/spotify-data-visualization/input
- Download `songs_normalize.csv`
- Place it in `genre_classification/data/`

### 2. Create a Virtual Environment

```bash
cd genre_classification
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Run the scripts in order:

```bash
# Phase 1: EDA and visualizations
python phase1_eda.py

# Phase 2: Feature engineering
python phase2_features.py

# Phase 3: Train and compare models
python phase3_models.py

# Phase 4: Launch the dashboard
streamlit run app.py
```

---

## What Each Phase Does

**Phase 1 (EDA):**
- Null checks and class balance inspection
- Distribution plots for each audio feature
- Correlation heatmap across all features
- Radar chart per genre showing average audio profiles
- t-SNE dimensionality reduction plot colored by genre

**Phase 2 (Feature Engineering):**
- StandardScaler normalization on 11 numeric features
- LabelEncoder for key, mode, time_signature
- Derived features: energy x loudness, valence x tempo
- Feature importance analysis using ExtraTreesClassifier

**Phase 3 (Model Comparison):**
- Logistic Regression (baseline)
- Random Forest (200 estimators)
- XGBoost (200 estimators)
- K-Nearest Neighbors (k=11)
- Confusion matrices + accuracy/F1 comparison chart

**Phase 4 (Streamlit App):**
- Spotify dark theme with green accents
- Sidebar sliders for all audio feature inputs
- Real-time genre prediction with confidence scores
- Dynamic radar chart vs genre average
- Top 5 nearest songs from the dataset by cosine similarity
- EDA plots and model comparison accessible in tabs

---

## Streamlit Cloud Deployment

1. Push the project to a GitHub repository.
2. Ensure `outputs/models/best_model.pkl` and `data/songs_normalize.csv` are committed (or use DVC/LFS for large files).
3. Go to https://streamlit.io/cloud and connect your GitHub repo.
4. Set the main file as `app.py`.
5. Deploy.

---

## Dataset Credits

Top Hits Spotify from 2000-2019
Source: https://www.kaggle.com/code/varunsaikanuri/spotify-data-visualization/input

---

## Tech Stack

- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost
- plotly, streamlit
