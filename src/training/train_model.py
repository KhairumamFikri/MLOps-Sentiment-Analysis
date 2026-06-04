import os
import glob
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# =========================
# LOAD DATASET
# =========================

LABELED_DIR = "data/labeled"

list_files = glob.glob(
    f"{LABELED_DIR}/*.csv"
)

if not list_files:
    raise Exception(
        "Tidak ada labeled dataset!"
    )

DATASET_PATH = (
    "data/processed/processed_comments_20260516_110522.csv"
)

print(f"Dataset: {DATASET_PATH}")

df = pd.read_csv(
    DATASET_PATH,
    on_bad_lines="skip"
)
# =========================
# CLEAN DATA
# =========================

df = df.dropna(
    subset=["clean_text", "sentiment"]
)

df["clean_text"] = (
    df["clean_text"]
    .astype(str)
    .str.strip()
)

df["sentiment"] = (
    df["sentiment"]
    .astype(str)
    .str.strip()
)

df = df[
    (df["clean_text"] != "") &
    (df["sentiment"] != "")
]

# =========================
# FEATURES & LABEL
# =========================

X = df["clean_text"]
y = df["sentiment"]

# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# EXPERIMENT PARAMETER
# =========================

MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
C = 0.5

# =========================
# SET EXPERIMENT
# =========================

mlflow.set_experiment(
    "sentiment-analysis"
)

# =========================
# START RUN
# =========================

DVC_METADATA_FILE = (
    "data/labeled/final_labeled_v2.csv.dvc"
)

with mlflow.start_run():

    # =========================
    # LOG DVC METADATA
    # =========================

    with open(
        DVC_METADATA_FILE,
        "r"
    ) as f:

        dvc_metadata = f.read()

    mlflow.log_text(
        dvc_metadata,
        "dataset_version.txt"
    )

    # =========================
    # LOG PARAMETERS
    # =========================

    mlflow.log_param(
        "max_features",
        MAX_FEATURES
    )

    mlflow.log_param(
        "ngram_range",
        str(NGRAM_RANGE)
    )

    mlflow.log_param(
        "C",
        C
    )

    # =========================
    # PIPELINE
    # =========================

    pipeline = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                max_features=MAX_FEATURES,
                ngram_range=NGRAM_RANGE,
                lowercase=True
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=C,
                random_state=42
            )
        )

    ])

    print("Training pipeline model...")

    # =========================
    # TRAINING
    # =========================

    pipeline.fit(
        X_train,
        y_train
    )

    # =========================
    # PREDICTION
    # =========================

    y_pred = pipeline.predict(
        X_test
    )

    # =========================
    # METRICS
    # =========================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    print("\n=== HASIL EVALUASI ===")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # =========================
    # LOG METRICS
    # =========================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    # =========================
    # SAVE LOCAL MODEL
    # =========================

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        "models/sentiment_pipeline.pkl"
    )

    # =========================
    # LOG MODEL TO MLFLOW
    # =========================

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name="sentiment-analysis-model"
    )

    print("\nModel berhasil disimpan.")