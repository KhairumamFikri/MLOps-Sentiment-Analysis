import pandas as pd
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

LABELED_DIR = "data/labeled"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================

list_files = glob.glob(f"{LABELED_DIR}/*.csv")

if not list_files:
    raise Exception("Tidak ada labeled dataset!")

latest_file = max(list_files, key=os.path.getctime)

print(f"Membaca dataset: {latest_file}")

df = pd.read_csv(
    latest_file,
    on_bad_lines="skip"
)
# =========================
# LOAD FEEDBACK DATA
# =========================

feedback_file = "data/feedback/feedback.csv"

if os.path.exists(feedback_file):

    print("Menggabungkan feedback dataset...")

    feedback_df = pd.read_csv(feedback_file)

    df = pd.concat(
        [df, feedback_df],
        ignore_index=True
    )

    # Hapus duplicate
    df = df.drop_duplicates(
        subset=["clean_text"]
    )
# =========================
# CLEAN DATA
# =========================

df = df.dropna(subset=["clean_text", "sentiment"])

df = df[df["clean_text"].str.strip() != ""]

# =========================
# FEATURES & LABEL
# =========================

X = df["clean_text"]

y = df["sentiment"]

# =========================
# TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_vectorized = vectorizer.fit_transform(X)

# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# TRAIN MODEL
# =========================

model = LogisticRegression(
    max_iter=1000
)

print("Training model...")

model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\n=== HASIL EVALUASI ===")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    model,
    f"{MODEL_DIR}/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    f"{MODEL_DIR}/tfidf_vectorizer.pkl"
)

print("\nModel berhasil disimpan.")