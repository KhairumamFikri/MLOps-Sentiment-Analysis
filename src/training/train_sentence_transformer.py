import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/processed/processed_20260525_055827.csv"
)

df = df.dropna(
    subset=["clean_text", "sentiment"]
)

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

# =========================
# EMBEDDING MODEL
# =========================

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Generating embeddings...")

X_train_emb = embedding_model.encode(
    X_train.tolist(),
    show_progress_bar=True
)

X_test_emb = embedding_model.encode(
    X_test.tolist(),
    show_progress_bar=True
)

# =========================
# LOGISTIC REGRESSION
# =========================

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=2.0,
    solver="lbfgs"
)

# =========================
# MLFLOW
# =========================

mlflow.set_experiment(
    "sentence-transformer-sentiment"
)

with mlflow.start_run():

    mlflow.log_param(
        "embedding_model",
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    mlflow.log_param(
        "classifier",
        "LogisticRegression"
    )

    model.fit(
        X_train_emb,
        y_train
    )

    y_pred = model.predict(
        X_test_emb
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    print("\nAccuracy:", accuracy)
    print("\nF1:", f1)

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="sentence-transformer-sentiment"
    )

# =========================
# SAVE LOCAL
# =========================

joblib.dump(
    model,
    "models/sentence_lr.pkl"
)

print("\nModel saved.")