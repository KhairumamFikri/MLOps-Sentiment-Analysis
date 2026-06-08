import os
import json
import joblib
import pandas as pd

import mlflow
import mlflow.pyfunc

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# ==================================================
# CUSTOM PYFUNC MODEL
# ==================================================

class SentimentPipeline(mlflow.pyfunc.PythonModel):

    def load_context(self, context):

        self.encoder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.classifier = joblib.load(
            context.artifacts["classifier"]
        )

    def predict(
        self,
        context,
        model_input
    ):

        texts = model_input["text"].tolist()

        embeddings = self.encoder.encode(
            texts
        )

        predictions = self.classifier.predict(
            embeddings
        )

        return predictions


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(
    "data/processed/processed_20260525_055827.csv"
)

df = df.dropna(
    subset=["clean_text", "sentiment"]
)

# ==================================================
# SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

# ==================================================
# EMBEDDING
# ==================================================

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

# ==================================================
# CLASSIFIER
# ==================================================

classifier = LogisticRegression(
    max_iter=2500,
    class_weight="balanced",
    C=3.0,
    solver="lbfgs"
)

classifier.fit(
    X_train_emb,
    y_train
)

# ==================================================
# EVALUATION
# ==================================================

y_pred = classifier.predict(
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

print("\nAccuracy:", accuracy)
print("\nF1:", f1)

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==================================================
# SAVE TEMP CLASSIFIER
# ==================================================

os.makedirs(
    "artifacts",
    exist_ok=True
)

classifier_path = (
    "artifacts/logistic_classifier.pkl"
)

joblib.dump(
    classifier,
    classifier_path
)

# ==================================================
# MLFLOW CONFIG
# ==================================================

os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

mlflow.set_tracking_uri(
    "http://localhost:5000"
)

mlflow.set_experiment(
    "sentence-transformer-sentiment"
)

# ==================================================
# LOG TO MLFLOW
# ==================================================

with mlflow.start_run():

    mlflow.log_param(
        "embedding_model",
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    mlflow.log_param(
        "classifier",
        "LogisticRegression"
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    metrics = {
        "accuracy": float(accuracy),
        "f1_score": float(f1)
    }

    with open(
        "metrics.json",
        "w"
    ) as f:
        json.dump(metrics, f)

    mlflow.log_artifact(
        "metrics.json"
    )

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SentimentPipeline(),
        artifacts={
            "classifier": classifier_path
        },
        registered_model_name="sentence-transformer-sentiment"
    )

print(
    "\n✅ Model berhasil diregister ke MLflow"
)