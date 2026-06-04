import mlflow
import mlflow.pyfunc
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer

# ==================================
# CUSTOM MODEL
# ==================================

class SentimentPipeline(
    mlflow.pyfunc.PythonModel
):

    def load_context(self, context):

        self.embedder = SentenceTransformer(
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

        embeddings = self.embedder.encode(
            texts
        )

        predictions = self.classifier.predict(
            embeddings
        )

        return predictions

# ==================================
# REGISTER MODEL
# ==================================

mlflow.set_experiment(
    "sentence-transformer-sentiment"
)

with mlflow.start_run():

    mlflow.pyfunc.log_model(

        artifact_path="model",

        python_model=SentimentPipeline(),

        artifacts={
            "classifier":
            "models/sentence_lr.pkl"
        },

        registered_model_name=
        "sentence-transformer-pipeline"
    )

print(
    "Model pipeline berhasil diregister."
)