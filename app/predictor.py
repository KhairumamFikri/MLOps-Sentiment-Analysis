import mlflow.pyfunc
from sentence_transformers import SentenceTransformer

MODEL_URI = (
    "models:/sentence-transformer-sentiment/Production"
)

embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

classifier = None


def load_model():

    global classifier

    if classifier is None:

        classifier = mlflow.pyfunc.load_model(
            MODEL_URI
        )

    return classifier


def predict_sentiment(text):

    model = load_model()

    embedding = embedder.encode(
        [text]
    )

    prediction = model.predict(
        embedding
    )

    return prediction[0]