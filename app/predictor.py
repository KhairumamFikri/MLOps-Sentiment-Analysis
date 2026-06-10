import pandas as pd
import mlflow.pyfunc
import mlflow

mlflow.set_tracking_uri(
    "http://mlflow:5000"
)

MODEL_URI = "models:/sentence-transformer-sentiment@production"

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

    prediction = model.predict(
        pd.DataFrame({
            "text": [text]
        })
    )

    return prediction[0]
