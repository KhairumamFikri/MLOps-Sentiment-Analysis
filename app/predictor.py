import pandas as pd
import mlflow.pyfunc

MODEL_URI = "models:/sentence-transformer-sentiment/1"

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