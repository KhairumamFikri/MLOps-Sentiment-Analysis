import requests

MODEL_URL = (
    "http://model-serving:1234/invocations"
)

def predict_sentiment(text):

    response = requests.post(
        MODEL_URL,
        json={
            "dataframe_records": [
                {
                    "text": text
                }
            ]
        }
    )

    response.raise_for_status()

    result = response.json()

    return result["predictions"][0]