from fastapi import FastAPI

from app.schemas import (
    PredictionRequest,
    PredictionResponse
)

from app.predictor import (
    predict_sentiment
)

app = FastAPI(
    title="Sentiment Analysis API"
)


@app.get("/")
def root():

    return {
        "message": "API running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: PredictionRequest
):

    sentiment = predict_sentiment(
        request.text
    )

    return {
        "sentiment": sentiment
    }