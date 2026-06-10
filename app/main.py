from fastapi import FastAPI
from fastapi.responses import Response

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from time import time

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

# =====================================
# Metrics
# =====================================

REQUEST_COUNT = Counter(
    "request_count_total",
    "Total prediction requests"
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Prediction latency"
)

POSITIVE_COUNT = Counter(
    "prediction_positive_total",
    "Positive predictions"
)

NEGATIVE_COUNT = Counter(
    "prediction_negative_total",
    "Negative predictions"
)

NEUTRAL_COUNT = Counter(
    "prediction_neutral_total",
    "Neutral predictions"
)

# =====================================

@app.get("/")
def root():

    return {
        "message": "API running"
    }


@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: PredictionRequest
):

    REQUEST_COUNT.inc()

    start = time()

    sentiment = predict_sentiment(
        request.text
    )

    latency = time() - start

    REQUEST_LATENCY.observe(
        latency
    )

    if sentiment == "positive":
        POSITIVE_COUNT.inc()

    elif sentiment == "negative":
        NEGATIVE_COUNT.inc()

    else:
        NEUTRAL_COUNT.inc()

    return {
        "sentiment": sentiment
    }
