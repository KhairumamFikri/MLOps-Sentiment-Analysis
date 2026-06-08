import os
import mlflow
import pandas as pd

os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

mlflow.set_tracking_uri(
    "http://localhost:5000"
)

model = mlflow.pyfunc.load_model(
    "models:/sentence-transformer-sentiment@production"
)

while True:

    text = input(
        "\nMasukkan kalimat (ketik exit): "
    )

    if text.lower() == "exit":
        break

    prediction = model.predict(
        pd.DataFrame({
            "text": [text]
        })
    )

    print(
        f"Prediksi: {prediction[0]}"
    )