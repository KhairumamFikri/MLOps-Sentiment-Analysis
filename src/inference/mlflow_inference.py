import mlflow
import pandas as pd

model = mlflow.pyfunc.load_model(
    "models:/sentence-transformer-pipeline@production"
)

print(
    "Production model berhasil dimuat."
)

while True:

    text = input(
        "\nMasukkan kalimat (ketik 'exit'): "
    )

    if text.lower() == "exit":
        break

    prediction = model.predict(

        pd.DataFrame({
            "text": [text]
        })

    )[0]

    print(
        f"Prediksi: {prediction}"
    )