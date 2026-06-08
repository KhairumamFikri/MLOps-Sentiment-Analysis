import mlflow

client = mlflow.MlflowClient(
    tracking_uri="http://localhost:5000"
)

for v in client.search_model_versions(
    "name='sentence-transformer-sentiment'"
):
    print(v.version)