from minio import Minio

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

BUCKET_NAME = "sentiment-data"

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)


def upload_file(
    local_path: str,
    object_name: str
):

    client.fput_object(
        BUCKET_NAME,
        object_name,
        local_path
    )

    print(
        f"Uploaded {local_path} -> {object_name}"
    )


def download_file(
    object_name: str,
    local_path: str
):

    client.fget_object(
        BUCKET_NAME,
        object_name,
        local_path
    )

    print(
        f"Downloaded {object_name}"
    )
