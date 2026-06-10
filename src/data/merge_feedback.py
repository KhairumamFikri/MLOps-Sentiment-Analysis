import pandas as pd

from src.utils.minio_utils import (
    download_file,
    upload_file
)

# =========================
# Download Dataset Utama
# =========================

download_file(
    "training/processed_20260525_055827.csv",
    "dataset.csv"
)

# =========================
# Download Feedback
# =========================

download_file(
    "feedback/feedback_dataset.csv",
    "feedback_dataset.csv"
)

# =========================
# Load Data
# =========================

dataset = pd.read_csv(
    "dataset.csv"
)

feedback = pd.read_csv(
    "feedback_dataset.csv"
)

print(
    f"Dataset lama: {len(dataset)}"
)

print(
    f"Feedback baru: {len(feedback)}"
)

# =========================
# Merge
# =========================

merged = pd.concat(
    [dataset, feedback],
    ignore_index=True
)

merged.drop_duplicates(
    subset=["clean_text", "sentiment"],
    inplace=True
)

print(
    f"Dataset gabungan: {len(merged)}"
)

# =========================
# Save
# =========================

merged.to_csv(
    "dataset_latest.csv",
    index=False
)

# =========================
# Upload ke MinIO
# =========================

upload_file(
    "dataset_latest.csv",
    "processed/dataset_latest.csv"
)

print(
    "Dataset terbaru berhasil diupload"
)
