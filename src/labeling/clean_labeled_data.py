import pandas as pd
import glob
import os
from datetime import datetime

LABELED_DIR = "data/labeled"

# =========================
# AMBIL FILE TERBARU
# =========================

list_files = glob.glob(f"{LABELED_DIR}/*.csv")

if not list_files:
    raise Exception("Tidak ada labeled dataset!")

latest_file = max(list_files, key=os.path.getctime)

print(f"Membaca file: {latest_file}")

df = pd.read_csv(
    latest_file,
    encoding="utf-8",
    on_bad_lines="skip"
)

# =========================
# PILIH KOLOM YANG DIPERLUKAN
# =========================

required_columns = [
    "clean_text",
    "sentiment"
]

df = df[required_columns]

# =========================
# HAPUS DATA KOSONG
# =========================

df = df.dropna()

df = df[df["clean_text"].str.strip() != ""]

# =========================
# HAPUS DUPLIKAT
# =========================

df = df.drop_duplicates(
    subset=["clean_text"]
)

# =========================
# LIMIT DATA
# =========================

MAX_DATA = 1000

df = df.head(MAX_DATA)

# =========================
# RESET INDEX
# =========================

df = df.reset_index(drop=True)

# =========================
# SAVE
# =========================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_file = (
    f"{LABELED_DIR}/final_labeled_{timestamp}.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("Dataset final berhasil dibuat.")
print(f"Jumlah data: {len(df)}")
print(f"Disimpan ke: {output_file}")