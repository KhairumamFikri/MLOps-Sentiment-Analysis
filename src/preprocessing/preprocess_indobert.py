import pandas as pd
import glob
import os
import re
from datetime import datetime

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =========================
# LOAD DATA TERBARU
# =========================

list_files = glob.glob(
    f"{RAW_DIR}/*.csv"
)

if not list_files:
    raise Exception(
        "Tidak ada dataset raw."
    )

latest_file = max(
    list_files,
    key=os.path.getctime
)

print(
    f"Membaca: {latest_file}"
)

df = pd.read_csv(
    latest_file
)

# =========================
# NORMALISASI RINGAN
# =========================

NORMALIZATION = {

    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",

    "tdk": "tidak",

    "bgt": "banget",

    "tp": "tapi",

    "yg": "yang",

    "krn": "karena",

    "utk": "untuk",

    "dr": "dari",

    "trmksh": "terima kasih",
    
}

# =========================
# CLEAN FUNCTION
# =========================

def clean_text(text):

    text = str(text)

    text = text.lower()

    # hapus url
    text = re.sub(
        r"http\S+",
        "",
        text
    )

    # hapus mention
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # rapikan spasi
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    tokens = []

    for word in text.split():

        tokens.append(
            NORMALIZATION.get(
                word,
                word
            )
        )

    return " ".join(
        tokens
    ).strip()

# =========================
# CLEANING
# =========================

df = df.dropna(
    subset=["text"]
)

df["clean_text"] = (
    df["text"]
    .apply(clean_text)
)

# hapus kosong
df = df[
    df["clean_text"]
    .str.strip() != ""
]

# hanya ambil clean_text
df = df[
    ["clean_text"]
]

# buat kolom label kosong
df["sentiment"] = ""

# ambil maksimum 1000 random
if len(df) > 1000:

    df = df.sample(
        n=1000,
        random_state=42
    )

df = df.reset_index(
    drop=True
)

# =========================
# SAVE
# =========================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_file = (
    f"{OUTPUT_DIR}/"
    f"processed_indobert_{timestamp}.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\n=== SELESAI ===")

print(
    f"Jumlah data: {len(df)}"
)

print(
    f"Output: {output_file}"
)

print(
    "\nKolom:"
)

print(
    df.columns.tolist()
)