import pandas as pd
import re
import glob
import os
import string
import csv
from datetime import datetime

from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# =========================
# LOAD DATA TERBARU
# =========================

list_files = glob.glob(f"{RAW_DIR}/*.csv")

if not list_files:
    raise Exception("Tidak ada file raw data!")

latest_file = max(list_files, key=os.path.getctime)

print(f"Memproses file: {latest_file}")

df = pd.read_csv(
    latest_file,
    encoding="utf-8",
    on_bad_lines="skip"
)

# =========================
# NORMALIZATION MAP
# =========================

NORMALIZATION_MAP = {
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "tdk": "tidak",
    "kagak": "tidak",

    "bgt": "sangat",
    "banget": "sangat",

    "udah": "sudah",
    "sdh": "sudah",

    "tp": "tapi",
    "tpi": "tapi",

    "klo": "kalau",
    "kalo": "kalau",

    "utk": "untuk",
    "yg": "yang",
    "dgn": "dengan",
    "jd": "jadi",

    "blm": "belum",
    "krn": "karena"
}

# =========================
# STOPWORDS
# =========================

stop_factory = StopWordRemoverFactory()

stopwords = set(
    stop_factory.get_stop_words()
)

NEGATION_WORDS = {
    "tidak",
    "bukan",
    "jangan",
    "belum",
    "kurang",
    "tanpa",
    "tapi",
    "namun"
}

ADDITIONAL_STOPWORDS = {
    "nya",
    "sih",
    "lah",
    "deh",
    "pun",
    "kok"
}

stopwords = (
    stopwords | ADDITIONAL_STOPWORDS
) - NEGATION_WORDS

# =========================
# CLEAN FUNCTION
# =========================

def clean_text(text):

    text = str(text).lower()

    # Hapus URL, mention, hashtag
    text = re.sub(
        r"http\S+|@\w+|#\w+",
        "",
        text
    )

    # JANGAN HAPUS ANGKA

    # Hapus emoji/non-ascii
    text = text.encode(
        "ascii",
        "ignore"
    ).decode()

    # Hapus punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Rapikan whitespace
    text = " ".join(text.split())

    # Tokenization
    tokens = text.split()

    cleaned_tokens = []

    for word in tokens:

        # Normalisasi slang
        word = NORMALIZATION_MAP.get(
            word,
            word
        )

        # Stopword removal
        if word not in stopwords:

            # Pertahankan token > 1 karakter
            if len(word) > 1:

                cleaned_tokens.append(word)

    return " ".join(cleaned_tokens)

# =========================
# DATA CLEANING
# =========================

df = df.dropna(subset=["text"])

df = df.drop_duplicates(
    subset=["text"]
)

print("Melakukan preprocessing text...")

df["clean_text"] = df["text"].apply(
    clean_text
)

# Hapus hasil kosong
df = df[
    df["clean_text"].str.strip() != ""
]

# =========================
# RANDOM SAMPLE 1000 DATA
# =========================

MAX_DATA = 1000

if len(df) > MAX_DATA:

    df = df.sample(
        n=MAX_DATA,
        random_state=42
    )

# =========================
# FORMAT FINAL
# =========================

df = df[["clean_text"]]

# Tambah kolom sentiment kosong
df["sentiment"] = ""

# Reset index
df = df.reset_index(drop=True)

# =========================
# SAVE
# =========================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_file = (
    f"{PROCESSED_DIR}/processed_comments_{timestamp}.csv"
)

df.to_csv(
    output_file,
    index=False,
    quoting=csv.QUOTE_ALL
)

print("\n--- SELESAI ---")
print(f"Jumlah data akhir: {len(df)}")
print(f"Hasil disimpan ke: {output_file}")