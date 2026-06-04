import pandas as pd
import random
import os
from datetime import datetime

# =========================
# DATA POSITIVE
# =========================

positive_samples = [

    "program bagus",
    "program sangat membantu",
    "maju terus pemerintah",
    "kerja bagus",
    "mantap programnya",
    "sangat bermanfaat",
    "rakyat sangat terbantu",
    "program keren",
    "semoga terus berjalan",
    "luar biasa pemerintah",
    "gas terus",
    "program berhasil",
    "sangat bagus",
    "kerja nyata",
    "kebijakan bagus",
    "terima kasih pemerintah",
    "anak anak jadi terbantu",
    "program ini hebat",
    "semangat terus",
    "saya mendukung program ini"

]

# =========================
# DATA NEGATIVE
# =========================

negative_samples = [

    "program jelek",
    "program gagal",
    "anggaran mubazir",
    "tidak bermanfaat",
    "program buruk",
    "tidak jelas",
    "banyak korupsi",
    "program kacau",
    "tidak membantu rakyat",
    "program mengecewakan",
    "kebijakan gagal",
    "rakyat dirugikan",
    "program hancur",
    "sangat buruk",
    "tidak efektif",
    "uang rakyat habis",
    "program tidak berguna",
    "pelaksanaannya buruk",
    "program penuh masalah",
    "banyak penyelewengan"

]

# =========================
# GENERATE DATA
# =========================

rows = []

# Generate Positive
for _ in range(100):

    text = random.choice(
        positive_samples
    )

    rows.append({

        "clean_text": text,
        "sentiment": "positive"

    })

# Generate Negative
for _ in range(100):

    text = random.choice(
        negative_samples
    )

    rows.append({

        "clean_text": text,
        "sentiment": "negative"

    })

# Shuffle dataset
random.shuffle(rows)

# =========================
# SAVE DATASET
# =========================

os.makedirs(
    "data/augmentation",
    exist_ok=True
)

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_file = (
    f"data/augmentation/"
    f"balanced_short_sentiment_{timestamp}.csv"
)

df = pd.DataFrame(rows)

df.to_csv(
    output_file,
    index=False
)

print("Dataset augmentasi berhasil dibuat.")
print(f"Jumlah data: {len(df)}")
print(f"Disimpan di: {output_file}")