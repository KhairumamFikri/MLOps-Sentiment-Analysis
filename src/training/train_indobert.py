import pandas as pd
import mlflow
import mlflow.transformers
import transformers

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
print("Transformers Version:", transformers.__version__)
# =========================
# CONFIG
# =========================

MODEL_NAME = "distilbert-base-multilingual-cased"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/processed/processed_indobert_20260525_055827.csv"
)

mapping = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label"] = df["sentiment"].map(mapping)

# pastikan tidak ada label kosong
df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)

print("\nDistribusi Label:")
print(df["sentiment"].value_counts())

# =========================
# SPLIT
# =========================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"\nTrain: {len(train_df)}")
print(f"Test : {len(test_df)}")

# =========================
# DATASET
# =========================

train_dataset = Dataset.from_pandas(
    train_df[["clean_text", "label"]]
)

test_dataset = Dataset.from_pandas(
    test_df[["clean_text", "label"]]
)

# =========================
# TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

def tokenize(batch):

    return tokenizer(
        batch["clean_text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)

train_dataset = train_dataset.remove_columns(
    ["clean_text"]
)

test_dataset = test_dataset.remove_columns(
    ["clean_text"]
)

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# =========================
# MODEL
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    ignore_mismatched_sizes=True
)

# =========================
# METRIC
# =========================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = logits.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),
        "f1": f1_score(
            labels,
            predictions,
            average="macro"
        )
    }

# =========================
# TRAINING ARGUMENT
# =========================

args = TrainingArguments(
    output_dir="outputs",

    num_train_epochs=2,

    learning_rate=2e-5,

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_steps=10,

    report_to="none"
)

# =========================
# TRAINER
# =========================

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# =========================
# MLFLOW
# =========================

mlflow.set_experiment(
    "indobert-sentiment"
)

with mlflow.start_run():

    mlflow.log_param(
        "model",
        MODEL_NAME
    )

    mlflow.log_param(
        "epochs",
        2
    )

    mlflow.log_param(
        "batch_size",
        2
    )

    print("\nTraining dimulai...\n")

    trainer.train()

    metrics = trainer.evaluate()

    print("\nHasil Evaluasi:")
    print(metrics)

    mlflow.log_metric(
        "accuracy",
        metrics["eval_accuracy"]
    )

    mlflow.log_metric(
        "f1",
        metrics["eval_f1"]
    )

    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer
        },
        artifact_path="model",
        registered_model_name="indobert-sentiment"
    )

print("\nTraining selesai.")