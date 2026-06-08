import pandas as pd

def test_dataset_exists():

    df = pd.read_csv(
        "data/processed/processed_20260525_055827.csv"
    )

    assert len(df) > 0


def test_required_columns():

    df = pd.read_csv(
        "data/processed/processed_20260525_055827.csv"
    )

    assert "clean_text" in df.columns
    assert "sentiment" in df.columns