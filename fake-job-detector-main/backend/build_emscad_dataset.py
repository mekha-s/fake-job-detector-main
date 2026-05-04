"""
Build a balanced training dataset from raw EMSCAD (Employment Scam Aegean
Dataset). Replaces the synthetic-template fakes in the original
extract_dataset.py with real labelled scams.

Steps:
  1. Read data/emscad.csv (17,880 rows, 866 fakes, 17,014 reals).
  2. Concatenate title + company_profile + description + requirements
     into one `description` column matching the original training format.
  3. Take all 866 fakes; sample 866 reals (1:1 balance) and a larger
     "validation real" pool from the remainder.
  4. Output:
       data/emscad_balanced.csv   1:1 train+val pool (~1732 rows)
       data/emscad_test_real.csv  Held-out real postings (~3000 rows)
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = DATA / "emscad.csv"
OUT_BALANCED = DATA / "emscad_balanced.csv"
OUT_TEST = DATA / "emscad_test_real.csv"


def clean(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def main():
    print(f"Loading {SRC} ...")
    df = pd.read_csv(SRC)
    print(f"  raw shape: {df.shape}")
    print(f"  fraud distribution: {df['fraudulent'].value_counts().to_dict()}")

    # Concatenate the most informative free-text fields into one description
    parts = [
        df["title"].fillna(""),
        df["company_profile"].fillna(""),
        df["description"].fillna(""),
        df["requirements"].fillna(""),
        df["benefits"].fillna(""),
    ]
    df["description_full"] = parts[0]
    for p in parts[1:]:
        df["description_full"] = df["description_full"] + " " + p
    df["description_full"] = df["description_full"].apply(clean)

    df = df[df["description_full"].str.len() > 60]
    df = df.drop_duplicates(subset=["description_full"])
    print(f"  after cleaning + dedup: {df.shape}")

    fakes = df[df["fraudulent"] == 1].reset_index(drop=True)
    reals = df[df["fraudulent"] == 0].reset_index(drop=True)
    print(f"  fakes available: {len(fakes)}  reals available: {len(reals)}")

    # 1:1 balance for training
    n = len(fakes)
    reals_sampled = reals.sample(n=n, random_state=42).reset_index(drop=True)
    reals_remaining = reals.drop(reals_sampled.index, errors="ignore").reset_index(drop=True)

    balanced = pd.concat([fakes, reals_sampled], ignore_index=True)
    balanced = balanced.rename(columns={"fraudulent": "label"})
    balanced = balanced[["description_full", "label"]].rename(
        columns={"description_full": "description"}
    )
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced.to_csv(OUT_BALANCED, index=False)
    print(f"  wrote {OUT_BALANCED}  shape={balanced.shape}  "
          f"label_dist={balanced['label'].value_counts().to_dict()}")

    # Held-out "real" pool for ad-hoc testing of false-positive rate
    n_test_real = min(3000, len(reals_remaining))
    test_real = reals_remaining.sample(n=n_test_real, random_state=24).rename(
        columns={"fraudulent": "label"}
    )
    test_real = test_real[["description_full", "label"]].rename(
        columns={"description_full": "description"}
    ).reset_index(drop=True)
    test_real.to_csv(OUT_TEST, index=False)
    print(f"  wrote {OUT_TEST}  shape={test_real.shape}")


if __name__ == "__main__":
    main()
