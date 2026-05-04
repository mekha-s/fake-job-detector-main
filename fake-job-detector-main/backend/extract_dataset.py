import pandas as pd
import zipfile
import os
import re
import random

# =====================================================
# DATASET FOLDER
# =====================================================

DATASET_FOLDER = "dataset"

# =====================================================
# ZIP FILES INSIDE DATASET FOLDER
# =====================================================

ZIP_FILES = [
    "fake_job_postings.csv (1).zip",
    "postings.csv.zip"
]

# =====================================================
# CSV FILE NAMES AFTER EXTRACTION
# =====================================================

FAKE_FILE = "fake_job_postings.csv"
LINKEDIN_FILE = "postings.csv"
SALARY_FILE = "ds_salaries.csv"

# =====================================================
# STEP 1 — EXTRACT ZIP FILES
# =====================================================

print("\nExtracting datasets...")

for zip_name in ZIP_FILES:

    zip_path = os.path.join(DATASET_FOLDER, zip_name)

    if os.path.exists(zip_path):

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_FOLDER)

        print("Extracted:", zip_name)

    else:
        print("Zip not found:", zip_name)

# =====================================================
# STEP 2 — LOAD DATASETS
# =====================================================

print("\nLoading datasets...")

fake_df = pd.read_csv(os.path.join(DATASET_FOLDER, FAKE_FILE))
linkedin_df = pd.read_csv(os.path.join(DATASET_FOLDER, LINKEDIN_FILE))

print("Fake jobs:", fake_df.shape)
print("LinkedIn jobs:", linkedin_df.shape)

# =====================================================
# CLEAN TEXT FUNCTION
# =====================================================

def clean_text(text):

    text = str(text)

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()

# =====================================================
# CLEAN FAKE JOB DATASET
# =====================================================

print("\nCleaning fake job dataset...")

fake_df = fake_df[["description", "fraudulent"]]

fake_df = fake_df.rename(columns={"fraudulent": "label"})

fake_df = fake_df.dropna(subset=["description"])

fake_df["description"] = fake_df["description"].astype(str)

fake_df["description"] = fake_df["description"].apply(clean_text)

fake_df = fake_df.drop_duplicates(subset=["description"])

fake_df = fake_df[fake_df["description"].str.len() > 40]

fake_df["label"] = 1

print("Fake cleaned:", fake_df.shape)

# =====================================================
# GENERATE SYNTHETIC FAKE JOBS
# =====================================================

print("\nGenerating additional fake jobs...")

fake_templates = [
"urgent hiring data entry operator work from home salary weekly contact whatsapp recruiter {}",
"immediate hiring python developer remote job telegram recruiter registration fee required {}",
"work from home online typing job easy payment daily apply through whatsapp {}",
"urgent software job no interview instant joining contact telegram recruiter {}",
"online part time data entry job for students work from home registration fee required {}"
]

synthetic_fake = []

for i in range(20000):

    template = random.choice(fake_templates)

    synthetic_fake.append({
        "description": template.format(i),
        "label": 1
    })

synthetic_fake_df = pd.DataFrame(synthetic_fake)

fake_df = pd.concat([fake_df, synthetic_fake_df])

print("Fake dataset after augmentation:", fake_df.shape)

# =====================================================
# CLEAN LINKEDIN DATASET
# =====================================================

print("\nCleaning LinkedIn dataset...")

linkedin_df = linkedin_df[["description"]]

linkedin_df = linkedin_df.dropna(subset=["description"])

linkedin_df["description"] = linkedin_df["description"].astype(str)

linkedin_df["description"] = linkedin_df["description"].apply(clean_text)

linkedin_df = linkedin_df.drop_duplicates(subset=["description"])

linkedin_df = linkedin_df[linkedin_df["description"].str.len() > 120]

linkedin_df["label"] = 0

# keep manageable size
linkedin_df = linkedin_df.sample(30000, random_state=42)

print("LinkedIn cleaned:", linkedin_df.shape)

# =====================================================
# MERGE DATASETS
# =====================================================

print("\nMerging datasets...")

dataset = pd.concat([fake_df, linkedin_df])

dataset = dataset.drop_duplicates(subset=["description"])

dataset = dataset.sample(frac=1).reset_index(drop=True)

print("Merged dataset:", dataset.shape)

# =====================================================
# CREATE 30K BALANCED DATASET
# =====================================================

print("\nCreating balanced dataset...")

real = dataset[dataset.label == 0].sample(15000, random_state=42)
fake = dataset[dataset.label == 1].sample(15000, random_state=42)

dataset = pd.concat([real, fake])

dataset = dataset.sample(frac=1).reset_index(drop=True)

print("Balanced dataset:", dataset.shape)

# =====================================================
# SAVE DATASET
# =====================================================

OUTPUT_FILE = "balanced_it_jobs_dataset_30k.csv"

dataset.to_csv(OUTPUT_FILE, index=False)

print("\nDataset saved:", OUTPUT_FILE)