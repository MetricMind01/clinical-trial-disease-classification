"""Step 2: Clean the clinical trial text and save a smaller dataset."""

import os
import pandas as pd
from text_preprocessing import clean_text


BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_FOLDER, "data", "clinical_trials_raw_patient2trial_conditions.csv")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "outputs")
CLEAN_FILE = os.path.join(OUTPUT_FOLDER, "clinical_trials_cleaned.csv")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("Loading raw data...")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "Dataset not found. Put clinical_trials_raw_patient2trial_conditions.csv "
        "inside the data folder."
    )

df = pd.read_csv(DATA_FILE, low_memory=False)
original_rows = len(df)

# Keep only the columns needed for this project.
required_columns = ["nct_id", "title", "brief_summary", "source_condition_query"]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"The dataset is missing these columns: {missing_columns}")

df = df[required_columns].copy()

# Fill empty text values, then remove rows without a summary or a category.
df["brief_summary"] = df["brief_summary"].fillna("").astype(str)
df["source_condition_query"] = df["source_condition_query"].fillna("").astype(str).str.strip()
df["title"] = df["title"].fillna("").astype(str)

df = df[df["brief_summary"].str.strip() != ""]
df = df[df["source_condition_query"] != ""]

# Remove duplicate trial IDs so that the same trial is not counted twice.
df = df.drop_duplicates(subset="nct_id")

print("Cleaning the brief summaries. This can take a little time...")
df["cleaned_summary"] = df["brief_summary"].apply(clean_text)

# Remove any summary that becomes empty after cleaning.
df = df[df["cleaned_summary"].str.strip() != ""]

df.to_csv(CLEAN_FILE, index=False)

print("\nPreprocessing completed.")
print("Rows before cleaning:", original_rows)
print("Rows after cleaning:", len(df))
print("Rows removed:", original_rows - len(df))
print("Number of disease categories:", df["source_condition_query"].nunique())
print("\nExample of cleaned text:")
print(df["cleaned_summary"].iloc[0][:500])
print("\nCleaned dataset saved to:", CLEAN_FILE)
