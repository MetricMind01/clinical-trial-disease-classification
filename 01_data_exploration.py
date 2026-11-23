"""Step 1: Load the raw clinical trial data and explore it."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_FOLDER, "data", "clinical_trials_raw_patient2trial_conditions.csv")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("Loading the clinical trial dataset...")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "Dataset not found. Put clinical_trials_raw_patient2trial_conditions.csv "
        "inside the data folder."
    )

df = pd.read_csv(DATA_FILE, low_memory=False)

print("\n1. Dataset size")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. Column names")
print(df.columns.tolist())

print("\n3. Data types")
print(df.dtypes)

print("\n4. Missing values")
print(df.isnull().sum().sort_values(ascending=False))

print("\n5. Duplicate records")
print("Duplicate complete rows:", df.duplicated().sum())
if "nct_id" in df.columns:
    print("Duplicate trial IDs:", df["nct_id"].duplicated().sum())

print("\n6. Disease category counts")
category_counts = df["source_condition_query"].value_counts()
print(category_counts)

# Chart 1: number of trials in each category.
plt.figure(figsize=(11, 6))
plt.barh(category_counts.index, category_counts.values)
plt.title("Clinical Trials by Disease Category")
plt.xlabel("Number of trials")
plt.ylabel("Disease category")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "category_distribution.png"), dpi=300)
plt.close()

# Chart 2: length of each brief summary.
df["summary_word_count"] = df["brief_summary"].fillna("").astype(str).str.split().str.len()
plt.figure(figsize=(10, 5))
plt.hist(df["summary_word_count"], bins=40)
plt.title("Length of Clinical Trial Brief Summaries")
plt.xlabel("Number of words")
plt.ylabel("Number of trials")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "summary_length_distribution.png"), dpi=300)
plt.close()

# Chart 3: most common words in the brief summaries.
summary_text = df["brief_summary"].fillna("").astype(str)
word_vectorizer = CountVectorizer(stop_words="english", max_features=20)
word_matrix = word_vectorizer.fit_transform(summary_text)
word_totals = word_matrix.sum(axis=0).A1
word_names = word_vectorizer.get_feature_names_out()
word_table = pd.DataFrame({"word": word_names, "count": word_totals})
word_table = word_table.sort_values("count", ascending=False)

plt.figure(figsize=(11, 6))
plt.barh(word_table["word"], word_table["count"])
plt.title("20 Most Frequent Words in Brief Summaries")
plt.xlabel("Word frequency")
plt.ylabel("Word")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "most_common_words.png"), dpi=300)
plt.close()

# Save a small table so the results can be used in the final report.
category_counts.rename_axis("disease_category").reset_index(name="trial_count").to_csv(
    os.path.join(OUTPUT_FOLDER, "category_counts.csv"), index=False
)
word_table.to_csv(os.path.join(OUTPUT_FOLDER, "most_common_words.csv"), index=False)

print("\nEDA completed.")
print("Charts and summary files have been saved in the outputs folder.")
