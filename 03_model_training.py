"""Step 3: Train and compare two machine learning models."""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
CLEAN_FILE = os.path.join(BASE_FOLDER, "outputs", "clinical_trials_cleaned.csv")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "outputs")
MODEL_FOLDER = os.path.join(BASE_FOLDER, "models")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


print("Loading the cleaned dataset...")
if not os.path.exists(CLEAN_FILE):
    raise FileNotFoundError("Run 02_data_preprocessing.py before training the model.")

df = pd.read_csv(CLEAN_FILE)
df = df.dropna(subset=["cleaned_summary", "source_condition_query"])

X = df["cleaned_summary"]
y = df["source_condition_query"]

print("Total records:", len(df))
print("Disease categories:", y.nunique())

# Keep the same category balance in both training and testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# TF-IDF changes text into numbers that a machine learning model can use.
print("Converting text into TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# We are comparing two common models.
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=400,
        class_weight="balanced",
    ),
}

results = []
trained_models = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, predictions, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, predictions, average="macro", zero_division=0)

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
    })
    trained_models[model_name] = model

    print("Accuracy:", round(accuracy, 4))
    print("Weighted F1-score:", round(f1_weighted, 4))
    print("Macro F1-score:", round(f1_macro, 4))

# Select the model with the best macro F1-score.
# Macro F1 gives each disease category equal importance.
results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
results_df.to_csv(os.path.join(OUTPUT_FOLDER, "model_evaluation.csv"), index=False)

best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]
best_predictions = best_model.predict(X_test_tfidf)

print("\nBest model:", best_model_name)
print("\nDetailed classification report:")
report = classification_report(y_test, best_predictions, zero_division=0)
print(report)

with open(os.path.join(OUTPUT_FOLDER, "classification_report.txt"), "w", encoding="utf-8") as file:
    file.write("Best model: " + best_model_name + "\n\n")
    file.write(report)

# Draw a confusion matrix for the selected model.
category_names = sorted(y.unique())
matrix = confusion_matrix(y_test, best_predictions, labels=category_names)

plt.figure(figsize=(10, 8))
plt.imshow(matrix, interpolation="nearest")
plt.title("Confusion Matrix - " + best_model_name)
plt.colorbar()
plt.xticks(range(len(category_names)), category_names, rotation=45, ha="right")
plt.yticks(range(len(category_names)), category_names)
plt.xlabel("Predicted category")
plt.ylabel("Actual category")

# Write the number inside each square so the chart is easy to read.
for row in range(len(category_names)):
    for column in range(len(category_names)):
        plt.text(column, row, str(matrix[row, column]), ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"), dpi=300)
plt.close()

# Save the vectorizer and model together for the Streamlit application.
model_bundle = {
    "vectorizer": vectorizer,
    "model": best_model,
    "best_model_name": best_model_name,
}
joblib.dump(model_bundle, os.path.join(MODEL_FOLDER, "disease_category_model.pkl"))

print("\nModel training completed.")
print("Evaluation results saved in the outputs folder.")
print("Model saved in the models folder.")
