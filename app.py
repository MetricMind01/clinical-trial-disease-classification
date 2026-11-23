import os
import joblib
import pandas as pd
import streamlit as st

from text_preprocessing import clean_text


# File locations
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_FOLDER, "models", "disease_category_model.pkl")
CLEAN_FILE = os.path.join(BASE_FOLDER, "outputs", "clinical_trials_cleaned.csv")
METRICS_FILE = os.path.join(BASE_FOLDER, "outputs", "model_evaluation.csv")
REPORT_FILE = os.path.join(BASE_FOLDER, "outputs", "classification_report.txt")
CONFUSION_FILE = os.path.join(BASE_FOLDER, "outputs", "confusion_matrix.png")


# Page settings
st.set_page_config(
    page_title="Clinical Trial Category Classifier",
    page_icon="🧬",
    layout="wide"
)

st.title("Clinical Trial Disease Category Classification")
st.write(
    "Enter a clinical trial summary to predict its dataset category. "
    "You can also explore the dataset and check the model results."
)

st.caption(
    "This project predicts the category attached to a clinical trial record. "
    "It does not provide a medical diagnosis."
)

st.divider()


# Load the saved model
if not os.path.exists(MODEL_FILE):
    st.error("Model file not found. Please run 03_model_training.py first.")
    st.stop()

try:
    model_bundle = joblib.load(MODEL_FILE)
    vectorizer = model_bundle["vectorizer"]
    model = model_bundle["model"]
    best_model_name = model_bundle["best_model_name"]
except Exception as error:
    st.error("There was a problem loading the saved model.")
    st.write(error)
    st.stop()


# Create the three main sections
prediction_tab, insights_tab, performance_tab = st.tabs(
    ["Prediction", "Dataset Insights", "Model Performance"]
)


# --------------------------------------------------
# TAB 1: PREDICTION
# --------------------------------------------------

with prediction_tab:
    st.header("Predict a Dataset Category")

    summary = st.text_area(
        "Enter the clinical trial brief summary",
        height=220,
        placeholder="Paste the clinical trial summary here..."
    )

    if st.button("Predict Category", type="primary"):
        if len(summary.strip()) < 20:
            st.warning("Please enter a longer clinical trial summary.")
        else:
            cleaned_summary = clean_text(summary)

            if cleaned_summary.strip() == "":
                st.warning("No useful words were found in the summary.")
            else:
                text_features = vectorizer.transform([cleaned_summary])
                prediction = model.predict(text_features)[0]

                st.success("Predicted Category: " + str(prediction).title())
                st.write("Model used:", best_model_name)

                # Show the three highest model scores
                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(text_features)[0]

                    score_table = pd.DataFrame({
                        "Category": model.classes_,
                        "Model Score": scores
                    })

                    score_table = score_table.sort_values(
                        "Model Score",
                        ascending=False
                    ).head(3)

                    st.subheader("Top 3 Category Scores")
                    st.bar_chart(
                        score_table.set_index("Category")
                    )

                    score_table["Model Score"] = (
                        score_table["Model Score"] * 100
                    ).round(2).astype(str) + "%"

                    st.dataframe(
                        score_table,
                        width="stretch",
                        hide_index=True
                    )

                st.caption(
                    "Model scores are not medically validated probabilities "
                    "and should not be used for diagnosis."
                )


# --------------------------------------------------
# TAB 2: DATASET INSIGHTS
# --------------------------------------------------

with insights_tab:
    st.header("Dataset Insights")

    if os.path.exists(CLEAN_FILE):
        try:
            # Load only the category column to keep this step simple.
            data = pd.read_csv(
                CLEAN_FILE,
                usecols=["source_condition_query"]
            )

            category_counts = (
                data["source_condition_query"]
                .value_counts()
                .reset_index()
            )

            category_counts.columns = ["Disease Category", "Trial Count"]

            total_trials = len(data)
            total_categories = len(category_counts)

            col1, col2 = st.columns(2)
            col1.metric("Total Clinical Trials", f"{total_trials:,}")
            col2.metric("Number of Categories", total_categories)

            st.subheader("Number of Trials in Each Category")

            st.bar_chart(
                category_counts.set_index("Disease Category")
            )

            st.subheader("Category Details")
            category_counts["Percentage"] = (
                category_counts["Trial Count"] / total_trials * 100
            ).round(2)

            st.dataframe(
                category_counts,
                width="stretch",
                hide_index=True
            )

        except Exception as error:
            st.error("Could not read the cleaned dataset.")
            st.write(error)
    else:
        st.warning(
            "Cleaned dataset not found. Please run "
            "02_data_preprocessing.py first."
        )


# --------------------------------------------------
# TAB 3: MODEL PERFORMANCE
# --------------------------------------------------

with performance_tab:
    st.header("Model Performance")

    if os.path.exists(METRICS_FILE):
        try:
            metrics = pd.read_csv(METRICS_FILE)

            st.subheader("Comparison of Machine Learning Models")

            display_metrics = metrics.copy()

            # Convert metric values to percentages for easier reading.
            metric_columns = [
                "accuracy",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
                "f1_macro"
            ]

            for column in metric_columns:
                display_metrics[column] = (
                    display_metrics[column] * 100
                ).round(2).astype(str) + "%"

            display_metrics = display_metrics.rename(
                columns={
                    "model": "Model",
                    "accuracy": "Accuracy",
                    "precision_weighted": "Weighted Precision",
                    "recall_weighted": "Weighted Recall",
                    "f1_weighted": "Weighted F1",
                    "f1_macro": "Macro F1"
                }
            )

            st.dataframe(
                display_metrics,
                width="stretch",
                hide_index=True
            )

            # Show the results of the selected model.
            best_result = metrics[
                metrics["model"] == best_model_name
            ]

            if not best_result.empty:
                result = best_result.iloc[0]

                st.subheader("Best Model: " + best_model_name)

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{result['accuracy']:.2%}")
                col2.metric("Weighted F1", f"{result['f1_weighted']:.2%}")
                col3.metric("Macro F1", f"{result['f1_macro']:.2%}")

            st.subheader("Accuracy Comparison")

            accuracy_data = metrics[["model", "accuracy"]].copy()
            accuracy_data["accuracy"] = accuracy_data["accuracy"] * 100

            st.bar_chart(
                accuracy_data.set_index("model")
            )

        except Exception as error:
            st.error("Could not read the model evaluation results.")
            st.write(error)
    else:
        st.warning(
            "Model evaluation file not found. Please run "
            "03_model_training.py first."
        )

    # Show the detailed classification report
    if os.path.exists(REPORT_FILE):
        st.subheader("Classification Report")

        with open(REPORT_FILE, "r", encoding="utf-8") as file:
            report = file.read()

        st.code(report)

    # Show the confusion matrix
    if os.path.exists(CONFUSION_FILE):
        st.subheader("Confusion Matrix")
        st.image(
            CONFUSION_FILE,
            caption="Confusion matrix for the selected model",
            width="stretch"
        )

