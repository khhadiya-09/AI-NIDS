import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
TEST_DIR = BASE_DIR / "data" / "test"
RESULTS_DIR = BASE_DIR / "results"

MODEL_FILE = MODEL_DIR / "ai_nids_multiclass_rf_H.joblib"
FEATURE_FILE = MODEL_DIR / "ai_nids_feature_columns.json"
METADATA_FILE = MODEL_DIR / "ai_nids_model_metadata.json"
PREPROCESSING_FILE = (
    MODEL_DIR / "ai_nids_multiclass_rf_H_preprocessing.json"
)


# ============================================================
# CLASS MAPPING
# ============================================================

def map_label(label):
    """
    Convert original CIC-IDS2017 labels into the
    7-class AI-NIDS label scheme.
    """

    label = str(label).strip()

    if label == "BENIGN":
        return "BENIGN"

    if label == "DDoS":
        return "DDoS"

    if label == "PortScan":
        return "PortScan"

    if label in [
        "DoS Hulk",
        "DoS GoldenEye",
        "DoS slowloris",
        "DoS Slowhttptest",
    ]:
        return "DoS"

    if label in [
        "FTP-Patator",
        "SSH-Patator",
    ]:
        return "Brute Force"

    if label in [
        "Web Attack � Brute Force",
        "Web Attack � XSS",
        "Web Attack � Sql Injection",
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Web Attack - Sql Injection",
    ]:
        return "Web Attack"

    if label == "Bot":
        return "Bot"

    return None


# ============================================================
# LOAD MODEL PACKAGE
# ============================================================

print("=" * 70)
print("AI-NIDS LOCAL VALIDATION TEST")
print("=" * 70)

print("\nLoading model...")

model = joblib.load(MODEL_FILE)

print("Model loaded:", type(model).__name__)


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

with open(FEATURE_FILE, "r") as f:
    feature_data = json.load(f)

if isinstance(feature_data, list):
    feature_columns = feature_data
else:
    feature_columns = feature_data["feature_columns"]

print("Features:", len(feature_columns))


# ============================================================
# LOAD METADATA
# ============================================================

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

BOT_THRESHOLD = float(
    metadata.get("bot_threshold", 0.50)
)

print("Bot threshold:", BOT_THRESHOLD)


# ============================================================
# LOAD TRAINING MEDIANS
# ============================================================

with open(PREPROCESSING_FILE, "r") as f:
    preprocessing = json.load(f)

training_medians = pd.Series(
    preprocessing["training_medians"],
    dtype=float
)

print("Training medians:", len(training_medians))


# ============================================================
# FIND TEST CSV
# ============================================================

csv_files = list(TEST_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV file found in {TEST_DIR}"
    )

test_file = csv_files[0]

print("\nTest CSV:")
print(test_file.name)


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading CSV...")

data = pd.read_csv(test_file)

print("Original dataset shape:", data.shape)

# CIC-IDS2017 contains leading spaces in some headers.
data.columns = data.columns.str.strip()

print("Column names normalized.")


# ============================================================
# CHECK LABEL COLUMN
# ============================================================

if "Label" not in data.columns:
    raise ValueError(
        "The CSV does not contain the Label column."
    )

print("Ground-truth Label column found.")


# ============================================================
# TAKE SAMPLE
# ============================================================

SAMPLE_SIZE = 1000

if len(data) > SAMPLE_SIZE:
    sample = data.sample(
        n=SAMPLE_SIZE,
        random_state=42
    ).reset_index(drop=True)
else:
    sample = data.copy()

print("Rows selected:", len(sample))


# ============================================================
# CREATE GROUND-TRUTH LABELS
# ============================================================

sample["True_Label"] = sample["Label"].apply(map_label)

unknown_labels = sample[
    sample["True_Label"].isna()
]["Label"].unique()

if len(unknown_labels) > 0:

    print("\nUnknown labels found:")

    for label in unknown_labels:
        print(" -", repr(label))

    raise ValueError(
        "Some labels could not be mapped to the AI-NIDS classes."
    )

print("Ground-truth labels mapped successfully.")


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

missing_features = [
    feature
    for feature in feature_columns
    if feature not in sample.columns
]

if missing_features:

    print("\nMissing features:")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        f"{len(missing_features)} required features are missing."
    )

print("All 70 required features found.")


# ============================================================
# SELECT FEATURES IN EXACT TRAINING ORDER
# ============================================================

X = sample[feature_columns].copy()


# ============================================================
# REPLACE INFINITY WITH NaN
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

nan_before = int(
    X.isna().sum().sum()
)

print(
    "\nNaN/non-finite values before imputation:",
    nan_before
)


# ============================================================
# APPLY TRAINING-ONLY MEDIANS
# ============================================================

X = X.fillna(training_medians)


# ============================================================
# VERIFY CLEANING
# ============================================================

nan_after = int(
    X.isna().sum().sum()
)

inf_after = int(
    np.isinf(X.to_numpy()).sum()
)

print("NaN values after imputation:", nan_after)
print("Infinity values after imputation:", inf_after)

assert nan_after == 0
assert inf_after == 0


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

print("\nRunning Random Forest predictions...")

probabilities = model.predict_proba(X)


# ============================================================
# NORMAL MULTICLASS PREDICTION
# ============================================================

class_names = list(model.classes_)

normal_prediction_indices = np.argmax(
    probabilities,
    axis=1
)

normal_predictions = np.array(
    [
        class_names[index]
        for index in normal_prediction_indices
    ]
)


# ============================================================
# APPLY LOCKED BOT THRESHOLD
# ============================================================

bot_index = class_names.index("Bot")

bot_probabilities = probabilities[:, bot_index]

predictions = normal_predictions.copy()

bot_override_mask = (
    bot_probabilities >= BOT_THRESHOLD
)

predictions[bot_override_mask] = "Bot"


# ============================================================
# CONFIDENCE
# ============================================================

confidence = probabilities.max(axis=1)


# ============================================================
# STORE RESULTS
# ============================================================

results = sample.copy()

results["Prediction"] = predictions
results["Confidence"] = confidence
results["Bot_Probability"] = bot_probabilities


# ============================================================
# SAVE PREDICTION RESULTS
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_file = RESULTS_DIR / "prediction_results.csv"

results.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("PREDICTION RESULTS SAVED")
print("=" * 70)

print("Saved to:")
print(output_file)

print("Rows saved:", len(results))


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

print(
    pd.Series(predictions)
    .value_counts()
)


# ============================================================
# BOT THRESHOLD INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BOT THRESHOLD INFORMATION")
print("=" * 70)

print("Locked Bot threshold:", BOT_THRESHOLD)
print("Bot overrides:", int(bot_override_mask.sum()))


# ============================================================
# TRUE LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRUE LABEL DISTRIBUTION")
print("=" * 70)

print(
    sample["True_Label"]
    .value_counts()
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    sample["True_Label"],
    predictions
)

print("\n" + "=" * 70)
print("LOCAL VALIDATION RESULTS")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        sample["True_Label"],
        predictions,
        labels=model.classes_,
        zero_division=0,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    sample["True_Label"],
    predictions,
    labels=model.classes_
)

cm_df = pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm_df)


# ============================================================
# DISPLAY FIRST 20 PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 20 PREDICTIONS")
print("=" * 70)

print(
    results[
        [
            "True_Label",
            "Prediction",
            "Confidence",
            "Bot_Probability",
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("✅ LOCAL AI-NIDS VALIDATION COMPLETED")
print("=" * 70)