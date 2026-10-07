import json
import joblib
from pathlib import Path


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "ai_nids_multiclass_rf_H.joblib"
FEATURE_FILE = MODEL_DIR / "ai_nids_feature_columns.json"
METADATA_FILE = MODEL_DIR / "ai_nids_model_metadata.json"
PREPROCESSING_FILE = MODEL_DIR / "ai_nids_multiclass_rf_H_preprocessing.json"


print("=" * 70)
print("AI-NIDS MODEL PACKAGE VERIFICATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Check files
# ------------------------------------------------------------

files_to_check = {
    "Model": MODEL_FILE,
    "Feature columns": FEATURE_FILE,
    "Metadata": METADATA_FILE,
    "Preprocessing": PREPROCESSING_FILE
}

print("\nChecking files...\n")

for name, path in files_to_check.items():
    status = "FOUND" if path.exists() else "MISSING"
    print(f"{name:20} : {status}")

    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")


# ------------------------------------------------------------
# 2. Load model
# ------------------------------------------------------------

print("\nLoading Random Forest model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")
print("Model type:", type(model).__name__)


# ------------------------------------------------------------
# 3. Load feature columns
# ------------------------------------------------------------

print("\nLoading feature configuration...")

with open(FEATURE_FILE, "r") as f:
    feature_data = json.load(f)

# Handle the saved JSON structure
if isinstance(feature_data, list):
    feature_columns = feature_data
elif isinstance(feature_data, dict):
    if "feature_columns" in feature_data:
        feature_columns = feature_data["feature_columns"]
    else:
        feature_columns = list(feature_data.keys())
else:
    raise ValueError("Unexpected feature JSON format.")

print("Number of features:", len(feature_columns))
print("Model feature count:", model.n_features_in_)

assert len(feature_columns) == model.n_features_in_

print("Feature count matches: TRUE")


# ------------------------------------------------------------
# 4. Load metadata
# ------------------------------------------------------------

print("\nLoading metadata...")

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

print("Model name:", metadata.get("model_name"))
print("Experiment:", metadata.get("experiment"))
print("Bot threshold:", metadata.get("bot_threshold"))
print("Classes:", metadata.get("classes"))


# ------------------------------------------------------------
# 5. Load preprocessing
# ------------------------------------------------------------

print("\nLoading preprocessing configuration...")

with open(PREPROCESSING_FILE, "r") as f:
    preprocessing = json.load(f)

preprocess_info = preprocessing["preprocessing"]
training_medians = preprocessing["training_medians"]

print("Preprocessing method:",
      preprocess_info["imputation"])

print("Median source:",
      preprocess_info["median_source"])

print("Number of saved medians:",
      len(training_medians))

print("Preprocessing feature count:",
      preprocess_info["n_features"])


# ------------------------------------------------------------
# 6. Verify preprocessing
# ------------------------------------------------------------

assert len(training_medians) == 70
assert preprocess_info["median_source"] == "training_only"
assert preprocess_info["n_features"] == 70

print("Training-only median configuration: TRUE")
print("Preprocessing feature count: TRUE")


# ------------------------------------------------------------
# 7. Verify classes
# ------------------------------------------------------------

expected_classes = [
    "BENIGN",
    "Bot",
    "Brute Force",
    "DDoS",
    "DoS",
    "PortScan",
    "Web Attack"
]

print("\nModel classes:")
for cls in model.classes_:
    print(" -", cls)

assert list(model.classes_) == expected_classes

print("Class configuration matches: TRUE")


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("✅ AI-NIDS MODEL PACKAGE VERIFICATION SUCCESSFUL")
print("=" * 70)