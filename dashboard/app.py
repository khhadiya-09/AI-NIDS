import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-NIDS | Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "ai_nids_multiclass_rf_H.joblib"
FEATURE_FILE = MODEL_DIR / "ai_nids_feature_columns.json"
METADATA_FILE = MODEL_DIR / "ai_nids_model_metadata.json"
PREPROCESSING_FILE = (
    MODEL_DIR / "ai_nids_multiclass_rf_H_preprocessing.json"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 12px;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .threat-banner {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #fecaca;
        background-color: #fef2f2;
    }

    .safe-banner {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #bbf7d0;
        background-color: #f0fdf4;
    }

    .evaluation-note {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #bfdbfe;
        background-color: #eff6ff;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding-top: 2rem;
        padding-bottom: 1rem;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL PACKAGE
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)

    with open(FEATURE_FILE, "r") as f:
        feature_data = json.load(f)

    if isinstance(feature_data, list):
        feature_columns = feature_data
    else:
        feature_columns = feature_data["feature_columns"]

    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    with open(PREPROCESSING_FILE, "r") as f:
        preprocessing = json.load(f)

    training_medians = pd.Series(
        preprocessing["training_medians"],
        dtype=float
    )

    return (
        model,
        feature_columns,
        metadata,
        training_medians
    )


# ============================================================
# LABEL MAPPING
# ============================================================

def map_label(label):

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
        "DoS Slowhttptest"
    ]:
        return "DoS"

    if label in [
        "FTP-Patator",
        "SSH-Patator"
    ]:
        return "Brute Force"

    if label in [
        "Web Attack � Brute Force",
        "Web Attack � XSS",
        "Web Attack � Sql Injection",
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Web Attack - Sql Injection"
    ]:
        return "Web Attack"

    if label == "Bot":
        return "Bot"

    return None


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_data(
    data,
    model,
    feature_columns,
    training_medians,
    bot_threshold
):

    data = data.copy()

    # Normalize CIC-IDS2017 headers
    data.columns = data.columns.str.strip()

    # Check required features
    missing_features = [
        feature
        for feature in feature_columns
        if feature not in data.columns
    ]

    if missing_features:
        return None, missing_features

    # Select exact training features
    X = data[feature_columns].copy()

    # Replace infinity
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Training-only median imputation
    X = X.fillna(training_medians)

    # Safety checks
    if X.isna().sum().sum() > 0:
        raise ValueError(
            "NaN values remain after preprocessing."
        )

    if np.isinf(X.to_numpy()).sum() > 0:
        raise ValueError(
            "Infinity values remain after preprocessing."
        )

    # Model probabilities
    probabilities = model.predict_proba(X)

    class_names = list(model.classes_)

    # Normal multiclass prediction
    prediction_indices = np.argmax(
        probabilities,
        axis=1
    )

    predictions = np.array(
        [
            class_names[index]
            for index in prediction_indices
        ]
    )

    # Bot threshold
    bot_index = class_names.index("Bot")

    bot_probability = probabilities[:, bot_index]

    bot_override = (
        bot_probability >= bot_threshold
    )

    predictions[bot_override] = "Bot"

    # Confidence
    confidence = probabilities.max(axis=1)

    # Results
    results = data.copy()

    results["Prediction"] = predictions
    results["Confidence"] = confidence
    results["Bot_Probability"] = bot_probability

    return results, None


# ============================================================
# LOAD MODEL
# ============================================================

try:

    (
        model,
        feature_columns,
        metadata,
        training_medians
    ) = load_model()

except Exception as e:

    st.error(
        f"Failed to load AI-NIDS model package: {e}"
    )

    st.stop()


BOT_THRESHOLD = float(
    metadata.get("bot_threshold", 0.50)
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ AI-NIDS")

    st.caption(
        "AI-Based Network Intrusion Detection System"
    )

    st.divider()

    st.markdown("### Model")

    st.write("**Algorithm:** Random Forest")
    st.write("**Dataset:** CIC-IDS2017")
    st.write("**Features:** 70")
    st.write("**Classes:** 7")
    st.write(
        f"**Bot threshold:** {BOT_THRESHOLD:.2f}"
    )

    st.divider()

    st.markdown("### Detection Classes")

    for class_name in model.classes_:

        if class_name == "BENIGN":
            st.write("🟢 BENIGN")

        elif class_name == "Bot":
            st.write("🔴 Bot")

        elif class_name == "DDoS":
            st.write("🔴 DDoS")

        elif class_name == "DoS":
            st.write("🟠 DoS")

        elif class_name == "PortScan":
            st.write("🟠 PortScan")

        elif class_name == "Brute Force":
            st.write("🟠 Brute Force")

        elif class_name == "Web Attack":
            st.write("🟠 Web Attack")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ AI-Based Network Intrusion Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine-learning-based network traffic detection and '
    'multi-class attack classification using CIC-IDS2017.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("🔍 Model Information"):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "Features",
            "70"
        )

    with col3:
        st.metric(
            "Classes",
            "7"
        )

    with col4:
        st.metric(
            "Bot Threshold",
            f"{BOT_THRESHOLD:.2f}"
        )

    st.write(
        "**Classes:**",
        ", ".join(model.classes_)
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📂 Upload Network Traffic'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a CIC-IDS2017-style CSV file",
    type=["csv"],
    help=(
        "Upload a network-flow CSV containing the "
        "70 CIC-IDS2017 features."
    )
)


# ============================================================
# PROCESS FILE
# ============================================================

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

    except Exception as e:

        st.error(
            f"Could not read the CSV file: {e}"
        )

        st.stop()

    st.success(
        f"Successfully loaded {len(data):,} "
        f"network-flow records."
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing network traffic with AI-NIDS..."
    ):

        results, missing_features = predict_data(
            data,
            model,
            feature_columns,
            training_medians,
            BOT_THRESHOLD
        )

    # --------------------------------------------------------
    # Missing features
    # --------------------------------------------------------

    if missing_features:

        st.error(
            f"{len(missing_features)} required features are missing."
        )

        with st.expander("View missing features"):

            for feature in missing_features:
                st.write(f"- {feature}")

        st.stop()


    # ========================================================
    # SUMMARY
    # ========================================================

    total_flows = len(results)

    benign_flows = int(
        (
            results["Prediction"] == "BENIGN"
        ).sum()
    )

    attack_flows = (
        total_flows - benign_flows
    )

    attack_percentage = (
        attack_flows / total_flows * 100
        if total_flows > 0
        else 0
    )


    # ========================================================
    # SECURITY STATUS
    # ========================================================

    if attack_flows > 0:

        st.markdown(
            f"""
            <div class="threat-banner">
            🚨 <strong>Threats Detected</strong><br>
            AI-NIDS identified
            <strong>{attack_flows:,}</strong>
            suspicious network flows
            ({attack_percentage:.2f}% of analyzed traffic).
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="safe-banner">
            🟢 <strong>No Threats Detected</strong><br>
            All analyzed network flows were classified as benign.
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Detection Summary'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Flows",
            f"{total_flows:,}"
        )

    with col2:
        st.metric(
            "Benign",
            f"{benign_flows:,}"
        )

    with col3:
        st.metric(
            "Attacks",
            f"{attack_flows:,}"
        )

    with col4:
        st.metric(
            "Attack Rate",
            f"{attack_percentage:.2f}%"
        )


    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Prediction Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    prediction_counts = (
        results["Prediction"]
        .value_counts()
        .reindex(
            model.classes_,
            fill_value=0
        )
    )

    st.bar_chart(
        prediction_counts,
        width="stretch"
    )


    # ========================================================
    # ATTACK BREAKDOWN
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🚨 Attack Breakdown'
        '</div>',
        unsafe_allow_html=True
    )

    attack_counts = (
        results[
            results["Prediction"] != "BENIGN"
        ]["Prediction"]
        .value_counts()
    )

    if len(attack_counts) > 0:

        attack_table = (
            attack_counts
            .rename("Detected Flows")
            .to_frame()
        )

        attack_table["Percentage"] = (
            attack_table["Detected Flows"]
            / attack_flows
            * 100
        ).round(2)

        st.dataframe(
            attack_table,
            width="stretch"
        )

    else:

        st.info(
            "No attacks were detected."
        )


    # ========================================================
    # EVALUATION MODE
    # ========================================================

    if "Label" in results.columns:

        st.markdown(
            '<div class="section-title">'
            '🔬 Model Evaluation'
            '</div>',
            unsafe_allow_html=True
        )

        results["True_Label"] = (
            results["Label"]
            .apply(map_label)
        )

        evaluable_mask = (
            results["True_Label"].notna()
        )

        evaluable_results = results[
            evaluable_mask
        ].copy()

        excluded_rows = (
            len(results) - len(evaluable_results)
        )

        st.markdown(
            """
            <div class="evaluation-note">
            <strong>Evaluation Mode</strong><br>
            Ground-truth labels were detected in the uploaded
            dataset. Metrics below compare the AI-NIDS predictions
            with the mapped CIC-IDS2017 labels.
            </div>
            """,
            unsafe_allow_html=True
        )

        if excluded_rows > 0:

            st.warning(
                f"{excluded_rows:,} rows were excluded from "
                "evaluation because their labels are outside "
                "the project's primary 7-class mapping."
            )

        y_true = (
            evaluable_results["True_Label"]
        )

        y_pred = (
            evaluable_results["Prediction"]
        )

        evaluation_labels = list(
            model.classes_
        )


        # ====================================================
        # OVERALL EVALUATION METRICS
        # ====================================================

        eval_accuracy = accuracy_score(
            y_true,
            y_pred
        )

        eval_precision = precision_score(
            y_true,
            y_pred,
            labels=evaluation_labels,
            average="macro",
            zero_division=0
        )

        eval_recall = recall_score(
            y_true,
            y_pred,
            labels=evaluation_labels,
            average="macro",
            zero_division=0
        )

        eval_macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=evaluation_labels,
            average="macro",
            zero_division=0
        )

        eval_weighted_f1 = f1_score(
            y_true,
            y_pred,
            labels=evaluation_labels,
            average="weighted",
            zero_division=0
        )


        # ====================================================
        # METRIC CARDS
        # ====================================================

        metric1, metric2, metric3, metric4, metric5 = (
            st.columns(5)
        )

        with metric1:
            st.metric(
                "Accuracy",
                f"{eval_accuracy * 100:.2f}%"
            )

        with metric2:
            st.metric(
                "Macro Precision",
                f"{eval_precision * 100:.2f}%"
            )

        with metric3:
            st.metric(
                "Macro Recall",
                f"{eval_recall * 100:.2f}%"
            )

        with metric4:
            st.metric(
                "Macro F1",
                f"{eval_macro_f1 * 100:.2f}%"
            )

        with metric5:
            st.metric(
                "Weighted F1",
                f"{eval_weighted_f1 * 100:.2f}%"
            )


        # ====================================================
        # DISTRIBUTION COMPARISON
        # ====================================================

        st.markdown(
            "#### Ground-Truth vs Predicted Distribution"
        )

        eval_col1, eval_col2 = st.columns(2)

        with eval_col1:

            st.markdown(
                "##### Ground-Truth Distribution"
            )

            true_counts = (
                y_true
                .value_counts()
                .reindex(
                    evaluation_labels,
                    fill_value=0
                )
            )

            st.bar_chart(
                true_counts,
                width="stretch"
            )

        with eval_col2:

            st.markdown(
                "##### Predicted Distribution"
            )

            pred_counts = (
                y_pred
                .value_counts()
                .reindex(
                    evaluation_labels,
                    fill_value=0
                )
            )

            st.bar_chart(
                pred_counts,
                width="stretch"
            )


        # ====================================================
        # PER-CLASS PERFORMANCE
        # ====================================================

        st.markdown(
            "#### 📊 Per-Class Performance"
        )

        class_precision = precision_score(
            y_true,
            y_pred,
            labels=evaluation_labels,
            average=None,
            zero_division=0
        )

        class_recall = recall_score(
            y_true,
            y_pred,
            labels=evaluation_labels,
            average=None,
            zero_division=0
        )

        class_f1 = f1_score(
            y_true,
            y_pred,
            labels=evaluation_labels,
            average=None,
            zero_division=0
        )

        support = (
            y_true
            .value_counts()
            .reindex(
                evaluation_labels,
                fill_value=0
            )
        )

        per_class_df = pd.DataFrame(
            {
                "Class": evaluation_labels,
                "Precision": (
                    class_precision * 100
                ).round(2),
                "Recall": (
                    class_recall * 100
                ).round(2),
                "F1 Score": (
                    class_f1 * 100
                ).round(2),
                "Support": support.values
            }
        )

        st.dataframe(
            per_class_df,
            width="stretch",
            hide_index=True
        )

        st.caption(
            "Precision, Recall and F1 are shown as percentages. "
            "Support represents the number of actual samples "
            "belonging to each class."
        )


        # ====================================================
        # CONFUSION MATRIX HEATMAP
        # ====================================================

        st.markdown(
            "#### 🔥 Confusion Matrix"
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=evaluation_labels
        )

        # ----------------------------------------------------
        # Compact heatmap
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=evaluation_labels,
            yticklabels=evaluation_labels,
            linewidths=0.5,
            linecolor="white",
            cbar=True,
            annot_kws={
                "fontsize": 9
            },
            ax=ax
        )

        ax.set_xlabel(
            "Predicted Class",
            fontsize=10
        )

        ax.set_ylabel(
            "Actual Class",
            fontsize=10
        )

        ax.set_title(
            "AI-NIDS Confusion Matrix",
            fontsize=12,
            fontweight="bold",
            pad=10
        )

        plt.xticks(
            rotation=35,
            ha="right",
            fontsize=9
        )

        plt.yticks(
            rotation=0,
            fontsize=9
        )

        plt.tight_layout()

        # IMPORTANT:
        # "content" keeps the heatmap compact instead of
        # stretching it across the entire dashboard.
        st.pyplot(
            fig,
            width="content"
        )

        plt.close(fig)

        st.caption(
            "Rows represent actual classes and columns represent "
            "predicted classes. Darker cells indicate larger "
            "numbers of predictions. Diagonal cells represent "
            "correct classifications."
        )


        # ====================================================
        # NUMERICAL CONFUSION MATRIX
        # ====================================================

        with st.expander(
            "📋 View Numerical Confusion Matrix"
        ):

            cm_df = pd.DataFrame(
                cm,
                index=evaluation_labels,
                columns=evaluation_labels
            )

            st.dataframe(
                cm_df,
                width="stretch"
            )


    # ========================================================
    # FILTER RESULTS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔎 Investigate Predictions'
        '</div>',
        unsafe_allow_html=True
    )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:

        selected_classes = st.multiselect(
            "Prediction classes",
            options=list(model.classes_),
            default=list(model.classes_)
        )

    with filter_col2:

        min_confidence = st.slider(
            "Minimum confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )


    filtered_results = results[
        results["Prediction"].isin(
            selected_classes
        )
        &
        (
            results["Confidence"]
            >= min_confidence
        )
    ]


    st.caption(
        f"Showing {len(filtered_results):,} "
        f"of {len(results):,} predictions."
    )


    # ========================================================
    # RESULTS TABLE
    # ========================================================

    display_columns = [
        "Prediction",
        "Confidence",
        "Bot_Probability"
    ]

    preferred_original_columns = [
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Bytes/s",
        "Flow Packets/s"
    ]

    for column in preferred_original_columns:

        if column in filtered_results.columns:

            display_columns.insert(
                0,
                column
            )


    available_columns = [
        column
        for column in display_columns
        if column in filtered_results.columns
    ]


    st.dataframe(
        filtered_results[
            available_columns
        ],
        width="stretch",
        height=450
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '💾 Export Results'
        '</div>',
        unsafe_allow_html=True
    )

    csv_output = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction Results CSV",
        data=csv_output,
        file_name="ai_nids_prediction_results.csv",
        mime="text/csv",
        width="stretch"
    )


else:

    st.info(
        "Upload a CIC-IDS2017 CSV above to begin detection."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    'AI-NIDS • Random Forest - Experiment H • CIC-IDS2017'
    '</div>',
    unsafe_allow_html=True
)