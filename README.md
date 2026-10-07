# AI-Based Network Intrusion Detection System (AI-NIDS)

A machine-learning-based Network Intrusion Detection System (NIDS) developed using the **CIC-IDS2017** dataset, with a focus on **generalization, data representation, leakage-controlled evaluation, and minority-class detection**.

The project evaluates how the representation of attack traffic in training data affects the ability of machine-learning models to detect known and previously unseen attack scenarios.

---

## 🔬 Research Question

> **How does the representation of attack traffic in training data affect the generalization and minority-class detection performance of machine-learning-based network intrusion detection systems?**

Rather than relying only on conventional random train-test splits, this project includes **scenario-aware and leakage-controlled experiments** to investigate whether high benchmark performance translates into reliable detection when attack scenarios are poorly represented or completely unseen during training.

---

## 🎯 Objectives

- Develop a machine-learning-based network intrusion detection system using CIC-IDS2017.
- Perform systematic dataset auditing and preprocessing.
- Prevent duplicate-row leakage between training and testing.
- Compare binary and multiclass intrusion detection.
- Evaluate generalization to an **unseen DDoS scenario**.
- Study how small amounts of representative attack traffic affect detection performance.
- Investigate minority-class detection, particularly **Bot traffic**.
- Compare Random Forest and XGBoost models.
- Build a Streamlit-based dashboard for practical inference and evaluation.

---

## 📊 Dataset

This project uses the **CIC-IDS2017** dataset developed by the Canadian Institute for Cybersecurity.

The dataset contains benign network traffic and multiple attack categories, including:

- DDoS
- DoS
- PortScan
- Brute Force
- Web Attacks
- Bot
- Infiltration
- Heartbleed

For the primary multiclass experiments, attack labels were consolidated into seven classes:

| Class |
|---|
| BENIGN |
| Bot |
| Brute Force |
| DDoS |
| DoS |
| PortScan |
| Web Attack |

### Dataset Audit

The original dataset contained approximately:

- **2.83 million rows**
- **79 columns**
- **78 network-flow features**
- **15 original labels**
- **308,381 exact duplicate rows**

Duplicate and conflicting feature-label patterns were investigated before model development.

The raw and processed datasets are **not included in this repository** because of their size. Users should obtain CIC-IDS2017 separately and place the required files in the appropriate `data/` directories.

---

## 🧪 Methodology

The project follows a research-oriented evaluation pipeline:

```text
CIC-IDS2017
     │
     ▼
Dataset Audit
     │
     ├── Duplicate analysis
     ├── Label analysis
     ├── Missing/non-finite value analysis
     └── Feature consistency checks
     │
     ▼
Preprocessing
     │
     ├── Feature selection
     ├── Label consolidation
     └── Training-only median imputation
     │
     ▼
Leakage-Controlled Splitting
     │
     ├── Exact-row hashing
     └── Group-based train/test separation
     │
     ▼
Experiments
     │
     ├── Binary detection
     ├── Unseen DDoS generalization
     ├── DDoS exposure ablation
     ├── Multiclass detection
     └── Bot minority-class analysis
     │
     ▼
Model Comparison
     │
     ├── Random Forest
     └── XGBoost
     │
     ▼
Final Deployment Model
     │
     └── Random Forest
     ```

## 📈 Key Research Findings
### 1. Random Splits Can Produce Extremely Optimistic Results
Under a leakage-controlled random group split, the Random Forest binary classifier achieved:
Metric	Score
Accuracy	99.80%
Precision	99.73%
Recall	99.26%
F1 Score	99.49%
ROC-AUC	99.97%
PR-AUC	99.82%


However, this performance changed substantially when an entire DDoS scenario was held out from training.
### 2. Unseen DDoS Generalization
When the complete Friday Afternoon DDoS scenario was excluded from training, the Random Forest achieved:
Metric	Score
Accuracy	79.27%
Precision	99.91%
Recall	63.50%
F1 Score	77.65%
ROC-AUC	93.79%
PR-AUC	95.01%


The model therefore maintained very high precision but missed a substantial portion of the unseen DDoS traffic.
36.50% of DDoS traffic was missed.
This demonstrates the difference between high benchmark performance and genuine scenario generalization.

### 3. Small Amounts of Attack Exposure Can Dramatically Improve Generalization
A controlled DDoS exposure experiment investigated whether introducing a small representative subset of DDoS traffic into training could improve detection.
DDoS Exposure	Recall
0%	63.50%
1%	99.79%
2%	99.82%
5%	99.88%


Introducing only 1% representative DDoS exposure improved recall by approximately 36.29 percentage points.
This experiment highlights the importance of representative attack diversity in the training data.

## 🤖 Multiclass Detection
The final multiclass experiment used seven classes and a leakage-controlled validation strategy.
The final Random Forest achieved:
Metric	Score
Accuracy	98.96%
Macro Precision	95.72%
Macro Recall	97.05%
Macro F1	96.01%
Weighted F1	98.93%


Per-Class Performance
Class	Precision	Recall	F1
BENIGN	98.83%	99.09%	99.36%
DoS	99.69%	89.11%	93.91%
DDoS	99.92%	99.93%	99.96%
PortScan	99.48%	99.10%	99.29%
Brute Force	100.00%	99.38%	99.68%
Web Attack	99.29%	98.35%	98.78%
Bot	72.60%	94.40%	82.08%



## 🕵️ Minority-Class Analysis: Bot Traffic
Bot traffic presented a particularly interesting minority-class detection problem.
The final model achieved:
- 94.40% Bot recall
- 72.60% Bot precision
- 82.08% Bot F1
The model detected:
371 / 393 Bot samples

The experiments showed that increasing Bot representation in training substantially improved Bot recall.
However, increasing sensitivity also introduced false positives where benign traffic was classified as Bot.
This demonstrates an important practical trade-off between minority-class recall and false-positive control.

## ⚖️ Model Comparison
Two tree-based machine-learning approaches were evaluated:
- Random Forest
- XGBoost
XGBoost achieved stronger aggregate performance in the final comparison, but Random Forest provided substantially better Bot recall, which was an important objective of this project.
Model	Accuracy	Macro F1	Bot Recall	Bot F1
Random Forest	98.96%	96.01%	94.40%	82.08%
XGBoost	99.87%	96.99%	78.63%	82.07%


Final Model Selection
Random Forest was retained as the deployment model because it provided substantially higher Bot recall while maintaining strong overall multiclass performance.

## 🔥 Final Confusion Matrix
The final evaluation revealed strong separation between most classes.
The largest attack-to-benign error was:
5,110 DoS samples classified as BENIGN

This highlights an important limitation of aggregate accuracy: a model can achieve excellent overall performance while still producing a meaningful number of misses for a particular attack category.

## 🖥️ Streamlit Dashboard
The project includes an interactive Streamlit dashboard for model inference and evaluation.
The dashboard supports:
- CSV upload
- Network traffic prediction
- Attack/benign distribution
- Per-class prediction breakdown
- Ground-truth evaluation when labels are available
- Accuracy, precision, recall and F1 metrics
- Per-class performance analysis
- Confusion matrix visualization
- Prediction filtering
- CSV result export
Run Locally
Create and activate the virtual environment:
python -m venv .venv

Windows PowerShell:
.venv\Scripts\Activate.ps1

Install dependencies:
pip install -r requirements.txt

Run the dashboard:
streamlit run dashboard/app.py

The application will open in your browser.
## 📁 Repository Structure
AI-NIDS/
│
├── dashboard/
│   └── app.py
│
├── models/
│   ├── ai_nids_multiclass_rf_H.joblib
│   ├── ai_nids_feature_columns.json
│   ├── ai_nids_model_metadata.json
│   └── ai_nids_multiclass_rf_H_preprocessing.json
│
├── results/
│   └── figures/
│       ├── 01_generalization_shift.png
│       ├── 02_model_comparison.png
│       ├── 03_ddos_exposure_ablation.png
│       ├── 04_multiclass_per_class.png
│       ├── 05_bot_exposure.png
│       └── 06_final_confusion_matrix.png
│
├── src/
│   ├── predict.py
│   └── verify_model.py
│
├── .gitignore
├── requirements.txt
└── README.md

## ⚠️ Limitations
- CIC-IDS2017 is a benchmark dataset and may not fully represent modern production network environments.
- Dataset-specific patterns can make machine-learning models appear more generalizable than they actually are.
- The unseen-scenario experiment focused specifically on DDoS traffic.
- Minority-class results can be sensitive to the amount and diversity of available training examples.
- The current deployment model is trained on CIC-IDS2017 features and therefore requires compatible network-flow feature extraction for real-world deployment.
## 🔮 Future Work
Potential extensions include:
- Evaluation on additional modern intrusion-detection datasets.
- Cross-dataset generalization experiments.
- Real-time network-flow ingestion.
- Online/continual learning.
- Explainable AI for intrusion predictions.
- Adversarial robustness testing.
- Drift detection for changing network environments.
- Evaluation on IoT-specific network traffic.
- Integration with SIEM/SOC workflows.
- Deployment as a real-time detection service.
🛠️ Technologies
- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- CIC-IDS2017
📚 Dataset Reference
Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A.
Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.
International Conference on Information Systems Security and Privacy (ICISSP), 2018.
👩‍💻 Author
Hadiya Khan
Computer Science Engineering
Cybersecurity | Network Security | AI Security | IoT Security
⭐ Project Focus
This project emphasizes that reliable intrusion detection evaluation should not rely solely on aggregate accuracy.
Instead, robust NIDS research should consider:
data representation → leakage control → scenario generalization → minority-class detection → class-specific metrics

### One VERY important thing

When you paste this version, **do not add any extra ` ``` ` at the beginning or end**.

The README should literally start with:

```text
# AI-Based Network Intrusion Detection System (AI-NIDS)

and end with:
**data representation → leakage control → scenario generalization → minority-class detection → class-specific metrics**
