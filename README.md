# 🧬 Enzyme Classification Project

## 📌 Overview

This project focuses on **classifying proteins into enzymes vs. non-enzymes** using only their amino acid sequences.

We start from **raw FASTA files**, clean and preprocess the data, extract biologically meaningful features, and train machine learning models (Logistic Regression, Random Forest, and XGBoost). The goal is to evaluate baseline classifiers and establish a foundation for more advanced sequence-based function prediction.

---

## 📂 Project Structure

```
Week_2_Enzyme_Classification/
│── data/
│   ├── sprot_enz_seq.fasta.txt      # Enzyme sequences
│   ├── sprot_nonenz_seq.fasta.txt   # Non-enzyme sequences
│
│── notebooks/
│   ├── enzyme_classification.ipynb  # Main analysis notebook
│
│── README.md                        # Project documentation
```

---

## 🔬 Workflow

The project is structured into several key steps:

1. **Dataset Preparation**

   * Parse enzyme and non-enzyme FASTA files using Biopython.
   * Build a combined dataset with labels (`1 = enzyme`, `0 = non-enzyme`).

2. **Data Cleaning**

   * Remove duplicate sequences.
   * Remove overlapping sequences present in both classes.
   * Drop sequences with invalid amino acids or length > 2500 residues.

3. **Feature Engineering**

   * Sequence length.
   * Amino acid composition (fraction of each residue).
   * Create feature matrix (`X`) and labels (`y`).

4. **Exploratory Data Analysis (EDA)**

   * Histograms of sequence lengths.
   * Distributions of amino acid fractions.

5. **Preprocessing**

   * Scale sequence length feature with `StandardScaler`.
   * Preserve amino acid composition fractions.

6. **Modeling & Evaluation**

   * Train/test split (80/20 stratified).
   * Train baseline models: Logistic Regression, Random Forest, XGBoost.
   * Evaluate with **Accuracy, Precision, Recall, F1, ROC-AUC**.
   * Plot confusion matrices and ROC curves.
   * Run cross-validation and hyperparameter tuning (GridSearchCV).

---

## 📊 Results (Baseline)

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
| ------------------- | -------- | --------- | ------ | ------ | ------- |
| Logistic Regression | \~0.85   | \~0.83    | \~0.86 | \~0.84 | \~0.90  |
| Random Forest       | \~0.90   | \~0.89    | \~0.91 | \~0.90 | \~0.95  |
| XGBoost             | \~0.91   | \~0.90    | \~0.92 | \~0.91 | \~0.96  |

*(Values are approximate, results depend on dataset splits & tuning.)*

---

## ⚙️ Installation & Requirements

Clone the repository:

```bash
git clone https://github.com/your-username/Week_2_Enzyme_Classification.git
cd Week_2_Enzyme_Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**Key Python packages**:

* `biopython`
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `xgboost` *(optional)*

---

## 🚀 Usage

Run the notebook step by step:

```bash
jupyter notebook notebooks/enzyme_classification.ipynb
```

Or execute the script version (if added later):

```bash
python enzyme_classifier.py
```

---

## 📌 Future Work

* Use **transformer-based embeddings** (e.g., ProtBERT, ESM) instead of handcrafted features.
* Explore **homology reduction** to avoid sequence redundancy.
* Expand dataset with more diverse enzymes.
* Apply deep learning methods (CNNs, RNNs, Transformers).

---

## 🧑‍💻 Author

**Rithwik Swarnkar**
Machine Learning Enthusiast | Bioinformatics Learner

