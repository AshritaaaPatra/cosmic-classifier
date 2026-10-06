# 🌌 Cosmic Classifier

A small machine learning project that classifies astronomical objects as **stars**, **galaxies**, or **quasars (QSO)** using telescope measurements from the Sloan Digital Sky Survey (SDSS).

## Overview

Telescopes capture huge numbers of objects, and sorting them by hand is not practical. This project trains a **Random Forest** model to recognize each object's type from its brightness in five color filters plus its redshift.

## Dataset

- **Source:** SDSS-based star classification dataset from Kaggle
- **Size:** 100,000 objects, 18 columns
- **Classes:**

| Class | Count |
|---|---|
| Galaxy | 59,445 |
| Star | 21,594 |
| Quasar (QSO) | 18,961 |

### Features used

| Feature | Meaning |
|---|---|
| `u` | Brightness through the ultraviolet filter |
| `g` | Brightness through the green filter |
| `r` | Brightness through the red filter |
| `i` | Brightness through the near-infrared filter |
| `z` | Brightness through the infrared filter |
| `redshift` | How much the object's light is stretched toward red, a measure of distance and speed of recession |

> **Note:** `z` is a telescope filter, not redshift. They are separate features.
> Values are *magnitudes*, so a **smaller number means a brighter object**.

## Method

1. Load the CSV with pandas
2. Select the features (`u, g, r, i, z, redshift`)
3. Split the data: 80% training, 20% testing (stratified, so class proportions stay the same)
4. Train a `RandomForestClassifier` with 100 trees
5. Evaluate with accuracy, a classification report, a confusion matrix, and feature importance

## Results

**Overall accuracy: 97.94%** on 20,000 test objects.

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Galaxy | 0.98 | 0.99 | 0.98 |
| Quasar | 0.97 | 0.93 | 0.95 |
| Star | 1.00 | 1.00 | 1.00 |

### Key findings

- **Stars are almost perfectly classified.** Only 2 of 4,319 were misclassified.
- **Quasars are the hardest class.** 276 quasars were labelled as galaxies, because distant quasars can look like faint galaxies.
- **Redshift is the most important feature** (about 0.65 importance). Stars have a redshift near 0, galaxies have low to moderate values, and quasars are often very high, so redshift alone separates the classes well.

## Metrics explained

- **Precision:** when the model predicts a class, how often is it correct?
- **Recall:** of all real objects in a class, how many did the model find?
- **F1-score:** a single number balancing precision and recall.
- **Support:** the number of real examples of that class in the test set.

Accuracy alone can be misleading because galaxies make up about 60% of the data, so per-class metrics matter.

## How to run

### Option 1: Google Colab (no installation)

1. Open [colab.research.google.com](https://colab.research.google.com) and create a new notebook
2. Upload the dataset CSV using the folder icon in the sidebar
3. Paste the code from `classifier.py` into a cell and run it

### Option 2: Locally

```bash
pip install pandas scikit-learn matplotlib
python classifier.py
```

Make sure the CSV file is in the same folder as `classifier.py`, and that the filename in `pd.read_csv(...)` matches it exactly.

## Project structure

```
Galaxy/
├── classifier.py
├── stars_classification.csv
└── README.md
```

## Possible improvements

- Add **color indices** (`u-g`, `g-r`, `r-i`, `i-z`) as extra features
- Try other models such as Gradient Boosting or a neural network
- Tune hyperparameters with cross-validation
- Handle class imbalance with class weights

## Tools used

Python, pandas, scikit-learn, matplotlib

## Acknowledgements

Data from the Sloan Digital Sky Survey (SDSS), obtained via Kaggle.
