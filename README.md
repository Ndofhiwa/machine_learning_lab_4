<div align="center">

# 🧹 Machine Learning Lab 4 — Data Preprocessing & Feature Engineering

**Before the model comes the data. This lab is all about getting it right.**

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-orange?style=flat-square&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-DataFrames-150458?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243?style=flat-square&logo=numpy)

</div>

---

## 📋 Overview

`machine_learning_lab4.ipynb` covers the full data preprocessing pipeline — the essential steps that happen *before* any model is trained. Working across a synthetic dataset and the real-world **Wine dataset (UCI)**, this lab builds hands-on intuition for cleaning, encoding, scaling, and selecting features.

| Section | Topic |
|---|---|
| 1 | Handling Missing Data |
| 2 | Handling Categorical Data |
| 3 | Partitioning a Dataset |
| 4 | Feature Scaling |
| 5 | Selecting Meaningful Features |

---

## 🔍 Section 1 — Handling Missing Data

> *You can't model what you don't have. First, find it. Then decide what to do with it.*

**Exercise 1.1 — Identifying Missing Values**
Constructs a small CSV in-memory using `StringIO` and loads it into a DataFrame. Uses `isnull().sum()` to count missing values per column. Observes how duplicating a row with missing values changes the count — a practical reminder that data entry errors compound.

**Exercise 1.2 — Eliminating Missing Values**
Explores `dropna()` with five different strategies:
- `axis=0` — drop rows with any missing value
- `axis=1` — drop entire columns with any missing value
- `how='all'` — only drop rows where *every* value is missing
- `thresh=4` — keep only rows with at least 4 non-null values
- `subset=['C']` — drop rows only where a specific column is null

Includes a written analysis of *when* each strategy makes sense — dropping rows is appropriate when missing data is rare (<1% of samples); dropping columns is justified when a feature is overwhelmingly empty or irrelevant.

**Exercise 1.3 — Imputing Missing Values**
Uses `sklearn`'s `SimpleImputer` to fill gaps rather than discard them. Compares three strategies side by side:
- `mean` — fills with the column average
- `median` — fills with the column midpoint (robust to outliers)
- `most_frequent` — fills with the most common value, ideal for **categorical features** where mean/median have no meaning

---

## 🏷️ Section 2 — Handling Categorical Data

> *Algorithms speak numbers. Teach them to understand colour, size, and class.*

**Exercise 2.1 — Mapping Ordinal Features**
Creates a clothing DataFrame with a `size` column (`S`, `M`, `L`, `XL`). Maps size labels to integers using a manually defined `size_mapping` dictionary, preserving the natural ordering. Also demonstrates inverse mapping to recover the original labels from integers.

Extended to include a fourth size (`S → 0`), reinforcing the principle that ordinal encoding must reflect the true rank order of the feature.

**Exercise 2.2 — Encoding Class Labels**
Demonstrates *why* raw string labels break sklearn classifiers by attempting to fit a `LogisticRegression` on un-encoded data — then fixing it properly:
- `LabelEncoder` — converts class labels (`class1`, `class2`) to integers and back
- `OneHotEncoder` — creates binary dummy columns for nominal features like `color`
- `pd.get_dummies()` — a pandas shortcut for one-hot encoding with optional `drop_first=True` to **prevent multicollinearity** (where dummy columns become linearly dependent, destabilizing linear models)

Full one-hot encoding is applied to both `color` and `size` simultaneously on the complete DataFrame.

---

## ✂️ Section 3 — Partitioning a Dataset

> *Split smart. A bad split is a silent lie about how well your model works.*

Loads the **Wine dataset** directly from the UCI ML Repository — 178 samples, 3 wine classes, 13 chemical features. Splits into training and test sets using `train_test_split` with the `stratify=y` parameter, ensuring each split preserves the original class proportions.

Compares splits at `test_size=0.3` vs `test_size=0.2` and prints normalised class distributions for both, making the effect of stratification visible. Explains that without `stratify=y`, a random split can create class imbalances that cause the model to appear more (or less) accurate than it truly is.

---

## ⚖️ Section 4 — Feature Scaling

> *A 1-unit difference in alcohol content shouldn't dominate a 1000-unit difference in proline. Scale everything.*

**Exercise 4.1 — Min-Max Normalization**
Applies `MinMaxScaler` to compress all Wine features into the `[0, 1]` range. Fits the scaler on the training set only, then applies it to the test set — preventing data leakage. Plots histograms of the `Alcohol` feature before and after scaling to visualise how the shape is preserved while the range changes.

**Exercise 4.2 — Standardization**
Applies `StandardScaler` to shift features to zero mean and unit variance. Verifies the result by printing the mean (~0.000) and standard deviation (~1.000) for every feature in the scaled training set.

Includes a written comparison: use **standardization** when features have different scales and algorithms assume centered data (e.g. SVM, Logistic Regression); use **normalization** when a fixed output range like [0, 1] is required (e.g. neural networks, image data).

---

## 🎯 Section 5 — Selecting Meaningful Features

> *More features ≠ better model. Find the ones that actually matter.*

**Exercise 5.1 — L1 Regularization for Sparsity**
Trains L1-penalised `LogisticRegression` on standardized Wine data across multiple values of `C` (`0.01` to `100`). Plots how each feature's coefficient evolves as regularization strength changes — features with coefficients driven to zero are effectively eliminated. Demonstrates that weaker regularization (higher `C`) allows more features to remain active, while strong regularization forces the model to select only the most predictive ones.

**Exercise 5.2 — Sequential Backward Selection (SBS)**
Implements a full custom `SBS` class from scratch using `sklearn`'s `BaseEstimator`. The algorithm:
1. Starts with all features
2. At each step, removes the feature whose removal costs the least in accuracy
3. Repeats until the target number of features (`k_features`) is reached

Applied using `KNeighborsClassifier` as the wrapped estimator. Plots accuracy vs. number of features to find the point where fewer features still yields strong performance. Identifies the top 3 features selected by SBS and evaluates their standalone test accuracy.

**Exercise 5.3 — Feature Importance with Random Forests**
Trains a 500-tree `RandomForestClassifier` on the Wine dataset and ranks all 13 features by their Gini importance. Plots a sorted bar chart of importances. Then applies `SelectFromModel` with a `threshold=0.1` to automatically select the most important features and re-evaluates the model on only those features.

Closes with a comparison between features selected by **Random Forest importance** and those selected by **SBS** — two different philosophies (embedded vs wrapper method) that often arrive at overlapping but not identical subsets.

---

## 💡 Key Takeaways

- Proper imputation strategy depends on the data type — mean/median for numeric, mode for categorical
- Ordinal encoding preserves rank; one-hot encoding avoids false ordinality on nominal features
- Always fit scalers on training data only — never on the test set
- L1 regularization is an *embedded* feature selector built into the training process
- SBS is a *wrapper* method — model-agnostic and exhaustive but computationally heavier
- Stratified splitting is non-negotiable on imbalanced datasets

---

## 🔧 Setup

```bash
pip install numpy pandas matplotlib scikit-learn
```

> Notebook developed and run in **Jupyter Notebook / JupyterLab**.

---

## 📊 Dataset Used

- **Wine Dataset (UCI)** — 178 samples, 3 wine cultivar classes, 13 chemical features  
  Loaded directly from: `https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data`

---

<div align="center">
  <sub>Part of a Machine Learning coursework series · Python · scikit-learn · pandas</sub>
</div>
