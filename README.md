# Supervised Pattern Recognition: KNN Tissue Classification
## 📊 Project Pivot & Scope Alignment
This project updates the previous framework, shifting from unsupervised K-Means clustering to a **Supervised K-Nearest Neighbors (KNN) Classification** model. Using structural profiles from the dataset, the updated framework directly predicts diagnostic classes (`Malignant` vs. `Benign`) using an optimized, distance-scaled pipeline.
---
## 🛠️ Supervised Data Preprocessing and Normalization
1. **Target Feature Assignment:** Handled data loading with fallback encodings (`latin1`) to fix character parsing errors. The `diagnosis` attribute was mapped to a binary classification target variable (`Malignant = 1`, `Benign = 0`).
2. **Stratified Partitioning:** Implemented an 80/20 train/test split with stratification (`stratify=y`) to maintain the same ratio of class labels across both partitions.
3. **Distance Standardization Matrix:** Because KNN uses distance metrics (such as Euclidean distance) to identify neighboring points:
   $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
   Features with large raw numerical values (like `area_mean`, which ranges up to 1000+) would dominate features with smaller scales (like `smoothness_mean`, which is less than 1). We apply `StandardScaler` to normalize all variables to a mean of 0 and a variance of 1, ensuring every feature contributes equally to the distance calculation.
---
## 🧬 Hyperparameter Tuning via 5-Fold Cross-Validation
Instead of using the Elbow Method (which applies only to unsupervised cluster variance), we optimized the neighbor hyperparameter ($k \in [1, 3, 5, 7, 9, 11]$) using 5-fold cross-validation on the training set.
* **k = 1:** Yielded a mean cross-validation accuracy of **`0.9250`** (at risk of overfitting due to high local variance).
* **k = 3:** Yielded a mean cross-validation accuracy of **`0.9375`**.
* **k = 5:** Achieved a peak mean cross-validation accuracy of **`0.9625`**.
* **k = 7, 9, 11:** Performance stabilized at **`0.9500`**.
**Optimal Model Decision:** The system selected **$k = 5$** as the optimal choice, balancing local boundary details with global generalization capability.
---
## 🏁 Comprehensive Classification Evaluation Metrics
The final configuration ($k=5$) was validated against the independent 20% holdout split, yielding the following results:
* **Out-of-Sample Overall Accuracy:** **`0.8500`** (85.0% correct classifications on unseen test records).
* **Granular Class Performance Profile:**
  * **Benign Class (0):** Precision = `0.83` | Recall = `0.71` | F1-Score = `0.77`
  * **Malignant Class (1):** Precision = `0.86` | Recall = `0.92` | F1-Score = `0.89`
* **Confusion Matrix Breakdown:**
  * **True Benign (TN):** 5 instances correctly classified.
  * **False Malignant (FP - False Alarms):** 2 instances.
  * **False Benign (FN - Misses):** 1 instance.
  * **True Malignant (TP):** 12 instances correctly identified.
This evaluation demonstrates strong performance. In a diagnostic setting, the high recall for malignant cases (`0.92`) is particularly valuable, as it minimizes dangerous false negatives (missing an actual abnormality).
---
## 📦 Local Installation and Execution (`requirements.txt`)
To run this pipeline locally, install the necessary project dependencies via pip:
```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
