# Diagnostic Analysis via Instance-Based KNN Learning
## 📈 Executive Summary and Project Goals
This project implements a K-Nearest Neighbors (KNN) classification framework to accurately identify whether observed tissues are Malignant or Benign. The pipeline handles data cleaning, normalizes features to prevent scaling bias, tunes the value of $k$ through cross-validation, and provides comprehensive performance metrics.
---
## 🛠️ Data Wrangling and Normalization Process
1.  **Ingestion & Format Adjustments:** The raw tracking document contains an initial line grouping error. To fix this, the loading pipeline skips the corrupted header tracking line and structures the feature metrics manually into distinct columns (`radius_mean`, `texture_mean`, `perimeter_mean`, etc.).
2.  **Missing Value Mitigation:** Rows containing incomplete observations are removed to protect the distance space from imputation bias.
3.  **Why Feature Scaling is Crucial for KNN:**
    KNN relies completely on distance metrics (like Euclidean distance) to determine point proximity:
    $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
    Features with large numeric variations (such as `area_mean` which ranges up to 1000+) will completely overwhelm features with small values (such as `smoothness_mean` which stays below 1). We apply `StandardScaler` to bring all features to a mean of 0 and variance of 1, ensuring every attribute contributes equally to the distance calculations.
---
## 🧪 Hyperparameter Optimization Tuning
To find the optimal balance between underfitting and overfitting, we evaluate several values for neighbor parameters ($k = 1, 3, 5, 7, 9, 11$) using 5-fold cross-validation across the training partition.

* **Low values of $k$ (e.g., $k=1$):** High variance, capturing noise and creating complex, erratic decision boundaries.
* **High values of $k$ (e.g., $k=11$):** High bias, smoothing the decision boundaries too much and potentially missing localized patterns.
* **Optimal Setting:** The script automatically flags the neighbor parameter that achieves the highest mean cross-validation accuracy, which serves as our final configuration model.
---
## 📋 Comprehensive Model Performance Framework
The final optimized KNN model is validated using a held-out test split (20% of the dataset) to ensure an unbiased evaluation. Performance is assessed using several key classification metrics:
* **Accuracy:** Overall proportion of correct classifications.
* **Precision:** Out of all cases predicted as malignant, the percentage that are truly malignant (critical for minimizing false alarms).
* **Recall (Sensitivity):** Out of all actual malignant cases, the percentage the model successfully identified (vital for ensuring no dangerous cases are missed).
* **F1-Score:** The harmonic mean of Precision and Recall, providing a reliable single measure of performance for balanced classes.
* **Confusion Matrix Heatmap:** Provides a clear visual breakdown of true positives, true negatives, false positives, and false negatives.
---
## 🚀 Environment Requirements (`requirements.txt`)
To run this pipeline locally, save these dependencies in a text file named `requirements.txt` and install them via pip:
```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
