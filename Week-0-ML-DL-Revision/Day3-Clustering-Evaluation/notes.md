1️⃣ K-Means Clustering

What it is

Groups data into K clusters where each point belongs to the cluster with the nearest centroid. It is unsupervised — no labels needed.

Supervised   → we know the answers (classification, regression)

Unsupervised → no answers given, find hidden patterns (clustering)

How K-Means Works — Step by Step

Step 1: Choose K (number of clusters)

Step 2: Randomly place K centroids


Step 3: Assign each point to nearest centroid

Step 4: Move centroid to mean of its assigned points

Step 5: Repeat Steps 3-4 until centroids stop moving


Visual Example

Before:          After K-Means (K=2):
● ● ●  ■ ■       ● ● ●  ■ ■
  ● ●    ■ ■       ● ●  ✦  ■ ■
● ● ●  ■ ■       ● ● ●  ■ ■
  ✦ = centroid

Inertia — How to Measure Quality

Inertia = sum of squared distances from each point to its centroid


Low inertia  = points are close to centroids = good clusters ✅

High inertia = points are far from centroids = bad clusters ❌

Choosing K — The Elbow Method

Run K-Means for K = 1, 2, 3, 4, 5, 6...

Plot inertia for each K

Find the "elbow" — where inertia stops dropping fast


Inertia
  |
8 |*
6 |  *
4 |    *
3 |      * ← elbow = best K!
2 |        * *
1 |            * *
  └──────────────── K
     1  2  3  4  5

Silhouette Score

Measures how well each point fits its cluster vs other clusters


Score = (b - a) / max(a, b)


a = average distance to points in SAME cluster

b = average distance to points in NEAREST other cluster


Score = +1 → perfectly clustered ✅

Score =  0 → on the boundary between clusters

Score = -1 → wrongly clustered ❌

Limitations of K-Means

❌ Must specify K in advance

❌ Sensitive to outliers

❌ Assumes clusters are spherical and equal size

❌ Different random init can give different results


2️⃣ PCA — Principal Component Analysis

What it is

Reduces the number of features while keeping as much information as possible.

100 features → PCA → 2 features (but keeping 95% of info!)

Why We Need It

Problem 1: Too many features → slow training, overfitting

Problem 2: Features are correlated → redundant information

Problem 3: Can't visualize more than 3 dimensions


PCA solves all 3!

The Intuition

Imagine data as a cloud of points in 3D space.

PCA finds the directions where data spreads the MOST.


PC1 = direction of maximum variance (most spread)

PC2 = direction of second most variance (perpendicular to PC1)

PC3 = direction of least variance


We keep only PC1 and PC2 → 2D projection of 3D data

How PCA Works — Step by Step

Step 1: Standardize data (mean=0, std=1)

Step 2: Compute covariance matrix

Step 3: Find eigenvectors and eigenvalues

Step 4: Sort by eigenvalue (highest = most variance)

Step 5: Pick top K eigenvectors → principal components

Step 6: Project data onto these components


Explained Variance

Each principal component explains some % of total variance


PC1: explains 72% of variance

PC2: explains 23% of variance

PC3: explains  4% of variance

PC4: explains  1% of variance


Keep PC1+PC2 → we keep 95% of information with just 2 features!

When to Use PCA

✅ Too many features (dimensionality reduction)

✅ Features are highly correlated

✅ Want to visualize high dimensional data in 2D/3D

✅ Want to speed up model training

❌ Need interpretable features (PCA components are hard to explain)


3️⃣ Evaluation Metrics

For Classification

Confusion Matrix
                 Predicted
                 Pos    Neg
Actual  Pos  |  TP  |  FN  |
        Neg  |  FP  |  TN  |


TP = True Positive  (predicted YES, actually YES) ✅

TN = True Negative  (predicted NO,  actually NO)  ✅

FP = False Positive (predicted YES, actually NO)  ❌ Type I error

FN = False Negative (predicted NO,  actually YES) ❌ Type II error

Accuracy

Accuracy = (TP + TN) / (TP + TN + FP + FN)


Use when: classes are balanced
Avoid when: imbalanced classes
Example: 99% accuracy on 99% majority class = useless model!

Precision

Precision = TP / (TP + FP)

"Of all positive predictions, how many were actually positive?"
High precision = few false alarms
Use when: false positives are costly
Example: spam filter (don't want real emails in spam)

Recall (Sensitivity)

Recall = TP / (TP + FN)
"Of all actual positives, how many did we catch?"
High recall = few missed positives
Use when: false negatives are costly
Example: cancer detection (don't want to miss cancer!)

F1 Score
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Harmonic mean of precision and recall
Use when: classes are imbalanced AND both FP and FN matter

Precision vs Recall Tradeoff

Lower threshold → more positives predicted → higher recall, lower precision

Higher threshold → fewer positives predicted → lower recall, higher precision


Example (spam filter):

Strict threshold → misses some spam (low recall) but no real emails in spam (high precision)

Loose threshold  → catches all spam (high recall) but some real emails flagged (low precision)

ROC-AUC

ROC curve: plots TPR (recall) vs FPR at different thresholds

AUC: area under the ROC curve

AUC = 1.0 → perfect classifier ✅
AUC = 0.5 → random guessing ❌
AUC = 0.0 → perfectly wrong ❌


Use when: comparing classifiers regardless of threshold
Avoid when: highly imbalanced data → use PR-AUC instead


For Regression
MAE — Mean Absolute Error
MAE = (1/n) × Σ|predicted - actual|

Easy to interpret: average error in same units as target
Less sensitive to outliers than MSE

MSE — Mean Squared Error

MSE = (1/n) × Σ(predicted - actual)²

Penalizes large errors more heavily
Sensitive to outliers
RMSE — Root Mean Squared Error
RMSE = √MSE


Same units as target variable
Most commonly reported metric for regression
R² Score
R² = 1 - (SS_residual / SS_total)

R² = 1.0 → perfect predictions ✅
R² = 0.0 → model is as good as predicting the mean
R² < 0.0 → model is worse than predicting the mean ❌

Interpretation: R²=0.85 means model explains 85% of variance in data


4️⃣ Cross Validation

What it is

A technique to reliably estimate model performance on unseen data.

K-Fold Cross Validation

Split data into K folds (usually 5 or 10)


Fold 1: [TEST] [train] [train] [train] [train]

Fold 2: [train] [TEST] [train] [train] [train]

Fold 3: [train] [train] [TEST] [train] [train]

Fold 4: [train] [train] [train] [TEST] [train]

Fold 5: [train] [train] [train] [train] [TEST]


Final score = average of 5 test scores


More reliable than single train/test split!
Stratified K-Fold
Same as K-Fold but ensures each fold has
same proportion of each class as the full dataset


Use for: imbalanced classification problems


5️⃣ Quick Reference — Which Metric to Use?
┌─────────────────────────────────────────────────────────────┐
│ Situation                          → Use This Metric        │
├─────────────────────────────────────────────────────────────┤
│ Balanced classes                   → Accuracy               │
│ Imbalanced classes                 → F1 Score or ROC-AUC    │
│ Missing positives is costly        → Recall (cancer, fraud) │
│ False alarms are costly            → Precision (spam)       │
│ Compare classifiers overall        → ROC-AUC                │
│ Regression, interpretable error    → MAE                    │
│ Regression, penalize big errors    → RMSE                   │
│ Regression, % variance explained   → R²                     │
└─────────────────────────────────────────────────────────────┘


🔑 10 Key Points to Remember

1.  K-Means is unsupervised — groups similar points together

2.  Elbow method finds the best K by plotting inertia vs K

3.  Silhouette score measures cluster quality (+1 best, -1 worst)

4.  PCA finds directions of maximum variance in data

5.  PC1 explains the most variance, PC2 second most, etc.

6.  Keep enough PCs to explain 95%+ of variance

7.  Accuracy is misleading on imbalanced datasets

8.  Precision = quality of positive predictions

9.  Recall = coverage of actual positives

10. F1 = balance between precision and recall 
