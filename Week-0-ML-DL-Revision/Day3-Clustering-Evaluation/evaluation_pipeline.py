import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)

# ═══════════════════════════════════════════════
# COMPLETE EVALUATION PIPELINE
# ═══════════════════════════════════════════════

# ─── 1. LOAD DATA ───────────────────────────────
print("="*55)
print("  COMPLETE ML EVALUATION PIPELINE")
print("="*55)

# Use breast cancer dataset (binary classification)
data    = load_breast_cancer()
X, y    = data.data, data.target

print(f"Dataset  : Breast Cancer")
print(f"Samples  : {X.shape[0]}")
print(f"Features : {X.shape[1]}")
print(f"Classes  : {data.target_names}")
print(f"Class 0  : {sum(y==0)} malignant")
print(f"Class 1  : {sum(y==1)} benign")

# ─── 2. PREPARE DATA ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

# ─── 3. TRAIN MODEL ─────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# ─── 4. ALL CLASSIFICATION METRICS ─────────────
print("\n" + "="*55)
print("  CLASSIFICATION METRICS")
print("="*55)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"Precision : {precision:.4f}  ({precision*100:.1f}%)")
print(f"Recall    : {recall:.4f}  ({recall*100:.1f}%)")
print(f"F1 Score  : {f1:.4f}  ({f1*100:.1f}%)")
print(f"ROC-AUC   : {roc_auc:.4f}  ({roc_auc*100:.1f}%)")

# ─── 5. CROSS VALIDATION ────────────────────────
print("\n" + "="*55)
print("  5-FOLD CROSS VALIDATION")
print("="*55)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"F1 per fold : {cv_scores.round(4)}")
print(f"Mean F1     : {cv_scores.mean():.4f}")
print(f"Std F1      : {cv_scores.std():.4f}")

# ─── 6. REGRESSION METRICS DEMO ─────────────────
print("\n" + "="*55)
print("  REGRESSION METRICS DEMO")
print("="*55)

# Use probabilities as "predictions" to demo regression metrics
y_true_reg = y_test.astype(float)
y_pred_reg = y_pred_prob

mae  = mean_absolute_error(y_true_reg, y_pred_reg)
mse  = mean_squared_error(y_true_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2   = r2_score(y_true_reg, y_pred_reg)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f} (model explains {r2*100:.1f}% of variance)")

# ─── 7. PLOTS ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Confusion Matrix
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=data.target_names)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Confusion Matrix")

# Add metric annotations
axes[0].text(0.02, -0.18,
    f"Accuracy={accuracy:.3f}  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}",
    transform=axes[0].transAxes, fontsize=9, color='navy')

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC={roc_auc:.3f})')
axes[1].plot([0,1],[0,1], color='navy', linestyle='--', label='Random')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='darkorange')
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate (Recall)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
axes[2].plot(rec, prec, color='green', lw=2, label=f'F1={f1:.3f}')
axes[2].fill_between(rec, prec, alpha=0.1, color='green')
axes[2].set_title("Precision-Recall Curve")
axes[2].set_xlabel("Recall")
axes[2].set_ylabel("Precision")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("evaluation_pipeline.png")
plt.show()
print("\nevaluation_pipeline.png saved!")

# ─── 8. OVERFITTING CHECK ───────────────────────
print("\n" + "="*55)
print("  OVERFITTING CHECK")
print("="*55)

train_f1 = f1_score(y_train, model.predict(X_train))
test_f1  = f1_score(y_test,  model.predict(X_test))
gap      = train_f1 - test_f1

print(f"Train F1 : {train_f1:.4f}")
print(f"Test  F1 : {test_f1:.4f}")
print(f"Gap      : {gap:.4f}")
print(f"Status   : {'⚠️ Overfitting!' if gap > 0.05 else '✅ Good fit!'}")
