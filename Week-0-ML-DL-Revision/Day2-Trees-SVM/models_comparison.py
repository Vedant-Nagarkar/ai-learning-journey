import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ═══════════════════════════════════════════════
# ALL 4 MODELS COMPARISON
# ═══════════════════════════════════════════════

# ─── 1. LOAD & PREPARE DATA ─────────────────────
print("="*55)
print("  MODEL COMPARISON — DECISION TREE vs RF vs GB vs SVM")
print("="*55)

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Dataset    : Iris ({len(X)} samples, {X.shape[1]} features)")
print(f"Classes    : {iris.target_names}")
print(f"Train/Test : {len(X_train)}/{len(X_test)} samples")

# ─── 2. DEFINE ALL MODELS ───────────────────────
models = {
    "Decision Tree"     : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"     : RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting" : GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM"               : SVC(kernel='rbf', C=1.0, random_state=42),
}

# ─── 3. TRAIN & EVALUATE ALL MODELS ────────────
print("\n" + "="*55)
print("  RESULTS")
print("="*55)
print(f"{'Model':<22} {'Train Acc':>10} {'Test Acc':>10} {'CV Score':>10}")
print("-"*55)

results = {}

for name, model in models.items():
    # Use scaled data for SVM, raw for tree models
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        train_acc = model.score(X_train_scaled, y_train)
        test_acc  = model.score(X_test_scaled,  y_test)
        cv_scores = cross_val_score(model,
                    scaler.fit_transform(X), y, cv=5)
    else:
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc  = model.score(X_test,  y_test)
        cv_scores = cross_val_score(model, X, y, cv=5)

    results[name] = {
        "train_acc" : train_acc,
        "test_acc"  : test_acc,
        "cv_mean"   : cv_scores.mean(),
        "model"     : model
    }

    print(f"{name:<22} {train_acc:>10.4f} {test_acc:>10.4f} {cv_scores.mean():>10.4f}")

# ─── 4. FIND WINNER ─────────────────────────────
best_model = max(results, key=lambda x: results[x]["cv_mean"])
print("-"*55)
print(f"🏆 Best Model (by CV score): {best_model}")

# ─── 5. FEATURE IMPORTANCE ──────────────────────
print("\n" + "="*55)
print("  FEATURE IMPORTANCE (Random Forest)")
print("="*55)

rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_

for feat, imp in zip(iris.feature_names, importances):
    bar = "█" * int(imp * 50)
    print(f"{feat:<25} {imp:.4f}  {bar}")

# ─── 6. PLOT 1 — ACCURACY COMPARISON ───────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Bar chart
model_names  = list(results.keys())
test_accs    = [results[m]["test_acc"]  for m in model_names]
train_accs   = [results[m]["train_acc"] for m in model_names]
cv_scores    = [results[m]["cv_mean"]   for m in model_names]

x = np.arange(len(model_names))
width = 0.25

axes[0].bar(x - width, train_accs, width, label="Train",    color="steelblue")
axes[0].bar(x,         test_accs,  width, label="Test",     color="darkorange")
axes[0].bar(x + width, cv_scores,  width, label="CV Mean",  color="green")
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=15, ha='right')
axes[0].set_ylim(0.8, 1.05)
axes[0].set_title("Model Accuracy Comparison")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Feature importance
feat_names = iris.feature_names
feat_imp   = rf_model.feature_importances_
colors     = ["steelblue", "darkorange", "green", "red"]

axes[1].barh(feat_names, feat_imp, color=colors)
axes[1].set_title("Feature Importance (Random Forest)")
axes[1].set_xlabel("Importance")
axes[1].grid(axis='x', alpha=0.3)

for i, v in enumerate(feat_imp):
    axes[1].text(v + 0.005, i, f"{v:.3f}", va='center')

# Confusion matrix for best model
best = results[best_model]["model"]
if best_model == "SVM":
    y_pred = best.predict(X_test_scaled)
else:
    y_pred = best.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=iris.target_names)
disp.plot(ax=axes[2], colorbar=False)
axes[2].set_title(f"Confusion Matrix — {best_model}")

plt.tight_layout()
plt.savefig("models_comparison.png")
plt.show()
print("\nmodels_comparison.png saved!")

# ─── 7. OVERFITTING CHECK ───────────────────────
print("\n" + "="*55)
print("  OVERFITTING CHECK")
print("  (big gap between train & test = overfitting)")
print("="*55)
print(f"{'Model':<22} {'Train':>8} {'Test':>8} {'Gap':>8} {'Status':>12}")
print("-"*55)

for name in model_names:
    train = results[name]["train_acc"]
    test  = results[name]["test_acc"]
    gap   = train - test
    status = "⚠️ Overfit" if gap > 0.05 else "✅ Good"
    print(f"{name:<22} {train:>8.4f} {test:>8.4f} {gap:>8.4f} {status:>12}") 
