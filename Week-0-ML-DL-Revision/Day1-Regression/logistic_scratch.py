import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ═══════════════════════════════════════════════
# LOGISTIC REGRESSION FROM SCRATCH
# Goal: classify flowers (class 0 vs class 1)
# ═══════════════════════════════════════════════

# ─── 1. LOAD DATASET ────────────────────────────
iris = load_iris()

# Use only first 2 classes (binary classification)
# Use only first 2 features (easier to visualize)
X = iris.data[:100, :2]
y = iris.target[:100]

print("Dataset loaded!")
print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"Classes : {np.unique(y)}")

# ─── 2. PREPROCESS ──────────────────────────────
# Normalize features (important for gradient descent!)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size : {X_train.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")

# ─── 3. LOGISTIC REGRESSION CLASS ───────────────
class LogisticRegressionScratch:

    def __init__(self, lr=0.1, epochs=1000):
        self.lr     = lr
        self.epochs = epochs
        self.w      = None
        self.b      = 0.0
        self.losses = []

    def sigmoid(self, z):
        # Squash any number to 0–1
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        # Returns probability of class 1
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def predict(self, X):
        # Returns 0 or 1
        return (self.predict_proba(X) >= 0.5).astype(int)

    def binary_cross_entropy(self, y_pred, y_true):
        # Clip to avoid log(0) error
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

    def fit(self, X, y):
        n, n_features = X.shape
        # Initialize weights to zero
        self.w = np.zeros(n_features)

        for epoch in range(self.epochs):

            # ── FORWARD PASS ──────────────────────
            y_pred = self.predict_proba(X)

            # ── COMPUTE LOSS ──────────────────────
            loss = self.binary_cross_entropy(y_pred, y)
            self.losses.append(loss)

            # ── COMPUTE GRADIENTS ─────────────────
            dw = (1/n) * np.dot(X.T, (y_pred - y))
            db = (1/n) * np.sum(y_pred - y)

            # ── UPDATE WEIGHTS ────────────────────
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # ── PRINT EVERY 100 EPOCHS ────────────
            if epoch % 100 == 0:
                acc = np.mean(self.predict(X) == y)
                print(f"Epoch {epoch:4d} | Loss={loss:.4f} | Accuracy={acc:.4f}")

    def plot_loss(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.losses, color='green')
        plt.title("Loss Curve — Logistic Regression")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross Entropy Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("logistic_loss.png")
        plt.show()
        print("logistic_loss.png saved!")


# ─── 4. TRAIN THE MODEL ─────────────────────────
print("\n" + "="*50)
print("TRAINING LOGISTIC REGRESSION FROM SCRATCH")
print("="*50)

model = LogisticRegressionScratch(lr=0.1, epochs=1000)
model.fit(X_train, y_train)

# ─── 5. EVALUATE ────────────────────────────────
print("\n" + "="*50)
print("RESULTS")
print("="*50)

train_acc = np.mean(model.predict(X_train) == y_train)
test_acc  = np.mean(model.predict(X_test)  == y_test)

print(f"Train Accuracy : {train_acc:.4f} ({train_acc*100:.1f}%)")
print(f"Test  Accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")

# ─── 6. COMPARE WITH SKLEARN ────────────────────
print("\n" + "="*50)
print("COMPARISON WITH SKLEARN")
print("="*50)

sk_model = LogisticRegression()
sk_model.fit(X_train, y_train)
sk_acc = sk_model.score(X_test, y_test)

print(f"Scratch Accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")
print(f"Sklearn Accuracy : {sk_acc:.4f}  ({sk_acc*100:.1f}%)")
print(f"Difference       : {abs(sk_acc - test_acc):.4f} (should be very small!)")

# ─── 7. PLOT LOSS CURVE ─────────────────────────
model.plot_loss() 
