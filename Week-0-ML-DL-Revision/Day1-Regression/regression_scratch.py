import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════
# LINEAR REGRESSION FROM SCRATCH
# Goal: learn y = 2x + 1 from data
# ═══════════════════════════════════════════════

# ─── 1. CREATE FAKE DATASET ─────────────────────
np.random.seed(42)
X = np.random.randn(100)          # 100 random input values
y = 2 * X + 1 + np.random.randn(100) * 0.5  # true: w=2, b=1

print("Dataset created!")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"First 5 X values: {X[:5].round(2)}")
print(f"First 5 y values: {y[:5].round(2)}")

# ─── 2. LINEAR REGRESSION CLASS ─────────────────
class LinearRegressionScratch:

    def __init__(self, lr=0.1, epochs=1000):
        self.lr     = lr        # learning rate
        self.epochs = epochs    # number of iterations
        self.w      = 0.0       # weight (start at 0)
        self.b      = 0.0       # bias  (start at 0)
        self.losses = []        # store loss history

    def predict(self, X):
        # y = w*x + b
        return self.w * X + self.b

    def mse_loss(self, y_pred, y_true):
        # Mean Squared Error
        return np.mean((y_pred - y_true) ** 2)

    def fit(self, X, y):
        n = len(y)

        for epoch in range(self.epochs):

            # ── FORWARD PASS ──────────────────────
            y_pred = self.predict(X)

            # ── COMPUTE LOSS ──────────────────────
            loss = self.mse_loss(y_pred, y)
            self.losses.append(loss)

            # ── COMPUTE GRADIENTS ─────────────────
            # How much does loss change if we change w?
            dw = (2/n) * np.sum((y_pred - y) * X)
            # How much does loss change if we change b?
            db = (2/n) * np.sum(y_pred - y)

            # ── UPDATE WEIGHTS ────────────────────
            # Move in opposite direction of gradient
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # ── PRINT EVERY 100 EPOCHS ────────────
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss={loss:.4f} | w={self.w:.4f} | b={self.b:.4f}")

    def plot_loss(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.losses, color='blue')
        plt.title("Loss Curve — Should Go Down Smoothly")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("loss_curve.png")
        plt.show()
        print("loss_curve.png saved!")

    def plot_fit(self, X, y):
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, alpha=0.5, label="Actual Data", color='steelblue')
        X_line = np.linspace(X.min(), X.max(), 100)
        plt.plot(X_line, self.predict(X_line), color='red', linewidth=2,
                 label=f"Predicted: y={self.w:.2f}x+{self.b:.2f}")
        plt.title("Linear Regression from Scratch")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("regression_fit.png")
        plt.show()
        print("regression_fit.png saved!")


# ─── 3. TRAIN THE MODEL ─────────────────────────
print("\n" + "="*50)
print("TRAINING LINEAR REGRESSION FROM SCRATCH")
print("="*50)

model = LinearRegressionScratch(lr=0.1, epochs=1000)
model.fit(X, y)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Learned w = {model.w:.4f}  (true value = 2.0)")
print(f"Learned b = {model.b:.4f}  (true value = 1.0)")
print(f"Final Loss = {model.losses[-1]:.4f}")

# ─── 4. PLOT RESULTS ────────────────────────────
model.plot_loss()
model.plot_fit(X, y)

# ─── 5. COMPARE WITH SKLEARN ────────────────────
print("\n" + "="*50)
print("COMPARISON WITH SKLEARN")
print("="*50)

from sklearn.linear_model import LinearRegression
sk = LinearRegression()
sk.fit(X.reshape(-1, 1), y)

print(f"Sklearn  w = {sk.coef_[0]:.4f}")
print(f"Scratch  w = {model.w:.4f}")
print(f"Sklearn  b = {sk.intercept_:.4f}")
print(f"Scratch  b = {model.b:.4f}")
print(f"Difference = {abs(sk.coef_[0] - model.w):.6f} (should be tiny!)")
