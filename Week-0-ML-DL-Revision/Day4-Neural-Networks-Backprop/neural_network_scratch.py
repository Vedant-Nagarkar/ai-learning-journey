import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier

# ═══════════════════════════════════════════════
# NEURAL NETWORK FROM SCRATCH
# Architecture: 4 → 16 → 8 → 3
# ═══════════════════════════════════════════════

class NeuralNetworkScratch:

    def __init__(self, layer_sizes, lr=0.01, epochs=1000):
        self.layer_sizes = layer_sizes  # e.g. [4, 16, 8, 3]
        self.lr          = lr
        self.epochs      = epochs
        self.weights     = []
        self.biases      = []
        self.losses      = []
        self._init_weights()

    # ── 1. INITIALIZE WEIGHTS ─────────────────
    def _init_weights(self):
        np.random.seed(42)
        for i in range(len(self.layer_sizes) - 1):
            n_in  = self.layer_sizes[i]
            n_out = self.layer_sizes[i+1]
            # He initialization (good for ReLU)
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)
        print(f"Network initialized: {self.layer_sizes}")
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            print(f"  Layer {i+1}: W={W.shape}, b={b.shape}")

    # ── 2. ACTIVATION FUNCTIONS ───────────────
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    # ── 3. FORWARD PROPAGATION ────────────────
    def forward(self, X):
        self.activations = [X]   # store all activations
        self.z_values    = []    # store all z values

        a = X
        for i in range(len(self.weights)):
            # z = W·a + b
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)

            # Apply activation
            if i == len(self.weights) - 1:
                # Last layer → Softmax
                a = self.softmax(z)
            else:
                # Hidden layers → ReLU
                a = self.relu(z)

            self.activations.append(a)

        return a  # final output = ŷ

    # ── 4. LOSS — CATEGORICAL CROSS ENTROPY ──
    def compute_loss(self, y_pred, y_true):
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    # ── 5. BACKPROPAGATION ────────────────────
    def backward(self, y_true):
        n = y_true.shape[0]
        # Gradient of loss w.r.t. softmax output
        delta = self.activations[-1] - y_true  # dL/dz (output layer)

        for i in reversed(range(len(self.weights))):
            a_prev = self.activations[i]

            # Gradients for weights and biases
            dW = (a_prev.T @ delta) / n
            db = np.mean(delta, axis=0, keepdims=True)

            # Gradient for previous layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * \
                        self.relu_derivative(self.z_values[i-1])

            # Update weights
            self.weights[i] -= self.lr * dW
            self.biases[i]  -= self.lr * db

    # ── 6. TRAIN ──────────────────────────────
    def fit(self, X, y):
        print(f"\nTraining for {self.epochs} epochs...")
        print(f"Learning rate: {self.lr}")
        print("-" * 45)

        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute loss
            loss = self.compute_loss(y_pred, y)
            self.losses.append(loss)

            # Backward pass
            self.backward(y)

            # Print every 100 epochs
            if epoch % 100 == 0:
                acc = self.accuracy(X, np.argmax(y, axis=1))
                print(f"Epoch {epoch:4d} | Loss={loss:.4f} | Acc={acc:.4f}")

    # ── 7. PREDICT ────────────────────────────
    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def accuracy(self, X, y_true_labels):
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true_labels)

    # ── 8. PLOT LOSS ──────────────────────────
    def plot_loss(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.losses, color='purple', linewidth=2)
        plt.title("Loss Curve — Neural Network from Scratch")
        plt.xlabel("Epoch")
        plt.ylabel("Categorical Cross Entropy Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("nn_loss_curve.png")
        plt.show()
        print("nn_loss_curve.png saved!")


# ─── 1. LOAD & PREPARE DATA ─────────────────────
print("="*55)
print("  NEURAL NETWORK FROM SCRATCH")
print("  Architecture: 4 → 16 → 8 → 3")
print("="*55)

iris = load_iris()
X, y = iris.data, iris.target

# Normalize features
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode labels
# y: [0,1,2] → [[1,0,0], [0,1,0], [0,0,1]]
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Train/test split
X_train, X_test, y_train, y_test, y_train_oh, y_test_oh = \
    train_test_split(X_scaled, y, y_onehot,
                    test_size=0.2, random_state=42)

print(f"\nDataset    : Iris")
print(f"Train size : {X_train.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")
print(f"y_onehot shape: {y_train_oh.shape}")
print(f"\nOne-hot example:")
print(f"  y=0 → {y_train_oh[y_train==0][0]}")
print(f"  y=1 → {y_train_oh[y_train==1][0]}")
print(f"  y=2 → {y_train_oh[y_train==2][0]}")

# ─── 2. BUILD & TRAIN ───────────────────────────
nn = NeuralNetworkScratch(
    layer_sizes = [4, 16, 8, 3],
    lr          = 0.01,
    epochs      = 1000
)

nn.fit(X_train, y_train_oh)

# ─── 3. EVALUATE ────────────────────────────────
print("\n" + "="*55)
print("  RESULTS")
print("="*55)

train_acc = nn.accuracy(X_train, y_train)
test_acc  = nn.accuracy(X_test,  y_test)

print(f"Train Accuracy : {train_acc:.4f} ({train_acc*100:.1f}%)")
print(f"Test  Accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")

# ─── 4. COMPARE WITH SKLEARN MLP ────────────────
print("\n" + "="*55)
print("  COMPARISON WITH SKLEARN MLP")
print("="*55)

mlp = MLPClassifier(
    hidden_layer_sizes = (16, 8),
    activation         = 'relu',
    learning_rate_init = 0.01,
    max_iter           = 1000,
    random_state       = 42
)
mlp.fit(X_train, y_train)
sk_acc = mlp.score(X_test, y_test)

print(f"Scratch Accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")
print(f"Sklearn Accuracy : {sk_acc:.4f}  ({sk_acc*100:.1f}%)")
print(f"Difference       : {abs(sk_acc - test_acc):.4f}")

# ─── 5. PLOT ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(nn.losses, color='purple', linewidth=2)
axes[0].set_title("Loss Curve — Scratch Neural Network")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

# Sklearn loss curve
axes[1].plot(mlp.loss_curve_, color='darkorange', linewidth=2)
axes[1].set_title("Loss Curve — Sklearn MLP")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("nn_comparison.png")
plt.show()
print("\nnn_comparison.png saved!")

# ─── 6. WEIGHT VISUALIZATION ────────────────────
print("\n" + "="*55)
print("  LEARNED WEIGHTS SUMMARY")
print("="*55)

for i, (W, b) in enumerate(zip(nn.weights, nn.biases)):
    print(f"Layer {i+1}: W shape={W.shape} | "
            f"W mean={W.mean():.4f} | W std={W.std():.4f}") 
