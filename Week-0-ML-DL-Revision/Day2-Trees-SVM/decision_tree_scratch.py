import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ═══════════════════════════════════════════════
# DECISION TREE FROM SCRATCH
# ═══════════════════════════════════════════════

# ─── 1. NODE CLASS ──────────────────────────────
class Node:
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature   = feature    # which feature to split on
        self.threshold = threshold  # split value
        self.left      = left       # left child node
        self.right     = right      # right child node
        self.value     = value      # prediction (only for leaf nodes)

    def is_leaf(self):
        return self.value is not None


# ─── 2. DECISION TREE CLASS ─────────────────────
class DecisionTreeScratch:

    def __init__(self, max_depth=10, min_samples=2):
        self.max_depth   = max_depth    # how deep tree can grow
        self.min_samples = min_samples  # minimum samples to split
        self.root        = None         # root node

    # ── Gini Impurity ─────────────────────────
    def gini(self, y):
        classes = np.unique(y)
        impurity = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            impurity -= p ** 2
        return impurity

    # ── Information Gain ──────────────────────
    def information_gain(self, y, left_y, right_y):
        # weighted average gini of children
        n = len(y)
        weighted = (len(left_y)/n) * self.gini(left_y) + \
                   (len(right_y)/n) * self.gini(right_y)
        return self.gini(y) - weighted

    # ── Find Best Split ───────────────────────
    def best_split(self, X, y):
        best_gain      = -1
        best_feature   = None
        best_threshold = None

        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                # split data
                left_mask  = X[:, feature] <= threshold
                right_mask = X[:, feature] >  threshold

                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue

                left_y  = y[left_mask]
                right_y = y[right_mask]

                # compute gain
                gain = self.information_gain(y, left_y, right_y)

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # ── Build Tree Recursively ────────────────
    def build_tree(self, X, y, depth=0):

        n_samples = len(y)
        n_classes = len(np.unique(y))

        # ── STOPPING CONDITIONS ───────────────
        # Stop if: max depth reached, too few samples, or pure node
        if (depth >= self.max_depth or
            n_samples < self.min_samples or
            n_classes == 1):
            # return leaf node with most common class
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        # ── FIND BEST SPLIT ───────────────────
        feature, threshold = self.best_split(X, y)

        if feature is None:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        # ── SPLIT DATA ────────────────────────
        left_mask  = X[:, feature] <= threshold
        right_mask = X[:, feature] >  threshold

        # ── BUILD CHILDREN RECURSIVELY ────────
        left  = self.build_tree(X[left_mask],  y[left_mask],  depth+1)
        right = self.build_tree(X[right_mask], y[right_mask], depth+1)

        return Node(feature, threshold, left, right)

    # ── Train ─────────────────────────────────
    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        print("Tree built successfully!")

    # ── Predict One Sample ────────────────────
    def predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    # ── Predict All Samples ───────────────────
    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])

    # ── Accuracy ──────────────────────────────
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# ─── 3. LOAD DATA ───────────────────────────────
print("="*50)
print("DECISION TREE FROM SCRATCH")
print("="*50)

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size : {X_train.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")
print(f"Features   : {iris.feature_names}")
print(f"Classes    : {iris.target_names}")

# ─── 4. TRAIN ───────────────────────────────────
print("\nBuilding tree...")
tree = DecisionTreeScratch(max_depth=5)
tree.fit(X_train, y_train)

# ─── 5. EVALUATE ────────────────────────────────
train_acc = tree.accuracy(X_train, y_train)
test_acc  = tree.accuracy(X_test,  y_test)

print(f"\nTrain Accuracy : {train_acc:.4f} ({train_acc*100:.1f}%)")
print(f"Test  Accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")

# ─── 6. COMPARE WITH SKLEARN ────────────────────
print("\n" + "="*50)
print("COMPARISON WITH SKLEARN")
print("="*50)

from sklearn.tree import DecisionTreeClassifier
sk_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
sk_tree.fit(X_train, y_train)
sk_acc = sk_tree.score(X_test, y_test)

print(f"Scratch Accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")
print(f"Sklearn Accuracy : {sk_acc:.4f}  ({sk_acc*100:.1f}%)")
print(f"Difference       : {abs(sk_acc - test_acc):.4f}")  
