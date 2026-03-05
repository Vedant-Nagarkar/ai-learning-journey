📓 Day 2 — Theory Notes | Decision Trees, Random Forest, Gradient Boosting & SVM

1️⃣ Decision Trees

What it is

A model that makes decisions by asking yes/no questions about features — like a flowchart.
Is age > 30?
├── YES → Is salary > 50k?
│         ├── YES → Will Buy ✅
│         └── NO  → Won't Buy ❌
└── NO  → Won't Buy ❌
Key Terms
Root Node   = first question (top of tree)
Branch      = answer to a question (yes/no)
Leaf Node   = final answer (prediction)
Depth       = how many levels deep the tree goes
How it Splits — Gini Impurity
Measures how mixed a node is:
Gini = 1 - Σ(pᵢ²)

pᵢ = proportion of class i in the node

Pure node   (all same class): Gini = 0   ✅ best
Mixed node  (50/50 classes):  Gini = 0.5 ❌ worst
Gini Example
Node has 10 samples: 8 cats, 2 dogs
p_cat = 8/10 = 0.8
p_dog = 2/10 = 0.2

Gini = 1 - (0.8² + 0.2²)
     = 1 - (0.64 + 0.04)
     = 1 - 0.68
     = 0.32
Information Gain
Measures how much a split reduces impurity:
Information Gain = Gini(parent) - weighted average Gini(children)

Higher gain = better split → tree picks this split
Pros and Cons
Pros:
✅ Easy to understand and visualize
✅ No need to normalize features
✅ Works for both classification and regression

Cons:
❌ Overfits easily (memorizes training data)
❌ Unstable — small data change = very different tree
❌ Biased toward features with more categories

2️⃣ Random Forest
What it is
Builds many decision trees and combines their predictions.
Single tree = one doctor's opinion
Random Forest = 100 doctors voting → majority wins
How it Works — Bagging
Step 1: Take random sample of data (with replacement) → "Bootstrap"
Step 2: Build a decision tree on that sample
Step 3: At each split, only consider random subset of features
Step 4: Repeat 100+ times → 100 different trees
Step 5: Final prediction = majority vote (classification)
                         = average (regression)
Why Random Forest Beats Single Tree
Single Tree:  high variance → small data change = completely different tree
Random Forest: averages out errors → much more stable and accurate

Key insight: trees make DIFFERENT errors
             averaging different errors cancels them out ✅
Key Parameters
n_estimators  = number of trees (more = better, but slower)
max_depth     = how deep each tree grows
max_features  = how many features to consider at each split
Bias vs Variance
Bias     = how wrong the model is on average (underfitting)
Variance = how much model changes with different data (overfitting)

Single Tree:    low bias,  HIGH variance ❌
Random Forest:  low bias,  LOW variance  ✅ (bagging reduces variance)

3️⃣ Gradient Boosting
What it is
Builds trees sequentially — each tree fixes the mistakes of the previous one.
Tree 1: makes predictions → has errors
Tree 2: learns to predict Tree 1's errors
Tree 3: learns to predict remaining errors
...
Final = Tree1 + Tree2 + Tree3 + ... (weighted sum)
Bagging vs Boosting
Bagging (Random Forest):
→ Trees built in PARALLEL (independently)
→ Reduces VARIANCE
→ Each tree sees different data sample

Boosting (Gradient Boosting):
→ Trees built SEQUENTIALLY (one after another)
→ Reduces BIAS
→ Each tree focuses on previous tree's mistakes
Popular Implementations
GradientBoostingClassifier  → sklearn (slow but reliable)
XGBoost                     → faster, regularization built in
LightGBM                    → very fast, great for large data
CatBoost                    → handles categorical features well
When to Use
Tabular data competition? → XGBoost or LightGBM almost always wins
Need interpretability?    → Single Decision Tree
Need robustness?          → Random Forest

4️⃣ Support Vector Machine (SVM)
What it is
Finds the best boundary (hyperplane) that separates two classes with the maximum margin.
Class A: ● ● ●
                  ←margin→  |decision boundary|  ←margin→
Class B: ■ ■ ■
Key Concepts
Hyperplane
In 2D: a line that separates classes
In 3D: a plane that separates classes
In nD: a hyperplane
Support Vectors
The data points CLOSEST to the decision boundary
These are the only points that matter for SVM
Move any other point → boundary stays the same
Move a support vector → boundary changes!
Margin
Distance between the decision boundary and nearest points
SVM goal: MAXIMIZE this margin
Large margin = more confident separation = better generalization
Hard Margin vs Soft Margin
Hard Margin: no points allowed inside margin (only works if data is perfectly separable)
Soft Margin: allows some points inside margin (real world data — not perfectly separable)

C parameter controls this:
High C → narrow margin, few errors allowed (can overfit)
Low C  → wide margin, more errors allowed (can underfit)
The Kernel Trick — For Non-Linear Data
What if data is NOT linearly separable?
Original 2D data:     After kernel transform:
  ● inside ■ ring  →  ● and ■ perfectly separable in 3D!

Kernel maps data to higher dimensions implicitly
(never actually computes the transformation — very efficient!)
Common Kernels
Linear  → use when data is linearly separable
RBF     → use for non-linear data (most common)
Poly    → use for polynomial boundaries
Sigmoid → rarely used
When to Use SVM
✅ Small to medium datasets
✅ High dimensional data (text classification)
✅ Clear margin of separation exists
❌ Very large datasets (slow to train)
❌ Lots of noise in data

5️⃣ Comparison Table
┌──────────────────┬──────────┬───────────────┬──────────────────┬──────────┐
│                  │ Decision │    Random     │    Gradient      │   SVM    │
│                  │  Tree    │    Forest     │    Boosting      │          │
├──────────────────┼──────────┼───────────────┼──────────────────┼──────────┤
│ Type             │ Single   │  Ensemble     │  Ensemble        │ Margin   │
│ Speed            │ Fast     │  Medium       │  Slow            │ Medium   │
│ Overfitting      │ High     │  Low          │  Medium          │ Low      │
│ Interpretable    │ Yes ✅   │  Partial      │  No              │ No       │
│ Best for         │ Simple   │  Most tasks   │  Competitions    │ High dim │
│ Key parameter    │ depth    │  n_estimators │  learning_rate   │ C, kernel│
└──────────────────┴──────────┴───────────────┴──────────────────┴──────────┘

6️⃣ When to Use Which Model
Start here → Try Random Forest first (almost always good baseline)
      ↓
Need more accuracy?  → Try Gradient Boosting (XGBoost)
Need interpretability? → Use single Decision Tree
High dimensional text? → Use SVM with linear kernel
Small dataset?         → SVM often works well

🔑 10 Key Points to Remember
1.  Decision Tree splits data by asking yes/no questions
2.  Gini impurity measures how mixed a node is (0=pure, 0.5=mixed)
3.  Information gain = how much a split reduces impurity
4.  Decision Trees overfit easily — use Random Forest instead
5.  Random Forest = many trees + majority vote → reduces variance
6.  Bagging = parallel trees on random data samples
7.  Boosting = sequential trees, each fixing previous errors
8.  SVM finds the boundary with maximum margin between classes
9.  Support vectors = only the closest points define the boundary
10. Kernel trick maps non-linear data to higher dimensions
