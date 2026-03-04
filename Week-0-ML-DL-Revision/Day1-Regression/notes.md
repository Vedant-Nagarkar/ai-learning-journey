  
📓 Day 1 — Theory Notes | Linear & Logistic Regression

1️⃣ What is Machine Learning?
Teaching a computer to learn patterns from data instead of writing rules manually.
Traditional Programming:
Rules + Data → Output

Machine Learning:
Data + Output → Rules (learned automatically)
3 Types of ML:
Supervised   → learns from labeled data (we know the answer)
Unsupervised → finds patterns in unlabeled data
Reinforcement→ learns by trial and error (reward/penalty)

2️⃣ Linear Regression
What it does
Predicts a continuous number
Examples:
- Predict house price from size
- Predict salary from experience
- Predict temperature from date
The Equation
y = w×x + b

y = output (what we predict)
x = input  (our feature)
w = weight (slope of the line)
b = bias   (where line crosses y-axis)
Simple Example
Predict salary from years of experience:

salary = 50000 × experience + 30000
w = 50000
b = 30000

Experience = 3 years → salary = 50000×3 + 30000 = 180000
What the model learns
Before training:  w=0,  b=0  (random/zero start)
After training:   w=2,  b=1  (learned from data)

3️⃣ Cost Function (MSE — Mean Squared Error)
What it is
A number that tells us how wrong our model is.
High cost = model is very wrong ❌
Low cost  = model is accurate  ✅
Goal      = minimize the cost
Formula
MSE = (1/n) × Σ(predicted - actual)²

n         = total number of data points
predicted = w×x + b  (model's guess)
actual    = real value from dataset
Σ         = sum over all data points
Step by Step Example
Data points: 3 houses

House 1: actual=200k, predicted=190k → error=10k  → squared=100
House 2: actual=300k, predicted=310k → error=-10k → squared=100
House 3: actual=250k, predicted=260k → error=-10k → squared=100

MSE = (100 + 100 + 100) / 3 = 100
Why square the errors?
Reason 1: removes negative signs (−10 becomes 100, not −100)
Reason 2: punishes BIG errors more than small ones
           error of 10  → 100  (small punishment)
           error of 100 → 10000 (big punishment)

4️⃣ Gradient Descent
What it is
The algorithm that minimizes the cost by updating weights step by step.
Simple Intuition
Imagine you are blindfolded on a hilly mountain:
→ You want to reach the lowest point (valley)
→ At each step you feel which direction goes downhill
→ You take one small step in that direction
→ Repeat until you reach the bottom

The mountain    = cost function
The lowest point= minimum cost = best weights
Each step       = one weight update
The Update Formula
w = w - lr × dw
b = b - lr × db

w   = current weight
lr  = learning rate (step size)
dw  = gradient (which direction is uphill for w)
Why SUBTRACT the gradient?
Gradient points UPHILL (direction of steepest increase)
We want to go DOWNHILL (decrease the cost)
So we SUBTRACT → we move opposite to gradient
Gradient Formulas for Linear Regression
dw = (2/n) × Σ(predicted - actual) × x
db = (2/n) × Σ(predicted - actual)
One Full Step Example
Current: w=0, b=0, lr=0.1
Data: x=2, actual y=5

Step 1 - Forward pass:
predicted = 0×2 + 0 = 0

Step 2 - Compute error:
error = predicted - actual = 0 - 5 = -5

Step 3 - Compute gradients:
dw = 2 × (-5) × 2 = -20
db = 2 × (-5)     = -10

Step 4 - Update weights:
w = 0 - 0.1×(-20) = 0 + 2 = 2
b = 0 - 0.1×(-10) = 0 + 1 = 1

Step 5 - Next prediction:
predicted = 2×2 + 1 = 5 ✅ (correct already!)

5️⃣ Learning Rate
What it is
Controls how big each step is in gradient descent.
Effect of Different Values
Too HIGH (e.g. lr=10):
loss: 5 → 20 → 8 → 50 → ...  ❌ bouncing, never converges

Too LOW (e.g. lr=0.0001):
loss: 5 → 4.99 → 4.98 → ...  ❌ too slow, takes forever

Just RIGHT (e.g. lr=0.1):
loss: 5 → 3 → 1.5 → 0.5 → 0.1 ✅ smooth decrease
Visualization
High LR:           Low LR:          Good LR:
    *                  *               *
        *              *              *
    *                  *             *
        *              *            *
                       *           *  ← converges smoothly
Values to try
Start with: 0.1
Too slow?   Try: 0.3 or 0.5
Bouncing?   Try: 0.01 or 0.001

6️⃣ Logistic Regression
What it does
Predicts a category (0 or 1, yes or no, spam or not spam).
Examples:
- Email: spam(1) or not spam(0)?
- Tumor: malignant(1) or benign(0)?
- Customer: will buy(1) or not buy(0)?
Why not use Linear Regression for this?
Linear output can be: -500, 0, 1, 999...
We need probabilities: only between 0 and 1

If we predict 999 for "spam" — what does that mean? ❌
We need: 0.95 probability of spam ✅
The Sigmoid Function — The Solution
Squashes ANY number to between 0 and 1:
sigmoid(z) = 1 / (1 + e^(-z))

Input  → Output
z = -10  → sigmoid ≈ 0.00005  (almost 0)
z = -2   → sigmoid ≈ 0.12
z =  0   → sigmoid = 0.5
z = +2   → sigmoid ≈ 0.88
z = +10  → sigmoid ≈ 0.99999  (almost 1)
Full Logistic Regression Flow
Step 1: z = w×x + b          (same as linear regression)
Step 2: ŷ = sigmoid(z)        (squash to 0–1)
Step 3: if ŷ >= 0.5 → class 1
        if ŷ <  0.5 → class 0
Example
Email spam detection:
z = w×x + b = 2.5
ŷ = sigmoid(2.5) = 0.92

0.92 >= 0.5 → SPAM ✅ (92% confident it's spam)

7️⃣ Binary Cross Entropy — Loss for Classification
Why not MSE for classification?
MSE with sigmoid = wavy, non-convex loss surface
Gradient descent gets stuck in wrong places ❌
Binary Cross Entropy = smooth, convex loss surface ✅
Formula
BCE = -(1/n) × Σ[ y×log(ŷ) + (1-y)×log(1-ŷ) ]

y  = actual label (0 or 1)
ŷ  = predicted probability (0.0 to 1.0)
Intuition with Examples
Case 1: actual=1, predicted=0.99
loss = -(1×log(0.99)) = -(-0.004) = 0.004  ← very LOW ✅

Case 2: actual=1, predicted=0.01
loss = -(1×log(0.01)) = -(-4.6)   = 4.6    ← very HIGH ❌

Case 3: actual=0, predicted=0.01
loss = -(log(1-0.01)) = -(-0.004) = 0.004  ← very LOW ✅

Rule: confident and correct = low loss
      confident and wrong   = HIGH loss (big penalty)

8️⃣ Regularization — Preventing Overfitting
What is Overfitting?
Model memorizes training data but fails on new data

Overfitting:  Train=99%, Test=55% ❌ (memorized, not learned)
Good fit:     Train=92%, Test=89% ✅ (actually learned patterns)
Underfitting: Train=60%, Test=58% ❌ (too simple, didn't learn)
L1 Regularization (Lasso)
Adds penalty: λ × Σ|w|

Effect: pushes unimportant weights to exactly ZERO
Result: automatic feature selection
Use when: you have many useless features
L2 Regularization (Ridge)
Adds penalty: λ × Σw²

Effect: shrinks all weights toward zero (but never zero)
Result: all features kept but with smaller influence
Use when: all features are somewhat useful
Quick Decision Rule
Many irrelevant features? → L1 (Lasso)
All features matter?      → L2 (Ridge)
Not sure?                 → Try L2 first
Both?                     → ElasticNet (mix of L1 + L2)

9️⃣ Summary Comparison Table
┌─────────────────────┬──────────────────────┬──────────────────────┐
│                     │  Linear Regression   │ Logistic Regression  │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Task                │ Predict a number     │ Predict a category   │
│ Output              │ Any number           │ 0 to 1 (probability) │
│ Activation          │ None                 │ Sigmoid              │
│ Loss Function       │ MSE                  │ Binary Cross Entropy │
│ Decision            │ Output IS prediction │ >= 0.5 → class 1     │
│ Example             │ House price          │ Spam detection       │
└─────────────────────┴──────────────────────┴──────────────────────┘

🔑 10 Key Points to Remember
1.  Linear Regression → predicts numbers. Logistic → predicts categories
2.  Cost function measures how wrong the model is — we minimize it
3.  MSE = average of squared errors — used for regression
4.  Gradient descent = step-by-step weight updates to reduce cost
5.  Gradient points uphill → we subtract it to go downhill
6.  Learning rate = step size. Too high = bounce. Too low = slow
7.  Sigmoid squashes any number to 0–1 (probability)
8.  Binary Cross Entropy = loss for classification
9.  Overfitting = model memorizes data, fails on new data
10. L1 = sparsity (zeros out weights). L2 = shrinks all weights

✅ Self Test — Answer These Without Looking
Q1. What is the difference between Linear and Logistic Regression?
Q2. What does the cost function measure?
Q3. Why do we subtract the gradient in gradient descent?
Q4. What happens when learning rate is too high?
Q5. Why does Logistic Regression use sigmoid instead of linear output?
Q6. What is overfitting and how does regularization fix it?
Q7. What is the difference between L1 and L2 regularization?
