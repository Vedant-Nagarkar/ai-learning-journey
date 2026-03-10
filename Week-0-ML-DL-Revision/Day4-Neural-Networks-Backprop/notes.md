1️⃣ What is a Neural Network?
A neural network is a series of layers of neurons that learn to map inputs to outputs by adjusting weights through training.
Input Layer → Hidden Layers → Output Layer
   (data)      (learning)      (prediction)
Inspired by the human brain — neurons connected to neurons, passing signals forward.

2️⃣ Forward Propagation
The Two Steps Per Layer
Step 1 — Linear:     z = W·x + b
Step 2 — Non-linear: a = activation(z)

Without Step 2, any deep network collapses to just:
y = Wx + b  (simple linear regression — useless for complex problems!)
Full Network Example (2 hidden layers)
Layer 1:  z¹ = W¹·X  + b¹  →  a¹ = ReLU(z¹)
Layer 2:  z² = W²·a¹ + b²  →  a² = ReLU(z²)
Output:   z³ = W³·a² + b³  →  ŷ  = Sigmoid(z³)
Activation Functions
┌──────────┬─────────────────────┬────────────┬─────────────────────┐
│ Name     │ Formula             │ Output     │ Use When            │
├──────────┼─────────────────────┼────────────┼─────────────────────┤
│ Sigmoid  │ 1/(1+e⁻ᶻ)          │ 0 to 1     │ Binary output       │
│ Tanh     │ (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) │ -1 to 1    │ Hidden (old)        │
│ ReLU     │ max(0, z)           │ 0 to ∞     │ Hidden layers ✅    │
│ Softmax  │ eᶻⁱ/Σeᶻʲ           │ 0 to 1 sum │ Multiclass output   │
└──────────┴─────────────────────┴────────────┴─────────────────────┘

ReLU is the default choice for hidden layers because:
→ Simple and fast to compute
→ No vanishing gradient (derivative = 1 for z>0)
→ Works well in practice

3️⃣ Loss Functions
What Loss Measures
Prediction: ŷ = 0.9   True label: y = 1   → Loss is low  ✅
Prediction: ŷ = 0.1   True label: y = 1   → Loss is high ❌

Loss = how wrong the model is
Goal = minimize loss during training
MSE — For Regression
L = (1/n) Σ(ŷᵢ - yᵢ)²

Example:
  Predicted house price: $250k
  Actual house price:    $300k
  Error = ($250k - $300k)² = very large penalty!
Binary Cross Entropy — For Binary Classification
L = -(1/n) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]

Intuition:
  y=1 → we want ŷ close to 1 → -log(ŷ) is small when ŷ≈1 ✅
  y=0 → we want ŷ close to 0 → -log(1-ŷ) is small when ŷ≈0 ✅
Categorical Cross Entropy — For Multiclass
L = -(1/n) Σ Σ yᵢⱼ · log(ŷᵢⱼ)

Used with Softmax output
Example: classifying digits 0-9
Choosing the Right Loss
Problem Type          → Loss Function        → Output Activation
──────────────────────────────────────────────────────────────────
Regression            → MSE / MAE            → Linear (none)
Binary Classification → Binary Cross Entropy → Sigmoid
Multiclass            → Categorical CE       → Softmax

4️⃣ Chain Rule of Derivatives
Why We Need It
We want: dL/dW  (how does loss change when we change weight W?)

But W is buried deep inside the network:
L depends on ŷ
ŷ depends on a²
a² depends on z²
z² depends on W²

Chain rule lets us "chain" these derivatives together!
Chain Rule Formula
If  L = f(a),  a = g(z),  z = h(w)

Then:  dL/dw = dL/da × da/dz × dz/dw

Like peeling an onion backwards — one layer at a time
Concrete Example
Forward:
  z = wx + b
  a = sigmoid(z)
  L = (a - y)²

Derivatives:
  dL/da = 2(a - y)                          ← loss derivative
  da/dz = sigmoid(z) × (1 - sigmoid(z))    ← sigmoid derivative
  dz/dw = x                                 ← linear derivative

Chain rule:
  dL/dw = dL/da × da/dz × dz/dw
        = 2(a-y) × sigmoid(z)(1-sigmoid(z)) × x

Weight update:
  w = w - lr × dL/dw  ✅

5️⃣ Backpropagation
What it is
Backpropagation = applying chain rule repeatedly from output layer back to input layer to compute gradients for every weight.
FORWARD PASS  →→→→→→→→→→→→→→→→→→→→→→→→→→→→→
X → z¹ → a¹ → z² → a² → z³ → ŷ → Loss

BACKWARD PASS ←←←←←←←←←←←←←←←←←←←←←←←←←←
dW¹ ← da¹ ← dz² ← da² ← dz³ ← dŷ ← dLoss
Full Training Loop
For each epoch:
  ┌─────────────────────────────────────────┐
  │  1. Forward Pass  → compute ŷ and Loss  │
  │  2. Backward Pass → compute all dL/dW   │
  │  3. Update Weights: W = W - lr × dL/dW  │
  └─────────────────────────────────────────┘
  Repeat until loss converges ✅
Why Backprop is Powerful
A network with 1 million weights needs 1 million gradients
Without backprop → compute each separately → impossibly slow
With backprop    → compute all at once efficiently → fast! ✅

6️⃣ Vanishing Gradient Problem
Sigmoid derivative:  max value = 0.25
After 5 layers:      0.25 × 0.25 × 0.25 × 0.25 × 0.25 = 0.001

Gradient shrinks to near zero!
Early layers get almost no signal → don't learn

┌─────────────────────────────────────────────┐
│ Solution: Use ReLU in hidden layers          │
│ ReLU derivative = 1 (when z > 0)            │
│ 1 × 1 × 1 × 1 × 1 = 1  → gradient stays!  │
└─────────────────────────────────────────────┘

7️⃣ Key Hyperparameters
Learning Rate (lr):
  Too high  → overshoots minimum → loss explodes
  Too low   → learns too slowly  → takes forever
  Just right → smooth convergence ✅ (try 0.01 or 0.001)

Epochs:
  Too few  → underfitting (hasn't learned enough)
  Too many → overfitting  (memorized training data)

Batch Size:
  Full batch  → slow but accurate gradients
  Mini batch  → fast, good gradients (most common: 32, 64, 128)
  Stochastic  → very fast, noisy gradients (batch size = 1)

Hidden Layer Size:
  Too small → can't learn complex patterns
  Too large → overfits, slow to train


🔑 10 Key Points
1.  Neural network = layers of neurons doing z=Wx+b then activation
2.  Activation functions add non-linearity (essential!)
3.  Without activations = just linear regression no matter how deep
4.  ReLU is best for hidden layers (fast, no vanishing gradient)
5.  Loss = how wrong predictions are (minimize this!)
6.  MSE for regression, Cross Entropy for classification
7.  Chain rule = derivative of composed functions
8.  Backprop = chain rule applied from output to input
9.  Vanishing gradient = sigmoid kills gradients in deep networks
10. Learning rate controls how big each weight update step is
  
