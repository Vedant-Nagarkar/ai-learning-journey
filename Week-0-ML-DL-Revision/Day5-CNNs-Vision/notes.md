1️⃣ Why CNNs? The Problem with Regular NNs for Images
A 28×28 grayscale image = 784 pixels
A 224×224 RGB image     = 224×224×3 = 150,528 pixels

Regular NN: every pixel connected to every neuron
→ 150,528 × 1000 neurons = 150 MILLION weights in one layer!
→ Too slow, too much memory, overfits easily

CNN solution:
→ Looks at small patches of image at a time
→ Shares weights across the image
→ Learns spatial patterns (edges, shapes, faces)
→ Far fewer parameters ✅

2️⃣ Key CNN Components
Convolution Layer — The Core Operation
Filter (kernel) slides across the image
At each position: element-wise multiply + sum = one output value

Input image:        3×3 Filter:      Output:
1 2 3 0            1 0 1            
0 1 2 3     ×      0 1 0     =   ?
1 0 1 2            1 0 1
2 1 0 1

Calculation at top-left:
(1×1)+(2×0)+(3×1)+(0×0)+(1×1)+(2×0)+(1×1)+(0×0)+(1×1) = 8
What Filters Learn
Early layers  → simple patterns
  Filter 1: detects horizontal edges  ──
  Filter 2: detects vertical edges    |
  Filter 3: detects diagonal edges    /

Middle layers → complex patterns
  Circles, corners, textures

Deep layers   → semantic patterns
  Eyes, wheels, faces, fur
Padding
Valid padding: no padding → output smaller than input
Same padding:  pad with zeros → output same size as input

Input: 5×5, Filter: 3×3
  Valid → output: 3×3  (shrinks!)
  Same  → output: 5×5  (stays same size)
Stride
How many pixels the filter moves each step

Stride=1 → filter moves 1 pixel at a time (default)
Stride=2 → filter moves 2 pixels → output is half the size

Use stride=2 instead of pooling to downsample
Output Size Formula
Output size = (Input - Filter + 2×Padding) / Stride + 1

Example:
  Input=28, Filter=3, Padding=1, Stride=1
  Output = (28 - 3 + 2×1) / 1 + 1 = 28 ✅ (same padding)

3️⃣ Pooling Layer
What it Does
Reduces spatial dimensions while keeping important information.
Max Pooling (2×2, stride=2):
  Takes MAXIMUM value in each 2×2 region

  Input:          Output:
  1 3 2 4         3 4
  5 6 1 2   →     6 3
  3 2 1 0         3 2
  1 2 3 2

Why maximum? Keeps the strongest activation (most important feature)
Why Pooling?
✅ Reduces computation (smaller feature maps)
✅ Adds translation invariance
   (feature detected even if slightly shifted)
✅ Reduces overfitting

4️⃣ CNN Architecture — Putting It Together
Input Image
    ↓
[Conv → ReLU] × N    ← learn features
    ↓
[Pooling]            ← reduce size
    ↓
[Conv → ReLU] × N    ← learn higher features
    ↓
[Pooling]            ← reduce size again
    ↓
[Flatten]            ← convert 3D → 1D
    ↓
[Fully Connected]    ← combine features
    ↓
[Softmax Output]     ← final prediction
Feature Map Size Through Network
Input:      28×28×1   (MNIST digit)
Conv1:      26×26×32  (32 filters, 3×3, valid)
Pool1:      13×13×32  (2×2 max pool)
Conv2:      11×11×64  (64 filters, 3×3, valid)
Pool2:       5×5×64   (2×2 max pool)
Flatten:    1600      (5×5×64)
FC:         128
Output:     10        (10 digit classes)

5️⃣ Famous CNN Architectures
LeNet-5 (1998):
→ First successful CNN
→ Used for handwritten digit recognition
→ Architecture: Conv→Pool→Conv→Pool→FC→FC→Output

AlexNet (2012):
→ Won ImageNet competition by huge margin
→ Introduced ReLU, Dropout, GPU training
→ Started the deep learning revolution!

VGG (2014):
→ Very deep network (16-19 layers)
→ Only uses 3×3 filters throughout
→ Simple but effective

ResNet (2015):
→ 152 layers deep!
→ Introduced skip connections (residual connections)
→ Solved vanishing gradient in very deep networks
→ Still widely used today
ResNet Skip Connections
Normal:   x → [Conv→ReLU→Conv] → output

ResNet:   x → [Conv→ReLU→Conv] → + x → output
              (learn residual)    ↑
                                  └── skip connection

If conv layers learn nothing useful → output = x (identity)
Network can always fall back to identity → safe to go very deep!

6️⃣ Transfer Learning
What it is
Use a model pretrained on millions of images and adapt it to your task.
Instead of training from scratch (weeks of compute):
→ Take ResNet pretrained on ImageNet (1.2M images, 1000 classes)
→ Remove final layer
→ Add your own output layer (e.g. 2 classes: cat vs dog)
→ Train only the new layer (or fine-tune all layers)

Result: great accuracy with very little data and training time! ✅
When to Use Transfer Learning
Your dataset is small    → freeze pretrained layers, train only head
Your dataset is medium   → fine-tune last few layers
Your dataset is large    → fine-tune entire network
Your domain is different → fine-tune more layers
Freeze vs Fine-tune
Freeze:    pretrained weights stay fixed, only new layers train
           → fast, works well with small datasets

Fine-tune: pretrained weights also updated (slowly)
           → slower, works better with more data

7️⃣ Data Augmentation
What it is
Artificially increase dataset size by applying transformations.
Original image → augmented versions:
  Flip horizontal  → mirror image
  Rotate ±15°      → slightly rotated
  Zoom in/out      → scaled version
  Brightness shift → lighter/darker
  Random crop      → different region

Each augmented image = new training example!
Why it Works
Model sees same image in many variations
→ Learns robust features (not memorizing exact pixels)
→ Reduces overfitting significantly
→ Essential when dataset is small

8️⃣ What We'll Build Today
Part 1: CNN from scratch with PyTorch on MNIST
  → Conv layers, pooling, forward pass
  → Train and evaluate

Part 2: Transfer Learning with ResNet18
  → Load pretrained ResNet18
  → Adapt for CIFAR-10 (10 classes)
  → Compare training from scratch vs transfer learning

🔑 10 Key Points
1.  CNNs use filters that slide across images → detect spatial patterns
2.  Filters learn automatically during training (not hand-crafted)
3.  Early layers = edges, deep layers = complex features
4.  Pooling reduces size and adds translation invariance
5.  Flatten converts 3D feature maps to 1D for FC layers
6.  ReLU after every conv layer (same reason as regular NNs)
7.  AlexNet (2012) started the deep learning revolution
8.  ResNet skip connections solve vanishing gradient in deep nets
9.  Transfer learning = pretrained model + your task's output layer
10. Data augmentation reduces overfitting with small datasets

