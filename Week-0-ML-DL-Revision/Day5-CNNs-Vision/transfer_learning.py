import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

# ═══════════════════════════════════════════════
# TRANSFER LEARNING WITH RESNET50 ON CIFAR-10
# ═══════════════════════════════════════════════

print("="*55)
print("  TRANSFER LEARNING — ResNet50 on CIFAR-10")
print("="*55)

# ─── 1. LOAD CIFAR-10 ───────────────────────────
print("\nLoading CIFAR-10...")
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Train : {X_train.shape}")
print(f"Test  : {X_test.shape}")
print(f"Classes: {class_names}")

# Normalize
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

# Use subset for faster training
X_train_sub = X_train[:10000]
y_train_sub = y_train[:10000]
X_test_sub  = X_test[:2000]
y_test_sub  = y_test[:2000]

print(f"\nUsing subset: {X_train_sub.shape[0]} train, {X_test_sub.shape[0]} test")

# ─── 2. VISUALIZE SAMPLES ───────────────────────
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i])
    ax.set_title(class_names[y_train[i][0]])
    ax.axis('off')
plt.suptitle("Sample CIFAR-10 Images", fontsize=14)
plt.tight_layout()
plt.savefig("cifar_samples.png")
plt.show()
print("cifar_samples.png saved!")

# ─── 3. MODEL A — SIMPLE CNN FROM SCRATCH ───────
print("\n" + "="*55)
print("  MODEL A: Simple CNN from Scratch")
print("="*55)

def build_simple_cnn():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu',
                      padding='same', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

model_scratch = build_simple_cnn()
model_scratch.compile(
    optimizer = 'adam',
    loss      = 'sparse_categorical_crossentropy',
    metrics   = ['accuracy']
)
model_scratch.summary()

print("\nTraining CNN from scratch...")
history_scratch = model_scratch.fit(
    X_train_sub, y_train_sub,
    epochs           = 10,
    batch_size       = 64,
    validation_split = 0.1,
    verbose          = 1
)

scratch_acc = model_scratch.evaluate(
    X_test_sub, y_test_sub, verbose=0)[1]
print(f"\nScratch CNN Test Accuracy: {scratch_acc:.4f} ({scratch_acc*100:.2f}%)")

# ─── 4. MODEL B — TRANSFER LEARNING ────────────
print("\n" + "="*55)
print("  MODEL B: Transfer Learning with ResNet50")
print("="*55)

# Resize images to 224x224 (ResNet expects this)
X_train_resized = tf.image.resize(X_train_sub, (64, 64))
X_test_resized  = tf.image.resize(X_test_sub,  (64, 64))

# Load ResNet50 pretrained on ImageNet
base_model = ResNet50(
    weights     = 'imagenet',
    include_top = False,           # remove final FC layer
    input_shape = (64, 64, 3)
)

# Freeze pretrained weights
base_model.trainable = False
print(f"ResNet50 layers: {len(base_model.layers)}")
print(f"Trainable params: {base_model.count_params():,} (all frozen)")

# Add our own head
inputs = keras.Input(shape=(64, 64, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model_tl = keras.Model(inputs, outputs)
model_tl.compile(
    optimizer = 'adam',
    loss      = 'sparse_categorical_crossentropy',
    metrics   = ['accuracy']
)

print(f"\nTotal params    : {model_tl.count_params():,}")
print(f"Trainable params: ", end="")
print(f"{sum([tf.size(w).numpy() for w in model_tl.trainable_weights]):,}")

print("\nTraining Transfer Learning model...")
history_tl = model_tl.fit(
    X_train_resized, y_train_sub,
    epochs           = 10,
    batch_size       = 64,
    validation_split = 0.1,
    verbose          = 1
)

tl_acc = model_tl.evaluate(X_test_resized, y_test_sub, verbose=0)[1]
print(f"\nTransfer Learning Test Accuracy: {tl_acc:.4f} ({tl_acc*100:.2f}%)")

# ─── 5. COMPARE BOTH MODELS ─────────────────────
print("\n" + "="*55)
print("  FINAL COMPARISON")
print("="*55)
print(f"CNN from Scratch    : {scratch_acc*100:.2f}%")
print(f"Transfer Learning   : {tl_acc*100:.2f}%")
print(f"Improvement         : +{(tl_acc-scratch_acc)*100:.2f}%")

# ─── 6. PLOT COMPARISON ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_scratch.history['val_accuracy'],
             label='Scratch CNN', color='blue', linewidth=2)
axes[0].plot(history_tl.history['val_accuracy'],
             label='Transfer Learning', color='red', linewidth=2)
axes[0].set_title("Validation Accuracy Comparison")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(['CNN Scratch', 'Transfer Learning'],
            [scratch_acc, tl_acc],
            color=['blue', 'red'], alpha=0.7)
axes[1].set_title("Final Test Accuracy")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
for i, v in enumerate([scratch_acc, tl_acc]):
    axes[1].text(i, v + 0.01, f"{v*100:.1f}%", ha='center', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("transfer_learning_comparison.png")
plt.show()
print("transfer_learning_comparison.png saved!")
