import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ═══════════════════════════════════════════════
# CNN FROM SCRATCH ON MNIST
# ═══════════════════════════════════════════════

print("="*55)
print("  CNN ON MNIST — TensorFlow/Keras")
print("="*55)
print(f"TensorFlow version: {tf.__version__}")

# ─── 1. LOAD & PREPARE DATA ─────────────────────
print("\nLoading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"Train shape : {X_train.shape}")
print(f"Test shape  : {X_test.shape}")
print(f"Labels      : {np.unique(y_train)}")

# Normalize 0-255 → 0-1
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

# Add channel dimension: (60000, 28, 28) → (60000, 28, 28, 1)
X_train = X_train[..., np.newaxis]
X_test  = X_test[...,  np.newaxis]

print(f"\nAfter reshape:")
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

# ─── 2. VISUALIZE SAMPLES ───────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].squeeze(), cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.suptitle("Sample MNIST Images", fontsize=14)
plt.tight_layout()
plt.savefig("mnist_samples.png")
plt.show()
print("mnist_samples.png saved!")

# ─── 3. BUILD CNN MODEL ─────────────────────────
print("\n" + "="*55)
print("  BUILDING CNN MODEL")
print("="*55)

def build_cnn():
    model = keras.Sequential([
        # ── Block 1 ───────────────────────────
        layers.Conv2D(32, (3,3), activation='relu',
                      padding='same', input_shape=(28,28,1)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # ── Block 2 ───────────────────────────
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # ── Classifier ────────────────────────
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = build_cnn()
model.summary()

# ─── 4. COMPILE ─────────────────────────────────
model.compile(
    optimizer = 'adam',
    loss      = 'sparse_categorical_crossentropy',
    metrics   = ['accuracy']
)

# ─── 5. TRAIN ───────────────────────────────────
print("\n" + "="*55)
print("  TRAINING")
print("="*55)

history = model.fit(
    X_train, y_train,
    epochs          = 10,
    batch_size      = 128,
    validation_split= 0.1,
    verbose         = 1
)

# ─── 6. EVALUATE ────────────────────────────────
print("\n" + "="*55)
print("  RESULTS")
print("="*55)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Loss     : {test_loss:.4f}")

# ─── 7. PLOT TRAINING HISTORY ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'],
             label='Train', color='blue', linewidth=2)
axes[0].plot(history.history['val_accuracy'],
             label='Val', color='orange', linewidth=2)
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'],
             label='Train', color='blue', linewidth=2)
axes[1].plot(history.history['val_loss'],
             label='Val', color='orange', linewidth=2)
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cnn_training.png")
plt.show()
print("cnn_training.png saved!")

# ─── 8. VISUALIZE PREDICTIONS ───────────────────
print("\n" + "="*55)
print("  SAMPLE PREDICTIONS")
print("="*55)

y_pred = np.argmax(model.predict(X_test[:25]), axis=1)

fig, axes = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].squeeze(), cmap='gray')
    color = 'green' if y_pred[i] == y_test[i] else 'red'
    ax.set_title(f"P:{y_pred[i]} T:{y_test[i]}", color=color)
    ax.axis('off')
plt.suptitle("Predictions (Green=Correct, Red=Wrong)", fontsize=14)
plt.tight_layout()
plt.savefig("cnn_predictions.png")
plt.show()
print("cnn_predictions.png saved!")

# ─── 9. VISUALIZE FILTERS ───────────────────────
print("\n" + "="*55)
print("  WHAT FILTERS LEARNED")
print("="*55)

filters = model.layers[0].get_weights()[0]
print(f"First conv layer filters shape: {filters.shape}")

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    if i < filters.shape[3]:
        ax.imshow(filters[:,:,0,i], cmap='viridis')
    ax.axis('off')
plt.suptitle("Learned Filters — Layer 1 (32 filters)", fontsize=14)
plt.tight_layout()
plt.savefig("cnn_filters.png")
plt.show()
print("cnn_filters.png saved!")  
