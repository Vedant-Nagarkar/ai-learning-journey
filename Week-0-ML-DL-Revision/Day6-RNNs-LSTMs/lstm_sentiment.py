import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ═══════════════════════════════════════════════
# LSTM SENTIMENT ANALYSIS ON IMDB DATASET
# Task: Movie review → Positive or Negative
# ═══════════════════════════════════════════════

print("="*55)
print("  LSTM SENTIMENT ANALYSIS — IMDB Reviews")
print("="*55)
print(f"TensorFlow: {tf.__version__}")

# ─── 1. LOAD IMDB DATASET ───────────────────────
print("\nLoading IMDB dataset...")

VOCAB_SIZE  = 10000   # keep top 10k most common words
MAX_LEN     = 200     # truncate/pad reviews to 200 words

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

print(f"Train samples : {len(X_train)}")
print(f"Test samples  : {len(X_test)}")
print(f"Labels        : 0=Negative, 1=Positive")
print(f"Positive train: {sum(y_train==1)}")
print(f"Negative train: {sum(y_train==0)}")

# ─── 2. EXPLORE DATA ────────────────────────────
print("\n" + "="*55)
print("  DATA EXPLORATION")
print("="*55)

lengths = [len(x) for x in X_train]
print(f"Review length — Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.0f}")
print(f"\nSample review (raw integers):")
print(f"  {X_train[0][:20]}...")
print(f"  Label: {'Positive' if y_train[0]==1 else 'Negative'}")

# Decode a review back to words
word_index   = imdb.get_word_index()
reverse_index = {v+3: k for k, v in word_index.items()}
reverse_index[0] = '<PAD>'
reverse_index[1] = '<START>'
reverse_index[2] = '<UNK>'

def decode_review(encoded):
    return ' '.join([reverse_index.get(i, '?') for i in encoded])

print(f"\nDecoded review sample:")
print(f"  {decode_review(X_train[0])[:200]}...")

# ─── 3. PREPROCESS ──────────────────────────────
print("\n" + "="*55)
print("  PREPROCESSING")
print("="*55)

# Pad/truncate all sequences to MAX_LEN
X_train_pad = pad_sequences(X_train, maxlen=MAX_LEN,
                            padding='post', truncating='post')
X_test_pad  = pad_sequences(X_test,  maxlen=MAX_LEN,
                            padding='post', truncating='post')

print(f"After padding:")
print(f"X_train : {X_train_pad.shape}")
print(f"X_test  : {X_test_pad.shape}")
print(f"\nPadded review sample:")
print(f"  {X_train_pad[0][:20]}...")

# ─── 4. BUILD MODELS ────────────────────────────

# ── Model A: Simple RNN ───────────────────────
def build_simple_rnn():
    model = keras.Sequential([
        layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
        layers.SimpleRNN(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='SimpleRNN')
    return model

# ── Model B: LSTM ─────────────────────────────
def build_lstm():
    model = keras.Sequential([
        layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='LSTM')
    return model

# ── Model C: Bidirectional LSTM ───────────────
def build_bilstm():
    model = keras.Sequential([
        layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='BiLSTM')
    return model

# ─── 5. TRAIN ALL MODELS ────────────────────────
print("\n" + "="*55)
print("  TRAINING ALL 3 MODELS")
print("="*55)

models_config = [
    ("Simple RNN",          build_simple_rnn()),
    ("LSTM",                build_lstm()),
    ("Bidirectional LSTM",  build_bilstm()),
]

histories = {}
results   = {}

for name, model in models_config:
    print(f"\n{'─'*45}")
    print(f"  Training: {name}")
    print(f"{'─'*45}")

    model.compile(
        optimizer = 'adam',
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy']
    )

    model.summary()

    history = model.fit(
        X_train_pad, y_train,
        epochs           = 5,
        batch_size       = 128,
        validation_split = 0.1,
        verbose          = 1
    )

    loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
    histories[name] = history
    results[name]   = {"acc": acc, "loss": loss, "model": model}

    print(f"  ✅ {name} Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# ─── 6. COMPARE RESULTS ─────────────────────────
print("\n" + "="*55)
print("  FINAL COMPARISON")
print("="*55)
print(f"{'Model':<25} {'Test Acc':>10} {'Test Loss':>10}")
print("-"*45)
for name, result in results.items():
    print(f"{name:<25} {result['acc']:>10.4f} {result['loss']:>10.4f}")

best = max(results, key=lambda x: results[x]['acc'])
print(f"\n🏆 Best Model: {best}")

# ─── 7. PLOT TRAINING HISTORY ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['blue', 'green', 'red']

for (name, _), color in zip(models_config, colors):
    axes[0].plot(histories[name].history['val_accuracy'],
                label=name, color=color, linewidth=2)
    axes[1].plot(histories[name].history['val_loss'],
                label=name, color=color, linewidth=2)

axes[0].set_title("Validation Accuracy — RNN vs LSTM vs BiLSTM")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Validation Loss — RNN vs LSTM vs BiLSTM")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lstm_comparison.png")
plt.show()
print("\nlstm_comparison.png saved!")

# ─── 8. TEST ON CUSTOM REVIEWS ──────────────────
print("\n" + "="*55)
print("  CUSTOM REVIEW PREDICTIONS")
print("="*55)

best_model = results[best]['model']

# Sample reviews to test
custom_reviews = [
    "This movie was absolutely fantastic! I loved every moment of it.",
    "Terrible film. Waste of time and money. Worst movie ever made.",
    "It was okay, nothing special but not bad either.",
    "Outstanding performance by the cast. A masterpiece of cinema!",
    "I fell asleep halfway through. Completely boring and predictable."
]

# Encode custom reviews
def encode_review(text):
    words   = text.lower().split()
    encoded = [word_index.get(w, 2) + 3 for w in words]
    encoded = [min(i, VOCAB_SIZE-1) for i in encoded]
    return pad_sequences([encoded], maxlen=MAX_LEN,
                        padding='post', truncating='post')

print(f"Using best model: {best}\n")
for review in custom_reviews:
    encoded  = encode_review(review)
    score    = best_model.predict(encoded, verbose=0)[0][0]
    sentiment = "😊 POSITIVE" if score > 0.5 else "😞 NEGATIVE"
    print(f"Review  : {review[:60]}...")
    print(f"Score   : {score:.4f}  →  {sentiment}")
    print() 
