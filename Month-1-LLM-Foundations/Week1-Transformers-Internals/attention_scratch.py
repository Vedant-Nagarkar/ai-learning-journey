import numpy as np

# 1. SCALED DOT-PRODUCT ATTENTION 
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Formula:
    Attention(Q,K,V) = softmax(QKᵀ / √d_k) × V

    Args:
        Q: Query matrix  (seq_len, d_k)
        K: Key matrix    (seq_len, d_k)
        V: Value matrix  (seq_len, d_v)
        mask: optional mask for decoder (seq_len, seq_len)

    Returns:
        output: attended values (seq_len, d_v)
        weights: attention weights (seq_len, seq_len)
    """
    d_k = Q.shape[-1]   # dimension of key vectors

    #  Step 1: Compute attention scores 
    # QKᵀ = similarity between each query and key
    # Shape: (seq_len, seq_len)
    scores = np.matmul(Q, K.T)
    print(f"  Raw scores shape    : {scores.shape}")
    print(f"  Raw scores (first row): {scores[0].round(3)}")

    #  Step 2: Scale by √d_k 
    # Prevents large values that make softmax peaked
    scores = scores / np.sqrt(d_k)
    print(f"  Scaled scores (first row): {scores[0].round(3)}")

    #  Step 3: Apply mask (for decoder) 
    # Set future positions to -infinity
    # so softmax gives them 0 weight
    if mask is not None:
        scores = scores + (mask * -1e9)
        print(f"  Mask applied!")

    #  Step 4: Softmax → attention weights 
    # Each row sums to 1.0
    weights = softmax(scores)
    print(f"  Attention weights (first row): {weights[0].round(3)}")
    print(f"  Sum of weights (should be 1): {weights[0].sum():.4f}")

    #  Step 5: Weighted sum of Values 
    output = np.matmul(weights, V)
    print(f"  Output shape: {output.shape}")

    return output, weights


def softmax(x):
    """
    Compute softmax along last axis.
    Subtract max for numerical stability.

    Without stability trick:
    e^1000 = overflow!

    With trick:
    e^(1000-1000) = e^0 = 1 (safe!)
    """
    # subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


#  2. MULTI-HEAD ATTENTION 
class MultiHeadAttention:
    """
    Multi-Head Attention from scratch.

    Runs attention h times in parallel
    each with different learned projections.

    MultiHead(Q,K,V) = Concat(head1,...,headh) × Wo
    where headi = Attention(Q×Wiq, K×Wik, V×Wiv)
    """

    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model   : total embedding dimension (e.g. 512)
            num_heads : number of attention heads (e.g. 8)
        """
        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads!"

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads  # 512//8 = 64

        print(f"MultiHeadAttention initialized:")
        print(f"  d_model   = {d_model}")
        print(f"  num_heads = {num_heads}")
        print(f"  d_k       = {self.d_k} (per head)")

        #  Weight matrices 
        # Each head has its own projection matrices
        # Shape: (d_model, d_model)
        np.random.seed(42)
        scale = np.sqrt(2.0 / d_model)

        self.Wq = np.random.randn(d_model, d_model) * scale
        self.Wk = np.random.randn(d_model, d_model) * scale
        self.Wv = np.random.randn(d_model, d_model) * scale
        self.Wo = np.random.randn(d_model, d_model) * scale

    def split_heads(self, x):
        """
        Split embedding into num_heads pieces.

        Input : (seq_len, d_model)
        Output: (num_heads, seq_len, d_k)

        Example:
        d_model=512, num_heads=8, d_k=64
        (10, 512) → (8, 10, 64)
        """
        seq_len = x.shape[0]

        # Reshape: (seq_len, d_model) → (seq_len, num_heads, d_k)
        x = x.reshape(seq_len, self.num_heads, self.d_k)

        # Transpose: (seq_len, num_heads, d_k) → (num_heads, seq_len, d_k)
        x = x.transpose(1, 0, 2)

        return x

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass through multi-head attention.

        Args:
            Q, K, V: input matrices (seq_len, d_model)
            mask: optional mask

        Returns:
            output: (seq_len, d_model)
        """
        # Step 1: Linear projections 
        # Project Q, K, V with learned weight matrices
        Q_proj = np.matmul(Q, self.Wq)  # (seq_len, d_model)
        K_proj = np.matmul(K, self.Wk)
        V_proj = np.matmul(V, self.Wv)

        # Step 2: Split into heads 
        # (seq_len, d_model) → (num_heads, seq_len, d_k)
        Q_heads = self.split_heads(Q_proj)
        K_heads = self.split_heads(K_proj)
        V_heads = self.split_heads(V_proj)

        # Step 3: Attention per head 
        head_outputs = []
        for i in range(self.num_heads):
            output, weights = scaled_dot_product_attention(
                Q_heads[i], K_heads[i], V_heads[i], mask
            )
            head_outputs.append(output)

        # Step 4: Concatenate heads 
        # (num_heads, seq_len, d_k) → (seq_len, d_model)
        concat = np.concatenate(head_outputs, axis=-1)

        # Step 5: Final linear projection 
        output = np.matmul(concat, self.Wo)

        return output


# 3. POSITIONAL ENCODING 
def positional_encoding(seq_len, d_model):
    """
    Generate positional encoding for a sequence.

    Uses sin/cos functions:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len : length of sequence
        d_model : embedding dimension

    Returns:
        PE: positional encoding (seq_len, d_model)
    """
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # even dimensions → sine
            PE[pos, i] = np.sin(
                pos / (10000 ** (2*i / d_model))
            )
            # odd dimensions → cosine
            if i + 1 < d_model:
                PE[pos, i+1] = np.cos(
                    pos / (10000 ** (2*i / d_model))
                )

    return PE


# 4. DECODER MASK 
def create_causal_mask(seq_len):
    """
    Create causal (look-ahead) mask for decoder.

    Prevents decoder from attending to future tokens.

    For seq_len=4:
    [[0, 1, 1, 1],    ← position 0 can only see itself
    [0, 0, 1, 1],    ← position 1 can see 0,1
    [0, 0, 0, 1],    ← position 2 can see 0,1,2
    [0, 0, 0, 0]]    ← position 3 can see all

    1 = masked (set to -infinity in attention)
    0 = not masked (allowed to attend)

    Args:
        seq_len: length of sequence

    Returns:
        mask: (seq_len, seq_len)
    """
    # Upper triangular matrix of ones
    # then remove diagonal (k=1 means above diagonal)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask


# 5. DEMO 
if __name__ == "__main__":

    print("=" * 55)
    print("  ATTENTION MECHANISM FROM SCRATCH")
    print("=" * 55)

    # ── Hyperparameters ───────────────────────
    seq_len = 6      # 6 words in sentence
    d_model = 512    # embedding dimension
    d_k     = 64     # key/query dimension
    d_v     = 64     # value dimension

    np.random.seed(42)

    # ── Simulate word embeddings ──────────────
    # In real transformer these come from
    # an embedding lookup table
    # Here we just use random vectors
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    print(f"\nSentence: {sentence}")
    print(f"seq_len : {seq_len}")
    print(f"d_model : {d_model}")

    # Random embeddings (seq_len, d_model)
    embeddings = np.random.randn(seq_len, d_model)

    # ── Add positional encoding ───────────────
    PE = positional_encoding(seq_len, d_model)
    x  = embeddings + PE   # add position info!
    print(f"\nEmbeddings + PE shape: {x.shape}")

    # ── Single head attention ─────────────────
    print("\n" + "─"*55)
    print("SINGLE HEAD ATTENTION:")
    print("─"*55)

    # Simple weight matrices for single head
    Wq = np.random.randn(d_model, d_k) * 0.1
    Wk = np.random.randn(d_model, d_k) * 0.1
    Wv = np.random.randn(d_model, d_v) * 0.1

    Q = np.matmul(x, Wq)  # (seq_len, d_k)
    K = np.matmul(x, Wk)
    V = np.matmul(x, Wv)

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")

    output, weights = scaled_dot_product_attention(Q, K, V)

    print(f"\nAttention output shape: {output.shape}")
    print(f"\nAttention Weight Matrix:")
    print(f"(rows=queries, cols=keys)")
    print(weights.round(3))

    # ── Multi-head attention ──────────────────
    print("\n" + "─"*55)
    print("MULTI-HEAD ATTENTION:")
    print("─"*55)

    mha    = MultiHeadAttention(d_model=512, num_heads=8)
    output = mha.forward(x, x, x)  # self-attention: Q=K=V=x
    print(f"\nMulti-head output shape: {output.shape}")
    print(f"(should be {seq_len} x {d_model})")

    # ── Causal mask (decoder) ─────────────────
    print("\n" + "─"*55)
    print("CAUSAL MASK (for decoder):")
    print("─"*55)

    mask = create_causal_mask(seq_len)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask:\n{mask.astype(int)}")
    print("(1 = masked/blocked, 0 = allowed)")

    # Masked attention
    output_masked, weights_masked = \
        scaled_dot_product_attention(Q, K, V, mask)
    print(f"\nMasked attention weights (first row):")
    print(weights_masked[0].round(3))
    print("(future tokens should have ~0 weight)")

    # ── Positional encoding visualization ─────
    print("\n" + "─"*55)
    print("POSITIONAL ENCODING:")
    print("─"*55)

    PE_small = positional_encoding(seq_len, 8)
    print(f"PE shape: {PE_small.shape}")
    print(f"PE (first 3 positions, 8 dims):")
    print(PE_small[:3].round(3))
    print("(each position has unique encoding)")

    print("\n" + "="*55)
    print("ATTENTION FROM SCRATCH COMPLETE!")
    print("="*55)
