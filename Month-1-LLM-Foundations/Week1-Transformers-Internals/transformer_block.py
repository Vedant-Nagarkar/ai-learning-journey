

import numpy as np
from attention_scratch import (
    MultiHeadAttention,
    positional_encoding,
    create_causal_mask
)

# 1. LAYER NORMALIZATION 
class LayerNorm:
    """
    Layer Normalization.

    Normalizes across the feature dimension
    for each sample independently.

    Why LayerNorm instead of BatchNorm?
    BatchNorm: normalizes across BATCH dimension
               → depends on batch size
               → doesn't work well for sequences

    LayerNorm: normalizes across FEATURE dimension
               → independent of batch size
               → works perfectly for sequences ✅

    Formula:
    y = γ × (x - mean) / (std + ε) + β

    γ (gamma) = learnable scale
    β (beta)  = learnable shift
    ε (eps)   = small number to avoid division by 0
    """

    def __init__(self, d_model, eps=1e-6):
        """
        Args:
            d_model: feature dimension (e.g. 512)
            eps    : small constant for stability
        """
        self.eps   = eps
        self.gamma = np.ones(d_model)   # scale (init to 1)
        self.beta  = np.zeros(d_model)  # shift (init to 0)

    def forward(self, x):
        """
        Args:
            x: input (seq_len, d_model)

        Returns:
            normalized x (seq_len, d_model)
        """
        # Compute mean and std across feature dimension
        # axis=-1 means across d_model dimension
        mean = x.mean(axis=-1, keepdims=True)
        std  = x.std(axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / (std + self.eps)

        # Scale and shift with learned parameters
        return self.gamma * x_norm + self.beta


# 2. FEED-FORWARD NETWORK 
class FeedForward:
    """
    Position-wise Feed-Forward Network.

    Applied to each position INDEPENDENTLY.
    Same FFN applied to every word.

    Architecture:
    Input(512) → Linear(2048) → ReLU → Linear(512)

    Why 4× expansion (512 → 2048)?
    → Gives model more capacity to learn
    → Bottleneck design (compress, expand, compress)
    → Similar to how CNNs use bottleneck blocks

    While attention MIXES information between words
    FFN PROCESSES each word's representation
    They complement each other! ✅
    """

    def __init__(self, d_model, d_ff):
        """
        Args:
            d_model: input/output dimension (512)
            d_ff   : hidden dimension (2048)
        """
        self.d_model = d_model
        self.d_ff    = d_ff

        # Two linear layers
        # W1: (d_model, d_ff)   → expand
        # W2: (d_ff, d_model)   → compress back
        scale     = np.sqrt(2.0 / d_model)
        self.W1   = np.random.randn(d_model, d_ff)   * scale
        self.b1   = np.zeros(d_ff)
        self.W2   = np.random.randn(d_ff, d_model)   * scale
        self.b2   = np.zeros(d_model)

    def relu(self, x):
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)

    def forward(self, x):
        """
        Args:
            x: input (seq_len, d_model)

        Returns:
            output (seq_len, d_model)
        """
        # First linear layer + ReLU
        # (seq_len, d_model) × (d_model, d_ff)
        # = (seq_len, d_ff)
        hidden = self.relu(np.matmul(x, self.W1) + self.b1)

        # Second linear layer
        # (seq_len, d_ff) × (d_ff, d_model)
        # = (seq_len, d_model)
        output = np.matmul(hidden, self.W2) + self.b2

        return output


# 3. ENCODER BLOCK 
class EncoderBlock:
    """
    Single Encoder Block (one layer of the encoder).

    Structure:
    Input
      ↓
    Multi-Head Self-Attention
      ↓
    Add & Norm (residual connection)
      ↓
    Feed-Forward Network
      ↓
    Add & Norm (residual connection)
      ↓
    Output

    Key concepts:
    1. Self-attention: Q=K=V (all from same input)
       Every word attends to every other word

    2. Residual connection: output = LayerNorm(x + sublayer(x))
       Allows gradient to flow directly
       Network learns INCREMENTAL updates

    3. Applied N times (N=6 in original paper)
       Each layer builds more abstract understanding
    """

    def __init__(self, d_model, num_heads, d_ff):
        """
        Args:
            d_model  : embedding dimension (512)
            num_heads: number of attention heads (8)
            d_ff     : feed-forward hidden dim (2048)
        """
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn       = FeedForward(d_model, d_ff)
        self.norm1     = LayerNorm(d_model)
        self.norm2     = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x   : input (seq_len, d_model)
            mask: optional padding mask

        Returns:
            output (seq_len, d_model)
        """
        # ── Sub-layer 1: Self-Attention ────────
        # Self-attention: Q = K = V = x
        # Every word attends to every other word
        attn_output = self.attention.forward(x, x, x, mask)

        # ── Add & Norm 1 ──────────────────────
        # Residual connection + layer normalization
        # x + attn_output = skip connection
        # prevents vanishing gradient!
        x = self.norm1.forward(x + attn_output)

        # ── Sub-layer 2: Feed-Forward ──────────
        # Process each position independently
        ffn_output = self.ffn.forward(x)

        # ── Add & Norm 2 ──────────────────────
        x = self.norm2.forward(x + ffn_output)

        return x


# 4. DECODER BLOCK 
class DecoderBlock:
    """
    Single Decoder Block (one layer of the decoder).

    Structure:
    Input (target sequence)
      ↓
    Masked Multi-Head Self-Attention
      ↓
    Add & Norm
      ↓
    Cross-Attention (encoder-decoder attention)
      ↓
    Add & Norm
      ↓
    Feed-Forward Network
      ↓
    Add & Norm
      ↓
    Output

    3 key differences from encoder:
    1. Masked self-attention (can't see future)
    2. Cross-attention (attends to encoder output)
    3. Takes 2 inputs: target sequence + encoder output
    """

    def __init__(self, d_model, num_heads, d_ff):
        """
        Args:
            d_model  : embedding dimension
            num_heads: number of attention heads
            d_ff     : feed-forward hidden dim
        """
        # 3 attention layers!
        self.masked_attention = MultiHeadAttention(
            d_model, num_heads)    # masked self-attention
        self.cross_attention  = MultiHeadAttention(
            d_model, num_heads)    # encoder-decoder attention
        self.ffn   = FeedForward(d_model, d_ff)

        # 3 layer norms (one per sublayer)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None):
        """
        Args:
            x              : target sequence (seq_len, d_model)
            encoder_output : from encoder (src_len, d_model)
            src_mask       : causal mask for target

        Returns:
            output (seq_len, d_model)
        """
        seq_len = x.shape[0]

        # ── Sub-layer 1: Masked Self-Attention ─
        # Target attends to itself
        # But MASKED so future tokens are hidden!
        causal_mask = create_causal_mask(seq_len)

        masked_attn = self.masked_attention.forward(
            x, x, x,         # Q=K=V=x (self-attention)
            causal_mask       # mask future positions!
        )
        x = self.norm1.forward(x + masked_attn)

        # ── Sub-layer 2: Cross-Attention ───────
        # Q comes from DECODER (what do I need?)
        # K,V come from ENCODER (what did input say?)
        # This is how decoder "reads" the encoder!
        cross_attn = self.cross_attention.forward(
            x,               # Q = decoder state
            encoder_output,  # K = encoder output
            encoder_output   # V = encoder output
        )
        x = self.norm2.forward(x + cross_attn)

        # ── Sub-layer 3: Feed-Forward ──────────
        ffn_output = self.ffn.forward(x)
        x = self.norm3.forward(x + ffn_output)

        return x


# 5. FULL ENCODER 
class Encoder:
    """
    Full Encoder = N stacked EncoderBlocks

    Each block builds more abstract representation
    Block 1: learns basic patterns (word relationships)
    Block 2: learns higher level patterns
    ...
    Block 6: learns very abstract representations

    Like how CNNs stack layers:
    Layer 1: edges
    Layer 2: shapes
    Layer 3: objects
    """

    def __init__(self, d_model, num_heads, d_ff, N=6):
        """
        Args:
            d_model  : embedding dimension
            num_heads: number of attention heads
            d_ff     : feed-forward hidden dim
            N        : number of encoder blocks
        """
        self.N      = N
        self.layers = [
            EncoderBlock(d_model, num_heads, d_ff)
            for _ in range(N)
        ]

    def forward(self, x, mask=None):
        """
        Pass input through all N encoder blocks.

        Args:
            x   : input (seq_len, d_model)
            mask: optional mask

        Returns:
            output (seq_len, d_model)
        """
        for i, layer in enumerate(self.layers):
            x = layer.forward(x, mask)
            print(f"  Encoder block {i+1}/{self.N} "
                f"output shape: {x.shape}")
        return x


# 6. FULL DECODER 
class Decoder:
    """
    Full Decoder = N stacked DecoderBlocks

    Takes:
    1. Target sequence (what we've generated so far)
    2. Encoder output (understanding of input)

    Generates next token at each step!
    """

    def __init__(self, d_model, num_heads, d_ff, N=6):
        self.N      = N
        self.layers = [
            DecoderBlock(d_model, num_heads, d_ff)
            for _ in range(N)
        ]

    def forward(self, x, encoder_output, src_mask=None):
        """
        Args:
            x             : target (seq_len, d_model)
            encoder_output: from encoder
            src_mask      : optional mask

        Returns:
            output (seq_len, d_model)
        """
        for i, layer in enumerate(self.layers):
            x = layer.forward(x, encoder_output, src_mask)
            print(f"  Decoder block {i+1}/{self.N} "
                f"output shape: {x.shape}")
        return x


# 7. DEMO 
if __name__ == "__main__":

    print("=" * 55)
    print("  TRANSFORMER BLOCK FROM SCRATCH")
    print("=" * 55)

    # ── Hyperparameters ───────────────────────
    # These match the original paper!
    d_model   = 512    # embedding dimension
    num_heads = 8      # attention heads
    d_ff      = 2048   # feed-forward hidden dim
    N         = 2      # number of blocks (2 for speed)
    src_len   = 6      # source sequence length
    tgt_len   = 4      # target sequence length

    np.random.seed(42)

    # ── Create fake input sequences ───────────
    # Simulate word embeddings + positional encoding
    # Source: "The cat sat on the mat" (6 words)
    # Target: "Le chat assis" (4 words - French!)
    src = np.random.randn(src_len, d_model)
    tgt = np.random.randn(tgt_len, d_model)

    # Add positional encoding
    src = src + positional_encoding(src_len, d_model)
    tgt = tgt + positional_encoding(tgt_len, d_model)

    print(f"\nSource shape : {src.shape} "
        f"(English sentence)")
    print(f"Target shape : {tgt.shape} "
        f"(French sentence)")

    # ── Layer Normalization Test ───────────────
    print("\n" + "─"*55)
    print("LAYER NORMALIZATION:")
    print("─"*55)

    ln   = LayerNorm(d_model)
    test = np.random.randn(src_len, d_model) * 100
    norm = ln.forward(test)

    print(f"Before norm - mean: {test.mean():.2f}, "
        f"std: {test.std():.2f}")
    print(f"After norm  - mean: {norm.mean():.4f}, "
        f"std: {norm.std():.4f}")
    print("(mean should be ~0, std should be ~1)")

    # ── Feed-Forward Test ──────────────────────
    print("\n" + "─"*55)
    print("FEED-FORWARD NETWORK:")
    print("─"*55)

    ffn    = FeedForward(d_model=512, d_ff=2048)
    output = ffn.forward(src)
    print(f"FFN input  shape: {src.shape}")
    print(f"FFN output shape: {output.shape}")
    print("(shape preserved!)")

    # ── Single Encoder Block ───────────────────
    print("\n" + "─"*55)
    print("SINGLE ENCODER BLOCK:")
    print("─"*55)

    enc_block  = EncoderBlock(d_model, num_heads, d_ff)
    enc_output = enc_block.forward(src)
    print(f"Input  shape: {src.shape}")
    print(f"Output shape: {enc_output.shape}")
    print("(shape preserved!)")

    # ── Full Encoder ───────────────────────────
    print("\n" + "─"*55)
    print(f"FULL ENCODER ({N} blocks):")
    print("─"*55)

    encoder        = Encoder(d_model, num_heads, d_ff, N=N)
    encoder_output = encoder.forward(src)
    print(f"Final encoder output: {encoder_output.shape}")

    # ── Single Decoder Block ───────────────────
    print("\n" + "─"*55)
    print("SINGLE DECODER BLOCK:")
    print("─"*55)

    dec_block  = DecoderBlock(d_model, num_heads, d_ff)
    dec_output = dec_block.forward(tgt, encoder_output)
    print(f"Input  shape: {tgt.shape}")
    print(f"Output shape: {dec_output.shape}")

    # ── Full Decoder ───────────────────────────
    print("\n" + "─"*55)
    print(f"FULL DECODER ({N} blocks):")
    print("─"*55)

    decoder        = Decoder(d_model, num_heads, d_ff, N=N)
    decoder_output = decoder.forward(tgt, encoder_output)
    print(f"Final decoder output: {decoder_output.shape}")

    # ── Final Linear + Softmax ─────────────────
    print("\n" + "─"*55)
    print("FINAL OUTPUT LAYER:")
    print("─"*55)

    # vocab_size = 10 (small for demo)
    # In real GPT vocab_size = 50,257!
    vocab_size = 10
    W_out      = np.random.randn(d_model, vocab_size) \
                 * 0.01

    # Project to vocabulary size
    logits = np.matmul(decoder_output, W_out)
    print(f"Logits shape : {logits.shape}")

    # Softmax → probabilities over vocabulary
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1,
                                keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    probs = softmax(logits)
    print(f"Probs shape  : {probs.shape}")
    print(f"Probs sum    : {probs[0].sum():.4f} "
        f"(should be 1.0)")
    print(f"Predicted token at pos 0: "
        f"{np.argmax(probs[0])}")

    print("\n" + "="*55)
    print("TRANSFORMER COMPLETE!")
    print("="*55)
    print(f"\nFull pipeline:")
    print(f"Input  ({src_len}, {d_model})")
    print(f"  → Encoder × {N}")
    print(f"  → ({src_len}, {d_model})")
    print(f"  → Decoder × {N}")
    print(f"  → ({tgt_len}, {d_model})")
    print(f"  → Linear + Softmax")
    print(f"  → ({tgt_len}, {vocab_size})")
    print(f"  → Next token prediction! ✅")
