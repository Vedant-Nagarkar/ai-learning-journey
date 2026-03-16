1️⃣ Why RNNs? The Problem with Regular NNs for Sequences
Regular NN: processes each input independently
→ No memory of previous inputs
→ Can't understand context or order

Examples where ORDER matters:
→ "The cat sat on the mat" ← words depend on each other
→ Stock prices over time  ← today depends on yesterday
→ Audio signal            ← each sample depends on previous

RNN solution: maintains a hidden state
→ Passes information from one step to the next
→ Has "memory" of previous inputs ✅

2️⃣ RNN — Recurrent Neural Network
Core Idea
At each time step t:
  hₜ = tanh(Wₓ·xₜ + Wₕ·hₜ₋₁ + b)

  xₜ   = current input
  hₜ₋₁ = previous hidden state (memory!)
  hₜ   = new hidden state
  Wₓ   = input weights (same at every step)
  Wₕ   = hidden weights (same at every step)
Unrolled Through Time
x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → output
       ↑              ↑              ↑
   same weights   same weights   same weights
Weight Sharing in RNNs
Same Wₓ and Wₕ used at EVERY time step
→ Can handle sequences of any length
→ Far fewer parameters than regular NN
→ Learns patterns that repeat across time
Types of RNN Tasks
One to One:    x → y              (regular NN)
One to Many:   x → y₁y₂y₃        (image captioning)
Many to One:   x₁x₂x₃ → y        (sentiment analysis)
Many to Many:  x₁x₂x₃ → y₁y₂y₃  (machine translation)
RNN Limitations
❌ Vanishing gradient over long sequences
   (gradient shrinks as it travels back through many time steps)
❌ Forgets information from far back in sequence
❌ Hard to train on very long sequences
❌ Sequential computation → can't parallelize

3️⃣ LSTM — Long Short-Term Memory
Why LSTM?
RNN struggles to remember information from 100 steps ago
LSTM specifically designed to remember long-term dependencies

Key innovation: CELL STATE (Cₜ)
→ Like a conveyor belt running through the sequence
→ Information can flow unchanged across many steps
→ Gates control what to add/remove from cell state
The Three Gates
┌─────────────────────────────────────────────────────┐
│                    LSTM CELL                         │
│                                                      │
│  Forget Gate:  fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)         │
│  → What to FORGET from cell state (0=forget, 1=keep)│
│                                                      │
│  Input Gate:   iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)         │
│  → What new info to ADD to cell state               │
│                                                      │
│  Output Gate:  oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)         │
│  → What to OUTPUT from cell state                   │
└─────────────────────────────────────────────────────┘
Cell State Update
Step 1: Forget  →  Cₜ = fₜ × Cₜ₋₁
Step 2: Add     →  Cₜ = Cₜ + iₜ × tanh(Wc·[hₜ₋₁,xₜ] + bc)
Step 3: Output  →  hₜ = oₜ × tanh(Cₜ)
Intuition with Example
Reading: "I grew up in France... I speak fluent ____"

Forget gate: forgets irrelevant info (weather, lunch)
Input gate:  remembers "France" when first mentioned
Output gate: outputs "French" when blank needs filling

Regular RNN: forgets "France" by the time it reaches "____"
LSTM: remembers "France" all the way through! ✅

4️⃣ GRU — Gated Recurrent Unit
What it is
Simplified version of LSTM with only 2 gates:
Reset Gate:  rₜ = σ(Wr·[hₜ₋₁, xₜ])
Update Gate: zₜ = σ(Wz·[hₜ₋₁, xₜ])

hₜ = (1-zₜ)×hₜ₋₁ + zₜ×tanh(Wh·[rₜ×hₜ₋₁, xₜ])
LSTM vs GRU
┌──────────────┬──────────┬────────────────────────────┐
│              │  LSTM    │  GRU                        │
├──────────────┼──────────┼────────────────────────────┤
│ Gates        │  3       │  2 (simpler)                │
│ Cell state   │  Yes     │  No (merged with hidden)    │
│ Parameters   │  More    │  Fewer (faster)             │
│ Performance  │  Better  │  Similar on most tasks      │
│ When to use  │  Long    │  Shorter sequences          │
│              │  sequences│  or limited compute        │
└──────────────┴──────────┴────────────────────────────┘

5️⃣ Backpropagation Through Time (BPTT)
Regular backprop: goes backwards through layers
BPTT: goes backwards through TIME STEPS

Forward:
t=1 → t=2 → t=3 → t=4 → Loss

Backward:
dL/dW flows back: t=4 → t=3 → t=2 → t=1

Problem: gradient multiplied at EACH time step
→ 100 time steps → gradient multiplied 100 times
→ Vanishing gradient gets very bad!

LSTM solution: cell state highway
→ Gradient flows directly through cell state
→ Avoids repeated multiplication ✅

6️⃣ Sequence Tasks We'll Build
Sentiment Analysis (Many to One)
Input:  "This movie was absolutely amazing!" → sequence of words
Output: Positive (1) or Negative (0)

Architecture:
  Embedding → LSTM → Dense → Sigmoid

Embedding layer:
  Converts words to dense vectors
  "amazing" → [0.2, 0.8, -0.3, 0.5, ...]
  Similar words have similar vectors
Text Generation (Many to Many)
Input:  "To be or not to be that is the"
Output: "question"  ← predicts next word/character

Architecture:
  Embedding → LSTM → LSTM → Dense → Softmax

7️⃣ Practical Tips for RNNs/LSTMs
Sequence length:
→ Truncate/pad all sequences to same length
→ Use masking to ignore padding tokens

Stacking LSTMs:
→ First LSTM: return_sequences=True
→ Last LSTM:  return_sequences=False

Bidirectional LSTM:
→ Processes sequence forward AND backward
→ Better context understanding
→ Use for: classification, NER (not generation)

Dropout in RNNs:
→ Use recurrent_dropout (not regular dropout)
→ Applied to recurrent connections

Gradient clipping:
→ Prevents exploding gradients
→ optimizer = Adam(clipnorm=1.0)

8️⃣ RNN → LSTM → Transformer Evolution
1990s: RNN
→ First sequence model
→ Vanishing gradient problem

1997: LSTM
→ Solved vanishing gradient
→ State of the art for 20 years

2014: GRU
→ Simplified LSTM
→ Faster, similar performance

2017: Transformer ← YOU'LL BUILD THIS IN MONTH 1!
→ No recurrence at all
→ Pure attention mechanism
→ Parallel computation
→ GPT, BERT, ChatGPT all use this!

🔑 10 Key Points
1.  RNNs process sequences by maintaining hidden state
2.  Same weights used at every time step (weight sharing)
3.  RNNs suffer from vanishing gradient over long sequences
4.  LSTM uses 3 gates to control information flow
5.  Cell state is the "memory highway" of LSTM
6.  Forget gate decides what to remove from memory
7.  Input gate decides what new info to store
8.  Output gate decides what to output
9.  GRU is simpler than LSTM, similar performance
10. Transformers replaced RNNs in 2017 (Month 1 topic!)
