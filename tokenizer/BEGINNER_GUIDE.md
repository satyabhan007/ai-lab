# The Amateur's Guide to Tokenizers (BPE)

### Phase 2 of the roadmap · text → numbers, explained from zero · with visuals

> **Files in this folder:** `bpe.py` (the engine, ~100 lines) · `step1_why_tokenize.py` ·
> `step2_train_bpe.py` · `step3_encode_decode.py` — run them in order, each explains itself.
> All numbers below are from actual runs. Difficulty: 🟢 basic · 🟡 intermediate · 🔴 advanced.

---

## 1. Why does a tokenizer exist? 🟢

A neural network is math. Math needs numbers. You type "hello" — the network must receive
`[104, 101, 108, 108, 111]`. The tokenizer is the **translator** at both ends:

```
  "the cat" ──encode──▶ [116, 258, 263, 32] ──▶ ┌──────────────┐
                                                │ THE NETWORK  │
  "the cat" ◀──decode── [116, 258, 263, 32] ◀── └──────────────┘
   (exact same text back!)
```

Every LLM you've ever used — GPT, Claude, Gemini — starts and ends with exactly this step.
No tokenizer, no LLM.

---

## 2. The three possible strategies (and why BPE wins) 🟢

| Strategy | Idea | Fatal flaw |
| ---------- | ------ | ------------ |
| **Per letter** | 26 tokens + symbols | Sequences super long; "q" alone means nothing |
| **Per word** | 1 token per word | Vocabulary explodes (millions of words); typos & new words = crash |
| **Per byte + BPE** ✅ | Start with 256 bytes, glue common pairs | …none. Handles ANY text, any language, any emoji |

```
   letters:   "the cat"  →  [t][h][e][ ][c][a][t]          7 tokens, no meaning
   bytes:     "the cat"  →  [116][104][101][32]...        7 numbers, no meaning
   BPE:       "the cat"  →  [116][258][263][32]           4 tokens WITH meaning
                              ▲      ▲     ▲
                           't'   'he '  'cat'   ← learned from data!
```

🟢 **LEGO analogy:** bytes are 1-stud bricks — you can build ANYTHING with them, but a castle
made of 1-stud bricks takes forever. BPE studies your city and glues the most common
combinations into bigger pre-made blocks. Building gets 4× faster, and each block *means
something*.

---

## 3. How BPE training works (count → glue → repeat) 🟢

No neural network. No calculus. Just counting:

```
   LOOP:
     1. COUNT every adjacent pair          "the cat" → (t,h)(h,e)(e,␣)(␣,c)...
     2. PICK the most frequent pair        (e,␣) appeared 340 times!
     3. GLUE it into a new token           'e'+'␣' → token 256 = 'e '
     4. REPEAT                             next: ' t', 'he ', ' the ', 'at', 'cat'...
```

🟡 **Real output from our training run** (corpus = 1 paragraph × 20, vocab 300):

```
  merge  1: (101, 32) -> 256   b'e '     (seen 340x)
  merge  2: (32, 116) -> 257   b' t'     (seen 279x)
  merge  3: (104, 256) -> 258  b'he '    (seen 200x)
  merge  4: (257, 258) -> 259  b' the '  (seen 199x)   ← a whole phrase, one token!
  merge  5: (97, 116) -> 260   b'at'     (seen 160x)
  ...
  merge 14: ... -> 269  b'learn'   (seen 80x)
  merge 23: ... -> 278  b'build'   (seen 60x)
```

Notice: **nobody told it about words.** It just counted and glued — and "discovered" English.

### The merge tree (how 'the ' was built) 🟡

```
              ' the '  (token 259)
             ╱         ╲
       ' t'(257)        'he '(258)
          │               ╱    ╲
       't'(116)     'h'(104)   'e '(256)
                               ╱     ╲
                           'e'(101)  '␣'(32)

  Encode replays this tree BOTTOM-UP (smallest merges first).
  Decode just looks up each id in a table. That's all.
```

---

## 4. Encode, decode, and the golden rule 🟡

```
  ENCODE = replay the merges (earliest-learned first) until none apply
  DECODE = pure table lookup:  id → its bytes → join → UTF-8 text
```

🟡 **Real results from our run:**

```
  "the cat sat on the mat."   (SEEN text)
    23 raw bytes  →  5 tokens  [116, 258, 263, 293, 46]   = 4.60× compression
    meaning:       't' + 'he ' + 'cat' + ' sat on the mat' + '.'

  "I love pizza! The moon is cheese."   (UNSEEN text)
    33 raw bytes  →  30 tokens                            = only 1.10× compression
    'pizza' was never in training → stays as boring byte-by-byte tokens
```

**The golden rule:** `decode(encode(x)) == x` — for ANY x. Our run proved it on plain text,
numbers, symbols, and a 4-byte emoji 😀 — **6/6 PASS**. Byte-level BPE can never crash on
unknown text: worst case it falls back to raw bytes, which always decode perfectly.

---

## 5. Compression = meaning (the deep insight) 🟡

```
  raw bytes:   ████████████████████████   24 tokens
  after BPE:   ██████████                 10 tokens      (2.4× shorter)

  shorter sequence = fewer network steps = cheaper, faster, smarter.
  A token like ' the ' carries 5 bytes of pattern in ONE number.
  The network doesn't have to re-learn "t+h+e go together" — the tokenizer
  already did that job.
```

🔴 **A warning from our own run:** the last merges became giant memorized phrases like
`' sat on the mat. the cat saw the dog. the '` — because our corpus is tiny and repetitive.
Real tokenizers train on **billions of characters** so they learn *reusable pieces*, not
memorized sentences. Data diversity = better tokens.

---

## 6. How the big models do it 🔴

```
  Ours:    vocab = 300        trained on 1 paragraph      4.2× on seen, 1.1× on unseen
  GPT-4:   vocab ≈ 100,000    trained on trillions of chars   ~4× even on UNSEEN text
           (tiktoken — the very same algorithm, just more data + bigger vocab)
```

Fun fact: in GPT tokenizers, `' the'` (with the leading **space**) is one token. Spaces are
part of words! That's why `' the'` and `'the'` are different ids.

---

## 7. Where this fits in your journey 🟢

```
   ✅ Project 1: micrograd      → how networks LEARN (gradients)
   ✅ Project 2: tokenizer      → how TEXT becomes numbers      ← you are here
   ⬜ Project 3: matmul kernel  → the math hardware actually runs
   ⬜ Project 4: Transformer    → tokens IN → next-token prediction OUT
                                            └── then the LLM is born
```

The tokenizer's output (token ids) is exactly what the future Transformer will eat. And when
the model generates text, it outputs token ids — and THIS tokenizer decodes them back.

---

## 8. Homework 🟢

1. **Bigger vocab:** train with `vocab_size=400`. Find the token id for `'cat'`. Did
   compression improve?
2. **Your own corpus:** replace `TRAIN_TEXT` with a paragraph you write (include your name).
   Retrain. Which merges surprise you?
3. **Break attempt:** try to find ANY text that fails `decode(encode(x)) == x`. (You won't —
   explain why byte-level BPE is unbreakable.)
4. **Count the savings:** take a paragraph from a news site. Compute raw bytes vs tokens with
   your tokenizer. Compare with the 4.24× we got.

> **One sentence to remember:** *A tokenizer is a learned dictionary that compresses text into
> meaningful numbers — built by nothing more than counting pairs and gluing the winners.*

<!-- auto-sync test: 22:17:45 -->
