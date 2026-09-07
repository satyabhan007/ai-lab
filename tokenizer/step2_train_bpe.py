"""
tokenizer/step2_train_bpe.py
============================
LESSON 2: TRAIN the tokenizer - watch it discover common patterns.

The BPE training loop (dead simple, no neural network involved):
  1. COUNT every ADJACENT pair of tokens in the text
  2. Take the MOST COMMON pair and GLUE it into a new token
  3. Repeat until the vocabulary reaches the size we want

That's it. Counting + gluing. Watch it find 'e ', ' t', 'he ', ' the '...
then whole words like 'cat', 'learn', 'build'.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from bpe import BPETokenizer

# A small, repetitive training corpus (patterns appear often on purpose)
TRAIN_TEXT = (
    "the cat sat on the mat. the cat saw the dog. "
    "the dog ran to the cat. the cat and the dog sat. "
    "we learn to build ai from scratch. we learn to code. "
    "we build to learn. the more we build, the more we learn. "
) * 20

print("=" * 60)
print("STEP 2: Training the BPE tokenizer")
print("=" * 60)

n_bytes = len(TRAIN_TEXT.encode('utf-8'))
print(f"\nTraining text: {len(TRAIN_TEXT)} characters = {n_bytes} raw bytes")
print(f"Goal: vocabulary of 300 tokens (256 bytes + 44 learned merges)\n")

tok = BPETokenizer()
ids = tok.train(TRAIN_TEXT, vocab_size=300, verbose=True)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"\nBefore training: {n_bytes} tokens (one per raw byte)")
print(f"After training:  {len(ids)} tokens")
ratio = n_bytes / len(ids)
print(f"Compression: {ratio:.2f}x  (each token now carries ~{ratio:.1f} bytes of meaning)")

print("\nSome learned tokens (id -> what it means):")
shown = 0
for idx in sorted(tok.vocab):
    if idx >= 256 and shown < 12:
        print(f"  token {idx} = {tok.vocab[idx]!r}")
        shown += 1

print("\nNotice what the tokenizer discovered on its own:")
print("  - 'e ', ' t', 'he ', ' the '  (the most common English patterns,")
print("    spaces included!)")
print("  - 'at', 'cat', 'learn', 'build', 'we '  (whole words and chunks)")
print("  - NOBODY told it about words. It just counted and glued.")
print("\nFun detail: the LAST merges are giant phrases like")
print("  ' sat on the mat. the cat saw the dog. the '")
print("  because our corpus is tiny and repetitive. Real tokenizers train")
print("  on BILLIONS of characters so they learn reusable pieces, not")
print("  memorized sentences.")
