"""
tokenizer/step3_encode_decode.py
================================
LESSON 3: USE the trained tokenizer - encode, decode, and the roundtrip test.

  encode("the cat")  ->  [116, 258, 263, 32]   (numbers for the network)
  decode(ids)        ->  "the cat"             (exact original text back)

The GOLDEN RULE of tokenizers:  decode(encode(x)) == x   for ANY x.
We will prove it on seen text, unseen text, numbers, and emoji.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from bpe import BPETokenizer

TRAIN_TEXT = (
    "the cat sat on the mat. the cat saw the dog. "
    "the dog ran to the cat. the cat and the dog sat. "
    "we learn to build ai from scratch. we learn to code. "
    "we build to learn. the more we build, the more we learn. "
) * 20

print("=" * 60)
print("STEP 3: Encode / Decode with the trained tokenizer")
print("=" * 60)

tok = BPETokenizer()
tok.train(TRAIN_TEXT, vocab_size=300)

# ------------------------------------------------------------------
# 1) ENCODE a sentence from the training data
# ------------------------------------------------------------------
sentence = "the cat sat on the mat."
ids = tok.encode(sentence)
print(f"\nText:   {sentence!r}")
print(f"Tokens: {ids}")
print(f"Each id means: {[tok.vocab[i] for i in ids]}")
print(f"Length: {len(sentence.encode('utf-8'))} raw bytes -> {len(ids)} tokens "
      f"({len(sentence.encode('utf-8'))/len(ids):.2f}x compression)")

# ------------------------------------------------------------------
# 2) ENCODE something the tokenizer has NEVER seen
# ------------------------------------------------------------------
unseen = "I love pizza! The moon is cheese."
u_ids = tok.encode(unseen)
print(f"\nUNSEEN text: {unseen!r}")
print(f"Tokens: {u_ids}")
print(f"Length: {len(unseen.encode('utf-8'))} raw bytes -> {len(u_ids)} tokens "
      f"({len(unseen.encode('utf-8'))/len(u_ids):.2f}x compression)")
print("  -> unseen words can't use the fancy merges, so they stay longer.")
print("     (Real tokenizers trained on billions of pages know 'pizza'.)")

# ------------------------------------------------------------------
# 3) THE GOLDEN RULE: decode(encode(x)) == x  — always
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("THE ROUNDTRIP TEST: decode(encode(x)) must equal x")
print("=" * 60)
tests = [
    "hello",
    "the cat sat on the mat",
    "I love pizza! The moon is cheese.",
    "numbers 12345 and symbols !@#$%",
    "the cat sat on the mat. " * 5,
]
all_ok = True
for t in tests:
    ok = tok.decode(tok.encode(t)) == t
    all_ok = all_ok and ok
    print(f"  {'PASS' if ok else 'FAIL'}  {t[:45]!r}")

# emoji test (multi-byte UTF-8)
emoji_text = "great job! \U0001F600"
ok = tok.decode(tok.encode(emoji_text)) == emoji_text
all_ok = all_ok and ok
print(f"  {'PASS' if ok else 'FAIL'}  {emoji_text!r}  (4-byte emoji!)")

print(f"\nALL ROUNDTRIPS PASS: {all_ok}")
print("  -> byte-level BPE can NEVER fail on any text: worst case it")
print("     falls back to raw bytes, which always decode perfectly.")

# ------------------------------------------------------------------
# 4) How the big models do it
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("FROM OUR 300 TOKENS TO GPT-4'S ~100,000")
print("=" * 60)
print("""
  Ours:      vocab=300, trained on 1 repetitive paragraph
             -> 4.2x on training text, but only 1.1x on unseen text
             (the late merges memorized whole sentences - tiny corpus!)
  GPT-4:     vocab~100,000, trained on trillions of chars -> ~4x compression
             EVEN on text it has never seen (uses tiktoken; same algorithm,
             way more data, bigger vocab)

  Fun fact: in GPT tokenizers, ' the' (with a leading SPACE) is ONE token.
  Spaces are part of words! That's why ' the' and 'the' are different ids.

  NEXT PHASES (per the roadmap):
    Project 3: matrix multiplication kernel (what hardware actually runs)
    Project 4: the Transformer - tokens go IN, predictions come OUT
""")