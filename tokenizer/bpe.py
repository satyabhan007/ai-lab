"""
tokenizer/bpe.py
================
Project 2 of the AI-engineering roadmap: a Byte-Pair Encoding (BPE) tokenizer,
built from scratch in ~100 lines. (Same design as Karpathy's minbpe.)

WHY THIS EXISTS
  Neural networks eat NUMBERS, not text. The tokenizer is the translator:

      text ──encode──▶ token ids ──▶ [ NEURAL NETWORK ] ──▶ ids ──decode──▶ text

  Every LLM (GPT, Claude, Gemini) starts and ends with exactly this step.

THE IDEA (LEGO analogy)
  Start with the 256 possible bytes (the smallest LEGO bricks — enough to
  build ANY text in ANY language). Then repeatedly:
      1. COUNT which two bricks sit next to each other most often
      2. GLUE them into one new, bigger brick
  Common patterns ("th", " the", "ing") become single bricks.
  Result: shorter sequences + bricks that carry meaning.

API
  tok = BPETokenizer()
  tok.train(text, vocab_size=300)   # learn the glue rules
  ids = tok.encode("hello")         # text    -> numbers
  txt = tok.decode(ids)             # numbers -> text  (always roundtrips!)
"""


def count_pairs(ids):
    """
    Count how often each ADJACENT pair occurs.

    >>> count_pairs([1, 2, 3, 1, 2])
    {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    Replace every occurrence of `pair` inside `ids` with the single id `idx`.

    >>> merge([1, 2, 3, 1, 2], (1, 2), 9)
    [9, 3, 9]
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2          # both elements became ONE token — skip them
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    def __init__(self):
        self.merges = {}    # (id, id) -> new id     the learned glue rules
        self.vocab = {}     # id -> bytes            the decode lookup table

    # ------------------------------------------------------------------
    # TRAIN — learn which pairs to glue (pure counting, no neural net!)
    # ------------------------------------------------------------------
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256, "vocab must at least cover the 256 bytes"
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}   # bytes 0..255

        ids = list(text.encode('utf-8'))                   # start: raw bytes
        num_merges = vocab_size - 256

        for i in range(num_merges):
            stats = count_pairs(ids)
            if not stats:                                   # nothing left to merge
                break
            pair = max(stats, key=stats.get)                # MOST common pair
            idx = 256 + i                                   # its new token id
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"  merge {i + 1:3d}: {pair} -> {idx}   "
                      f"{self.vocab[idx]!r}   (seen {stats[pair]}x)")
        return ids

    # ------------------------------------------------------------------
    # ENCODE — text -> ids (replay the learned merges, earliest first)
    # ------------------------------------------------------------------
    def encode(self, text):
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            stats = count_pairs(ids)
            # apply the merge that was learned EARLIEST (lowest new id) —
            # order matters: 'th' must exist before 'the' can form
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break                                       # no known pair left
            ids = merge(ids, pair, self.merges[pair])
        return ids

    # ------------------------------------------------------------------
    # DECODE — ids -> text (pure lookup; ALWAYS roundtrips)
    # ------------------------------------------------------------------
    def decode(self, ids):
        tokens = b''.join(self.vocab[i] for i in ids)
        return tokens.decode('utf-8', errors='replace')