# Attention — The Beginner Guide
> Phase 4 · How GPT decides what to look at · Prerequisites: phases 1–3

---

## The one-sentence version

> **Attention lets every word look at every earlier word and ask
> "how relevant are you to what I mean?" — then rebuild its own meaning
> as a weighted average of the answers.**

That's it. GPT stacks this simple move 100 times and suddenly it writes poetry.

---

## 1 · The cocktail-party analogy 🍸

You're at a loud party, mid-conversation, and someone across the room says
your name. Instantly your brain *re-weights* the audio: the name gets 80%
of your attention, the background chatter drops to noise.

A token inside GPT does exactly the same:

| Party | Transformer |
|---|---|
| every sound around you | every token in the context |
| "is that relevant to me?" | `query · key` (dot product) |
| focus strength | softmax → weights summing to 100% |
| what you actually take away | `Σ weight · value` (the mix) |

Nobody programmed "listen for your name." The *weights* (what to attend to)
are **learned**. Same for GPT: nobody hard-codes that `"it"` should look at
`"animal"` — training discovered it.

## 2 · Words must become numbers first: embeddings

A computer can't dot-product the word "cat". So every token gets a **vector** —
a list of numbers, its coordinates in *meaning space*.

In our lab we craft 4–5 dimensions by hand (`animal-ness`, `size`, `pet-ness`…).
In real GPT the ~12,000 dimensions are **learned** during training, but the
property we rely on is identical:

> **similar meaning → similar direction** → big dot product → high attention.

This is why `king − man + woman ≈ queen` works: *relationships become
directions in meaning space* (see `step1_similarity.py`).

## 3 · The 4 moves of one attention head

```
scores = query · key / √d        ① every word scores every word
scores[future] = −∞              ② causal mask: no peeking ahead
weights = softmax(scores)        ③ scores → a 100% attention budget
context = Σ weightᵢ · valueᵢ     ④ new meaning = weighted average
```

### ① Score — the similarity meter
`query · key` is step 1's dot product. Dividing by **√d** is pure hygiene:
big vectors would give huge scores, softmax would saturate, gradients die.
(The "scaled" in *scaled dot-product attention*.)

### ② Mask — the one-way mirror 🎭
GPT is trained to predict the **next** word. If position *i* could see
position *i+1*, the task is trivial: copy the answer. So the mask sets all
future scores to −∞. After softmax that's **exactly 0%** — verifiably zero,
which our tester proves with `future_mass < 1e-12`.

Bonus: this is why GPT can train on *all* positions of a sequence in ONE
pass — every row of the attention table is a legal, independent
past-only lookup.

### ③ Softmax — the budget
Exponentiate and normalize: every word ends up with a percentage of the
attention budget, always summing to 100%. Negative scores are fine — they
just become small percentages.

### ④ Mix — the takeaway
The word's new representation is the weighted average of the values it
looked at. Look 23% at "animal", 22% at "didn't", … — your new embedding
*contains those meanings, in those proportions*. That mix is what feeds the
next layer (and, in GPT, the prediction of the next token).

## 4 · Temperature — the spice dial 🌡

Softmax has one knob, and you've already used it in ChatGPT's settings:

| temperature | effect on softmax | feels like |
|---|---|---|
| 0.2 | winner takes ~99% | deterministic, boring, safe |
| 1.0 | honest probabilities | balanced |
| 2.0+ | nearly flat | chaotic, "creative" |

Same logits, same attention — a different **sharpening**. Our temperature
scenario proves it: T=0.2 samples 1 distinct word in 30 draws, T=2.5
samples 3. (`lab/scenarios_tester.py` → scenario 11.)

## 5 · Positional encoding — why words need page numbers 📄

Attention is a **bag of vectors**: shuffle the words and every dot product
stays the same. "dog bites man" ≡ "man bites dog". Disaster.

Fix: **add** a position fingerprint to each embedding. The classic recipe
(Vaswani 2017) uses sine waves of different wavelengths — like a multi-hand
clock where fast hands track "next to" and slow hands track "chapter-length"
distance. Nearby positions → similar fingerprints (our tester checks
`cos(pe₀, pe₁) > cos(pe₀, pe₅₀)`).

## 6 · Why everyone says "O(n²)"

n tokens → every token scores every token → **n² scores**.

| context | scores needed |
|---|---|
| 1k tokens | 1,000,000 |
| 8k tokens | 67,000,000 |
| 128k tokens | 16,400,000,000 |

4× the context = **16×** the compute. This single fact explains long-context
pricing, Flash Attention, KV caching, and why your 200-page PDF upload costs
what it costs (scenario 13).

## 7 · What we did NOT build (yet)

Real transformers add: **learned** Q/K/V projections (weight matrices that
let each word ask different questions), **multi-head** attention (8–96
parallel perspectives), residuals & layernorm, feed-forward blocks, and
full next-token training. Our head uses embeddings directly as
queries/keys — deliberately, so the mechanism stays visible. You now know
every move; scale is "just" engineering (see `AI_ENGINEERING_NOTES.md` →
projects 4–7).

---

## Run the lessons

```bash
cd attention
python step1_similarity.py       # meaning as direction, the queen analogy
python step2_self_attention.py   # score → mask → softmax → mix, on a real puzzle
python step3_mini_transformer.py # train the pipeline end-to-end with micrograd
```

## Check yourself

- [ ] Explain why softmax, not "pick the max".
- [ ] What breaks if we delete the causal mask? (Hint: the loss becomes trivially 0.)
- [ ] Why √d and not d or nothing?
- [ ] Why does temperature change *creativity* but not the argmax?
- [ ] Your tokenizer merges "the cat" into one token. What happens to attention length?
