# 🧪 The AI Lab — Visual Testers & Real-World Scenarios

Everything from phases 1–5 (autograd → nets → tokenizer → attention → matmul), made
**touchable**: a browser playground where you can *see* the concepts work, and a
terminal tester with **52 PASS/FAIL checks** across every layer.

---

## 1. The Visual Playground (open in a browser)

**File:** `playground.html` — double-click it, no server, no install, no libraries.

| Panel | What you play with | Real-world twin |
| ------- | -------------------- | ----------------- |
| ☂ **Umbrella Neuron** | Drag the 3 weather sliders; watch the color gauge + verdict change instantly | Any instant yes/no feature: spam flag, credit pre-check, "cancel subscription?" |
| 🏋 **Train a Network LIVE** | Click +1/+10/+100 steps; loss chart falls, accuracy table fills with ✅. Drag learning rate to 0.5 → watch it bounce (the shower knob!), drag to 0.005 → watch it crawl | Spam filters, price predictors — any training run, at human speed |
| 🔢 **Tokenizer Lab** | Train your own BPE on the editable corpus, then type anything: every token becomes a colored chip with its id + meaning; roundtrip check ✅/❌ shown live | Autocomplete, byte-fallback for any language/emoji |
| 👀 **Attention Visualizer** | The famous sentence: *"the animal didn't cross the street because it was too …"* — click TIRED or WIDE and watch REAL softmax attention weights re-route between `animal` and `street` | How transformers resolve "it", translate, and summarize — the core of GPT |
| 🌡 **Temperature Explorer** | The model's honest next-word guess is fixed; drag temperature 0.1→3.0 and watch the distribution sharpen into a laser or flatten into chaos, plus 10 live samples | The exact creativity dial behind every ChatGPT answer |
| 🧮 **Context Cost (O(n²))** | Drag context length 128→32k tokens; watch pairwise scores explode quadratically, with memory footprint, $-to-read, and why Flash Attention/KV caches exist | Why long context costs more than linear — pricing, kernels, caching |
| 🧲 **MatMul Visualizer** | Step through the triple loop `C[i][j] += A[i][k]·B[k][j]` one multiply at a time and watch a cell of C fill in | The exact loop every GPU runs billions of times per second |
| 💰 **Cost Calculator** | The same token count → API cost at GPT-4o-mini rates ($0.15 / 1M tokens) | How ChatGPT actually bills you: per TOKEN, not per word |
| 🌍 **Real-World Map** | Cards linking each product you use daily → the exact concept + panel that powers it | Spam filter, Zillow pricing, ChatGPT billing, autocomplete, multilingual BPE |

The playground contains faithful JavaScript ports of YOUR Python code — same 4 backward rules
(add, mul swap, pow, tanh), same `+=` accumulation, same topo-sort backward, same BPE
count-and-glue, same softmax-with-max-stability and 1/√d scaling. Nothing hidden.

## 2. The Scenarios Tester (terminal, CI-friendly)

**File:** `scenarios_tester.py` — run `python scenarios_tester.py` from this folder.

Two parts — **layer-by-layer unit tests** (is every building block correct?) and
**real-world scenarios** (do the blocks solve actual problems?) — 52 checks total:

```
  PART 1 · LAYER-BY-LAYER UNIT TESTS
  ⚙ engine ops         6/6   every op's gradient vs numerical differentiation
  🧩 net anatomy       4/4   Neuron/Layer/MLP structure + the 41-parameter count
  🏋 mlp learns        2/2   fresh weights → 100% accuracy by backprop
  🔢 tokenizer         4/4   BPE roundtrips, compression, multi-byte merges
  👀 attention         6/6   softmax, temperature, √d scaling, causal mask, PE
  🧮 matmul            6/6   naive/tiled/Strassen agree · n³ flops · matvec = attention

  PART 2 · REAL-WORLD SCENARIOS
  ☂ umbrella           4/4   frozen-weight neuron decides 4 weather cases
  📧 spam filter       2/2   trains on 6 mails → classifies 2 UNSEEN mails
  🏠 house price       2/2   MSE regression: predicts 90 m² within $30k
  💬 API billing       3/3   tokenizer costs 3 messages; emoji roundtrip exact
  🔬 autograd unit     1/1   engine gradient vs numerical: |diff| < 1e-3
  🌡 temperature       2/2   T=0.2 is deterministic, T=2.5 is exploratory
  👤 pronouns          2/2   attention resolves "it" → animal or street
  📈 attn cost         4/4   n tokens → n² pairwise scores; 4× tokens = 16× work
  🎭 causal mask       1/1   zero attention mass on the future, rows sum to 1
  🤖 mini transformer  3/3   tokens→PE→attention→next-word, generalizes UNSEEN

  TOTAL: 52/52  ──  ✅ ALL SYSTEMS GO
```

Exit code 0 when everything passes (1 on failure), so CI fails loudly if you break a layer.

## 3. What each scenario teaches

| Scenario | Concept from your studies | The lesson |
| ---------- | --------------------------- | ------------ |
| Umbrella | tanh neuron, frozen weights | Inference = one dot product + squash. Decisions are cheap; training built the opinions. |
| Spam filter | hinge loss, unseen-data testing | The REAL test is data the model never saw (generalization beats memorization) |
| House price | MSE, normalization | Regression = same loop, different loss. Normalizing inputs (÷40) makes training stable. |
| API billing | BPE tokens | Tokens = money. Compression (fewer tokens) literally lowers your bill. |
| Autograd unit | central difference | Every ML team sanity-checks engines this way: analytical vs numerical gradient. |
| Temperature | softmax(x/T) | T is the creativity dial: same logits, different courage. Low = safe, high = adventurous. |
| Pronouns | attention weights | "it" resolves by re-weighting context — the mechanism behind translation & summarization |
| Attn cost | O(n²) scores | Long context is quadratically expensive — the reason for Flash Attention + KV caches |
| Causal mask | -inf masking | GPT trains in parallel without peeking: future positions get exactly 0% |
| Mini transformer | end-to-end pipeline | Embeddings + positions + attention + linear head — every piece of GPT, at toy scale |

## 4. Study path with the lab

1. Open `playground.html` → drag the umbrella sliders until the verdict flips. Find the flip line.
2. Train the network live with lr=0.05 (smooth), then lr=0.5 (bounces — shower knob!), then 0.005 (crawls).
3. Type your own sentence in the tokenizer; watch `' the '`-style chips appear after training.
4. Click **…TIRED** then **…WIDE** in the Attention Visualizer — watch 'it' switch its
   attention from `animal` to `street`. That re-routing IS attention.
5. Drag temperature: find where 10 samples become identical (T≈0.1) and where rare
   words appear (T≥2). You've now *felt* the creativity dial.
6. Drag context length and read the O(n²) numbers — this single panel explains
   long-context pricing, Flash Attention and KV caches.
6b. Click **▶ step** in the MatMul Visualizer and trace C[0][0] = 1·9 + 2·6 + 3·3 = 42 by hand — that is the exact loop every GPU runs.
7. Run `python scenarios_tester.py` — read every PASS line and trace WHY it passed using
   `micrograd/BEGINNER_GUIDE.md` and `attention/BEGINNER_GUIDE.md`.
8. Modify & break: change a weight in Scenario 1 or a mask in the tester until a test
   FAILS — then explain the failure. That's what a real ML engineer does all day.

> **Remember:** a model is only "done" when its real-world scenario tests pass —
> exactly what these two environments let you practice, visually and automatically.

---

## 5. Running the Lab on GitHub (2 superpowers)

Both testers are already **hosted and automated for free** on GitHub:

### A. The visual playground is a live website (GitHub Pages)

| Where | URL |
|---|---|
| Landing page | https://satyabhan007.github.io/AI-ML/ |
| Interactive playground | https://satyabhan007.github.io/AI-ML/lab/playground.html |

How it works:

1. The workflow `.github/workflows/deploy-pages.yml` deploys the **entire repository
   root** (landing page + assets + lab) automatically on every push to `main`.
2. One-time setup (already done): **Settings → Pages → Build and deployment →
   Source: `GitHub Actions`**.
3. Every later push redeploys the site. Share the link — anyone can drag the
   umbrella sliders, resolve the pronoun with attention, and train your network,
   no install needed.

### B. The scenarios tester runs itself on every push (GitHub Actions)

`.github/workflows/lab-tests.yml` runs automatically on every push/PR:

- runs `lab/scenarios_tester.py` — all **52 layer + real-world checks**
- re-runs all 6 micrograd + tokenizer lessons **and** all 3 attention lessons as
  regression checks
- green ✅ = everything still passes; red ❌ = you broke something (and CI tells you what)

See results in the [Actions tab](https://github.com/satyabhan007/AI-ML/actions).
The root README already carries the badge:

```markdown
![AI Lab Tests](https://github.com/satyabhan007/AI-ML/actions/workflows/lab-tests.yml/badge.svg)
```

### Keeping GitHub in sync (auto-sync watcher)

The workspace folder (`account_rotator/`) is the source of truth. The background
watcher `watch_and_push.ps1` polls `micrograd/`, `tokenizer/`, `attention/`,
`lab/`, `assets/`, `.github/` and the root site files (`index.html`, `404.html`,
`.gitignore`, `README.md`) every 2 seconds. On any save it copies the changed file
into the local `AI-ML` clone, commits and pushes — which redeploys Pages and
re-runs CI automatically.

```powershell
# start the watcher (leave the window open while you work)
powershell -ExecutionPolicy Bypass -File "D:\test\account rotate\account_rotator\watch_and_push.ps1"
```

Manual push (if the watcher isn't running):

```powershell
cd "D:\test\account rotate\AI-ML"
git add -A
git commit -m "update AI-ML content"
git push origin main
```

> One-time setup recap: `gh repo create AI-ML --public` → push → enable
> **Settings → Pages → Source: GitHub Actions** once. Every later push is automatic.
