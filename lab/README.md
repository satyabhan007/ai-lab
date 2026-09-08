# 🧪 The AI Lab — Visual Testers & Real-World Scenarios

Everything from phases 1–2 (micrograd + tokenizer), made **touchable**: two tester
environments where you can *see* the concepts work on real problems.

---

## 1. The Visual Playground (open in a browser)

**File:** `playground.html` — double-click it, no server, no install, no libraries.

| Panel | What you play with | Real-world twin |
| ------- | -------------------- | ----------------- |
| ☂ **Umbrella Neuron** | Drag the 3 weather sliders; watch the color gauge + verdict change instantly | Any instant yes/no feature: spam flag, credit pre-check, "cancel subscription?" |
| 🏋 **Train a Network LIVE** | Click +1/+10/+100 steps; loss chart falls, accuracy table fills with ✅. Drag learning rate to 0.5 → watch it bounce (the shower knob!), drag to 0.005 → watch it crawl | Spam filters, price predictors — any training run, at human speed |
| 🔢 **Tokenizer Lab** | Train your own BPE on the editable corpus, then type anything: every token becomes a colored chip with its id + meaning; roundtrip check ✅/❌ shown live | Autocomplete, byte-fallback for any language/emoji |
| 💰 **Cost Calculator** | The same token count → API cost at GPT-4o-mini rates ($0.15 / 1M tokens) | How ChatGPT actually bills you: per TOKEN, not per word |
| 🌍 **Real-World Map** | 6 cards linking each product you use daily → the exact concept + panel that powers it | Spam filter, Zillow pricing, ChatGPT billing, autocomplete, multilingual BPE |

The playground contains faithful JavaScript ports of YOUR Python code — same 4 backward rules
(add, mul swap, pow, tanh), same `+=` accumulation, same topo-sort backward, same BPE
count-and-glue. Nothing hidden.

## 2. The Scenarios Tester (terminal, CI-friendly)

**File:** `scenarios_tester.py` — run `python scenarios_tester.py` from this folder.

Five real-world situations, each with built-in PASS/FAIL checks and a final scoreboard:

```
  ☂ Umbrella        4/4   frozen-weight neuron decides 4 weather cases
  📧 Spam filter    2/2   trains on 6 mails → classifies 2 UNSEEN mails correctly
  🏠 House price    2/2   MSE regression: learns price = 50 + 2×size, predicts 90 m² within $30k
  💬 API billing    3/3   tokenizer costs 3 messages; roundtrip exact on emoji too
  🔬 Autograd unit  1/1   engine gradient vs numerical gradient: |diff| = 1.8e-11

  TOTAL: 12/12  ──  ✅ ALL SYSTEMS GO
```

Exit code 0 when everything passes (1 on failure), so you can wire it into CI later.

## 3. What each scenario teaches

| Scenario | Concept from your studies | The lesson |
| ---------- | --------------------------- | ------------ |
| Umbrella | tanh neuron, frozen weights | Inference = one dot product + squash. Decisions are cheap; training built the opinions. |
| Spam filter | hinge loss, unseen-data testing | The REAL test is data the model never saw (generalization beats memorization) |
| House price | MSE, normalization | Regression = same loop, different loss. Normalizing inputs (÷40) makes training stable. |
| API billing | BPE tokens | Tokens = money. Compression (fewer tokens) literally lowers your bill. |
| Autograd unit | central difference | Every ML team sanity-checks engines this way: analytical vs numerical gradient. |

## 4. Study path with the lab

1. Open `playground.html` → drag the umbrella sliders until the verdict flips. Find the flip line.
2. Train the network live with lr=0.05 (smooth), then lr=0.5 (bounces — shower knob!), then 0.005 (crawls).
3. Type your own sentence in the tokenizer; watch `' the '`-style chips appear after training.
4. Run `python scenarios_tester.py` — read every PASS line and trace WHY it passed using
   `micrograd/BEGINNER_GUIDE.md` sections referenced in each scenario.
5. Modify & break: change a weight in Scenario 1 until a test FAILS — then explain the failure.
   That's what a real ML engineer does all day.

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
   umbrella sliders and train your network, no install needed.

### B. The scenarios tester runs itself on every push (GitHub Actions)

`.github/workflows/lab-tests.yml` runs automatically on every push/PR:

- runs `lab/scenarios_tester.py` (the 12 real-world tests)
- re-runs all 6 micrograd + tokenizer lessons as regression checks
- green ✅ = everything still passes; red ❌ = you broke something (and CI tells you what)

See results in the [Actions tab](https://github.com/satyabhan007/AI-ML/actions).
The root README already carries the badge:

```markdown
![AI Lab Tests](https://github.com/satyabhan007/AI-ML/actions/workflows/lab-tests.yml/badge.svg)
```

### Keeping GitHub in sync (auto-sync watcher)

The workspace folder (`account_rotator/`) is the source of truth. The background
watcher `watch_and_push.ps1` polls `micrograd/`, `tokenizer/`, `lab/`, `assets/`,
`.github/` and the root site files (`index.html`, `404.html`, `.gitignore`) every
2 seconds. On any save it copies the changed file into the local `AI-ML` clone,
commits and pushes — which redeploys Pages and re-runs CI automatically.

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
