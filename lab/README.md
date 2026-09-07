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

Once this repo is pushed to GitHub, you get both testers **hosted and automated for free**:

### A. The visual playground becomes a live website (GitHub Pages)

1. Push the repo (commands below).
2. On GitHub: **Settings → Pages → Build and deployment → Source: `GitHub Actions`**.
3. The included workflow `.github/workflows/deploy-pages.yml` deploys `lab/` automatically
   on every push. Your playground goes live at:

   `https://<your-username>.github.io/<repo-name>/playground.html`

   Share that link — anyone can drag the umbrella sliders and train your network,
   no install needed.

### B. The scenarios tester runs itself on every push (GitHub Actions)

`.github/workflows/lab-tests.yml` runs automatically on every push/PR:

- runs `lab/scenarios_tester.py` (the 12 real-world tests)
- re-runs all 6 micrograd + tokenizer lessons as regression checks
- green ✅ = everything still passes; red ❌ = you broke something (and CI tells you what)

See results in the repo's **Actions** tab. Badge for your README:

```markdown
![AI Lab Tests](https://github.com/<your-username>/<repo-name>/actions/workflows/lab-tests.yml/badge.svg)
```

### Push it (first time)

```powershell
cd "d:\test\account rotate\account_rotator"
git init
git add .
git commit -m "AI Lab: micrograd + tokenizer + visual playground + scenario tests"

# Option 1 — GitHub CLI (easiest; creates the remote repo too):
gh repo create ai-lab --public --source=. --push

# Option 2 — manual: create an empty repo on github.com, then:
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

After the first push: enable Pages (step A2 above) once — every later `git push`
redeploys the site and re-runs the tests automatically.
