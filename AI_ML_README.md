# AI-ML — From Scratch AI Engineering Curriculum

> Hands-on, zero-black-box implementations of modern AI and Machine Learning fundamentals from scratch in pure Python.

---

## 🗺️ Curriculum & Repository Structure

```
AI-ML/
├── micrograd/               # Phase 1: Autograd Engine & Neural Networks
│   ├── engine.py            # Scalar autograd engine (Value, topological backprop)
│   ├── nn.py                # Neuron, Layer, MLP architectures
│   ├── explainer.py         # Step-by-step engine verification & visualization
│   ├── step1_just_values.py # Building blocks: forward math & computational graphs
│   ├── step2_one_neuron.py  # Weights, biases, activations, and backprop on 1 neuron
│   ├── step3_full_network.py# Multi-layer perceptron training on 2D classification
│   ├── homework4.py         # Advanced gradient checks & edge cases
│   ├── BEGINNER_GUIDE.md    # Intuitive explanation with everyday analogies
│   ├── DEEP_EXPLANATION.md  # Line-by-line breakdown of backpropagation math
│   ├── VISUAL_GUIDE.md      # Visual graphs and forward/backward walkthroughs
│   ├── MINDMAPS.md          # ASCII concept maps and mental models
│   ├── REPORT.md            # Comprehensive reference manual (1800+ lines)
│   └── AI_ENGINEERING_NOTES.md # Session notes, Q&A, and roadmap insights
│
├── tokenizer/               # Phase 2: Byte-Pair Encoding (BPE) Tokenizer
│   ├── bpe.py               # Minimal ~100-line BPE tokenizer (like Karpathy's minbpe)
│   ├── step1_why_tokenize.py# Why text must become numbers; byte vs vocab tradeoff
│   ├── step2_train_bpe.py   # Training loop: count adjacent pairs & iterative merge
│   ├── step3_encode_decode.py# Encoding text to token IDs and roundtrip decoding
│   └── BEGINNER_GUIDE.md    # The Amateur's Guide to Tokenizers with diagrams
│
└── lab/                     # Interactive Playground & Real-World Scenarios
    ├── playground.html      # Self-contained browser GUI: visual neuron & tokenizer
    └── scenarios_tester.py  # 5 end-to-end tests: umbrella, spam, housing, billing, autograd
```

---

## ⚡ Quick Start

### 1. Run the Real-World Scenarios Tester (100% PASS)
Tests everything built across Phase 1 and Phase 2 against 5 realistic scenarios:
```bash
cd lab
python scenarios_tester.py
```
- **Scenario 1 (☂ Umbrella decision):** Frozen neuron evaluating weather signals
- **Scenario 2 (📧 Spam filter):** MLP trained on emails classifying unseen messages
- **Scenario 3 (🏠 House price estimate):** Regression network predicting home values
- **Scenario 4 (💬 Chat API billing):** BPE token counts calculating LLM API costs
- **Scenario 5 (🔬 Autograd unit test):** Analytical gradients verified against numerical nudges ($\Delta < 10^{-10}$)

### 2. Launch the Interactive Browser Lab
Open `lab/playground.html` in any browser (or double click it) for:
- Live visual neuron with interactive weight/bias sliders and activation curves
- Live BPE tokenizer inspector: type text, see tokens highlighted, inspect compression ratios

### 3. Train the Tokenizer from Scratch
```bash
cd tokenizer
python step2_train_bpe.py
python step3_encode_decode.py
```

### 4. Train the Neural Network from Scratch
```bash
cd micrograd
python step3_full_network.py
```

---

## 📚 Explainer Documentation Index

| Guide | Topic | Description |
|---|---|---|
| [`micrograd/REPORT.md`](micrograd/REPORT.md) | Autograd & Backprop | Exhaustive 1800+ line reference manual covering engine internals |
| [`micrograd/BEGINNER_GUIDE.md`](micrograd/BEGINNER_GUIDE.md) | Neural Nets 101 | Plain English analogies, step-by-step calculations |
| [`micrograd/VISUAL_GUIDE.md`](micrograd/VISUAL_GUIDE.md) | Graph Walkthroughs | Visual diagrams of DAG backward passes and chain rule flow |
| [`micrograd/MINDMAPS.md`](micrograd/MINDMAPS.md) | Conceptual Maps | Memory aids and system maps for rapid revision |
| [`tokenizer/BEGINNER_GUIDE.md`](tokenizer/BEGINNER_GUIDE.md) | BPE Tokenization | How raw bytes turn into tokens, compression ratios, and UTF-8 handling |
| [`micrograd/AI_ENGINEERING_NOTES.md`](micrograd/AI_ENGINEERING_NOTES.md) | Roadmap | The journey from autograd to LLMs and AI Agents |

---

## 🔄 Automatic Repository Sync
This repository is automatically kept in sync using a background watcher script that monitors changes to markdown, python, and lab files in the workspace and pushes them to GitHub.
