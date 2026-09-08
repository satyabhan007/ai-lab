# AI-ML — From Scratch AI Engineering Curriculum
[![GitHub Pages](https://img.shields.io/badge/Live%20Site-GitHub%20Pages-252932?logo=githubpages&labelColor=252932&color=3b82f6)](https://satyabhan007.github.io/AI-ML/)
[![AI Lab Tests](https://github.com/satyabhan007/AI-ML/actions/workflows/lab-tests.yml/badge.svg)](https://github.com/satyabhan007/AI-ML/actions/workflows/lab-tests.yml)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776ab?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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
├── attention/               # Phase 4: Attention & the Transformer Heart
│   ├── attention.py         # softmax · √d scaling · causal mask · positional encoding
│   ├── step1_similarity.py  # Dot product & cosine similarity: the "relevance meter"
│   ├── step2_self_attention.py # Queries/keys/values: pronouns resolved by attention
│   ├── step3_mini_transformer.py # Embeddings + PE + causal attention, trained end-to-end
│   └── BEGINNER_GUIDE.md    # Everyday analogies for attention, temperature & O(n²)
│
└── lab/                     # Interactive Playground & Real-World Scenarios
    ├── playground.html      # Browser GUI: 6 live labs (neuron, trainer, tokenizer, attention, temperature, O(n²))
    └── scenarios_tester.py  # 46 automated checks: layer-by-layer unit tests + 15 real-world scenarios
```

---

## ⚡ Quick Start

### 1. Run the Lab Tester (46/46 PASS)
Two kinds of checks — every **layer** unit-tested, then 15 real-world **scenarios**:
```bash
cd lab
python scenarios_tester.py
```
- **Part 1 · Layer-by-layer unit tests:** every engine op's gradient vs numerical differentiation · Neuron/Layer/MLP anatomy (41 params) · a fresh MLP trains to 100% · BPE roundtrips + byte fallback · softmax, temperature, √d scaling, causal mask, positional encoding
- **Part 2 · Real-world scenarios:** ☂ umbrella · 📧 spam on UNSEEN mail · 🏠 house prices · 💬 API billing · 🔬 autograd unit test · 🌡 temperature sampling · 👤 pronoun resolution · 📈 O(n²) cost · 🎭 causal-mask proof · 🤖 mini transformer trained end-to-end

### 2. Launch the Interactive Browser Lab
Open `lab/playground.html` in any browser (or double click it) — **6 live labs**, zero install:
- ☂ Umbrella neuron sliders · 🏋 Train-a-network LIVE (loss chart + accuracy table)
- 🔢 BPE tokenizer chips + 💰 API cost calculator
- 👀 Attention visualizer (pronoun resolution) · 🌡 Temperature explorer · 🧮 O(n²) context cost

### 3. Walk the Attention Chapter (Phase 4)
```bash
cd attention
python step1_similarity.py
python step2_self_attention.py
python step3_mini_transformer.py
```

### 4. Train the Classics
```bash
cd tokenizer && python step2_train_bpe.py && python step3_encode_decode.py
cd ..\micrograd && python step3_full_network.py
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
| [`attention/attention.py`](attention/attention.py) | Attention Core | Softmax, √d scaling, causal masks, positional encoding — pure Python |
| [`attention/BEGINNER_GUIDE.md`](attention/BEGINNER_GUIDE.md) | Transformers 101 | Everyday analogies for queries, keys, values — and the O(n²) bill |

---

## 🔄 Automatic Repository Sync
This repository is automatically kept in sync using a background watcher script that monitors changes to markdown, python, and lab files in the workspace and pushes them to GitHub.
