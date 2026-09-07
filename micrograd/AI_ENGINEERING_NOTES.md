# AI Engineering Learning — Session Notes
> Date: 2026-09-06 | Status: In Progress

---

## Questions Asked This Session

---

### Q1 — Which AI Engineering project should I start with?

**Context:** Given a list of 50+ AI engineering projects (Autograd, Transformer, RAG, CUDA kernels, etc.), asked which to start with to learn AI implementation from basics to advanced.

**Answer — The recommended order (full path):**

#### Foundation Layer
1. **Autograd engine** (`Micrograd`) — understand backpropagation & gradients
2. **Tokenizer** (`BPE`) — understand how text becomes numbers
3. **Matrix Multiplication kernel** — understand what hardware executes

#### Core Models
4. **Transformer from scratch** (`Attention is All You Need`)
5. **Small Language Model (SLM)** — apply transformer end-to-end
6. **Softmax kernel optimization** — numerical stability
7. **Flash Attention kernel** (CUDA) — understand why it exists

#### Agent & Reasoning
8. **Chain of Thought Reasoner**
9. **ReAct Agent loop** — tool use + reasoning
10. **Function Calling router**
11. **Structured Output parser** (CFGs)

#### Memory & Retrieval
12. **Vector Database** (HNSW index)
13. **Embedding model**
14. **RAG pipeline**
15. **Graph RAG system**

#### Training & Optimization
16. **LoRA trainer**
17. **DPO loss function**
18. **RLHF pipeline** (PPO)
19. **Quantization library** (Int8/FP4)
20. **KV Cache paging system** (vLLM-style)

**TL;DR — 3-project starter pack:**
```
1. Autograd engine  →  understand learning
2. Transformer from scratch  →  understand models
3. ReAct Agent loop  →  understand agents
```

---

### Q2 — Where do I start learning the 3 foundational projects?

**Answer — Concrete resources & schedule:**

#### 1. Autograd Engine (Week 1–2)
- **Watch:** Andrej Karpathy — *"The spelled-out intro to neural networks and backpropagation: building micrograd"*
  - `youtube.com/watch?v=VMj-3S1tku0`
- **Read:** `github.com/karpathy/micrograd` — 150 lines, read every line
- **Core concept:** Chain rule, computational graphs, `.grad` on every node

#### 2. Transformer from Scratch (Week 3–6)
- **Watch:** Andrej Karpathy — *"Let's build GPT from scratch"*
  - `youtube.com/watch?v=kCc8FmEb1nY`
- **Paper:** *"Attention Is All You Need"* — Vaswani et al. 2017 (focus on Figure 1 + Section 3)
- **Read:** `github.com/karpathy/nanoGPT`
- **Core concept:** Self-attention, positional encoding, why transformers replaced RNNs

#### 3. ReAct Agent Loop (Week 7–8)
- **Read:** *"ReAct: Synergizing Reasoning and Acting in Language Models"* — Yao et al. 2022
  - `arxiv.org/abs/2210.03629` (just abstract + Figure 1)
- **Core concept:** Thought → Action → Observation loop
- **Core code pattern:**
  ```python
  while not done:
      thought = llm(prompt + history)
      action = parse_action(thought)
      observation = tools[action.name](action.input)
      history += f"Thought: {thought}\nObservation: {observation}\n"
  ```

**Weekly Schedule:**

| Week | Task | Time/day |
|------|------|----------|
| 1 | Watch Karpathy micrograd video | 2h |
| 2 | Rewrite micrograd from scratch (no peeking) | 2h |
| 3–4 | Watch Karpathy GPT video, code along | 2h |
| 5–6 | Train nanoGPT on a small text dataset | 1–2h |
| 7 | Read ReAct paper, build basic tool loop | 2h |
| 8 | Add 3 real tools (search, calculator, code runner) | 2h |

**Prerequisites:**
- Python (intermediate)
- NumPy — matrix ops (`@`, reshape, broadcasting)
- Basic calculus — just derivatives: `d/dx(x²) = 2x`

> **Golden Rule:** Build, don't just watch. After every video, close it and rewrite the code from a blank file.

---

### Q3 — Let's start micrograd from scratch

**Answer — Built a complete micrograd implementation:**

Files created in `micrograd/`:

| File | Purpose |
|------|---------|
| [`engine.py`](engine.py) | The autograd engine — `Value` class with forward + backward pass |
| [`nn.py`](nn.py) | `Neuron`, `Layer`, `MLP` built on top of engine |
| [`step1_just_values.py`](step1_just_values.py) | Gradients & chain rule with manual math verification |
| [`step2_one_neuron.py`](step2_one_neuron.py) | One neuron, gradient descent, full training loop |
| [`step3_full_network.py`](step3_full_network.py) | Full MLP → 100% accuracy on toy dataset |

**Results from running:**
```
Step 3 — MLP training on 4-sample classification dataset:
  Step  0:  loss=1.014010  accuracy=50%
  Step 20:  loss=0.219246  accuracy=100%
  Step 40:  loss=0.005319  accuracy=100%

Final Predictions:
  Sample 1: pred=+1.0201  target=+1  ✓
  Sample 2: pred=-1.7564  target=-1  ✓
  Sample 3: pred=+1.2068  target=+1  ✓
  Sample 4: pred=-1.0829  target=-1  ✓
```

**The core engine in ~20 lines (conceptually):**
```python
class Value:
    def __init__(self, data):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set()

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad  += out.grad * other.data  # chain rule
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def backward(self):
        # topological sort → seed grad=1.0 → call _backward() in reverse
        ...
```

**Homework before moving to Transformer:**
- [ ] Delete `__mul__` from `engine.py`. Rewrite it from memory.
- [ ] Modify step2 — get output to exactly `0.999`. Experiment with learning rate.
- [ ] Add a 5th sample to step3 with inputs `[0.0, 0.0, 0.0]` and label `+1`. Does it still converge?
- [ ] Draw the computation graph for `e = a*b + (a+b)` on paper. Trace the backward pass manually.

---

### Q4 — What is micrograd?

**Answer:**

Micrograd is a ~150-line implementation of **backpropagation** — the algorithm that makes neural networks learn.

**The problem it solves:**
> "If I increase weight `w1` by a tiny amount, does the loss go up or down, and by how much?"
> That's `d(loss)/d(w1)` — the gradient. Micrograd computes it automatically for every weight at once.

**How it works:**
1. Every number is wrapped in a `Value` object that remembers the operation that created it
2. Every operation records how to compute the gradient backwards (a `_backward` closure)
3. `.backward()` walks the chain in reverse — chain rule, automated

**The mountain analogy:**
> You're lost in fog in the mountains. You can't see the valley (minimum loss). But you can feel whether the ground slopes up or down. The slope = gradient. Walking downhill = gradient descent. Micrograd = the instrument that measures the slope.

**In one line:**
> Micrograd is the algorithm that lets a neural network learn — stripped to 150 lines so you can understand it completely. Every framework (PyTorch, JAX, TensorFlow) does this same thing, just at massive scale.

---

## Next Steps

- [ ] Complete micrograd homework exercises above
- [ ] Watch Karpathy's GPT video (Week 3)
- [ ] Build Transformer from scratch
- [ ] Build ReAct Agent loop

---

*Run micrograd lessons:*
```bash
cd micrograd
$env:PYTHONIOENCODING='utf-8'
python step1_just_values.py
python step2_one_neuron.py
python step3_full_network.py
```
