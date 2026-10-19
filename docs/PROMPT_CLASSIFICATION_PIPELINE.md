# LLMOptiFit — Prompt Classification & Model Router Pipeline

## 1. Executive Summary & Business Objective

In high-throughput Large Language Model (LLM) serving architectures, forwarding every user prompt to frontier models (e.g., Claude 3.5 Sonnet, GPT-4o) incurs unsustainable inference costs and latency penalties. Conversely, routing complex, multi-constraint reasoning tasks to lightweight models (e.g., LLaMA-3-8B, Nemotron-Nano) degrades response quality.

**LLMOptiFit** solves this optimization challenge by classifying human prompts across **4 core dimensions** before dispatching them to model endpoints:

| # | Dimension | Learning Paradigm | Winning Algorithm | Performance Metric | Production Latency |
|---|---|---|---|---|---|
| **1** | **Topic Discovery** | Unsupervised | **MiniBatch KMeans ($k=15$)** | Silhouette: `0.2265` | $<1\text{ ms}$ (Centroid) |
| **2** | **Task Type** | Supervised NLP | **Linear SVM (`SGDClassifier`)** | Macro-F1: `0.6650` | $3\text{ ms}$ / 4.7k batch |
| **3** | **Complexity & Tier** | Supervised Tabular | **Gradient Boosting** | Macro-F1: `0.9816` | $4\text{ ms}$ / 4.7k batch |
| **4** | **Prompt Specificity** *(New)* | Composite Rule-Based | **Orthogonal Ambiguity Index** | $0.0 \to 1.0$ continuous | $<1\text{ ms}$ (Formula) |

All steps have been implemented, benchmarked, and executed end-to-end inside [`prompt.ipynb`](../prompt.ipynb).

---

## 2. Exploratory Data Analysis (EDA) Highlights

Analysis performed on **23,613 English prompts** from the LMSYS Chatbot Arena (`data/arena/arena_prompts_english.parquet`):

```
       prompt_len_chars  prompt_len_words  prompt_len_sentences  avg_word_len
count          23613.00          23613.00              23613.00      23613.00
mean             202.94             35.09                  2.91          5.44
std              405.02             73.66                  4.10          1.37
min                1.00              1.00                  1.00          1.00
50% (Median)      99.00             18.00                  2.00          5.21
75%              210.00             45.00                  3.00          5.73
90%              471.00            111.00                  6.00          6.41
99%             2202.00            589.00                 20.00         10.50
max             4129.00           1240.00                128.00         48.00
```

### Key Statistical Takeaways:
1. **Extreme Right-Tail Length**: The median prompt is concise (18 words, 99 characters), but the 99th percentile spans 589 words (with maximums over 4,000 words).
2. **Structural Signals**:
   - Explicit Questions (`?`): **54.2%**
   - Code Blocks / Syntax Cues (`def`, `import`, `SELECT`, ```): **14.8%**
   - Numbered / Ordered Lists: **4.1%**
   - Bulleted Lists: **2.8%**
   - Structured Output Directives (`json`, `yaml`, `schema`): **5.3%**
   - Constraint Markers (`must`, `never`, `ensure`): Mean of 0.35 constraints per prompt, scaling up to 18 in complex queries.
3. **Arena Battle Preference**: Model B won 34.2%, Model A won 33.8%, 18.6% ties, and 13.4% both bad.

---

## 3. Feature Engineering Framework

The pipeline computes two orthogonal feature representations:

### A. 14-Parameter Semantic Rubric Features
Extracted using `PromptFeatureExtractor` (`tools/prompt_classifier.py`):
- `reasoning_complexity` (Weight: 0.20)
- `task_type_score` (Weight: 0.15)
- `decision_search_complexity` (Weight: 0.12)
- `accuracy_requirement` (Weight: 0.10)
- `instruction_complexity` (Weight: 0.09)
- `context_requirement` (Weight: 0.08)
- `output_complexity` (Weight: 0.07)
- `iteration_verification_need` (Weight: 0.06)
- `domain_difficulty` (Weight: 0.04)
- `tool_requirement` (Weight: 0.03)
- `freshness_requirement` (Weight: 0.02)
- `ambiguity` (Weight: 0.02)
- `latency_sensitivity` (Weight: 0.01)
- `cost_sensitivity` (Weight: 0.01)

*Correlation finding*: `reasoning_complexity` and `decision_search_complexity` exhibit the highest cross-correlation ($r = 0.74$).

### B. Structural Text Features (11 Dimensions)
- Length metrics: `prompt_len_chars`, `prompt_len_words`, `prompt_len_sentences`, `avg_word_len`
- Syntax & structure: `has_question`, `has_code_block`, `has_numbered_list`, `has_bullet_list`
- Logic density: `n_conditionals`, `n_constraints`, `has_json_request`

### C. Lexical TF-IDF Vectorization
- 8,000 features, unigram + bigram range, sublinear term-frequency scaling, English stopword removal.

---

## 4. Multi-Method Solution Benchmarking

### Pillar 1: Topic Discovery (Unsupervised)
Tested $k \in \{6, 8, 10, 12, 15\}$ on 50 TruncatedSVD components:
- **Agglomerative Clustering**: Captures tree hierarchies but suffers from $O(n^2)$ complexity and lacks centroids for single-prompt streaming inference.
- **MiniBatch KMeans**: Scalable, fast, produces cluster centroids for sub-millisecond production inference.
- **Winner**: **MiniBatch KMeans ($k=15$)**, Silhouette Score = **0.2265** (Agglomerative best: 0.2189).
- Discovered domains: Comparisons, Python/Software, Creative Writing, SQL/Database, Mathematics/Physics, Translation, and Business Analysis.

### Pillar 2: Task Type Classification (Supervised)
8 classes: `generation`, `coding`, `analysis`, `extraction`, `planning`, `mathematical_reasoning`, `research`, `classification`.

| Classifier | Macro-F1 | Weighted-F1 | Train Time | Inference Latency (4,723 test prompts) |
|---|---|---|---|---|
| **Linear SVM (`SGDClassifier`)** | **0.6650** | **0.8653** | **0.07s** | **3 ms** ($<0.001\text{ ms}$/prompt) |
| Logistic Regression (L-BFGS) | 0.4526 | 0.8174 | 0.39s | 3 ms |
| Random Forest (100 trees) | 0.2565 | 0.7584 | 0.18s | 52 ms |

**Why Linear SVM Won**: Maximum-margin hyperplanes combined with balanced class weights prevent rare classes (`extraction`, `planning`) from being overwhelmed by the majority `generation` class on sparse 8,000-dimensional vectors.

### Pillar 3: Complexity & Model Tier Routing (Supervised)
Target classes: `nano`, `super`, `ultra`. Evaluated on 25 tabular features.

| Router Model | Macro-F1 | Precision | Recall | Accuracy | Inference Time |
|---|---|---|---|---|---|
| **Gradient Boosting** | **0.9816** | **0.99** | **0.98** | **1.00** | **4 ms** |
| Random Forest (200 trees) | 0.9786 | 0.99 | 0.97 | 1.00 | 48 ms |
| Logistic Regression + Scaler | 0.9142 | 0.95 | 0.89 | 0.98 | 1 ms |

**Why Gradient Boosting Won**: Non-linear boundary learning isolates difficult prompts that require frontier reasoning without false-positive routing of easy prompts.
- **Top 3 Predictive Drivers**: `reasoning_complexity` (38%), `accuracy_requirement` (19%), `instruction_complexity` (14%).

---

## 5. Pillar 4: Prompt Specificity (New Dimension)

### Motivation
Complexity (Easy vs. Hard) does not capture Ambiguity (Vague vs. Specific):
- **Vague & Simple** (*"how do I fix my website?"*): Needs a strong frontier model to infer user intent and avoid generic answers.
- **Specific & Simple** (*"extract all dates from this text as a JSON array formatted as YYYY-MM-DD"*): Can be executed cheaply by a `nano` model.

### Formula ($0.0 \to 1.0$)
$$\text{Specificity} = 0.20 \cdot \min(1, \frac{W}{60}) + 0.20 \cdot \min(1, \frac{C}{4}) + 0.20 \cdot I_{\text{rubric}} + 0.15 \cdot J + 0.10 \cdot L + 0.15 \cdot (1 - A)$$

- **`vague`** ($<0.20$): 48.3%
- **`moderate`** ($0.20 - 0.40$): 40.8%
- **`specific`** ($0.40 - 0.65$): 10.2%
- **`highly_specific`** ($\ge 0.65$): 0.7%

---

## 6. Additional Production Router Dimensions

1. **Safety & Risk Level**: Diverts high-liability domains (medical, legal, financial, toxicity) to aligned models with system prompt guardrails.
2. **Response Modality & Tool Execution**: Prompts requesting images, charts, sandbox execution, or database access route to tool-augmented agents.

---

## 7. Nebius Token Factory Model Catalog & Pareto Recommendation Engine

### Live Catalog Integration
Connected directly to the official Nebius Token Factory live catalog ([`https://tokenfactory.nebius.com/models/catalog`](https://tokenfactory.nebius.com/models/catalog)) and its public endpoint ([`https://tokenfactory.nebius.com/api/public/models_info`](https://tokenfactory.nebius.com/api/public/models_info)). 

All model metadata, specifications, and live pricing are structured in [`config/nebius_catalog.json`](../config/nebius_catalog.json):

| Model Tier | Live Nebius Model ID | Vendor | Size | Input / Output per 1M | Tokens/sec | Context Window |
|---|---|---|---|---|---|---|
| **NANO** | `nvidia/Nemotron-3_5-Lightning` | NVIDIA | 30B | **$0.06 / $0.24** | 267 t/s | 64k |
| **NANO** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | NVIDIA | 30B | **$0.06 / $0.24** | 58 t/s | 128k |
| **NANO** | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Qwen | 30B | **$0.08 / $0.28** | 67 t/s | 128k |
| **SUPER** | `deepseek-ai/DeepSeek-V4-Flash-0731` | DeepSeek | 67B | **$0.14 / $0.28** | 236 t/s | 128k |
| **SUPER** | `nvidia/nemotron-3-super-120b-a12b` | NVIDIA | 120B | **$0.30 / $0.90** | 93 t/s | 128k |
| **SUPER** | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Qwen | 235B | **$0.20 / $0.60** | 75 t/s | 128k |
| **ULTRA** | `NousResearch/Hermes-4-405B` | NousResearch | 405B | **$1.00 / $3.00** | 48 t/s | 128k |
| **ULTRA** | `nvidia/Nemotron-3-Ultra-550b-a55b` | NVIDIA | 550B | **$1.00 / $3.00** | 332 t/s | 128k |
| **ULTRA** | `deepseek-ai/DeepSeek-V4-Pro` | DeepSeek | 671B | **$1.75 / $3.50** | 42 t/s | 128k |

### Pareto Frontier & Utility Optimization Formulation
The recommendation engine ([`tools/nebius_recommender.py`](../tools/nebius_recommender.py)) computes:

1. **Exact Token Costs**:
   $$\text{Cost}_{\text{USD}} = \frac{T_{\text{in}} \cdot P_{\text{in}} + T_{\text{out}} \cdot P_{\text{out}}}{1,000,000}$$
2. **Estimated Latency**:
   $$\text{Latency} = 0.20\text{s (TTFT)} + \frac{T_{\text{out}}}{\text{Tokens/sec}}$$
3. **Task & Tier Affinity Alignment ($Q(m)$)**:
   $$Q(m) = Q_{\text{base}} \pm 15 \cdot \Delta_{\text{tier}} + 7 \cdot \mathbb{I}[\text{task}] + 4 \cdot \mathbb{I}[\text{NVIDIA}]$$
4. **Pareto Dominance**:
   Candidate $A$ strictly dominates $B$ if $\text{Cost}(A) \le \text{Cost}(B)$ and $Q(A) \ge Q(B)$ (with at least one strict inequality).
5. **Utility Function**:
   $$U(m) = Q(m) - 10 \cdot (1 + \omega_{\text{cost}}) \cdot \text{Cost} \times 10^3 - 3 \cdot (1 + \omega_{\text{lat}}) \cdot \frac{\text{Latency}}{2}$$

---

## 8. Artifacts & Usage

1. **Interactive Jupyter Notebook**: [`prompt.ipynb`](../prompt.ipynb) — Fully executed with 28 code cells, visual figures, and Nebius live routing outputs.
2. **Model Catalog**: [`config/nebius_catalog.json`](../config/nebius_catalog.json) — 24 Nebius Token Factory live models with exact specifications and pricing.
3. **Recommender Module**: [`tools/nebius_recommender.py`](../tools/nebius_recommender.py) — Pareto frontier filtering, cost calculations, and utility ranking.
4. **Executable Demo Script**: [`examples/nebius_routing_demo.py`](../examples/nebius_routing_demo.py) — Demonstrates routing across simple, structured, coding, and frontier reasoning prompts.
5. **Live Classifier & Recommendation Usage**:
```python
from tools.nebius_recommender import recommend_for_prompt

# 1. Simple factual query -> Routes to NVIDIA Nemotron 3.5 Lightning (92% savings)
rec_easy = recommend_for_prompt("What is the capital of Australia?")
print(f"Model: {rec_easy.recommended_model_id} | Cost: ${rec_easy.estimated_cost_usd:.6f} | Savings: {rec_easy.cost_savings_vs_frontier_pct:.1f}%")
# Model: nvidia/Nemotron-3_5-Lightning | Cost: $0.000085 | Savings: 92.0%

# 2. Complex distributed systems architecture -> Routes to NVIDIA Nemotron-3-Ultra-550B
rec_hard = recommend_for_prompt("Architect a globally distributed, Byzantine fault-tolerant consensus protocol for financial ledgers.")
print(f"Model: {rec_hard.recommended_model_id} | Tier: {rec_hard.tier} | Latency: {rec_hard.estimated_latency_sec:.2f}s")
# Model: nvidia/Nemotron-3-Ultra-550b-a55b | Tier: ultra | Latency: 1.06s
```

