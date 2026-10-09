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

## 7. Artifacts & Usage

1. **Interactive Jupyter Notebook**: [`prompt.ipynb`](../prompt.ipynb) — Fully executed with 28 code cells and embedded visual figures.
2. **Classified Parquet Dataset**: `data/arena/arena_prompts_classified.parquet` — 23,613 prompts enriched with 21 classification features.
3. **Live Router Function**:
```python
from tools.prompt_classifier import classify_prompt

res = classify_prompt("Design an async Python microservice with Redis pub/sub and OpenTelemetry.")
print(f"Tier: {res.route_tier} | Model: {res.recommended_model} | Score: {res.weighted_score}")
# Tier: ultra | Model: nemotron-ultra | Score: 0.612
```
