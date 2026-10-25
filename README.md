# LLMOptiFit

## Installable web app (current release)

The public app uses FastAPI and the mobile-friendly interface in `web/`. It supports free prompt analysis, model comparisons, Android/iPhone Home Screen installation, and optional generation using **the owner's server-side Nebius key**. Visitors never enter or receive that key.

### Run locally

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-runtime.txt
.venv/bin/uvicorn api.server:app --host 127.0.0.1 --port 8000 --no-proxy-headers --no-access-log
```

Open <http://localhost:8000>. The server checks both bundled classifier artifacts at startup. `/api/health` reports readiness and generation availability; `/api/catalog` reports the snapshot version and verification date. API requests use JSON; prompts are limited to 20,000 characters. API responses are never cached. Interactive API docs are disabled in the public app.

Analysis requires no paid model calls. For generation, set `NEBIUS_API_KEY` in local `.env` (already ignored by Git) or in your hosting secret manager. Set `GENERATION_ENABLED=true` only when you are ready to let visitors consume your Nebius balance. The default is `false`; this preserves the $0 constraint until billing is resolved. The public API ignores visitor key headers and never accepts a provider URL.

**Request limits are not spending limits.** A public server using your key can incur charges. Promotional credits can expire or be exhausted; do not enable public generation on the assumption that a trial account cannot bill you. This app does not implement a durable monetary budget. Configure a provider-enforced credit-only cap if available, or explicitly accept usage charges before enabling generation.

### Free Render deployment

1. Push the reviewed changes to your own repository. Preserve `cache/models/*.pkl`; never commit `.env` or a provider key.
2. Connect that repository in Render and apply `render.yaml`. Confirm the **Free** compute plan, no payment method, no database, and no paid add-ons. The app uses the free HTTPS `onrender.com` address.
3. Leave `GENERATION_ENABLED=false` for the initial analysis-only deployment. When billing is resolved, add `NEBIUS_API_KEY` as a secret in Render and explicitly enable generation.
4. Check `/api/health`, analyze a sample prompt, and test Home Screen installation on a real iPhone and Android device. Only perform a real generation request when you intentionally accept its provider usage.
5. If a deployment fails, use Render's rollback to the last successful revision. Restore the previous Git revision and redeploy if it is outside Render's free rollback window. Manual deployments are configured; pushes do not automatically publish.

Render sleeps idle free services and may take about a minute to wake them. It can suspend service when free allowances are exhausted. Do not enable paid upgrades, paid databases, or automated keep-alive traffic. See [Render's current free-tier limits](https://render.com/docs/free).

The single-worker process keeps hashed IP rate-limit counters in memory for one minute (30 analyses and 5 generation attempts/minute/IP). A semaphore permits two concurrent generation calls. Counters reset on restart and are availability controls, not abuse-proof quotas or billing controls. The deployment disables Uvicorn access logs; application logs contain only known route names, status codes, and elapsed time. Hosting infrastructure may retain network metadata under its policies.

### Tests and maintenance

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests block outbound provider traffic and cover owner-key isolation, disabled generation, routing, usage accounting, limits, safe errors, and static file access. Optional browser tests use an isolated Chromium profile and mocked inference:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python tests/browser_smoke.py --browser '/path/to/chromium'
```

The browser test creates and stops its own local server/browser, checks mobile and desktop layouts, safe rendering, consent, catalog, installation help and offline reload, and saves screenshots under `/tmp/llmoptifit-browser`. It never attaches to your personal browser profile. Actual Android/iPhone installation still needs a device check.

A clean Ubuntu CI job validates the pinned runtime and runs the tests. Update model metadata with `python tools/refresh_catalog.py` (public metadata only; no inference charge), review the catalog diff, then redeploy. The script updates existing models only; it does not invent rankings for newly released models. Version the service-worker cache name whenever interface assets change. Browser updates activate after old tabs close.

The sections below describe the broader research project and its local demo tools; they are not additional public-app features.

---

**Compare LLMs. Optimize cost. Choose the right fit.**

LLMOptiFit is an AI model comparison and recommendation platform designed to help users identify the most suitable Large Language Model for a specific task.

With the growing number of LLM providers, model families, and model variants, choosing the right model is becoming increasingly difficult. Different models vary significantly in reasoning ability, accuracy, context limits, latency, token consumption, and pricing.

LLMOptiFit aims to make that decision easier.

Instead of asking:

> Which LLM is the best?

LLMOptiFit focuses on a more useful question:

> Which LLM is the best fit for this task, considering quality, performance, tokens, and cost?

---

## Why LLMOptiFit?

The LLM ecosystem is expanding rapidly.

Users today may need to choose between models from providers such as:

- OpenAI
- Anthropic
- Google
- Meta
- Mistral
- DeepSeek
- xAI
- Cohere
- Qwen
- and many others

Each provider may also offer multiple model variants optimized for different workloads.

A powerful model may produce excellent results but be unnecessarily expensive for a simple task.

A cheaper model may perform equally well for summarization, classification, extraction, or lightweight coding.

The goal of LLMOptiFit is to help users make that tradeoff intelligently.

---

# Core Idea

LLMOptiFit evaluates models across multiple dimensions such as:

- Task suitability
- Accuracy
- Correctness
- Reasoning capability
- Token consumption
- Input token pricing
- Output token pricing
- Estimated task cost
- Context window
- Latency
- Model capabilities
- Provider
- Model family
- Overall value

The platform can then identify models that offer the best balance between **performance and cost**.

---

# How It Works

```text
User Requirement
       │
       ▼
Understand the Task
       │
       ▼
Identify Candidate LLMs
       │
       ▼
Compare Model Capabilities
       │
       ├── Accuracy
       ├── Correctness
       ├── Reasoning
       ├── Tokens
       ├── Cost
       ├── Context Window
       └── Performance
       │
       ▼
Calculate Task Fit
       │
       ▼
Recommend Best-Fit Models
```

---

# Model Comparison

LLMOptiFit is designed to provide side-by-side comparison between different models.

Example:

| Model | Task Fit | Accuracy | Input Tokens | Output Tokens | Estimated Cost | Speed |
|---|---:|---:|---:|---:|---:|---|
| Model A | 96% | 95% | 12K | 2K | $0.42 | Medium |
| Model B | 92% | 93% | 12K | 2K | $0.18 | Fast |
| Model C | 87% | 89% | 12K | 2K | $0.07 | Very Fast |

Instead of simply declaring one model as universally better, LLMOptiFit can provide different recommendations such as:

- **Best Overall Fit:** Model A
- **Best Price-to-Performance:** Model B
- **Lowest Cost:** Model C
- **Best for Complex Reasoning:** Model A

---

# Prompt-Based Model Recommendation

LLMOptiFit also includes a planned prompt analysis capability.

Users can provide the actual prompt or task they intend to run.

```text
Prompt
   │
   ▼
Prompt Analysis
   │
   ├── Task Type
   ├── Complexity
   ├── Reasoning Requirement
   ├── Expected Context
   ├── Expected Output
   └── Accuracy Requirement
   │
   ▼
Candidate Model Evaluation
   │
   ▼
Best-Fit Model Recommendation
```

Example:

```text
Prompt:

"Analyze this financial report and identify the major
business risks, revenue trends, and potential warning signs."
```

LLMOptiFit could classify the task as:

```text
Task Type: Financial Analysis
Complexity: High
Reasoning Requirement: High
Context Requirement: Large
Accuracy Importance: High
Expected Output: Medium
```

The system can then rank models according to their suitability for that workload.

---

# Planned Recommendation Modes

LLMOptiFit aims to support several recommendation strategies:

- **Best Overall:** Prioritizes output quality and task suitability.
- **Best Value:** Balances quality with token and API cost.
- **Lowest Cost:** Finds the least expensive model capable of completing the task satisfactorily.
- **Best Performance:** Prioritizes capability, reasoning, and correctness.
- **Fastest:** Prioritizes response latency.
- **Best Context Fit:** Recommends models capable of handling the required input size efficiently.

---

# Key Features

- **Model Explorer:** Explore available LLMs and their important characteristics.
- **Model Comparison:** Compare multiple models side-by-side.
- **Pricing Comparison:** Compare input and output token pricing across providers.
- **Token Analysis:** Estimate token requirements for a workload.
- **Cost Estimation:** Estimate the expected cost of running a prompt on different models.
- **Task Fit Scoring:** Estimate how suitable each model is for a particular task.
- **Prompt Analyzer:** Analyze a prompt and understand its complexity and model requirements.
- **Model Recommendation:** Recommend the best model based on user priorities.
- **Cost vs Quality Analysis:** Understand whether paying more for a larger model actually provides meaningful improvement.

---

# Example Use Cases

LLMOptiFit can help users select models for tasks such as:

- Coding & Code Review
- Research & Academic Synthesis
- Document Analysis & Processing
- Summarization & Data Extraction
- Text Classification
- Creative & Copy Writing
- Mathematical Reasoning
- Financial Analysis
- Question Answering
- Translation & Localization
- Long-Context Document Processing
- Agentic Workflows
- Retrieval-Augmented Generation (RAG)
- Customer Support & Chatbots
- Structured Data Generation (JSON / Schema validation)

---

# Example User Journey

A user wants to summarize and analyze a 100-page document.

Instead of manually researching dozens of models, the user enters the requirement into LLMOptiFit.

The system evaluates:

```text
Document size
Task complexity
Context requirement
Reasoning requirement
Expected output size
Accuracy requirement
Available models
Model pricing
Token consumption
```

The result could be:

```text
Recommended Model: Model A

Task Fit: 95%
Estimated Input Tokens: 72,000
Estimated Output Tokens: 4,500
Estimated Cost: $1.24

Reason:
Strong long-context capability and high performance for document analysis.
```

Alternative recommendations are also provided:

```text
Best Value: Model B
Lowest Cost: Model C
Fastest: Model D
```

---

# The Problem We Are Trying to Solve

Choosing an LLM currently requires users to manually evaluate multiple factors:

- Which provider?
- Which model?
- Which model version?
- How capable is the model?
- How expensive is it?
- How many tokens will my task require?
- Is a premium model actually necessary?
- Can a smaller model achieve similar results?
- Which model performs best for my specific workload?

LLMOptiFit brings these questions into a single decision framework.

---

# Project Vision

The long-term vision for LLMOptiFit is to become an **intelligence layer for AI model selection**.

Rather than forcing users to understand every model in the rapidly changing LLM ecosystem, the platform should understand the user's requirement and help identify the most appropriate model automatically.

```text
Prompt
  ↓
Task Understanding
  ↓
Model Selection
  ↓
Cost Optimization
  ↓
Execution
```

Eventually, LLMOptiFit could evolve from a comparison platform into a model-routing system capable of dynamically selecting models based on workload requirements.

---

# Tech Stack & Architecture

- **Backend:** Python (FastAPI / Uvicorn)
- **Frontend / Dashboard:** Streamlit / Gradio
- **Token Estimator & Analysis:** `tiktoken`, Hugging Face `tokenizers`
- **Data & Model Knowledge Base:** SQLite / DuckDB, Pandas

```text
                    ┌──────────────────┐
                    │     Frontend     │
                    │   LLMOptiFit UI  │
                    │ (Streamlit/Gradio)
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  FastAPI Backend │
                    │ Prompt Analyzer  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌────────────────────┐
                    │ Task Classification │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Model Knowledge DB │
                    └─────────┬──────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
          Capability      Token Cost      Benchmark
           Analysis        Analysis         Data
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                    ┌────────────────────┐
                    │ Ranking / Scoring  │
                    │      Engine        │
                    └─────────┬──────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Model Recommendation │
                   └──────────────────────┘
```

---

# Evaluation Framework

A future LLMOptiFit recommendation score considers factors such as:

```text
Model Fit Score =
    Task Suitability
  + Accuracy
  + Correctness
  + Reasoning Capability
  + Context Compatibility
  + Latency
  + Cost Efficiency
  + Token Efficiency
```

The weighting adapts depending on user priorities:

### Quality First
- Accuracy: 30%
- Reasoning: 25%
- Task Fit: 25%
- Correctness: 15%
- Cost: 5%

### Cost First
- Cost: 35%
- Token Usage: 25%
- Task Fit: 20%
- Accuracy: 15%
- Latency: 5%

### Balanced
- Task Fit: 25%
- Accuracy: 20%
- Correctness: 15%
- Cost: 15%
- Token Usage: 10%
- Reasoning: 10%
- Latency: 5%

*These scoring methods are experimental and will evolve as the project develops.*

---

# Roadmap

### Phase 1 — Model Comparison
- [ ] Model database & provider metadata
- [ ] Provider & model family comparison
- [ ] Model capability & context window comparison
- [ ] Token pricing comparison table
- [ ] Interactive frontend comparison dashboard

### Phase 2 — Cost Intelligence
- [ ] Workload token estimator
- [ ] Prompt token estimation
- [ ] Estimated API cost calculations
- [ ] Price-to-performance analysis

### Phase 3 — Prompt Intelligence
- [ ] Prompt classification
- [ ] Task complexity estimation
- [ ] Context requirement estimation
- [ ] Reasoning requirement estimation
- [ ] Task-to-model matching

### Phase 4 — Recommendation Engine
- [ ] Multi-metric model ranking
- [ ] Best-fit & best-value recommendations
- [ ] Lowest-cost recommendation mode
- [ ] User-defined ranking weights

### Phase 5 — Intelligent Routing
- [ ] Automatic model selection & proxy API
- [ ] Dynamic, cost-aware model routing
- [ ] Performance-aware routing
- [ ] Model fallback & retry strategies

---

# Research Questions

LLMOptiFit also explores several broader questions around model selection:

- Does every task require a frontier model?
- How much quality improvement is gained by using a more expensive model?
- Can prompt characteristics predict which model will perform best?
- Can model selection reduce LLM API costs without significantly reducing output quality?
- How should accuracy, latency, token consumption, and price be combined into a model-selection score?
- Can smaller models outperform larger models for specialized workloads?
- Can task-aware routing improve the overall efficiency of multi-model AI systems?

---

# Project Status

LLMOptiFit is currently under active development as a hackathon project.

The architecture, evaluation methodology, scoring system, supported models, and recommendation algorithms may change as the project evolves.

---

# Contributions

Contributions, research ideas, feature suggestions, and discussions are welcome!

Potential contribution areas include:
- LLM benchmarking & evaluation datasets
- Model pricing data updates
- Token estimation & prompt classification
- Recommendation & routing algorithms
- Frontend UI development & data visualization
- Benchmark validation

---

# Datasets, Model & Prompt Complexity Classifier

LLMOptiFit includes real user prompt logs, a 14-parameter prompt complexity scoring system, and a pre-trained model router.

## 1. What is this Data?

### A. LMSYS Chatbot Arena User Prompts (`data/arena/`)
* **Source:** [`lmsys/chatbot_arena_conversations`](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)
* **Description:** Real human user prompts collected from the LMSYS Chatbot Arena, filtered for English conversations, deduplicated, and formatted for prompt classification and router training.
* **Key Files:**
  * `data/arena/arena_prompts_english.parquet`: **23,613 unique English user prompts** with human preference votes (`winner`), model pairings, toxicity tags, and placeholder schema fields for task classification.
  * `data/arena/arena_prompts_english.csv`: CSV export of the processed Arena dataset. Routing and complexity fields are placeholders for future annotation; the bundled classifier is currently trained on synthetic labels.

### B. Pre-trained Model & Router (`cache/models/` & `tools/`)
* **Model File:** `cache/models/prompt_router.pkl` — Pre-trained `RandomForestClassifier` trained on synthetic labeled rubric feature vectors to classify prompts into **Nano** (`nemotron-nano`), **Super** (`nemotron-super`), or **Ultra** (`nemotron-ultra`) model tiers.
* **Classifier:** `tools/prompt_classifier.py` — Feature extractor scoring prompts on a **14-parameter rubric** (Reasoning Complexity, Task Type, Decision Search, Accuracy Requirement, Context Length, Output Structure, Domain Difficulty, etc.).

---

## 2. How to Read and Load the Data

### Reading Parquet Datasets in Python

```python
import pandas as pd

# Load the 23.6K row LMSYS Arena English User Prompts
df_arena = pd.read_parquet("data/arena/arena_prompts_english.parquet")
print(f"Arena prompts: {len(df_arena):,}")
print(df_arena[["prompt_id", "prompt", "model_a", "model_b", "human_preference"]].head())
```

### Classifying Prompts with the Router Model

```python
from tools.prompt_classifier import classify_prompt

prompt = "Write an async Python microservice in FastAPI to stream stock tickers from Redis PubSub."

result = classify_prompt(prompt)
print(f"Route Tier:        {result.route_tier.upper()}")
print(f"Recommended Model: {result.recommended_model}")
print(f"Weighted Score:    {result.weighted_score:.3f}")
print(f"Task Type:         {result.scores.predicted_task_type}")

# Access full 14-parameter rubric breakdown
for driver in result.rationale["top_drivers"]:
    print(f"  {driver['parameter']}: score={driver['score']:.2f}, contribution={driver['contribution']:.3f}")
```

### Testing the Model Recommendation Engine (Live Nebius Token Factory Catalog)

Run [`test_recommendation.py`](test_recommendation.py) to test the recommendation engine on any user prompt:

```bash
# 1. Test a single prompt via CLI
python test_recommendation.py "Write a Python script to scrape product prices from an HTML table and save to SQLite."

# 2. Interactive prompt testing mode
python test_recommendation.py

# 3. Run the 4-tier benchmark evaluation suite
python test_recommendation.py --demo

# 4. Programmatic JSON output
python test_recommendation.py "What is photosynthesis?" --json-output

# 5. Run the 1,000 Prompt Synthetic Routing Regression Benchmark
python tools/benchmark_1000_prompts.py
```

# Running the Applications

LLMOptiFit provides two full application experiences, plus a one-click launcher:

### 1. Interactive Streamlit Dashboard (Recommended for Demos & Hackathon Evaluation)
Features real-time prompt telemetry, interactive Altair Pareto frontier tradeoff curves, quick benchmark presets, and live Nebius Token Factory generation:

```bash
# Run with launcher:
./run_app.sh streamlit

# Or directly with Streamlit:
.venv/bin/streamlit run streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

---

### 2. FastAPI Modular Application & OpenAPI Swagger UI
FastAPI backend for the installable web app:

```bash
# Run with launcher:
./run_app.sh api

# Or directly with Uvicorn:
.venv/bin/uvicorn api.server:app --reload --port 8000
```
- Public API docs are disabled; see `api/schemas.py` for request/response contracts.
- Web Frontend: **`http://localhost:8000/`**
- API Endpoints: `/api/health`, `/api/catalog`, `/api/analyze`, `/api/generate`

---

# Disclaimer

LLM performance varies depending on prompts, datasets, model versions, provider updates, temperature settings, system instructions, and evaluation methodology.

LLMOptiFit recommendations should therefore be treated as decision-support information rather than an absolute measure of model quality. Model pricing and capabilities may also change over time and should be verified with the respective model provider.
