# LLMOptiFit: Hackathon Project Plan

## 1. Project summary

**LLMOptiFit** is an intelligent LLM router. It analyzes a prompt using a trained data science model, predicts prompt characteristics, and uses those predictions plus user priorities to recommend and select a suitable large language model (LLM). The selected model then answers the prompt through Nebius Token Factory.

The core idea is to route each request according to what it needs instead of sending every prompt to the same model. A short extraction task may favor a fast, economical model; a complex coding or multi-step reasoning task may need a more capable model.

### One-sentence pitch

> LLMOptiFit predicts what a prompt needs, then routes it to a suitable LLM with a visible explanation of the tradeoffs.

### Track fit

The strongest fit is **Best Apps and Agents**: LLMOptiFit is an application that routes user requests to models through Nebius Token Factory. The hackathon requires projects to run on Nebius Token Factory or Nebius AI Cloud and use at least one NVIDIA open-source model. Include an NVIDIA model as a real model candidate and demonstrate an actual inference request to it during the demo—not only a mention in the documentation.

## 2. Problem and audience

### Problem

Users and developers often default to one LLM for every request. That can waste time or inference budget on simple tasks, while more demanding prompts may need a stronger model. Choosing among models manually also requires users to understand model capabilities and tradeoffs.

### Intended users

- Developers integrating multiple LLMs into an application.
- Teams that want a transparent starting point for matching task needs to model capabilities.
- Hackathon judges evaluating a practical use of Nebius-hosted open models.

### Product value

LLMOptiFit makes a model choice using explicit prompt signals, exposes why it made that choice, and lets users inspect the resulting answer. It is a recommendation and routing aid; it should not claim to identify the universally best model unless that claim is supported by a defined benchmark.

## 3. Goals and scope

### Hackathon MVP goals

1. Accept a prompt and optional user priorities.
2. Run the existing trained prompt-parameter model.
3. Display the predicted parameters and confidence where available.
4. Apply a deterministic, inspectable routing policy to select a model.
5. Send the prompt to the selected model through Nebius Token Factory.
6. Include an NVIDIA open-source model in the available routes and demonstrate it in a real request.
7. Show the selected model, routing explanation, answer, and basic request metrics.
8. Compare the router with a simple baseline using a small, documented evaluation set.

### Explicitly out of scope for the MVP

- Supporting every LLM provider or model on the market.
- Claiming a universal quality ranking across models.
- Training a new model during the demo.
- Fully autonomous dynamic routing based on production-scale online learning.
- Storing user prompts or answers beyond what is needed for the live session, unless explicitly designed and disclosed.

## 4. Product flow

```text
User prompt + optional priorities
              ↓
Prompt-parameter prediction model
              ↓
Task type, complexity, reasoning need, ambiguity
              ↓
Routing policy selects an eligible model
              ↓
Nebius Token Factory inference request
              ↓
Answer + routing explanation + observed metrics
```

### Example interaction

1. The user enters: “Write a Python function to merge overlapping intervals and explain its time complexity.”
2. The classifier predicts: `task_type=coding`, `complexity=medium`, `reasoning_requirement=medium`, `ambiguity=low`.
3. The routing policy considers those predictions and the user’s priority, such as balanced quality and latency.
4. LLMOptiFit chooses a model from the configured catalog. The catalog includes at least one NVIDIA open-source model.
5. The backend sends the request to Nebius Token Factory and returns the answer.
6. The UI displays the prediction, selected model, reason, answer, latency, and token counts if provided by the API.

## 5. System architecture

Keep the prompt classifier, routing policy, model catalog, and inference provider as separate components. This lets the project explain and evaluate each decision independently.

### Components

| Component | Responsibility | Input | Output |
|---|---|---|---|
| UI | Collect prompt and priorities; display results | Prompt, preferences | Request to backend; rendered result |
| API endpoint | Validate request and coordinate flow | Request payload | Structured response |
| Parameter predictor | Load and invoke the existing trained model | Prompt and any model-required features | Predicted parameter labels/scores |
| Router | Apply policy to predictions, constraints, and preferences | Labels/scores, model catalog, priorities | Selected model and explanation |
| Model catalog | Store model IDs and capability/availability metadata | Configuration | Eligible model candidates |
| Nebius provider adapter | Call Token Factory and normalize response | Model ID and prompt | Text response and usage metadata |
| Evaluation module | Compare routing against baselines | Evaluation prompts and outcomes | Quality, routing, latency, and cost summaries |

### Suggested repository layout

Adapt this layout to the current repository’s language and framework. It is a proposed organization, not an assumption about files that already exist.

```text
llmoptifit/
├── app/                         # Existing UI or frontend
├── api/                         # Backend request handling
│   ├── routes.py                # Example: POST /api/route-and-generate
│   ├── schemas.py               # Request/response validation
│   ├── predictor.py             # Wrapper around the trained parameter model
│   ├── router.py                # Routing policy
│   └── providers/
│       └── nebius.py            # Token Factory inference adapter
├── config/
│   └── model_catalog.yaml       # Model IDs and routing metadata
├── models/                      # Model artifact, or instructions to fetch it
├── evaluation/
│   ├── prompts.jsonl            # Small, representative evaluation set
│   ├── run_evaluation.py
│   └── results/                 # Generated evaluation summary
├── tests/
│   ├── test_predictor.py
│   ├── test_router.py
│   └── test_api.py
├── .env.example                 # Variable names only; no secrets
├── README.md
├── LICENSE
└── requirements.txt             # Or the package manager manifest in use
```

### Request sequence

1. The client sends the prompt and routing preferences to the backend.
2. The backend validates inputs and applies reasonable length limits.
3. The predictor returns prompt parameters and confidence scores, if supported.
4. The router filters candidates by availability and hard constraints, then scores candidates against task fit and user priorities.
5. The provider adapter sends the prompt to the selected model using the server-side Nebius API key.
6. The backend returns a normalized response containing predictions, route, explanation, answer, and metrics.

## 6. Prompt-parameter model integration

The existing data science model is the project’s decision input. Wrap it behind a stable interface rather than embedding model-specific loading or inference code in the API route.

### Predictor interface

```python
# api/predictor.py — illustrative interface; adapt to the existing model

def predict_parameters(prompt: str) -> dict:
    """Return the prompt attributes expected by the router."""
    return {
        "task_type": "coding",
        "complexity": "medium",
        "reasoning_requirement": "medium",
        "ambiguity": "low",
        "confidence": {
            "task_type": 0.94,
            "complexity": 0.78,
            "reasoning_requirement": 0.81,
            "ambiguity": 0.88,
        },
    }
```

The example values above are illustrative only. Use the actual parameter names, labels, and output format from the trained model. If the model has no calibrated confidence scores, omit them rather than inventing values.

### Proposed parameter contract

| Parameter | Example values | How the router can use it |
|---|---|---|
| `task_type` | coding, extraction, summarization, creative, reasoning | Match task to known model strengths |
| `complexity` | low, medium, high | Increase capability threshold as complexity rises |
| `reasoning_requirement` | low, medium, high | Prefer stronger reasoning for multi-step tasks |
| `ambiguity` | low, medium, high | Prefer models suited to clarification or robust interpretation; optionally ask the user a follow-up |
| `confidence` | Per-parameter probability, if available | Flag uncertain predictions or use a conservative route |

Keep labels and enum values consistent between the trained model, API schema, evaluation set, and UI. Document any preprocessing needed by the predictor.

## 7. Routing policy

The router should be deterministic and explainable for the MVP. Do not make a second LLM call to decide the model unless that is itself an explicit, evaluated part of the project; otherwise the classifier you built is no longer visibly driving the decision.

### Candidate model catalog

Maintain a small catalog with exact model IDs copied from the current Nebius Token Factory model catalog. Confirm availability before the demo. Include at least one NVIDIA open-source model, such as an available Nemotron model, and make sure it can be selected through the normal routing path.

Example configuration from live Nebius catalog (`config/nebius_catalog.json`):

```yaml
models:
  - id: "nvidia/Nemotron-3_5-Lightning"
    display_name: "Nemotron-3.5-Lightning"
    provider: nebius_token_factory
    vendor: "NVIDIA"
    strengths: [general, summarization, low_latency, cost_efficiency]
    reasoning_tier: nano
    input_price_per_m: 0.06
    output_price_per_m: 0.24
    nvidia_open_model: true
    enabled: true

  - id: "deepseek-ai/DeepSeek-V4-Flash-0731"
    display_name: "DeepSeek-V4-Flash"
    provider: nebius_token_factory
    vendor: "DeepSeek"
    strengths: [coding, analysis, high_throughput]
    reasoning_tier: super
    input_price_per_m: 0.14
    output_price_per_m: 0.28
    nvidia_open_model: false
    enabled: true

  - id: "nvidia/nemotron-3-super-120b-a12b"
    display_name: "Nemotron-3-Super-120B"
    provider: nebius_token_factory
    vendor: "NVIDIA"
    strengths: [reasoning, coding, math]
    reasoning_tier: super
    input_price_per_m: 0.30
    output_price_per_m: 0.90
    nvidia_open_model: true
    enabled: true

  - id: "nvidia/Nemotron-3-Ultra-550b-a55b"
    display_name: "Nemotron-3-Ultra-550B"
    provider: nebius_token_factory
    vendor: "NVIDIA"
    strengths: [frontier_reasoning, system_architecture, math, planning]
    reasoning_tier: ultra
    input_price_per_m: 1.00
    output_price_per_m: 3.00
    nvidia_open_model: true
    enabled: true
```

The catalog metadata is an initial routing hypothesis. Validate it with observed results; do not present capability tags as objective truth without evidence.

### Simple scoring approach

A transparent score can combine task fit and user priorities:

```text
candidate_score =
    task_fit_weight       * task_fit
  + reasoning_weight      * reasoning_fit
  + quality_weight        * expected_quality
  - latency_weight        * expected_latency
  - cost_weight           * estimated_cost
```

For an MVP, `task_fit` and `reasoning_fit` can be categorical scores defined in the catalog or policy. If no measured latency or cost data exists yet, call those values estimates or omit them. Keep user constraints (for example, allowed models or maximum latency) as filters rather than soft score terms.

### Router interface

```python
# api/router.py — illustrative pseudocode

def choose_model(prediction: dict, preference: str, catalog: list[dict]) -> dict:
    candidates = [m for m in catalog if m["enabled"]]
    # Apply hard constraints first, then score candidates against predicted needs.
    # Return selected model ID and a human-readable explanation.
    return {
        "model_id": "<selected-model-id>",
        "reason": "Selected for the predicted task and reasoning needs.",
        "scores": {},
    }
```

Return enough information for the UI to explain the route, but avoid exposing internal secrets or irrelevant model internals.

### Low-confidence behavior

If the predictor reports low confidence or an unsupported label:

- Use a documented fallback route, such as the balanced model; or
- Ask the user a short clarifying question if ambiguity materially changes the route.

Record which fallback was used so evaluation can measure it. Do not fail silently or route to an unconfigured model.

## 8. Nebius Token Factory integration

Nebius Token Factory provides an OpenAI-compatible inference API. The documented base URL is `https://api.tokenfactory.nebius.com/v1/`; requests use a model ID, chat messages, and a bearer API key. Keep this integration in the backend provider adapter. Never place the API key in browser JavaScript, a checked-in file, logs, or the public repository.

### Python example

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
)

def generate(model_id: str, prompt: str) -> dict:
    result = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "text": result.choices[0].message.content,
        "model": getattr(result, "model", model_id),
        "usage": getattr(result, "usage", None),
    }
```

### Environment setup

`.env.example` should contain names and safe placeholders only:

```dotenv
NEBIUS_API_KEY=replace_me
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
```

For local development, load a private `.env` file that is excluded by `.gitignore`. For deployment, set `NEBIUS_API_KEY` in the host’s server-side secret configuration. Do not commit a real key. Add request timeout and user-friendly error handling for rate limits, unavailable models, and provider failures.

### Hackathon requirement evidence

In the application, README, and demo video, make clear:

- Which component runs on Nebius Token Factory.
- Which NVIDIA open-source model is in the catalog.
- A visible live request where that model is selected and returns an answer.
- How the user’s prompt parameters caused that route.

## 9. API contract

An endpoint such as `POST /api/route-and-generate` can accept:

```json
{
  "prompt": "Explain this SQL query and identify possible performance issues.",
  "preference": "balanced",
  "allowed_models": []
}
```

A normalized response can look like:

```json
{
  "prediction": {
    "task_type": "analysis",
    "complexity": "medium",
    "reasoning_requirement": "medium",
    "ambiguity": "low"
  },
  "routing": {
    "selected_model_id": "<actual-model-id>",
    "reason": "Selected for the predicted analysis and reasoning requirements.",
    "fallback_used": false
  },
  "answer": "<model response>",
  "metrics": {
    "latency_ms": 0,
    "prompt_tokens": null,
    "completion_tokens": null,
    "estimated_cost": null
  }
}
```

Use `null` or omit unavailable metrics. Do not report fabricated or falsely precise cost or token data.

## 10. User experience

### Main screen

- Prompt input with a few example prompts.
- Optional priority selector: `balanced`, `lowest latency`, or `lowest estimated cost`.
- Optional model restriction for comparison/testing.
- “Analyze and run” action.

### Results view

Show the following in order:

1. The generated answer.
2. The selected model and a plain-language routing explanation.
3. Predicted prompt parameters, with confidence only if the classifier provides it.
4. Observed latency and API usage data where available.
5. Optional “Compare another model” action for a controlled comparison.

Make the chosen model visible before or alongside its answer. The main demonstration is the parameter model driving a real route, so keep that chain easy to follow.

### Error states

- Missing or invalid API key: explain that server configuration is incomplete without revealing secret values.
- Model unavailable or rate limited: return a useful message and optionally use a documented fallback.
- Predictor error: show that analysis failed and offer a conservative configured route only if this is an intentional fallback.
- Invalid prompt or unsupported task: ask for a revised prompt or use the documented general route.

## 11. Evaluation plan

Evaluate both the parameter model and the end-to-end routing outcome. A classifier can have good label accuracy while still routing poorly, so report these separately.

### Evaluation set

Create a small, fixed set of representative prompts across the supported task types and difficulty levels. Include simple and challenging examples, ambiguous prompts, and prompts likely to distinguish the candidate models. Store prompts without sensitive user data.

For each example, define expected prompt parameters and, where practical, a rubric for response quality. Clearly label whether the expected parameters are human-labeled, derived from existing training data, or generated by another method.

### Baselines

At minimum compare:

1. **Always-one-model baseline:** send all prompts to one chosen model.
2. **Router:** use the classifier’s parameters and routing policy.

Optionally compare a direct, static rule-based router to isolate the value added by the trained classifier.

### Metrics

| Area | Metric | Notes |
|---|---|---|
| Parameter model | Accuracy, macro-F1, or per-label precision/recall | Choose metrics appropriate to class balance and task |
| Routing | Route distribution and eligible-route rate | Confirm the intended candidate pool is actually used |
| Answer quality | Human rubric or task-specific correctness checks | Define criteria before reviewing outputs |
| Efficiency | Median and p95 latency | Measure repeated requests where feasible |
| Cost | Estimated cost per request or tokens used | Use only documented pricing/usage; state assumptions |
| Safety/reliability | Error and fallback rate | Include provider failures and low-confidence cases |

Do not claim the router is “better” based on one hand-picked example. Show the test size, metric definitions, and limitations alongside any result.

## 12. Privacy, security, and reliability

- Keep the Nebius API key server-side and out of version control.
- Do not log full prompts or model outputs by default. If debugging logs are needed, use redaction and a short retention period.
- Avoid sending sensitive personal information in the demo prompts.
- Validate prompt length and request payload shape.
- Use request timeouts and handle rate limits/provider errors.
- Treat classifier output as untrusted input: validate it against known labels before routing.
- Restrict model IDs to the configured catalog; never accept arbitrary model IDs directly from the client.
- If using a model artifact, document how it is obtained and ensure it is included only if its license permits.
- State that model recommendations are heuristic and depend on the available catalog and evaluation data.

## 13. Build plan

### Phase 1 — Inspect and connect existing model

- Identify the current app framework, entry points, and deployment path.
- Confirm the trained model’s loading method, input preprocessing, parameter names, output labels, and artifact location.
- Wrap it in a predictor interface and validate outputs on representative examples.

**Exit condition:** a backend function accepts a prompt and returns a validated parameter object.

### Phase 2 — Implement routing policy

- Define model catalog entries using exact currently available Nebius model IDs.
- Include an NVIDIA open-source candidate that is usable through the normal request flow.
- Define task-fit and reasoning-fit rules, priorities, hard constraints, and low-confidence fallback.
- Return a selected model ID and explanation.

**Exit condition:** deterministic test cases select expected configured candidates and produce explanations.

### Phase 3 — Add Nebius inference

- Create the provider adapter using the OpenAI-compatible API.
- Configure `NEBIUS_API_KEY` only in server-side secrets.
- Test a live inference request with each model required for the demo, including the NVIDIA model.
- Normalize response text, usage, latency, and provider errors.

**Exit condition:** a live server-side request returns an answer from the selected model.

### Phase 4 — Connect UI and API

- Add prompt input and optional priority selection.
- Connect the request flow from predictor to router to provider.
- Display parameters, selected model, explanation, answer, and metrics.
- Implement loading, validation, and error states.

**Exit condition:** a user can complete the entire route-and-generate flow in the app.

### Phase 5 — Evaluate and refine

- Create a fixed prompt set and label or document the evaluation method.
- Compare against an always-one-model baseline.
- Check model availability, route coverage, latency, answer quality, and errors.
- Revise routing weights or rules based on measured outcomes.

**Exit condition:** README and demo can report a small, reproducible evaluation with limitations.

### Phase 6 — Package and submit

- Prepare a public repository with an open-source license at the repository root.
- Add setup instructions, environment variable names, model usage details, and Nebius/NVIDIA attribution.
- Deploy a working demo URL.
- Record a public video no longer than three minutes showing key modules in action.
- Verify the repository, demo, and video links before submission.

## 14. Demo script (about 2–3 minutes)

1. **Problem (15 seconds):** “Different prompts have different needs; LLMOptiFit predicts those needs and routes to an appropriate model.”
2. **Show a straightforward task (25 seconds):** submit a short extraction or summary prompt and show its predicted parameters, selected route, and answer.
3. **Show a demanding task (35 seconds):** submit a multi-step reasoning or coding prompt. Show how the parameters change and how that affects the model choice.
4. **Show NVIDIA model on Nebius (30 seconds):** select or trigger a route that uses the configured NVIDIA open-source model, then show the live response and visible model ID.
5. **Explain the router (25 seconds):** briefly show the routing reason and the policy’s use of task, complexity, reasoning, and user priority.
6. **Show evaluation (25 seconds):** present the small baseline comparison and explain the test size and metrics.
7. **Close (10 seconds):** point to the working demo and public repository.

The video must show the actual app modules working and explain the use of Nebius Token Factory and the NVIDIA model.

## 15. Submission checklist

- [ ] Project is categorized under the best-fit hackathon track.
- [ ] Runnable project uses Nebius Token Factory or Nebius AI Cloud.
- [ ] At least one NVIDIA open-source model is used in live inference.
- [ ] The parameter model visibly drives the route selection.
- [ ] Working demo URL is accessible to judges.
- [ ] Public code repository is accessible.
- [ ] Open-source license is visible at the repository root.
- [ ] README includes setup and run instructions.
- [ ] README identifies Nebius services, NVIDIA model(s), and how they are used.
- [ ] Demo video is public, no longer than three minutes, and shows the application in action.
- [ ] Submission includes requested feedback on Nebius and NVIDIA tools/models.
- [ ] No API keys, private prompts, or credentials are committed or shown in the demo.
- [ ] Model IDs and availability are rechecked before recording and submission.

## 16. README outline

1. Project name and one-sentence pitch.
2. Problem and intended users.
3. How LLMOptiFit works (parameter model → routing policy → Nebius inference).
4. Architecture diagram.
5. NVIDIA open-source model(s) used and where they appear in the live flow.
6. Nebius Token Factory integration details.
7. Demo link and video link.
8. Setup instructions and required environment variables.
9. How to run tests and evaluation.
10. Results, metric definitions, and limitations.
11. Privacy/security notes.
12. License.

## 17. Key implementation decisions to confirm in the repository

Before writing code, map this plan to the existing project:

- Framework and current UI/API entry points.
- Predictor model format, loading code, preprocessing, and output schema.
- Whether inference runs in Python or another backend runtime.
- Current hosting/deployment target and its secret-management support.
- The exact model IDs available to the Nebius account at build time.
- Whether the classifier has calibrated confidence outputs.
- Which evaluation examples and labels can be used without exposing sensitive data.

## References

- Hackathon overview and requirements: <https://nebiusglobalaihackathon.devpost.com/>
- Nebius Token Factory API introduction: <https://docs.tokenfactory.nebius.com/api-reference/introduction>
- Nebius Token Factory quickstart: <https://docs.tokenfactory.nebius.com/quickstart>
- Nebius structured output and JSON guide: <https://docs.tokenfactory.nebius.com/ai-models-inference/json>
