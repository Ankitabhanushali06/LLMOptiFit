import json
import pickle
import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SAVE_PATH = PROJECT_ROOT / "cache/models/task_classifier.pkl"
BENCHMARK_PATH = PROJECT_ROOT / "data/benchmark_1000_prompts.json"

TASK_DATA = {
    "coding": [
        "write sql query to {action}",
        "write a postgres sql query that {action}",
        "write an optimized sql query with joins between {a} and {b}",
        "write a python script using {lib} to {action}",
        "implement a function in {lang} that {action}",
        "debug this {lang} code that throws {err}",
        "refactor this function to be more efficient and clean",
        "solve the two-sum problem in {lang} with O(n) complexity",
        "write a regex pattern to validate email addresses and phone numbers",
        "create a REST API endpoint in FastAPI with pydantic validation",
        "implement a binary search tree in {lang} with insert and delete operations",
        "write a bash shell script to automate database backups and log errors",
        "implement rate limiting middleware in Node.js using Redis",
        "write a Dockerfile and docker-compose file for a microservice architecture",
        "implement Dijkstra's shortest path algorithm in {lang}",
        "write unit tests using pytest for this payment gateway integration",
        "implement an LRU cache with get and put methods in O(1) time",
        "write a React hook in TypeScript for managing WebSocket connections",
        "create a GraphQL schema and resolver for fetching user orders",
        "write code to serialize and deserialize JSON objects in {lang}",
    ],
    "mathematical_reasoning": [
        "solve for x: {math_eq}",
        "solve the quadratic equation {math_eq}",
        "find the roots of the polynomial {math_poly}",
        "calculate the derivative of f(x) = {math_func}",
        "evaluate the definite integral of {math_func} from 0 to 1",
        "compute the limit of {math_lim} as x approaches infinity",
        "prove by induction that {math_claim}",
        "find the eigenvalues and eigenvectors of matrix {math_matrix}",
        "calculate the probability of {math_prob}",
        "derive the Bayes' theorem posterior probability given prior {math_val}",
        "solve the system of linear equations: {math_sys}",
        "compute the dot product and cross product of two 3D vectors",
        "factor the algebraic expression: {math_expr}",
        "simplify the following fraction: {math_expr}",
        "prove that the square root of 2 is an irrational number",
        "calculate the standard deviation, variance, and mean for the dataset",
        "solve the differential equation dy/dx + 2y = 4x",
        "compute the Taylor series expansion of sin(x) around x=0 up to degree 5",
        "determine if the infinite series sum(1/n^2) converges or diverges",
        "calculate the expected value and variance of a Poisson distribution",
    ],
    "extraction": [
        "extract all email addresses and phone numbers from the following text: {text}",
        "parse the invoice text and extract invoice number, date, vendor, and amount: {text}",
        "extract all dates and dollar amounts from this contract: {text}",
        "extract names, addresses, and phone numbers from this contact dump: {text}",
        "pull out all named entities (Person, Organization, Location) from this news snippet: {text}",
        "extract all chemical compound names and formulas from this research paragraph: {text}",
        "extract the job title, employer, and years of experience from this resume: {text}",
        "parse this JSON string and extract all active user IDs: {text}",
        "extract all URLs, links, and domain names from this raw HTML string: {text}",
        "extract the error code and stack trace from this system log file: {text}",
        "pull out all numerical values and percentage figures from this earnings call transcript: {text}",
        "extract the flight numbers, departure cities, and arrival times from this itinerary: {text}",
        "parse the XML document and extract all product SKUs and inventory levels: {text}",
        "extract the key terms, definitions, and acronyms from this technical documentation: {text}",
        "extract table headers and row records from this unstructured text dump: {text}",
    ],
    "analysis": [
        "compare {tech1} versus {tech2} in terms of performance and scalability",
        "analyze the trade-offs between {tech1} and {tech2} for real-time streaming",
        "conduct a root cause analysis for the database outage and explain why it failed",
        "analyze the pros and cons of microservices versus monolithic architecture",
        "explain why {phenom} occurs and analyze its consequences for {domain}",
        "critique this system design proposal and identify potential single points of failure",
        "evaluate the financial health and debt risk of {company} based on quarterly filings",
        "compare relational SQL databases with NoSQL document stores for this use case",
        "analyze the security risks and attack vectors of using JWT tokens in local storage",
        "examine the performance bottlenecks in this multi-threaded architecture: {text}",
        "compare convolutional neural networks with vision transformers for medical imaging",
        "analyze the cost-benefit ratio of moving from on-premise servers to AWS cloud",
        "evaluate the impact of interest rate hikes on venture capital investment trends",
        "conduct a competitive analysis between Stripe and Adyen for global payment processing",
        "analyze the ethical and legal implications of autonomous AI agents in healthcare",
    ],
    "planning": [
        "create a step-by-step implementation plan for migrating a database to the cloud",
        "design an architecture roadmap for building a high-availability payment system",
        "draft a 4-week project sprint schedule and milestone tracker for our MVP launch",
        "create a comprehensive disaster recovery plan for our multi-region Kubernetes cluster",
        "outline the phases and rollout plan for adopting zero-trust enterprise security",
        "outline the sequential steps required to implement zero-trust network access in an enterprise",
        "outline the step-by-step procedure required to implement this architecture migration",
        "plan the rollout and sequential implementation steps for enterprise deployment",
        "develop a 90-day onboarding and training plan for incoming software engineers",
        "design a capacity planning strategy to prepare for 10x traffic surges during Black Friday",
        "create a project charter and risk management plan for launching our mobile application",
        "draft a security compliance checklist and audit remediation plan for SOC 2 Type II",
        "plan the migration strategy and deprecation timeline for our legacy API v1",
        "create a detailed itinerary and logistics plan for a 3-day technical conference",
        "develop a strategic roadmap for integrating generative AI into customer support",
    ],
    "classification": [
        "classify the sentiment of this review as positive, negative, or neutral: '{text}'",
        "categorize the following customer support ticket into billing, technical, or shipping: '{text}'",
        "is the following email spam or legitimate ham? classify it: '{text}'",
        "classify this news headline as clickbait or factual news: '{text}'",
        "assign the appropriate medical specialty (cardiology, oncology, neurology) to this clinical note: '{text}'",
        "classify the severity of this bug report as P0 (critical), P1 (high), or P2 (low): '{text}'",
        "categorize this credit card transaction as grocery, travel, utility, or dining: '{text}'",
        "determine the intent of this chatbot user query (informational, navigational, transactional): '{text}'",
        "label the toxicity of this comment as benign, toxic, or severely toxic: '{text}'",
        "classify whether this financial statement indicates high bankruptcy risk or low risk: '{text}'",
        "categorize this customer query into sales, returns, or technical troubleshooting: '{text}'",
    ],
    "research": [
        "provide a comprehensive literature review on quantum error correction breakthroughs",
        "what is the current state of academic research regarding room-temperature superconductivity?",
        "summarize published scientific studies on the neurological impact of microplastics",
        "survey the academic literature on transformer attention mechanism optimizations from 2023 to 2025",
        "synthesize recent peer-reviewed medical papers on mRNA vaccine applications in oncology",
        "what are the leading empirical findings on the economic impact of universal basic income?",
        "summarize recent research into solid-state battery electrolytes and manufacturing challenges",
        "survey the published research on reinforcement learning with human feedback (RLHF)",
        "what is known in the scientific literature about cellular senescence reversal using senolytics?",
        "provide an overview of historical and contemporary research on behavioral economics nudges",
        "review academic papers exploring dark matter candidates beyond WIMPs and axions",
    ],
    "generation": [
        "write a polite email to a client explaining a 2-day delivery delay due to weather",
        "draft a compelling 500-word blog post announcing the launch of our new product",
        "compose a science fiction short story about an astronaut marooned on Titan",
        "write an engaging LinkedIn post sharing 5 lessons learned from failing at my first startup",
        "draft a motivational keynote speech for graduating engineering students",
        "write an attractive product description for an ergonomic mechanical keyboard",
        "compose a persuasive pitch deck script to pitch our climate startup to investors",
        "write a friendly welcome email for new newsletter subscribers with onboarding tips",
        "draft a formal press release announcing our partnership with a leading bank",
        "write a creative poem about the autumn leaves falling in a quiet forest",
        "compose a thoughtful cover letter applying for a Senior Data Scientist role",
    ],
}

FILLERS = {
    "action": [
        "fetch data from an API and cache in Redis",
        "calculate monthly revenue and group by region",
        "find all records where manager salary is less than employee salary",
        "clean null values and normalize date formats",
        "filter active subscribers and export to CSV",
    ],
    "a": ["orders", "users", "transactions", "customers"],
    "b": ["invoices", "shipments", "payments", "accounts"],
    "lib": ["pandas", "numpy", "matplotlib", "httpx", "sqlalchemy", "scipy"],
    "lang": ["Python", "JavaScript", "Go", "Rust", "Java", "C++", "TypeScript", "SQL"],
    "err": ["IndexError", "KeyError", "ZeroDivisionError", "NullPointerException", "Deadlock"],
    "tech1": ["PostgreSQL", "Kafka", "Redis", "gRPC", "Docker", "Kubernetes", "AWS DynamoDB"],
    "tech2": ["MongoDB", "RabbitMQ", "Memcached", "REST", "Podman", "Nomad", "Cassandra"],
    "phenom": ["the vanishing gradient", "cache thrashing", "split-brain consensus", "memory fragmentation"],
    "domain": ["deep learning", "high-frequency trading", "distributed databases", "operating systems"],
    "company": ["Tesla Inc.", "Apple Inc.", "Amazon.com", "Microsoft Corp.", "Alphabet Inc."],
    "text": [
        "Product arrived shattered, support refused refund after 3 weeks.",
        "Loved the experience! The design is sleek, checkout was fast.",
        "Need to reset password for user account #84920 on portal.",
        "John Doe (john@acme.com) paid $4,500 on 2026-03-15 for order #8819.",
        "Patient presented on 2026-02-10 with acute bronchitis, prescribed 500mg.",
    ],
    "math_eq": [
        "3x^2 - 12x + 9 = 0",
        "2x^2 + 5x - 3 = 0",
        "x^3 - 4x^2 + x + 6 = 0",
        "4x - 7 = 3x + 15",
        "e^(2x) - 5e^x + 6 = 0",
        "cos(2x) = 1/2 for x in [0, 2pi]",
        "sqrt(2x + 5) = x - 1",
    ],
    "math_poly": ["2x^2 + 7x + 3", "x^3 - 6x^2 + 11x - 6", "4x^2 - 25"],
    "math_func": ["3x^4 - 5x^2 + 7x - 2", "ln(x^2 + 1)", "e^(3x) * sin(2x)", "x / (x^2 + 1)"],
    "math_lim": ["(sin(3x)) / x", "(x^2 - 4) / (x - 2)", "(sqrt(x + 1) - 1) / x"],
    "math_claim": [
        "1 + 2 + ... + n = n(n+1)/2 for all n >= 1",
        "2^n > n for all positive integers n",
    ],
    "math_matrix": ["[[2, 1], [1, 2]]", "[[4, 2], [1, 3]]", "[[3, 0], [0, -2]]"],
    "math_prob": [
        "drawing two aces consecutively without replacement",
        "getting at least 3 heads in 5 fair coin flips",
    ],
    "math_val": ["P(A) = 0.01, P(B|A) = 0.95, P(B|not A) = 0.05"],
    "math_sys": ["2x + 3y = 12 and x - 2y = -1", "3x + 4y = 10 and 2x - y = 3"],
    "math_expr": ["(x^2 - 9)/(x + 3)", "sqrt(50) + sqrt(18) - sqrt(8)"],
}


def build_training_dataset(samples_per_class: int = 400, seed: int = 42) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    texts, labels = [], []

    for task_type, templates in TASK_DATA.items():
        for _ in range(samples_per_class):
            tmpl = rng.choice(templates)
            formatted = tmpl
            for key, options in FILLERS.items():
                token = "{" + key + "}"
                while token in formatted:
                    chosen = rng.choice(options)
                    formatted = formatted.replace(token, chosen, 1)

            # Add stylistic noise / user prefixes
            prefix_roll = rng.random()
            if prefix_roll < 0.20:
                prefix = rng.choice([
                    "Please ", "Can you ", "Could you please ", "I need you to ", "Help me: ",
                    "Kindly ", "Hey, ", "Quick question: ", "Prompt: "
                ])
                formatted = prefix + formatted
            elif prefix_roll < 0.35:
                suffix = rng.choice([
                    " Thanks!", " Please explain in detail.", " Output formatted as clean markdown.",
                    " Be concise and accurate.", " Show your reasoning step by step."
                ])
                formatted = formatted + suffix

            texts.append(formatted)
            labels.append(task_type)

    # Combine and shuffle
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


def train_and_save():
    print("Generating comprehensive training dataset across 8 task types...")
    x_train, y_train = build_training_dataset(samples_per_class=450, seed=42)
    print(f"Total training samples: {len(x_train)} across {len(set(y_train))} classes")

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_features=15000,
                sublinear_tf=True,
                token_pattern=r"(?u)\b[\w\^=\-\+\*/]+\b|\b\w+\b",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=3.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    print("Fitting statistical NLP pipeline (TF-IDF + Calibrated Logistic Regression)...")
    pipeline.fit(x_train, y_train)

    # Validate against the 1,000 prompt benchmark
    print(f"\nEvaluating against 1,000 ground-truth benchmark prompts ({BENCHMARK_PATH})...")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    bench_x = [item["prompt"] for item in bench_data]
    bench_y = [item["true_task_type"] for item in bench_data]

    bench_preds = pipeline.predict(bench_x)
    acc = accuracy_score(bench_y, bench_preds)
    f1 = f1_score(bench_y, bench_preds, average="macro")

    print("\n" + "=" * 70)
    print("MODEL VALIDATION ON 1,000 UNSEEN BENCHMARK PROMPTS")
    print("=" * 70)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Macro-F1 : {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(bench_y, bench_preds, digits=4))

    # Save trained pipeline
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nTrained task classifier saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_and_save()
