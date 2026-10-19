"""
Generator for 1,000 diverse, high-quality, ground-truth labeled benchmark prompts.
Spans 8 canonical task types (125 prompts each = 1,000 total):
  1. coding
  2. mathematical_reasoning
  3. extraction
  4. analysis
  5. planning
  6. classification
  7. research
  8. generation
"""

import json
import random
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data/benchmark_1000_prompts.json"

CODING_TEMPLATES = [
    # SQL
    "Write an SQL query to find {entity} where {condition}.",
    "Write a SQL query that extracts {entity} having {condition}.",
    "Construct a PostgreSQL query with window functions to calculate {metric} partitioned by {entity}.",
    "Optimize this slow MySQL query with joins across {entity1} and {entity2}.",
    "Write an SQL statement to delete duplicate records from the {entity} table keeping the latest.",
    # Python
    "Write a Python function using {library} to {action}.",
    "Implement an async Python script with FastAPI to stream {metric} from {system}.",
    "Debug this Python code that raises {error}: {snippet}",
    "Refactor this Python class to follow SOLID principles and add type hints.",
    "Write a Python generator that yields {entity} from a large {source}.",
    # Data Structures & Algorithms
    "Implement a LRU cache in {language} with O(1) get and put time complexity.",
    "Write a function in {language} to detect a cycle in a directed graph using Kahn's algorithm.",
    "Implement binary search on a rotated sorted array in {language}.",
    "Write a regex pattern in Python to match valid international {entity} formats.",
    "Solve the two-sum problem in {language} with optimal time complexity.",
    # Web & Systems
    "Create a React component in TypeScript that renders a {component} with debounced search.",
    "Write a bash script to monitor CPU and memory usage and send alerts if threshold exceeds {metric}.",
    "Write a Dockerfile and docker-compose.yml for a {tech_stack} application.",
    "Implement rate limiting middleware in Node.js using Redis token bucket algorithm.",
    "Write a Go service that handles graceful shutdown on SIGTERM.",
]

MATH_TEMPLATES = [
    # Algebra
    "Solve for x: {eq}.",
    "Solve the system of linear equations: {sys_eq}.",
    "Factor the quadratic polynomial {poly}.",
    "Find the roots of {poly} using the quadratic formula.",
    "Simplify the algebraic expression: {expr}.",
    # Calculus
    "Calculate the derivative of f(x) = {func}.",
    "Evaluate the definite integral of {func} from {a} to {b}.",
    "Find the local extrema and inflection points of f(x) = {func}.",
    "Compute the Taylor series expansion of {func} centered at x = 0 up to degree 4.",
    "Find the limit of {limit_expr} as x approaches {lim_val}.",
    # Probability & Statistics
    "Calculate the probability of drawing {prob_event}.",
    "Derive the maximum likelihood estimator for the parameter lambda of a Poisson distribution.",
    "Given a normal distribution with mean {mean} and standard deviation {std}, find P(X > {val}).",
    "Calculate Bayes' theorem posterior probability given P(A) = {p1}, P(B|A) = {p2}.",
    "Compute the expectation and variance of the discrete random variable {rand_var}.",
    # Proofs & Advanced Math
    "Prove by mathematical induction that {induction_claim}.",
    "Find the eigenvalues and eigenvectors of the 2x2 matrix {matrix}.",
    "Prove that the square root of {num} is irrational.",
    "Compute the dot product and cross product of vectors u = {vec1} and v = {vec2}.",
    "Solve the first-order differential equation dy/dx + {coeff}y = {rhs}.",
]

EXTRACTION_TEMPLATES = [
    "Extract all {entities} from the following text and format as a JSON array: {context}",
    "Parse the following invoice text and extract the invoice number, date, vendor, and total amount: {context}",
    "Extract email addresses and phone numbers from the following raw contact dump: {context}",
    "Pull out all chemical compound names and their molecular weights mentioned in this paragraph: {context}",
    "Extract the start date, end date, and job titles from the resume text below: {context}",
    "Extract all URLs and hyperlinks found in this HTML snippet: {context}",
    "Parse this CSV log and extract all rows with HTTP status code 500: {context}",
    "Extract key entities (Person, Organization, Location) from this news excerpt: {context}",
    "Extract the delivery address, zip code, and recipient name from this order confirmation: {context}",
    "Extract all financial figures, revenue growth percentages, and EBITDA margins from this earnings snippet: {context}",
]

ANALYSIS_TEMPLATES = [
    "Compare {tech1} versus {tech2} in terms of performance, scalability, and operational complexity.",
    "Analyze the trade-offs between {approach1} and {approach2} for distributed data storage.",
    "Explain why {phenomenon} occurs and discuss its implications for {domain}.",
    "Conduct a root cause analysis for {incident} in a microservices deployment.",
    "Analyze the pros and cons of using {pattern1} compared to {pattern2}.",
    "Evaluate the financial health of {company} based on its debt-to-equity ratio and free cash flow.",
    "Compare monolithic architecture with event-driven microservices architecture.",
    "Analyze the security vulnerabilities associated with {protocol} and explain how to mitigate them.",
    "Critique this system design proposal and identify potential single points of failure: {context}",
    "Compare convolutional neural networks (CNNs) with vision transformers (ViTs) for image classification.",
]

PLANNING_TEMPLATES = [
    "Create a step-by-step implementation plan for migrating a monolith to microservices on AWS.",
    "Design an architecture roadmap for building a high-throughput, fault-tolerant real-time payment gateway.",
    "Draft a 4-week project sprint schedule for delivering an MVP of an e-commerce platform.",
    "Create a disaster recovery plan for a multi-region PostgreSQL database cluster.",
    "Outline the sequential steps required to implement zero-trust network access in an enterprise.",
    "Develop a 90-day onboarding and technical upskilling plan for senior data engineers.",
    "Design a capacity planning strategy for handling 10x traffic spikes during Black Friday.",
    "Create a product launch checklist and milestone schedule for a B2B SaaS analytics tool.",
    "Plan the architecture and rollout phases for adopting Kubernetes across legacy servers.",
    "Draft a security audit and compliance remediation plan for SOC 2 Type II certification.",
]

CLASSIFICATION_TEMPLATES = [
    "Classify the sentiment of the following customer review as positive, negative, or neutral: '{text}'",
    "Categorize the following support ticket into billing, technical, account, or shipping: '{text}'",
    "Is the following news headline clickbait or factual? Classify and explain: '{text}'",
    "Classify whether the following email is spam or legitimate ham: '{text}'",
    "Assign the appropriate medical specialty (cardiology, neurology, oncology, pediatrics) to this clinical note: '{text}'",
    "Classify this financial transaction description as grocery, utility, travel, or entertainment: '{text}'",
    "Determine the intent of the following user chatbot query (informational, navigational, transactional): '{text}'",
    "Classify the severity of this bug report as P0 (critical), P1 (high), P2 (medium), or P3 (low): '{text}'",
    "Categorize the following legal clause as indemnity, confidentiality, termination, or governing law: '{text}'",
    "Label the toxicity level of the following social media comment (toxic, severely toxic, benign): '{text}'",
]

RESEARCH_TEMPLATES = [
    "Provide a comprehensive literature review on recent breakthroughs in quantum error correction.",
    "What is the current state of research regarding room-temperature superconductivity?",
    "Summarize the academic consensus and recent findings on the neurological effects of microplastics.",
    "Survey the published literature on transformer attention mechanism optimizations from 2023 to 2025.",
    "Synthesize recent peer-reviewed studies on mRNA vaccine platforms against autoimmune diseases.",
    "What are the leading empirical findings on the economic impact of universal basic income trials?",
    "Provide an overview of recent research into solid-state battery electrolytes and commercialization barriers.",
    "Summarize current academic research regarding dark matter candidates beyond WIMPs and axions.",
    "Survey the historical evolution and contemporary findings on behavioral nudging in public policy.",
    "Review recent scientific papers investigating cellular senescence reversal using senolytics.",
]

GENERATION_TEMPLATES = [
    "Write a polite email to a client explaining a 2-day delivery delay due to inclement weather.",
    "Draft a compelling 500-word blog post announcing the launch of our new AI productivity tool.",
    "Compose a creative science fiction short story about an astronaut discovering an abandoned outpost on Titan.",
    "Write an engaging LinkedIn post sharing 5 lessons learned from failing at my first startup.",
    "Draft a motivational speech for a graduating class of computer science students.",
    "Write a product description highlighting the benefits of an ergonomic mechanical keyboard.",
    "Compose a persuasive pitch deck script introducing a climate tech venture to angel investors.",
    "Draft a press release announcing a partnership between a fintech company and a major bank.",
    "Write a friendly welcome email for new newsletter subscribers offering a free onboarding guide.",
    "Draft a respectful follow-up message to a job interviewer inquiring about the status of an application.",
]

def generate_benchmark(n_per_category: int = 125, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    benchmark = []

    categories = [
        ("coding", CODING_TEMPLATES, "super"),
        ("mathematical_reasoning", MATH_TEMPLATES, "super"),
        ("extraction", EXTRACTION_TEMPLATES, "nano"),
        ("analysis", ANALYSIS_TEMPLATES, "super"),
        ("planning", PLANNING_TEMPLATES, "ultra"),
        ("classification", CLASSIFICATION_TEMPLATES, "nano"),
        ("research", RESEARCH_TEMPLATES, "super"),
        ("generation", GENERATION_TEMPLATES, "nano"),
    ]

    fillers = {
        "entity": ["employees", "customers", "orders", "transactions", "inventory_items", "sensor_logs", "patient_records"],
        "entity1": ["users", "accounts", "customers", "students"],
        "entity2": ["orders", "transactions", "invoices", "enrollments"],
        "condition": [
            "salary is higher than the department average",
            "created_at is within the last 30 days and status = 'completed'",
            "manager_id is NULL or manager salary is less than employee salary",
            "total_spend > 5000 and country = 'US'",
            "inventory_count < reorder_threshold",
        ],
        "metric": ["moving 7-day average revenue", "percentile rank", "cumulative sum of sales", "lagged price"],
        "library": ["pandas", "numpy", "matplotlib", "asyncio", "httpx", "sqlalchemy", "scipy"],
        "action": [
            "load parquet files, filter null values, and compute group statistics",
            "fetch JSON from a REST API with exponential backoff and save to SQLite",
            "parse an XML sitemap and extract all canonical links",
            "calculate the rolling Sharpe ratio over daily asset returns",
            "encrypt and decrypt sensitive credentials using AES-GCM",
        ],
        "language": ["Python", "Rust", "Go", "C++", "TypeScript", "Java"],
        "error": ["IndexError: list index out of range", "KeyError: 'user_id'", "Deadlock detected in transaction"],
        "snippet": ["def get_val(data):\n    return [d['key'] for d in data if d]", "for i in range(len(arr)):\n    arr.pop(i)"],
        "system": ["Kafka", "RabbitMQ", "Redis pub/sub", "WebSocket server"],
        "source": ["compressed tar archive", "multi-gigabyte NDJSON file", "Postgres stream"],
        "component": ["virtualized data grid table", "dynamic autocomplete dropdown", "paginated file tree viewer"],
        "tech_stack": ["FastAPI, PostgreSQL, and Redis", "Next.js frontend and Node backend", "Flask and Celery worker"],
        # Math fillers
        "eq": [
            "3x^2 - 12x + 9 = 0",
            "2x^2 + 5x - 3 = 0",
            "x^3 - 4x^2 + x + 6 = 0",
            "4x - 7 = 3x + 15",
            "e^(2x) - 5e^x + 6 = 0",
            "ln(2x - 1) = 3",
            "cos(2x) = 1/2 for x in [0, 2pi]",
            "sqrt(2x + 5) = x - 1",
            "x^4 - 5x^2 + 4 = 0",
            "5^(x-1) = 125",
        ],
        "sys_eq": [
            "2x + 3y = 12 and x - 2y = -1",
            "3x + 4y - z = 8, 2x - y + 3z = 7, and x + 2y + 2z = 9",
            "5x - 2y = 10 and 2x + 5y = 33",
        ],
        "poly": ["2x^2 + 7x + 3", "x^3 - 6x^2 + 11x - 6", "4x^2 - 25", "x^2 - 10x + 21"],
        "expr": ["(x^2 - 9)/(x + 3)", "sqrt(50) + sqrt(18) - sqrt(8)", "(2a^3 b^-2)^3 / (4a^-1 b^4)"],
        "func": ["3x^4 - 5x^2 + 7x - 2", "ln(x^2 + 1)", "e^(3x) * sin(2x)", "x / (x^2 + 1)", "sqrt(4 - x^2)"],
        "a": ["0", "1", "-1", "2"],
        "b": ["1", "4", "pi", "e"],
        "limit_expr": ["(sin(3x)) / x", "(x^2 - 4) / (x - 2)", "(sqrt(x + 1) - 1) / x"],
        "lim_val": ["0", "2", "infinity"],
        "prob_event": [
            "two aces in a row from a standard 52-card deck without replacement",
            "at least 3 heads in 5 fair coin flips",
            "a sum of 8 or higher when rolling two fair 6-sided dice",
        ],
        "mean": ["100", "50", "0"],
        "std": ["15", "10", "1"],
        "val": ["115", "65", "1.96"],
        "p1": ["0.01", "0.05", "0.10"],
        "p2": ["0.95", "0.80", "0.99"],
        "rand_var": ["X with P(X=1)=0.2, P(X=2)=0.5, P(X=3)=0.3", "Y ~ Binomial(n=10, p=0.4)"],
        "induction_claim": [
            "1 + 2 + ... + n = n(n+1)/2 for all n >= 1",
            "1^2 + 2^2 + ... + n^2 = n(n+1)(2n+1)/6 for all n >= 1",
            "2^n > n for all positive integers n",
        ],
        "matrix": ["[[2, 1], [1, 2]]", "[[4, 2], [1, 3]]", "[[3, 0], [0, -2]]"],
        "num": ["2", "3", "7"],
        "vec1": ["(1, 2, 3)", "(2, -1, 4)", "(0, 3, -2)"],
        "vec2": ["(4, -1, 2)", "(1, 2, 0)", "(-3, 1, 5)"],
        "coeff": ["2", "3", "-1"],
        "rhs": ["e^(3x)", "x^2", "cos(x)"],
        # Text/Context fillers
        "entities": ["dates and amounts", "SKU codes and prices", "flight numbers and arrival times"],
        "context": [
            "John Doe (john@acme.corp) submitted report on 2026-04-12 regarding $4,500 server costs.",
            "Invoice #INV-2026-889 issued by CloudCorp on March 15, 2026 for $12,450.00 due net 30.",
            "Patient presented on 2026-03-01 with acute bronchitis; prescribed Amoxicillin 500mg TID.",
        ],
        "tech1": ["PostgreSQL", "Apache Kafka", "Redis", "gRPC", "Docker", "Kubernetes"],
        "tech2": ["MongoDB", "RabbitMQ", "Memcached", "REST", "Podman", "Nomad"],
        "approach1": ["eventual consistency", "optimistic locking", "single-leader replication"],
        "approach2": ["strong consistency", "pessimistic locking", "multi-leader replication"],
        "phenomenon": ["the vanishing gradient problem", "cache thrashing", "split-brain in distributed consensus"],
        "domain": ["deep neural network convergence", "high-frequency trading architectures", "cloud storage availability"],
        "incident": ["cascading connection pool exhaustion", "memory leak under sustained load", "database replication lag"],
        "pattern1": ["microservices", "CQRS", "serverless lambda functions"],
        "pattern2": ["modular monolith", "traditional CRUD", "long-running container services"],
        "company": ["Tesla Inc.", "Apple Inc.", "Amazon.com", "Microsoft Corp."],
        "protocol": ["OAuth 2.0 PKCE", "JWT token storage", "mTLS handshake"],
        "text": [
            "The product arrived broken and customer support refused to refund my money after 3 weeks.",
            "Absolutely magnificent experience! The UI is slick, response times are lightning fast, highly recommend.",
            "I need to update my credit card on file for subscription renewal #99201.",
            "URGENT: Click here to claim your $1,000 lottery voucher before it expires in 10 minutes!",
            "Patient demonstrates normal sinus rhythm with no ST elevation or signs of myocardial infarction.",
        ],
    }

    prompt_id = 1
    for task_type, templates, default_tier in categories:
        for i in range(n_per_category):
            tmpl = templates[i % len(templates)]
            # Format template with fillers
            formatted = tmpl
            for key, val_list in fillers.items():
                token = "{" + key + "}"
                while token in formatted:
                    chosen = rng.choice(val_list)
                    formatted = formatted.replace(token, chosen, 1)

            # Assign expected tier based on task complexity
            tier = default_tier
            if task_type in ("coding", "mathematical_reasoning"):
                if len(formatted.split()) > 30 or "architecture" in formatted.lower() or "proof" in formatted.lower() or "differential" in formatted.lower():
                    tier = "ultra"
                elif len(formatted.split()) > 15:
                    tier = "super"
                else:
                    tier = "nano"
            elif task_type in ("extraction", "classification"):
                tier = "nano"
            elif task_type in ("planning",):
                tier = "ultra" if "disaster" in formatted.lower() or "enterprise" in formatted.lower() else "super"
            elif task_type in ("analysis", "research"):
                tier = "ultra" if "quantum" in formatted.lower() or "senolytics" in formatted.lower() or "distributed" in formatted.lower() else "super"
            elif task_type in ("generation",):
                tier = "super" if len(formatted.split()) > 40 else "nano"

            benchmark.append({
                "id": f"prompt_{prompt_id:04d}",
                "prompt": formatted,
                "true_task_type": task_type,
                "true_tier": tier,
            })
            prompt_id += 1

    rng.shuffle(benchmark)
    return benchmark

if __name__ == "__main__":
    bm = generate_benchmark(n_per_category=125)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(bm, f, indent=2)
    print(f"Generated {len(bm)} benchmark prompts saved to: {OUTPUT_PATH}")
