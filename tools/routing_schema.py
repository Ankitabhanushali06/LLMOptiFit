"""
Schema and Categorical Variable Extractor for LLMOptiFit.

Defines the 11 task types, the 15-variable minimum routing schema, and
heuristic/rule-based extractors to generate categorical and ordinal
features from user prompts for downstream model training and routing.
"""

from __future__ import annotations

import enum
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# =====================================================================
# 1. Enums for Categorical Variables
# =====================================================================

class TaskType(str, enum.Enum):
    SQL_DATA = "sql_data"
    CODE = "code"
    MATH_REASONING = "math_reasoning"
    FACTUAL_QA = "factual_qa"
    SUMMARIZE = "summarize"
    REWRITE_TRANSLATE = "rewrite_translate"
    EXTRACT_TRANSFORM = "extract_transform"
    CLASSIFY = "classify"
    PLANNING_TOOL_USE = "planning_tool_use"
    CREATIVE_GENERATION = "creative_generation"
    GENERAL_CHAT = "general_chat"


TASK_TYPE_METADATA: dict[TaskType, dict[str, str]] = {
    TaskType.SQL_DATA: {
        "description": "Write, debug, optimize, explain, or transform SQL; database, schema, ETL, data engineering, analytics, Snowflake, Spark, Pandas, or tabular-data work.",
        "recommended_model_capability": "coding + structured-output",
        "primary_evaluation": "SQL parsing, execution, schema-reference validation",
    },
    TaskType.CODE: {
        "description": "Write, debug, refactor, explain, test, review, or optimize source code.",
        "recommended_model_capability": "coding",
        "primary_evaluation": "compile, unit tests, static analysis",
    },
    TaskType.MATH_REASONING: {
        "description": "Arithmetic, algebra, probability, logic puzzles, algorithmic reasoning, or multi-step quantitative work.",
        "recommended_model_capability": "reasoning",
        "primary_evaluation": "exact answer, numeric tolerance, symbolic validation",
    },
    TaskType.FACTUAL_QA: {
        "description": "Questions whose answer depends on real-world facts, current events, research, products, places, laws, or named entities.",
        "recommended_model_capability": "retrieval-grounded",
        "primary_evaluation": "citation support and factual-grounding score",
    },
    TaskType.SUMMARIZE: {
        "description": "Condense a supplied article, document, transcript, meeting notes, code, or conversation without adding information.",
        "recommended_model_capability": "long-context",
        "primary_evaluation": "faithfulness, coverage, requested length",
    },
    TaskType.REWRITE_TRANSLATE: {
        "description": "Rewrite, proofread, paraphrase, change tone, simplify, translate, or adapt content for an audience.",
        "recommended_model_capability": "language-generation",
        "primary_evaluation": "meaning preservation and instruction following",
    },
    TaskType.EXTRACT_TRANSFORM: {
        "description": "Extract named fields, entities, tables, dates, facts, or convert text into JSON, CSV, XML, Markdown, or another specified schema.",
        "recommended_model_capability": "structured-output",
        "primary_evaluation": "schema validation, exact field match",
    },
    TaskType.CLASSIFY: {
        "description": "Assign a fixed label, sentiment, topic, risk level, priority, intent, or category to supplied input.",
        "recommended_model_capability": "fast-classification",
        "primary_evaluation": "label accuracy, F1, precision/recall",
    },
    TaskType.PLANNING_TOOL_USE: {
        "description": "Create a plan, choose tools, orchestrate steps, schedule work, query a database, call an API, or complete multi-step agent work.",
        "recommended_model_capability": "tool-calling + reasoning",
        "primary_evaluation": "tool success rate and final-goal completion",
    },
    TaskType.CREATIVE_GENERATION: {
        "description": "Write stories, poems, marketing copy, scripts, social-media posts, role-play, ideas, or brainstorming output.",
        "recommended_model_capability": "creative-writing",
        "primary_evaluation": "preference score, style adherence",
    },
    TaskType.GENERAL_CHAT: {
        "description": "Casual conversation, explanations, opinions, personal advice, greetings, and prompts that do not fit another task class.",
        "recommended_model_capability": "general-instruction",
        "primary_evaluation": "helpfulness and relevance",
    },
}


class Language(str, enum.Enum):
    EN = "en"
    ZH = "zh"
    ES = "es"
    HI = "hi"
    FR = "fr"
    DE = "de"
    JA = "ja"
    OTHER = "other"


class Modality(str, enum.Enum):
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    IMAGE = "image"
    DOCUMENT = "document"


class ContextCompleteness(str, enum.Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"


class OutputContract(str, enum.Enum):
    SCHEMA = "schema"
    FORMAT_ONLY = "format_only"
    NONE = "none"


class ConstraintCompleteness(str, enum.Enum):
    EXPLICIT_BOUNDED = "explicit_bounded"
    PARTIAL = "partial"
    NONE = "none"


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    REGULATED = "regulated"


class PrivacyRequirement(str, enum.Enum):
    NONE = "none"
    SENSITIVE = "sensitive"
    LOCAL_ONLY = "local_only"


class UrgencyLatencyTier(str, enum.Enum):
    REALTIME = "realtime"
    INTERACTIVE = "interactive"
    BATCH = "batch"


# =====================================================================
# 2. Minimum Routing Schema & Feature Container
# =====================================================================

@dataclass
class PromptRoutingFeatures:
    """Complete 15-variable feature representation of a user prompt."""
    task_type: TaskType
    language: Language
    modality: Modality
    complexity_score: int          # 1 to 5
    ambiguity_score: int           # 1 to 5
    context_completeness: ContextCompleteness
    output_contract: OutputContract
    constraint_completeness: ConstraintCompleteness
    needs_retrieval: bool
    needs_tools: bool
    risk_level: RiskLevel
    privacy_requirement: PrivacyRequirement
    urgency_latency_tier: UrgencyLatencyTier
    estimated_input_tokens: int
    estimated_output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "language": self.language.value,
            "modality": self.modality.value,
            "complexity_score": int(self.complexity_score),
            "ambiguity_score": int(self.ambiguity_score),
            "context_completeness": self.context_completeness.value,
            "output_contract": self.output_contract.value,
            "constraint_completeness": self.constraint_completeness.value,
            "needs_retrieval": bool(self.needs_retrieval),
            "needs_tools": bool(self.needs_tools),
            "risk_level": self.risk_level.value,
            "privacy_requirement": self.privacy_requirement.value,
            "urgency_latency_tier": self.urgency_latency_tier.value,
            "estimated_input_tokens": int(self.estimated_input_tokens),
            "estimated_output_tokens": int(self.estimated_output_tokens),
        }


# =====================================================================
# 3. Categorical & Variable Feature Extractor from Raw Text
# =====================================================================

class PromptFeatureExtractor:
    """Extracts the 15 categorical/numerical variables from raw prompt text."""

    def __init__(self) -> None:
        self._init_regex_patterns()

    def _init_regex_patterns(self) -> None:
        self.sql_data_re = re.compile(r"\b(select|from|where|join|group by|order by|create table|drop table|alter table|sql|sqlite|postgresql|postgres|mysql|snowflake|bigquery|spark|pyspark|pandas|dataframe|etl|database|schema)\b", re.I)
        self.code_re = re.compile(r"(\bdef\s+\w+|\bclass\s+\w+|\bfunction\b|\bimport\s+\w+|=>|[{};]\s*$|```[a-z]*|#include|console\.log|println!|\bhtml\b|\bcss\b|\bjavascript\b|\bpython\b|\bc\+\+\b|\bjava\b|\brust\b|\bgolang\b)", re.I | re.M)
        self.math_re = re.compile(r"(\bcalculate\b|\bsolve\b|\bequation\b|\bprobability\b|\bderivative\b|\bintegral\b|\bmatrix\b|\balgebra\b|\barithmetic\b|\bformula\b|\btheorem\b|\bproof\b|[0-9]+\s*[\+\-\*\/\^=]\s*[0-9]+)", re.I)
        self.factual_re = re.compile(r"\b(what is the capital|who was the president|when was|where is|history of|explain how|what causes|definition of|current price|latest news|released in)\b", re.I)
        self.summarize_re = re.compile(r"\b(summarize|summary|tldr|tl;dr|condense|digest|briefly explain the text below|key takeaways|bullet points from this)\b", re.I)
        self.rewrite_re = re.compile(r"\b(rewrite|rephrase|paraphrase|proofread|correct grammar|translate|translation|change tone|make it more professional|simplify)\b", re.I)
        self.extract_re = re.compile(r"\b(extract|parse|convert to json|return as csv|json schema|markdown table|yaml format|extract entities|extract fields|regex)\b", re.I)
        self.classify_re = re.compile(r"\b(classify|categorize|sentiment|label this|is this spam|positive or negative|rate the severity|assign a tag)\b", re.I)
        self.planning_re = re.compile(r"\b(step by step plan|roadmap|orchestrate|call the api|search for|agent workflow|multi-step|schedule|automate)\b", re.I)
        self.creative_re = re.compile(r"\b(write a story|poem|marketing copy|haiku|screenplay|fiction|creative|song lyrics|blog post about|roleplay|pretend you are)\b", re.I)

        # Output contract patterns
        self.schema_re = re.compile(r"\b(json|xml|yaml|schema|strict json|json format|pydantic|table with columns)\b", re.I)
        self.format_re = re.compile(r"\b(bullet points|numbered list|markdown|table|csv|short paragraph|max \d+ words)\b", re.I)

        # Constraint patterns
        self.constraints_re = re.compile(r"\b(only|must|do not|never|exclude|limit to|maximum|minimum|under \d+ words|strictly|within \d+)\b", re.I)

        # Retrieval / Tools patterns
        self.retrieval_re = re.compile(r"\b(current|recent|today|latest|news|stock price|weather|lookup|search online|sources|cite|references)\b", re.I)
        self.tools_re = re.compile(r"\b(run python|execute|call api|search web|browse|query database|tool|function call|calculate)\b", re.I)

        # Risk & Privacy patterns
        self.risk_re = re.compile(r"\b(medical advice|diagnosis|dosage|legal contract|exploit|vulnerability|hack|bypass|confidential|ssn|credit card)\b", re.I)
        self.privacy_re = re.compile(r"\b(private data|pii|confidential|internal only|proprietary|gdpr|hipaa|nda|local only|do not share)\b", re.I)

    def extract(self, prompt: str, language: str = "en") -> PromptRoutingFeatures:
        p = prompt.strip()
        tokens_est = max(1, len(p) // 4)

        # 1. Task Type Classification
        task_type = self._detect_task_type(p)

        # 2. Modality
        if "```" in p or "def " in p or "class " in p or "select " in p.lower():
            modality = Modality.CODE
        elif "|" in p and "-|-" in p:
            modality = Modality.TABLE
        elif tokens_est > 1500:
            modality = Modality.DOCUMENT
        else:
            modality = Modality.TEXT

        # 3. Complexity Score (1 to 5)
        complexity = self._estimate_complexity(p, task_type, tokens_est)

        # 4. Ambiguity Score (1 to 5)
        ambiguity = self._estimate_ambiguity(p)

        # 5. Context Completeness
        if tokens_est < 10 and any(w in p.lower() for w in ["fix this", "explain this", "summarize", "it", "why did that happen"]):
            context_completeness = ContextCompleteness.MISSING
        elif tokens_est < 25 and ("refer to" in p.lower() or "following text" in p.lower() and ":" not in p):
            context_completeness = ContextCompleteness.PARTIAL
        else:
            context_completeness = ContextCompleteness.COMPLETE

        # 6. Output Contract
        if self.schema_re.search(p):
            output_contract = OutputContract.SCHEMA
        elif self.format_re.search(p):
            output_contract = OutputContract.FORMAT_ONLY
        else:
            output_contract = OutputContract.NONE

        # 7. Constraint Completeness
        constraint_hits = len(self.constraints_re.findall(p))
        if constraint_hits >= 2:
            constraint_completeness = ConstraintCompleteness.EXPLICIT_BOUNDED
        elif constraint_hits == 1:
            constraint_completeness = ConstraintCompleteness.PARTIAL
        else:
            constraint_completeness = ConstraintCompleteness.NONE

        # 8. Needs Retrieval & Tools
        needs_retrieval = bool(self.retrieval_re.search(p)) or (task_type == TaskType.FACTUAL_QA and ("latest" in p.lower() or "2024" in p or "2025" in p or "2026" in p))
        needs_tools = bool(self.tools_re.search(p)) or task_type == TaskType.PLANNING_TOOL_USE

        # 9. Risk Level & Privacy
        risk_hits = len(self.risk_re.findall(p))
        if risk_hits >= 2:
            risk_level = RiskLevel.REGULATED if any(k in p.lower() for k in ["medical", "dosage", "legal"]) else RiskLevel.HIGH
        elif risk_hits == 1:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        if "local only" in p.lower() or "offline only" in p.lower():
            privacy = PrivacyRequirement.LOCAL_ONLY
        elif self.privacy_re.search(p):
            privacy = PrivacyRequirement.SENSITIVE
        else:
            privacy = PrivacyRequirement.NONE

        # 10. Urgency Latency Tier
        if "realtime" in p.lower() or "sub-second" in p.lower() or "instant" in p.lower() or task_type == TaskType.CLASSIFY:
            urgency = UrgencyLatencyTier.REALTIME
        elif tokens_est > 3000 or task_type in [TaskType.PLANNING_TOOL_USE, TaskType.SUMMARIZE]:
            urgency = UrgencyLatencyTier.BATCH
        else:
            urgency = UrgencyLatencyTier.INTERACTIVE

        # 11. Estimated Output Tokens
        if output_contract == OutputContract.SCHEMA:
            est_output = min(2048, max(128, tokens_est // 2))
        elif task_type in [TaskType.SUMMARIZE, TaskType.CLASSIFY]:
            est_output = min(512, max(64, tokens_est // 4))
        elif task_type in [TaskType.CODE, TaskType.CREATIVE_GENERATION, TaskType.PLANNING_TOOL_USE]:
            est_output = min(4096, max(384, tokens_est * 2))
        else:
            est_output = min(1024, max(128, tokens_est))

        lang_val = Language.EN if language.lower().startswith("en") else Language.OTHER

        return PromptRoutingFeatures(
            task_type=task_type,
            language=lang_val,
            modality=modality,
            complexity_score=complexity,
            ambiguity_score=ambiguity,
            context_completeness=context_completeness,
            output_contract=output_contract,
            constraint_completeness=constraint_completeness,
            needs_retrieval=needs_retrieval,
            needs_tools=needs_tools,
            risk_level=risk_level,
            privacy_requirement=privacy,
            urgency_latency_tier=urgency,
            estimated_input_tokens=tokens_est,
            estimated_output_tokens=est_output,
        )

    def _detect_task_type(self, p: str) -> TaskType:
        if self.sql_data_re.search(p):
            return TaskType.SQL_DATA
        if self.extract_re.search(p):
            return TaskType.EXTRACT_TRANSFORM
        if self.classify_re.search(p):
            return TaskType.CLASSIFY
        if self.code_re.search(p):
            return TaskType.CODE
        if self.math_re.search(p):
            return TaskType.MATH_REASONING
        if self.summarize_re.search(p):
            return TaskType.SUMMARIZE
        if self.rewrite_re.search(p):
            return TaskType.REWRITE_TRANSLATE
        if self.planning_re.search(p):
            return TaskType.PLANNING_TOOL_USE
        if self.creative_re.search(p):
            return TaskType.CREATIVE_GENERATION
        if self.factual_re.search(p) or p.strip().endswith("?"):
            return TaskType.FACTUAL_QA
        return TaskType.GENERAL_CHAT

    def _estimate_complexity(self, p: str, task_type: TaskType, tokens: int) -> int:
        score = 1
        if tokens > 50:
            score += 1
        if tokens > 250:
            score += 1
        if task_type in [TaskType.SQL_DATA, TaskType.CODE, TaskType.MATH_REASONING, TaskType.PLANNING_TOOL_USE]:
            score += 1
        if any(w in p.lower() for w in ["optimize", "architecture", "tradeoff", "multi-step", "asynchronous", "distributed", "concurrency", "rigorous"]):
            score += 1
        return min(5, max(1, score))

    def _estimate_ambiguity(self, p: str) -> int:
        length = len(p.split())
        if length <= 3:
            return 5
        if length <= 6:
            return 4
        if "?" in p and length < 12 and not any(k in p.lower() for k in ["specifically", "given that", "assuming"]):
            return 3
        if length > 25:
            return 1
        return 2


# =====================================================================
# 4. Scikit-Learn Encoding Pipeline for ML Models
# =====================================================================

def build_routing_feature_preprocessor() -> ColumnTransformer:
    """Creates a Scikit-Learn ColumnTransformer for the categorical and numerical schema."""
    nominal_categorical_cols = [
        "task_type",
        "language",
        "modality",
        "context_completeness",
        "output_contract",
        "constraint_completeness",
        "privacy_requirement",
        "urgency_latency_tier",
    ]

    ordinal_cols = [
        "risk_level",
    ]
    risk_categories = [["low", "medium", "high", "regulated"]]

    numerical_cols = [
        "complexity_score",
        "ambiguity_score",
        "needs_retrieval",
        "needs_tools",
        "estimated_input_tokens",
        "estimated_output_tokens",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("nom_cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_categorical_cols),
            ("ord_cat", OrdinalEncoder(categories=risk_categories, handle_unknown="use_encoded_value", unknown_value=-1), ordinal_cols),
            ("num", StandardScaler(), numerical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def extract_features_df(prompts: List[str], languages: Optional[List[str]] = None) -> pd.DataFrame:
    """Batch extracts 15 categorical/numerical variables into a structured pandas DataFrame."""
    extractor = PromptFeatureExtractor()
    records: List[dict[str, Any]] = []

    if languages is None:
        languages = ["en"] * len(prompts)

    for prompt, lang in zip(prompts, languages):
        feat = extractor.extract(str(prompt or ""), language=str(lang or "en"))
        records.append(feat.to_dict())

    return pd.DataFrame.from_records(records)
