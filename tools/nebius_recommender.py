"""
Nebius Token Factory Model Recommendation Engine for LLMOptiFit.

Integrates the live Nebius Token Factory model catalog (https://tokenfactory.nebius.com/models/catalog)
with the prompt parameter classification pipeline.

Calculates exact dollar costs, estimated latency, task-specific quality, and
identifies the Pareto-optimal model candidate for any incoming prompt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

CATALOG_PATH = Path(__file__).resolve().parent.parent / "config/nebius_catalog.json"
FALLBACK_FRONTIER_ID = "NousResearch/Hermes-4-405B"


@dataclass
class NebiusModelCandidate:
    model_id: str
    name: str
    vendor: str
    tier: str  # nano, super, ultra
    size_b: float
    quality_score: float
    context_window_tokens: int
    input_price_per_million: float
    output_price_per_million: float
    tokens_per_second: float
    supports_json: bool
    supports_tools: bool
    is_nvidia: bool
    task_affinities: list[str]
    description: str


@dataclass
class ModelEvaluation:
    model: NebiusModelCandidate
    cost_usd: float
    estimated_latency_sec: float
    quality_match: float
    utility_score: float
    is_pareto_optimal: bool = False


@dataclass
class RecommendationResult:
    prompt_preview: str
    recommended_model_id: str
    model_name: str
    vendor: str
    tier: str
    is_nvidia: bool
    estimated_cost_usd: float
    estimated_latency_sec: float
    quality_score: float
    cost_savings_vs_frontier_pct: float
    cost_savings_usd: float
    rationale: str
    top_pareto_candidates: list[dict[str, Any]] = field(default_factory=list)
    all_candidates: list[dict[str, Any]] = field(default_factory=list)


class NebiusModelCatalog:
    """Loads, caches, and queries models from the Nebius Token Factory catalog."""

    def __init__(self, catalog_path: Path = CATALOG_PATH):
        self.catalog_path = Path(catalog_path)
        self.models: list[NebiusModelCandidate] = []
        self._load_catalog()

    def _load_catalog(self):
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Nebius catalog file not found: {self.catalog_path}")

        with open(self.catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.models = [NebiusModelCandidate(**item) for item in data]

    def get_model(self, model_id: str) -> Optional[NebiusModelCandidate]:
        for m in self.models:
            if m.model_id == model_id:
                return m
        return None

    def filter_candidates(
        self,
        min_context: int = 4000,
        needs_json: bool = False,
        preferred_tier: Optional[str] = None,
    ) -> list[NebiusModelCandidate]:
        candidates = []
        for m in self.models:
            if m.context_window_tokens < min_context:
                continue
            if needs_json and not m.supports_json:
                continue
            candidates.append(m)
        return candidates


class NebiusParetoRecommender:
    """
    Computes real-dollar costs, task affinities, and the Pareto frontier to
    recommend the optimal Nebius Token Factory model for a given prompt.
    """

    def __init__(self, catalog: Optional[NebiusModelCatalog] = None):
        self.catalog = catalog or NebiusModelCatalog()

    @staticmethod
    def _estimate_tokens(prompt_text: str, task_type: str) -> tuple[int, int]:
        """Estimate prompt input tokens and expected generation output tokens."""
        words = len(prompt_text.split())
        in_tokens = max(10, int(words * 1.33))

        # Expected output tokens based on canonical task type
        task_out_tokens = {
            "coding": 450,
            "mathematical_reasoning": 350,
            "analysis": 450,
            "planning": 500,
            "research": 400,
            "generation": 350,
            "extraction": 120,
            "classification": 80,
        }
        out_tokens = task_out_tokens.get(task_type.lower(), 300)
        return in_tokens, out_tokens

    def recommend(
        self,
        prompt_text: str,
        task_type: str = "generation",
        complexity_score: float = 0.20,
        route_tier: str = "nano",
        specificity_score: float = 0.25,
        cost_sensitivity: float = 0.50,
        latency_sensitivity: float = 0.50,
        has_json_request: bool = False,
        prefer_nvidia: bool = True,
        frontier_benchmark_id: str = FALLBACK_FRONTIER_ID,
    ) -> RecommendationResult:
        in_tokens, out_tokens = self._estimate_tokens(prompt_text, task_type)
        total_tokens = in_tokens + out_tokens

        candidates = self.catalog.filter_candidates(
            min_context=total_tokens + 500,
            needs_json=has_json_request,
        )
        if not candidates:
            # Fallback to all models if filtering was too strict
            candidates = self.catalog.models

        evaluations: list[ModelEvaluation] = []
        for m in candidates:
            # 1. Real Dollar Cost Calculation
            cost_usd = (
                in_tokens * m.input_price_per_million
                + out_tokens * m.output_price_per_million
            ) / 1_000_000.0

            # 2. Latency Estimation (TTFT base + output throughput)
            latency_sec = 0.20 + (out_tokens / max(10.0, m.tokens_per_second))

            # 3. Prompt-Aligned Quality Score Q(m, prompt)
            q = float(m.quality_score)

            # Tier Alignment Logic
            tier_rank = {"nano": 1, "super": 2, "ultra": 3}
            target_rank = tier_rank.get(route_tier.lower(), 1)
            model_rank = tier_rank.get(m.tier.lower(), 1)

            if model_rank == target_rank:
                q += 15.0  # Perfect tier match reward
            elif model_rank > target_rank:
                # Over-provisioning penalty: discourage using expensive tiers on simple prompts
                q -= 14.0 * (model_rank - target_rank)
            else:
                # Under-provisioning penalty: severely penalize weak models on hard tasks
                q -= 30.0 * (target_rank - model_rank)

            # Task Affinity Bonus
            if task_type.lower() in [a.lower() for a in m.task_affinities]:
                q += 7.0

            # Specificity Synergy: if prompt is crystal clear, cheaper models get a bonus
            if specificity_score >= 0.55 and m.tier in ["nano", "super"]:
                q += 4.0

            # NVIDIA Open-Source Hackathon Preference
            if prefer_nvidia and m.is_nvidia:
                q += 4.0

            # 4. Utility Score (Balancing Quality vs Cost vs Latency)
            # Normalize cost and latency penalties
            norm_cost = cost_usd * 1000.0  # scale to comparable unit
            norm_latency = latency_sec / 2.0

            c_weight = 10.0 * (1.0 + cost_sensitivity)
            l_weight = 3.0 * (1.0 + latency_sensitivity)

            utility = q - (c_weight * norm_cost) - (l_weight * norm_latency)

            evaluations.append(
                ModelEvaluation(
                    model=m,
                    cost_usd=cost_usd,
                    estimated_latency_sec=latency_sec,
                    quality_match=q,
                    utility_score=utility,
                )
            )

        # 5. Pareto Dominance Filtering
        # Model A dominates Model B if A has lower/equal cost AND higher/equal quality
        for ev in evaluations:
            is_dominated = False
            for other in evaluations:
                if other.model.model_id == ev.model.model_id:
                    continue
                if (
                    other.cost_usd <= ev.cost_usd
                    and other.quality_match >= ev.quality_match
                    and (other.cost_usd < ev.cost_usd or other.quality_match > ev.quality_match)
                ):
                    is_dominated = True
                    break
            ev.is_pareto_optimal = not is_dominated

        # Strict Pareto Frontier Selection:
        # The recommended model must be strictly on the Pareto Optimal frontier.
        pareto_candidates = [ev for ev in evaluations if ev.is_pareto_optimal]
        if not pareto_candidates:
            pareto_candidates = evaluations

        # Sort Pareto candidates by Utility Score to select the single best model
        pareto_candidates.sort(key=lambda x: x.utility_score, reverse=True)
        winner_eval = pareto_candidates[0]
        winner = winner_eval.model


        # Calculate Savings vs Benchmark Frontier Model
        frontier_model = self.catalog.get_model(frontier_benchmark_id) or candidates[-1]
        frontier_cost = (
            in_tokens * frontier_model.input_price_per_million
            + out_tokens * frontier_model.output_price_per_million
        ) / 1_000_000.0

        savings_usd = max(0.0, frontier_cost - winner_eval.cost_usd)
        savings_pct = (savings_usd / frontier_cost * 100.0) if frontier_cost > 0 else 0.0

        # Rationale Construction
        preview = (prompt_text[:70] + "...") if len(prompt_text) > 70 else prompt_text
        nvidia_tag = " (NVIDIA Open-Source)" if winner.is_nvidia else ""
        rationale = (
            f"Prompt matches {route_tier.upper()} tier ({task_type}, complexity={complexity_score:.2f}). "
            f"Selected {winner.name}{nvidia_tag} as the Pareto-optimal candidate: "
            f"delivers {winner_eval.quality_match:.1f}/100 quality fit at ${winner_eval.cost_usd:.6f} "
            f"({savings_pct:.1f}% cheaper than frontier {frontier_model.name})."
        )

        sorted_pareto = sorted(pareto_candidates, key=lambda x: x.utility_score, reverse=True)
        top_candidates = [
            {
                "model_id": ev.model.model_id,
                "name": ev.model.name,
                "vendor": ev.model.vendor,
                "tier": ev.model.tier,
                "is_nvidia": ev.model.is_nvidia,
                "cost_usd": round(ev.cost_usd, 6),
                "latency_sec": round(ev.estimated_latency_sec, 2),
                "quality_match": round(ev.quality_match, 1),
                "utility": round(ev.utility_score, 2),
                "is_pareto_optimal": ev.is_pareto_optimal,
            }
            for ev in sorted_pareto[:5]
        ]

        all_candidate_dicts = [
            {
                "model_id": ev.model.model_id,
                "name": ev.model.name,
                "vendor": ev.model.vendor,
                "tier": ev.model.tier,
                "is_nvidia": ev.model.is_nvidia,
                "cost_usd": round(ev.cost_usd, 6),
                "latency_sec": round(ev.estimated_latency_sec, 2),
                "quality_match": round(ev.quality_match, 1),
                "utility": round(ev.utility_score, 2),
                "is_pareto_optimal": ev.is_pareto_optimal,
                "is_winner": (ev.model.model_id == winner.model_id),
                "context_window_tokens": ev.model.context_window_tokens,
                "input_price_per_million": ev.model.input_price_per_million,
                "output_price_per_million": ev.model.output_price_per_million,
            }
            for ev in sorted(evaluations, key=lambda x: x.cost_usd)
        ]

        return RecommendationResult(
            prompt_preview=preview,
            recommended_model_id=winner.model_id,
            model_name=winner.name,
            vendor=winner.vendor,
            tier=winner.tier,
            is_nvidia=winner.is_nvidia,
            estimated_cost_usd=winner_eval.cost_usd,
            estimated_latency_sec=winner_eval.estimated_latency_sec,
            quality_score=winner_eval.quality_match,
            cost_savings_vs_frontier_pct=savings_pct,
            cost_savings_usd=savings_usd,
            rationale=rationale,
            top_pareto_candidates=top_candidates,
            all_candidates=all_candidate_dicts,
        )


def recommend_for_prompt(prompt_text: str, **kwargs) -> RecommendationResult:
    """Helper to classify a prompt and immediately recommend the optimal Nebius model."""
    from tools.prompt_classifier import classify_prompt

    classifier_res = classify_prompt(prompt_text)
    scores = classifier_res.scores

    recommender = NebiusParetoRecommender()
    has_json = bool(re.search(r"\bjson\b|\byaml\b|\bschema\b", prompt_text, re.I))

    params = {
        "prompt_text": prompt_text,
        "task_type": scores.predicted_task_type,
        "complexity_score": classifier_res.weighted_score,
        "route_tier": classifier_res.route_tier,
        "specificity_score": 0.35,
        "cost_sensitivity": scores.cost_sensitivity,
        "latency_sensitivity": scores.latency_sensitivity,
        "has_json_request": has_json,
    }
    params.update(kwargs)

    return recommender.recommend(**params)

