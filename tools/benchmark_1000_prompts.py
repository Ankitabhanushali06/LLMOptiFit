#!/usr/bin/env python3
"""
LLMOptiFit — 1,000 Prompt Benchmark Stress-Test & Evaluation Suite
==================================================================

Validates the full 4-pillar prompt classification pipeline and live
Nebius Token Factory model recommender across 1,000 diverse ground-truth
prompts spanning all 8 canonical task types:
  • coding (SQL, Python, systems, algorithms)
  • mathematical_reasoning (algebra, calculus, probability, proofs)
  • extraction (JSON, entities, contacts, financial metrics)
  • analysis (trade-offs, architectural comparisons, root-cause)
  • planning (sprints, cloud migrations, disaster recovery)
  • classification (sentiment, support tickets, spam, intents)
  • research (literature reviews, academic surveys, breakthroughs)
  • generation (creative writing, emails, copywriting)

Usage:
    python tools/benchmark_1000_prompts.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project root and .venv site-packages are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

venv_lib = PROJECT_ROOT / ".venv" / "lib"
if venv_lib.exists():
    import site
    for site_pkg in venv_lib.glob("python*/site-packages"):
        if site_pkg.exists() and str(site_pkg) not in sys.path:
            site.addsitedir(str(site_pkg))

from tools.prompt_classifier import classify_prompt
from tools.nebius_recommender import recommend_for_prompt

BENCHMARK_PATH = PROJECT_ROOT / "data/benchmark_1000_prompts.json"


def run_benchmark():
    if not BENCHMARK_PATH.exists():
        print(f"Error: Benchmark file not found at {BENCHMARK_PATH}")
        sys.exit(1)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    total = len(prompts_data)
    correct_task = 0
    misclassifications = []

    per_task_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    recommended_models = Counter()
    tiers = Counter()

    print("\n" + "═" * 80)
    print("  LLMOptiFit — 1,000 PROMPT BENCHMARK STRESS TEST")
    print("═" * 80)
    print(f"Loading {total} ground-truth prompts from {BENCHMARK_PATH.name}...")

    for idx, item in enumerate(prompts_data, 1):
        prompt = item["prompt"]
        true_task = item["true_task_type"]

        # Run full classification pipeline
        clf = classify_prompt(prompt)
        pred_task = clf.scores.predicted_task_type

        # Run Nebius recommendation
        rec = recommend_for_prompt(prompt)
        recommended_models[rec.recommended_model_id] += 1
        tiers[rec.tier] += 1

        per_task_stats[true_task]["total"] += 1

        if pred_task == true_task:
            correct_task += 1
            per_task_stats[true_task]["tp"] += 1
        else:
            per_task_stats[pred_task]["fp"] += 1
            per_task_stats[true_task]["fn"] += 1
            misclassifications.append({
                "id": item["id"],
                "prompt": prompt,
                "true_task": true_task,
                "pred_task": pred_task,
                "recommended_model": rec.recommended_model_id,
            })

    accuracy = correct_task / total * 100.0

    print("\n" + "─" * 80)
    print(f"  OVERALL PERFORMANCE SUMMARY")
    print("─" * 80)
    print(f"  • Total Evaluated Prompts : {total}")
    print(f"  • Classification Accuracy : {correct_task} / {total} ({accuracy:.2f}%)")
    print(f"  • Total Misclassifications: {len(misclassifications)}")
    print("─" * 80)
    print(f"  {'Task Category':<25} {'Total':<7} {'TP':<6} {'FP':<5} {'FN':<5} {'Precision':<10} {'Recall':<10} {'F1-Score'}")
    print("  " + "─" * 78)

    f1_scores = []
    for cat in sorted(per_task_stats.keys()):
        stats = per_task_stats[cat]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        tot = stats["total"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec_val / (prec + rec_val) if (prec + rec_val) > 0 else 0.0
        f1_scores.append(f1)
        print(f"  {cat:<25} {tot:<7} {tp:<6} {fp:<5} {fn:<5} {prec*100:<9.1f}% {rec_val*100:<9.1f}% {f1:.4f}")

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print("  " + "─" * 78)
    print(f"  Macro-Averaged F1-Score   : {macro_f1:.4f}")
    print("═" * 80)

    print("\n🏆 Top Recommended Models on Nebius Token Factory:")
    for m_id, count in recommended_models.most_common():
        pct = count / total * 100.0
        print(f"  • {m_id:<42}: {count:>4} prompts ({pct:>5.1f}%)")

    if misclassifications:
        print(f"\n⚠️  Misclassifications Found ({len(misclassifications)}):")
        for m in misclassifications[:10]:
            print(f"  • [{m['id']}] Expected: {m['true_task']} | Pred: {m['pred_task']} | \"{m['prompt'][:70]}...\"")
    else:
        print("\n✅ Zero misclassifications! All 1,000 prompts classified with 100% precision.")

    print("\n" + "═" * 80 + "\n")


if __name__ == "__main__":
    run_benchmark()
