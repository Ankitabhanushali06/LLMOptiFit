"""
Evaluates the LLMOptiFit prompt classifier and recommender across the 1,000 benchmark prompts.
Computes accuracy, confusion matrix, per-category precision/recall/F1, and logs all misclassifications.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.prompt_classifier import classify_prompt
from tools.nebius_recommender import recommend_for_prompt

BENCHMARK_PATH = PROJECT_ROOT / "data/benchmark_1000_prompts.json"


def evaluate():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    total = len(prompts_data)
    correct_task = 0
    correct_tier = 0
    misclassifications = []

    y_true_task = []
    y_pred_task = []

    per_task_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    recommended_models = Counter()
    tiers = Counter()

    for idx, item in enumerate(prompts_data, 1):
        prompt = item["prompt"]
        true_task = item["true_task_type"]
        true_tier = item["true_tier"]

        # Run classifier
        clf = classify_prompt(prompt)
        pred_task = clf.scores.predicted_task_type
        pred_tier = clf.route_tier

        # Run recommender
        rec = recommend_for_prompt(prompt)
        recommended_models[rec.recommended_model_id] += 1
        tiers[rec.tier] += 1

        y_true_task.append(true_task)
        y_pred_task.append(pred_task)

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
                "pred_tier": pred_tier,
                "recommended_model": rec.recommended_model_id,
            })

    accuracy = correct_task / total * 100.0

    print("=" * 80)
    print(f"LLMOptiFit — 1,000 BENCHMARK EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Evaluated Prompts : {total}")
    print(f"Task Classification Acc : {correct_task}/{total} ({accuracy:.2f}%)")
    print(f"Total Misclassifications: {len(misclassifications)}")
    print("-" * 80)
    print(f"{'Task Category':<25} {'Total':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'F1-Score'}")
    print("-" * 80)

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
        print(f"{cat:<25} {tot:<8} {tp:<6} {fp:<6} {fn:<6} {prec*100:<9.1f}% {rec_val*100:<9.1f}% {f1:.3f}")

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print("-" * 80)
    print(f"Macro-Averaged F1-Score : {macro_f1:.4f}")
    print("=" * 80)

    print("\nTop Recommended Models Across 1,000 Prompts:")
    for m_id, count in recommended_models.most_common(5):
        print(f"  • {m_id:<38}: {count} ({count/total*100:.1f}%)")

    print("\nSample Misclassifications (First 15):")
    for m in misclassifications[:15]:
        print(f"  • [ID {m['id']}] Expected: {m['true_task'].upper():<22} | Predicted: {m['pred_task'].upper():<12}")
        print(f"    Prompt: \"{m['prompt'][:85]}...\"")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "misclassifications": misclassifications,
    }


if __name__ == "__main__":
    evaluate()
