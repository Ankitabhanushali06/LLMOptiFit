"""
Demonstration: Nebius Token Factory Model Recommendation Engine for LLMOptiFit.

Tests live prompt routing using the exact Nebius model catalog
(https://tokenfactory.nebius.com/models/catalog) and calculates real dollar savings.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.nebius_recommender import recommend_for_prompt


def main():
    test_prompts = [
        # 1. Simple extraction / Q&A -> Should route to Nano (e.g. Nemotron-3-Nano)
        "What is the capital of Australia?",
        
        # 2. Structured extraction task -> Nano with JSON support
        "Extract the user names and email addresses from the text above and return them as a JSON list.",
        
        # 3. Medium complexity analysis & coding -> Super (e.g. Nemotron-3-Super or Qwen3-30B)
        "Write a Python script using pandas and matplotlib to read sales_data.csv, calculate monthly rolling revenue, and plot a seasonal trend line.",
        
        # 4. Heavy multi-step distributed systems design -> Ultra (e.g. Nemotron-3-Ultra or Hermes-4-405B)
        "Architect a globally distributed, Byzantine fault-tolerant consensus protocol for a financial settlement network. "
        "Formally specify state transitions, edge case handling during network partition, cryptographic proof requirements, "
        "and provide formal pseudo-code with safety and liveness invariants."
    ]

    print("=" * 85)
    print("LLMOptiFit — NEBIUS TOKEN FACTORY MODEL RECOMMENDATION DEMO")
    print("Connected to live catalog: https://tokenfactory.nebius.com/models/catalog")
    print("=" * 85)

    for i, p in enumerate(test_prompts, 1):
        print(f"\n[Test Case {i}] Prompt: \"{p[:80]}...\"" if len(p) > 80 else f"\n[Test Case {i}] Prompt: \"{p}\"")
        result = recommend_for_prompt(p)

        print(f"  ★ Recommended Model : {result.recommended_model_id} ({result.model_name})")
        print(f"  • Model Tier        : {result.tier.upper()} | Vendor: {result.vendor.upper()} | NVIDIA Model: {result.is_nvidia}")
        print(f"  • Estimated Cost    : ${result.estimated_cost_usd:.6f}")
        print(f"  • Estimated Latency : {result.estimated_latency_sec:.2f}s")
        print(f"  • Quality Score     : {result.quality_score:.1f}/100")
        print(f"  • Dollar Savings    : {result.cost_savings_vs_frontier_pct:.1f}% cheaper than Frontier (${result.cost_savings_usd:.6f} saved)")
        print(f"  • Rationale         : {result.rationale}")
        
        print("\n  Top 3 Ranked Candidates for this Prompt:")
        for rank, cand in enumerate(result.top_pareto_candidates[:3], 1):
            pareto_tag = " [Pareto Optimal]" if cand["is_pareto_optimal"] else ""
            print(f"    {rank}. {cand['model_id']:<38} | Cost: ${cand['cost_usd']:.6f} | Q: {cand['quality_match']:<4.1f} | Latency: {cand['latency_sec']:.2f}s{pareto_tag}")
        print("-" * 85)


if __name__ == "__main__":
    main()
