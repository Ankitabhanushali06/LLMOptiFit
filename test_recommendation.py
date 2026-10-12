#!/usr/bin/env python3
"""
LLMOptiFit — Interactive & CLI Model Recommendation Test Tool
============================================================

Tests prompt classification and live Nebius Token Factory model routing.
Takes any user prompt from CLI arguments or interactive console input,
analyzes prompt parameters, and suggests the Pareto-optimal Nebius model.

Usage:
    # 1. Single prompt via CLI:
    python test_recommendation.py "What is the capital of Australia?"

    # 2. Interactive prompt mode:
    python test_recommendation.py

    # 3. Benchmark suite across all tiers:
    python test_recommendation.py --demo

    # 4. JSON output mode for programmatic testing:
    python test_recommendation.py "Write a Python script" --json-output
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root and .venv site-packages are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Automatically hook into repository .venv site-packages if running with system python3
venv_lib = PROJECT_ROOT / ".venv" / "lib"
if venv_lib.exists():
    import site
    for site_pkg in venv_lib.glob("python*/site-packages"):
        if site_pkg.exists() and str(site_pkg) not in sys.path:
            site.addsitedir(str(site_pkg))

try:
    from tools.prompt_classifier import classify_prompt
    from tools.nebius_recommender import NebiusParetoRecommender, recommend_for_prompt
except ImportError as e:
    print(f"Error importing LLMOptiFit tools: {e}")
    print("Please make sure you are in the LLMOptiFit project root and virtual environment is active.")
    sys.exit(1)


# Preset test prompts for benchmark demonstration
DEMO_BENCHMARK_PROMPTS = [
    {
        "category": "Nano Tier / General Q&A (Fast & Low Cost)",
        "prompt": "What is the capital of Australia?",
    },
    {
        "category": "Super Tier / Structured Data Extraction",
        "prompt": "Extract the user names and email addresses from the text above and return them as a validated JSON list.",
    },
    {
        "category": "Super Tier / Data Science & Python Scripting",
        "prompt": "Write a Python script using pandas and matplotlib to read sales_data.csv, calculate monthly rolling revenue, and plot a seasonal trend line.",
    },
    {
        "category": "Ultra Tier / Frontier Systems Architecture",
        "prompt": "Architect a globally distributed, Byzantine fault-tolerant consensus protocol for a financial settlement network. Formally specify state transitions, partition handling, and cryptographic invariants.",
    },
]


def print_recommendation_card(
    prompt: str,
    cost_sensitivity: float = 0.5,
    latency_sensitivity: float = 0.5,
    show_alternatives: bool = False,
    generate_response: bool = True,
) -> None:


    """Classifies the prompt and prints the single final Pareto-optimal model recommendation."""
    # 1. Run full prompt classification
    clf_result = classify_prompt(prompt)
    scores = clf_result.scores
    
    # 2. Run Nebius Token Factory Pareto Recommendation
    rec = recommend_for_prompt(
        prompt,
        cost_sensitivity=cost_sensitivity,
        latency_sensitivity=latency_sensitivity,
    )

    nvidia_badge = " [NVIDIA Open-Source 🟢]" if rec.is_nvidia else ""
    tier_upper = rec.tier.upper()

    print("\n" + "═" * 80)
    print("  LLMOptiFit — FINAL MODEL RECOMMENDATION (PARETO OPTIMAL)")
    print("═" * 80)

    # Prompt Preview
    prompt_display = prompt.strip()
    if len(prompt_display) > 140:
        prompt_display = prompt_display[:137] + "..."
    print(f"\n📝 PROMPT:")
    print(f"   \"{prompt_display}\"")

    # Prompt Analysis Parameters
    print(f"\n🔍 PROMPT ANALYSIS:")
    print(f"   • Predicted Task Type : {scores.predicted_task_type.upper()}")
    print(f"   • Recommended Tier    : {clf_result.route_tier.upper()} (Complexity Score: {clf_result.weighted_score:.3f})")
    print(f"   • Context / Tokens    : ~{len(prompt.split())} words (~{int(len(prompt.split()) * 1.33)} input tokens)")
    if clf_result.rationale and "top_drivers" in clf_result.rationale:
        top_drivers = clf_result.rationale["top_drivers"][:2]
        driver_str = ", ".join(f"{d['parameter']} ({d['score']:.2f})" for d in top_drivers)
        print(f"   • Complexity Drivers  : {driver_str}")

    # Single Winning Model Recommendation
    print(f"\n🏆 RECOMMENDED MODEL (ONLY OPTIMAL FIT):")
    print(f"   ★ Model ID            : {rec.recommended_model_id}{nvidia_badge}")
    print(f"   • Model Name          : {rec.model_name}")
    print(f"   • Vendor & Tier       : {rec.vendor.upper()} ({tier_upper} tier)")
    print(f"   • Status              : Pareto Optimal (Unbeatable Cost/Quality Frontier)")
    print(f"   • Estimated Cost      : ${rec.estimated_cost_usd:.6f} USD")
    print(f"   • Estimated Latency   : {rec.estimated_latency_sec:.2f} seconds")
    print(f"   • Quality Fit Score   : {rec.quality_score:.1f} / 100")
    if rec.cost_savings_vs_frontier_pct > 0:
        print(f"   • Cost Savings        : {rec.cost_savings_vs_frontier_pct:.1f}% cheaper than Frontier baseline (${rec.cost_savings_usd:.6f} saved)")
    else:
        print(f"   • Cost Savings        : Frontier-tier capability deployed")

    # Selection Rationale
    print(f"\n💡 WHY THIS MODEL:")
    print(f"   {rec.rationale}")

    # Only show alternatives if explicitly requested
    if show_alternatives and rec.top_pareto_candidates:
        print(f"\n📊 ALTERNATIVE PARETO CANDIDATES:")
        print(f"   {'Rank':<5} {'Model ID':<38} {'Cost ($)':<12} {'Latency':<10} {'Quality':<9} {'Status'}")
        print(f"   {'-'*4} {'-'*37} {'-'*11} {'-'*9} {'-'*8} {'-'*15}")
        for idx, cand in enumerate(rec.top_pareto_candidates[:4], 1):
            pareto_str = "Pareto Optimal" if cand["is_pareto_optimal"] else "Dominated"
            nv_flag = " (NVIDIA)" if cand.get("is_nvidia") else ""
            model_label = (cand['model_id'][:35] + ".." if len(cand['model_id']) > 37 else cand['model_id'])
            print(f"   #{idx:<4} {model_label:<38} ${cand['cost_usd']:<11.6f} {cand['latency_sec']:<9.2f}s {cand['quality_match']:<8.1f} {pareto_str}{nv_flag}")

    # Live Inference Call if requested
    if generate_response:
        from tools.nebius_client import route_and_generate, NebiusTokenFactoryClient
        client = NebiusTokenFactoryClient()
        if client.is_configured():
            print("\n" + "─" * 80)
            print(f"🚀 CALLING NEBIUS TOKEN FACTORY ({rec.recommended_model_id})...")
            gen_res = route_and_generate(
                prompt,
                cost_sensitivity=cost_sensitivity,
                latency_sensitivity=latency_sensitivity,
            )
            if gen_res.error:
                print(f"❌ Error during inference: {gen_res.error}")
            else:
                print(f"⚡ Round-Trip Latency : {gen_res.roundtrip_latency_sec:.2f}s")
                print(f"📊 Tokens Consumed    : {gen_res.input_tokens} prompt / {gen_res.output_tokens} completion ({gen_res.total_tokens} total)")
                print(f"💰 Actual Cost (USD)  : ${gen_res.cost_usd:.6f}")
                print("\n💬 RESPONSE:")
                print("─" * 80)
                print(gen_res.generated_text.strip())
                print("─" * 80)
        else:
            print("\n" + "─" * 80)
            print("💬 NEBIUS TOKEN FACTORY LIVE INFERENCE:")
            print("   ℹ️  Ready to execute live requests on Nebius Token Factory!")
            print("   To execute real inference with the recommended model, provide your API key:")
            print("     export NEBIUS_API_KEY=\"your_key_here\"")
            print("   or add to .env file:")
            print("     NEBIUS_API_KEY=your_key_here")
            print("─" * 80)

    print("═" * 80 + "\n")




def run_interactive_mode() -> None:
    """Runs a continuous interactive CLI loop allowing users to test multiple prompts."""
    print("=" * 80)
    print("  LLMOptiFit — Interactive Nebius Model Recommendation Shell")
    print("  Connected to: https://tokenfactory.nebius.com/models/catalog")
    print("=" * 80)
    print("Type your prompt and press [Enter] to get an instant recommendation.")
    print("Type 'demo' to run the benchmark suite, or 'exit' / 'quit' to stop.\n")

    while True:
        try:
            user_input = input("Enter prompt > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "demo":
            run_demo_suite()
            continue

        try:
            print_recommendation_card(user_input)
        except Exception as e:
            print(f"❌ Error evaluating prompt: {e}\n")


def run_demo_suite() -> None:
    """Runs the 4 diverse benchmark test cases covering all model tiers."""
    print("=" * 80)
    print("  LLMOptiFit — BENCHMARK EVALUATION SUITE")
    print("  Testing 4 distinct complexity tiers on Nebius Token Factory")
    print("=" * 80)

    for i, test_case in enumerate(DEMO_BENCHMARK_PROMPTS, 1):
        print(f"\n>>> Benchmark Case {i}/{len(DEMO_BENCHMARK_PROMPTS)}: {test_case['category']}")
        print_recommendation_card(test_case["prompt"])


def main():
    parser = argparse.ArgumentParser(
        description="LLMOptiFit: Recommend the best Nebius Token Factory model for a given prompt."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="The input user prompt to classify and recommend a model for. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the benchmark test suite across all 4 difficulty tiers.",
    )
    parser.add_argument(
        "--cost-sensitivity",
        type=float,
        default=0.5,
        help="Sensitivity to dollar cost (0.0 = performance priority, 1.0 = maximum cost savings). Default: 0.5",
    )
    parser.add_argument(
        "--latency-sensitivity",
        type=float,
        default=0.5,
        help="Sensitivity to response latency (0.0 = ignore latency, 1.0 = fastest response). Default: 0.5",
    )
    parser.add_argument(
        "--no-generate",
        "--recommend-only",
        action="store_true",
        help="Only show model recommendation without executing live inference on Nebius Token Factory.",
    )
    parser.add_argument(
        "--show-alternatives",
        action="store_true",
        help="Also display alternative candidate models (by default only the single winning Pareto Optimal model is shown).",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output raw structured JSON instead of human-readable text report.",
    )

    args = parser.parse_args()
    should_generate = not args.no_generate

    # Case 1: Run benchmark demo suite
    if args.demo:
        run_demo_suite()
        return

    # Case 2: Direct prompt supplied via CLI argument
    if args.prompt:
        if args.json_output:
            if should_generate:
                from tools.nebius_client import route_and_generate
                gen_res = route_and_generate(
                    args.prompt,
                    cost_sensitivity=args.cost_sensitivity,
                    latency_sensitivity=args.latency_sensitivity,
                )
                output = {
                    "prompt": args.prompt,
                    "recommended_model_id": gen_res.recommended_model_id,
                    "model_name": gen_res.model_name,
                    "tier": gen_res.tier,
                    "is_nvidia": gen_res.is_nvidia,
                    "is_live_call": gen_res.is_live_call,
                    "generated_text": gen_res.generated_text,
                    "input_tokens": gen_res.input_tokens,
                    "output_tokens": gen_res.output_tokens,
                    "total_tokens": gen_res.total_tokens,
                    "cost_usd": gen_res.cost_usd,
                    "latency_sec": gen_res.roundtrip_latency_sec,
                    "error": gen_res.error,
                }
                print(json.dumps(output, indent=2))
            else:
                clf_result = classify_prompt(args.prompt)
                rec = recommend_for_prompt(
                    args.prompt,
                    cost_sensitivity=args.cost_sensitivity,
                    latency_sensitivity=args.latency_sensitivity,
                )
                rec_dict = {
                    "recommended_model_id": rec.recommended_model_id,
                    "model_name": rec.model_name,
                    "vendor": rec.vendor,
                    "tier": rec.tier,
                    "status": "Pareto Optimal",
                    "is_nvidia": rec.is_nvidia,
                    "estimated_cost_usd": rec.estimated_cost_usd,
                    "estimated_latency_sec": rec.estimated_latency_sec,
                    "quality_score": rec.quality_score,
                    "cost_savings_vs_frontier_pct": rec.cost_savings_vs_frontier_pct,
                    "cost_savings_usd": rec.cost_savings_usd,
                    "rationale": rec.rationale,
                }
                if args.show_alternatives:
                    rec_dict["top_pareto_candidates"] = rec.top_pareto_candidates

                output = {
                    "prompt": args.prompt,
                    "classification": {
                        "task_type": clf_result.scores.predicted_task_type,
                        "complexity_score": clf_result.weighted_score,
                        "route_tier": clf_result.route_tier,
                    },
                    "recommendation": rec_dict,
                }
                print(json.dumps(output, indent=2))
        else:
            print_recommendation_card(
                args.prompt,
                cost_sensitivity=args.cost_sensitivity,
                latency_sensitivity=args.latency_sensitivity,
                show_alternatives=args.show_alternatives,
                generate_response=should_generate,
            )
        return


    # Case 3: No CLI argument provided -> Launch interactive mode
    run_interactive_mode()



if __name__ == "__main__":
    main()
