"""
Example: Running the 3-Pillar Prompt Classification Framework on Arena Prompts.

Pillars:
  1. Unsupervised Topic Discovery (Clusters & Keywords)
  2. Task Type Classification (Coding, Math, Analysis, etc.)
  3. Complexity & Model Tier Routing (Nano, Super, Ultra)
"""

from pathlib import Path
import pandas as pd
from tools.prompt_pipeline import PromptClassificationPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data/arena/arena_prompts_english.parquet"


def main():
    print(f"Loading data from {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df):,} total prompts.\n")

    # Run pipeline on a sample (e.g., 500 prompts for quick demonstration)
    pipeline = PromptClassificationPipeline(n_clusters=6)
    print("Running 3-Pillar Classification on 500 sample prompts...")
    processed_df, cluster_keywords = pipeline.process_dataframe(df, sample_size=500)

    print("\n" + "=" * 70)
    print("PILLAR 1: DISCOVERED TOPIC CLUSTERS (Unsupervised TF-IDF + KMeans)")
    print("=" * 70)
    for cluster_id, keywords in cluster_keywords.items():
        count = (processed_df["topic_cluster_id"] == cluster_id).sum()
        print(f"  Cluster {cluster_id} ({count} prompts): {', '.join(keywords)}")

    print("\n" + "=" * 70)
    print("PILLAR 2: TASK TYPE DISTRIBUTION (Multi-Class)")
    print("=" * 70)
    task_counts = processed_df["task_type"].value_counts()
    for task, count in task_counts.items():
        pct = (count / len(processed_df)) * 100
        print(f"  {task:<25}: {count:3d} ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print("PILLAR 3: COMPLEXITY & ROUTE TIER BREAKDOWN")
    print("=" * 70)
    tier_counts = processed_df["route_tier"].value_counts()
    for tier, count in tier_counts.items():
        pct = (count / len(processed_df)) * 100
        print(f"  Tier {tier.upper():<6} -> {count:3d} prompts ({pct:.1f}%)")

    print("\nAverage Complexity Score:", round(processed_df["complexity_score"].mean(), 3))

    print("\n" + "=" * 70)
    print("SAMPLE PROCESSED PROMPTS (All 3 Dimensions)")
    print("=" * 70)
    cols = ["prompt", "topic_name", "task_type", "complexity_score", "route_tier"]
    for idx, row in processed_df[cols].head(4).iterrows():
        p_preview = row["prompt"][:80].replace("\n", " ") + ("..." if len(row["prompt"]) > 80 else "")
        print(f"\nPrompt:      \"{p_preview}\"")
        print(f"Topic:       {row['topic_name']}")
        print(f"Task Type:   {row['task_type']}")
        print(f"Complexity:  {row['complexity_score']} -> {row['route_tier'].upper()}")


if __name__ == "__main__":
    main()
