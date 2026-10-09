"""
Unified Prompt Analysis & Classification Pipeline for LLMOptiFit.

Addresses 3 core pillars:
1. Unsupervised Topic Discovery (TF-IDF + KMeans clustering & keyword extraction)
2. Task Type Classification (8 canonical task categories)
3. Complexity & Model Tier Routing (14-parameter rubric -> Nano / Super / Ultra)
"""

from __future__ import annotations

import re
from typing import Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

from tools.prompt_classifier import (
    PromptFeatureExtractor,
    classify_prompt,
    TASK_TYPES,
)


class TopicDiscoverer:
    """
    Pillar 1: Unsupervised Discovery & Clustering.
    Groups raw prompts into k clusters using TF-IDF and identifies top distinctive keywords.
    """

    def __init__(self, n_clusters: int = 8, max_features: int = 5000, random_state: int = 42):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        )
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=1024,
            n_init=3,
        )
        self.cluster_labels_: dict[int, str] = {}

    def fit_transform(self, prompts: list[str]) -> tuple[np.ndarray, dict[int, list[str]]]:
        """Fit clustering on prompts, assign clusters, and generate top keywords per cluster."""
        tfidf_matrix = self.vectorizer.fit_transform(prompts)
        cluster_assignments = self.kmeans.fit_predict(tfidf_matrix)

        # Extract top keywords per cluster
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        cluster_keywords = {}
        for cluster_id in range(self.n_clusters):
            center = self.kmeans.cluster_centers_[cluster_id]
            top_indices = np.argsort(center)[::-1][:6]
            top_words = feature_names[top_indices].tolist()
            cluster_keywords[cluster_id] = top_words
            self.cluster_labels_[cluster_id] = f"Cluster {cluster_id}: {', '.join(top_words[:3])}"

        return cluster_assignments, cluster_keywords


class PromptClassificationPipeline:
    """
    Unified Pipeline executing all 3 pillars:
      1. Discovered Topics (Unsupervised Clustering)
      2. Task Type Classification (Rule/Heuristic & Pattern Detection)
      3. Complexity Score & Route Tier (14 Rubric Dimensions -> Nano/Super/Ultra)
    """

    def __init__(self, n_clusters: int = 8):
        self.topic_discoverer = TopicDiscoverer(n_clusters=n_clusters)
        self.feature_extractor = PromptFeatureExtractor()

    def process_dataframe(
        self,
        df: pd.DataFrame,
        prompt_col: str = "prompt",
        sample_size: int | None = None,
    ) -> tuple[pd.DataFrame, dict[int, list[str]]]:
        """
        Process DataFrame and append:
          - 'discovered_topic'
          - 'task_type'
          - 'complexity_score'
          - 'route_tier'
          - 'recommended_model'
        """
        target_df = df.sample(sample_size, random_state=42).copy() if sample_size else df.copy()
        prompts = target_df[prompt_col].fillna("").astype(str).tolist()

        # Pillar 1: Unsupervised Clustering
        clusters, cluster_keywords = self.topic_discoverer.fit_transform(prompts)
        target_df["topic_cluster_id"] = clusters
        target_df["topic_name"] = [self.topic_discoverer.cluster_labels_[c] for c in clusters]

        # Pillar 2 & 3: Task Types + Complexity Scoring & Model Tier Routing
        task_types = []
        complexity_scores = []
        route_tiers = []
        recommended_models = []

        for p in prompts:
            res = classify_prompt(p)
            task_types.append(res.scores.predicted_task_type)
            complexity_scores.append(round(res.weighted_score, 4))
            route_tiers.append(res.route_tier)
            recommended_models.append(res.recommended_model)

        target_df["task_type"] = task_types
        target_df["complexity_score"] = complexity_scores
        target_df["route_tier"] = route_tiers
        target_df["recommended_model"] = recommended_models

        return target_df, cluster_keywords
