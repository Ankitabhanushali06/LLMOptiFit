"""Read-only analysis resources, loaded once per process."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import re
from pathlib import Path
from api.schemas import AnalyzeRequest, AnalyzeResponse, ClassificationResult
from tools.nebius_recommender import NebiusModelCatalog, NebiusParetoRecommender
from tools.prompt_classifier import MODEL_PATH, PromptComplexityModel

ROOT = Path(__file__).resolve().parent.parent


class AnalysisService:
    def __init__(self):
        self.classifier = PromptComplexityModel()
        self.classifier.load(MODEL_PATH)
        if self.classifier.extractor._task_model is None:
            raise RuntimeError("Task classifier artifact could not be loaded.")
        self.catalog = NebiusModelCatalog()
        if not self.catalog.models:
            raise RuntimeError("Catalog is empty.")
        self.recommender = NebiusParetoRecommender(self.catalog)
        self.version = hashlib.sha256(self.catalog.catalog_path.read_bytes()).hexdigest()[:12]
        path = ROOT / "config/catalog_metadata.json"
        self.metadata = json.loads(path.read_text()) if path.exists() else {}
        self.verified_at = (self.metadata.get("verified_at")
                            if self.metadata.get("catalog_version") == self.version else None)

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResponse:
        clf = self.classifier.classify(req.prompt)
        rec = self.recommender.recommend(
            prompt_text=req.prompt, task_type=clf.scores.predicted_task_type,
            complexity_score=clf.weighted_score, route_tier=clf.route_tier,
            specificity_score=0.35, cost_sensitivity=req.cost_sensitivity,
            latency_sensitivity=req.latency_sensitivity, prefer_nvidia=req.prefer_nvidia,
            has_json_request=bool(re.search(r"\bjson\b|\byaml\b|\bschema\b", req.prompt, re.I)),
        )
        return AnalyzeResponse(
            prompt=req.prompt,
            classification=ClassificationResult(
                task_type=clf.scores.predicted_task_type, route_tier=clf.route_tier,
                complexity_score=clf.weighted_score,
                model_source="Task classifier + rules; rule-based routing tier",
                scores=asdict(clf.scores), top_drivers=clf.rationale.get("top_drivers", []),
            ), recommendation=asdict(rec), catalog_version=self.version,
            catalog_verified_at=self.verified_at,
        )


@lru_cache(maxsize=1)
def get_service() -> AnalysisService:
    return AnalysisService()
