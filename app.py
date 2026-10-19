#!/usr/bin/env python3
"""
LLMOptiFit — Web Application Server
===================================

Lightweight, high-performance web server powering the LLMOptiFit frontend.
Serves static UI assets and provides REST API endpoints for:
  - Prompt complexity classification & parameter scoring
  - Pareto-optimal Nebius Token Factory model routing
  - Live LLM inference execution via Nebius Token Factory
  - Real-time model catalog queries & health checks

Zero external server dependencies (uses Python 3 standard library ThreadingHTTPServer).
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Hook into .venv site-packages if present
venv_lib = PROJECT_ROOT / ".venv" / "lib"
if venv_lib.exists():
    import site
    for site_pkg in venv_lib.glob("python*/site-packages"):
        if site_pkg.exists() and str(site_pkg) not in sys.path:
            site.addsitedir(str(site_pkg))

from tools.prompt_classifier import classify_prompt
from tools.nebius_recommender import NebiusParetoRecommender, recommend_for_prompt
from tools.nebius_client import NebiusTokenFactoryClient, route_and_generate

WEB_DIR = PROJECT_ROOT / "web"
CATALOG_PATH = PROJECT_ROOT / "config" / "nebius_catalog.json"


class LLMOptiFitRequestHandler(SimpleHTTPRequestHandler):
    """Handles static files from web/ and provides /api/* endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        # Enable CORS and caching headers
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _parse_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length).decode("utf-8")
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._handle_health()
        elif path == "/api/catalog":
            self._handle_catalog()
        else:
            # Fallback to serving static UI files
            if path == "/" or not path:
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/analyze":
            self._handle_analyze()
        elif path == "/api/generate":
            self._handle_generate()
        else:
            self._send_json({"error": f"Endpoint not found: {path}"}, status=404)

    def _handle_health(self):
        client = NebiusTokenFactoryClient()
        models_count = 0
        if CATALOG_PATH.exists():
            try:
                with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                    models_count = len(json.load(f))
            except Exception:
                pass

        self._send_json({
            "status": "healthy",
            "service": "LLMOptiFit",
            "version": "1.0.0",
            "api_key_configured": client.is_configured(),
            "models_available": models_count,
            "timestamp": time.time(),
        })

    def _handle_catalog(self):
        if not CATALOG_PATH.exists():
            self._send_json({"error": "Catalog file missing"}, status=500)
            return

        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                catalog_data = json.load(f)
            self._send_json({"catalog": catalog_data, "count": len(catalog_data)})
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_analyze(self):
        body = self._parse_json_body()
        prompt = body.get("prompt", "").strip()

        if not prompt:
            self._send_json({"error": "Prompt cannot be empty."}, status=400)
            return

        cost_sens = float(body.get("cost_sensitivity", 0.5))
        lat_sens = float(body.get("latency_sensitivity", 0.5))
        prefer_nv = bool(body.get("prefer_nvidia", True))

        try:
            # 1. Classify prompt parameters
            clf = classify_prompt(prompt)
            scores_dict = {
                "predicted_task_type": clf.scores.predicted_task_type,
                "reasoning_complexity": round(clf.scores.reasoning_complexity, 3),
                "accuracy_requirement": round(clf.scores.accuracy_requirement, 3),
                "instruction_complexity": round(clf.scores.instruction_complexity, 3),
                "output_complexity": round(clf.scores.output_complexity, 3),
                "context_requirement": round(clf.scores.context_requirement, 3),
                "decision_search_complexity": round(clf.scores.decision_search_complexity, 3),
                "domain_difficulty": round(clf.scores.domain_difficulty, 3),
            }

            top_drivers = []
            if clf.rationale and "top_drivers" in clf.rationale:
                top_drivers = [
                    {"parameter": d["parameter"], "score": round(d["score"], 3)}
                    for d in clf.rationale["top_drivers"]
                ]

            # 2. Pareto Optimal Recommendation
            recommender = NebiusParetoRecommender()
            rec = recommender.recommend(
                prompt_text=prompt,
                task_type=clf.scores.predicted_task_type,
                complexity_score=clf.weighted_score,
                route_tier=clf.route_tier,
                cost_sensitivity=cost_sens,
                latency_sensitivity=lat_sens,
                prefer_nvidia=prefer_nv,
            )

            response = {
                "success": True,
                "prompt": prompt,
                "classification": {
                    "task_type": clf.scores.predicted_task_type,
                    "route_tier": clf.route_tier,
                    "complexity_score": round(clf.weighted_score, 3),
                    "model_source": getattr(clf, "source", "supervised_nlp"),
                    "scores": scores_dict,
                    "top_drivers": top_drivers,
                },
                "recommendation": {
                    "recommended_model_id": rec.recommended_model_id,
                    "model_name": rec.model_name,
                    "vendor": rec.vendor,
                    "tier": rec.tier,
                    "is_nvidia": rec.is_nvidia,
                    "estimated_cost_usd": round(rec.estimated_cost_usd, 6),
                    "estimated_latency_sec": round(rec.estimated_latency_sec, 2),
                    "quality_score": round(rec.quality_score, 1),
                    "cost_savings_vs_frontier_pct": round(rec.cost_savings_vs_frontier_pct, 1),
                    "cost_savings_usd": round(rec.cost_savings_usd, 6),
                    "rationale": rec.rationale,
                    "top_pareto_candidates": rec.top_pareto_candidates,
                    "all_candidates": rec.all_candidates,
                },
            }
            self._send_json(response)
        except Exception as e:
            self._send_json({"error": f"Analysis failed: {str(e)}"}, status=500)

    def _handle_generate(self):
        body = self._parse_json_body()
        prompt = body.get("prompt", "").strip()

        if not prompt:
            self._send_json({"error": "Prompt cannot be empty."}, status=400)
            return

        cost_sens = float(body.get("cost_sensitivity", 0.5))
        lat_sens = float(body.get("latency_sensitivity", 0.5))
        forced_model_id = body.get("model_id")

        try:
            client = NebiusTokenFactoryClient()
            if not client.is_configured():
                self._send_json({
                    "success": False,
                    "error": "Nebius API key is not configured in .env. Please set NEBIUS_API_KEY.",
                }, status=401)
                return

            if forced_model_id:
                # Direct generation with specified model
                start_t = time.perf_counter()
                text, usage, lat = client.generate(model_id=forced_model_id, prompt=prompt)
                dur = time.perf_counter() - start_t
                in_tok = usage.get("prompt_tokens", len(prompt.split()) * 2)
                out_tok = usage.get("completion_tokens", len(text.split()) * 2)
                tot_tok = usage.get("total_tokens", in_tok + out_tok)
                
                # Model lookup for price
                recommender = NebiusParetoRecommender()
                model_meta = recommender.catalog.get_model(forced_model_id)
                cost = 0.0
                if model_meta:
                    cost = (in_tok * model_meta.input_price_per_million + out_tok * model_meta.output_price_per_million) / 1_000_000.0

                self._send_json({
                    "success": True,
                    "prompt": prompt,
                    "model_id": forced_model_id,
                    "model_name": model_meta.name if model_meta else forced_model_id,
                    "vendor": model_meta.vendor if model_meta else "custom",
                    "tier": model_meta.tier if model_meta else "unknown",
                    "is_nvidia": model_meta.is_nvidia if model_meta else ("nvidia" in forced_model_id.lower()),
                    "generated_text": text,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": tot_tok,
                    "cost_usd": round(cost, 6),
                    "roundtrip_latency_sec": round(dur, 2),
                    "is_live_call": True,
                })
            else:
                # Automated route & generate
                gen_res = route_and_generate(
                    prompt=prompt,
                    cost_sensitivity=cost_sens,
                    latency_sensitivity=lat_sens,
                )
                if gen_res.error:
                    self._send_json({"success": False, "error": gen_res.error}, status=500)
                    return

                self._send_json({
                    "success": True,
                    "prompt": prompt,
                    "model_id": gen_res.recommended_model_id,
                    "model_name": gen_res.model_name,
                    "vendor": gen_res.vendor,
                    "tier": gen_res.tier,
                    "is_nvidia": gen_res.is_nvidia,
                    "generated_text": gen_res.generated_text,
                    "input_tokens": gen_res.input_tokens,
                    "output_tokens": gen_res.output_tokens,
                    "total_tokens": gen_res.total_tokens,
                    "cost_usd": round(gen_res.cost_usd, 6),
                    "roundtrip_latency_sec": round(gen_res.roundtrip_latency_sec, 2),
                    "is_live_call": gen_res.is_live_call,
                })

        except Exception as e:
            self._send_json({"success": False, "error": f"Generation failed: {str(e)}"}, status=500)


def run_server(port: int = 8000, host: str = "127.0.0.1") -> None:
    """Starts the LLMOptiFit web server."""
    if not WEB_DIR.exists():
        WEB_DIR.mkdir(parents=True, exist_ok=True)

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, LLMOptiFitRequestHandler)
    print("=" * 70)
    print(f"  🚀 LLMOptiFit Web Application running at http://{host}:{port}/")
    print("  Connected to Nebius Token Factory & NVIDIA Nemotron Models")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8000
    run_server(port=port_arg)
