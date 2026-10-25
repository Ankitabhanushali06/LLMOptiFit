"""
Nebius Token Factory Inference Client for LLMOptiFit.
====================================================

Connects LLMOptiFit's model recommendation output to live Nebius Token Factory
inference endpoints (https://api.tokenfactory.nebius.com/v1/).

Dispatches prompts to the recommended Pareto-optimal model and returns the
generated answer, latency, and token consumption metrics.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(env_path: Path = ENV_FILE) -> None:
    """Lightweight .env loader that populates os.environ without external dependencies."""
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


class ProviderError(RuntimeError):
    """Content-free provider error safe to expose publicly."""
    def __init__(self, code: str, message: str, status: int = 502):
        super().__init__(message)
        self.code, self.status = code, status


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward Authorization to another endpoint.


@dataclass
class GenerationResult:
    prompt: str
    recommended_model_id: str
    model_name: str
    vendor: str
    tier: str
    is_nvidia: bool
    generated_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    roundtrip_latency_sec: float
    is_live_call: bool
    error: Optional[str] = None


class NebiusTokenFactoryClient:
    """HTTP client for dispatching completions to Nebius Token Factory."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_sec: float = 60.0,
        allow_env: bool = True,
    ):
        if allow_env:
            load_env_file()
        self.api_key = api_key if not allow_env else (api_key or os.environ.get("NEBIUS_API_KEY"))
        self.base_url = ((base_url or os.environ.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL)
                         if allow_env else DEFAULT_BASE_URL).rstrip("/")
        self.timeout_sec = timeout_sec

    def is_configured(self) -> bool:
        """Returns True if a valid API key is present."""
        return bool(self.api_key and self.api_key != "replace_me" and len(self.api_key) > 5)

    def generate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> tuple[str, dict[str, Any], float]:
        """
        Sends a chat completion request to Nebius Token Factory.

        Returns:
            (generated_text, usage_dict, latency_seconds)
        """
        if not self.is_configured():
            raise ValueError(
                "NEBIUS_API_KEY is not set.\n"
                "Please set it in your terminal:\n"
                "  export NEBIUS_API_KEY=\"your_key_here\"\n"
                "or create a .env file in the LLMOptiFit project root:\n"
                "  NEBIUS_API_KEY=your_key_here"
            )

        endpoint = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "LLMOptiFit/1.0",
            },
            method="POST",
        )

        start_time = time.perf_counter()
        try:
            with urllib.request.build_opener(NoRedirects()).open(req, timeout=self.timeout_sec) as response:
                resp_bytes = response.read(2_000_001)
                if len(resp_bytes) > 2_000_000:
                    raise ProviderError("provider_response", "The provider response was too large.")
                latency = time.perf_counter() - start_time
                resp_json = json.loads(resp_bytes.decode("utf-8"))

                choices = resp_json.get("choices", [])
                if not choices:
                    raise ProviderError("provider_response", "The provider returned no answer.")

                msg = choices[0].get("message", {})
                content = (msg.get("content") or "").strip()
                generated_text = content or choices[0].get("text", "") or ""
                if not isinstance(generated_text, str) or not generated_text.strip():
                    raise ProviderError("provider_response", "The provider returned no final answer. Usage may still be billed.")

                usage = resp_json.get("usage") or {}
                if not isinstance(usage, dict):
                    raise ProviderError("provider_response", "The provider returned invalid usage data.")
                return generated_text, usage, latency

        except urllib.error.HTTPError as e:
            code = e.code
            e.close()
            if code in (401, 403):
                raise ProviderError("provider_auth", "Nebius rejected your key or model access.", 401) from None
            if code == 429:
                raise ProviderError("provider_limit", "Your Nebius account is rate limited. Try again later.", 429) from None
            if code in (400, 404, 422):
                raise ProviderError("provider_model", "Nebius could not run this request. Check model availability and context limits.") from None
            raise ProviderError("provider_unavailable", "Nebius is temporarily unavailable.") from None
        except urllib.error.URLError as e:
            raise ProviderError("provider_unavailable", "Could not reach Nebius. No automatic retry was made.") from None
        except TimeoutError:
            raise ProviderError("provider_timeout", "Nebius timed out. The request may still be billed; retry only if you choose.", 504) from None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError):
            raise ProviderError("provider_response", "Nebius returned an unreadable response.") from None


def route_and_generate(
    prompt: str,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    cost_sensitivity: float = 0.5,
    latency_sensitivity: float = 0.5,
    simulation_if_missing_key: bool = True,
    prefer_nvidia: bool = True,
) -> GenerationResult:
    """
    End-to-end LLMOptiFit pipeline:
      1. Classifies prompt intent and complexity
      2. Recommends the Pareto-optimal Nebius model
      3. Dispatches inference to Nebius Token Factory using the recommended model
      4. Returns text, latency, token usage, and dollar cost
    """
    from tools.nebius_recommender import recommend_for_prompt

    # 1. Recommend Pareto-optimal model
    rec = recommend_for_prompt(
        prompt,
        cost_sensitivity=cost_sensitivity,
        latency_sensitivity=latency_sensitivity,
        prefer_nvidia=prefer_nvidia,
    )

    client = NebiusTokenFactoryClient(api_key=api_key)

    if client.is_configured():
        # Live inference call to Nebius Token Factory
        try:
            generated_text, usage, latency = client.generate(
                model_id=rec.recommended_model_id,
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            in_tok = usage.get("prompt_tokens", int(len(prompt.split()) * 1.33))
            out_tok = usage.get("completion_tokens", len(generated_text.split()))
            tot_tok = usage.get("total_tokens", in_tok + out_tok)

            # Recalculate exact dollar cost from observed tokens
            catalog_model = rec.recommended_model_id
            # Retrieve model prices
            from tools.nebius_recommender import NebiusModelCatalog
            cat = NebiusModelCatalog()
            m = cat.get_model(catalog_model)
            if m:
                cost = (in_tok * m.input_price_per_million + out_tok * m.output_price_per_million) / 1_000_000.0
            else:
                cost = rec.estimated_cost_usd

            return GenerationResult(
                prompt=prompt,
                recommended_model_id=rec.recommended_model_id,
                model_name=rec.model_name,
                vendor=rec.vendor,
                tier=rec.tier,
                is_nvidia=rec.is_nvidia,
                generated_text=generated_text,
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=tot_tok,
                cost_usd=cost,
                roundtrip_latency_sec=latency,
                is_live_call=True,
            )
        except Exception as e:
            return GenerationResult(
                prompt=prompt,
                recommended_model_id=rec.recommended_model_id,
                model_name=rec.model_name,
                vendor=rec.vendor,
                tier=rec.tier,
                is_nvidia=rec.is_nvidia,
                generated_text="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=rec.estimated_cost_usd,
                roundtrip_latency_sec=0.0,
                is_live_call=False,
                error=str(e),
            )

    # If key is missing and simulation is requested, provide simulated execution response
    in_tok = int(len(prompt.split()) * 1.33)
    out_tok = 80
    return GenerationResult(
        prompt=prompt,
        recommended_model_id=rec.recommended_model_id,
        model_name=rec.model_name,
        vendor=rec.vendor,
        tier=rec.tier,
        is_nvidia=rec.is_nvidia,
        generated_text=(
            f"[Nebius Token Factory live generation requires NEBIUS_API_KEY]\n"
            f"Model {rec.recommended_model_id} selected via Pareto frontier.\n"
            f"To execute live inference, set NEBIUS_API_KEY in your environment or .env file."
        ),
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=in_tok + out_tok,
        cost_usd=rec.estimated_cost_usd,
        roundtrip_latency_sec=rec.estimated_latency_sec,
        is_live_call=False,
        error="NEBIUS_API_KEY not configured",
    )
