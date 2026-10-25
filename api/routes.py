"""Anonymous analysis and opt-in generation using the server owner's key."""
from dataclasses import asdict
from threading import BoundedSemaphore
from fastapi import APIRouter, HTTPException
from api.schemas import AnalyzeRequest, AnalyzeResponse, GenerateRequest, GenerateResponse, HealthResponse
from api.service import get_service
from api.settings import get_settings
from tools.nebius_client import NebiusTokenFactoryClient, ProviderError

router = APIRouter(prefix="/api")
generation_slots = BoundedSemaphore(2)


def fail(status: int, code: str, message: str):
    raise HTTPException(status, detail={"code": code, "message": message})


@router.get("/health", response_model=HealthResponse)
def health():
    service = get_service()
    return HealthResponse(models_available=len(service.catalog.models), catalog_version=service.version, generation_enabled=get_settings().can_generate)


@router.get("/catalog")
def catalog():
    service = get_service()
    return {"success": True, "catalog": [asdict(m) for m in service.catalog.models],
            "count": len(service.catalog.models), "version": service.version,
            "verified_at": service.verified_at, "pricing_is_estimate": True}


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        return get_service().analyze(req)
    except ValueError:
        fail(422, "no_eligible_model", "No catalog model meets this prompt's context and format requirements.")


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    settings = get_settings()
    if not settings.can_generate:
        fail(503, "generation_disabled", "Live generation is not enabled for this pilot. Recommendations remain available.")
    service = get_service()
    analysis = analyze(req)
    model_id = req.model_id or analysis.recommendation["recommended_model_id"]
    model = service.catalog.get_model(model_id)
    if model is None:
        fail(422, "unknown_model", "Select a model from the current catalog.")
    eligible = {m["model_id"] for m in analysis.recommendation["all_candidates"]}
    if model_id not in eligible:
        fail(422, "ineligible_model", "This model cannot meet the prompt's context or format requirements.")
    if not generation_slots.acquire(blocking=False):
        fail(429, "busy", "Two requests are already running. Please try again shortly.")
    try:
        client = NebiusTokenFactoryClient(api_key=settings.api_key.get_secret_value(), allow_env=False)
        text, usage, latency = client.generate(model_id=model_id, prompt=req.prompt, max_tokens=1024)
        observed = all(type(usage.get(k)) is int and usage[k] >= 0
                       for k in ("prompt_tokens", "completion_tokens"))
        in_tokens = usage["prompt_tokens"] if observed else max(1, (len(req.prompt.encode("utf-8")) + 3) // 4)
        out_tokens = usage["completion_tokens"] if observed else max(1, (len(text.encode("utf-8")) + 3) // 4)
        cost = (in_tokens * model.input_price_per_million + out_tokens * model.output_price_per_million) / 1e6
        return GenerateResponse(
            prompt=req.prompt, model_id=model_id, model_name=model.name, vendor=model.vendor,
            tier=model.tier, is_nvidia=model.is_nvidia, generated_text=text,
            input_tokens=in_tokens, output_tokens=out_tokens, total_tokens=in_tokens + out_tokens,
            usage_source="provider" if observed else "estimated", cost_usd=round(cost, 8),
            roundtrip_latency_sec=round(latency, 3), catalog_version=service.version,
        )
    except ProviderError as exc:
        fail(exc.status, exc.code, str(exc))
    except Exception:
        fail(502, "generation_failed", "Generation failed. No automatic retry was made; the request may still be billed.")
    finally:
        generation_slots.release()
