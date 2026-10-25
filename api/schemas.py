"""Public web API contracts. Credentials never belong in request bodies."""
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=20_000)
    cost_sensitivity: float = Field(0.5, ge=0, le=1)
    latency_sensitivity: float = Field(0.5, ge=0, le=1)
    prefer_nvidia: bool = True

    @field_validator("prompt")
    @classmethod
    def clean_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Enter a prompt.")
        value.encode("utf-8")
        return value


class GenerateRequest(AnalyzeRequest):
    model_id: str | None = Field(None, min_length=1, max_length=200)


class ClassificationResult(BaseModel):
    task_type: str
    route_tier: str
    complexity_score: float
    model_source: str
    scores: dict[str, Any]
    top_drivers: list[dict[str, Any]]


class AnalyzeResponse(BaseModel):
    success: bool = True
    prompt: str
    classification: ClassificationResult
    recommendation: dict[str, Any]
    catalog_version: str
    catalog_verified_at: str | None


class GenerateResponse(BaseModel):
    success: bool = True
    prompt: str
    model_id: str
    model_name: str
    vendor: str
    tier: str
    is_nvidia: bool
    generated_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    usage_source: str
    cost_usd: float
    cost_is_estimate: bool = True
    roundtrip_latency_sec: float
    is_live_call: bool = True
    catalog_version: str


class HealthResponse(BaseModel):
    status: str = "ready"
    service: str = "LLMOptiFit"
    version: str = "1.1.0"
    generation_mode: str = "server_key"
    generation_enabled: bool
    models_available: int
    catalog_version: str
