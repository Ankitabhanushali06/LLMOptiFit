"""Single-origin, server-funded web application."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles
from api.routes import router
from api.security import WebGuard, error_response
from api.service import get_service


request_log = logging.getLogger("llmoptifit.requests")
if not request_log.handlers:
    request_log.addHandler(logging.StreamHandler())
request_log.setLevel(logging.INFO)
request_log.propagate = False


@asynccontextmanager
async def lifespan(app):
    get_service()
    yield


app = FastAPI(title="LLMOptiFit", version="1.1.0", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(WebGuard)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # Pydantic's default errors include rejected input; never echo it.
    return error_response(422, "invalid_request", "Enter 1–20,000 characters and valid preferences between 0 and 1.")


@app.exception_handler(HTTPException)
async def http_error(request, exc):
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    return error_response(exc.status_code, detail.get("code", "request_failed"),
                          detail.get("message", "The requested resource is unavailable."))


app.include_router(router)
# StaticFiles rejects traversal and symlinks outside web/; API routes come first.
app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "web", html=True), name="web")
