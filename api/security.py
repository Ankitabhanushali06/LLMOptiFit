"""Bounded, process-local limits for a single-worker free deployment."""
from collections import deque
import hashlib
import ipaddress
import logging
import os
import time
from starlette.responses import JSONResponse

log = logging.getLogger("llmoptifit.requests")
MAX_BODY = 100_000


def error_response(status, code, message):
    return JSONResponse({"success": False, "error": {"code": code, "message": message}}, status_code=status)


class WebGuard:
    def __init__(self, app):
        self.app = app
        self.buckets = {}

    def allowed(self, identity, group):
        now = time.monotonic()
        # Bound retained identifiers and erase inactive buckets after a minute.
        expired = [k for k, q in self.buckets.items() if not q or q[-1] <= now - 60]
        for key in expired:
            del self.buckets[key]
        key = (hashlib.sha256(identity.encode()).digest(), group)
        if key not in self.buckets and len(self.buckets) >= 4096:
            return False
        queue = self.buckets.setdefault(key, deque())
        while queue and queue[0] <= now - 60:
            queue.popleft()
        limit = {"analyze": 30, "generate": 5}.get(group, 60)
        if len(queue) >= limit:
            return False
        queue.append(now)
        return True

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        is_api = path.startswith("/api/")
        headers = dict(scope.get("headers", []))
        start = time.monotonic()
        response_started = False

        async def secured_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                extra = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    (b"content-security-policy", b"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"),
                    (b"cache-control", b"no-store" if is_api else b"no-cache"),
                ]
                message["headers"] = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"] + extra
                if is_api:
                    # Never log raw URLs, IPs, headers, bodies, or exception text.
                    route = path if path in ("/api/health", "/api/catalog", "/api/analyze", "/api/generate") else "/api/unknown"
                    log.info("route=%s status=%s elapsed_ms=%d", route, message["status"], int((time.monotonic()-start)*1000))
            await send(message)

        async def reject(status, code, message):
            await error_response(status, code, message)(scope, receive, secured_send)

        if is_api:
            identity = (scope.get("client") or ("unknown", 0))[0]
            # Render is the only trusted ingress. Use its nearest forwarded hop,
            # never the attacker-controlled leftmost address. Disable Uvicorn's
            # proxy rewriting so local deployments trust the socket peer only.
            if os.environ.get("RENDER") == "true" and b"x-forwarded-for" in headers:
                try:
                    identity = str(ipaddress.ip_address(headers[b"x-forwarded-for"].decode().split(",")[-1].strip()))
                except (ValueError, UnicodeError):
                    pass
            group = path.rsplit("/", 1)[-1] if path in ("/api/analyze", "/api/generate") else "read"
            if not self.allowed(identity, group):
                return await reject(429, "rate_limited", "Too many requests. Please wait a minute.")
            if scope["method"] == "POST":
                origin = headers.get(b"origin")
                if origin and origin.split(b"://", 1)[-1] != headers.get(b"host"):
                    return await reject(403, "origin_denied", "Use this app's own website to send requests.")
                if headers.get(b"content-type", b"").split(b";", 1)[0].lower() != b"application/json":
                    return await reject(415, "json_required", "Send an application/json request.")
                chunks, size = [], 0
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size > MAX_BODY:
                        return await reject(413, "request_too_large", "The request is too large.")
                    chunks.append(chunk)
                    if not message.get("more_body", False):
                        break
                body = b"".join(chunks)
                delivered = False
                original_receive = receive

                async def buffered_receive():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    return await original_receive()
                receive = buffered_receive
        try:
            await self.app(scope, receive, secured_send)
        except Exception:
            # Unexpected application exceptions must not reach Uvicorn traceback logs.
            if not response_started:
                await reject(500, "internal_error", "The service could not complete this request.")
