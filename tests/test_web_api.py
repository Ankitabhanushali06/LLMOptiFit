"""Regression tests: no real inference calls and no test-only dependency installs."""
import asyncio
import io
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError
from pydantic import SecretStr

from api.server import app
from api.service import get_service
from api.settings import Settings
from api.routes import generation_slots
from tools.nebius_client import NebiusTokenFactoryClient, ProviderError, NoRedirects, DEFAULT_BASE_URL

OWNER = "unit-test-owner-secret"
VISITOR = "unit-test-visitor-secret"
SETTINGS = Settings(SecretStr(OWNER), True)


async def request(method, path, payload=None, headers=None, client="127.0.0.1", raw=None, chunks=None):
    body = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else b"")
    fields = {"host":"testserver", **(headers or {})}
    if payload is not None: fields.setdefault("content-type", "application/json")
    scope = {"type":"http", "asgi":{"version":"3.0"}, "http_version":"1.1", "method":method,
             "scheme":"http", "path":path, "raw_path":path.encode(), "query_string":b"", "root_path":"",
             "headers":[(k.lower().encode(),v.encode()) for k,v in fields.items()], "client":(client,1234), "server":("testserver",80)}
    incoming = list(chunks) if chunks is not None else [body]
    sent = []
    async def receive():
        if incoming:
            chunk = incoming.pop(0)
            return {"type":"http.request", "body":chunk, "more_body":bool(incoming)}
        await asyncio.Event().wait()
        return {"type":"http.disconnect"}
    async def send(message): sent.append(message)
    await app(scope, receive, send)
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1, sent
    data = b"".join(m.get("body",b"") for m in sent)
    response_headers = dict(starts[0]["headers"])
    if b"application/json" in response_headers.get(b"content-type",b""): data = json.loads(data)
    return starts[0]["status"], data, response_headers


class WebTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        app.middleware_stack = None
        self.settings = patch("api.routes.get_settings", return_value=SETTINGS)
        self.settings.start()
        # A test must never accidentally use the network, even with a local .env.
        self.network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Network forbidden in tests"))
        self.network.start()
    def tearDown(self):
        self.settings.stop(); self.network.stop()

    async def test_health_does_not_disclose_credentials(self):
        status,data,headers=await request("GET","/api/health")
        self.assertEqual(status,200); self.assertEqual(data["generation_mode"],"server_key")
        self.assertTrue(data["generation_enabled"])
        self.assertNotIn(OWNER,json.dumps(data)); self.assertNotIn("api_key_configured",data)
        self.assertEqual(headers[b"cache-control"],b"no-store")

    async def test_analyze_tasks_and_preferences(self):
        for prompt in ("What is 1+1?", "Write a Python sorting function.", "Extract names as JSON: Ada, Lin.", "Design a fault-tolerant distributed database and prove its consistency properties."):
            status,data,_=await request("POST","/api/analyze",{"prompt":prompt,"prefer_nvidia":False,"cost_sensitivity":1,"latency_sensitivity":0})
            self.assertEqual(status,200)
            rec=data["recommendation"]
            self.assertIn(rec["recommended_model_id"],[m["model_id"] for m in rec["all_candidates"]])
            self.assertNotIn("/100",rec["rationale"])
            self.assertIn("rule-based",data["classification"]["model_source"])

    async def test_owner_key_only_and_real_usage(self):
        seen=[]
        def generate(client, **kwargs):
            seen.append((client.api_key,client.base_url,kwargs))
            return "Answer",{"prompt_tokens":10,"completion_tokens":12,"total_tokens":22},0.2
        with patch.dict(os.environ,{"NEBIUS_BASE_URL":"https://attacker.invalid"}), patch.object(NebiusTokenFactoryClient,"generate",generate):
            status,data,_=await request("POST","/api/generate",{"prompt":"Hi"},headers={"X-Nebius-API-Key":VISITOR})
        self.assertEqual(status,200); self.assertEqual(data["total_tokens"],22)
        self.assertEqual(data["usage_source"],"provider"); self.assertTrue(data["cost_is_estimate"])
        self.assertEqual(seen[0][0],OWNER); self.assertEqual(seen[0][1],DEFAULT_BASE_URL)
        self.assertEqual(seen[0][2]["max_tokens"],1024)
        self.assertNotIn(OWNER,json.dumps(data)); self.assertNotIn(VISITOR,json.dumps(data))

    async def test_generation_disabled_or_missing_owner_key(self):
        for settings in (Settings(SecretStr(OWNER),False),Settings(SecretStr(""),True)):
            with patch("api.routes.get_settings",return_value=settings), patch.object(NebiusTokenFactoryClient,"generate") as provider:
                status,data,_=await request("POST","/api/generate",{"prompt":"Hi"},headers={"X-Nebius-API-Key":VISITOR})
                self.assertEqual(status,503); provider.assert_not_called()

    async def test_auto_and_manual_generation(self):
        payload={"prompt":"Write a Python sorting function.","prefer_nvidia":False,"cost_sensitivity":0.9}
        _,analysis,_=await request("POST","/api/analyze",payload)
        selected=analysis["recommendation"]["recommended_model_id"]
        with patch.object(NebiusTokenFactoryClient,"generate",return_value=("answer",{},0.2)) as provider:
            status,data,_=await request("POST","/api/generate",payload)
            self.assertEqual(status,200); self.assertEqual(data["model_id"],selected)
            self.assertEqual(data["usage_source"],"estimated")
            other=analysis["recommendation"]["all_candidates"][-1]["model_id"]
            status,data,_=await request("POST","/api/generate",{**payload,"model_id":other})
            self.assertEqual(status,200); self.assertEqual(data["model_id"],other)
            self.assertEqual(provider.call_count,2)

    async def test_unknown_models_and_invalid_input(self):
        with patch.object(NebiusTokenFactoryClient,"generate") as provider:
            for payload in ({"prompt":"Hi","model_id":"unknown"},{"prompt":" "},{"prompt":"x"*20001},{"prompt":"Hi","cost_sensitivity":2},{"prompt":"Hi","api_key":OWNER},{"prompt":"Hi","base_url":"https://example.com"}):
                status,data,_=await request("POST","/api/generate",payload,client=str(id(payload)))
                self.assertEqual(status,422); self.assertNotIn(OWNER,json.dumps(data))
            provider.assert_not_called()

    async def test_impossible_context_never_falls_back(self):
        service=get_service()
        with patch.object(service.catalog,"models",[replace(m,context_window_tokens=10) for m in service.catalog.models]):
            status,data,_=await request("POST","/api/analyze",{"prompt":"A long prompt"})
            self.assertEqual(status,422); self.assertEqual(data["error"]["code"],"no_eligible_model")

    async def test_json_constraint_applies_to_analysis_and_generation(self):
        service=get_service()
        with patch.object(service.catalog,"models",[replace(m,supports_json=False) for m in service.catalog.models]), patch.object(NebiusTokenFactoryClient,"generate") as provider:
            for path in ("/api/analyze","/api/generate"):
                status,_,_=await request("POST",path,{"prompt":"Extract JSON from Ada and Lin"})
                self.assertEqual(status,422)
            provider.assert_not_called()

    async def test_provider_errors_are_sanitized_without_retry(self):
        for exc,expected in [(ProviderError("provider_auth","Rejected",401),401),(ProviderError("provider_timeout","Timed out",504),504),(RuntimeError(OWNER + " secret prompt"),502)]:
            with patch.object(NebiusTokenFactoryClient,"generate",side_effect=exc) as provider:
                status,data,_=await request("POST","/api/generate",{"prompt":"Hi"})
                self.assertEqual(status,expected); self.assertNotIn(OWNER,json.dumps(data)); self.assertEqual(provider.call_count,1)

    async def test_generation_concurrency_limit(self):
        generation_slots.acquire(); generation_slots.acquire()
        try:
            status,data,_=await request("POST","/api/generate",{"prompt":"Hi"})
            self.assertEqual(status,429); self.assertEqual(data["error"]["code"],"busy")
        finally: generation_slots.release(); generation_slots.release()

    async def test_request_rate_limits(self):
        for _ in range(30):
            status,_,_=await request("POST","/api/analyze",{"prompt":"Hello"})
            self.assertEqual(status,200)
        status,_,_=await request("POST","/api/analyze",{"prompt":"Hello"})
        self.assertEqual(status,429)
        with patch.object(NebiusTokenFactoryClient,"generate",return_value=("Hi",{},0.1)):
            for _ in range(5):
                status,_,_=await request("POST","/api/generate",{"prompt":"Hi"})
                self.assertEqual(status,200)
            status,_,_=await request("POST","/api/generate",{"prompt":"Hi"})
            self.assertEqual(status,429)

    async def test_oversized_chunked_body_and_foreign_origin(self):
        status,_,_=await request("POST","/api/analyze",headers={"content-type":"application/json"},chunks=[b"x"*60000,b"x"*60000])
        self.assertEqual(status,413)
        status,_,_=await request("POST","/api/analyze",{"prompt":"Hi"},headers={"origin":"https://evil.example"})
        self.assertEqual(status,403)

    async def test_static_traversal_and_private_files(self):
        for path in ("/../.env","/icons/../../.env","/.env","/api/server.py","/cache/models/prompt_router.pkl","/not-found"):
            status,_,_=await request("GET",path)
            self.assertEqual(status,404,path)
        status,body,headers=await request("GET","/")
        self.assertEqual(status,200); self.assertIn(b'LLMOptiFit',body)
        self.assertIn(b"script-src 'self'",headers[b"content-security-policy"])

    async def test_pwa_assets_and_cache_contract(self):
        for path in ("/manifest.webmanifest","/service-worker.js","/privacy.html","/icons/icon-192.png","/icons/icon-512.png","/icons/apple-touch-icon.png"):
            status,_,_=await request("GET",path); self.assertEqual(status,200,path)
        sw=Path("web/service-worker.js").read_text()
        self.assertIn('request.method !== "GET"',sw)
        self.assertIn('!ASSETS.includes(url.pathname)',sw)
        self.assertNotIn('"/api/',sw)


class ProviderTests(unittest.TestCase):
    def test_explicit_client_ignores_env_and_redirects(self):
        with patch.dict(os.environ,{"NEBIUS_API_KEY":OWNER,"NEBIUS_BASE_URL":"https://evil.example"}), patch("tools.nebius_client.load_env_file") as load:
            client=NebiusTokenFactoryClient(api_key="",allow_env=False)
            self.assertFalse(client.is_configured()); self.assertEqual(client.base_url,DEFAULT_BASE_URL)
            load.assert_not_called()
        self.assertIsNone(NoRedirects().redirect_request(None,None,302,"",{},"https://evil.example"))

    def test_response_validation_and_http_errors(self):
        for response in ({},{"choices":[]},{"choices":[{"message":{"content":""}}]},{"choices":[{"message":{"content":42}}]}):
            fake=MagicMock(); fake.__enter__.return_value.read.return_value=json.dumps(response).encode()
            with patch("urllib.request.OpenerDirector.open",return_value=fake), self.assertRaises(ProviderError):
                NebiusTokenFactoryClient(api_key=OWNER,allow_env=False).generate("test","prompt")
        error=HTTPError("https://example.com",401,"bad",{},io.BytesIO(OWNER.encode()))
        with patch("urllib.request.OpenerDirector.open",side_effect=error):
            with self.assertRaises(ProviderError) as caught:
                NebiusTokenFactoryClient(api_key=OWNER,allow_env=False).generate("test","prompt")
            self.assertEqual(caught.exception.status,401); self.assertNotIn(OWNER,str(caught.exception))


if __name__ == "__main__": unittest.main()
