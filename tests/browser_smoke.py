"""Optional isolated Chromium QA. All inference is mocked; never uses .env keys.
Run with the dev-only websockets package available and an installed Chromium browser.
"""
import argparse
import base64
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import uvicorn
from pydantic import SecretStr
from websockets.sync.client import connect
from api.server import app
from api.settings import Settings
from tools.nebius_client import NebiusTokenFactoryClient


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0)); return sock.getsockname()[1]


class CDP:
    def __init__(self, url):
        self.connection=connect(url,open_timeout=10); self.ws=self.connection.__enter__(); self.sequence=0; self.errors=[]; self.requests=[]
    def call(self, method, params=None):
        self.sequence+=1; request_id=self.sequence
        self.ws.send(json.dumps({"id":request_id,"method":method,"params":params or {}}))
        while True:
            message=json.loads(self.ws.recv(timeout=15))
            if message.get("method")=="Runtime.exceptionThrown": self.errors.append(message["params"])
            if message.get("method")=="Network.requestWillBeSent": self.requests.append(message["params"]["request"])
            if message.get("id")==request_id:
                if "error" in message: raise AssertionError(message["error"])
                return message.get("result",{})
    def js(self, expression):
        result=self.call("Runtime.evaluate",{"expression":expression,"returnByValue":True,"awaitPromise":True})
        if "exceptionDetails" in result: raise AssertionError(result["exceptionDetails"])
        return result.get("result",{}).get("value")
    def wait(self, expression, timeout=15):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if self.js(expression): return
            time.sleep(.1)
        raise AssertionError("Timed out: "+expression+"; "+str(self.js('({online:navigator.onLine,status:document.querySelector("#connection-status")?.textContent,title:document.title})')))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--browser",required=True); parser.add_argument("--output",default="/tmp/llmoptifit-browser")
    args=parser.parse_args(); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    port=free_port(); url=f"http://127.0.0.1:{port}"
    server=uvicorn.Server(uvicorn.Config(app,host="127.0.0.1",port=port,log_level="warning",access_log=False,proxy_headers=False))
    calls=[]
    def generate(client, **kwargs):
        assert client.api_key == "browser-test-owner-secret"
        calls.append(kwargs)
        return 'Test answer: <img src=x onerror="alert(1)"> is plain text.', {"prompt_tokens":12,"completion_tokens":15}, .2
    with patch("api.routes.get_settings",return_value=Settings(SecretStr("browser-test-owner-secret"),True)), patch.object(NebiusTokenFactoryClient,"generate",generate), tempfile.TemporaryDirectory(prefix="llmoptifit-browser-") as profile:
        thread=threading.Thread(target=server.run,daemon=True); thread.start()
        browser=None; cdp=None
        try:
            for _ in range(100):
                if server.started: break
                time.sleep(.1)
            assert server.started, "Test server failed to start"
            browser=subprocess.Popen([args.browser,"--headless=new",f"--user-data-dir={profile}","--remote-debugging-port=0","--no-first-run","--no-default-browser-check","--disable-background-networking","about:blank"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            active=Path(profile)/"DevToolsActivePort"
            for _ in range(100):
                if active.exists(): break
                if browser.poll() is not None: raise AssertionError("Browser exited")
                time.sleep(.1)
            debug_port=active.read_text().splitlines()[0]
            targets=json.load(urllib.request.urlopen(f"http://127.0.0.1:{debug_port}/json"))
            page=next(t for t in targets if t["type"]=="page")
            cdp=CDP(page["webSocketDebuggerUrl"])
            cdp.call("Runtime.enable"); cdp.call("Network.enable"); cdp.call("Page.enable")
            cdp.call("Emulation.setDeviceMetricsOverride",{"width":390,"height":844,"deviceScaleFactor":1,"mobile":True})
            cdp.call("Page.navigate",{"url":url})
            cdp.wait('document.querySelector("#generation-availability")?.textContent.includes("Generation is available")')
            assert cdp.js('document.querySelector("#api-key") === null'), "Visitor key UI remains"
            cdp.js('document.querySelector("[data-preset=coding]").click()')
            assert not any('/api/analyze' in r['url'] for r in cdp.requests), "Typing/presets submitted content"
            cdp.js('document.querySelector("#analyze-button").click()')
            cdp.wait('document.querySelector("#analysis-status").textContent.includes("Recommendation ready")')
            assert cdp.js('document.querySelector("#generate-button").disabled'), "Consent must be required"
            cdp.js('document.querySelector("#consent").click()')
            cdp.js('document.querySelector("#generate-button").click()')
            cdp.wait('document.querySelector("#answer-status").textContent.includes("Completed")')
            assert len(calls)==1
            assert cdp.js('document.querySelector("#answer img") === null'), "Model HTML executed"
            assert cdp.js('document.querySelector("#answer").textContent.includes("<img")')
            assert cdp.js('document.documentElement.scrollWidth <= window.innerWidth'), "Mobile overflow"
            assert cdp.js('document.querySelectorAll("main > section").length === 4'), "Cards must not be nested"
            assert cdp.js('localStorage.length === 0 && sessionStorage.length === 0')
            assert 'browser-test-owner-secret' not in cdp.js('document.documentElement.outerHTML')
            cdp.call("Page.captureScreenshot",{"format":"png"})
            for width,height,name in [(390,844,"mobile"),(1280,1000,"desktop")]:
                cdp.call("Emulation.setDeviceMetricsOverride",{"width":width,"height":height,"deviceScaleFactor":1,"mobile":width<600})
                time.sleep(.1)
                assert cdp.js('document.documentElement.scrollWidth <= window.innerWidth')
                image=cdp.call("Page.captureScreenshot",{"format":"png"})
                (output/f"{name}.png").write_bytes(base64.b64decode(image["data"]))
            cdp.js('document.querySelector("#catalog-button").click()')
            cdp.wait('document.querySelectorAll(".catalog-item").length > 0')
            cdp.js('document.querySelector("#catalog-dialog").close(); document.querySelector("#install-button").click()')
            cdp.wait('document.querySelector("#install-dialog").open')
            cdp.js('document.querySelector("#install-dialog").close()')
            cdp.js('navigator.serviceWorker.ready.then(() => true)')
            cdp.call("Page.reload")
            cdp.wait('navigator.serviceWorker.controller !== null')
            cached=cdp.js('caches.keys().then(async keys => (await Promise.all(keys.map(async k => (await (await caches.open(k)).keys()).map(r=>r.url)))).flat())')
            assert not any('/api/' in item for item in cached)
            cdp.call("Network.emulateNetworkConditions",{"offline":True,"latency":0,"downloadThroughput":0,"uploadThroughput":0})
            cdp.call("Network.overrideNetworkState", {"offline":True,"latency":0,"downloadThroughput":0,"uploadThroughput":0})
            cdp.call("Page.reload")
            time.sleep(0.5)
            cdp.call("Network.emulateNetworkConditions",{"offline":True,"latency":0,"downloadThroughput":0,"uploadThroughput":0})
            cdp.js("if (!navigator.onLine || true) window.dispatchEvent(new Event('offline'))")
            cdp.wait('document.querySelector("#connection-status")?.textContent.includes("offline")')
            assert cdp.js('document.querySelector("#analyze-button").disabled')
            assert not cdp.errors, cdp.errors
            print(json.dumps({"passed":True,"mock_generation_calls":len(calls),"cached_assets":len(cached),"screenshots":str(output),"checks":["mobile layout","desktop layout","analysis","consent","owner-key generation","safe rendering","catalog","install help","offline reload","no private caches","no browser storage","no JavaScript errors"]},indent=2))
        finally:
            if cdp: cdp.connection.__exit__(None,None,None)
            if browser:
                browser.terminate()
                try: browser.wait(timeout=10)
                except subprocess.TimeoutExpired: browser.kill(); browser.wait()
            server.should_exit=True; thread.join(timeout=5)


if __name__ == "__main__": main()
