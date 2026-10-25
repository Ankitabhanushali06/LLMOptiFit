"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const presets = {
    simple: "What is the capital of Australia?",
    coding: "Write a Python function to merge overlapping intervals and explain its time complexity.",
    extraction: 'Extract the names and emails as a JSON array: Ada (ada@example.com), Lin (lin@example.com).',
    reasoning: "Design a Byzantine fault-tolerant consensus protocol for a financial settlement network. Explain invariants, partition handling, and tradeoffs."
  };
  let analysis = null, revision = 0, busy = false, answer = "", catalog = null, installPrompt = null, generationEnabled = false;
  let analysisController = null;
  const money = n => `$${Number(n).toFixed(6)}`;
  const payload = () => ({ prompt: $("prompt").value.trim(), cost_sensitivity: Number($("cost").value)/100,
    latency_sensitivity: Number($("speed").value)/100, prefer_nvidia: $("prefer-nvidia").checked });
  function say(id, text, error = false) { $(id).textContent = text; $(id).classList.toggle("error", error); }
  function updateButtons() {
    $("analyze-button").disabled = busy || !navigator.onLine;
    $("generate-button").disabled = busy || !navigator.onLine || !analysis || !generationEnabled || !$("consent").checked;
    $("generation-model").disabled = busy || !analysis;
    $("prompt-form").setAttribute("aria-busy", String(busy));
  }
  function invalidate() {
    revision++;
    if (analysisController) analysisController.abort();
    analysis = null;
    $("character-count").textContent = `${$("prompt").value.length.toLocaleString()} / 20,000`;
    $("classification-card").hidden = true;
    $("recommendation-content").hidden = true;
    $("recommendation-empty").hidden = false;
    $("recommendation-title").textContent = "Your recommendation appears here";
    $("generation-model").replaceChildren(new Option("Analyze a prompt first", ""));
    say("generation-estimate", "Analyze this prompt to review the model and estimated cost.");
    say("analysis-status", "Changes are kept here until you choose Analyze.");
    updateButtons();
  }
  async function api(path, options = {}) {
    if (!navigator.onLine) throw new Error("You are offline. Reconnect to analyze or generate.");
    const controller = new AbortController();
    const external = options.signal;
    const abort = () => controller.abort();
    external?.addEventListener("abort", abort, { once:true });
    if (external?.aborted) controller.abort();
    const timeout = setTimeout(abort, 90000);
    try {
      const response = await fetch(path, { ...options, signal:controller.signal, cache:"no-store", credentials:"omit" });
      let data;
      try { data = await response.json(); }
      catch { throw new Error("The server may be waking up. Wait about a minute and try again."); }
      if (!response.ok || data.success === false) throw new Error(data.error?.message || "The request could not be completed.");
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The request was interrupted or timed out. Generation is never retried automatically and may still be billed.");
      if (error instanceof TypeError) throw new Error("Could not reach the server. It may be waking up; try again after a minute.");
      throw error;
    } finally {
      clearTimeout(timeout);
      external?.removeEventListener("abort", abort);
    }
  }
  function renderAnalysis(data) {
    const c = data.classification, r = data.recommendation;
    $("classification-card").hidden = false;
    $("task-value").textContent = c.task_type.replaceAll("_", " ");
    $("tier-value").textContent = c.route_tier.toUpperCase();
    $("complexity-value").textContent = c.complexity_score.toFixed(3);
    $("drivers").replaceChildren();
    c.top_drivers.slice(0, 4).forEach(driver => {
      const tag = document.createElement("span"); tag.className = "chip";
      tag.textContent = `${driver.parameter.replaceAll("_", " ")}: ${driver.score.toFixed(2)}`;
      $("drivers").append(tag);
    });
    $("recommendation-empty").hidden = true;
    $("recommendation-content").hidden = false;
    $("recommendation-title").textContent = r.model_name;
    $("model-id").textContent = r.recommended_model_id;
    $("estimated-cost").textContent = money(r.estimated_cost_usd);
    $("estimated-latency").textContent = `${r.estimated_latency_sec.toFixed(2)}s`;
    $("fit-score").textContent = r.quality_score.toFixed(1);
    $("rationale").textContent = r.rationale;
    say("catalog-date", data.catalog_verified_at ? `Catalog verified ${new Date(data.catalog_verified_at).toLocaleDateString()}. Prices and availability can change.` : "Catalog snapshot has not been reverified. Confirm current pricing and availability with Nebius before generating.");
    $("generation-model").replaceChildren();
    r.all_candidates.forEach(m => $("generation-model").append(new Option(`${m.name} · ${money(m.cost_usd)} estimated`, m.model_id)));
    $("generation-model").value = r.recommended_model_id;
    $("alternatives").replaceChildren();
    const alternatives = r.top_pareto_candidates.filter(m => m.model_id !== r.recommended_model_id);
    if (!alternatives.length) {
      const note = document.createElement("p"); note.className = "hint";
      note.textContent = "This is the only candidate on the cost/fit frontier. Other eligible models are available in the model selector.";
      $("alternatives").append(note);
    }
    alternatives.forEach(m => {
      const row = document.createElement("div"); row.className = "alternative";
      const text = document.createElement("div"), title = document.createElement("strong"), detail = document.createElement("small");
      title.textContent = m.name; detail.textContent = `${money(m.cost_usd)} estimated · ${m.latency_sec.toFixed(2)}s · fit ${m.quality_match.toFixed(1)}`;
      text.append(title, detail);
      const button = document.createElement("button"); button.type = "button"; button.className = "button secondary"; button.textContent = "Select";
      button.addEventListener("click", () => { if (!busy) { $("generation-model").value = m.model_id; reviewModel(); $("generation-model").focus(); } });
      row.append(text, button); $("alternatives").append(row);
    });
    reviewModel();
  }
  function reviewModel() {
    if (!analysis) return;
    const chosen = analysis.recommendation.all_candidates.find(m => m.model_id === $("generation-model").value);
    say("generation-estimate", chosen ? `${chosen.name}: about ${money(chosen.cost_usd)} per request at expected output length. Actual usage may be higher, up to 1,024 output tokens. Selecting a model does not send a generation request.` : "Choose an eligible model.");
    updateButtons();
  }
  $("prompt").addEventListener("input", invalidate);
  for (const name of ["cost", "speed"]) $(name).addEventListener("input", () => { $(`${name}-value`).textContent = `${$(name).value}%`; invalidate(); });
  $("prefer-nvidia").addEventListener("change", invalidate);
  document.querySelectorAll("[data-preset]").forEach(button => button.addEventListener("click", () => {
    $("prompt").value = presets[button.dataset.preset]; invalidate(); $("prompt").focus();
  }));
  $("prompt-form").addEventListener("submit", async event => {
    event.preventDefault();
    if (busy || !$("prompt-form").reportValidity()) return;
    const current = revision;
    busy = true; analysisController = new AbortController(); updateButtons();
    say("analysis-status", "Analyzing… an idle free server can take about a minute to wake up.");
    try {
      const data = await api("/api/analyze", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload()), signal:analysisController.signal });
      if (current !== revision) return;
      analysis = data; renderAnalysis(data);
      say("analysis-status", "Recommendation ready. No generation or paid AI call was made.");
    } catch (error) { if (current === revision) say("analysis-status", error.message, true); }
    finally { busy = false; analysisController = null; updateButtons(); }
  });
  $("generation-model").addEventListener("change", reviewModel);
  $("consent").addEventListener("change", updateButtons);
  $("generate-button").addEventListener("click", async () => {
    if ($("generate-button").disabled || !analysis) return;
    const request = { ...payload(), model_id:$("generation-model").value };
    busy = true; updateButtons();
    $("generate-button").textContent = "Generating…";
    answer = ""; $("answer").textContent = "Waiting for Nebius…";
    $("answer-metrics").textContent = "";
    $("copy-answer").disabled = $("share-answer").disabled = true;
    say("answer-status", `Generating with ${request.model_id}. Your prompt is sent to Nebius. Please wait for this request to complete.`);
    try {
      const data = await api("/api/generate", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(request) });
      answer = data.generated_text;
      $("answer").textContent = answer;
      say("answer-status", `Completed using ${data.model_name}.`);
      $("answer-metrics").textContent = `${data.roundtrip_latency_sec.toFixed(2)}s measured · ${data.total_tokens} tokens (${data.usage_source}) · ${money(data.cost_usd)} calculated at catalog prices; provider billing may differ.`;
      $("copy-answer").disabled = $("share-answer").disabled = false;
    } catch (error) { $("answer").textContent = "No answer received."; say("answer-status", error.message, true); }
    finally { busy = false; $("generate-button").textContent = "Generate answer"; updateButtons(); }
  });
  async function copy(text, button) {
    try { await navigator.clipboard.writeText(text); const old = button.textContent; button.textContent = "Copied!"; setTimeout(() => button.textContent = old, 1800); }
    catch { say("connection-status", "Clipboard unavailable. Select and copy the text manually.", true); }
  }
  $("copy-model").addEventListener("click", () => copy($("model-id").textContent, $("copy-model")));
  $("copy-answer").addEventListener("click", () => copy(answer, $("copy-answer")));
  $("share-answer").addEventListener("click", async () => {
    if (!navigator.share) return copy(answer, $("share-answer"));
    try { await navigator.share({ title:"LLMOptiFit answer", text:answer }); }
    catch (error) { if (error.name !== "AbortError") say("answer-status", "Sharing unavailable. Use Copy instead."); }
  });
  document.querySelectorAll("[data-close]").forEach(button => button.addEventListener("click", () => $(button.dataset.close).close()));
  $("catalog-button").addEventListener("click", async () => {
    $("catalog-dialog").showModal();
    say("catalog-status", "Loading catalog… an idle server may take about a minute.");
    try { catalog = await api("/api/catalog"); renderCatalog(); }
    catch (error) { say("catalog-status", error.message, true); }
  });
  $("catalog-search").addEventListener("input", renderCatalog);
  function renderCatalog() {
    if (!catalog) return;
    const query = $("catalog-search").value.toLowerCase();
    const models = catalog.catalog.filter(m => `${m.name} ${m.model_id} ${m.vendor} ${m.tier} ${m.task_affinities.join(" ")}`.toLowerCase().includes(query));
    say("catalog-status", `${models.length} of ${catalog.count} models. ${catalog.verified_at ? `Snapshot checked ${new Date(catalog.verified_at).toLocaleDateString()}.` : "Snapshot not reverified."} Prices and availability can change.`);
    $("catalog-list").replaceChildren();
    models.forEach(m => {
      const item = document.createElement("article"); item.className = "catalog-item";
      const name = document.createElement("h3"), id = document.createElement("code"), detail = document.createElement("p");
      name.textContent = m.name; id.textContent = m.model_id; detail.className = "hint";
      detail.textContent = `${m.vendor} · ${m.tier} · ${m.context_window_tokens.toLocaleString()} context tokens · $${m.input_price_per_million}/M input · $${m.output_price_per_million}/M output · ${m.tokens_per_second} tokens/s estimated`;
      item.append(name, id, detail); $("catalog-list").append(item);
    });
  }
  window.addEventListener("beforeinstallprompt", event => {
    event.preventDefault(); installPrompt = event; $("native-install").hidden = false;
  });
  $("install-button").addEventListener("click", () => $("install-dialog").showModal());
  $("native-install").addEventListener("click", async () => {
    if (!installPrompt) return;
    const event = installPrompt; installPrompt = null; $("native-install").hidden = true;
    try { await event.prompt(); }
    catch { say("connection-status", "Use the browser menu to install this app."); }
  });
  function connectivity() {
    say("connection-status", navigator.onLine ? "Recommendations are free. Live generation is provided by the app when enabled. An idle server may take about a minute to wake up." : "You are offline. The interface is available; reconnect to analyze or generate.", !navigator.onLine);
    updateButtons();
  }
  window.addEventListener("online", connectivity);
  window.addEventListener("offline", connectivity);
  async function loadAvailability() {
    try {
      const health = await api("/api/health");
      generationEnabled = health.generation_enabled === true;
      say("generation-availability", generationEnabled ? "Generation is available. No API key or payment is required from visitors." : "Live generation is currently disabled. You can still analyze prompts and compare models.");
    } catch { say("generation-availability", "Unable to check generation availability. Reconnect or reload to try again."); }
    updateButtons();
  }
  window.addEventListener("online", loadAvailability);
  loadAvailability();
  if ("serviceWorker" in navigator && navigator.onLine) navigator.serviceWorker.register("/service-worker.js").catch(() => {
    if (navigator.onLine) say("connection-status", "The app works online, but offline installation is unavailable in this browser.");
  });
  connectivity();
})();
