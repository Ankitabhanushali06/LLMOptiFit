/**
 * LLMOptiFit — Modern Frontend Logic & Pareto Frontier Orchestrator
 * =================================================================
 * Connects prompt input to LLMOptiFit classification, Pareto frontier visualizer,
 * and live Nebius Token Factory generation.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM References
  const promptInput = document.getElementById("prompt-input");
  const inTokenCount = document.getElementById("in-token-count");
  const presetPills = document.getElementById("preset-pills");
  const costSlider = document.getElementById("cost-slider");
  const costVal = document.getElementById("cost-val");
  const latencySlider = document.getElementById("latency-slider");
  const latencyVal = document.getElementById("latency-val");
  const preferNvidiaToggle = document.getElementById("prefer-nvidia-toggle");
  
  const analyzeBtn = document.getElementById("analyze-btn");
  const generateBtn = document.getElementById("generate-btn");
  const btnSpinner = document.getElementById("btn-spinner");
  const btnText = document.getElementById("btn-text");

  // Telemetry DOM
  const telemetryTaskIcon = document.getElementById("telemetry-task-icon");
  const telemetryTaskName = document.getElementById("telemetry-task-name");
  const telemetryTierPill = document.getElementById("telemetry-tier-pill");
  const telemetryComplexityNum = document.getElementById("telemetry-complexity-num");
  const telemetryComplexityFill = document.getElementById("telemetry-complexity-fill");
  const telemetryTokens = document.getElementById("telemetry-tokens");
  const driverTags = document.getElementById("driver-tags");

  // Winner DOM
  const winnerCard = document.getElementById("winner-card");
  const winnerModelName = document.getElementById("winner-model-name");
  const winnerModelId = document.getElementById("winner-model-id");
  const winnerSavingsPill = document.getElementById("winner-savings-pill");
  const winnerVendorName = document.getElementById("winner-vendor-name");
  const winnerNvSpark = document.getElementById("winner-nv-spark");
  const copyModelIdBtn = document.getElementById("copy-model-id-btn");
  const copyBtnLabel = document.getElementById("copy-btn-label");
  const kpiCost = document.getElementById("kpi-cost");
  const kpiSavingsText = document.getElementById("kpi-savings-text");
  const kpiLatency = document.getElementById("kpi-latency");
  const kpiQuality = document.getElementById("kpi-quality");
  const kpiStatus = document.getElementById("kpi-status");
  const winnerRationale = document.getElementById("winner-rationale");

  // Terminal DOM
  const terminalModelName = document.getElementById("terminal-model-name");
  const statLatency = document.getElementById("stat-latency");
  const statTokens = document.getElementById("stat-tokens");
  const statCost = document.getElementById("stat-cost");
  const terminalContent = document.getElementById("terminal-content");
  const copyOutputBtn = document.getElementById("copy-output-btn");

  // Catalog Modal DOM
  const catalogModal = document.getElementById("catalog-modal-backdrop");
  const openCatalogBtn = document.getElementById("open-catalog-btn");
  const closeCatalogBtn = document.getElementById("close-catalog-btn");
  const catalogSearch = document.getElementById("catalog-search");
  const catalogTierFilters = document.getElementById("catalog-tier-filters");
  const catalogTableBody = document.getElementById("catalog-table-body");

  // Application State
  let currentAnalysis = null;
  let catalogCache = [];
  let currentFilterTier = "all";
  let chartPointData = []; // for mouse hover detection

  // Task Icon Map
  const TASK_ICONS = {
    coding: "💻",
    mathematical_reasoning: "🔢",
    analysis: "📊",
    planning: "🗺️",
    research: "🔬",
    generation: "✍️",
    extraction: "📋",
    classification: "🏷️",
  };

  // Preset Prompts Catalog
  const PRESET_PROMPTS = {
    coding: "Write a fast Python script to scrape product prices from an HTML table and save them into a SQLite database with indexed lookups.",
    sql: "write sql query that extract manager salary is less than employee salary",
    math: "Solve for x: 3x^2 - 12x + 9 = 0. Show the step-by-step factorization and roots.",
    architecture: "Architect a globally distributed, Byzantine fault-tolerant consensus protocol for a financial settlement network. Formally specify state transitions, partition handling, and cryptographic invariants.",
    extraction: "Extract all key technical contact names, emails, and primary cloud providers from this incident log and output valid JSON conforming to the schema.",
    nano: "what is 1+1",
  };

  // 1. Initialize Token Meter & Debounced Real-Time Analysis
  let debounceTimer = null;
  function updateTokenMeter() {
    const text = promptInput.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    const estTokens = Math.max(0, Math.round(words * 1.33));
    inTokenCount.textContent = `~${estTokens}`;
    telemetryTokens.textContent = `~${estTokens} tokens`;
  }

  promptInput.addEventListener("input", () => {
    // Clear active preset highlight when typing custom prompt
    presetPills.querySelectorAll(".pill-btn").forEach(p => p.classList.remove("active"));
    updateTokenMeter();
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      runAnalysis();
    }, 180);
  });
  updateTokenMeter();

  // 2. Preset Pill Buttons
  presetPills.addEventListener("click", (e) => {
    const pill = e.target.closest(".pill-btn");
    if (!pill) return;

    presetPills.querySelectorAll(".pill-btn").forEach(p => p.classList.remove("active"));
    pill.classList.add("active");

    const presetKey = pill.getAttribute("data-preset");
    if (PRESET_PROMPTS[presetKey]) {
      promptInput.value = PRESET_PROMPTS[presetKey];
      updateTokenMeter();
      runAnalysis();
    }
  });

  // 3. Tradeoff Sliders (Real-time updates)
  costSlider.addEventListener("input", () => {
    costVal.textContent = `${costSlider.value}%`;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runAnalysis, 100);
  });

  latencySlider.addEventListener("input", () => {
    latencyVal.textContent = `${latencySlider.value}%`;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runAnalysis, 100);
  });

  preferNvidiaToggle.addEventListener("change", runAnalysis);

  // 4. API Analysis (Inspect Route)
  async function runAnalysis() {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    try {
      const payload = {
        prompt,
        cost_sensitivity: parseInt(costSlider.value) / 100,
        latency_sensitivity: parseInt(latencySlider.value) / 100,
        prefer_nvidia: preferNvidiaToggle.checked,
      };

      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json();
      if (!data.success) {
        throw new Error(data.error || "Analysis failed");
      }

      currentAnalysis = data;
      renderAnalysis(data);
    } catch (err) {
      console.error("Analysis Error:", err);
    }
  }

  // 5. Render Analysis & Telemetry
  function renderAnalysis(data) {
    const { classification, recommendation } = data;

    // Telemetry updates
    const taskType = classification.task_type.toLowerCase();
    telemetryTaskName.textContent = classification.task_type.toUpperCase();
    telemetryTaskIcon.textContent = TASK_ICONS[taskType] || "⚡";
    telemetryTierPill.textContent = classification.route_tier.toUpperCase();

    // Tier badge color
    if (classification.route_tier === "ultra") {
      telemetryTierPill.style.background = "rgba(139, 92, 246, 0.2)";
      telemetryTierPill.style.color = "#C4B5FD";
      telemetryTierPill.style.borderColor = "rgba(139, 92, 246, 0.4)";
    } else if (classification.route_tier === "super") {
      telemetryTierPill.style.background = "rgba(6, 182, 212, 0.2)";
      telemetryTierPill.style.color = "#67E8F9";
      telemetryTierPill.style.borderColor = "rgba(6, 182, 212, 0.4)";
    } else {
      telemetryTierPill.style.background = "rgba(16, 185, 129, 0.2)";
      telemetryTierPill.style.color = "#6EE7B7";
      telemetryTierPill.style.borderColor = "rgba(16, 185, 129, 0.4)";
    }

    const cScore = classification.complexity_score;
    telemetryComplexityNum.textContent = cScore.toFixed(3);
    telemetryComplexityFill.style.width = `${Math.min(100, Math.round(cScore * 100))}%`;

    // Drivers
    driverTags.innerHTML = "";
    if (classification.top_drivers && classification.top_drivers.length) {
      classification.top_drivers.slice(0, 3).forEach(d => {
        const span = document.createElement("span");
        span.className = "driver-chip";
        span.innerHTML = `<span class="driver-name">${d.parameter}</span>: <strong>${d.score.toFixed(2)}</strong>`;
        driverTags.appendChild(span);
      });
    }

    // Winner Hero Card Updates
    winnerModelName.textContent = recommendation.model_name;
    winnerModelId.textContent = recommendation.recommended_model_id;
    winnerVendorName.textContent = recommendation.vendor.toUpperCase();

    if (recommendation.is_nvidia) {
      winnerCard.style.borderColor = "rgba(16, 185, 129, 0.55)";
      winnerCard.style.boxShadow = "0 10px 30px -5px rgba(0, 0, 0, 0.7), 0 0 25px -3px rgba(16, 185, 129, 0.35)";
      winnerNvSpark.style.display = "inline-block";
    } else {
      winnerCard.style.borderColor = "rgba(6, 182, 212, 0.45)";
      winnerCard.style.boxShadow = "0 10px 30px -5px rgba(0, 0, 0, 0.7), 0 0 25px -3px rgba(6, 182, 212, 0.25)";
      winnerNvSpark.style.display = "none";
    }

    if (recommendation.cost_savings_vs_frontier_pct > 0) {
      winnerSavingsPill.textContent = `Save ${recommendation.cost_savings_vs_frontier_pct.toFixed(1)}% vs Frontier`;
      winnerSavingsPill.style.display = "inline-block";
      kpiSavingsText.textContent = `${recommendation.cost_savings_vs_frontier_pct.toFixed(1)}% cheaper than 405B`;
    } else {
      winnerSavingsPill.textContent = `Frontier Capable`;
      kpiSavingsText.textContent = `Deployed at Frontier Scale`;
    }

    kpiCost.textContent = `$${recommendation.estimated_cost_usd.toFixed(6)}`;
    kpiLatency.textContent = `${recommendation.estimated_latency_sec.toFixed(2)}s`;
    kpiQuality.textContent = `${recommendation.quality_score.toFixed(1)} / 100`;
    winnerRationale.textContent = recommendation.rationale;

    terminalModelName.textContent = recommendation.recommended_model_id;
  }

  // 7. Live Generation Handler
  generateBtn.addEventListener("click", async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    // Synchronize telemetry and recommendation analysis immediately
    runAnalysis();

    generateBtn.classList.add("loading");
    btnSpinner.style.display = "inline-block";
    btnText.textContent = "Calling Nebius...";
    terminalContent.textContent = "Connecting to Nebius Token Factory endpoint...\nDispatching chat completion request to Pareto-optimal model...";

    try {
      const payload = {
        prompt,
        cost_sensitivity: parseInt(costSlider.value) / 100,
        latency_sensitivity: parseInt(latencySlider.value) / 100,
        prefer_nvidia: preferNvidiaToggle.checked,
      };

      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!data.success) {
        throw new Error(data.error || "Generation request failed");
      }

      // Update terminal stats
      terminalModelName.textContent = data.model_id;
      statLatency.textContent = `⚡ ${data.roundtrip_latency_sec.toFixed(2)}s`;
      statTokens.textContent = `📊 ${data.total_tokens} tok (${data.input_tokens} in / ${data.output_tokens} out)`;
      statCost.textContent = `💰 $${data.cost_usd.toFixed(6)}`;

      // Typewriter Effect
      streamTypewriter(data.generated_text);

      // Re-run analysis in parallel to update visual state if not yet analyzed
      if (!currentAnalysis) {
        runAnalysis();
      }

    } catch (err) {
      console.error("Generation Error:", err);
      terminalContent.textContent = `❌ Error: ${err.message}\n\nPlease check your NEBIUS_API_KEY in .env.`;
    } finally {
      generateBtn.classList.remove("loading");
      btnSpinner.style.display = "none";
      btnText.textContent = "Route & Generate Live";
    }
  });

  analyzeBtn.addEventListener("click", runAnalysis);

  // Typewriter streaming effect
  function streamTypewriter(fullText) {
    terminalContent.textContent = "";
    let i = 0;
    const speed = fullText.length > 800 ? 5 : 12; // fast streaming for long responses

    function typeChar() {
      if (i < fullText.length) {
        terminalContent.textContent += fullText.charAt(i);
        i++;
        setTimeout(typeChar, speed);
      }
    }
    typeChar();
  }

  // 8. Copy to Clipboard Handlers
  copyModelIdBtn.addEventListener("click", () => {
    const text = winnerModelId.textContent;
    navigator.clipboard.writeText(text).then(() => {
      copyBtnLabel.textContent = "Copied!";
      setTimeout(() => copyBtnLabel.textContent = "Copy", 2000);
    });
  });

  copyOutputBtn.addEventListener("click", () => {
    const text = terminalContent.textContent;
    navigator.clipboard.writeText(text).then(() => {
      const orig = copyOutputBtn.innerHTML;
      copyOutputBtn.innerHTML = `<span>Copied!</span>`;
      setTimeout(() => copyOutputBtn.innerHTML = orig, 2000);
    });
  });

  // 9. Catalog Modal Handlers
  openCatalogBtn.addEventListener("click", openCatalog);
  closeCatalogBtn.addEventListener("click", () => catalogModal.classList.remove("open"));
  catalogModal.addEventListener("click", (e) => {
    if (e.target === catalogModal) catalogModal.classList.remove("open");
  });

  async function openCatalog() {
    catalogModal.classList.add("open");
    if (catalogCache.length === 0) {
      try {
        const res = await fetch("/api/catalog");
        const data = await res.json();
        catalogCache = data.catalog || [];
        renderCatalogTable();
      } catch (err) {
        console.error("Failed to load catalog:", err);
      }
    } else {
      renderCatalogTable();
    }
  }

  catalogSearch.addEventListener("input", renderCatalogTable);

  catalogTierFilters.addEventListener("click", (e) => {
    const btn = e.target.closest(".filter-pill");
    if (!btn) return;
    catalogTierFilters.querySelectorAll(".filter-pill").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilterTier = btn.getAttribute("data-filter");
    renderCatalogTable();
  });

  function renderCatalogTable() {
    const query = catalogSearch.value.toLowerCase().trim();
    catalogTableBody.innerHTML = "";

    const filtered = catalogCache.filter(m => {
      const matchesSearch = m.name.toLowerCase().includes(query) ||
                            m.model_id.toLowerCase().includes(query) ||
                            m.vendor.toLowerCase().includes(query);

      let matchesTier = true;
      if (currentFilterTier === "nvidia") {
        matchesTier = m.is_nvidia;
      } else if (currentFilterTier !== "all") {
        matchesTier = m.tier.toLowerCase() === currentFilterTier;
      }

      return matchesSearch && matchesTier;
    });

    filtered.forEach(m => {
      const tr = document.createElement("tr");
      if (m.is_nvidia) tr.className = "nv-row";

      const nvIcon = m.is_nvidia ? `<span style="color:#10B981; margin-right:4px;">◆</span>` : "";
      const ctxK = Math.round(m.context_window_tokens / 1000);

      tr.innerHTML = `
        <td>
          <div style="font-weight:600; color:#FFFFFF;">${nvIcon}${m.name}</div>
          <div class="font-mono" style="font-size:11px; color:#64748B;">${m.model_id}</div>
        </td>
        <td><span style="text-transform:uppercase; font-size:11px; font-weight:600;">${m.vendor}</span></td>
        <td><span class="tier-pill" style="font-size:11px;">${m.tier.toUpperCase()}</span></td>
        <td class="font-mono">${ctxK}k tok</td>
        <td class="font-mono">$${m.input_price_per_million.toFixed(2)}</td>
        <td class="font-mono">$${m.output_price_per_million.toFixed(2)}</td>
        <td class="font-mono">${Math.round(m.tokens_per_second)} t/s</td>
        <td>
          <button class="select-model-btn" data-id="${m.model_id}">Select</button>
        </td>
      `;

      tr.querySelector(".select-model-btn").addEventListener("click", () => {
        catalogModal.classList.remove("open");
        // Trigger live generation forced on this model
        triggerDirectModel(m.model_id);
      });

      catalogTableBody.appendChild(tr);
    });
  }

  async function triggerDirectModel(modelId) {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    generateBtn.classList.add("loading");
    btnSpinner.style.display = "inline-block";
    btnText.textContent = `Calling ${modelId.split('/')[1] || modelId}...`;
    terminalContent.textContent = `Dispatching request directly to selected model: ${modelId}...`;

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, model_id: modelId }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error);

      terminalModelName.textContent = data.model_id;
      statLatency.textContent = `⚡ ${data.roundtrip_latency_sec.toFixed(2)}s`;
      statTokens.textContent = `📊 ${data.total_tokens} tok (${data.input_tokens} in / ${data.output_tokens} out)`;
      statCost.textContent = `💰 $${data.cost_usd.toFixed(6)}`;
      streamTypewriter(data.generated_text);
    } catch (err) {
      terminalContent.textContent = `❌ Error: ${err.message}`;
    } finally {
      generateBtn.classList.remove("loading");
      btnSpinner.style.display = "none";
      btnText.textContent = "Route & Generate Live";
    }
  }

  // Initial Run on Page Load
  runAnalysis();
});
