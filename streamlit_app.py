"""
LLMOptiFit — Streamlit Intelligent LLM Router Dashboard
======================================================
Production-grade interactive application for Pareto-optimal model routing
powered by trained prompt parameter classification and Nebius Token Factory.

Usage:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Setup paths and environment
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.prompt_classifier import classify_prompt
from tools.nebius_recommender import NebiusParetoRecommender
from tools.nebius_client import NebiusTokenFactoryClient

CATALOG_PATH = PROJECT_ROOT / "config" / "nebius_catalog.json"

# Page configuration
st.set_page_config(
    page_title="LLMOptiFit — Pareto-Optimal LLM Router",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Cyber-Aesthetic matching Nebius & NVIDIA design)
st.markdown("""
<style>
    /* Dark cyber theme accents */
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Gradient Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 182, 212, 0.08) 50%, rgba(139, 92, 246, 0.12) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #10B981, #06B6D4, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 14px;
        margin-top: 4px;
    }
    
    /* Metric & KPI Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .winner-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15);
        margin-bottom: 20px;
    }
    
    .tier-badge-nano {
        background: rgba(16, 185, 129, 0.2);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
    }
    .tier-badge-super {
        background: rgba(6, 182, 212, 0.2);
        color: #67E8F9;
        border: 1px solid rgba(6, 182, 212, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
    }
    .tier-badge-ultra {
        background: rgba(139, 92, 246, 0.2);
        color: #C4B5FD;
        border: 1px solid rgba(139, 92, 246, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
    }
    
    .nvidia-tag {
        background: rgba(118, 185, 0, 0.18);
        color: #76B900;
        border: 1px solid rgba(118, 185, 0, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Load Catalog
@st.cache_data
def load_catalog() -> list[dict]:
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

catalog_data = load_catalog()
nebius_client = NebiusTokenFactoryClient()

# Sidebar: Configuration & Tradeoffs
with st.sidebar:
    st.markdown("### 🎛️ Routing Controls")
    
    cost_sensitivity = st.slider(
        "Cost Sensitivity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Higher values favor cheaper models (Nano tier / 8B-14B models)."
    )
    
    latency_sensitivity = st.slider(
        "Latency Sensitivity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Higher values favor high-throughput / low-latency models."
    )
    
    prefer_nvidia = st.toggle(
        "Prefer NVIDIA Open-Source",
        value=True,
        help="Applies a positive utility bonus for NVIDIA Nemotron models hosted on Nebius."
    )
    
    st.markdown("---")
    st.markdown("### 🔌 Nebius Connection")
    if nebius_client.is_configured():
        st.success("✅ Nebius API Key configured")
    else:
        st.warning("⚠️ Nebius API Key not set in `.env` (Demo mode)")
    
    st.markdown(f"**Models Available:** {len(catalog_data)} models")
    
    # Presets
    st.markdown("---")
    st.markdown("### ⚡ Quick Benchmarks")
    preset_choice = st.radio(
        "Load Preset Prompt:",
        [
            "Custom Prompt",
            "💻 Python Scraper (Coding)",
            "⚙ Byzantine Consensus (Planning/Architecture)",
            "⚡ Simple '1 + 1' (Nano)",
            "📊 Manager Salary SQL (Analysis)",
            "🔬 Location Lookup (Research)",
            "📋 Extract JSON Schema (Extraction)",
        ],
        index=0,
    )

# Prompt mapping
PRESETS = {
    "💻 Python Scraper (Coding)": "Write a fast Python script to scrape product prices from an HTML table and save them into a SQLite database with indexed lookups.",
    "⚙ Byzantine Consensus (Planning/Architecture)": "Architect a globally distributed, Byzantine fault-tolerant consensus protocol for a financial settlement network. Formally specify state transitions, partition handling, and cryptographic invariants.",
    "⚡ Simple '1 + 1' (Nano)": "what is 1+1",
    "📊 Manager Salary SQL (Analysis)": "Write an optimized SQL query to extract all employees whose current salary is greater than their direct reporting manager's salary.",
    "🔬 Location Lookup (Research)": "Where is the headquarters of the European Organization for Nuclear Research (CERN), and what is their primary research accelerator?",
    "📋 Extract JSON Schema (Extraction)": "Extract all key technical contact names, email addresses, and server regions from the following error logs and format as valid RFC8259 JSON.",
}

default_prompt = PRESETS.get(preset_choice, "Write a Python script to sort a dictionary by value.")

# Top Banner
st.markdown("""
<div class="hero-banner">
    <div>
        <h1 class="hero-title">LLMOptiFit — Pareto-Optimal Router</h1>
        <div class="hero-subtitle">Intelligent Prompt Classifier &bull; Nebius Token Factory &bull; NVIDIA Nemotron Acceleration</div>
    </div>
    <div>
        <span class="nvidia-tag">◆ NVIDIA Open-Source Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Prompt Input
prompt_text = st.text_area(
    "📝 Prompt Input",
    value=default_prompt,
    height=110,
    placeholder="Enter your prompt here to see real-time task classification and Pareto model selection..."
)

# Estimate tokens
word_count = len(prompt_text.split()) if prompt_text.strip() else 0
est_tokens = max(0, int(word_count * 1.33))

c1, c2 = st.columns([1, 1])

# Classify and recommend
if prompt_text.strip():
    clf = classify_prompt(prompt_text)
    recommender = NebiusParetoRecommender()
    rec = recommender.recommend(
        prompt_text=prompt_text,
        task_type=clf.scores.predicted_task_type,
        complexity_score=clf.weighted_score,
        route_tier=clf.route_tier,
        cost_sensitivity=cost_sensitivity,
        latency_sensitivity=latency_sensitivity,
        prefer_nvidia=prefer_nvidia,
    )
    
    # Left Column: Prompt Classification Telemetry
    with c1:
        st.markdown("### 🔍 Classification Telemetry")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Task Type", clf.scores.predicted_task_type.upper())
        m2.metric("Route Tier", clf.route_tier.upper())
        m3.metric("Complexity", f"{clf.weighted_score:.3f}")
        m4.metric("Est. Tokens", f"~{est_tokens}")
        
        # Complexity progress bar
        st.progress(min(1.0, max(0.0, float(clf.weighted_score))))
        
        # Top Drivers
        st.markdown("**Top Rubric Drivers:**")
        if clf.rationale and "top_drivers" in clf.rationale:
            driver_cols = st.columns(len(clf.rationale["top_drivers"]))
            for i, d in enumerate(clf.rationale["top_drivers"]):
                driver_cols[i].markdown(f"• **{d['parameter']}**: `{d['score']:.2f}`")
        
        # Rubric parameters breakdown chart
        rubric_data = {
            "Parameter": [
                "Reasoning", "Task Load", "Decision Search", "Accuracy",
                "Instruction", "Context", "Output Spec", "Domain Difficulty"
            ],
            "Score": [
                clf.scores.reasoning_complexity,
                clf.scores.task_type,
                clf.scores.decision_search_complexity,
                clf.scores.accuracy_requirement,
                clf.scores.instruction_complexity,
                clf.scores.context_requirement,
                clf.scores.output_complexity,
                clf.scores.domain_difficulty,
            ]
        }
        df_rubric = pd.DataFrame(rubric_data)
        
        chart_rubric = alt.Chart(df_rubric).mark_bar(cornerRadius=4).encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Parameter:N", sort="-x"),
            color=alt.Color("Score:Q", scale=alt.Scale(scheme="tealblues")),
            tooltip=["Parameter", "Score"]
        ).properties(height=240)
        
        st.altair_chart(chart_rubric, width="stretch")

    # Right Column: Winning Recommendation & Pareto Frontier
    with c2:
        st.markdown("### 🏆 Pareto-Optimal Recommendation")
        
        nv_tag = '<span class="nvidia-tag">◆ NVIDIA Model</span>' if rec.is_nvidia else ''
        tier_class = f"tier-badge-{rec.tier.lower()}"
        
        st.markdown(f"""
        <div class="winner-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="color:#10B981; font-weight:700; font-size:13px;">★ WINNING MODEL</span>
                {nv_tag}
            </div>
            <div style="font-size:22px; font-weight:800; color:#FFFFFF;">{rec.model_name}</div>
            <div style="font-family:monospace; font-size:12px; color:#94A3B8; margin-bottom:14px;">{rec.recommended_model_id}</div>
            <div style="display:flex; gap:12px; margin-bottom:14px;">
                <span class="{tier_class}">{rec.tier.upper()} TIER</span>
                <span style="color:#10B981; font-weight:600; font-size:13px;">Save {rec.cost_savings_vs_frontier_pct:.1f}% vs Frontier</span>
            </div>
            <div style="font-size:13px; color:#CBD5E1; line-height:1.5; background:rgba(0,0,0,0.3); padding:10px 14px; border-radius:8px;">
                {rec.rationale}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Est. Cost / Req", f"${rec.estimated_cost_usd:.6f}")
        k2.metric("Est. Latency", f"{rec.estimated_latency_sec:.2f}s")
        k3.metric("Quality Score", f"{rec.quality_score:.1f} / 100")
        
        # Pareto Candidates Plot
        st.markdown("**Pareto Frontier Tradeoff:**")
        candidates_data = []
        for c in rec.all_candidates:
            candidates_data.append({
                "Model": c["name"],
                "Vendor": c["vendor"],
                "Tier": c["tier"].upper(),
                "Cost ($)": c["cost_usd"],
                "Latency (s)": c["latency_sec"],
                "Quality": c["quality_match"],
                "Is_Winner": c.get("is_winner", (c["model_id"] == rec.recommended_model_id)),
                "Is_NVIDIA": c["is_nvidia"],
            })
        df_candidates = pd.DataFrame(candidates_data)
        
        scatter = alt.Chart(df_candidates).mark_circle(size=130).encode(
            x=alt.X("Latency (s):Q", title="Estimated Latency (s)"),
            y=alt.Y("Quality:Q", scale=alt.Scale(domain=[60, 100]), title="Quality Match (0-100)"),
            color=alt.Color("Tier:N", scale=alt.Scale(domain=["NANO", "SUPER", "ULTRA"], range=["#10B981", "#06B6D4", "#8B5CF6"])),
            tooltip=["Model", "Vendor", "Tier", "Cost ($)", "Latency (s)", "Quality"]
        ).properties(height=240)
        
        winner_df = df_candidates[df_candidates["Is_Winner"]]
        if not winner_df.empty:
            winner_point = alt.Chart(winner_df).mark_point(
                size=320, shape="cross", strokeWidth=3, color="#F59E0B"
            ).encode(
                x="Latency (s):Q",
                y="Quality:Q",
                tooltip=["Model", "Cost ($)", "Quality"]
            )
            combined_chart = scatter + winner_point
        else:
            combined_chart = scatter
        
        st.altair_chart(combined_chart, width="stretch")

# Live Inference Section
st.markdown("---")
st.markdown("### 🚀 Live Generation Execution")

gen_col1, gen_col2 = st.columns([1, 4])
with gen_col1:
    execute_button = st.button("⚡ Route & Generate Live", type="primary", use_container_width=True)

with gen_col2:
    override_model = st.selectbox(
        "Manual Model Override (Optional):",
        options=["Auto-Route (Pareto Optimal)"] + [m["model_id"] for m in catalog_data],
        index=0
    )

if execute_button:
    chosen_id = None if override_model == "Auto-Route (Pareto Optimal)" else override_model
    
    with st.spinner("Connecting to Nebius Token Factory & Generating..."):
        start_t = time.time()
        try:
            if chosen_id:
                res = nebius_client.generate(model_id=chosen_id, prompt=prompt_text)
                target_model_name = chosen_id
            else:
                from tools.nebius_client import route_and_generate
                res = route_and_generate(
                    prompt=prompt_text,
                    cost_sensitivity=cost_sensitivity,
                    latency_sensitivity=latency_sensitivity,
                    prefer_nvidia=prefer_nvidia,
                )
                target_model_name = getattr(res, "model_name", getattr(res, "recommended_model_id", "Selected Model"))
            
            dur = time.time() - start_t
            
            st.success(f"Generation completed in {dur:.2f}s using **{target_model_name}**")
            
            # Response metrics
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Roundtrip Latency", f"{getattr(res, 'roundtrip_latency_sec', dur):.2f}s")
            r2.metric("Input Tokens", getattr(res, "input_tokens", est_tokens))
            r3.metric("Output Tokens", getattr(res, "output_tokens", 0))
            r4.metric("Realized Cost", f"${getattr(res, 'cost_usd', 0.0):.6f}")
            
            # Text output
            st.markdown("**Generated Response:**")
            st.code(getattr(res, "generated_text", str(res)), language="markdown")
            
        except Exception as e:
            st.error(f"Inference execution failed: {str(e)}")

# Model Catalog Explorer Tab
with st.expander("📚 Explore Complete Nebius Model Catalog (24 Models)"):
    if catalog_data:
        df_cat = pd.DataFrame(catalog_data)
        display_cols = ["name", "model_id", "vendor", "tier", "context_window_tokens", "input_price_per_million", "output_price_per_million", "tokens_per_second", "is_nvidia"]
        available_cols = [c for c in display_cols if c in df_cat.columns]
        st.dataframe(
            df_cat[available_cols],
            width="stretch",
            column_config={
                "name": "Model Name",
                "model_id": "Nebius Model ID",
                "vendor": "Vendor",
                "tier": "Tier",
                "context_window_tokens": st.column_config.NumberColumn("Context Window", format="%d tok"),
                "input_price_per_million": st.column_config.NumberColumn("Input $/M", format="$%.2f"),
                "output_price_per_million": st.column_config.NumberColumn("Output $/M", format="$%.2f"),
                "tokens_per_second": st.column_config.NumberColumn("Speed", format="%.0f t/s"),
                "is_nvidia": st.column_config.CheckboxColumn("NVIDIA Model"),
            }
        )
