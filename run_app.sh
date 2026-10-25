#!/usr/bin/env bash
# ==============================================================================
# LLMOptiFit — Application Launcher
# ==============================================================================
# Usage:
#   ./run_app.sh             # Launches Streamlit Interactive Dashboard (Default)
#   ./run_app.sh streamlit   # Launches Streamlit Interactive Dashboard
#   ./run_app.sh api         # Launches FastAPI Production Backend & Swagger UI
#   ./run_app.sh web         # Launches Classic Web Server (app.py)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect Python venv
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    STREAMLIT=".venv/bin/streamlit"
    UVICORN=".venv/bin/uvicorn"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
    STREAMLIT="streamlit"
    UVICORN="uvicorn"
else
    echo "❌ Error: Python 3 not found."
    exit 1
fi

MODE="${1:-streamlit}"

echo "========================================================================"
echo "  🚀 LLMOptiFit — Pareto-Optimal LLM Router"
echo "  Connected to Nebius Token Factory & NVIDIA Nemotron Models"
echo "========================================================================"

case "$MODE" in
    streamlit|dashboard|ui)
        echo "Starting Streamlit Interactive Dashboard on http://localhost:8501 ..."
        "$STREAMLIT" run streamlit_app.py --server.port 8501 --server.headless false
        ;;
    api|fastapi|server)
        echo "Starting FastAPI Application Server on http://localhost:8000 ..."
        echo "📖 Interactive OpenAPI Swagger documentation: http://localhost:8000/docs"
        "$UVICORN" api.server:app --host 127.0.0.1 --port 8000 --reload
        ;;
    web|classic)
        echo "Starting Classic Web Server on http://localhost:8000 ..."
        "$PYTHON" app.py 8000
        ;;
    *)
        echo "Usage: ./run_app.sh [streamlit|api|web]"
        exit 1
        ;;
esac
