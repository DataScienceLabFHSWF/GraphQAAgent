#!/bin/bash
# Start both the API server and the Streamlit frontend.
#
# Usage:  ./scripts/run_demo.sh
# Stop:   Ctrl+C (kills both processes)

set -e

echo "=== GraphQA Agent Demo ==="
echo ""

echo "Starting GraphQA API on :8080..."
uvicorn kgrag.api.server:app --host 0.0.0.0 --port 8080 &
API_PID=$!

echo "Starting Streamlit UI on :8501..."
streamlit run src/kgrag/frontend/app.py --server.port 8501 --server.headless true &
UI_PID=$!

trap "kill $API_PID $UI_PID 2>/dev/null; echo 'Demo stopped.'" EXIT

echo ""
echo "Demo running:"
echo "  API:  http://localhost:8080/docs"
echo "  UI:   http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop."
wait
