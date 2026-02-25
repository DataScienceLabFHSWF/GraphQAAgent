#!/usr/bin/env bash
# helper for bringing up the full GraphQAAgent demo stack
# usage: ./scripts/start_demo.sh [--no-frontend] [--no-docker]

set -euo pipefail

# ports we care about (will be adjusted if occupied)
API_PORT=8080
STREAMLIT_PORT=8501

# find a free TCP port on the host (not bound by local process or Docker container)
function port_in_use() {
    local p=$1
    # check local listeners
    if lsof -iTCP:$p -sTCP:LISTEN -t >/dev/null; then
        return 0
    fi
    # check docker port bindings
    if docker ps --format '{{.Ports}}' | grep -q "0.0.0.0:$p->"; then
        return 0
    fi
    return 1
}

function find_free_port() {
    local start=$1
    local p=$start
    while port_in_use $p; do
        p=$((p+1))
    done
    echo $p
}

function kill_on_port() {
    local port=$1
    if lsof -iTCP:$port -sTCP:LISTEN -t >/dev/null ; then
        echo "killing process listening on port $port"
        kill -9 $(lsof -iTCP:$port -sTCP:LISTEN -t) || true
    fi
}

# parse options
NO_FRONTEND=false
NO_DOCKER=false
EXTERNAL_NEO4J=false
EXTERNAL_FUSEKI=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-frontend) NO_FRONTEND=true; shift ;; 
        --no-docker) NO_DOCKER=true; shift ;; 
        --external-neo4j) EXTERNAL_NEO4J=true; shift ;;
        --external-fuseki) EXTERNAL_FUSEKI=true; shift ;;
        *) shift ;;
    esac
done

# ensure .env is loaded for cli tools too
if [[ -f ".env" ]]; then
    export $(grep -v '^#' .env | xargs)
fi

if [[ "$NO_DOCKER" == false ]]; then
    # decide which docker command to use
    if command -v docker-compose >/dev/null 2>&1; then
        DC="docker-compose"
    else
        DC="docker compose"
    fi

    # dynamically build service list, skipping ones with existing listeners
    SERVICES=()
    if [[ "$EXTERNAL_NEO4J" == false ]]; then
        if ! port_in_use 7474 && ! port_in_use 7687; then
            SERVICES+=(neo4j)
        else
            echo "Detected existing Neo4j; not launching new container"
        fi
    else
        echo "--external-neo4j specified; skipping Neo4j container"
    fi

    if ! port_in_use 6333; then
        SERVICES+=(qdrant)
    else
        echo "Detected existing Qdrant; not launching new container"
    fi

    # fuseki occupies 3030
    if [[ "$EXTERNAL_FUSEKI" == false ]]; then
        if ! port_in_use 3030; then
            SERVICES+=(fuseki)
        else
            echo "Detected existing Fuseki; not launching new container"
        fi
    else
        echo "--external-fuseki specified; skipping Fuseki container"
    fi

    # always bring up ollama, qa-agent, frontend (qa-agent may reuse a free port later)
    SERVICES+=(ollama qa-agent frontend)

    echo "Starting required containers (${SERVICES[*]}) using $DC"
    $DC up -d ${SERVICES[*]}
    echo "Waiting 5s for services to settle..."
    sleep 5
fi

# check connectivity quickly
echo "Checking Neo4j & Fuseki endpoints"
if curl -sSf "http://localhost:7474" >/dev/null; then
    echo "Neo4j reachable at http://localhost:7474"
else
    echo "Warning: Neo4j not responding"
fi
if curl -sSf "http://localhost:3030/" >/dev/null; then
    echo "Fuseki reachable at http://localhost:3030"
else
    echo "Warning: Fuseki not responding"
fi

# ensure ports are free and adjust if necessary
API_PORT=$(find_free_port $API_PORT)
STREAMLIT_PORT=$(find_free_port $STREAMLIT_PORT)

# start API server
echo "Starting GraphQAAgent API on port $API_PORT"
kill_on_port $API_PORT
source .venv/bin/activate
nohup uvicorn src.kgrag.api.server:app --host 0.0.0.0 --port $API_PORT &> /tmp/qa-agent.log &
API_PID=$!
echo "API PID is $API_PID"

echo "Configured frontend port $STREAMLIT_PORT"

if [[ "$NO_FRONTEND" == false ]]; then
    if [[ "$NO_DOCKER" == true ]]; then
        # run local streamlit if not using docker
        echo "Starting Streamlit frontend on port $STREAMLIT_PORT"
        kill_on_port $STREAMLIT_PORT
        nohup streamlit run src/kgrag/frontend/app.py --server.port $STREAMLIT_PORT &> /tmp/qa-frontend.log &
        FRONT_PID=$!
        echo "Frontend PID is $FRONT_PID"
        echo "You can now visit http://localhost:$STREAMLIT_PORT"
        echo "To expose with ngrok run: ngrok http $STREAMLIT_PORT"
    else
        echo "Frontend container started; visit http://localhost:$STREAMLIT_PORT"
        echo "To expose with ngrok run: ngrok http $STREAMLIT_PORT"
    fi
fi

echo "Demo stack launched."  
# prints logs tails
function tail_logs() {
    echo "--- API log ---"
    tail -20 /tmp/qa-agent.log || true
    echo "--- Frontend log ---"
    tail -20 /tmp/qa-frontend.log || true
}

tail_logs

# end of file
