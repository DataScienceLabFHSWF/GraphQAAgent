#!/bin/bash
# KG-RAG Ollama Setup Script
# This script sets up the dedicated Ollama instance for KG-RAG

set -e

echo "🚀 Setting up KG-RAG Ollama instance..."

# Check if docker-compose.kgrag.yml exists
if [ ! -f "docker-compose.kgrag.yml" ]; then
    echo "❌ docker-compose.kgrag.yml not found!"
    exit 1
fi

# Start the Ollama service
echo "📦 Starting KG-RAG Ollama service on port 18136..."
docker compose -f docker-compose.kgrag.yml up -d kgrag-ollama

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to be ready..."
sleep 10

# Check if Ollama is running
if ! docker compose -f docker-compose.kgrag.yml exec -T kgrag-ollama ollama list > /dev/null 2>&1; then
    echo "❌ Ollama service failed to start properly"
    exit 1
fi

echo "✅ Ollama service is running!"

# Pull required models
echo "📥 Pulling required models..."

echo "Pulling qwen3:8b (generation model)..."
docker compose -f docker-compose.kgrag.yml exec -T kgrag-ollama ollama pull qwen3:8b

echo "Pulling qwen3-embedding:latest (embedding model)..."
docker compose -f docker-compose.kgrag.yml exec -T kgrag-ollama ollama pull qwen3-embedding:latest

echo "📋 Available models:"
docker compose -f docker-compose.kgrag.yml exec -T kgrag-ollama ollama list

echo ""
echo "🎉 KG-RAG Ollama setup complete!"
echo ""
echo "To start the full KG-RAG stack:"
echo "  docker compose -f docker-compose.kgrag.yml up -d"
echo ""
echo "To start only the QA agent (assuming external services are running):"
echo "  docker compose -f docker-compose.kgrag.yml up -d kgrag-agent"
echo ""
echo "To view logs:"
echo "  docker compose -f docker-compose.kgrag.yml logs -f kgrag-ollama"
echo "  docker compose -f docker-compose.kgrag.yml logs -f kgrag-agent"