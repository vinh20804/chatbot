#!/bin/bash
# =============================================================================
# LightRAG Server Setup — first-time install on Linux
#
# Run once when setting up a new SSH server.
#
# Usage:
#   cd ~/LightRAG
#   bash eval/setup_server.sh
# =============================================================================

set -e
set -u

# Auto-detect repo root (script may live in eval/ subfolder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_DIR/venv"
HF_CACHE="$HOME/hf_cache"
WORKSPACE_DIR="$HOME/tlu_workspace"

echo "[setup] Detected REPO_DIR=$REPO_DIR"

echo "============================================================"
echo "  LightRAG Server Setup"
echo "============================================================"

# ---- 1. Python venv ----
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/5] Creating Python venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
echo "[1/5] venv: $(python --version) at $VIRTUAL_ENV"

# ---- 2. Install LightRAG deps ----
echo "[2/5] Installing LightRAG deps..."
cd "$REPO_DIR"
pip install -e ".[api]" --quiet
echo "[2/5] LightRAG installed."

# ---- 3. Install eval deps ----
echo "[3/5] Installing eval deps..."
pip install --quiet \
    rank-bm25 \
    bert-score \
    ragas \
    langchain-openai \
    cohere \
    asyncpg \
    python-dotenv \
    sentence-transformers \
    transformers \
    torch
echo "[3/5] Eval deps installed."

# ---- 4. Download HF embed model ----
mkdir -p "$HF_CACHE"
echo "[4/5] Downloading HF embed model microsoft/harrier-oss-v1-270m..."
HF_HOME="$HF_CACHE" python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('microsoft/harrier-oss-v1-270m')
print('Embed model loaded. Dim:', m.get_sentence_embedding_dimension())
" || echo "[WARN] Embed model download failed — will retry on first use"

# ---- 5. Verify workspace ----
echo "[5/5] Verifying workspace..."
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "[WARN] Workspace not found: $WORKSPACE_DIR"
    echo "       Sync from local:"
    echo "         scp -r D:/Document/RAG_Learning/tlu_workspace tts@<server>:~/"
else
    echo "      Workspace: $WORKSPACE_DIR ($(du -sh $WORKSPACE_DIR | cut -f1))"
    GRAPHML="$WORKSPACE_DIR/graph_chunk_entity_relation.graphml"
    if [ -f "$GRAPHML" ]; then
        echo "      Graph file: $GRAPHML ($(du -h $GRAPHML | cut -f1))"
    else
        echo "[WARN] Graph file missing: $GRAPHML"
    fi
fi

# ---- Final ----
echo ""
echo "============================================================"
echo "  Setup complete."
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Configure .env:"
echo "     cp eval/.env.template .env"
echo "     vim .env  # fill in API keys"
echo ""
echo "  2. Start LightRAG server:"
echo "     source venv/bin/activate"
echo "     nohup python -u -m lightrag.api.lightrag_server > server.log 2>&1 &"
echo ""
echo "  3. Wait ~30s for server warmup, then test:"
echo "     curl http://localhost:9621/health"
echo ""
echo "  4. Run benchmark pipeline:"
echo "     bash eval/run_all.sh"
