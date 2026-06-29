# Server Benchmark Quickstart

## Prereqs (one-time)

```bash
# 1. Clone main branch
git clone --branch main --single-branch \
  https://github.com/minhquoctran2604/TLU-Chatbot.git ~/LightRAG
cd ~/LightRAG

# 2. Sync graph data from local Windows
#    Run on Windows:
#      scp -r D:\Document\RAG_Learning\tlu_workspace tts@192.168.3.252:~/

# 3. Install deps
bash eval/setup_server.sh

# 4. Start 9router (config dir ~/.9router, sync via web dashboard or scp DB)
npx 9router &

# 5. Configure .env
cp eval/.env.template .env
vim .env
# Fill in:
#   LLM_BINDING_API_KEY=<9router key on this machine>
#   POSTGRES_PASSWORD=<supabase password>
#   RERANK_BINDING_API_KEY=<cohere key>
#   WORKING_DIR=/home/tts/tlu_workspace   (already in template)

# 6. Start LightRAG server
source venv/bin/activate
nohup python -u -m lightrag.api.lightrag_server > server.log 2>&1 &

# 7. Wait warmup (~30s), verify
curl http://localhost:9621/health
```

## Run benchmark

```bash
cd ~/LightRAG
source venv/bin/activate
bash eval/run_all.sh
```

Output:
- `eval/results_raw.json` — raw responses
- `eval/results_eval.json` — BERTScore P/R/F1
- `eval/results_chunks.json` — chunks per (query, mode)
- `eval/results_ragas.json` — RAGAS 4 metrics
- `eval/report.md` — BERTScore aggregate report

## Retrieve results

```bash
# From Windows
scp tts@192.168.3.252:~/LightRAG/eval/results_*.json D:/Document/RAG_Learning/LightRAG/eval/
scp tts@192.168.3.252:~/LightRAG/eval/report.md D:/Document/RAG_Learning/LightRAG/eval/
```

## Options

```bash
# Skip RAGAS (faster, only BERTScore)
bash eval/run_all.sh --skip-ragas

# Subset of modes
bash eval/run_all.sh --modes naive,hybrid

# Re-eval without re-running bench
bash eval/run_all.sh --skip-bench

# Custom queries
bash eval/run_all.sh --queries eval/my_queries.json
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Server not responding at :9621` | Check `tail server.log`. Most common: missing tokenizer download, PG connection refused |
| `9router 401 Unauthorized` | Wrong `LLM_BINDING_API_KEY` — must match key on **this server's** 9router |
| `No module named 'rank_bm25'` | `pip install rank-bm25` (or re-run `setup_server.sh`) |
| `Graph file not found` | Sync `tlu_workspace/` from local (see step 2) |
| RAGAS timeout/quota | Re-run only failed: `python eval/retry_failed_ragas.py --metric faithfulness` |

## Architecture recap

```
Linux server (192.168.3.252)
├── ~/LightRAG/             ← main branch (code + eval scripts)
├── ~/source_docs/          ← master branch (PDF/MD source)
├── ~/tlu_workspace/        ← graph + image cache (scp'd from local)
├── ~/.9router/             ← 9router config (sqlite db)
└── ~/hf_cache/             ← HF model cache
```

Workflow:
- LightRAG → calls 9router (`localhost:20128`) → forwards to LLM providers
- LightRAG → reads chunks/entities from Supabase Postgres (`aws-1-ap-south-1.pooler.supabase.com`)
- LightRAG → reads graph from local file `tlu_workspace/graph_chunk_entity_relation.graphml`
- Embed model `microsoft/harrier-oss-v1-270m` cached at `hf_cache/`
