# rag-requirements-demo

## Purpose & links

- Reference project: hybrid RAG, requirement extraction and a small LangGraph agent over the EU Medical Device Regulation (MDR 2017/745, DE and EN). A demo, not a product. Public: `robinchoice/rag-requirements-demo`.
- Corpus comes from the EU Publications Office (Cellar API behind EUR-Lex). No copyrighted standards (IEC, ISO) in the repo.

## Checks

`uv sync --all-extras && uv run ruff check . && uv run ruff format --check . && uv run pytest -q`. Tests start Postgres/pgvector via Testcontainers (Docker needed) and use fake embeddings and a scripted fake LLM, never a real model.

## Deploy

None. No hosting, no domain. CI (`.github/workflows/ci.yml`) is the only pipeline: job `test` (ruff, pytest), then job `kind` builds the image, applies `deploy/k8s` to a kind cluster and smoke-tests `/readyz`, full-text `/search`, `/openapi.json` and `/metrics`. The kind cluster has no model server, so endpoints that need embeddings or the LLM are not part of the smoke test.

Local stack: `docker compose up -d`, then `uv run --env-file .env --extra ingest python -m mdr_rag.ingest`, then `scripts/demo`. On Robin's laptop `.env` moves the published ports (Postgres 5460, Ollama 11435, API 8010) because other projects use the defaults.

## Pitfalls

- `docling` pulls torch. `pyproject.toml` pins torch/torchvision to the CPU wheel index; without that, uv installs ~3 GB of CUDA libraries. Docling is an optional extra (`ingest`), the API does not need it.
- EUR-Lex answers scripted requests with a bot challenge (HTTP 202, empty body). `ingest.py` fetches from the Cellar REST API instead; do not work around the challenge.
- Ingest embeds ~1500 chunks with bge-m3 on the CPU: about an hour on a loaded laptop. It commits per batch and resumes, so interrupting is safe.
- `/agent` and `/extract-requirements` call the 7B model several times; on a laptop CPU that takes minutes. Ollama's context is set to 8192 tokens in `compose.yaml`, which is why `tree.READ_LIMIT` and `agent.MAX_STEPS` are small.
- Raw HTML, model weights and database volumes stay out of git (`data/`, Docker volumes).

## Deviations from the standard

- Python (uv, FastAPI, psycopg 3, pgvector, LangGraph) instead of the starter stack (Bun, Hono, SvelteKit, Drizzle): Robin's decision, the project exists to show Python, RAG and Kubernetes. Not generated from `robinchoice/starter`.
- No UI, no auth, no signup: the FastAPI docs and the terminal demo are the interface.
