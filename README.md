# rag-requirements-demo

Hybrid RAG, requirement extraction and a small agent over the EU Medical Device Regulation
(MDR 2017/745, German and English). Work in progress.

```sh
uv sync --all-extras
uv run --extra ingest python -m mdr_rag.ingest   # needs Postgres/pgvector and an OpenAI-compatible endpoint
uv run pytest
```
