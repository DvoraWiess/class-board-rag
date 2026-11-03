# Progress

## Stage 0 — project setup (done, 2026-11-03)
- venv (Python 3.13.5), git, `.gitignore`, `.env.example`, `requirements.txt`,
  `src/config.py` (`LLM_PROVIDER` switch), `src/check_setup.py`.
- Models: `gemini-3.5-flash-lite` (LLM), `gemini-embedding-2` (Embeddings, 3072 dimensions).
  `gemini-3.8-flash` returned 503 high demand — compare again in stage 7 (problems_log #2).
  Package defaults are outdated, so model names are set explicitly in `.env`.
- NetFree SSL: solved with `truststore` in `src/__init__.py` (problems_log #1).
  Rule: entry points outside `src/` must start with `import src`.
- Check: `python -m src.check_setup` passes.

## Next: Stage 1 — email ingestion (waiting for approval)
