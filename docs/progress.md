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

## Stage 1 — email ingestion (in progress, 2026-11-03)
- Decisions: stdlib `imaplib` (LlamaIndex Gmail reader needs OAuth and returns Documents, not JSON),
  extra field `body_type` (`text`/`html`) for stage 2, search `INBOX` only.
- `src/fetch_emails.py`: IMAP, `--days` (default 90), readonly + BODY.PEEK (no "mark as read"),
  checks Message-ID header before downloading the full email. File name = sha1(message_id)[:16].json.
- `src/import_export.py`: `.mbox` / `.eml` folder → same JSON, filtered by `SENDER_ADDRESS`.
- `tzdata` added (Windows has no time zone database for `zoneinfo`).
- IMAP port 993 reachable through NetFree (greeting OK). Synthetic parse tests pass.
- IMAP blocked (no 2FA → no app password), problems_log #3. Main path is now Takeout `.mbox` import.
  Emails come from a group (many senders → one address); `from` matters; informal messages exist.
  UI (stage 6): "ייבוא קובץ חדש" button instead of "סנכרון מיילים חדשים".
- Sample data: 21 invented emails in `data/sample_emails/` (see its README.md) — for development only.
  Submission results must use the real Takeout emails.
- Decisions: no sender filter by default; optional `--group` (To/Cc); `--dry-run` prints stats only.
  `import_export.py` searches folders recursively for `.mbox` / `.eml`.
- Samples committed to Git (`.gitignore`: `data/*` + `!data/sample_emails/`); generator in `tests/make_sample_emails.py`.
- CLAUDE.md updated: Takeout is the main path, emails come from a group, "ייבוא קובץ חדש" button.
- Next: dry-run on the real export in `data/raw/` (no saving), review results with the user.
