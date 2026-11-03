# Progress

## Stage 0 — project setup (done, 2026-11-03)
- venv (Python 3.13.5), git, `.gitignore`, `.env.example`, `requirements.txt`,
  `src/config.py` (`LLM_PROVIDER` switch), `src/check_setup.py`.
- Models: `gemini-3.5-flash-lite` (LLM), `gemini-embedding-2` (Embeddings, 3072 dimensions).
  `gemini-3.8-flash` returned 503 high demand — compare again in stage 7 (problems_log #2).
- NetFree SSL: solved with `truststore` in `src/__init__.py` (problems_log #1).
  Rule: entry points outside `src/` must start with `import src`.

## Stage 1 — message format and data sources (done, 2026-11-03)
- Direction: "לוח הודעות כיתתי עם בוט שאלות". Group = students only (no staff, no roles, no auth).
  History: IMAP blocked → static Takeout → messages through the app (problems_log #3).
- `src/messages.py`: `parse_message` (reads In-Reply-To), `save_email(record, folder)`, `email_filename`,
  `seed_samples()` (copies samples to `data/emails/` when it is empty).
- Format: message_id, date, subject, from, in_reply_to, body, body_type.
- `src/import_export.py`: optional import, recursive, `--dry-run`, `--group`.
- `fetch_emails.py` deleted (kept in commit `e34ceba`). Gmail vars removed from `.env.example`.
- Samples: 25 invented student messages (`tests/make_sample_emails.py`), in Git; see `data/sample_emails/README.md`.
  Cases: update, cancel, partial, contradiction (#21/#22), threads, HTML, quote, iPhone signature.
- Docs: CLAUDE.md, README.md (sample-data warning, IMAP history), problems_log #3.
- Planned for later stages: `index_new_messages()` (stage 3), tabs "שאלות" / "הוספת הודעה" (stage 6),
  contradiction prompt (stage 5).

## Next: Stage 2 — cleaning
