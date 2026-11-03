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

## Stage 2 — cleaning (done, waiting for approval/commit, 2026-11-03)
- Decisions: BeautifulSoup for HTML; own regex for quotes (email_reply_parser is English-only);
  clean at index time, raw body stays in JSON; informal messages are NOT filtered — measure in stage 4.
- `src/clean.py`: `clean_body(body, body_type)` = HTML → remove bidi marks → quotes → signature → whitespace.
  Quote header must contain a digit (date/time), so "המורה כתבה:" is kept.
  CLI: `python -m src.clean --summary` (lengths only), `python -m src.clean "<id>"` (before/after).
- Result on samples: exactly the 5 designed messages changed (#02, #10, #11, #13, #23). Edge cases pass.

## Stage 3 — indexing (done, waiting for approval/commit, 2026-11-03)
- `src/indexer.py`: Document per message (id = message_id), text = subject + clean body
  (+ reply: `בתגובה ל<name>: "<150 chars of parent>"` — decision B).
  Metadata: message_id, subject, from (name only), date (readable, with weekday), date_ts, in_reply_to.
  Embed sees text only; LLM sees text + from + date.
- Chroma 1.5.9 persistent in `chroma_db/`, collection "messages", cosine (`configuration`), embedding_function=None.
- `index_new_messages()`: incremental insert in batches of 10 messages; `index_info.json`
  (provider, model, dimension 3072, chunk settings) — mismatch stops with a Hebrew message and `--rebuild`.
- Result: 25 messages → 25 Nodes; second run adds 0; model change detected.

## Stage 4 — retrieval inspection (done, waiting for approval/commit, 2026-11-03)
- `src/inspect_retrieval.py`: top_k with LlamaIndex score, Chroma distance, cosine; no LLM. `TOP_K = 5` in config.
- Score: LlamaIndex Chroma returns `exp(-distance)`; Chroma cosine distance = 1 - cosine. Range here ~0.65-0.85.
  Embeddings use task types RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT (built into the library).
- Findings → problems_log #5-#8: no threshold separates in-domain "no info" (0.790) from answers (0.772);
  unrelated question 0.673; update ranked below original; relative "מחר"; informal messages rank high.
- Suggested cutoff for stage 5: 0.72 (score units).

## Stage 5 — answer generation (done, waiting for approval/commit, 2026-11-03)
- `src/query.py`: `as_query_engine` with TOP_K=10, `SimilarityPostprocessor(0.72)`, custom `ChronologicalNumbering`
  (FixedRecencyPostprocessor is newest-first, keeps 1, needs pandas), Hebrew `QA_PROMPT` with today's date,
  `<messages>`/`<message number>` tags, 9 rules (update, contradiction, relative dates, student questions,
  partial, no-info, injection, sources line).
- `ask()` = `engine.retrieve()` → empty → NO_INFO without LLM; else `engine.synthesize()`.
- Sources: LLM ends with "מקורות: N, M"; `split_sources` strips it; fallback = all sent nodes (no line / bad number).
  `Answer.latest_date`, `Answer.has_earlier`.
- Tokens: `TokenCountingHandler` — LLM counts are real (Gemini usage_metadata), embedding counts are estimates.
- `--show-prompt` prints the real prompt. Results: update, cancel, contradiction, partial, "מחר", no-info,
  off-topic (no LLM call), prompt injection — all correct. Problems #5-#7, #9 updated.
- For stage 6: hide "מקורות:" line while streaming (user request), show formatted sources instead.

## Next: Stage 6 — Streamlit UI
