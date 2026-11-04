# Run: python -m tests.evaluate --split tune [--top-k 10] [--cutoff 0.72] [--llm MODEL] [--no-judge] [--tag name]
import src  # noqa: F401  (Must: ראשון — מפעיל את truststore)

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

from llama_index.core import Settings

from src.config import NO_INFO_PENALTY, PROJECT_ROOT, SIMILARITY_CUTOFF, TOP_K, judge_llm, setup_models
from src.query import ask, attach_token_counter

EVAL_SET_PATH = PROJECT_ROOT / "tests" / "eval_set.json"
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
STAGES = ["rewrite", "embedding", "search", "llm"]

# Why: שיפוט מרוכז — קריאה אחת לשופט בכל ריצה. במסלול החינמי יש לשופט 20 בקשות ליום (יומן הבעיות #11)
JUDGE_PROMPT = """אתה בודק תשובות של בוט שעונה על שאלות לפי הודעות של קבוצת תלמידות.
לפניך {count} פריטים. בכל פריט: שאלה, התשובה הנכונה, והתשובה של הבוט.

כללים — מתי תשובת הבוט נכונה:
- המידע העיקרי מהתשובה הנכונה מופיע, ואין מידע שגוי (תאריך, שעה, מקום) שסותר אותה. פרטים נוספים נכונים — מותר.
- אם התשובה הנכונה היא "לא מצאתי מידע" — הבוט חייב לומר שאין לו מידע, ולא לענות מהידע הכללי שלו.
- אם התשובה הנכונה מציגה סתירה — הבוט חייב להציג את כל הגרסאות.
- אם התשובה הנכונה אומרת שמידע חסר — הבוט חייב לציין שהוא חסר, ולא להמציא.
- ניסוח שונה, סדר אחר, או תאריך עם שנה — לא משנים.
- שפוט כל פריט בנפרד, רק לפי הפריט עצמו.

{items}

ענה ב-JSON בלבד, בלי טקסט נוסף: מערך עם {count} איברים, אחד לכל פריט, עם מספר הפריט (id):
[{{"id": 1, "correct": true, "reason": "משפט אחד בעברית"}}, ...]"""

ITEM_TEMPLATE = """<item id="{id}">
שאלה: {question}
התשובה הנכונה: {expected}
התשובה של הבוט: {actual}
</item>"""


def parse_verdicts(raw, ids):
    """Parse: מחזיר {id: (נכון?, נימוק)}. פריט חסר או לא תקין — (None, נימוק), כלומר "?"."""
    verdicts = {}
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    try:
        for entry in json.loads(match.group(0)):
            if isinstance(entry, dict) and entry.get("id") in ids and entry["id"] not in verdicts:
                verdicts[entry["id"]] = (bool(entry["correct"]), entry.get("reason", ""))
    except (AttributeError, ValueError, KeyError, TypeError):
        pass
    missing = [i for i in ids if i not in verdicts]
    if missing:
        print(f"Warning: השופט החזיר {len(verdicts)} שיפוטים ל-{len(ids)} שאלות. חסרות: {missing} — מסומנות '?'")
    return {i: verdicts.get(i, (None, "Missing in judge output")) for i in ids}


def judge_all(judge_model, results):
    """Judge: מודל אחר מזה שכתב את התשובות שופט את כל התשובות בקריאה אחת."""
    to_judge = [r for r in results if r["error"] is None]
    ids = [r["id"] for r in to_judge]
    items = "\n\n".join(ITEM_TEMPLATE.format(id=r["id"], question=r["question"], expected=r["expected"],
                                             actual=r["answer"]) for r in to_judge)
    try:
        raw = judge_model.complete(JUDGE_PROMPT.format(count=len(ids), items=items)).text
    except Exception as error:
        # Why: שגיאה (למשל 429 / 503) לא מפילה את הריצה — כל השאלות מסומנות "?"
        print(f"Judge error: {type(error).__name__} {str(error)[:80]}")
        return {i: (None, f"Judge error: {type(error).__name__}") for i in ids}
    return parse_verdicts(raw, ids)


def retrieval_metrics(nodes, relevant_ids):
    """Retrieval: Recall@k = כמה מההודעות הנכונות הגיעו ל-LLM; MRR = 1/מקום (לפי ציון) של הראשונה מהן."""
    if not relevant_ids:
        return None, None
    ranked = [n.node.metadata["message_id"] for n in sorted(nodes, key=lambda n: n.score, reverse=True)]
    recall = sum(mid in ranked for mid in relevant_ids) / len(relevant_ids)
    rank = next((i for i, mid in enumerate(ranked, 1) if mid in relevant_ids), None)
    return round(recall, 3), round(1 / rank if rank else 0.0, 3)


def result_row(item, answer=None, nodes=(), error=None):
    """Row: שורת תוצאה אחת. בשגיאה — answer=None והשאלה לא נשפטת."""
    recall, mrr = retrieval_metrics(nodes, item["relevant_ids"]) if answer else (None, None)
    return {
        "id": item["id"], "type": item["type"], "question": item["question"], "expected": item["expected"],
        "search_query": answer.search_query if answer else None,
        "answer": answer.text if answer else f"ERROR: {error}", "error": error,
        "correct": None, "reason": "", "recall": recall, "mrr": mrr,
        "llm_called": answer.llm_called if answer else False,
        "tokens": answer.tokens if answer else {"llm_prompt": 0, "llm_completion": 0, "embedding": 0},
        "seconds": {k: round(v, 2) for k, v in answer.timings.items()} if answer else {},
        "sent_ids": [n.node.metadata["message_id"] for n in nodes],
        "cited": [s["message_id"] for s in answer.sources] if answer else [],
    }


def average(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarize(results):
    """Summary: נכונות, Recall@k, MRR, טוקנים וזמן ממוצע לכל שלב. שאלות עם "?" לא נספרות בנכונות."""
    judged = [r["correct"] for r in results if r["correct"] is not None]
    ok = [r for r in results if r["error"] is None]
    return {
        "questions": len(results),
        "not_judged": len(results) - len(judged),
        "accuracy": round(sum(judged) / len(judged), 3) if judged else None,
        "recall_at_k": average(r["recall"] for r in results),
        "mrr": average(r["mrr"] for r in results),
        "llm_calls": sum(r["llm_called"] for r in results),
        # Note: טוקני LLM — נתון אמיתי מ-Gemini; טוקני Embedding — הערכה (יומן הבעיות #9)
        "avg_llm_tokens_real": average(r["tokens"]["llm_prompt"] + r["tokens"]["llm_completion"] for r in ok),
        "avg_embedding_tokens_estimate": average(r["tokens"]["embedding"] for r in ok),
        "avg_seconds": {stage: average(r["seconds"].get(stage) for r in ok) for stage in STAGES}
                       | {"total": average(sum(r["seconds"].values()) for r in ok)},
    }


def write_markdown(path, settings, summary, results):
    """Report: טבלה לקריאה — כדי לעבור על הנימוקים של השופט."""
    lines = [f"# Evaluation — {settings['split']}", "", "```", json.dumps(settings, ensure_ascii=False, indent=2),
             json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
             "| # | type | correct | recall | MRR | question | search query | answer | judge reason |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        mark = {True: "✓", False: "✗", None: "?"}[r["correct"]]
        answer = r["answer"].replace("\n", " ").replace("|", "/")
        lines.append(f"| {r['id']} | {r['type']} | {mark} | {r['recall']} | {r['mrr']} | {r['question']} | "
                     f"{r['search_query'] or ''} | {answer} | {r['reason']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate: הרצת סט הבדיקות וחישוב מדדים")
    parser.add_argument("--split", choices=["tune", "test", "all"], default="tune", help="Split: איזו קבוצת שאלות")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--cutoff", type=float, default=SIMILARITY_CUTOFF)
    parser.add_argument("--penalty", type=float, default=NO_INFO_PENALTY, help="Penalty: הורדת ציון להודעות בלי מידע")
    parser.add_argument("--llm", help="LLM: מודל אחר לכתיבת התשובות (ניסוי lite מול flash)")
    parser.add_argument("--no-rewrite", action="store_true", help="No rewrite: בלי ניסוח מחדש (להשוואה)")
    parser.add_argument("--no-judge", action="store_true", help="No judge: רק מדדי אחזור וזמנים, בלי שופט")
    parser.add_argument("--tag", default="", help="Tag: שם קצר לניסוי, נכנס לשם הקובץ")
    # Note: ל-lite יש 15 בקשות לדקה, וכל שאלה = 2 קריאות (ניסוח + תשובה) — לכן 9 שניות בין שאלות
    parser.add_argument("--pause", type=float, default=9.0, help="Pause: שניות בין שאלות (מגבלת בקשות)")
    args = parser.parse_args()
    # Why: תשובות יכולות לכלול תווים שאין בקידוד cp1255 (יומן הבעיות #4)
    sys.stdout.reconfigure(errors="replace")

    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    items = [q for q in eval_set["questions"] if args.split == "all" or q["split"] == args.split]
    now = datetime.fromisoformat(eval_set["asked_on"])

    if args.llm:
        os.environ["GEMINI_LLM_MODEL"] = args.llm
    setup_models()
    counter = attach_token_counter()
    judge_model = None if args.no_judge else judge_llm(Settings.llm.model)
    settings = {"split": args.split, "top_k": args.top_k, "cutoff": args.cutoff, "penalty": args.penalty,
                "llm": Settings.llm.model,
                "judge": judge_model.model if judge_model else None, "rewrite": not args.no_rewrite,
                "asked_on": eval_set["asked_on"], "tag": args.tag}
    print(f"Settings: {settings}")

    results = []
    for item in items:
        # Note: שאלת המשך — השיחה הקודמת קבועה בקובץ, כדי שהבדיקה תהיה ניתנת לשחזור
        history = [(h["question"], h["answer"]) for h in item.get("history", [])]
        for attempt in (1, 2):
            try:
                answer, nodes = ask(item["question"], counter, history=history, rewrite=not args.no_rewrite,
                                    top_k=args.top_k, cutoff=args.cutoff, penalty=args.penalty, now=now)
                row = result_row(item, answer, nodes)
                break
            except Exception as error:
                row = result_row(item, error=f"{type(error).__name__}: {str(error)[:80]}")
                # Why: 429 של מגבלה לדקה (15 ל-lite) — מחכים חצי דקה ומנסים שוב פעם אחת
                if "429" not in str(error) or attempt == 2:
                    break
                print(f"Q{item['id']}: מגבלת בקשות לדקה — מחכים 30 שניות")
                time.sleep(30)
        results.append(row)
        print(f"Q{row['id']:<3}{row['type']:<14} recall={row['recall']} mrr={row['mrr']} "
              f"sec={row['seconds']} | {row['search_query'] or row['answer']}")
        time.sleep(args.pause)

    if judge_model:
        verdicts = judge_all(judge_model, results)
        for r in results:
            r["correct"], r["reason"] = verdicts.get(r["id"], (None, "Answer error"))
            print(f"Q{r['id']:<3}{ {True: 'OK', False: 'WRONG', None: '?'}[r['correct']]:<6} {r['reason']}")

    summary = summarize(results)
    print(f"Summary: {summary}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{args.split}_k{args.top_k}_c{args.cutoff}_{Settings.llm.model}" + (f"_{args.tag}" if args.tag else "")
    output = {"settings": settings, "summary": summary, "results": results}
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(RESULTS_DIR / f"{name}.md", settings, summary, results)
    print(f"Saved: tests/results/{name}.json + .md")


if __name__ == "__main__":
    main()
