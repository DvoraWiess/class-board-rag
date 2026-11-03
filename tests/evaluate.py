# Run: python -m tests.evaluate --split tune [--top-k 10] [--cutoff 0.72] [--tag name]
import src  # noqa: F401  (Must: ראשון — מפעיל את truststore)

import argparse
import json
import re
import sys
import time
from datetime import datetime

from llama_index.core import Settings

from src.config import PROJECT_ROOT, SIMILARITY_CUTOFF, TOP_K, judge_llm, setup_models
from src.query import ask, attach_token_counter

EVAL_SET_PATH = PROJECT_ROOT / "tests" / "eval_set.json"
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"

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
    to_judge = [r for r in results if not r["answer"].startswith("ERROR")]
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
    return recall, (1 / rank if rank else 0.0)


def average(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarize(results):
    """Summary: אחוז תשובות נכונות, Recall@k, MRR וטוקנים ממוצעים. שאלות עם "?" לא נספרות בדיוק."""
    judged = [r["correct"] for r in results if r["correct"] is not None]
    return {
        "questions": len(results),
        "not_judged": len(results) - len(judged),
        "accuracy": round(sum(judged) / len(judged), 3) if judged else None,
        "recall_at_k": average(r["recall"] for r in results),
        "mrr": average(r["mrr"] for r in results),
        "llm_calls": sum(r["llm_called"] for r in results),
        # Note: טוקני LLM — נתון אמיתי מ-Gemini; טוקני Embedding — הערכה (יומן הבעיות #9)
        "avg_llm_tokens_real": average(r["tokens"]["llm_prompt"] + r["tokens"]["llm_completion"] for r in results),
        "avg_embedding_tokens_estimate": average(r["tokens"]["embedding"] for r in results),
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
                     f"{r.get('search_query') or ''} | {answer} | {r['reason']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate: הרצת סט הבדיקות וחישוב מדדים")
    parser.add_argument("--split", choices=["tune", "test", "all"], default="tune", help="Split: איזו קבוצת שאלות")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--cutoff", type=float, default=SIMILARITY_CUTOFF)
    parser.add_argument("--tag", default="", help="Tag: שם קצר לניסוי, נכנס לשם הקובץ")
    parser.add_argument("--no-rewrite", action="store_true", help="No rewrite: בלי ניסוח מחדש (להשוואה לפני/אחרי)")
    # Note: ל-lite יש 15 בקשות לדקה, וכל שאלה = 2 קריאות (ניסוח + תשובה) — לכן 9 שניות בין שאלות
    parser.add_argument("--pause", type=float, default=9.0, help="Pause: שניות בין שאלות (מגבלת בקשות)")
    args = parser.parse_args()
    # Why: תשובות יכולות לכלול תווים שאין בקידוד cp1255 (יומן הבעיות #4)
    sys.stdout.reconfigure(errors="replace")

    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    items = [q for q in eval_set["questions"] if args.split == "all" or q["split"] == args.split]
    now = datetime.fromisoformat(eval_set["asked_on"])

    setup_models()
    counter = attach_token_counter()
    judge_model = judge_llm(Settings.llm.model)
    settings = {"split": args.split, "top_k": args.top_k, "cutoff": args.cutoff, "llm": Settings.llm.model,
                "judge": judge_model.model, "rewrite": not args.no_rewrite, "asked_on": eval_set["asked_on"],
                "tag": args.tag}
    print(f"Settings: {settings}")

    results = []
    for item in items:
        # Note: שאלת המשך — השיחה הקודמת קבועה בקובץ, כדי שהבדיקה תהיה ניתנת לשחזור
        history = [(h["question"], h["answer"]) for h in item.get("history", [])]
        try:
            answer, nodes = ask(item["question"], counter, history=history, rewrite=not args.no_rewrite,
                                top_k=args.top_k, cutoff=args.cutoff, now=now)
        except Exception as error:
            print(f"Q{item['id']:<3}ERROR  {type(error).__name__}: {str(error)[:80]}")
            results.append({"id": item["id"], "type": item["type"], "question": item["question"],
                            "expected": item["expected"], "answer": f"ERROR: {type(error).__name__}",
                            "correct": None, "reason": "Answer error", "recall": None, "mrr": None,
                            "llm_called": False, "tokens": {"llm_prompt": 0, "llm_completion": 0, "embedding": 0},
                            "sent_ids": [], "cited": []})
            continue
        recall, mrr = retrieval_metrics(nodes, item["relevant_ids"])
        results.append({
            "id": item["id"], "type": item["type"], "question": item["question"], "expected": item["expected"],
            "search_query": answer.search_query, "answer": answer.text, "correct": None, "reason": "",
            "recall": None if recall is None else round(recall, 3), "mrr": None if mrr is None else round(mrr, 3),
            "llm_called": answer.llm_called, "tokens": answer.tokens,
            "sent_ids": [n.node.metadata["message_id"] for n in nodes],
            "cited": [s["message_id"] for s in answer.sources],
        })
        print(f"Q{item['id']:<3}{item['type']:<14} recall={recall} mrr={mrr} | search: {answer.search_query}")
        time.sleep(args.pause)

    verdicts = judge_all(judge_model, results)
    for r in results:
        if r["id"] in verdicts:
            r["correct"], r["reason"] = verdicts[r["id"]]
        mark = {True: "OK", False: "WRONG", None: "?"}[r["correct"]]
        print(f"Q{r['id']:<3}{mark:<6} {r['reason']}")

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
