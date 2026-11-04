import argparse
import json
from datetime import datetime
from email.utils import parseaddr
from zoneinfo import ZoneInfo

import chromadb
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.clean import clean_body
from src.config import (CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, MESSAGES_DIR, PARENT_EXCERPT_CHARS, TEXT_VERSION,
                        TIMEZONE, setup_models)
from src.messages import seed_samples

INDEX_INFO_PATH = CHROMA_DIR / "index_info.json"
BATCH_SIZE = 10
WEEKDAYS = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]  # Note: weekday() מחזיר 0 ליום שני

# Why: מזהים ומספרים — רעש. date ו-from — רק ל-LLM (מתי ומי), בחיפוש הם היו מקרבים לפי תאריך/שם.
# Why: reply_context (ציטוט מההודעה המקורית) — רק לחיפוש; reply_to (מי ומתי) — רק ל-LLM (#12)
NEVER_KEYS = ["message_id", "date_ts", "in_reply_to", "subject", "has_info"]
EXCLUDED_EMBED_KEYS = NEVER_KEYS + ["date", "from", "reply_to"]
EXCLUDED_LLM_KEYS = NEVER_KEYS + ["reply_context"]

# Why: סימון לפי תוכן, לא לפי סימן שאלה — "אז המבחן ביום שני?" היא שאלה עם מידע (#8)
INFO_PROMPT = """לפניך הודעות מקבוצה של תלמידות. לכל הודעה החלט: האם היא מוסרת מידע
(תאריך, שעה, שינוי, ביטול, משימה, חומר למבחן, פרט עובדתי) — או שהיא רק שאלה, בקשה, תודה או ברכה, בלי מידע חדש?
תשובה לשאלה של מישהי ("כן, ביום שני 9.11") — מוסרת מידע.

{items}

ענה ב-JSON בלבד: [{{"id": 1, "has_info": true}}, ...] — איבר אחד לכל הודעה."""


def readable_date(iso_date):
    """Date: "יום שלישי 3.11.2026, 20:15" — היום בשבוע עוזר בשאלות כמו "מה יש ביום שני?" (#7)."""
    date = datetime.fromisoformat(iso_date).astimezone(ZoneInfo(TIMEZONE))
    return f"יום {WEEKDAYS[date.weekday()]} {date.day}.{date.month}.{date.year}, {date:%H:%M}"


def sender_name(record):
    """Sender: רק השם. "שם <כתובת>" (הודעות הדוגמה) → השם; "נועה לוי" (מהטופס) נשאר — parseaddr היה חותך (#10)."""
    if "<" not in record["from"] and "@" not in record["from"]:
        return record["from"].strip()
    name, address = parseaddr(record["from"])
    return name or address


def classify_info(records):
    """Info: קריאת LLM אחת למנה — לכל הודעה, האם יש בה מידע. בכשל — כולן "יש מידע" (בלי הורדת ציון)."""
    items = "\n".join(f'<message id="{i}">{r["subject"]}\n{clean_body(r["body"])}</message>'
                      for i, r in enumerate(records, 1))
    try:
        raw = Settings.llm.complete(INFO_PROMPT.format(items=items)).text
        labels = {e["id"]: bool(e["has_info"]) for e in json.loads(raw[raw.find("["):raw.rfind("]") + 1])}
    except Exception as error:
        print(f"Warning: סימון has_info נכשל ({type(error).__name__}) — כל ההודעות במנה מסומנות כבעלות מידע")
        labels = {}
    return [labels.get(i, True) for i in range(1, len(records) + 1)]


def build_document(record, parent, has_info=True):
    """Document: הודעה אחת = Document אחד. הטקסט = נושא + גוף נקי."""
    metadata = {"message_id": record["message_id"], "subject": record["subject"], "from": sender_name(record),
                "date": readable_date(record["date"]),
                "date_ts": int(datetime.fromisoformat(record["date"]).timestamp()),
                # Note: Chroma לא מקבל None — לכן מחרוזת ריקה
                "in_reply_to": record.get("in_reply_to") or "", "has_info": has_info}
    if parent:
        excerpt = " ".join(clean_body(parent["body"]).split())
        excerpt = excerpt[:PARENT_EXCERPT_CHARS] + "..." if len(excerpt) > PARENT_EXCERPT_CHARS else excerpt
        metadata["reply_context"] = f'בתגובה ל{sender_name(parent)}: "{excerpt}"'
        metadata["reply_to"] = f"בתגובה להודעה של {sender_name(parent)} מ{readable_date(parent['date'])}"
    return Document(id_=record["message_id"], text=f"{record['subject']}\n{clean_body(record['body'])}",
                    metadata=metadata, excluded_embed_metadata_keys=EXCLUDED_EMBED_KEYS,
                    excluded_llm_metadata_keys=EXCLUDED_LLM_KEYS)


def load_records():
    """Load: כל ההודעות מ-data/messages, לפי message_id."""
    records = (json.loads(path.read_text(encoding="utf-8")) for path in MESSAGES_DIR.glob("*.json"))
    return {record["message_id"]: record for record in records}


def current_settings():
    """Settings: מה שחייב להיות זהה בין האינדקס השמור לקוד — אחרת צריך בנייה מחדש."""
    return {"embed_model": Settings.embed_model.model_name, "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP, "text_version": TEXT_VERSION}


def read_index_info():
    """Info: index_info.json, או None אם אין (אינדקס חדש, או בנייה מחדש באמצע)."""
    return json.loads(INDEX_INFO_PATH.read_text(encoding="utf-8")) if INDEX_INFO_PATH.exists() else None


def index_built_at():
    """Built: מתי האינדקס נבנה — משתנה רק בבנייה מחדש (#13)."""
    info = read_index_info()
    return info.get("built_at", "legacy") if info else None


def check_index_info():
    """Check: מודל Embeddings, חלוקה או בניית טקסט שונים מהאינדקס השמור — עוצרים (ווקטורים לא ניתנים להשוואה)."""
    saved = read_index_info()
    if saved is None:
        return
    changed = [f"{k}: {saved.get(k)} -> {v}" for k, v in current_settings().items() if saved.get(k) != v]
    if changed:
        raise SystemExit(f"Error: הגדרות האינדקס השתנו ({', '.join(changed)}).\n"
                         "צריך לבנות את האינדקס מחדש: python -m src.indexer --rebuild")


def save_index_info(collection):
    """Save: ההגדרות שבהן האינדקס נבנה, ממד הווקטור וזמן הבנייה (הוספה מצטברת לא משנה את built_at)."""
    built_at = index_built_at()
    if built_at in (None, "legacy"):
        built_at = datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")
    dimension = len(collection.get(limit=1, include=["embeddings"])["embeddings"][0])
    info = current_settings() | {"dimension": dimension, "built_at": built_at}
    INDEX_INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def open_collection():
    """Open: אוסף ב-Chroma. cosine — המדד המקובל לטקסט; את הווקטורים מחשב Gemini, לא Chroma."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        "messages", configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)
    return client, collection


def open_index():
    """Index: עוטף את אוסף Chroma ב-VectorStoreIndex של LlamaIndex."""
    _, collection = open_collection()
    return VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection)), collection


def index_new_messages():
    """Incremental: מוסיף לאינדקס רק הודעות שעוד לא בו (insert), במנות. מחזיר כמה נוספו."""
    check_index_info()
    index, collection = open_index()
    records = load_records()
    already = {m["message_id"] for m in collection.get(include=["metadatas"])["metadatas"]}
    new_records = sorted((r for r in records.values() if r["message_id"] not in already), key=lambda r: r["date"])
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # Why: מנות לפי הודעות — כדי שהודעה לא תיכנס רק חלקית; מנה = קריאת סימון אחת (מגבלת בקשות)
    for start in range(0, len(new_records), BATCH_SIZE):
        batch = new_records[start:start + BATCH_SIZE]
        documents = [build_document(r, records.get(r.get("in_reply_to")), info)
                     for r, info in zip(batch, classify_info(batch))]
        index.insert_nodes(splitter.get_nodes_from_documents(documents))
        print(f"Progress: אונדקסו {start + len(batch)} מתוך {len(new_records)} הודעות חדשות")
    if new_records:
        save_index_info(collection)
    print(f"Done: {len(new_records)} הודעות חדשות נוספו, סה\"כ {collection.count()} באינדקס")
    return len(new_records)


def main():
    parser = argparse.ArgumentParser(description="Index: אינדוקס מצטבר של ההודעות ל-Chroma")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild: מחיקת האינדקס ובנייה מחדש")
    args = parser.parse_args()
    setup_models()
    if args.rebuild:
        open_collection()[0].delete_collection("messages")
        INDEX_INFO_PATH.unlink(missing_ok=True)
    seed_samples()
    index_new_messages()


if __name__ == "__main__":
    main()
