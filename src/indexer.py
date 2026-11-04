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
from src.config import (CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, EMAILS_DIR, EMBED_PROVIDER, PARENT_EXCERPT_CHARS,
                        TEXT_VERSION, TIMEZONE, setup_models)
from src.messages import seed_samples

COLLECTION_NAME = "messages"
INDEX_INFO_PATH = CHROMA_DIR / "index_info.json"
BATCH_SIZE = 10

# Note: datetime.weekday() מחזיר 0 ליום שני
WEEKDAYS = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]

# Why: מזהים ומספרים — רעש לחיפוש ול-LLM; subject כבר בטקסט עצמו.
# Why: date ו-from — רק ל-LLM (מתי ומי). בחיפוש הם היו מקרבים הודעות לפי תאריך/שם ולא לפי נושא.
# Why: reply_context (ציטוט מההודעה המקורית) — רק לחיפוש; reply_to (מי ומתי) — רק ל-LLM (יומן הבעיות #12)
NEVER_KEYS = ["message_id", "date_ts", "in_reply_to", "subject", "has_info"]

# Why: סימון לפי תוכן, לא לפי סימן שאלה ("אז המבחן ביום שני?" היא שאלה עם מידע — יומן הבעיות #8)
INFO_PROMPT = """לפניך הודעות מקבוצה של תלמידות. לכל הודעה החלט: האם היא מוסרת מידע
(תאריך, שעה, שינוי, ביטול, משימה, חומר למבחן, פרט עובדתי) — או שהיא רק שאלה, בקשה, תודה או ברכה, בלי מידע חדש?
תשובה לשאלה של מישהי ("כן, ביום שני 9.11") — מוסרת מידע.

{items}

ענה ב-JSON בלבד: [{{"id": 1, "has_info": true}}, ...] — איבר אחד לכל הודעה."""
EXCLUDED_EMBED_KEYS = NEVER_KEYS + ["date", "from", "reply_to"]
EXCLUDED_LLM_KEYS = NEVER_KEYS + ["reply_context"]


def readable_date(iso_date):
    """Date: "יום שלישי 3.11.2026, 20:15" — היום בשבוע עוזר בשאלות כמו "מה יש ביום שני?"."""
    date = datetime.fromisoformat(iso_date).astimezone(ZoneInfo(TIMEZONE))
    return f"יום {WEEKDAYS[date.weekday()]} {date.day}.{date.month}.{date.year}, {date:%H:%M}"


def sender_name(record):
    """Sender: רק השם, בלי הכתובת. אם אין שם — הכתובת."""
    # Why: הודעה מהמסך היא שם בלבד ("נועה לוי"). parseaddr היה חותך אותה ברווח (יומן הבעיות #10)
    if "<" not in record["from"] and "@" not in record["from"]:
        return record["from"].strip()
    name, address = parseaddr(record["from"])
    return name or address or record["from"]


def reply_metadata(parent):
    """Reply: לחיפוש — שורה קצרה מההודעה המקורית; ל-LLM — רק מי כתבה אותה ומתי."""
    excerpt = " ".join(clean_body(parent["body"], parent["body_type"]).split())
    if len(excerpt) > PARENT_EXCERPT_CHARS:
        excerpt = excerpt[:PARENT_EXCERPT_CHARS] + "..."
    return {
        "reply_context": f'בתגובה ל{sender_name(parent)}: "{excerpt}"',
        "reply_to": f"בתגובה להודעה של {sender_name(parent)} מ{readable_date(parent['date'])}",
    }


def classify_info(records):
    """Info: קריאת LLM אחת למנה — לכל הודעה, האם יש בה מידע. בכשל — כולן "יש מידע" (בלי הורדת ציון)."""
    items = "\n".join(f'<message id="{i}">{r["subject"]}\n{clean_body(r["body"], r["body_type"])}</message>'
                      for i, r in enumerate(records, 1))
    try:
        raw = Settings.llm.complete(INFO_PROMPT.format(items=items)).text
        labels = {entry["id"]: bool(entry["has_info"]) for entry in json.loads(raw[raw.find("["):raw.rfind("]") + 1])}
    except Exception as error:
        print(f"Warning: סימון has_info נכשל ({type(error).__name__}) — כל ההודעות במנה מסומנות כבעלות מידע")
        labels = {}
    return [labels.get(i, True) for i in range(1, len(records) + 1)]


def build_document(record, parent, has_info=True):
    """Document: הודעה אחת = Document אחד. הטקסט = נושא + גוף נקי."""
    metadata = {
        "message_id": record["message_id"],
        "subject": record["subject"],
        "from": sender_name(record),
        "date": readable_date(record["date"]),
        "date_ts": int(datetime.fromisoformat(record["date"]).timestamp()),
        # Note: Chroma לא מקבל None ב-metadata — לכן מחרוזת ריקה
        "in_reply_to": record.get("in_reply_to") or "",
        "has_info": has_info,
    }
    if parent:
        metadata |= reply_metadata(parent)
    return Document(
        id_=record["message_id"],
        text=f"{record['subject']}\n{clean_body(record['body'], record['body_type'])}",
        metadata=metadata,
        excluded_embed_metadata_keys=EXCLUDED_EMBED_KEYS,
        excluded_llm_metadata_keys=EXCLUDED_LLM_KEYS,
    )


def load_records():
    """Load: כל ההודעות מ-data/emails, לפי message_id."""
    records = (json.loads(path.read_text(encoding="utf-8")) for path in EMAILS_DIR.glob("*.json"))
    return {record["message_id"]: record for record in records}


def current_settings():
    """Settings: מה שחייב להיות זהה בין האינדקס השמור להגדרות הנוכחיות."""
    return {"embed_provider": EMBED_PROVIDER, "embed_model": Settings.embed_model.model_name,
            "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "text_version": TEXT_VERSION}


def check_index_info():
    """Check: אם מודל ה-Embeddings, החלוקה או בניית הטקסט השתנו — עוצרים ומבקשים בנייה מחדש."""
    if not INDEX_INFO_PATH.exists():
        return
    saved = json.loads(INDEX_INFO_PATH.read_text(encoding="utf-8"))
    changed = [f"{key}: {saved.get(key)} -> {value}" for key, value in current_settings().items()
               if saved.get(key) != value]
    if changed:
        raise SystemExit(f"Error: הגדרות האינדקס השתנו ({', '.join(changed)}).\n"
                         "צריך לבנות את האינדקס מחדש: python -m src.indexer --rebuild")


def index_built_at():
    """Built: מתי האינדקס נבנה (משתנה רק בבנייה מחדש). None = אין קובץ — כנראה בנייה מחדש באמצע."""
    if not INDEX_INFO_PATH.exists():
        return None
    return json.loads(INDEX_INFO_PATH.read_text(encoding="utf-8")).get("built_at", "legacy")


def save_index_info(collection):
    """Save: שומר לצד האינדקס את ההגדרות שבהן נבנה, ממד הווקטור וזמן הבנייה (יומן הבעיות #13)."""
    sample = collection.get(limit=1, include=["embeddings"])
    # Note: הוספה מצטברת לא משנה את built_at — רק בנייה מחדש (rebuild מוחק את הקובץ)
    built_at = index_built_at()
    if built_at in (None, "legacy"):
        built_at = datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")
    info = current_settings() | {"dimension": len(sample["embeddings"][0]), "built_at": built_at}
    INDEX_INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def open_collection():
    """Open: פותח (או יוצר) את האוסף ב-Chroma."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Why: cosine — המדד המקובל לטקסט. embedding_function=None — את הווקטורים מחשב Gemini, לא Chroma
    collection = client.get_or_create_collection(
        COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)
    return client, collection


def open_index():
    """Index: עוטף את אוסף Chroma ב-VectorStoreIndex של LlamaIndex."""
    _, collection = open_collection()
    return VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection)), collection


def index_new_messages():
    """Incremental: מוסיף לאינדקס רק הודעות שעוד לא בו (insert), בלי בנייה מחדש. מחזיר כמה נוספו."""
    check_index_info()
    index, collection = open_index()
    records = load_records()
    already = {metadata.get("message_id") for metadata in collection.get(include=["metadatas"])["metadatas"]}
    new_records = sorted((r for r in records.values() if r["message_id"] not in already), key=lambda r: r["date"])
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Why: מנות לפי הודעות (לא לפי Nodes) — כדי שהודעה לא תיכנס לאינדקס רק חלקית
    for start in range(0, len(new_records), BATCH_SIZE):
        batch = new_records[start:start + BATCH_SIZE]
        documents = [build_document(r, records.get(r.get("in_reply_to")), has_info)
                     for r, has_info in zip(batch, classify_info(batch))]
        index.insert_nodes(splitter.get_nodes_from_documents(documents))
        print(f"Progress: אונדקסו {start + len(batch)} מתוך {len(new_records)} הודעות חדשות")

    if new_records:
        save_index_info(collection)
    print(f"Done: {len(records)} הודעות בתיקייה, {len(new_records)} חדשות נוספו, "
          f"סה\"כ {collection.count()} Nodes באינדקס")
    return len(new_records)


def rebuild():
    """Rebuild: מוחק את האוסף ואת קובץ ההגדרות, כדי לבנות הכול מחדש."""
    client, _ = open_collection()
    client.delete_collection(COLLECTION_NAME)
    INDEX_INFO_PATH.unlink(missing_ok=True)
    print("Rebuild: האינדקס נמחק, בונים מחדש")


def main():
    parser = argparse.ArgumentParser(description="Index: אינדוקס מצטבר של ההודעות ל-Chroma")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild: מחיקת האינדקס ובנייה מחדש")
    args = parser.parse_args()

    setup_models()
    if args.rebuild:
        rebuild()
    copied = seed_samples()
    if copied:
        print(f"Seed: הועתקו {copied} הודעות דוגמה ל-data/emails")
    index_new_messages()


if __name__ == "__main__":
    main()
