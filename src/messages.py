import hashlib
import json
import shutil
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from src.config import MESSAGES_DIR, SAMPLE_MESSAGES_DIR, TIMEZONE


def message_filename(message_id):
    """Name: שם קובץ בטוח ל-Windows — hash קצר של ה-message_id (שמכיל תווים כמו <>@)."""
    return hashlib.sha1(message_id.encode("utf-8")).hexdigest()[:16] + ".json"


def new_post(name, subject, body, in_reply_to=None):
    """Post: הודעה חדשה מהטופס. התאריך (שעון ישראל) והמזהה נקבעים אוטומטית."""
    return {"message_id": f"<post-{uuid.uuid4().hex}@board.local>",
            "date": datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds"),
            "subject": subject.strip(), "from": name.strip(), "in_reply_to": in_reply_to, "body": body.strip()}


def save_message(record, folder=MESSAGES_DIR):
    """Save: שומר הודעה כקובץ JSON. מחזיר False אם היא כבר קיימת."""
    path = folder / message_filename(record["message_id"])
    if path.exists():
        return False
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def seed_samples():
    """Seed: אם data/messages ריקה — מעתיק אליה את הודעות הדוגמה. מחזיר כמה הועתקו."""
    if any(MESSAGES_DIR.glob("*.json")):
        return 0
    MESSAGES_DIR.mkdir(parents=True, exist_ok=True)
    files = list(SAMPLE_MESSAGES_DIR.glob("*.json"))
    for file in files:
        shutil.copy2(file, MESSAGES_DIR / file.name)
    return len(files)
