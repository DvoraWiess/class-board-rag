import email
import hashlib
import json
import shutil
import uuid
from datetime import datetime
from email import policy
from zoneinfo import ZoneInfo

from src.config import EMAILS_DIR, SAMPLE_EMAILS_DIR, TIMEZONE


def email_filename(message_id):
    """Name: שם קובץ בטוח ל-Windows — hash קצר של ה-message_id (שמכיל תווים כמו <>@)."""
    return hashlib.sha1(message_id.encode("utf-8")).hexdigest()[:16] + ".json"


def parse_date(msg):
    """Date: מחזיר את תאריך המייל בשעון ישראל, או None אם אין תאריך."""
    header = msg["date"]
    if header is None or header.datetime is None:
        return None
    date = header.datetime
    if date.tzinfo is None:
        # Note: תאריך בלי אזור זמן — מניחים שהוא בשעון ישראל
        date = date.replace(tzinfo=ZoneInfo(TIMEZONE))
    return date.astimezone(ZoneInfo(TIMEZONE))


def parse_body(msg):
    """Body: מחזיר (גוף, סוג) — עדיפות לטקסט רגיל, ואם אין אז HTML."""
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return "", "text"
    body_type = "text" if part.get_content_type() == "text/plain" else "html"
    return part.get_content(), body_type


def parse_message(raw_bytes):
    """Parse: ממיר מייל גולמי (bytes) למילון בפורמט האחיד, או None אם אי אפשר."""
    msg = email.message_from_bytes(raw_bytes, policy=policy.default)
    subject = str(msg["subject"] or "")
    sender = str(msg["from"] or "")
    date = parse_date(msg)
    if date is None:
        print("Warning: מייל בלי תאריך — דילגנו עליו")
        return None

    message_id = str(msg["message-id"] or "").strip()
    if not message_id:
        # Why: מזהה יציב — אותו מייל תמיד יקבל אותו מזהה, ולכן לא ייווצרו כפילויות
        stable_key = f"{sender}|{date.isoformat()}|{subject}"
        message_id = "generated-" + hashlib.sha1(stable_key.encode("utf-8")).hexdigest()
        print("Warning: מייל בלי Message-ID — נוצר מזהה מ-hash של שולח+תאריך+נושא")

    body, body_type = parse_body(msg)
    return {
        "message_id": message_id,
        "date": date.isoformat(),
        "subject": subject,
        "from": sender,
        "in_reply_to": str(msg["in-reply-to"] or "").strip() or None,
        "body": body,
        "body_type": body_type,
    }


def new_post(name, subject, body, in_reply_to=None):
    """Post: הודעה חדשה ממסך "הוספת הודעה" — באותו פורמט. תאריך ומזהה נקבעים אוטומטית."""
    return {
        "message_id": f"<post-{uuid.uuid4().hex}@board.local>",
        "date": datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds"),
        "subject": subject.strip(),
        "from": name.strip(),
        "in_reply_to": in_reply_to,
        "body": body.strip(),
        "body_type": "text",
    }


def save_email(record, folder=EMAILS_DIR):
    """Save: שומר הודעה כקובץ JSON. מחזיר False אם היא כבר קיימת."""
    path = folder / email_filename(record["message_id"])
    if path.exists():
        return False
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def seed_samples():
    """Seed: אם data/emails ריקה — מעתיק אליה את מיילי הדוגמה. מחזיר כמה הועתקו."""
    if EMAILS_DIR.exists() and any(EMAILS_DIR.glob("*.json")):
        return 0
    EMAILS_DIR.mkdir(parents=True, exist_ok=True)
    files = list(SAMPLE_EMAILS_DIR.glob("*.json"))
    for file in files:
        shutil.copy2(file, EMAILS_DIR / file.name)
    return len(files)
