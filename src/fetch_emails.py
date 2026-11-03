import argparse
import email
import hashlib
import imaplib
import json
from datetime import datetime, timedelta
from email import policy
from zoneinfo import ZoneInfo

from src.config import EMAILS_DIR, TIMEZONE, require_env

IMAP_HOST = "imap.gmail.com"
MAILBOX = "INBOX"
PROGRESS_EVERY = 20


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
        "body": body,
        "body_type": body_type,
    }


def save_email(record):
    """Save: שומר מייל כקובץ JSON. מחזיר False אם הוא כבר קיים."""
    path = EMAILS_DIR / email_filename(record["message_id"])
    if path.exists():
        return False
    EMAILS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def connect():
    """Connect: התחברות ל-Gmail ב-IMAP עם סיסמת אפליקציה."""
    address = require_env("GMAIL_ADDRESS")
    password = require_env("GMAIL_APP_PASSWORD")
    try:
        imap = imaplib.IMAP4_SSL(IMAP_HOST, timeout=30)
    except OSError as error:
        raise SystemExit(
            f"Error: אין חיבור ל-Gmail ({type(error).__name__}). "
            "ייתכן שהחיבור חסום בסינון — השתמשי בייבוא מקובץ (src.import_export)"
        )
    try:
        imap.login(address, password)
    except imaplib.IMAP4.error:
        raise SystemExit("Error: ההתחברות נכשלה — בדקי את GMAIL_ADDRESS ואת GMAIL_APP_PASSWORD ב-.env")
    return imap


def search_uids(imap, sender, days):
    """Search: מחזיר את מזהי ה-IMAP (UID) של המיילים מהשולח ב-N הימים האחרונים."""
    since = (datetime.now(ZoneInfo(TIMEZONE)) - timedelta(days=days)).strftime("%d-%b-%Y")
    # Note: readonly — כדי שהמשיכה לא תסמן מיילים כ"נקראו"
    imap.select(MAILBOX, readonly=True)
    _, data = imap.uid("SEARCH", None, "FROM", f'"{sender}"', "SINCE", since)
    return data[0].split()


def fetch_message_id(imap, uid):
    """Peek: מושך רק את כותרת Message-ID — כדי לבדוק אם המייל כבר קיים בלי להוריד את כולו."""
    _, data = imap.uid("FETCH", uid, "(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)])")
    header = email.message_from_bytes(data[0][1], policy=policy.default)
    return str(header["message-id"] or "").strip()


def fetch_raw(imap, uid):
    """Fetch: מושך את המייל המלא (bytes). PEEK = בלי לסמן כנקרא."""
    _, data = imap.uid("FETCH", uid, "(BODY.PEEK[])")
    return data[0][1]


def main():
    parser = argparse.ArgumentParser(description="Fetch: משיכת מיילים מ-Gmail ב-IMAP")
    parser.add_argument("--days", type=int, default=90, help="Days: כמה ימים אחורה למשוך")
    args = parser.parse_args()

    sender = require_env("SENDER_ADDRESS")
    imap = connect()
    try:
        uids = search_uids(imap, sender, args.days)
        print(f"Found: {len(uids)} מיילים מהשולח ב-{args.days} הימים האחרונים")
        new_count = existing_count = 0
        for i, uid in enumerate(uids, 1):
            message_id = fetch_message_id(imap, uid)
            if message_id and (EMAILS_DIR / email_filename(message_id)).exists():
                existing_count += 1
            else:
                record = parse_message(fetch_raw(imap, uid))
                if record and save_email(record):
                    new_count += 1
                elif record:
                    existing_count += 1
            if i % PROGRESS_EVERY == 0:
                print(f"Progress: {i}/{len(uids)}")
    finally:
        imap.logout()

    skipped = len(uids) - new_count - existing_count
    print(f"Done: {new_count} חדשים, {existing_count} כבר היו, {skipped} דולגו")


if __name__ == "__main__":
    main()
