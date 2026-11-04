import argparse
import email
import hashlib
import mailbox
from email import policy
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config import TIMEZONE
from src.messages import save_email


def parse_message(raw_bytes):
    """Parse: ממיר מייל גולמי (bytes) למילון בפורמט האחיד, או None אם אין לו תאריך."""
    msg = email.message_from_bytes(raw_bytes, policy=policy.default)
    if msg["date"] is None or msg["date"].datetime is None:
        print("Warning: מייל בלי תאריך — דילגנו עליו")
        return None
    date = msg["date"].datetime
    if date.tzinfo is None:
        # Note: תאריך בלי אזור זמן — מניחים שהוא בשעון ישראל
        date = date.replace(tzinfo=ZoneInfo(TIMEZONE))
    date = date.astimezone(ZoneInfo(TIMEZONE))
    subject, sender = str(msg["subject"] or ""), str(msg["from"] or "")

    message_id = str(msg["message-id"] or "").strip()
    if not message_id:
        # Why: מזהה יציב — אותו מייל תמיד יקבל אותו מזהה, ולכן לא ייווצרו כפילויות
        stable_key = f"{sender}|{date.isoformat()}|{subject}"
        message_id = "generated-" + hashlib.sha1(stable_key.encode("utf-8")).hexdigest()
        print("Warning: מייל בלי Message-ID — נוצר מזהה מ-hash של שולח+תאריך+נושא")

    # Note: עדיפות לטקסט רגיל; אם אין — HTML (clean.py ממיר אותו לטקסט באינדוקס)
    part = msg.get_body(preferencelist=("plain", "html"))
    return {
        "message_id": message_id,
        "date": date.isoformat(),
        "subject": subject,
        "from": sender,
        "in_reply_to": str(msg["in-reply-to"] or "").strip() or None,
        "body": part.get_content() if part else "",
        "body_type": "html" if part and part.get_content_type() == "text/html" else "text",
    }


def read_raw_messages(path):
    """Read: מחזיר מיילים גולמיים (bytes) מקובץ .mbox, מקובץ .eml, או מכל אלה שבתוך תיקייה."""
    if path.is_dir():
        # Why: rglob — חילוץ של Takeout יוצר תתי-תיקיות (Takeout/Mail/...)
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in (".mbox", ".eml"))
    elif path.is_file():
        files = [path]
    else:
        raise SystemExit(f"Error: הנתיב {path} לא קיים")
    if not files:
        raise SystemExit(f"Error: לא נמצאו קבצי .mbox או .eml בנתיב {path} (קובץ zip צריך לחלץ קודם)")

    for file in files:
        if file.suffix.lower() == ".eml":
            yield file.read_bytes()
        else:
            mbox = mailbox.mbox(file, create=False)
            for key in mbox.keys():
                yield mbox.get_bytes(key)


def import_file(path):
    """Import: מפענח ושומר את כל המיילים בפורמט האחיד. מחזיר (סה"כ בקובץ, נשמרו חדשים, נכשלו)."""
    total = saved = failed = 0
    for raw in read_raw_messages(path):
        total += 1
        record = parse_message(raw)
        if record is None:
            failed += 1
        elif save_email(record):
            saved += 1
    return total, saved, failed


def main():
    parser = argparse.ArgumentParser(description="Import: ייבוא מיילים מקובץ .mbox / .eml או מתיקייה")
    parser.add_argument("path", type=Path, help="Path: קובץ .mbox / .eml, או תיקייה שמכילה אותם")
    args = parser.parse_args()
    total, saved, failed = import_file(args.path)
    print(f"Done: {total} מיילים בקובץ, {saved} חדשים נשמרו, {failed} לא נקראו")


if __name__ == "__main__":
    main()
