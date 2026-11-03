import argparse
import mailbox
from collections import Counter
from email import policy
from email.parser import BytesHeaderParser
from email.utils import parseaddr
from pathlib import Path

from src.config import EMAILS_DIR
from src.fetch_emails import email_filename, parse_message, save_email


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


def is_to_group(raw, group):
    """Filter: האם כתובת הקבוצה מופיעה בנמענים (To / Cc). קורא רק כותרות, לא את הגוף."""
    headers = BytesHeaderParser(policy=policy.default).parsebytes(raw)
    recipients = f"{headers['to'] or ''} {headers['cc'] or ''}"
    return group.lower() in recipients.lower()


def load_records(path, group):
    """Load: מפענח את כל המיילים. מחזיר (רשומות, סה"כ בקובץ, לא לקבוצה, נכשלו)."""
    records = []
    total = not_to_group = failed = 0
    for raw in read_raw_messages(path):
        total += 1
        if group and not is_to_group(raw, group):
            not_to_group += 1
            continue
        record = parse_message(raw)
        if record is None:
            failed += 1
        else:
            records.append(record)
    return records, total, not_to_group, failed


def print_summary(records, total, not_to_group, failed):
    """Summary: סטטיסטיקה בלבד — בלי תוכן מיילים."""
    ids = Counter(r["message_id"] for r in records)
    unique_ids = list(ids)
    duplicates = len(records) - len(unique_ids)
    already_saved = sum((EMAILS_DIR / email_filename(i)).exists() for i in unique_ids)
    senders = {parseaddr(r["from"])[1].lower() for r in records}
    body_types = Counter(r["body_type"] for r in records)
    dates = sorted(r["date"] for r in records)

    print(f"Total in file: {total}")
    if not_to_group:
        print(f"Not to group: {not_to_group} (דולגו)")
    print(f"Failed to parse: {failed}")
    print(f"Parsed: {len(records)}")
    print(f"Duplicates in file: {duplicates} (אותו message_id יותר מפעם אחת)")
    print(f"Unique: {len(unique_ids)}, already saved: {already_saved}, new: {len(unique_ids) - already_saved}")
    if dates:
        print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Senders: {len(senders)} כתובות שונות")
    print(f"Body types: {dict(body_types)}")


def main():
    parser = argparse.ArgumentParser(description="Import: ייבוא מיילים מקובץ .mbox / .eml או מתיקייה")
    parser.add_argument("path", type=Path, help="Path: קובץ .mbox / .eml, או תיקייה שמכילה אותם")
    parser.add_argument("--group", help="Group: לשמור רק מיילים שנשלחו לכתובת הקבוצה הזו (אופציונלי)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run: רק סטטיסטיקה, בלי לשמור")
    args = parser.parse_args()

    records, total, not_to_group, failed = load_records(args.path, args.group)
    print_summary(records, total, not_to_group, failed)
    if args.dry_run:
        print("Dry run: לא נשמר שום קובץ")
        return

    new_count = sum(save_email(r) for r in records)
    print(f"Done: {new_count} מיילים חדשים נשמרו ב-{EMAILS_DIR}")


if __name__ == "__main__":
    main()
