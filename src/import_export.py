import argparse
import mailbox
from pathlib import Path

from src.messages import parse_message, save_email


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
