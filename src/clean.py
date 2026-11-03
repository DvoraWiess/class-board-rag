import argparse
import json
import re
import sys

from bs4 import BeautifulSoup

from src.config import EMAILS_DIR
from src.messages import email_filename

BLOCK_TAGS = ["p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"]

# Note: תווי כיווניות בלתי נראים (LRM, RLM, LRE...) — Gmail מוסיף אותם סביב שמות וכתובות
BIDI_MARKS = re.compile("[‎‏‪-‮⁦-⁩]")

# Why: כותרת של ציטוט תמיד כוללת תאריך או שעה — לכן דורשים ספרה בשורה.
# Note: כך שורה כמו "המורה כתבה:" (בלי ספרות) לא תיחשב ציטוט ולא תימחק
QUOTE_HEADERS = [
    re.compile(r"^(?=.*\d).*(כתב|כתבה|wrote):\s*$"),  # Gmail: "ב-29 באוק' 2026, 08:05, יעל כתבה:"
    re.compile(r"^בתאריך(?=.*\d).*מאת.*:\s*$"),  # Gmail בעברית: "בתאריך יום ה׳, 29 באוק׳ 2026 ב-8:05 מאת ...:"
    re.compile(r"^-{3,}\s*(Original Message|הודעה מקורית)\s*-{3,}$", re.IGNORECASE),  # Outlook
]

# Note: חתימה של מכשיר ("נשלח מה-iPhone שלי"), או שורת המפריד התקנית לחתימה "-- "
SIGNATURE_START = re.compile(r"^(נשלח מ|Sent from my |--\s*$)")


def html_to_text(html):
    """HTML: ממיר HTML לטקסט, ושומר ירידות שורה בין פסקאות."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    # Why: שורה חדשה רק אחרי תגיות בלוק — אחרת <u>8.11</u> היה נקרע לשורה נפרדת
    for tag in soup.find_all(BLOCK_TAGS):
        tag.append("\n")
    return soup.get_text()


def cut_at_first(lines, patterns):
    """Cut: מחזיר את השורות עד (ולא כולל) השורה הראשונה שמתאימה לאחת התבניות."""
    for i, line in enumerate(lines):
        if any(p.search(line.strip()) for p in patterns):
            return lines[:i]
    return lines


def remove_quotes(text):
    """Quotes: מוחק ציטוט של הודעה קודמת — מכותרת הציטוט והלאה, וגם שורות שמתחילות ב->."""
    lines = cut_at_first(text.split("\n"), QUOTE_HEADERS)
    return "\n".join(line for line in lines if not line.lstrip().startswith(">"))


def remove_signature(text):
    """Signature: מוחק חתימה — משורת החתימה והלאה."""
    return "\n".join(cut_at_first(text.split("\n"), [SIGNATURE_START]))


def normalize_whitespace(text):
    """Whitespace: רווחים בסוף שורות, יותר משורה ריקה אחת ברצף, ורווחים בהתחלה ובסוף."""
    text = text.replace(" ", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_body(body, body_type="text"):
    """Clean: הניקוי המלא של גוף הודעה. הסדר חשוב — קודם HTML, ורק אז חיפוש לפי שורות."""
    text = html_to_text(body) if body_type == "html" else body
    text = BIDI_MARKS.sub("", text)
    text = remove_quotes(text)
    text = remove_signature(text)
    return normalize_whitespace(text)


def load_message(message_id):
    """Load: טוען הודעה מ-data/emails לפי message_id."""
    path = EMAILS_DIR / email_filename(message_id)
    if not path.exists():
        raise SystemExit(f"Error: לא נמצאה הודעה עם message_id={message_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Clean: הצגת לפני/אחרי של ניקוי הודעות")
    parser.add_argument("message_ids", nargs="*", help="IDs: הודעות להצגה מלאה לפני/אחרי")
    parser.add_argument("--summary", action="store_true", help="Summary: אורך לפני/אחרי לכל ההודעות, בלי תוכן")
    args = parser.parse_args()
    # Why: תוכן הודעות יכול לכלול אימוג'י, שאין בקידוד cp1255 (יומן הבעיות #4) — מחליפים ב-? במקום לקרוס
    sys.stdout.reconfigure(errors="replace")

    if args.summary:
        changed = 0
        for path in sorted(EMAILS_DIR.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            before, after = len(record["body"]), len(clean_body(record["body"], record["body_type"]))
            if before != after:
                changed += 1
                print(f"{record['message_id']}: {before} to {after} chars ({record['body_type']})")
        print(f"Changed: {changed} הודעות")

    for message_id in args.message_ids:
        record = load_message(message_id)
        print(f"===== {message_id} ({record['body_type']}) =====")
        print("--- Before:")
        print(record["body"])
        print("--- After:")
        print(clean_body(record["body"], record["body_type"]))
        print()


if __name__ == "__main__":
    main()
