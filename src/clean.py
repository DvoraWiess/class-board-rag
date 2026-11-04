import re

from bs4 import BeautifulSoup

BLOCK_TAGS = ["p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"]
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")  # Note: תגית HTML ("<p>", "</div>") — לא סתם "<" בטקסט

# Why: כותרת ציטוט ("ב-29 באוק' 2026, 08:05, יעל כתבה:") תמיד כוללת תאריך או שעה — לכן דורשים ספרה,
# Note: וכך שורה כמו "המורה כתבה:" (בלי ספרות) לא נמחקת
QUOTE_HEADER = re.compile(r"^(?=.*\d).*(כתב|כתבה):\s*$")
SIGNATURE = re.compile(r"^נשלח מ")  # Note: "נשלח מה-iPhone שלי"


def html_to_text(html):
    """HTML: ממיר HTML לטקסט, ושומר ירידות שורה בין פסקאות."""
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    # Why: שורה חדשה רק אחרי תגיות בלוק — אחרת <u>8.11</u> היה נקרע לשורה נפרדת
    for tag in soup.find_all(BLOCK_TAGS):
        tag.append("\n")
    return soup.get_text()


def clean_body(body):
    """Clean: HTML → טקסט, ואז חיתוך מכותרת ציטוט או מחתימה, הסרת שורות ציטוט (">") ורווחים מיותרים."""
    text = html_to_text(body) if HTML_TAG.search(body) else body
    lines = []
    for line in text.split("\n"):
        if QUOTE_HEADER.search(line.strip()) or SIGNATURE.search(line.strip()):
            break
        if not line.lstrip().startswith(">"):
            lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines).replace(" ", " ")).strip()
