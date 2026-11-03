# Problems log

לכל בעיה: מה הבעיה, איך גילינו, מה הסיבה, מה תיקנו, מה השתנה אחרי התיקון.

<!-- Template:
## N. כותרת קצרה
- **Problem:**
- **How found:**
- **Cause:**
- **Fix:**
- **Result:**
-->

## 1. SSL error when connecting to Gemini (NetFree)
- **Problem:** `CERTIFICATE_VERIFY_FAILED: CA cert does not include key usage extension` בכל קריאה ל-`generativelanguage.googleapis.com`.
- **How found:** בדיקת חיבור בשלב 0, עוד לפני שהיה מפתח API.
- **Cause:** נטפרי פותח את תעבורת ה-HTTPS לבדיקה וחותם עליה בתעודה משלו (`SSL_CERT_FILE` מצביע על תעודת נטפרי).
  Python 3.13 מפעיל כברירת מחדל בדיקה מחמירה (`VERIFY_X509_STRICT`), ותעודת נטפרי לא עומדת בה.
- **Fix:** חבילת `truststore` — אימות דרך מאגר התעודות של Windows, שמקבל את תעודת נטפרי.
  הקריאה `truststore.inject_into_ssl()` נמצאת ב-`src/__init__.py`, שרץ לפני כל מודול ב-`src`.
  נקודות כניסה מחוץ ל-`src` (`app.py`, `tests/evaluate.py`) מתחילות ב-`import src`.
  נשקלו גם: ביטול הבדיקה המחמירה בקוד (פחות בטוח), מעבר ל-Python 3.12 (התקנה נוספת).
- **Result:** החיבור עובר. Embedding התקבל (3072 מספרים).

## 2. Gemini LLM returns 503 "high demand"
- **Problem:** `gemini-3.8-flash` מחזיר `503 UNAVAILABLE` גם אחרי 3 ניסיונות חוזרים.
- **How found:** `check_setup` בשלב 0 (2026-11-03). ה-Embeddings עבדו באותו זמן.
- **Cause:** עומס זמני בצד של Google — לא בעיה בקוד או בחיבור.
- **Fix:** מעבר ל-`gemini-3.5-flash-lite` בשינוי שורה אחת ב-`.env`, בלי לגעת בקוד.
  בשלב 7 נשווה שוב מול `gemini-3.8-flash` (איכות עברית וטיפול בסתירות).
- **Result:** ה-LLM עונה בעברית, `check_setup` עובר.
- **Lesson:** הרמז על נטפרי הוצג גם בשגיאה הזו והטעה — עכשיו הוא מוצג רק בשגיאת חיבור (`httpx.ConnectError`).

## 3. Data source: IMAP blocked → static Takeout → messages through the system
- **Problem:** אי אפשר למשוך מיילים ב-IMAP.
- **How found:** בשלב 1, בניסיון ליצור סיסמת אפליקציה (2026-11-03). החיבור לשרת עצמו עבד (`Gimap ready`).
- **Cause:** Gmail דורש סיסמת אפליקציה, שדורשת אימות דו-שלבי — ו-Google חסמה את הפעלת האימות הדו-שלבי בחשבון.
- **Fix, step 1 — Takeout:** ייבוא קובץ `.mbox` מ-Google Takeout דרך `src/import_export.py`
  (עם `--dry-run` לסטטיסטיקה בלי שמירה). הבעיה: Takeout הוא **תמונת מצב קפואה** — כל עדכון דורש
  ייצוא חדש ידני, והמערכת צריכה מידע שמתעדכן כל יום.
- **Fix, step 2 — messages through the system:** המערכת הפכה ל"לוח הודעות כיתתי עם בוט שאלות".
  הודעות נכנסות דרך מסך "הוספת הודעה" ב-Streamlit, נשמרות ב-`save_email` באותו פורמט JSON,
  ונכנסות מיד לאינדקס (`insert` מצטבר). הנתונים ההתחלתיים: הודעות דוגמה מומצאות.
  `import_export.py` נשאר כאפשרות ייבוא. `fetch_emails.py` נמחק (שמור ב-commit `e34ceba`),
  והפונקציות המשותפות עברו ל-`src/messages.py`. נוסף השדה `in_reply_to` לשרשורים.
- **Result:** מקור נתונים חי ופשוט, בלי תלות ב-Gmail. כל המקורות (מסך, דוגמה, ייבוא) — פורמט אחד.
- **Also learned:** הקבוצה היא שיחה בין תלמידות בלבד (הרבה שולחות, אין מורות). לכן:
  - סינון לפי `From` (כמו שתוכנן בהתחלה) לא מתאים — השולחות שונות בכל הודעה.
  - **אין מקור רשמי**: בסתירה — מציגים את שתי הגרסאות עם שם ותאריך (שלב 5).
  - השדה `from` חשוב לתשובות (מי מסרה את המידע).
  - יש הודעות לא רשמיות (שאלות, תודה) ושרשורים שעלולים להכניס רעש לאחזור — לטפל בשלב 2/3.

## 4. UnicodeEncodeError when printing to a Hebrew Windows terminal
- **Problem:** `print` של החץ `→` קרס עם `UnicodeEncodeError: 'charmap' codec can't encode character`.
- **How found:** בדיקת `--dry-run` מ-Git Bash (פלט מנותב, לא חלון PowerShell).
- **Cause:** כשהפלט מנותב, פייתון ב-Windows עברית משתמש בקידוד הישן `cp1255`, שיש בו עברית אבל אין בו `→` (וגם לא אימוג'י).
- **Fix:** החלפת `→` ב-`to`. כלל: בהדפסות למסך — רק עברית, אנגלית ומספרים. לא מדפיסים תוכן מיילים (שעלול להכיל אימוג'י) גם ככה.
- **Result:** `--dry-run` עובד גם ב-PowerShell וגם בפלט מנותב.
