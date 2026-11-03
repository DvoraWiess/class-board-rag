# Why: נטפרי חותם על תעבורת HTTPS בתעודה משלו, ו-Python 3.13 דוחה אותה בבדיקה המחמירה
# Fix: truststore מאמת תעודות דרך מאגר התעודות של Windows, שמכיר את תעודת נטפרי
# Note: הקובץ הזה רץ אוטומטית לפני כל מודול ב-src, לכן כל נקודת כניסה מכוסה.
# Rule: קבצים מחוץ ל-src (app.py, tests/evaluate.py) חייבים להתחיל ב-import src
import truststore

truststore.inject_into_ssl()
