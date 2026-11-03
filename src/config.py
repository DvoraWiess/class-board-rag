import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings

# Load: קריאת .env לתוך משתני הסביבה (רק בזמן ריצה, בלי להדפיס ערכים)
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
TIMEZONE = "Asia/Jerusalem"

# Paths: נתיבים יחסית לשורש הפרויקט, כדי שיעבדו מכל תיקייה שממנה מריצים
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMAILS_DIR = PROJECT_ROOT / "data" / "emails"
SAMPLE_EMAILS_DIR = PROJECT_ROOT / "data" / "sample_emails"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Why: טמפרטורה 0 — אנחנו רוצים תשובות עובדתיות ועקביות, לא יצירתיות
LLM_TEMPERATURE = 0.0

# Chunking: הודעות קצרות (10-100 טוקנים) — כמעט תמיד הודעה אחת = Chunk אחד
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50

# Replies: כמה תווים מההודעה המקורית מוסיפים לטקסט של תגובה
PARENT_EXCERPT_CHARS = 150

# Version: אופן בניית הטקסט שנכנס לאינדקס (ניקוי, שורת תגובה, metadata).
# Rule: כל שינוי ב-clean.py או ב-build_document באינדקסר — להעלות ב-1. המערכת תבקש בנייה מחדש.
TEXT_VERSION = 1

# Retrieval: שליפה רחבה — כדי שעדכון לא "ייפול" מחוץ לרשימה (יומן הבעיות #6)
TOP_K = 10

# Cutoff: מתחת לציון הזה ההודעה לא נשלחת ל-LLM. 0.72 = בין שאלה לא קשורה (0.67) לשאלות בתחום (0.77+)
# Note: הציון הוא של LlamaIndex: exp(-distance). ניסוי עם 0.65 / 0.72 / 0.8 — בשלב 7
SIMILARITY_CUTOFF = 0.72


def require_env(name):
    """Get: מחזיר משתנה סביבה, ועוצר עם הודעה ברורה אם הוא חסר."""
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing: המשתנה {name} חסר או ריק בקובץ .env")
    return value


def setup_models():
    """Setup: מגדיר את Settings.llm ו-Settings.embed_model לפי LLM_PROVIDER."""
    # Note: ה-import בתוך כל ענף, כדי שלא יידרשו חבילות של ספק שלא בשימוש
    if LLM_PROVIDER == "gemini":
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        from llama_index.llms.google_genai import GoogleGenAI

        api_key = require_env("GOOGLE_API_KEY")
        Settings.llm = GoogleGenAI(
            model=require_env("GEMINI_LLM_MODEL"),
            api_key=api_key,
            temperature=LLM_TEMPERATURE,
        )
        Settings.embed_model = GoogleGenAIEmbedding(
            model_name=require_env("GEMINI_EMBED_MODEL"),
            api_key=api_key,
        )
    elif LLM_PROVIDER == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI

        api_key = require_env("OPENAI_API_KEY")
        Settings.llm = OpenAI(
            model=require_env("OPENAI_LLM_MODEL"),
            api_key=api_key,
            temperature=LLM_TEMPERATURE,
        )
        Settings.embed_model = OpenAIEmbedding(
            model=require_env("OPENAI_EMBED_MODEL"),
            api_key=api_key,
        )
    else:
        raise SystemExit(f"Error: ערך לא מוכר LLM_PROVIDER={LLM_PROVIDER} (gemini או openai)")
