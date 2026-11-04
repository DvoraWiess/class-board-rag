import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# Load: קריאת .env לתוך משתני הסביבה (רק בזמן ריצה, בלי להדפיס ערכים)
load_dotenv()

# Provider: רק Gemini. נשמר ב-index_info.json — אם יוחלף ספק, המערכת תבקש בנייה מחדש
EMBED_PROVIDER = "gemini"
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
# Note: 2 = הציטוט מההודעה המקורית רק לחיפוש, ול-LLM רק "בתגובה להודעה של X מ-..." (יומן הבעיות #12)
# Note: 3 = סימון has_info לכל הודעה (יומן הבעיות #8)
TEXT_VERSION = 3

# Penalty: הציון של הודעה בלי מידע (שאלה, תודה, ברכה) מוכפל בזה. 1.0 = כבוי (יומן הבעיות #8)
NO_INFO_PENALTY = 0.95

# Retrieval: שליפה רחבה — כדי שעדכון לא "ייפול" מחוץ לרשימה (יומן הבעיות #6)
TOP_K = 10

# Cutoff: מתחת לציון הזה ההודעה לא נשלחת ל-LLM. 0.72 = בין שאלה לא קשורה (0.67) לשאלות בתחום (0.77+)
# Note: הציון הוא של LlamaIndex: exp(-distance). ניסוי עם 0.65 / 0.72 / 0.8 — בשלב 7
SIMILARITY_CUTOFF = 0.72

# Judge: מודל חזק יותר מזה שכותב את התשובות, ושונה ממנו (שלב 7)
DEFAULT_JUDGE_MODEL = "gemini-3.5-flash"


def require_env(name):
    """Get: מחזיר משתנה סביבה, ועוצר עם הודעה ברורה אם הוא חסר."""
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing: המשתנה {name} חסר או ריק בקובץ .env")
    return value


def setup_models():
    """Setup: מגדיר את Settings.llm ו-Settings.embed_model (Gemini)."""
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


def judge_llm(answer_model):
    """Judge: LLM נפרד ששופט תשובות בהערכה. עוצר אם הוא זהה למודל שכותב את התשובות."""
    model = os.getenv("GEMINI_JUDGE_MODEL", DEFAULT_JUDGE_MODEL).strip()
    if model == answer_model:
        raise SystemExit(f"Error: מודל השופט ({model}) זהה למודל שכותב את התשובות — מודל לא ישפוט את עצמו")
    # Note: callback_manager ריק — כדי שטוקני השופט לא ייספרו כטוקנים של המערכת עצמה
    # Note: max_retries גבוה — עומס (503) נפוץ במסלול החינמי
    return GoogleGenAI(model=model, api_key=require_env("GOOGLE_API_KEY"), temperature=0.0,
                       max_retries=8, callback_manager=CallbackManager([]))
