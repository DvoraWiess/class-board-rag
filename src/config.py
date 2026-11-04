import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# Load: קריאת .env (רק בזמן ריצה, בלי להדפיס ערכים)
load_dotenv()

TIMEZONE = "Asia/Jerusalem"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MESSAGES_DIR = PROJECT_ROOT / "data" / "messages"
SAMPLE_MESSAGES_DIR = PROJECT_ROOT / "data" / "sample_messages"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Chunking: הודעות קצרות — כמעט תמיד הודעה אחת = Chunk אחד
CHUNK_SIZE, CHUNK_OVERLAP = 1024, 50
# Replies: כמה תווים מההודעה המקורית נכנסים לחיפוש של תגובה (יומן הבעיות #12)
PARENT_EXCERPT_CHARS = 150
# Version: להעלות ב-1 בכל שינוי באופן בניית הטקסט באינדקס — המערכת תבקש בנייה מחדש
TEXT_VERSION = 3

# Retrieval: שליפה רחבה, כדי שעדכון לא ייפול מחוץ לרשימה (יומן הבעיות #6)
TOP_K = 10
# Cutoff: ציון LlamaIndex = exp(-distance). 0.72 = בין שאלה לא קשורה (0.67) לשאלות בתחום (0.77+) (#5)
SIMILARITY_CUTOFF = 0.72
# Penalty: הציון של הודעה בלי מידע (שאלה, תודה) מוכפל בזה. 1.0 = כבוי (יומן הבעיות #8)
NO_INFO_PENALTY = 0.95

JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.5-flash").strip()


def require_env(name):
    """Get: משתנה סביבה, או עצירה עם הודעה ברורה אם הוא חסר."""
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing: המשתנה {name} חסר או ריק בקובץ .env")
    return value


def setup_models():
    """Setup: מגדיר את Settings.llm ו-Settings.embed_model (Gemini). טמפרטורה 0 — תשובות עובדתיות."""
    api_key = require_env("GOOGLE_API_KEY")
    Settings.llm = GoogleGenAI(model=require_env("GEMINI_LLM_MODEL"), api_key=api_key, temperature=0.0)
    Settings.embed_model = GoogleGenAIEmbedding(model_name=require_env("GEMINI_EMBED_MODEL"), api_key=api_key)


def judge_llm(answer_model):
    """Judge: LLM שופט להערכה — חייב להיות מודל אחר מזה שכותב את התשובות."""
    if JUDGE_MODEL == answer_model:
        raise SystemExit(f"Error: מודל השופט ({JUDGE_MODEL}) זהה למודל שכותב את התשובות")
    # Note: callback_manager ריק — טוקני השופט לא נספרים; max_retries גבוה — 503 נפוץ במסלול החינמי
    return GoogleGenAI(model=JUDGE_MODEL, api_key=require_env("GOOGLE_API_KEY"), temperature=0.0,
                       max_retries=8, callback_manager=CallbackManager([]))
