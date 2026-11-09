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
TEXT_VERSION = 4

# Retrieval: שליפה רחבה, כדי שעדכון לא ייפול מחוץ לרשימה (יומן הבעיות #6)
TOP_K = 10
# Cutoff: ציון LlamaIndex = exp(-distance). תלוי במודל ה-Embeddings — לכייל מחדש בכל החלפה (#5, #26).
# OpenAI text-embedding-3-small: מחוץ לתחום 0.46-0.54, בתחום 0.60+ → 0.57. (Gemini היה 0.72.)
SIMILARITY_CUTOFF = 0.57
# Penalty: הציון של הודעה בלי מידע (שאלה, תודה) מוכפל בזה. 1.0 = כבוי (יומן הבעיות #8)
NO_INFO_PENALTY = 0.95

# Provider: openai (בתשלום — ברירת המחדל) או gemini (חינמי, מוגבל במכסות). קובע גם את ה-LLM וגם את ה-Embeddings.
# Note: החלפת ספק = מודל Embeddings אחר → חובה --rebuild (check_index_info עוצר ומודיע)
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
# Judge: נשאר Gemini בכל מקרה — מודל אחר (וגם ספק אחר) מזה שכותב את התשובות
JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.5-flash").strip()


def require_env(name):
    """Get: משתנה סביבה, או עצירה עם הודעה ברורה אם הוא חסר."""
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing: המשתנה {name} חסר או ריק בקובץ .env")
    return value


def setup_models():
    """Setup: מגדיר את Settings.llm ו-Settings.embed_model לפי MODEL_PROVIDER. טמפרטורה 0 — תשובות עובדתיות."""
    if MODEL_PROVIDER == "openai":
        # Note: import כאן — כדי שהחבילות של OpenAI יידרשו רק כשמשתמשים בו. truststore (src/__init__.py) מכסה גם אותו
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI

        api_key = require_env("OPENAI_API_KEY")
        Settings.llm = OpenAI(model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini").strip(), api_key=api_key,
                              temperature=0.0)
        Settings.embed_model = OpenAIEmbedding(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip(), api_key=api_key)
    elif MODEL_PROVIDER == "gemini":
        api_key = require_env("GOOGLE_API_KEY")
        Settings.llm = GoogleGenAI(model=require_env("GEMINI_LLM_MODEL"), api_key=api_key, temperature=0.0)
        Settings.embed_model = GoogleGenAIEmbedding(model_name=require_env("GEMINI_EMBED_MODEL"), api_key=api_key)
    else:
        raise SystemExit(f"Error: ערך לא מוכר MODEL_PROVIDER={MODEL_PROVIDER} (openai או gemini)")


def judge_llm(answer_model):
    """Judge: LLM שופט להערכה — חייב להיות מודל אחר מזה שכותב את התשובות."""
    if JUDGE_MODEL == answer_model:
        raise SystemExit(f"Error: מודל השופט ({JUDGE_MODEL}) זהה למודל שכותב את התשובות")
    # Note: callback_manager ריק — טוקני השופט לא נספרים; max_retries גבוה — 503 נפוץ במסלול החינמי
    return GoogleGenAI(model=JUDGE_MODEL, api_key=require_env("GOOGLE_API_KEY"), temperature=0.0,
                       max_retries=8, callback_manager=CallbackManager([]))
