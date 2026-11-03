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

# Why: טמפרטורה 0 — אנחנו רוצים תשובות עובדתיות ועקביות, לא יצירתיות
LLM_TEMPERATURE = 0.0


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
