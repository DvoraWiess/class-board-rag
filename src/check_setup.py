import httpx
from llama_index.core import Settings

from src.config import LLM_PROVIDER, setup_models

NETWORK_HINT = "Hint: ייתכן שהחיבור ל-API חסום בסינון (נטפרי) — בדקי שהדומיין generativelanguage.googleapis.com פתוח"


def check_llm():
    """LLM: שולח 'שלום' למודל ומדפיס את התשובה."""
    response = Settings.llm.complete("שלום! ענה במשפט אחד קצר.")
    print("LLM response:", response.text.strip())


def check_embedding():
    """Embedding: ממיר משפט אחד לווקטור ומדפיס את אורכו ואת 5 המספרים הראשונים."""
    vector = Settings.embed_model.get_text_embedding("מחר אין שיעור ראשון")
    print("Embedding length:", len(vector))
    print("First 5 numbers:", [round(x, 4) for x in vector[:5]])


def main():
    print(f"Provider: {LLM_PROVIDER}")
    try:
        setup_models()
        print(f"Models: LLM={Settings.llm.model}, Embeddings={Settings.embed_model.model_name}")
        check_llm()
        check_embedding()
    except SystemExit:
        raise
    except Exception as error:
        # Note: מדפיסים רק את סוג השגיאה וההודעה, בלי מפתחות
        print(f"Failed: {type(error).__name__}: {error}")
        # Why: הרמז על נטפרי רלוונטי רק לשגיאת חיבור, לא לשגיאה שהשרת עצמו החזיר
        if isinstance(error, httpx.ConnectError):
            print(NETWORK_HINT)
        raise SystemExit(1)
    print("OK: ההגדרה עובדת")


if __name__ == "__main__":
    main()
