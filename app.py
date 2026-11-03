# Run: .\.venv\Scripts\streamlit.exe run app.py
import src  # noqa: F401  (Must: ראשון — מפעיל את truststore)

import streamlit as st
from chromadb.api.client import SharedSystemClient

from src.config import PROJECT_ROOT, setup_models
from src.import_export import import_file
from src.indexer import index_built_at, index_new_messages, load_records, open_index, readable_date, sender_name
from src.messages import new_post, save_email, seed_samples
from src.query import NO_INFO_ANSWER, Answer, ask_stream, attach_token_counter, finish_stream

UPLOADS_DIR = PROJECT_ROOT / "data" / "raw" / "uploads"
REPLY_CHOICES = 30
ANSWER_ERROR = "אירעה שגיאה בזמן החיפוש או כתיבת התשובה (ייתכן עומס זמני ב-Gemini). נסי שוב בעוד רגע."

# Note: כיוון RTL לכל האפליקציה; קוד נשאר משמאל לימין
RTL_CSS = """
<style>
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] { direction: rtl; }
p, li, h1, h2, h3, label, input, textarea, [data-testid="stMarkdownContainer"] { text-align: right; }
code, pre { direction: ltr; text-align: left; }
</style>
"""


@st.cache_resource(show_spinner="Loading: טוען מודלים ומעדכן את האינדקס...")
def startup():
    """Startup: פעם אחת לכל הפעלה — מודלים, מונה טוקנים, הודעות דוגמה, אינדוקס מצטבר, ופתיחת האינדקס.

    Why: האינדקס נפתח כאן פעם אחת, ולא בכל שאלה (יומן הבעיות #16).
    """
    setup_models()
    counter = attach_token_counter()
    seed_samples()
    index_new_messages()
    index, _ = open_index()
    return {"counter": counter, "built_at": index_built_at(), "index": index}


def check_index_fresh(state):
    """Fresh: אם האינדקס נבנה מחדש מתהליך אחר מאז שהאפליקציה עלתה — טוענים מחדש.

    Why: Chroma מקומי לא מיועד לכמה תהליכים בו-זמנית; תהליך פתוח ראה מצב ישן (יומן הבעיות #13).
    """
    built_at = index_built_at()
    if built_at is None:
        # Note: אין index_info.json — בנייה מחדש באמצע. לא טוענים, כדי לא לאנדקס במקביל לתהליך האחר
        st.warning("האינדקס נבנה כרגע מחדש. נסי שוב בעוד רגע.")
        st.stop()
    if built_at != state["built_at"]:
        SharedSystemClient.clear_system_cache()
        st.cache_resource.clear()
        st.toast("האינדקס נבנה מחדש — נטען מחדש")
        st.rerun()


def index_now():
    """Index: אינדוקס מצטבר. אם נכשל — ההודעה כבר שמורה בקובץ ותיכנס בפעם הבאה."""
    try:
        return index_new_messages()
    except Exception as error:
        st.warning(f"Note: ההודעה נשמרה, אבל לא נכנסה עדיין לחיפוש ({type(error).__name__}). היא תיכנס בהפעלה הבאה.")
        return 0


def render_answer(answer, question):
    """Answer: ניסוח החיפוש, מקורות מעוצבים וטוקנים — מתחת לטקסט התשובה."""
    if answer.search_query and answer.search_query != question:
        st.caption(f"חיפשתי: {answer.search_query}")
    if answer.sources:
        earlier = " · יש גם הודעות קודמות בנושא" if answer.has_earlier else ""
        st.caption(f"מבוסס על הודעה מ{answer.latest_date}{earlier}")
        title = "כל ההודעות שנשלחו" if answer.sources_fallback else "מקורות"
        with st.expander(f"{title} ({len(answer.sources)})"):
            for s in answer.sources:
                st.markdown(f"**{s['date']}** · {s['from']} · {s['subject']}")
                st.caption(s["snippet"])
    t = answer.tokens
    if answer.llm_called:
        st.caption(
            f"טוקנים: LLM {t['llm_prompt']} נשלחו + {t['llm_completion']} בתשובה, כולל הניסוח לחיפוש "
            f"(נתון מ-Gemini) · Embedding {t['embedding']} (הערכה)"
        )
    else:
        st.caption(f"לא נמצאה הודעה מתאימה — לא נשלחה קריאה ל-LLM שעונה · Embedding {t['embedding']} (הערכה)")


def questions_tab(state):
    """Tab: שאלות. השאלה והתשובה האחרונות נשלחות רק לשלב הניסוח מחדש — לא ל-LLM שעונה."""
    counter = state["counter"]
    if st.button("שיחה חדשה"):
        st.session_state.history = []

    for item in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"].text)
            render_answer(item["answer"], item["question"])

    question = st.chat_input("שאלי שאלה על ההודעות...")
    if not question:
        return
    history = [(item["question"], item["answer"].text) for item in st.session_state.history]
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            stream, nodes, search_query = ask_stream(question, counter, history, state["index"])
            if stream is None:
                st.markdown(NO_INFO_ANSWER)
                answer = Answer(NO_INFO_ANSWER, search_query=search_query,
                                tokens={"llm_prompt": counter.prompt_llm_token_count,
                                        "llm_completion": counter.completion_llm_token_count,
                                        "embedding": counter.total_embedding_token_count})
            else:
                st.write_stream(stream)
                answer = finish_stream(stream, nodes, counter, search_query)
        except Exception as error:
            # Why: בלי זה שגיאה (למשל 503 מ-Gemini) הציגה Traceback של פייתון (יומן הבעיות #15).
            # Note: השאלה לא נכנסת להיסטוריה — כדי שלא תשמש כ"שאלה קודמת" לשאלת המשך
            print(f"Error: {type(error).__name__}: {error}")
            st.error(ANSWER_ERROR)
            return
        render_answer(answer, question)
    st.session_state.history.append({"question": question, "answer": answer})


def reply_subject(subject):
    """Re: תגובה לתגובה — בלי "Re: Re:"."""
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"


def reply_label(record):
    return f"{readable_date(record['date'])} · {sender_name(record)} · {record['subject']}"


def add_message_tab():
    """Tab: הוספת הודעה — שם, נושא, תוכן ו"בתגובה להודעה". נשמר ונכנס מיד לאינדקס."""
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    records = sorted(load_records().values(), key=lambda r: r["date"], reverse=True)[:REPLY_CHOICES]
    by_id = {r["message_id"]: r for r in records}
    # Why: מחוץ לטופס — כדי שבחירת הודעה תמלא מיד את הנושא ב-"Re: ..."
    reply_to = st.selectbox(
        "בתגובה להודעה (לא חובה)",
        options=[None] + list(by_id),
        format_func=lambda mid: "— הודעה חדשה —" if mid is None else reply_label(by_id[mid]),
    )
    default_subject = reply_subject(by_id[reply_to]["subject"]) if reply_to else ""

    with st.form("new_message", clear_on_submit=True):
        name = st.text_input("שם", value=st.session_state.get("name", ""))
        subject = st.text_input("נושא", value=default_subject)
        body = st.text_area("תוכן", height=150)
        submitted = st.form_submit_button("שליחה")

    if submitted:
        if not (name.strip() and subject.strip() and body.strip()):
            st.error("צריך למלא שם, נושא ותוכן.")
            return
        st.session_state.name = name.strip()
        save_email(new_post(name, subject, body, reply_to))
        added = index_now()
        st.session_state.flash = "ההודעה נשמרה ונכנסה לחיפוש." if added else "ההודעה נשמרה."
        st.rerun()

    import_section()


def import_section():
    """Import: ייבוא אופציונלי מקובץ מיילים (.mbox / .eml) — אותו פורמט, ואז אינדוקס מצטבר."""
    with st.expander("ייבוא מקובץ מיילים"):
        uploaded = st.file_uploader("קובץ .mbox (מ-Google Takeout) או .eml", type=["mbox", "eml"])
        if uploaded and st.button("ייבוא"):
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            path = UPLOADS_DIR / uploaded.name
            path.write_bytes(uploaded.getvalue())
            try:
                with st.spinner("מייבא..."):
                    total, saved, failed = import_file(path)
                    index_now()
            except (SystemExit, Exception) as error:
                st.error(f"הייבוא נכשל: {error}")
                return
            st.success(f"נקראו {total} מיילים, נשמרו {saved} חדשים, {failed} לא נקראו.")


def main():
    st.set_page_config(page_title="לוח הודעות כיתתי", page_icon="📋")
    st.markdown(RTL_CSS, unsafe_allow_html=True)
    st.title("לוח הודעות כיתתי")
    st.caption("ההודעות ההתחלתיות הן נתוני דוגמה מומצאים.")

    try:
        state = startup()
    except SystemExit as error:
        # Note: למשל — מודל ה-Embeddings השתנה ויש לבנות את האינדקס מחדש, או מפתח חסר
        st.error(str(error))
        st.stop()
    except Exception as error:
        print(f"Error: {type(error).__name__}: {error}")
        st.error("שגיאה בהפעלת המערכת (חיבור ל-Gemini או לאינדקס). בדקי את החיבור ונסי לרענן.")
        st.stop()
    check_index_fresh(state)

    st.session_state.setdefault("history", [])
    questions, add_message = st.tabs(["שאלות", "הוספת הודעה"])
    with questions:
        questions_tab(state)
    with add_message:
        add_message_tab()


main()
