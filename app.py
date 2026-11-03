# Run: .\.venv\Scripts\streamlit.exe run app.py
import src  # noqa: F401  (Must: ראשון — מפעיל את truststore)

import streamlit as st

from src import import_export
from src.config import PROJECT_ROOT, setup_models
from src.indexer import index_new_messages, load_records, readable_date, sender_name
from src.messages import new_post, save_email, seed_samples
from src.query import NO_INFO_ANSWER, Answer, ask_stream, attach_token_counter, finish_stream

UPLOADS_DIR = PROJECT_ROOT / "data" / "raw" / "uploads"
REPLY_CHOICES = 30

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
    """Startup: פעם אחת לכל הפעלה — מודלים, מונה טוקנים, הודעות דוגמה ואינדוקס מצטבר."""
    setup_models()
    counter = attach_token_counter()
    seed_samples()
    index_new_messages()
    return counter


def index_now():
    """Index: אינדוקס מצטבר. אם נכשל — ההודעה כבר שמורה בקובץ ותיכנס בפעם הבאה."""
    try:
        return index_new_messages()
    except Exception as error:
        st.warning(f"Note: ההודעה נשמרה, אבל לא נכנסה עדיין לחיפוש ({type(error).__name__}). היא תיכנס בהפעלה הבאה.")
        return 0


def render_answer(answer):
    """Answer: מקורות מעוצבים וטוקנים — מתחת לטקסט התשובה."""
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
            f"טוקנים: LLM {t['llm_prompt']} נשלחו + {t['llm_completion']} בתשובה (נתון מ-Gemini) · "
            f"Embedding {t['embedding']} (הערכה)"
        )
    else:
        st.caption(f"לא נמצאה הודעה מתאימה — לא נשלחה קריאה ל-LLM · Embedding {t['embedding']} (הערכה)")


def questions_tab(counter):
    """Tab: שאלות — היסטוריה בתצוגה בלבד; כל שאלה נשלחת ל-LLM בנפרד, בלי ההיסטוריה."""
    if st.button("שיחה חדשה"):
        st.session_state.history = []

    for item in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"].text)
            render_answer(item["answer"])

    question = st.chat_input("שאלי שאלה על ההודעות...")
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        stream, nodes = ask_stream(question, counter)
        if stream is None:
            st.markdown(NO_INFO_ANSWER)
            answer = Answer(NO_INFO_ANSWER, tokens={"llm_prompt": 0, "llm_completion": 0,
                                                    "embedding": counter.total_embedding_token_count})
        else:
            st.write_stream(stream)
            answer = finish_stream(stream, nodes, counter)
        render_answer(answer)
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
            with st.spinner("מייבא..."):
                records, total, _, failed = import_export.load_records(path, None)
                saved = sum(save_email(r) for r in records)
                index_now()
            st.success(f"נקראו {total} מיילים, נשמרו {saved} חדשים, {failed} לא נקראו.")


def main():
    st.set_page_config(page_title="לוח הודעות כיתתי", page_icon="📋")
    st.markdown(RTL_CSS, unsafe_allow_html=True)
    st.title("לוח הודעות כיתתי")
    st.caption("ההודעות ההתחלתיות הן נתוני דוגמה מומצאים.")

    try:
        counter = startup()
    except SystemExit as error:
        # Note: למשל — מודל ה-Embeddings השתנה ויש לבנות את האינדקס מחדש
        st.error(str(error))
        st.stop()

    st.session_state.setdefault("history", [])
    questions, add_message = st.tabs(["שאלות", "הוספת הודעה"])
    with questions:
        questions_tab(counter)
    with add_message:
        add_message_tab()


main()
