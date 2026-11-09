# Run: .\.venv\Scripts\streamlit.exe run app.py
import src  # noqa: F401  (Must: ראשון — מפעיל את truststore)

import streamlit as st
from chromadb.api.client import SharedSystemClient

from src.config import setup_models
from src.indexer import index_built_at, index_new_messages, load_records, open_index, readable_date, sender_name
from src.messages import new_post, save_message, seed_samples
from src.query import NO_INFO_ANSWER, Answer, ask_stream, attach_token_counter, build_answer, token_summary

ANSWER_ERROR = "אירעה שגיאה בזמן החיפוש או כתיבת התשובה (ייתכן עומס זמני אצל ספק המודלים). נסי שוב בעוד רגע."
RTL_CSS = """<style>
[data-testid="stAppViewContainer"] { direction: rtl; }
p, li, h1, label, input, textarea, [data-testid="stMarkdownContainer"] { text-align: right; }
</style>"""


@st.cache_resource(show_spinner="טוען מודלים ומעדכן את האינדקס...")
def startup():
    """Startup: פעם אחת — מודלים, מונה טוקנים, הודעות דוגמה, אינדוקס מצטבר, ופתיחת האינדקס (#16)."""
    setup_models()
    counter = attach_token_counter()
    seed_samples()
    index_new_messages()
    return {"counter": counter, "index": open_index()[0], "built_at": index_built_at()}


def check_index_fresh(state):
    """Fresh: אם האינדקס נבנה מחדש מתהליך אחר מאז שהאפליקציה עלתה — טוענים מחדש (#13)."""
    built_at = index_built_at()
    if built_at is None:
        # Note: בנייה מחדש באמצע — לא טוענים, כדי לא לאנדקס במקביל לתהליך האחר
        st.warning("האינדקס נבנה כרגע מחדש. נסי שוב בעוד רגע.")
        st.stop()
    if built_at != state["built_at"]:
        SharedSystemClient.clear_system_cache()
        st.cache_resource.clear()
        st.rerun()


def render_answer(answer, question):
    """Answer: ניסוח החיפוש, מקורות וטוקנים — מתחת לטקסט התשובה."""
    if answer.search_query != question:
        st.caption(f"חיפשתי: {answer.search_query}")
    if answer.sources:
        latest = max(answer.sources, key=lambda s: s["date_ts"])["date"]
        earlier = " · יש גם הודעות קודמות בנושא" if len(answer.sources) > 1 else ""
        st.caption(f"מבוסס על הודעה מ{latest}{earlier}")
        with st.expander(f"{'כל ההודעות שנשלחו' if answer.sources_fallback else 'מקורות'} ({len(answer.sources)})"):
            for s in answer.sources:
                st.markdown(f"**{s['date']}** · {s['from']} · {s['subject']}")
                st.caption(s["snippet"])
    t = answer.tokens
    llm = f"LLM {t['llm_prompt']} + {t['llm_completion']} (נתון מהספק, כולל הניסוח)" if answer.llm_called \
        else "לא נשלחה קריאה ל-LLM שעונה"
    st.caption(f"טוקנים: {llm} · Embedding {t['embedding']} (הערכה)")


def questions_tab(state):
    """Tab: שאלות. השאלה והתשובה האחרונות נשלחות רק לניסוח מחדש — לא ל-LLM שעונה."""
    if st.button("שיחה חדשה"):
        st.session_state.history = []
    for item in st.session_state.history:
        st.chat_message("user").markdown(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"].text)
            render_answer(item["answer"], item["question"])

    question = st.chat_input("שאלי שאלה על ההודעות...")
    if not question:
        return
    history = [(item["question"], item["answer"].text) for item in st.session_state.history]
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("מחפש בהודעות..."):
                stream, nodes, search_query = ask_stream(question, state["counter"], history, state["index"])
            if stream is None:
                st.markdown(NO_INFO_ANSWER)
                answer = Answer(NO_INFO_ANSWER, tokens=token_summary(state["counter"]), search_query=search_query)
            else:
                st.write_stream(stream)
                answer = build_answer(stream.full_text, nodes, state["counter"], search_query, {})
        except Exception as error:
            # Why: בלי זה שגיאה (למשל 503) הציגה Traceback; השאלה לא נכנסת להיסטוריה (#15)
            print(f"Error: {type(error).__name__}: {error}")
            st.error(ANSWER_ERROR)
            return
        render_answer(answer, question)
    st.session_state.history.append({"question": question, "answer": answer})


def add_message_tab():
    """Tab: הוספת הודעה — שם, נושא, תוכן ו"בתגובה להודעה". נשמרת ונכנסת מיד לאינדקס."""
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    recent = {r["message_id"]: r for r in sorted(load_records().values(), key=lambda r: r["date"])[-30:]}
    # Why: מחוץ לטופס — כדי שבחירת הודעה תמלא מיד את הנושא ב-"Re: ..."
    reply_to = st.selectbox("בתגובה להודעה (לא חובה)", [None] + list(reversed(recent)),
                            format_func=lambda m: "— הודעה חדשה —" if m is None else
                            f"{readable_date(recent[m]['date'])} · {sender_name(recent[m])} · {recent[m]['subject']}")
    subject = recent[reply_to]["subject"] if reply_to else ""
    if reply_to and not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"  # Note: תגובה לתגובה — בלי "Re: Re:"

    with st.form("new_message", clear_on_submit=True):
        name = st.text_input("שם", value=st.session_state.get("name", ""))
        subject = st.text_input("נושא", value=subject)
        body = st.text_area("תוכן", height=150)
        submitted = st.form_submit_button("שליחה")
    if not submitted:
        return
    if not (name.strip() and subject.strip() and body.strip()):
        st.error("צריך למלא שם, נושא ותוכן.")
        return
    st.session_state.name = name.strip()
    save_message(new_post(name, subject, body, reply_to))
    try:
        index_new_messages()
        st.session_state.flash = "ההודעה נשמרה ונכנסה לחיפוש."
    except Exception as error:
        # Note: ההודעה כבר שמורה בקובץ — תיכנס לאינדקס בהפעלה הבאה
        st.session_state.flash = f"ההודעה נשמרה, ותיכנס לחיפוש בהפעלה הבאה ({type(error).__name__})."
    st.rerun()


st.set_page_config(page_title="לוח הודעות כיתתי", page_icon="📋")
st.markdown(RTL_CSS, unsafe_allow_html=True)
st.title("לוח הודעות כיתתי")
st.caption("ההודעות ההתחלתיות הן נתוני דוגמה מומצאים.")
try:
    state = startup()
except SystemExit as error:  # Note: מפתח חסר, או אינדקס שצריך לבנות מחדש
    st.error(str(error))
    st.stop()
except Exception as error:
    print(f"Error: {type(error).__name__}: {error}")
    st.error("שגיאה בהפעלת המערכת (חיבור לספק המודלים או לאינדקס). בדקי את החיבור ונסי לרענן.")
    st.stop()
check_index_fresh(state)
st.session_state.setdefault("history", [])
questions, add_message = st.tabs(["שאלות", "הוספת הודעה"])
with questions:
    questions_tab(state)
with add_message:
    add_message_tab()
