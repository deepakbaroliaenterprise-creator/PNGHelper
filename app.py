import glob
import os
import re
from datetime import datetime, timezone

import gspread
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from google.oauth2.service_account import Credentials
from pypdf import PdfReader

# ---- Config (verify model names in Google AI Studio; they change over time) ----
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-flash-lite-latest"
DOCS_DIR = "docs"
CHUNK_SIZE, OVERLAP, TOP_K = 800, 150, 4

# ---- Unanswered-question review process (persisted in Google Sheets) ----
UNANSWERED_FIELDS = ["timestamp", "question", "status", "answer", "answered_at"]
NO_ANSWER_MARKER = "NO_ANSWER:"
GSHEET_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# ---- Most-asked-question tracking (persisted in Google Sheets) ----
QUESTION_USAGE_FIELDS = ["question", "answer", "count"]
TOP_QUESTIONS_SHOWN = 3
FAQ_QA_PATTERN = re.compile(
    r"^#{1,6}\s*\*{0,2}Q:\s*(?P<question>.+?)\*{0,2}\s*$\n+(?P<answer>.*?)(?=^#{1,6}\s|^-{3,}\s*$|\Z)",
    re.DOTALL | re.MULTILINE,
)

SYSTEM_PROMPT = (
    "You help society members with PNG (piped natural gas) connection applications. "
    "Answer ONLY from the provided context. Be brief, use bullet points and simple steps. "
    "Never ask for personal ID numbers. "
    "If, and only if, the answer to the question is not present in the context, reply with "
    f"EXACTLY this and nothing else: '{NO_ANSWER_MARKER} I don't know. Please contact the "
    "society committee or the gas provider.'"
)

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


@st.cache_resource
def get_worksheets():
    creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=GSHEET_SCOPES)
    sh = gspread.authorize(creds).open_by_key(st.secrets["GSHEET_ID"])

    try:
        questions_ws = sh.worksheet("Questions")
    except gspread.exceptions.WorksheetNotFound:
        questions_ws = sh.add_worksheet("Questions", rows=1000, cols=len(UNANSWERED_FIELDS))
        questions_ws.append_row(UNANSWERED_FIELDS, value_input_option="RAW")

    try:
        stats_ws = sh.worksheet("Stats")
    except gspread.exceptions.WorksheetNotFound:
        stats_ws = sh.add_worksheet("Stats", rows=10, cols=2)
        stats_ws.append_rows([["key", "value"], ["visits", 0], ["questions", 0]], value_input_option="RAW")

    try:
        usage_ws = sh.worksheet("QuestionUsage")
    except gspread.exceptions.WorksheetNotFound:
        usage_ws = sh.add_worksheet("QuestionUsage", rows=1000, cols=len(QUESTION_USAGE_FIELDS))
        usage_ws.append_row(QUESTION_USAGE_FIELDS, value_input_option="RAW")

    return questions_ws, stats_ws, usage_ws


def read_docs():
    texts = []
    for path in glob.glob(os.path.join(DOCS_DIR, "*")):
        if path.lower().endswith(".pdf"):
            text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        elif path.lower().endswith(".txt"):
            text = open(path, encoding="utf-8").read()
        else:
            continue
        texts.append((os.path.basename(path), text))

    answered = [r for r in read_unanswered() if r.get("status") == "answered" and r.get("answer")]
    if answered:
        admin_text = "\n\n".join(f"Q: {r['question']}\nA: {r['answer']}" for r in answered)
        texts.append(("Admin Answers (Google Sheet)", admin_text))

    return texts


def chunk(text):
    step = CHUNK_SIZE - OVERLAP
    return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), step) if text[i : i + CHUNK_SIZE].strip()]


def parse_faq_pairs(text):
    pairs = []
    for m in FAQ_QA_PATTERN.finditer(text):
        question = m.group("question").strip()
        answer = m.group("answer").strip()
        if question and answer:
            pairs.append((question, answer))
    return pairs


def embed(texts, task):
    vecs = []
    for i in range(0, len(texts), 50):  # batch to respect free-tier limits
        res = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i : i + 50],
            config=types.EmbedContentConfig(task_type=task),
        )
        vecs += [e.values for e in res.embeddings]
    m = np.array(vecs, dtype="float32")
    return m / np.linalg.norm(m, axis=1, keepdims=True)


@st.cache_resource(show_spinner="Indexing documents...")
def build_index():
    chunks, sources, chunk_qa = [], [], []
    for name, text in read_docs():
        pairs = parse_faq_pairs(text)
        if pairs:
            for question, answer in pairs:
                chunks.append(f"Q: {question}\nA: {answer}")
                sources.append(name)
                chunk_qa.append((question, answer))
        else:
            for c in chunk(text):
                chunks.append(c)
                sources.append(name)
                chunk_qa.append(None)
    if not chunks:
        return [], [], [], None
    return chunks, sources, chunk_qa, embed(chunks, "RETRIEVAL_DOCUMENT")


def retrieve(question, chunks, sources, chunk_qa, matrix):
    q = embed([question], "RETRIEVAL_QUERY")[0]
    idx = np.argsort(matrix @ q)[::-1][:TOP_K]
    hits = [(chunks[i], sources[i]) for i in idx]
    top_qa = chunk_qa[idx[0]] if len(idx) else None
    return hits, top_qa


def read_unanswered():
    questions_ws, _, _ = get_worksheets()
    values = questions_ws.get_all_values()
    if not values:
        return []
    header, *rows = values
    return [dict(zip(header, row)) for row in rows]


def log_unanswered(question):
    rows = read_unanswered()
    already_pending = any(
        r.get("question", "").strip().lower() == question.strip().lower() and r.get("status") == "pending"
        for r in rows
    )
    if already_pending:
        return
    questions_ws, _, _ = get_worksheets()
    questions_ws.append_row(
        [datetime.now(timezone.utc).isoformat(timespec="seconds"), question, "pending", "", ""],
        value_input_option="RAW",
    )


def resolve_unanswered(target, status, answer=""):
    questions_ws, _, _ = get_worksheets()
    values = questions_ws.get_all_values()
    header, *rows = values
    for i, row in enumerate(rows, start=2):  # sheet row 1 is the header
        row_dict = dict(zip(header, row))
        if row_dict.get("timestamp") == target["timestamp"] and row_dict.get("question") == target["question"]:
            answered_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            questions_ws.update(range_name=f"C{i}:E{i}", values=[[status, answer, answered_at]], value_input_option="RAW")
            break


def read_stats():
    _, stats_ws, _ = get_worksheets()
    stats = {}
    for row in stats_ws.get_all_values()[1:]:
        if len(row) >= 2:
            try:
                stats[row[0]] = int(row[1])
            except ValueError:
                stats[row[0]] = 0
    return stats


def bump_stat(key):
    _, stats_ws, _ = get_worksheets()
    values = stats_ws.get_all_values()
    header, *rows = values
    for i, row in enumerate(rows, start=2):
        if row and row[0] == key:
            current = int(row[1]) if len(row) > 1 and row[1].strip().isdigit() else 0
            stats_ws.update(range_name=f"B{i}", values=[[current + 1]], value_input_option="RAW")
            return current + 1
    stats_ws.append_row([key, 1], value_input_option="RAW")
    return 1


def bump_question_usage(question, answer):
    _, _, usage_ws = get_worksheets()
    values = usage_ws.get_all_values()
    header, *rows = values
    for i, row in enumerate(rows, start=2):
        if row and row[0] == question:
            current = int(row[2]) if len(row) > 2 and row[2].strip().isdigit() else 0
            usage_ws.update(range_name=f"C{i}", values=[[current + 1]], value_input_option="RAW")
            return
    usage_ws.append_row([question, answer, 1], value_input_option="RAW")


def top_questions(n=TOP_QUESTIONS_SHOWN):
    _, _, usage_ws = get_worksheets()
    values = usage_ws.get_all_values()
    if not values:
        return []
    header, *rows = values
    records = [dict(zip(header, row)) for row in rows]

    def count_of(r):
        try:
            return int(r.get("count", 0))
        except ValueError:
            return 0

    return sorted(records, key=count_of, reverse=True)[:n]


st.set_page_config(page_title="PNG Application Helper", page_icon="🔥", initial_sidebar_state="collapsed")
st.title("🔥 PNG Application Helper")
st.caption("Information sourced from society PNG rules. Please contact committee members or GAIL representatives for any further clarifications.")

if "visited" not in st.session_state:
    st.session_state.visited = True
    bump_stat("visits")

top_qs = top_questions()
if top_qs:
    st.subheader("📊 Most Asked Questions")
    for i, row in enumerate(top_qs, start=1):
        with st.expander(f"{i}. {row['question']}", expanded=False):
            st.caption(row["answer"])
    st.divider()

chunks, sources, chunk_qa, matrix = build_index()
if matrix is None:
    st.error("No documents found. Add PDF/TXT files to the 'docs' folder.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if question := st.chat_input("Ask about PNG application steps, documents, fees, forms..."):
    bump_stat("questions")
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})
    hits, top_qa = retrieve(question, chunks, sources, chunk_qa, matrix)
    if top_qa:
        bump_question_usage(*top_qa)
    context = "\n\n---\n\n".join(f"[{s}]\n{c}" for c, s in hits)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    try:
        resp = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
        )
        answer = resp.text.strip()
        if answer.startswith(NO_ANSWER_MARKER):
            answer = answer[len(NO_ANSWER_MARKER) :].strip()
            log_unanswered(question)
    except Exception:
        answer = "Sorry, the assistant is busy (free-tier limit). Please try again in a minute."
    answer += "\n\n_Sources: " + ", ".join(sorted({s for _, s in hits})) + "_"
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("🛠️ Admin")
    st.caption("Review questions the assistant couldn't answer and add answers to the knowledge base.")
    if not st.session_state.get("admin_authed"):
        pwd = st.text_input("Admin password", type="password", key="admin_pwd_input")
        if st.button("Log in", key="admin_login_btn"):
            if pwd and pwd == st.secrets.get("ADMIN_PASSWORD", ""):
                st.session_state.admin_authed = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    else:
        if st.button("Log out", key="admin_logout_btn"):
            st.session_state.admin_authed = False
            st.rerun()

        rows = read_unanswered()
        pending = [r for r in rows if r["status"] == "pending"]
        st.metric("Unanswered questions", len(pending))

        for row in pending:
            row_key = f"{row['timestamp']}_{row['question']}"
            with st.expander(row["question"]):
                st.caption(f"Asked: {row['timestamp']} UTC")
                new_answer = st.text_area("Answer to add", key=f"answer_{row_key}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save answer", key=f"save_{row_key}"):
                        if new_answer.strip():
                            resolve_unanswered(row, "answered", new_answer.strip())
                            build_index.clear()
                            st.success("Saved. The assistant will use this answer from now on.")
                            st.rerun()
                        else:
                            st.warning("Enter an answer before saving.")
                with col2:
                    if st.button("Dismiss", key=f"dismiss_{row_key}"):
                        resolve_unanswered(row, "dismissed")
                        st.rerun()

        with st.expander("Answered / dismissed history"):
            resolved = [r for r in rows if r["status"] != "pending"]
            if not resolved:
                st.caption("Nothing resolved yet.")
            for row in resolved:
                st.markdown(f"**{row['status'].title()}** — {row['question']}")
                if row["answer"]:
                    st.caption(row["answer"])

stats = read_stats()
st.markdown(
    f"""
    <style>
    .app-footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 4px 20px;
        background: rgba(14, 17, 23, 0.92);
        border-top: 1px solid rgba(250, 250, 250, 0.15);
        font-size: 0.75rem;
        color: rgba(250, 250, 250, 0.6);
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 12px;
        z-index: 999;
    }}
    </style>
    <div class="app-footer">
        <span>👀 Visits: {stats.get('visits', 0)}</span>
        <span>•</span>
        <span>💬 Questions asked: {stats.get('questions', 0)}</span>
        <span>•</span>
        <span>Created by Deepak Barolia aka DKB</span>
    </div>
    """,
    unsafe_allow_html=True,
)
