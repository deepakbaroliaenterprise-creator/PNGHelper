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

# "### **Q: ...**" / "### Q: ..." style (legacy FAQs.txt / admin-added answers)
FAQ_QA_PATTERN = re.compile(
    r"^#{1,6}\s*\*{0,2}Q:\s*(?P<question>.+?)\*{0,2}\s*$\n+(?P<answer>.*?)(?=^#{1,6}\s|^-{3,}\s*$|\Z)",
    re.DOTALL | re.MULTILINE,
)
# "**Question:** ... **Answer:** ..." style (GAIL_Gas_PNG_Resident_FAQ.md)
MD_QA_PATTERN = re.compile(
    r"\*\*Question:\*\*\s*(?P<question>.+?)\s*\n+\*\*Answer:\*\*\s*(?P<answer>.*?)(?=\n\*\*Keywords:\*\*|\n-{3,}|\n##\s|\Z)",
    re.DOTALL,
)
NEW_MD_QA_PATTERN = re.compile(
    r"^##\s+FAQ-\d+:\s*(?P<question>.+?)\s*$\n+"
    r"\*\*Example questions:\*\*\s*\n(?P<examples>.*?)(?=\n+\*\*Answer:\*\*)\n+"
    r"\*\*Answer:\*\*\s*(?P<answer>.*?)(?=\n+---\s*$|\Z)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)

NOT_AVAILABLE_MESSAGE = (
    "This information is not available in the current documents. "
    "Please verify the information through the relevant official channel."
)

SYSTEM_PROMPT = (
    "You are a RAG assistant answering questions using only the documents provided as context. "
    "Follow these rules:\n"
    "1. Use only information available in this knowledge base.\n"
    "2. Provide short and direct answers.\n"
    "3. Answer the specific question first.\n"
    "4. Do not repeat unrelated registration details.\n"
    "5. Provide the relevant official link only when required.\n"
    "6. Do not invent GAIL Gas policies, charges or timelines.\n"
    "7. If the requested information is not available, respond EXACTLY with this and nothing else: "
    f"'{NO_ANSWER_MARKER} {NOT_AVAILABLE_MESSAGE}'\n"
    "8. For payment-related questions, always remind residents to use the official GAIL Gas portal.\n"
    "9. Never advise residents to make payments to an individual representative.\n"
    "10. If multiple FAQs are relevant, combine only the minimum information needed to answer the "
    "user's question.\n"
    "Additionally, never ask residents for personal ID numbers."
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
    for path in glob.glob(os.path.join(DOCS_DIR, "**"), recursive=True):
        if not os.path.isfile(path):
            continue
        if path.lower().endswith(".pdf"):
            text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        elif path.lower().endswith((".txt", ".md")):
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
    for m in NEW_MD_QA_PATTERN.finditer(text):
        question = m.group("question").strip()
        examples = m.group("examples").strip()
        answer = m.group("answer").strip()
        if question and answer:
            pairs.append((question, answer, f"{question}\n{examples}"))
    if pairs:
        return pairs

    for pattern in (MD_QA_PATTERN, FAQ_QA_PATTERN):
        pairs = []
        for m in pattern.finditer(text):
            question = m.group("question").strip()
            answer = m.group("answer").strip()
            if question and answer:
                pairs.append((question, answer, question))
        if pairs:
            return pairs
    return []


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
            for question, answer, searchable_question in pairs:
                chunks.append(f"Q: {searchable_question}\nA: {answer}")
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
    st.error("No supported documents found. Add PDF, TXT or Markdown files to the 'docs' folder.")
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
