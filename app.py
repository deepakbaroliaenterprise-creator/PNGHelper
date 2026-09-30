import csv
import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# ---- Config (verify model names in Google AI Studio; they change over time) ----
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-flash-lite-latest"
DOCS_DIR = "docs"
CHUNK_SIZE, OVERLAP, TOP_K = 800, 150, 4

# ---- Unanswered-question review process ----
DATA_DIR = "data"
UNANSWERED_LOG = os.path.join(DATA_DIR, "unanswered_questions.csv")
UNANSWERED_FIELDS = ["timestamp", "question", "status", "answer", "answered_at"]
ADMIN_ANSWERS_FILE = os.path.join(DOCS_DIR, "admin_added_answers.txt")
NO_ANSWER_MARKER = "NO_ANSWER:"

# ---- Usage stats (visits / questions asked) ----
STATS_FILE = os.path.join(DATA_DIR, "stats.json")

SYSTEM_PROMPT = (
    "You help society members with PNG (piped natural gas) connection applications. "
    "Answer ONLY from the provided context. Be brief, use bullet points and simple steps. "
    "Never ask for personal ID numbers. "
    "If, and only if, the answer to the question is not present in the context, reply with "
    f"EXACTLY this and nothing else: '{NO_ANSWER_MARKER} I don't know. Please contact the "
    "society committee or the gas provider.'"
)

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


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
    return texts


def chunk(text):
    step = CHUNK_SIZE - OVERLAP
    return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), step) if text[i : i + CHUNK_SIZE].strip()]


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
    chunks, sources = [], []
    for name, text in read_docs():
        for c in chunk(text):
            chunks.append(c)
            sources.append(name)
    if not chunks:
        return [], [], None
    return chunks, sources, embed(chunks, "RETRIEVAL_DOCUMENT")


def retrieve(question, chunks, sources, matrix):
    q = embed([question], "RETRIEVAL_QUERY")[0]
    idx = np.argsort(matrix @ q)[::-1][:TOP_K]
    return [(chunks[i], sources[i]) for i in idx]


def read_unanswered():
    if not os.path.isfile(UNANSWERED_LOG):
        return []
    with open(UNANSWERED_LOG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_unanswered(rows):
    with open(UNANSWERED_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNANSWERED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def log_unanswered(question):
    rows = read_unanswered()
    already_pending = any(
        r["question"].strip().lower() == question.strip().lower() and r["status"] == "pending" for r in rows
    )
    if already_pending:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    rows.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": question,
            "status": "pending",
            "answer": "",
            "answered_at": "",
        }
    )
    write_unanswered(rows)


def resolve_unanswered(rows, target, status, answer=""):
    target["status"] = status
    target["answer"] = answer
    target["answered_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_unanswered(rows)


def append_faq_answer(question, answer):
    with open(ADMIN_ANSWERS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n### Q: {question}\n{answer}\n")


def read_stats():
    if not os.path.isfile(STATS_FILE):
        return {"visits": 0, "questions": 0}
    with open(STATS_FILE, encoding="utf-8") as f:
        return json.load(f)


def bump_stat(key):
    os.makedirs(DATA_DIR, exist_ok=True)
    stats = read_stats()
    stats[key] = stats.get(key, 0) + 1
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f)
    return stats


st.set_page_config(page_title="PNG Application Helper", page_icon="🔥", initial_sidebar_state="collapsed")
st.title("🔥 PNG Application Helper")
st.caption("Answers come from the society's PNG guidelines. Please confirm final details with the committee / gas provider.")

if "visited" not in st.session_state:
    st.session_state.visited = True
    bump_stat("visits")

chunks, sources, matrix = build_index()
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
    hits = retrieve(question, chunks, sources, matrix)
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
                            append_faq_answer(row["question"], new_answer.strip())
                            resolve_unanswered(rows, row, "answered", new_answer.strip())
                            build_index.clear()
                            st.success("Saved. The assistant will use this answer from now on.")
                            st.rerun()
                        else:
                            st.warning("Enter an answer before saving.")
                with col2:
                    if st.button("Dismiss", key=f"dismiss_{row_key}"):
                        resolve_unanswered(rows, row, "dismissed")
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
        justify-content: space-between;
        align-items: center;
        z-index: 999;
    }}
    </style>
    <div class="app-footer">
        <span>👀 Visits: {stats.get('visits', 0)} &nbsp;&nbsp;•&nbsp;&nbsp; 💬 Questions asked: {stats.get('questions', 0)}</span>
        <span>Created by @DKB</span>
    </div>
    """,
    unsafe_allow_html=True,
)
