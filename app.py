import glob
import os

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

SYSTEM_PROMPT = (
    "You help society members with PNG (piped natural gas) connection applications. "
    "Answer ONLY from the provided context. If the answer is not in the context, say you "
    "don't know and suggest contacting the society committee or the gas provider. "
    "Be brief, use bullet points and simple steps. Never ask for personal ID numbers."
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


st.set_page_config(page_title="PNG Application Helper", page_icon="🔥")
st.title("🔥 PNG Application Helper")
st.caption("Answers come from the society's PNG guidelines. Please confirm final details with the committee / gas provider.")

chunks, sources, matrix = build_index()
if matrix is None:
    st.error("No documents found. Add PDF/TXT files to the 'docs' folder.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if question := st.chat_input("Ask about PNG application steps, documents, fees, forms..."):
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
        answer = resp.text
    except Exception:
        answer = "Sorry, the assistant is busy (free-tier limit). Please try again in a minute."
    answer += "\n\n_Sources: " + ", ".join(sorted({s for _, s in hits})) + "_"
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
