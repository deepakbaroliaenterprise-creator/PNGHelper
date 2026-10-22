import re
from pathlib import Path

import streamlit as st


FAQ_PATH = Path(__file__).resolve().parents[1] / "docs" / "GAIL_Gas_PNG_Resident_FAQ.md"
FAQ_PATTERN = re.compile(
    r"^\*{0,2}##\s+FAQ-\d+:\s*(?P<question>.+?)\*{0,2}\s*$\n+"
    r"\*\*Example questions:\*\*\s*\n.*?(?=\n+\*\*Answer:\*\*)\n+"
    r"\*\*Answer:\*\*\s*(?P<answer>.*?)(?=\n+\*{0,2}---\*{0,2}\s*$|\Z)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)


def parse_faq(text):
    return [
        (match.group("question").strip(), match.group("answer").strip())
        for match in FAQ_PATTERN.finditer(text)
        if match.group("question").strip() and match.group("answer").strip()
    ]


def relevance(question, answer, query):
    question = question.casefold()
    answer = answer.casefold()
    score = 12 if query in question else 5 if query in answer else 0
    for term in query.split():
        score += 4 if term in question else 0
        score += 1 if term in answer else 0
    return score


st.set_page_config(page_title="GAIL Gas PNG FAQ", initial_sidebar_state="collapsed")
st.title("GAIL Gas PNG FAQ")
st.caption("Answers to common questions about GAIL Gas PNG connections.")
st.link_button("Ask the AI assistant", "https://gailgasconnection.streamlit.app/")

try:
    faq_items = parse_faq(FAQ_PATH.read_text(encoding="utf-8"))
except OSError:
    st.error("The FAQ source could not be loaded. Please try again later.")
    st.stop()

if not faq_items:
    st.error("No question-and-answer entries were found in the FAQ source.")
    st.stop()

query = st.text_input("Search questions and answers", placeholder="Security deposit or gas leak").strip().casefold()
matches = faq_items
if query:
    ranked = [(question, answer, relevance(question, answer, query)) for question, answer in faq_items]
    matches = [(question, answer) for question, answer, score in sorted(ranked, key=lambda item: item[2], reverse=True) if score > 0]

st.caption(f"{len(matches)} {'answer' if len(matches) == 1 else 'answers'}{' found' if query else ''}")
if not matches:
    st.info("No answers match that search.")

for question, answer in matches:
    st.subheader(question)
    st.markdown(answer)
    st.divider()