"""
app.py — RePurpose AI: circular economy advisor
Run: streamlit run app.py
"""

import os
import chromadb
from groq import Groq
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH  = "data/chroma_db"
MODEL        = os.getenv("GROQ_MODEL", "llama3-8b-8192")
SIMILARITY_THRESHOLD = 0.45

HAZARD_KEYWORDS = {
    "battery","batteries","e-waste","electronic","electronics","phone","laptop",
    "computer","tablet","tv","monitor","printer","chemical","paint","solvent",
    "fluorescent","cfl","mercury","refrigerant","fridge","freezer","aerosol","pesticide",
}

SYSTEM_PROMPT = """You are RePurpose AI, a circular economy advisor.

When a user describes an unwanted item, recommend the best circular pathway in this priority order:
1. Reuse As-Is  2. Repair  3. Repurpose/Upcycle  4. Donate  5. Refurbish  6. Recycle  7. Responsible Disposal

Always respond in this exact format:

**♻️ Recommended Pathway:** [pathway name]

**Why:** [2-3 sentences. If not recommending Reuse As-Is, explain why higher options don't apply.]

**Sources:**
- [list the source titles from the provided context, or "No references available" if none]

**Next Steps:**
- [1-3 specific, actionable things the user can do right now]

**Confidence:** [High / Moderate / Low] — [one sentence explaining why]

Rules:
- Never recommend home disposal of hazardous items. Always direct to certified channels.
- Only cite sources from the provided context. Do not fabricate references.
- If no context is provided, note "low knowledge base coverage" in Confidence.
- End with: *This recommendation is advisory only. For hazardous or high-value items, consult your local authority.*"""

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        col = client.get_collection("knowledge_base")
        count = col.count()
    except Exception:
        col, count = None, 0
    return model, col, count

# ── Helpers ───────────────────────────────────────────────────────────────────
def retrieve(query, model, col, top_k=5):
    if col is None or col.count() == 0:
        return []
    embedding = model.encode(query).tolist()
    results = col.query(
        query_embeddings=[embedding],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        sim = 1.0 - dist
        if sim >= SIMILARITY_THRESHOLD:
            chunks.append({"text": doc, "title": meta.get("title", "Unknown"), "sim": sim})
    return sorted(chunks, key=lambda x: x["sim"], reverse=True)

def is_hazardous(text):
    lower = text.lower()
    return any(kw in lower for kw in HAZARD_KEYWORDS)

def build_user_message(description, chunks):
    if chunks:
        context = "\n\n---\n\n".join(
            f"[Source: {c['title']}]\n{c['text']}" for c in chunks
        )
        return f"Item: {description}\n\nKnowledge base context:\n{context}"
    return f"Item: {description}\n\n(No knowledge base context found — use low confidence note.)"

def get_response(description, history, model, col):
    chunks = retrieve(description, model, col)
    user_msg = build_user_message(description, chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # last 5 turns of history
    for turn in history[-5:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_msg})

    client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    resp = client.chat.completions.create(model=MODEL, messages=messages, timeout=30)
    return resp.choices[0].message.content, chunks

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RePurpose AI", page_icon="♻️", layout="centered")
st.title("♻️ RePurpose AI")
st.caption("Describe an unwanted item and get a circular economy recommendation grounded in sustainability guidelines.")

embed_model, col, kb_count = load_resources()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        "**RePurpose AI** helps you find the best second life for unwanted items "
        "using a 7-level circular economy hierarchy: "
        "Reuse → Repair → Repurpose → Donate → Refurbish → Recycle → Dispose."
    )
    st.divider()
    st.metric("Knowledge base chunks", kb_count if kb_count else "Empty")
    if kb_count == 0:
        st.warning("Run `python ingest.py` to load your knowledge base.")
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.rerun()

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# Chat history display
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Describe your item (e.g. 'old wooden chair with broken leg')")

if user_input:
    user_input = user_input.strip()

    # Validate
    if len(user_input) < 5:
        st.warning("Please describe your item in a bit more detail (at least 5 characters).")
        st.stop()

    # Truncate silently at 1000
    if len(user_input) > 1000:
        user_input = user_input[:1000]

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Hazard check
    if is_hazardous(user_input):
        with st.chat_message("assistant"):
            st.warning(
                "⚠️ **Hazardous item detected.** This item requires specialist disposal — "
                "do NOT place in general waste. Take electronics to an e-waste collection point, "
                "batteries to a recycling bin, and chemicals to a household hazardous waste facility."
            )

    # Get recommendation
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                api_key = os.getenv("GROQ_API_KEY", "")
                if not api_key or api_key == "your_openai_api_key_here":
                    st.error("Groq API key not set. Add GROQ_API_KEY to your .env file.")
                    st.stop()
                reply, chunks = get_response(user_input, st.session_state.history, embed_model, col)
                st.markdown(reply)
                if chunks:
                    with st.expander(f"📚 {len(chunks)} source(s) retrieved"):
                        for c in chunks:
                            st.markdown(f"- **{c['title']}** (similarity: {c['sim']:.2f})")
            except Exception as e:
                reply = f"Something went wrong: {e}"
                st.error(reply)

    # Update history
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": reply})
