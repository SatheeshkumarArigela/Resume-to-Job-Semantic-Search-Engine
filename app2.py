"""
Resume-to-Job Semantic Search Engine

"""
import os
import re
import time
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# ---------- Config ----------
EMBED_MODEL = "gemini-embedding-001"   # if Google renames it, check the docs
LLM_MODEL = "gemini-3.8-flash"         # used only for "why this match" explanations
DIM = 768                              # smaller vectors = faster, still accurate
BATCH_SIZE = 20                        # small batches to respect free-tier limits
MAX_CHARS = 8000                       # truncate very long resumes
CHUNK_SIZE = 700                       # RAG: characters per chunk
CHUNK_OVERLAP = 120                    # RAG: overlap so ideas aren't cut in half

st.set_page_config(page_title="Resume-to-Job Semantic Search", page_icon="🔎", layout="wide")


# ---------- Helpers ----------
@st.cache_resource
def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None
    if not key:
        st.error("Set GEMINI_API_KEY as an environment variable or in .streamlit/secrets.toml")
        st.stop()
    return genai.Client(api_key=key)


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def extract_text_from_path(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def redact_pii(text: str) -> str:
    """Privacy step: mask emails and phone numbers before embedding / sending to an LLM."""
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", text)
    text = re.sub(r"(\+?\d[\d\s\-().]{8,}\d)", "[PHONE]", text)
    return text


def embed_texts(texts, task_type):
    """Return L2-normalised float32 embeddings (cosine similarity via inner product)."""
    client = get_client()
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for attempt in range(4):  # simple retry for rate limits
            try:
                resp = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type, output_dimensionality=DIM
                    ),
                )
                vectors.extend([e.values for e in resp.embeddings])
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 5)
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)  # needed because reduced-dimension embeddings aren't normalised
    return arr


def build_index(candidates):
    texts = [c["clean_text"][:MAX_CHARS] for c in candidates]
    vectors = embed_texts(texts, "RETRIEVAL_DOCUMENT")
    index = faiss.IndexFlatIP(DIM)
    index.add(vectors)
    return index


def explain_match(job_desc: str, resume_text: str) -> str:
    client = get_client()
    prompt = f"""You are a recruiting assistant. Compare the resume to the job description.
Reply in exactly this format:
**Strengths:** 2-3 short bullet points of skills/experience that match (even if wording differs)
**Gaps:** 1-2 short bullet points of missing requirements
**Verdict:** one sentence

JOB DESCRIPTION:
{job_desc[:4000]}

RESUME:
{resume_text[:5000]}"""
    resp = client.models.generate_content(model=LLM_MODEL, contents=prompt)
    return resp.text


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, preferring to cut at line breaks / sentence ends."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if len(c) > 30]


def build_chunk_index(candidates):
    """RAG indexing: chunk every resume, embed each chunk, keep resume name as metadata."""
    chunks = []
    for c in candidates:
        label = Path(c["name"]).stem.replace("_", " ")
        for piece in chunk_text(c["clean_text"][:MAX_CHARS]):
            chunks.append({"resume": c["name"], "text": piece, "embed_text": f"Candidate: {label}\n{piece}"})
    vectors = embed_texts([c["embed_text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    index = faiss.IndexFlatIP(DIM)
    index.add(vectors)
    return index, chunks


def answer_question(question: str, hits):
    """RAG generation: answer ONLY from retrieved chunks and cite the resume names."""
    client = get_client()
    context = "\n\n".join(f"[Source: {h['resume']}]\n{h['text']}" for h in hits)
    prompt = f"""You are a recruiting assistant answering questions about candidates.
Answer ONLY using the context below. If the context does not contain the answer, reply exactly:
"I couldn't find that in the uploaded resumes."
Cite the resume file name(s) in brackets after each claim, e.g. [01_Priya_Sharma_GenAI_Engineer.txt].
Be concise.

CONTEXT:
{context}

QUESTION: {question}"""
    resp = client.models.generate_content(model=LLM_MODEL, contents=prompt)
    return resp.text


# ---------- UI ----------
st.title("🔎 Resume-to-Job Semantic Search Engine")
st.caption("Finds relevant candidates by meaning, not exact keywords. Contact details are redacted before processing.")

with st.sidebar:
    st.header("1. Load resumes")
    source = st.radio("Source", ["Local folder", "Upload files"], horizontal=True)

    # each item: (name, size_key, loader_function)
    items = []
    if source == "Upload files":
        uploaded = st.file_uploader("PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
        items = [(f.name, f.size, (lambda f=f: extract_text(f))) for f in uploaded or []]
    else:
        folder = st.text_input("Folder path", placeholder=r"C:\Users\you\Desktop\sample_resumes")
        subfolders = st.checkbox("Include subfolders", value=False)
        if folder:
            root = Path(folder.strip().strip('"'))
            if root.is_dir():
                found = root.rglob("*") if subfolders else root.iterdir()
                paths = sorted(
                    p for p in found
                    if p.is_file()
                    and p.suffix.lower() in (".pdf", ".txt")
                    and not p.name.upper().startswith(("TEST_JOB", "EXPECTED"))
                )
                items = [(p.name, p.stat().st_size, (lambda p=p: extract_text_from_path(p))) for p in paths]
                st.caption(f"Found {len(items)} resume file(s)")
            else:
                st.error("Folder not found. Check the path.")

    top_k = st.slider("Candidates to show", 1, 10, 5)
    explain_top = st.slider("AI explanations for top N", 0, 5, 3,
                            help="Each explanation uses one LLM call (free-tier friendly: keep low).")
    chunks_k = st.slider("Chunks to retrieve for chat", 3, 10, 6)

if items:
    signature = tuple((name, size) for name, size, _ in items)
    if st.session_state.get("signature") != signature:
        with st.spinner("Reading resumes and building the vector index..."):
            candidates = []
            for name, _, load in items:
                raw = load()
                if raw.strip():
                    candidates.append(
                        {"name": name, "raw_text": raw, "clean_text": redact_pii(raw)}
                    )
            if not candidates:
                st.error("Could not read any text from the files.")
                st.stop()
            st.session_state["candidates"] = candidates
            st.session_state["index"] = build_index(candidates)
            st.session_state["chunk_index"], st.session_state["chunks"] = build_chunk_index(candidates)
            st.session_state.pop("results", None)
            st.session_state["chat"] = []
            st.session_state["signature"] = signature
        st.sidebar.success(f"Indexed {len(st.session_state['candidates'])} resumes")

st.header("2. Paste a job description")
job_desc = st.text_area("Job description", height=220,
                        placeholder="e.g. Looking for an engineer to build RAG pipelines with LangChain and vector databases...")

if st.button("Find best candidates", type="primary"):
    if "index" not in st.session_state:
        st.warning("Load some resumes first (sidebar).")
    elif not job_desc.strip():
        st.warning("Please paste a job description.")
    else:
        candidates = st.session_state["candidates"]
        k = min(top_k, len(candidates))
        with st.spinner("Searching..."):
            q = embed_texts([redact_pii(job_desc)], "RETRIEVAL_QUERY")
            scores, ids = st.session_state["index"].search(q, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            explanation = None
            if rank <= explain_top:
                with st.spinner(f"Generating explanation for #{rank}..."):
                    try:
                        explanation = explain_match(job_desc, candidates[idx]["clean_text"])
                    except Exception as e:
                        explanation = f"Explanation unavailable (rate limit?): {e}"
            results.append({"idx": int(idx), "score": float(score), "explanation": explanation})
        st.session_state["results"] = results

# Results are stored in session_state so they stay on screen while you use the chat below
if st.session_state.get("results"):
    st.header("3. Results")
    for rank, r in enumerate(st.session_state["results"], start=1):
        cand = st.session_state["candidates"][r["idx"]]
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.subheader(f"#{rank}  {cand['name']}")
            c2.metric("Match", f"{max(r['score'], 0) * 100:.0f}%")
            st.progress(float(min(max(r["score"], 0), 1)))
            if r["explanation"]:
                st.markdown(r["explanation"])
            with st.expander("View redacted resume text"):
                st.text(cand["clean_text"][:3000])

# ---------- RAG chat ----------
if "chunk_index" in st.session_state:
    st.divider()
    st.header("4. Ask questions about the candidates (RAG)")
    st.caption("Answers come only from retrieved resume chunks, with the source resume cited.")

    for msg in st.session_state.get("chat", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Retrieved chunks used for this answer"):
                    for h in msg["sources"]:
                        st.markdown(f"**{h['resume']}**  (similarity {h['score']:.2f})")
                        st.text(h["text"])

    question = st.chat_input("e.g. Which candidates have vector database experience?")
    if question:
        st.session_state["chat"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and answering..."):
                chunks = st.session_state["chunks"]
                k = min(chunks_k, len(chunks))
                q = embed_texts([redact_pii(question)], "RETRIEVAL_QUERY")
                scores, ids = st.session_state["chunk_index"].search(q, k)
                hits = [{**chunks[i], "score": float(sc)} for sc, i in zip(scores[0], ids[0])]
                try:
                    answer = answer_question(question, hits)
                except Exception as e:
                    answer = f"Could not generate an answer (rate limit?): {e}"
            st.markdown(answer)
            with st.expander("Retrieved chunks used for this answer"):
                for h in hits:
                    st.markdown(f"**{h['resume']}**  (similarity {h['score']:.2f})")
                    st.text(h["text"])
        st.session_state["chat"].append({"role": "assistant", "content": answer, "sources": hits})
