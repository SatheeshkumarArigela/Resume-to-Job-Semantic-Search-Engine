# 🔎 Resume-to-Job Semantic Search Engine

An AI-powered application that matches resumes with job descriptions using semantic search and vector embeddings. It also uses Retrieval-Augmented Generation (RAG) to answer questions about uploaded candidates and provides AI-generated explanations of resume-job matches.

## 🚀 Features

- **Resume Upload:** Supports PDF and TXT resumes.
- **Local Folder Loading:** Load resumes from a local directory, with an option to include subfolders.
- **Semantic Job Matching:** Finds relevant candidates based on meaning rather than exact keyword matches.
- **Vector Search with FAISS:** Uses vector embeddings and similarity search to rank candidates.
- **AI Match Explanations:** Generates strengths, skill gaps, and an overall verdict for selected matches.
- **RAG-Based Candidate Q&A:** Ask questions about candidates using retrieved resume content.
- **Source References:** Includes resume filenames in generated answers.
- **PII Redaction:** Masks detected email addresses and phone numbers before embedding and LLM processing.
- **Configurable Search:** Adjust the number of candidates, AI explanations, and retrieved chunks.
- **Retry Handling:** Retries embedding requests when API calls fail.

## 🧰 Tech Stack

- Python
- Streamlit
- Google Gemini API
- Gemini Embeddings
- FAISS
- NumPy
- pypdf
- Regular Expressions

## ⚙️ How It Works

### 1. Resume Ingestion
Users upload PDF/TXT resumes or provide a local folder path. The application extracts the text from each file.

### 2. Privacy Preprocessing
Email addresses and phone numbers detected by regular expressions are replaced with `[EMAIL]` and `[PHONE]`.

### 3. Text Embedding
The application converts resume text into numerical vectors using Google's Gemini embedding model.

### 4. Semantic Matching
FAISS indexes the resume vectors. When a job description is entered, it is embedded as a query and compared with the indexed resumes to retrieve the most relevant candidates.

### 5. AI-Generated Explanations
For selected candidates, Gemini generates:
- Matching strengths
- Missing or weaker requirements
- An overall match verdict

### 6. RAG-Based Candidate Questions
Resume text is divided into overlapping chunks. The chunks are embedded and indexed separately. When a user asks a question, the application retrieves relevant chunks and sends them to Gemini to generate a concise answer with resume filename references.

## 🧠 Key Concepts Used

- Text embeddings
- Semantic similarity search
- Vector databases and indexing with FAISS
- Cosine similarity using normalized vectors and inner-product search
- Retrieval-Augmented Generation (RAG)
- Text chunking and overlap
- Metadata and source attribution
- Personally identifiable information (PII) redaction
- Prompt engineering
- API error handling and retries

## 📁 Project Structure

```text
Resume-to-Job-Semantic-Search-Engine/
│
├── app.py
├── README.md
└── requirements.txt
```

- `app.py` — Resume ingestion, text processing, embeddings, FAISS search, AI explanations, RAG retrieval, and Streamlit interface.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.

## 🛠️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd Resume-to-Job-Semantic-Search-Engine
```

Replace `YOUR_REPOSITORY_URL` with your GitHub repository URL.

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

If you already have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the packages used by the application:

```bash
pip install streamlit google-genai faiss-cpu numpy pypdf
```

## 🔑 Configure the Gemini API Key

Get an API key from [Google AI Studio](https://aistudio.google.com/apikey).

**Windows Command Prompt:**

```cmd
set GEMINI_API_KEY=your_api_key_here
```

**Windows PowerShell:**

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

**macOS/Linux:**

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Alternatively, configure the key using Streamlit Secrets in `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Never commit API keys or secrets to GitHub.

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then:

1. Load sample resumes from a folder or upload PDF/TXT files.
2. Paste a job description.
3. Select how many candidates to display.
4. Click **Find best candidates**.
5. Review match scores and optional AI explanations.
6. Ask questions about the candidates using the RAG chat interface.

## 🔍 Example Questions

- Which candidates mention Python and machine learning?
- Which resumes show experience with vector databases?
- Which candidates have experience with retrieval-augmented generation?
- What relevant skills does a particular candidate have?
- Which job requirements are not clearly demonstrated in a resume?

The answers depend on the content of the resumes provided.

## ⚙️ Configuration

The current application defines these settings in its Python code:

| Setting | Purpose |
|---|---|
| `EMBED_MODEL` | Gemini embedding model |
| `LLM_MODEL` | Model used for match explanations and RAG answers |
| `DIM` | Embedding vector dimension |
| `BATCH_SIZE` | Number of texts processed per embedding batch |
| `MAX_CHARS` | Maximum resume characters processed |
| `CHUNK_SIZE` | Target size of each RAG text chunk |
| `CHUNK_OVERLAP` | Overlap between consecutive chunks |

The interface also lets users configure the number of candidates displayed, AI explanations generated, and chunks retrieved for chat.

## 🔐 Privacy and Responsible Use

- Test with sample or authorized resumes.
- Email and phone-number redaction is pattern-based and may not catch every personal detail.
- Redaction does not remove names, addresses, or all other potentially identifying information.
- Resume content is sent to external AI services for embedding and generation.
- Follow applicable consent, privacy, data-retention, and organizational requirements before processing real candidate information.
- Treat match scores and AI-generated explanations as decision-support information, not objective measures of candidate quality.

## ⚠️ Limitations

- Semantic similarity scores are not calibrated probabilities of hiring success.
- Results depend on resume quality, job-description clarity, embedding behavior, and retrieval settings.
- AI-generated explanations may be incomplete or inaccurate.
- The RAG assistant can only answer reliably when relevant information is present in the retrieved chunks.
- PDF text extraction may not work well with scanned or image-only documents.
- The application uses external API calls and may be affected by rate limits, availability, or model changes.
- This tool should not be used as the sole basis for hiring decisions.

## 🎯 Learning Outcomes

This project demonstrates practical exploration of:

- Building an embedding-based semantic search pipeline
- Indexing and retrieving vectors using FAISS
- Implementing a basic RAG workflow
- Chunking documents for retrieval
- Integrating LLMs with a Streamlit application
- Redacting selected personal information before processing
- Designing prompts for grounded answers and source attribution
- Handling API errors and configuring retrieval parameters

## 👨‍💻 Author

**Satheesh Kumar Arigela**

- **GitHub:** [SatheeshkumarArigela](https://github.com/SatheeshkumarArigela)
- **LinkedIn:** [Arigela Satheesh Kumar](https://www.linkedin.com/in/arigela-satheesh-kumar-59a00822b/)
