# 📚 Corrective RAG Document Assistant

An **Corrective Retrieval-Augmented Generation (RAG) Document Assistant** that allows users to upload multiple PDF documents and ask questions through a conversational Streamlit interface.

The system does not blindly generate an answer from retrieved documents. Instead, it evaluates the relevance of the retrieved context and dynamically decides whether the answer should be generated from:

- 📚 Uploaded document context
- 🌐 Web search context
- 🔀 A combination of document and web context

The RAG workflow is orchestrated using **LangGraph**, **FAISS** is used for vector similarity search, **NVIDIA Nemotron embeddings** are used for semantic representation, and **Tavily** provides web-search fallback.

---

LINK : https://corrective-rag-859h.onrender.com

# 🚀 Features

## 📄 Multiple PDF Upload

The application allows users to upload multiple PDF documents in a single session.

The system:

1. Loads the uploaded PDFs.
2. Extracts their text.
3. Splits the documents into smaller chunks.
4. Generates vector embeddings.
5. Stores the embeddings in a FAISS vector store.
6. Creates a retriever for semantic search.

---

## 🔍 Semantic Document Retrieval

The system uses:

- NVIDIA Nemotron embeddings
- FAISS vector store
- Similarity-based retrieval
- Recursive text splitting

For every user question, the retriever searches across all uploaded documents and returns the most relevant chunks.

Current configuration:

```text
Retrieval type: similarity
Top K: 4 chunks
```

---

# 🤖 Agentic Retrieval Evaluation

A major feature of this project is the **LLM-based evaluation of retrieved documents**.

Instead of directly passing retrieved chunks to the answer generator, each retrieved chunk is evaluated for relevance.

Each document receives a relevance score between:

```text
0.0 → Completely irrelevant
1.0 → Sufficiently relevant
```

The current thresholds are:

```python
UPPER_TH = 0.7
LOWER_TH = 0.3
```

Based on these scores, the system determines one of three verdicts:

```text
CORRECT
AMBIGUOUS
INCORRECT
```

---

# 📊 Retrieval Decision Logic

## 🟢 CORRECT

If **at least one retrieved document chunk has a score greater than 0.7**, the retrieved document context is considered sufficiently relevant.

```text
Retrieved Documents
        ↓
LLM Evaluation
        ↓
Score > 0.7
        ↓
CORRECT
        ↓
Generate Answer
```

The answer is generated using the relevant uploaded-document context.

The Streamlit UI displays:

```text
📚 Answer retrieved from the given document context.
```

---

## 🟡 AMBIGUOUS

If:

```text
No score > 0.7
AND
Not all scores < 0.3
```

the retrieved context is considered partially relevant but insufficient.

The system:

1. Keeps the relevant document chunks.
2. Rewrites the user's question into a web-search query.
3. Searches the web using Tavily.
4. Combines document and web context.
5. Generates the final answer.

```text
Retrieved Documents
        ↓
LLM Evaluation
        ↓
AMBIGUOUS
        ↓
Rewrite Query
        ↓
Tavily Web Search
        ↓
Document Context + Web Context
        ↓
Generate Answer
```

The UI displays:

```text
🔀 Answer generated using both the given document
context and web context because the provided context
was not sufficient.
```

---

## 🔴 INCORRECT

If **all retrieved chunks have scores below 0.3**, the uploaded document context is considered irrelevant.

The system does not rely on the retrieved document context.

Instead:

```text
Retrieved Documents
        ↓
LLM Evaluation
        ↓
INCORRECT
        ↓
Rewrite Query
        ↓
Tavily Web Search
        ↓
Generate Answer
```

The UI displays:

```text
🌐 Answer retrieved from web context because the
provided document context is not relevant.
```

---

# 🧠 System Architecture

```text
                         ┌──────────────────────┐
                         │      Streamlit       │
                         │      Frontend        │
                         │       app.py         │
                         └──────────┬───────────┘
                                    │
                              Upload PDFs
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   retriever.py       │
                         │                      │
                         │ PDF Loading          │
                         │ Text Splitting       │
                         │ Embeddings            │
                         │ FAISS Retriever      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    FAISS Vector      │
                         │        Store         │
                         └──────────┬───────────┘
                                    │
                              User Question
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       back.py        │
                         │                      │
                         │    LangGraph RAG     │
                         │      Pipeline        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Retrieve        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Evaluate Retrieved   │
                         │      Documents       │
                         └──────────┬───────────┘
                                    │
                      ┌─────────────┼─────────────┐
                      │             │             │
                      ▼             ▼             ▼
                  CORRECT       AMBIGUOUS     INCORRECT
                      │             │             │
                      │             ▼             ▼
                      │        Web Search    Web Search
                      │             │             │
                      │             └──────┬──────┘
                      │                    │
                      └──────────┬─────────┘
                                 ▼
                       ┌──────────────────────┐
                       │   Answer Generator   │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │   Streamlit Chat UI  │
                       └──────────────────────┘
```

---

# 🏗️ Project Structure

```text
RAG-Document-Assistant/
│
├── app.py
├── retriever.py
├── back.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
└── venv/
```

---

# 📂 File Responsibilities

## `app.py`

The **Streamlit frontend**.

Responsible for:

* PDF uploading
* Multiple document selection
* Processing documents
* Displaying uploaded document names
* Maintaining chat history
* Displaying user messages
* Displaying assistant responses
* Showing retrieval verdict
* Showing retrieval reasoning
* Providing the chat interface

The frontend calls the retriever-building function and the LangGraph application:

```python
from back import build_retriever, app
```

---

## `retriever.py`

Responsible for **document processing and semantic retrieval**.

Main responsibilities:

```text
PDF Files
   ↓
PyPDFLoader
   ↓
Document Pages
   ↓
RecursiveCharacterTextSplitter
   ↓
Text Chunks
   ↓
NVIDIA Embeddings
   ↓
FAISS Vector Store
   ↓
Retriever
```

Main function:

```python
build_retriever(pdf_paths)
```

The function accepts multiple PDF paths, processes them, creates embeddings, builds a FAISS vector store, and stores the resulting retriever in a module-level `retriever` variable.

`back.py` accesses the latest retriever instance by importing the module itself:

```python
import retriever as retriever_module
...
retriever_module.retriever.invoke(question)
```

This ensures `back.py` always reads the most up-to-date retriever after `build_retriever()` has been called, without needing a separate getter function.

---

## `back.py`

Contains the **LLM-powered Agentic RAG workflow**.

Responsible for:

* Retrieving relevant document chunks
* Evaluating document relevance
* Generating retrieval verdicts
* Rewriting web-search queries
* Performing Tavily web search
* Combining document and web context
* Generating final answers
* LangGraph state management
* Conditional routing

It also re-exports `build_retriever` (imported from `retriever.py`) alongside the compiled LangGraph application `app`, so the frontend only needs one import line.

The main workflow is compiled into a LangGraph application.

---

# 🔄 Complete RAG Workflow

When a user asks a question, the application follows this process.

## Step 1 — User Question

The user enters a question through the Streamlit chat interface.

Example:

```text
What is batch normalization?
```

---

## Step 2 — Document Retrieval

The question is passed to the FAISS retriever.

```text
User Question
      ↓
FAISS Retriever
      ↓
Top 4 Relevant Chunks
```

---

## Step 3 — LLM Relevance Evaluation

Each retrieved chunk is evaluated by the LLM.

The evaluator produces structured output similar to:

```json
{
    "score": 0.85,
    "reason": "The document directly discusses the requested concept."
}
```

The score ranges from:

```text
0.0 → 1.0
```

---

# Step 4 — Verdict Determination

The retrieved context is classified using:

```python
UPPER_TH = 0.7
LOWER_TH = 0.3
```

Decision logic:

```text
                    Retrieved Chunks
                           │
                           ▼
                    Evaluate Scores
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
        Score > 0.7    Mixed Scores   All < 0.3
             │             │             │
             ▼             ▼             ▼
          CORRECT      AMBIGUOUS      INCORRECT
```

---

# Step 5 — Conditional Routing

LangGraph decides the next step based on the verdict.

```text
CORRECT
   ↓
Generate using document context
```

```text
AMBIGUOUS
   ↓
Rewrite Query
   ↓
Tavily Search
   ↓
Document + Web Context
   ↓
Generate
```

```text
INCORRECT
   ↓
Rewrite Query
   ↓
Tavily Search
   ↓
Web Context
   ↓
Generate
```

---

# 🌐 Web Search Fallback

When document retrieval is insufficient, the system uses **Tavily Search**.

Before searching, the LLM rewrites the original question into a concise search query.

Example:

```text
Original Question:

What are the latest developments in transformer architectures?

                ↓

Rewritten Search Query:

latest developments transformer architectures
```

The rewritten query is then sent to Tavily.

The returned search results are converted into LangChain `Document` objects and passed to the answer-generation stage.

---

# 📝 Answer Generation

The answer generator receives context depending on the retrieval verdict.

## CORRECT

```text
Relevant Uploaded Document Context
                ↓
           Answer Generator
```

---

## INCORRECT

```text
Web Search Context
        ↓
Answer Generator
```

---

## AMBIGUOUS

```text
Relevant Document Context
          +
Web Search Context
          ↓
Answer Generator
```

The answer generation prompt instructs the model to answer using the supplied context and avoid unsupported information.

---

# 🧩 LangGraph Workflow

The backend is organized as a state-based LangGraph workflow.

Main nodes:

```text
retrieve
   ↓
eval_each_doc
   ↓
route_after_eval
   │
   ├── CORRECT ────────→ generate
   │
   ├── AMBIGUOUS ─────→ rewrite_query
   │                         ↓
   │                     web_search
   │                         ↓
   │                      generate
   │
   └── INCORRECT ──────→ rewrite_query
                             ↓
                         web_search
                             ↓
                          generate
```

---

# 📦 Technologies Used

| Technology                     | Purpose                         |
| ------------------------------ | -------------------------------- |
| Python                         | Backend development             |
| Streamlit                      | Frontend and chat interface     |
| LangChain                      | RAG and LLM components          |
| LangGraph                      | Agentic workflow orchestration  |
| FAISS                          | Vector similarity search        |
| NVIDIA Nemotron Embeddings     | Document embeddings             |
| OpenRouter                     | LLM access                      |
| Tavily                         | Web search                      |
| PyPDFLoader                    | PDF loading                     |
| RecursiveCharacterTextSplitter | Document chunking               |
| Pydantic                       | Structured LLM outputs          |
| python-dotenv                  | Environment variable management |

---

# ⚙️ Current Configuration

## Document Chunking

```python
chunk_size = 900
chunk_overlap = 150
```

---

## Retrieval

```python
search_type = "similarity"
k = 4
```

---

## Retrieval Evaluation

```python
UPPER_TH = 0.7
LOWER_TH = 0.3
```

---

## Web Search

```text
Maximum results = 5
```

---

## Embedding Model

```text
nvidia/nemotron-3-embed-1b
```

---

## LLM

```text
openrouter/free
```

---

# 📄 Multiple Document Processing

Multiple PDFs can be uploaded and processed together.

Example:

```text
PDF 1
 ├── Page 1
 ├── Page 2
 └── Page 3

PDF 2
 ├── Page 1
 ├── Page 2
 └── Page 3

PDF 3
 ├── Page 1
 └── Page 2
```

All extracted documents are combined into a collection of LangChain documents.

They are then:

```text
Documents
    ↓
Chunks
    ↓
Embeddings
    ↓
FAISS
```

The retriever can therefore search across all uploaded PDFs within the current session.

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

```env
NVIDIA_API_KEY=your_nvidia_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Replace the values with your actual API keys.

## Important

Never commit `.env` to GitHub.

Add the following to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd RAG-Document-Assistant
```

---

# 2. Create Virtual Environment

## Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

---

## Linux / macOS

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet, install the main dependencies:

```bash
pip install streamlit
pip install langchain
pip install langchain-community
pip install langchain-core
pip install langchain-text-splitters
pip install langchain-nvidia-ai-endpoints
pip install langchain-openrouter
pip install langgraph
pip install faiss-cpu
pip install pypdf
pip install python-dotenv
pip install tavily-python
```

---

# ▶️ Running the Application

Activate the virtual environment first.

Then run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 💬 Using the Application

## Step 1 — Upload PDFs

Use the sidebar to upload one or multiple PDF documents.

Example:

```text
📄 Machine_Learning.pdf
📄 Deep_Learning.pdf
📄 NLP.pdf
```

---

## Step 2 — Process Documents

Click:

```text
🚀 Process Documents
```

The application performs:

```text
Load PDFs
    ↓
Extract Pages
    ↓
Create Chunks
    ↓
Generate Embeddings
    ↓
Create FAISS Index
    ↓
Initialize Retriever
```

The interface also displays the number of processed pages and chunks.

---

# Step 3 — Ask Questions

After processing the documents, use the chat input.

Example:

```text
What is batch normalization?
```

The question is passed to the Agentic RAG pipeline.

---

# Step 4 — View the Answer

The application generates an answer and displays the retrieval verdict.

Possible verdicts:

```text
CORRECT
INCORRECT
AMBIGUOUS
```

---

# 🔎 Retrieval Transparency

The application does not silently switch between document and web sources.

Instead, it exposes the retrieval decision.

For each answer, the interface can show:

```text
Verdict
Reason
```

Example:

```text
Verdict: CORRECT

Reason:
The retrieved document directly contains information
relevant to the user's question.
```

This makes the RAG decision easier to understand and debug.

---

# 🧠 Why Retrieval Evaluation?

A basic RAG system typically works like:

```text
Question
   ↓
Retrieve
   ↓
Generate
```

The problem is that a retriever may return documents that are semantically similar but do not actually contain enough information to answer the question.

This project adds an LLM-based evaluation layer:

```text
Question
   ↓
Retrieve
   ↓
Evaluate Relevance
   ↓
Determine Verdict
   ↓
┌────────────┬────────────┬────────────┐
│  CORRECT   │ AMBIGUOUS  │  INCORRECT │
└────────────┴────────────┴────────────┘
       │           │             │
       ▼           ▼             ▼
    Document    Document +      Web
    Context     Web Context    Context
```

This creates a more adaptive RAG workflow instead of always depending on retrieved documents.

---

# 🔄 Example End-to-End Workflow

```text
                         USER
                           │
                           ▼
                    Upload PDF(s)
                           │
                           ▼
                  Process Documents
                           │
                           ▼
                     PDF Loading
                           │
                           ▼
                       Chunking
                           │
                           ▼
                 NVIDIA Embeddings
                           │
                           ▼
                        FAISS
                           │
                           ▼
                    Ask Question
                           │
                           ▼
                      Retrieval
                           │
                           ▼
                LLM Relevance Check
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
           CORRECT      AMBIGUOUS    INCORRECT
              │            │            │
              │            ▼            ▼
              │       Rewrite Query
              │            │            │
              │            ▼            │
              │       Tavily Search     │
              │            │            │
              │            └─────┬──────┘
              │                  │
              ▼                  ▼
        Document Context    Web Context
              │                  │
              │           + Document Context
              │                  │
              └─────────┬────────┘
                        ▼
                 Answer Generator
                        │
                        ▼
                  Streamlit Chat
```

---

# 🛠️ Future Improvements

## 1. Conversational RAG

Currently, the Streamlit interface maintains the visual conversation history.

A future version can pass previous conversation turns into the retrieval and query-rewriting stages.

For example:

```text
User:
What is batch normalization?

User:
What are its advantages?

User:
How is it different from layer normalization?
```

The system could use previous turns to understand the context of follow-up questions.

---

## 2. Source Citations

Display the exact source PDF and page number used for generating each answer.

Example:

```text
Sources:

📄 Deep_Learning.pdf — Page 12
📄 Neural_Networks.pdf — Page 27
```

---

## 3. Persistent Vector Database

Currently, the FAISS index is created during document processing.

A future version could use a persistent vector database for larger document collections and reuse existing embeddings.

---

## 4. Streaming Responses

Stream LLM responses token-by-token in the Streamlit interface instead of waiting for the complete response.

---

## 5. Document Management

Add functionality to:

```text
Add Documents
Remove Documents
View Documents
Switch Document Collections
```

---

## 6. Improved Retrieval

Possible improvements include:

```text
Hybrid Search
BM25
MMR Retrieval
Cross-Encoder Reranking
Metadata Filtering
Query Expansion
```

---

# 🔒 Security

API keys should never be hardcoded inside Python files.

Use environment variables:

```env
NVIDIA_API_KEY=...
OPENROUTER_API_KEY=...
TAVILY_API_KEY=...
```

Make sure `.env` is included in `.gitignore`.

Never upload API keys to GitHub.

---

# 🎯 Project Highlights

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation
* Agentic RAG
* Multi-document question answering
* Semantic vector search
* FAISS vector databases
* NVIDIA embedding models
* LLM-based retrieval evaluation
* LangGraph conditional workflows
* Query rewriting
* Web-augmented RAG
* Tavily web search
* Streamlit conversational UI
* Multiple PDF processing
* Conditional source selection
* Structured LLM outputs
* Retrieval transparency

---

# 👨‍💻 Author

**Sunkavalli Aditya**

B.Tech — Information Technology

National Institute of Technology Srinagar

---

# 📄 License

This project is intended for educational and portfolio purposes.

If you plan to distribute the project publicly, add an appropriate open-source license such as MIT.