# 📚 RAG Document Assistant

A **Retrieval-Augmented Generation (RAG) Document Assistant** that allows users to upload multiple PDF documents and ask questions about their content through a conversational Streamlit interface.

The system evaluates whether the retrieved document context is sufficiently relevant to the user's question. Based on the retrieval quality, it can answer using:

* 📚 **Uploaded document context**
* 🌐 **Web search context**
* 🔀 **Both document and web context**

The RAG workflow is orchestrated using **LangGraph**, while **FAISS** is used for vector similarity search and **NVIDIA Embeddings** are used to generate document embeddings.

---

## 🚀 Features

### 📄 Multiple PDF Upload

Users can upload multiple PDF documents through the Streamlit interface.

The application:

1. Loads all uploaded PDFs.
2. Extracts their text.
3. Splits the text into smaller chunks.
4. Generates embeddings.
5. Stores the embeddings in a FAISS vector database.

---

### 🔍 Semantic Document Retrieval

The system uses:

* NVIDIA Nemotron embeddings
* FAISS vector database
* Similarity-based retrieval

For every question, the system retrieves the most relevant document chunks.

The current configuration retrieves the top **4 chunks**.

---

### 🤖 LLM-Based Retrieval Evaluation

Retrieved chunks are evaluated by an LLM before generating the final answer.

Each chunk receives a relevance score between:

```text
0.0 → Completely irrelevant
1.0 → Sufficient to answer the question
```

The system uses two thresholds:

```text
UPPER_TH = 0.7
LOWER_TH = 0.3
```

The retrieved context is classified into three states.

#### CORRECT

At least one retrieved chunk has a score greater than `0.7`.

```text
📚 Answer retrieved from the given document context.
```

The answer is generated using the relevant document context.

---

#### INCORRECT

All retrieved chunks have a score below `0.3`.

```text
🌐 Answer retrieved from web context
because the provided document context is not relevant.
```

The system rewrites the question into a web-search query and searches the web using Tavily.

---

#### AMBIGUOUS

No chunk scores above `0.7`, but not all chunks score below `0.3`.

```text
🔀 Answer generated using both the given document
context and web context because the provided context
was not sufficient.
```

The system performs web search and combines:

```text
Relevant document chunks
        +
Web search results
```

before generating the final response.

---

## 🧠 System Architecture

```text
                    ┌──────────────────────┐
                    │      Streamlit       │
                    │      Frontend        │
                    └──────────┬───────────┘
                               │
                         Upload PDFs
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Document Processor   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │     PDF Loader       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Recursive Text       │
                    │ Splitter             │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ NVIDIA Embeddings    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       FAISS          │
                    │    Vector Store      │
                    └──────────┬───────────┘
                               │
                         User Question
                               │
                               ▼
                    ┌──────────────────────┐
                    │      LangGraph       │
                    │      RAG Pipeline    │
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
                    │ Documents            │
                    └──────────┬───────────┘
                               │
                  ┌────────────┼────────────┐
                  │            │            │
                  ▼            ▼            ▼
              CORRECT      AMBIGUOUS    INCORRECT
                  │            │            │
                  │            ▼            ▼
                  │       Web Search    Web Search
                  │            │            │
                  │            └─────┬──────┘
                  │                  │
                  └────────┬─────────┘
                           ▼
                    ┌──────────────────────┐
                    │    Answer Generator  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Chat UI    │
                    └──────────────────────┘
```

---

# 🏗️ Project Structure

```text
RAG-Document-Assistant/
│
├── app.py
│
├── document_processor.py
│
├── rag_pipeline.py
│
├── documents/
│
├── .env
│
├── requirements.txt
│
└── README.md
```

### `app.py`

The Streamlit frontend.

Responsible for:

* PDF uploading
* Displaying uploaded documents
* Processing documents
* Maintaining chat history
* Displaying user messages
* Displaying assistant responses
* Showing retrieval verdict
* Showing retrieval reasoning

---

### `document_processor.py`

Responsible for document processing and vector database creation.

Main responsibilities:

```text
PDF
 ↓
PyPDFLoader
 ↓
Document chunks
 ↓
Text cleaning
 ↓
NVIDIA Embeddings
 ↓
FAISS
 ↓
Retriever
```

Main function:

```python
build_retriever(pdf_paths)
```

This function accepts multiple PDF paths.

---

### `rag_pipeline.py`

Contains the LangGraph RAG workflow.

Responsibilities:

* Document retrieval
* Document relevance evaluation
* Verdict generation
* Query rewriting
* Web search
* Context construction
* Answer generation
* LangGraph routing

---

# 🔄 RAG Workflow

When a user asks a question, the following workflow takes place.

## Step 1 — Retrieve

The question is passed to the FAISS retriever.

```text
User Question
      ↓
FAISS Retriever
      ↓
Top 4 relevant chunks
```

---

## Step 2 — Evaluate Retrieved Chunks

Each retrieved chunk is sent to the LLM evaluator.

The evaluator returns:

```json
{
    "score": 0.85,
    "reason": "The chunk directly discusses the requested concept."
}
```

The score is between:

```text
0.0 and 1.0
```

---

## Step 3 — Determine Verdict

### CORRECT

```text
Any score > 0.7
```

The document context is considered sufficiently relevant.

Flow:

```text
Retrieve
   ↓
Evaluate
   ↓
CORRECT
   ↓
Generate
```

---

### INCORRECT

```text
All scores < 0.3
```

The document context is considered irrelevant.

Flow:

```text
Retrieve
   ↓
Evaluate
   ↓
INCORRECT
   ↓
Rewrite Query
   ↓
Tavily Web Search
   ↓
Generate
```

---

### AMBIGUOUS

```text
No score > 0.7
AND
Not all scores < 0.3
```

Flow:

```text
Retrieve
   ↓
Evaluate
   ↓
AMBIGUOUS
   ↓
Rewrite Query
   ↓
Tavily Web Search
   ↓
Relevant Document Context
        +
Web Context
   ↓
Generate
```

---

# 🌐 Web Search

When the uploaded document context is insufficient, the system uses **Tavily Search**.

Before searching, the LLM rewrites the user's question into a concise search query.

For example:

```text
Original:

What are the latest developments in transformer architectures?

        ↓

Rewritten Query:

latest developments transformer architectures
```

The rewritten query is sent to Tavily.

The search results are converted into LangChain `Document` objects before being passed to the answer-generation stage.

---

# 🧩 Technologies Used

| Technology        | Purpose                         |
| ----------------- | ------------------------------- |
| Python            | Backend development             |
| Streamlit         | Frontend / chat interface       |
| LangChain         | LLM and RAG components          |
| LangGraph         | RAG workflow orchestration      |
| FAISS             | Vector similarity search        |
| NVIDIA Embeddings | Text embeddings                 |
| OpenRouter        | LLM provider                    |
| Tavily            | Web search                      |
| PyPDFLoader       | PDF document loading            |
| Pydantic          | Structured LLM outputs          |
| python-dotenv     | Environment variable management |

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

Move into the project:

```bash
cd RAG-Document-Assistant
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have `requirements.txt` yet, install the main dependencies:

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

# 🔐 Environment Variables

Create a `.env` file in the project root.

```env
NVIDIA_API_KEY=your_nvidia_api_key

OPENROUTER_API_KEY=your_openrouter_api_key

TAVILY_API_KEY=your_tavily_api_key
```

Replace the values with your actual API keys.

### Important

Never commit `.env` to GitHub.

Add this to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

---

# ▶️ Running the Application

Activate your virtual environment first.

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

## Step 1 — Upload Documents

Use the sidebar:

```text
Upload PDF documents
```

Multiple PDFs can be selected at the same time.

For example:

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

The application will:

```text
Load PDFs
   ↓
Extract pages
   ↓
Create chunks
   ↓
Generate embeddings
   ↓
Create FAISS index
```

The UI also displays:

```text
Pages
Chunks
```

---

## Step 3 — Ask Questions

After processing the documents, use:

```text
Ask something about your documents...
```

For example:

```text
What is batch normalization?
```

---

## Step 4 — Check the Verdict

The application displays the source of the answer.

### Document Context

```text
📚 Answer retrieved from the given document context.
```

### Web Context

```text
🌐 Answer retrieved from web context because
the provided document context is not relevant.
```

### Both

```text
🔀 Answer generated using both the given document
context and web context because the provided
context was not sufficient.
```

---

# 🗂️ Multiple Document Handling

The application supports multiple PDFs in a single session.

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

All documents are combined into a single collection of chunks.

These chunks are embedded and stored in FAISS.

The retriever can therefore search across all uploaded documents.

---

# 📊 Current Configuration

### Chunking

```python
chunk_size = 900
chunk_overlap = 150
```

### Retrieval

```python
search_type = "similarity"
k = 4
```

### Evaluation Thresholds

```python
UPPER_TH = 0.7
LOWER_TH = 0.3
```

### Web Search

```python
max_results = 5
```

### Embedding Model

```text
nvidia/nemotron-3-embed-1b
```

### LLM

```text
openrouter/free
```

---

# 🧠 Why Use Retrieval Evaluation?

A normal RAG pipeline typically looks like:

```text
Question
   ↓
Retrieve
   ↓
Generate Answer
```

This can cause problems when the retrieved documents are unrelated to the question.

This project adds an evaluation stage:

```text
Question
   ↓
Retrieve
   ↓
Evaluate Relevance
   ↓
┌───────────┬────────────┬────────────┐
│ CORRECT   │ AMBIGUOUS  │ INCORRECT  │
└───────────┴────────────┴────────────┘
```

This allows the system to decide whether it should rely on the uploaded documents or use web search.

---

# 🔎 Retrieval Transparency

The application exposes the retrieval decision to the user rather than silently switching between sources.

Each generated answer has a corresponding verdict:

```text
CORRECT
INCORRECT
AMBIGUOUS
```

The application also provides a retrieval-details section containing:

```text
Verdict
Reason
```

This makes it easier to understand why the system selected a particular information source.

---

# 🛠️ Future Improvements

Possible improvements include:

### 1. Conversational RAG

Currently, the chat interface maintains the conversation visually.

A future version can pass previous conversation history into the retrieval/query-rewriting stage so questions such as:

```text
User:
What is batch normalization?

User:
What are its advantages?

User:
How is it different from layer normalization?
```

can be interpreted using previous turns.

---

### 2. Source Citations

Display the source PDF and page number used for each answer.

Example:

```text
Sources

📄 Deep_Learning.pdf — Page 12
📄 Neural_Networks.pdf — Page 27
```

---

### 3. Persistent Vector Database

Currently, the vector index can be rebuilt when documents are processed.

A persistent vector database could be used for larger document collections.

---

### 4. Streaming Responses

Stream LLM responses token-by-token in the Streamlit interface.

---

### 5. Document Management

Allow users to:

```text
Add documents
Remove documents
View documents
Switch document collections
```

---

### 6. Better Retrieval

Possible future retrieval improvements:

```text
Hybrid Search
BM25
MMR Retrieval
Reranking
Metadata Filtering
Cross-Encoder Reranking
```

---

# 🔒 Security

Do not expose API keys directly in Python files.

Use environment variables:

```env
NVIDIA_API_KEY=...
OPENROUTER_API_KEY=...
TAVILY_API_KEY=...
```

And make sure `.env` is included in `.gitignore`.

---

# 📌 Example Workflow

```text
                    USER
                     │
                     ▼
              Upload PDFs
                     │
                     ▼
          Process Documents
                     │
                     ▼
             PDF Extraction
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
           ┌─────────┼─────────┐
           │         │         │
           ▼         ▼         ▼
        CORRECT   AMBIGUOUS  INCORRECT
           │         │         │
           │         ▼         ▼
           │       Web Search
           │         │         │
           │         └────┬────┘
           │              │
           ▼              ▼
        Document       Document
        Context    +   Web Context
           │              │
           └──────┬───────┘
                  ▼
            Answer Generator
                  │
                  ▼
             Streamlit Chat
```

---

# 👨‍💻 Author

**Sunkavalli Aditya**

B.Tech — Information Technology

National Institute of Technology Srinagar

---

# ⭐ Project Highlights

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation
* Multi-document question answering
* Semantic vector search
* FAISS vector databases
* NVIDIA embedding models
* LLM-based retrieval evaluation
* LangGraph conditional workflows
* Query rewriting
* Web-augmented RAG
* Streamlit conversational UI
* Multiple PDF processing
* Source-aware retrieval decisions

---

## 📄 License

This project is intended for educational and portfolio purposes. Add an appropriate open-source license if you plan to distribute the project publicly.
