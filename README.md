# VERNA Agentic RAG App

VERNA is a lightweight agentic RAG application built with Streamlit, LangChain, LangGraph, OpenAI embeddings, and a local Chroma vector database. It lets users upload documents or provide URLs, build a knowledge base, and ask questions using different retrieval strategies.

Live demo: https://vernaagenticrag.streamlit.app/

---

## 1. Project architecture

The application is structured into a few clear layers:

### Frontend and orchestration
- app.py
  - Handles the Streamlit UI
  - Stores chat history and KB state in session state
  - Lets users enter OpenAI/Tavily keys
  - Provides UI for uploading files, adding URLs, building the KB, and querying the assistant
  - Builds and caches the LangGraph agent instance

### Knowledge ingestion
- loaders/document_loader.py
  - Loads content from uploaded files and web URLs
  - Extracts raw text from PDFs and web pages
  - Splits content into chunks before embedding

### Vector storage and retrieval
- store/vector_store.py
  - Creates and loads a local Chroma vector store
  - Uses OpenAI embeddings to encode document chunks
  - Builds a retriever for semantic search over the stored knowledge base

### Agent reasoning and RAG flow
- core/graph.py
  - Defines the LangGraph workflow for the assistant
  - Supports three modes:
    - Naive RAG
    - Corrective RAG
    - Adaptive RAG
  - Routes questions to the correct data source
  - Grades retrieved documents for relevance
  - Falls back to web search with Tavily when needed

### Placeholder app entry
- main.py
  - Minimal starter script
  - Not the main runtime for the app; the real application runs through Streamlit in app.py

---

## 2. Directory structure

```text
verna-agentic-rag-app/
├── app.py                     # Main Streamlit app
├── main.py                    # Starter entry point
├── pyproject.toml             # Python project metadata and dependencies
├── requirements.txt           # Installed Python packages
├── README.md                  # Project documentation
├── .env                       # Local environment variables (not committed)
├── chroma_db/                 # Local Chroma vector database (created at runtime)
├── core/
│   └── graph.py               # LangGraph agent workflow and RAG logic
├── loaders/
│   └── document_loader.py     # File and URL ingestion + text chunking
├── store/
│   └── vector_store.py        # Chroma store creation, retrieval, and persistence
└── .venv/                     # Local virtual environment
```

---

## 3. How the application works

### Step 1: User configures API keys
The sidebar in app.py asks for:
- OpenAI API key
- Tavily API key (optional, but required for web fallback/search)

The selected LLM model is then used to build the graph.

### Step 2: Knowledge base is built
The user can:
- upload one or more files, or
- enter one or more URLs

Then they click Build KB.

### Step 3: Documents are ingested
The loader pipeline does the following:
- reads uploaded files to temporary files
- extracts raw text from supported document sources
- fetches HTML from URLs and strips the page content
- stores the extracted text as LangChain Document objects

### Step 4: Text is chunked and embedded
The extracted content is split into smaller chunks using a recursive text splitter.

Then:
- embeddings are generated using OpenAI embedding model
- those chunks are stored in a local Chroma database
- the app creates a retriever to search the vector store semantically

### Step 5: A question is sent to the RAG graph
Each user question flows through a LangGraph workflow:
- retrieve relevant chunks
- optionally grade the chunks by relevance
- route to vector search or web search depending on the strategy
- generate the final response using the LLM

### Step 6: Response is shown with reasoning steps
The app displays:
- final answer
- optional reasoning details (the "Agent reasoning" expander)
- chat history for the current session

---

## 4. RAG modes

### Naive RAG
- Retrieve the most similar documents from Chroma
- Pass the retrieved context to the LLM
- Generate the final answer

Use when you want a basic retrieval-first pattern.

### Corrective RAG
- Retrieve documents
- Grade each document for relevance
- If no documents are relevant, switch to web search
- Generate the final answer using the best available context

Use when retrieval quality matters and you want extra validation.

### Adaptive RAG
- Route the question to either vector store or web search based on intent
- Rewrite the query for better retrieval when needed
- Grade retrieved results and continue with generation

Use when the app needs to decide whether the answer should come from local docs or external knowledge.

---

## 5. Step-by-step setup

### Step 1: Clone the repo
```bash
git clone <repo-url>
cd verna-agentic-rag-app
```

### Step 2: Create a virtual environment
```bash
python -m venv .venv
```

On Windows:
```powershell
.venv\Scripts\activate
```

On macOS/Linux:
```bash
source .venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

If you prefer the project metadata flow:
```bash
pip install -e .
```

### Step 4: Create environment variables
Create a file named `.env` in the project root:

```env
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

The app loads these values via Python-dotenv when it starts.

### Step 5: Run the Streamlit app
```bash
streamlit run app.py
```

The app will open in the browser and show the UI.

### Step 6: Build the knowledge base
In the sidebar:
1. Add your OpenAI API key
2. (Optional) add your Tavily key
3. Select the model and RAG strategy
4. Upload documents or paste URLs
5. Click Build KB

### Step 7: Ask questions
Once the KB is built:
- type a question in the chat box
- the agent will use the selected strategy to respond
- use the "Agent reasoning" panel to inspect intermediate steps

---

## 6. Key implementation details

### Streamlit state
The app uses Streamlit session state to maintain:
- chat messages
- KB version
- UI state

This lets the UI refresh properly after documents are loaded or the KB is rebuilt.

### Chroma persistence
The vector database is stored in the local folder:
```text
./chroma_db
```

This allows the app to reuse the same knowledge base between runs without reprocessing everything.

### Embedding model
The project uses:
```python
text-embedding-3-small
```

through OpenAI embeddings for document indexing.

### Retriever behavior
The retriever uses a maximum marginal relevance (MMR) search strategy to balance relevance and diversity of returned chunks.

---

## 7. File-level responsibilities

### app.py
This is the main orchestration layer.
It handles:
- configuration inputs
- knowledge base creation
- graph initialization
- chat loop and answer rendering
- error states and session management

### loaders/document_loader.py
This covers data ingestion.
It is responsible for:
- PDF parsing
- URL content extraction
- text cleanup
- chunk splitting

### store/vector_store.py
This covers persistence and retrieval.
It handles:
- Chroma initialization
- vector store creation
- embedding configuration
- retriever creation
- existence checks for the KB

### core/graph.py
This contains the actual reasoning workflow.
It includes:
- reusable prompts
- routing logic
- document grading
- web fallback logic
- answer generation
- state graph assembly

---

## 8. Current notes and limitations

- The application is designed around local Chroma persistence and OpenAI-based retrieval.
- The current document loader is most complete for PDF inputs and URL-based textual content.
- Additional file formats may require extra loader implementations and validation.
- The app depends on valid OpenAI and optionally Tavily credentials to work fully.

---

## 9. Suggested next improvements

- Add richer support for DOCX, CSV, Markdown, and HTML loading
- Add explicit error handling and retry logic for failed URL fetches
- Add chat export and document metadata filtering
- Add a backend API layer in addition to the Streamlit UI
- Add unit tests for loader, store, and graph functions

---

## 10. Quick start summary

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy NUL .env            # or create .env manually
streamlit run app.py
```

Then:
1. enter your OpenAI key
2. add Tavily key if needed
3. upload files or add URLs
4. click Build KB
5. ask a question

This gives a clean end-to-end flow from document ingestion to semantic retrieval and final answer generation.
