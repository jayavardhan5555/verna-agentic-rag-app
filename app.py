from __future__ import annotations

import os
import streamlit as st
from dotenv import load_dotenv
from loaders.document_loader import load_uploaded_files,load_urls, split_documents
from store.vector_store import build_vector_store, vector_store_exists,load_vector_store,get_retriver, CHROMA_DIR
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_tavily import TavilySearch
import shutil
from core.graph import build_graph

load_dotenv()

st.set_page_config(
    page_title="VERNA AGENTIC RAG", layout="wide", initial_sidebar_state="expanded"
)

# Session State Defaults

if "messages" not in st.session_state:
    st.session_state.messages:list[dict] =[]
if "kb_version" not in st.session_state:
    st.session_state.kb_version = 0

@st.cache_resource
def _get_graph(api_key:str, model: str, tavily_key: str, kb_version: int, rag_mode: str):

    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
    vs = load_vector_store(api_key)
    retriver = get_retriver(vs)

    tavily_tool = TavilySearch(max_results=4, tavily_api_key=tavily_key) if tavily_key else None

    return build_graph(rag_mode, llm, retriver, tavily_tool)


with st.sidebar:
    st.title("Configuration")

    with st.expander("API Keys", expanded= True):
                
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="Enter your OpenAI API key here")

        tavily_key = st.text_input("Tavily Key", type="password", placeholder="Enter your Tavily key here")

    selected_model = st.selectbox("Select Model", options=["gpt-4o-mini", "gpt-4o"], index=0, disabled=not api_key)

    _RAG_OPTIONS = {
            "Adaptive RAG" : "adaptive",
            "Corrective RAG": "corrective",
            "Naive RAG": "naive"
        }

    _RAG_HELP = (
            "**Naive RAG** - retrieve -> generate (fastest). \n"
            "**Corrective RAG** - retrieve -> grade -> web fallback if no relevant docs found. \n"
            "**Adaptive RAG** - LLM routed, graded, with query rewriting."
        )

    rag_label = st.selectbox("RAG Strategy",options=list(_RAG_OPTIONS.keys()),index= 0, help= _RAG_HELP)
    rag_mode: str = _RAG_OPTIONS[rag_label]

    st.divider()

    st.subheader("Knowledge Source")

    uploaded_file = st.file_uploader("Upload documents", type=["pdf", "docx", "csv", "md", "html"], accept_multiple_files=True)

    url_input = st.text_input("Enter a web URL", placeholder="https://example.com")

    col_build, col_clear = st.columns(2)

    with col_build:
        build_clicked = st.button(
            "Build KB", use_container_width= True, type= "primary", disabled= not api_key
        )
    with col_clear:
        if st.button("Clear KB", use_container_width= True):
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
            st.session_state.kb_version += 1
            _get_graph.clear()
            st.rerun()

    if build_clicked:
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if not uploaded_file and not urls:
            st.error("Upload at least one file or enter URL")
        else:
            with st.status("Building knowledge base...", expanded=True) as status:
                try:
                    all_docs =[]

                    if uploaded_file:
                        st.write(f"loading {len(uploaded_file)} file(s)...")
                        file_docs = load_uploaded_files(uploaded_file)
                        all_docs.extend(file_docs)
                        st.write(f"{len(file_docs)} selection(s) loaded from files")

                    if urls:
                        st.write(f"loading {len(urls)} urls(s)...")
                        url_docs = load_urls(urls)
                        all_docs.extend(url_docs)
                        st.write(f"{len(url_docs)} selection(s) loaded from urls")

                    if not all_docs:
                        status.update(label="No content extracted", state="error")
                        st.error("Could not extract content from the sources.")
                    else:
                        st.write("Splitting and embedding content...")
                        chunks = split_documents(all_docs)
                        build_vector_store(chunks, api_key)
                        st.session_state.kb_version += 1
                        _get_graph.clear()
                        status.update(label=f"Knowledge is ready - {len(chunks)} chunks", state="complete")

                except Exception as ex:
                    status.update(label=f"failed", state="error")
                    st.error(f"Error : {ex}")


    if vector_store_exists():
        st.success("Knowledge base is ready")
    else:
        st.info("Knowledge is not yet loaded")
    
    st.divider()
    if st.button("Clear Chat", use_container_width= True):
        st.session_state.messages = []
        st.rerun()

st.title("VERNA AGENTIC RAG ASSISTANT")

# if not api_key:
#     st.info("No open api key")
#     st.stop()

# if not vector_store_exists():
#     st.info("No vector store built")
#     st.stop()

if api_key and vector_store_exists():
    try:
        graph = _get_graph(api_key, selected_model, tavily_key, st.session_state.kb_version, rag_mode)

    except Exception as e:
        st.error(f"Failed to start the agent: {e}")
        st.stop()
else:
    st.info("Please provide an OpenAI API key and build the knowledge base before asking questions.")
    st.stop()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            with st.expander("Agent reasoning", expanded=False):
                for step in msg["steps"]:
                    st.write(step)

if prompt := st.chat_input("Ask anything about your docs"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    history = []

    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Verna Thinking ..........."):
            try:
                result = graph.invoke({
                    "question": prompt,
                    "chat_history" : history,
                    "documents": [],
                    "generation": "",
                    "web_search" : False,
                    "steps": []
                })

                answer = result.get(
                    "generation", "Sorry, I Could not get a response"
                )
                steps: list[str] = result.get("steps", [])

                st.markdown(answer)

                if steps:
                    with st.expander("Agent reasoning", expanded=False):
                        for step in steps:
                            st.write(step)
                    
                    st.session_state.messages.append(
                        {
                            "role": "assistant","content": answer,"steps": steps
                        }
                    )
            except Exception as e:
                st.session_state.messages.append(
                    {
                         "role": "assistant","content": e,"steps": []
                    }
                )
                st.error(e)
