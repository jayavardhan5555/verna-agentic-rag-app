from __future__ import annotations

import operator
from typing import Annotated, List, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END , StateGraph


RAGMode = Literal["naive", "corrective", "adaptive"]

# GRAPH STATE 
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: bool
    documents: List[Document]
    chat_history: List[BaseMessage]
    steps: Annotated[List[str], operator.add]

### STRAUCTUTRED OUTPUT SCHEMAS

class RouteQuery(BaseModel):
    """Route a query to the appropriate datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        description=(
            "Use 'vectorstore' for questions about the uploaded documents. "
            "Use 'web_search' for current events, general knowledge, or anything not covered by documents."
        )
    )

class GradeDocuments(BaseModel):
    """Binary relevance score for single retrived document"""
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question , 'no' otherwise"
    )



### SHARED METHODS
def _make_rag_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system",
            "You are a helpful AI assistant "
            "Use only the follwing retrived context to answer the question."
            "If the answer is not present in the context , say so and answer from your general knowledge."
            "Be clear and concise. \n\n{context}"
            ),
              MessagesPlaceholder(variable_name="chat_history",optional=True),
            ("human", "{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()

def _make_retrieval_grader(llm:ChatOpenAI):
    grader_llm = llm.with_structured_output(GradeDocuments)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a grader assessing weather a retrived document is relevant to the user question. Give a binary 'yes' or 'no'"
        ),
        ("human", "Document:\n\n{document}\n\nQuestion:{question}")
    ])
    return prompt | grader_llm

def _make_question_router(llm:ChatOpenAI):
    router_llm = llm.with_structured_output(RouteQuery)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert at routing user questions to the right data source.\n"
            "Route to 'vectorstore' when the question is about specific topics present in the user uploaded documents. "
            "Route to 'web_search' when the question is outside the uploaded documents or general events."
        ),
        ("human", "{question}")
    ])
    return prompt | router_llm

def _make_question_rewriter(llm:ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a query optimzation expert."
            "rewrite the user question to improve retrival or web search results."
            "Return only improvised question , no explanation"
        ),
        ("human", "Original Question:{question}")
    ])
    return prompt | llm | StrOutputParser()


### NAIVE RAG
def _build_naive_rag(llm:ChatOpenAI, retriver):
    """START -> retrive -> generate -> END"""

    rag_chain = _make_rag_chain(llm)

    def retrive(state: GraphState) -> dict:
        docs = retriver.invoke(state["question"])
        return {
            "documents" : docs,
            "steps": [f" [Naive] retrived {len(docs)} chunks from vectore store"]
        }


    def generate(state: GraphState) -> dict:
        context = "\n\n".join(
            d.page_content for d in (state.get("documents") or [])
        )

        answer = rag_chain.invoke(
            {
                "context": context or "No context available",
                "question": state["question"],
                "chat_history": state.get("chat_history") or []
            }
        )

        return {"generation": answer , "steps": ["Generated answer"]}
    
    wf = StateGraph(GraphState)
    wf.add_node("retrive", retrive)
    wf.add_node("generate", generate)
    wf.add_edge(START, "retrive")
    wf.add_edge("retrive","generate")
    wf.add_edge("generate", END)
    return wf.compile()

#Build Corrective RAG

def build_corrective_rag(llm, retriver,tavily_tool):

    rag_chain = _make_rag_chain(llm)
    retrival_grader = _make_retrieval_grader(llm)

    def retrive(state: GraphState) -> dict:
        docs = retriver.invoke(state["question"])
        return {
            "documents" : docs,
            "steps": [f" [Naive] retrived {len(docs)} chunks from vectore store"]
        }

    def grade_documents(state:GraphState) -> dict:
        relevant: List[Document] = []
        for doc in state["documents"]:
            grade = retrival_grader.invoke(
                {"document": doc.page_content , "question" : state["question"]}
            )
            if grade.binary_score == "yes":
                relevant.append(doc)
        need_web = len(relevant) == 0
        note = (
            f"{len(relevant)}/{len(state['documents'])} chunk(s) relevant" + (" -- switching to web search" if need_web else "")
        )
        return {"documents": relevant, "web_search": need_web, "steps": [note]}
    

    def web_search_node(state:GraphState) -> dict:
        if tavily_tool is None:
            return {"steps": ["Web search skipped (no tavily key provided)"]}
        
        raw = tavily_tool.invoke({"query": state["question"]})
        results = raw if isinstance(raw,list) else []
        web_docs = [
            Document(
                page_content=r.get("content", ""),
                metadate={"source": r.get("url","web")},
            )
            for r in results
            if r.get("content")
        ]
        return {
            "documents": web_docs,
            "steps" : [f"Web search results returned"]
        }
    def generate(state: GraphState) -> dict:
        context = "\n\n".join(
            d.page_content for d in (state.get("documents") or [])
        )

        answer = rag_chain.invoke(
            {
                "context": context or "No context available",
                "question": state["question"],
                "chat_history": state.get("chat_history") or []
            }
        )

        return {"generation": answer , "steps": ["Generated answer"]}
    
    def _after_grading(state:GraphState) -> str:
        return "web_search" if state.get("web_search") else "generate"
    

    wf = StateGraph(GraphState)
    wf.add_node("retrive", retrive)
    wf.add_node("grade_documents", grade_documents)
    wf.add_node("web_search", web_search_node)
    wf.add_node("generate", generate)
    wf.add_edge(START, "retrive")
    wf.add_edge("retrive", "grade_documents")
    wf.add_conditional_edges(
        "grade_documents",
        _after_grading,
        {"generate": "generate", "web_search": "web_search"}
    )
    wf.add_edge("web_search","generate")
    wf.add_edge("generate", END)
    return wf.compile()

def build_adaptive_rag(llm,retriver,tavily_tool):

    rag_chain = _make_rag_chain(llm)
    retrival_grader = _make_retrieval_grader(llm)
    question_router = _make_question_router(llm)
    question_rewriter = _make_question_rewriter(llm)

    def routequestion(state:GraphState) -> dict:
        result = question_router.invoke({"question": state["question"]})
        use_web = result.datasource == "web_search"
        label = "Routing web search " if use_web else " Routing Vector store"
        return {"web_search": use_web, "steps" :[label]} 


    def retrive(state: GraphState) -> dict:
        docs = retriver.invoke(state["question"])
        return {
            "documents" : docs,
            "steps": [f" [Naive] retrived {len(docs)} chunks from vectore store"]
        }

    def grade_documents(state:GraphState) -> dict:
        relevant: List[Document] = []
        for doc in state["documents"]:
            grade = retrival_grader.invoke(
                {"document": doc.page_content , "question" : state["question"]}
            )
            if grade.binary_score == "yes":
                relevant.append(doc)
        need_web = len(relevant) == 0
        note = (
            f"{len(relevant)}/{len(state['documents'])} chunk(s) relevant" + (" -- switching to web search" if need_web else "")
        )
        return {"documents": relevant, "web_search": need_web, "steps": [note]}
    
    def transform_query(state:GraphState) -> dict:
        better_q = question_rewriter.invoke({"question": state["question"]})

        return{
            "question": better_q,
            "steps": [f"Rewrote query : \"{better_q}\""]
        }
    

    def web_search_node(state:GraphState) -> dict:
        if tavily_tool is None:
            return {"steps": ["Web search skipped (no tavily key provided)"]}
        
        raw = tavily_tool.invoke({"query": state["question"]})
        results = raw if isinstance(raw,list) else []
        web_docs = [
            Document(
                page_content=r.get("content", ""),
                metadate={"source": r.get("url","web")},
            )
            for r in results
            if r.get("content")
        ]
        return {
            "documents": web_docs,
            "steps" : [f"Web search results returned"]
        }
    def generate(state: GraphState) -> dict:
        context = "\n\n".join(
            d.page_content for d in (state.get("documents") or [])
        )

        answer = rag_chain.invoke(
            {
                "context": context or "No context available",
                "question": state["question"],
                "chat_history": state.get("chat_history") or []
            }
        )

        return {"generation": answer , "steps": ["Generated answer"]}
    
    def _after_grading(state:GraphState) -> str:
        return "transform_query" if state.get("web_search") else "generate"
    
    def _after_routing(state:GraphState) -> str:
        return "web_search" if state.get("web_search") else "retrive"
    

    wf = StateGraph(GraphState)
    wf.add_node("routequestion",routequestion)
    wf.add_node("retrive", retrive)
    wf.add_node("grade_documents", grade_documents)
    wf.add_node("web_search", web_search_node)
    wf.add_node("transform_query", transform_query)
    wf.add_node("generate", generate)
    wf.add_edge(START, "routequestion")
    wf.add_conditional_edges("routequestion", _after_routing , {"web_search": "web_search" , "retrive": "retrive"})
    wf.add_edge("retrive", "grade_documents")
    wf.add_conditional_edges(
        "grade_documents",
        _after_grading,
        {"generate": "generate", "transform_query": "transform_query"}
    )
    wf.add_edge("transform_query", "web_search")
    wf.add_edge("web_search","generate")
    wf.add_edge("generate", END)
    return wf.compile()

def build_graph(mode: RAGMode, llm: ChatOpenAI, retriver, tavily_tool):

    if mode == "naive":
        return _build_naive_rag(llm,retriver)
    if mode == "corrective":
        return build_corrective_rag(llm,retriver,tavily_tool)
    return build_adaptive_rag(llm,retriver,tavily_tool)