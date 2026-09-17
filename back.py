from typing import List, TypedDict

from pydantic import BaseModel
from langchain_openrouter import ChatOpenRouter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_community.tools.tavily_search import TavilySearchResults

import retriever as retriever_module
from retriever import build_retriever


# =========================================================
# LLM
# =========================================================

llm = ChatOpenRouter(
    model="openrouter/free"
)


# =========================================================
# THRESHOLDS
# =========================================================

UPPER_TH = 0.7
LOWER_TH = 0.3


# =========================================================
# STATE
# =========================================================

class State(TypedDict):
    question: str

    docs: List[Document]
    good_docs: List[Document]

    verdict: str
    reason: str

    web_query: str
    web_docs: List[Document]

    answer: str


# =========================================================
# RETRIEVE
# =========================================================

def retrieve_node(state: State) -> State:

    if retriever_module.retriever is None:
        raise ValueError(
            "Retriever is not initialized. "
            "Upload documents first."
        )

    q = state["question"]

    return {
        "docs": retriever_module.retriever.invoke(q)
    }


# =========================================================
# DOCUMENT EVALUATION
# =========================================================

class DocEvalScore(BaseModel):
    score: float
    reason: str


doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only."
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Chunk:\n{chunk}"
        ),
    ]
)

doc_eval_chain = (
    doc_eval_prompt
    | llm.with_structured_output(DocEvalScore)
)


def eval_each_doc_node(state: State) -> State:

    q = state["question"]
    docs = state.get("docs", [])

    if not docs:
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": "No input documents were provided to evaluate."
        }

    scores = []
    good = []

    for d in docs:

        try:

            out = doc_eval_chain.invoke(
                {
                    "question": q,
                    "chunk": d.page_content
                }
            )

            current_score = (
                out.score if out is not None else 0.0
            )

        except Exception as e:

            print(
                f"Error calling LLM evaluator: {e}"
            )

            current_score = 0.0

        scores.append(current_score)

        if current_score > LOWER_TH:
            good.append(d)

    # CORRECT
    if any(s > UPPER_TH for s in scores):

        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason":
                f"At least one retrieved chunk scored > {UPPER_TH}."
        }

    # INCORRECT
    if all(s < LOWER_TH for s in scores):

        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason":
                f"All retrieved chunks scored < {LOWER_TH}."
        }

    # AMBIGUOUS
    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason":
            f"No chunk scored > {UPPER_TH}, "
            f"but not all were < {LOWER_TH}."
    }


# =========================================================
# WEB SEARCH
# =========================================================

class WebQuery(BaseModel):
    query: str


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query "
            "composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency, add a constraint "
            "like (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query"
        ),
        (
            "human",
            "Question: {question}"
        ),
    ]
)

rewrite_chain = (
    rewrite_prompt
    | llm.with_structured_output(WebQuery)
)


def rewrite_query_node(state: State) -> State:

    out = rewrite_chain.invoke(
        {
            "question": state["question"]
        }
    )

    return {
        "web_query": out.query
    }


tavily = TavilySearchResults(
    max_results=5
)


def web_search_node(state: State) -> State:

    q = (
        state.get("web_query")
        or state["question"]
    )

    results = tavily.invoke(
        {"query": q}
    )

    web_docs = []

    for r in results or []:

        title = r.get("title", "")
        url = r.get("url", "")
        content = (
            r.get("content", "")
            or r.get("snippet", "")
        )

        text = (
            f"TITLE: {title}\n"
            f"URL: {url}\n"
            f"CONTENT:\n{content}"
        )

        web_docs.append(
            Document(
                page_content=text,
                metadata={
                    "url": url,
                    "title": title
                }
            )
        )

    return {
        "web_docs": web_docs
    }


# =========================================================
# GENERATE
# =========================================================

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Answer the user's question using the provided context. "
            "Use only the information available in the context. "
            "If the context does not contain enough information "
            "to answer the question, say that the answer is not "
            "available in the provided context."
        ),
        (
            "human",
            "Context:\n{context}\n\n"
            "Question:\n{question}"
        ),
    ]
)


def generate(state: State) -> State:

    if state["verdict"] == "CORRECT":

        docs_to_use = state.get(
            "good_docs", []
        )

    elif state["verdict"] == "INCORRECT":

        docs_to_use = state.get(
            "web_docs", []
        )

    else:

        docs_to_use = (
            state.get("good_docs", [])
            +
            state.get("web_docs", [])
        )

    context = "\n\n".join(
        d.page_content
        for d in docs_to_use
    ).strip()

    out = (
        answer_prompt
        | llm
    ).invoke(
        {
            "question": state["question"],
            "context": context
        }
    )

    return {
        "answer": out.content
    }


# =========================================================
# ROUTING
# =========================================================

def route_after_eval(state: State) -> str:

    if state["verdict"] == "CORRECT":
        return "generate"

    return "rewrite_query"


# =========================================================
# GRAPH
# =========================================================

g = StateGraph(State)

g.add_node("retrieve", retrieve_node)
g.add_node("eval_each_doc",eval_each_doc_node)
g.add_node("rewrite_query",rewrite_query_node)
g.add_node("web_search",web_search_node)
g.add_node("generate",generate)
g.add_edge(START,"retrieve")
g.add_edge("retrieve","eval_each_doc")
g.add_conditional_edges("eval_each_doc",
    route_after_eval,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
    }
)
g.add_edge("rewrite_query","web_search")
g.add_edge("web_search","generate")
g.add_edge("generate",END)
app = g.compile()