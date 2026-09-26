from __future__ import annotations

import traceback
from pathlib import Path

import streamlit as st

from rag.ingestion import DocumentIngestion
from rag.pipeline import RAGPipeline
from agents.orchestrator import AgentOrchestrator


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic RAG AI",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Agentic RAG AI")

st.caption(
    "RAG + Hybrid Retrieval + Agents + Critic + "
    "Grounded Answer Generation"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

RAG_FILE = DATA_DIR / "rag_notes.txt"


# ============================================================
# RAG INITIALIZATION
# ============================================================

@st.cache_resource(show_spinner=False)
def build_rag():

    # --------------------------------------------------------
    # CHECK KNOWLEDGE BASE
    # --------------------------------------------------------

    if not RAG_FILE.exists():

        raise FileNotFoundError(
            f"Knowledge base not found:\n{RAG_FILE}"
        )

    # --------------------------------------------------------
    # INGESTION
    # --------------------------------------------------------

    ingestion = DocumentIngestion(
        chunk_size=500,
        overlap=100,
    )

    documents = ingestion.load(
        RAG_FILE
    )

    if not documents:

        raise ValueError(
            "No documents were loaded from:\n"
            f"{RAG_FILE}"
        )

    # --------------------------------------------------------
    # RAG PIPELINE
    # --------------------------------------------------------

    pipeline = RAGPipeline()

    # --------------------------------------------------------
    # RETRIEVER
    # --------------------------------------------------------

    retriever = pipeline.retriever

    if retriever is None:

        raise RuntimeError(
            "RAGPipeline did not create a retriever."
        )

    # --------------------------------------------------------
    # BUILD INDEX
    # --------------------------------------------------------

    retriever.build(
        documents
    )

    return retriever


# ============================================================
# INITIALIZE RAG
# ============================================================

try:

    with st.spinner(
        "Building RAG index..."
    ):

        retriever = build_rag()

    st.success(
        "RAG index ready."
    )

except Exception as exc:

    st.error(
        "RAG initialization failed."
    )

    st.code(
        traceback.format_exc(),
        language="text",
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System"
    )

    st.success(
        "RAG: READY"
    )

    st.write(
        "Knowledge base:"
    )

    st.code(
        str(RAG_FILE)
    )

    st.write(
        "Agent:"
    )

    st.code(
        "AgentOrchestrator"
    )

    st.write(
        "Retrieval:"
    )

    st.code(
        "Hybrid"
    )


# ============================================================
# QUERY INPUT
# ============================================================

st.subheader(
    "Ask the Agent"
)

query = st.text_area(
    "Question",
    placeholder="What is RAG?",
    height=120,
)


# ============================================================
# RUN AGENT
# ============================================================

run_agent = st.button(
    "🚀 Run Agent",
    type="primary",
    use_container_width=True,
)


if run_agent:

    # --------------------------------------------------------
    # VALIDATE QUERY
    # --------------------------------------------------------

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # --------------------------------------------------------
    # CREATE ORCHESTRATOR
    # --------------------------------------------------------

    try:

        orchestrator = AgentOrchestrator(
            retriever=retriever
        )

    except Exception:

        st.error(
            "Failed to initialize AgentOrchestrator."
        )

        st.code(
            traceback.format_exc(),
            language="text",
        )

        st.stop()

    # --------------------------------------------------------
    # EXECUTE
    # --------------------------------------------------------

    with st.spinner(
        "Agent is executing..."
    ):

        try:

            result = orchestrator.run(
                query.strip()
            )

        except Exception:

            st.error(
                "Agent execution failed."
            )

            st.code(
                traceback.format_exc(),
                language="text",
            )

            st.stop()

    # --------------------------------------------------------
    # RESULT VALIDATION
    # --------------------------------------------------------

    if result is None:

        st.error(
            "Agent returned None."
        )

        st.stop()

    if not isinstance(
        result,
        dict,
    ):

        st.error(
            "Agent returned an unexpected result type."
        )

        st.code(
            repr(result)
        )

        st.stop()

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 Answer"
    )

    answer = result.get(
        "answer",
        "No answer returned.",
    )

    st.markdown(
        answer
    )

    # ========================================================
    # METRICS
    # ========================================================

    st.divider()

    st.subheader(
        "Agent Metrics"
    )

    col1, col2, col3, col4 = st.columns(4)

    grounded = result.get(
        "grounded",
        False,
    )

    confidence = result.get(
        "confidence",
        0.0,
    )

    evidence = result.get(
        "evidence",
        [],
    )

    attempts = result.get(
        "attempt",
        1,
    )

    with col1:

        st.metric(
            "Grounded",
            "YES" if grounded else "NO",
        )

    with col2:

        try:
            confidence_value = float(
                confidence
            )
        except Exception:
            confidence_value = 0.0

        st.metric(
            "Confidence",
            f"{confidence_value:.2f}",
        )

    with col3:

        evidence_count = (
            len(evidence)
            if isinstance(
                evidence,
                list,
            )
            else 0
        )

        st.metric(
            "Evidence",
            evidence_count,
        )

    with col4:

        st.metric(
            "Attempts",
            attempts,
        )

    # ========================================================
    # TOOLS
    # ========================================================

    st.subheader(
        "🔧 Tools Used"
    )

    tools_used = result.get(
        "tools_used",
        [],
    )

    if tools_used:

        for tool in tools_used:

            st.write(
                f"• `{tool}`"
            )

    else:

        st.info(
            "No tools reported."
        )

    # ========================================================
    # CITATIONS
    # ========================================================

    st.subheader(
        "📚 Citations"
    )

    citations = result.get(
        "citations",
        [],
    )

    if citations:

        st.write(
            citations
        )

    else:

        st.info(
            "No citations returned."
        )

    # ========================================================
    # EVIDENCE
    # ========================================================

    st.subheader(
        "🔎 Retrieved Evidence"
    )

    if isinstance(
        evidence,
        list,
    ) and evidence:

        for index, item in enumerate(
            evidence,
            start=1,
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            source = item.get(
                "source",
                "unknown",
            )

            page = item.get(
                "page",
                0,
            )

            score = item.get(
                "score",
                0.0,
            )

            text = item.get(
                "text",
                "",
            )

            with st.expander(
                f"Evidence {index} — {source}"
            ):

                st.write(
                    f"**Source:** {source}"
                )

                st.write(
                    f"**Page:** {page}"
                )

                st.write(
                    f"**Score:** {score}"
                )

                st.write(
                    text
                )

    else:

        st.warning(
            "No evidence returned."
        )

    # ========================================================
    # CRITIC
    # ========================================================

    st.subheader(
        "🧠 Critic Evaluation"
    )

    critic = result.get(
        "critic",
        {},
    )

    if isinstance(
        critic,
        dict,
    ):

        critic_supported = critic.get(
            "supported",
            False,
        )

        critic_confidence = critic.get(
            "confidence",
            0.0,
        )

        if critic_supported:

            st.success(
                "Critic: Evidence supports the answer."
            )

        else:

            st.warning(
                "Critic: Evidence does not fully support the answer."
            )

        st.write(
            f"**Critic confidence:** "
            f"{critic_confidence}"
        )

        issues = critic.get(
            "issues",
            [],
        )

        if issues:

            st.write(
                "**Issues:**"
            )

            for issue in issues:

                st.write(
                    f"• {issue}"
                )

    else:

        st.info(
            "No critic information returned."
        )

    # ========================================================
    # FULL DEBUG RESULT
    # ========================================================

    with st.expander(
        "🛠 Full Agent Result"
    ):

        st.json(
            result
        )