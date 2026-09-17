import streamlit as st
import tempfile
import os

from back import build_retriever, app


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Document Chat",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

if "document_names" not in st.session_state:
    st.session_state.document_names = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📚 Document Chat")

    st.markdown(
        "Upload one or more PDF documents "
        "and start asking questions."
    )

    # -----------------------------------------------------
    # PDF UPLOADER
    # -----------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    # -----------------------------------------------------
    # SHOW SELECTED FILES
    # -----------------------------------------------------

    if uploaded_files:

        st.write(
            f"**{len(uploaded_files)} document(s) selected**"
        )

        for file in uploaded_files:
            st.write(
                f"📄 {file.name}"
            )

        # -------------------------------------------------
        # PROCESS DOCUMENTS
        # -------------------------------------------------

        if st.button(
            "🚀 Process Documents",
            use_container_width=True
        ):

            with st.spinner(
                "Processing documents..."
            ):

                temp_paths = []

                try:

                    # =====================================
                    # SAVE UPLOADED FILES TEMPORARILY
                    # =====================================

                    for uploaded_file in uploaded_files:

                        temp_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf"
                        )

                        temp_file.write(
                            uploaded_file.getbuffer()
                        )

                        temp_file.close()

                        temp_paths.append(
                            temp_file.name
                        )

                    # =====================================
                    # BUILD FAISS RETRIEVER
                    # =====================================

                    num_pages, num_chunks = (
                        build_retriever(
                            temp_paths
                        )
                    )

                    # =====================================
                    # UPDATE SESSION STATE
                    # =====================================

                    st.session_state.documents_loaded = True

                    st.session_state.document_names = [
                        file.name
                        for file in uploaded_files
                    ]

                    # New document collection
                    # means new conversation
                    st.session_state.messages = []

                    # =====================================
                    # SUCCESS MESSAGE
                    # =====================================

                    st.success(
                        f"Processed {len(uploaded_files)} "
                        f"document(s) successfully."
                    )

                    st.info(
                        f"📄 Pages: {num_pages}\n\n"
                        f"🧩 Chunks: {num_chunks}"
                    )

                except Exception as e:

                    st.error(
                        f"Error processing documents:\n\n{e}"
                    )

                finally:

                    # =====================================
                    # DELETE TEMPORARY FILES
                    # =====================================

                    for path in temp_paths:

                        if os.path.exists(path):

                            os.remove(path)

    # =====================================================
    # SIDEBAR DIVIDER
    # =====================================================

    st.divider()

    # =====================================================
    # LOADED DOCUMENTS
    # =====================================================

    if st.session_state.documents_loaded:

        st.subheader(
            "📂 Loaded Documents"
        )

        for name in st.session_state.document_names:

            st.write(
                f"📄 {name}"
            )

        st.divider()

        # =================================================
        # CLEAR CHAT
        # =================================================

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()


# =========================================================
# MAIN UI
# =========================================================

st.title(
    "📖 Corrective RAG"
)

st.caption(
    "Upload documents and ask questions using "
    "document context and web search."
)


# =========================================================
# DOCUMENT CHECK
# =========================================================

if not st.session_state.documents_loaded:

    st.info(
        "👈 Upload one or more PDF documents "
        "from the sidebar to begin."
    )

    st.stop()


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        # -------------------------------------------------
        # USER MESSAGE
        # -------------------------------------------------

        if message["role"] == "user":

            st.markdown(
                message["content"]
            )

        # -------------------------------------------------
        # ASSISTANT MESSAGE
        # -------------------------------------------------

        else:

            verdict = message.get(
                "verdict",
                ""
            )

            # =============================================
            # SHOW VERDICT
            # =============================================

            if verdict == "CORRECT":

                st.success(
                    "📚 Answer retrieved from the "
                    "given document context."
                )

            elif verdict == "INCORRECT":

                st.warning(
                    "🌐 Answer retrieved from web context "
                    "because the provided document context "
                    "is not relevant."
                )

            elif verdict == "AMBIGUOUS":

                st.info(
                    "🔀 Answer generated using both the "
                    "given document context and web context "
                    "because the provided context was not "
                    "sufficient."
                )

            # =============================================
            # SHOW ANSWER
            # =============================================

            st.markdown(
                message["content"]
            )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask something about your documents..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # =====================================================
    # SAVE USER MESSAGE
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # =====================================================
    # DISPLAY USER MESSAGE
    # =====================================================

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # =====================================================
    # ASSISTANT RESPONSE
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            try:

                # =========================================
                # RUN LANGGRAPH BACKEND
                # =========================================

                result = app.invoke(
                    {
                        "question": question,

                        "docs": [],
                        "good_docs": [],

                        "verdict": "",
                        "reason": "",

                        "web_query": "",
                        "web_docs": [],

                        "answer": ""
                    }
                )

                # =========================================
                # GET RESULT
                # =========================================

                answer = result.get(
                    "answer",
                    "No answer was generated."
                )

                verdict = result.get(
                    "verdict",
                    ""
                )

                reason = result.get(
                    "reason",
                    ""
                )

                # =========================================
                # SHOW VERDICT
                # =========================================

                if verdict == "CORRECT":

                    st.success(
                        "📚 Answer retrieved from the "
                        "given document context."
                    )

                elif verdict == "INCORRECT":

                    st.warning(
                        "🌐 Answer retrieved from web context "
                        "because the provided document context "
                        "is not relevant."
                    )

                elif verdict == "AMBIGUOUS":

                    st.info(
                        "🔀 Answer generated using both the "
                        "given document context and web context "
                        "because the provided context was not "
                        "sufficient."
                    )

                # =========================================
                # SHOW ANSWER
                # =========================================

                st.markdown(
                    answer
                )

                # =========================================
                # OPTIONAL: SHOW REASON
                # =========================================

                with st.expander(
                    "🔎 Retrieval details"
                ):

                    st.write(
                        f"**Verdict:** {verdict}"
                    )

                    st.write(
                        f"**Reason:** {reason}"
                    )

                # =========================================
                # SAVE ASSISTANT MESSAGE
                # =========================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "verdict": verdict,
                        "reason": reason
                    }
                )

            # =============================================
            # ERROR HANDLING
            # =============================================

            except Exception as e:

                error_message = (
                    "Something went wrong:\n\n"
                    f"{e}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "verdict": "ERROR",
                        "reason": str(e)
                    }
                )