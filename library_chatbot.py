import os
import sys
import streamlit as st

# ==============================
# SQLite (Chroma용) 충돌 방지
# ==============================
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# ==============================
# OpenAI API Key 안전 설정
# ==============================
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("❌ OPENAI_API_KEY가 Streamlit secrets에 설정되어 있지 않습니다.")
    st.stop()

# ==============================
# LangChain imports
# ==============================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)

# ==============================
# Cache functions
# ==============================
@st.cache_resource
def load_and_split_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


@st.cache_resource
def create_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    split_docs = splitter.split_documents(docs)

    return Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="./chroma_db",
    )


@st.cache_resource
def get_vectorstore(docs):
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(
                model="text-embedding-3-small"
            ),
        )
    return create_vector_store(docs)


# ==============================
# RAG Chain 초기화
# ==============================
@st.cache_resource
def initialize_components(selected_model: str):
    file_path = "/mount/src/library_chatbot/[챗봇프로그램및실습] 부경대학교 규정집.pdf"

    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 질문 재구성 프롬프트
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a chat history and the latest user question, "
                "reformulate it into a standalone question.",
            ),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # QA 프롬프트
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an assistant for question-answering tasks.
Use the retrieved context to answer the question.
If you don't know the answer, say you don't know.
Answer in Korean, politely, and use emojis 😊

{context}""",
            ),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model=selected_model)

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(
        history_aware_retriever,
        qa_chain,
    )


# ==============================
# Streamlit UI
# ==============================
st.header("📚 국립부경대 도서관 규정 Q&A 챗봇")

model_option = st.selectbox(
    "GPT 모델 선택",
    ("gpt-4o-mini", "gpt-3.5-turbo-0125"),
)

rag_chain = initialize_components(model_option)

chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

if user_input := st.chat_input("질문을 입력하세요"):
    st.chat_message("human").write(user_input)

    with st.chat_message("ai"):
        with st.spinner("답변 생성 중..."):
            response = conversational_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "default"}},
            )

            st.write(response["answer"])

            with st.expander("📄 참고 문서"):
                for doc in response["context"]:
                    st.markdown(
                        doc.metadata.get("source", "출처 없음"),
                        help=doc.page_content,
                    )
