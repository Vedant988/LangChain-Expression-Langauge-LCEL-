import os
import tempfile
import hashlib
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain


GOOGLE_API_KEY = ""
PINECONE_API_KEY = ("")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "react-memory-basics"
MEMORY_FILE = "memory.txt"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


pc = Pinecone(api_key=PINECONE_API_KEY)
existing = [idx.name for idx in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )


def load_memory() -> str:
    if os.path.exists(MEMORY_FILE):
        return open(MEMORY_FILE, "r", encoding="utf-8").read()
    return ""

def update_memory(user_q: str, assistant_a: str):
    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"You: {user_q}\nGemini: {assistant_a}\n\n")


def process_pdf(file)->list:
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    loader=PyPDFLoader(tmp_path)
    docs=loader.load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    return splitter.split_documents(docs)

def upsert_chunks(chunks: list, namespace: str):
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    return LangchainPinecone.from_documents(
        documents=chunks,
        embedding=embedder,
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace,
    )


@st.cache_resource
def create_chain(namespace: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vect = LangchainPinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedder,
        namespace=namespace,
    )
    retriever = vect.as_retriever()

    prompt_template = """
Use the following prior conversation and document context to answer the current question.
If the prior memory is helpful, use it; otherwise rely only on the document context.
If the answer is not found, say "Answer not found in the document."

Prior Chat Memory:
{memory}

Document Context:
{context}

Current Question:
{question}

Answer:
"""
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["memory", "context", "question"]
    )
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

    def run_chain(inputs: dict) -> dict:
        user_q = inputs["question"]
        mem_text = load_memory()
        docs = retriever.get_relevant_documents(user_q)
        answer = qa_chain.run(
            input_documents=docs,
            memory=mem_text,
            question=user_q,
            context="\n\n".join([d.page_content for d in docs]),
        )
        update_memory(user_q, answer)
        return {"answer": answer}

    return run_chain


st.set_page_config(page_title="Gemini PDF Chat with Memory", layout="centered")
st.title("Chat with PDF using Gemini + Pinecone + ReAct Memory")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    h = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:12]
    if st.session_state.get("last_file_hash") != h:
        with st.spinner("Indexing PDF into Pinecone..."):
            chunks = process_pdf(uploaded_file)
            upsert_chunks(chunks, namespace=h)
            st.session_state["last_file_hash"] = h
            st.session_state["chat_history"] = []
            st.success(f"Indexed {len(chunks)} chunks under namespace {h}")

    st.markdown("---")
    chat_chain = create_chain(namespace=h)

    user_q = st.text_input("Ask a question from the PDF:")
    if user_q:
        with st.spinner("Thinking..."):
            try:
                res = chat_chain({"question": user_q})
                st.session_state.chat_history.append(("You", user_q))
                st.session_state.chat_history.append(("Gemini", res["answer"]))
            except Exception as e:
                st.error(f"Error: {e}")

    for role,msg in st.session_state.chat_history:
        emoji = "🧑‍💬" if role == "You" else "🤖"
        st.markdown(f"**{emoji} {role}:** {msg}")

else:
    st.info("Please upload a PDF to get started.")