import os
import tempfile
import hashlib
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory

GOOGLE_API_KEY = ""
PINECONE_API_KEY = ""
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "memory-lcel"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
index_list = [index.name for index in pc.list_indexes()]
if PINECONE_INDEX_NAME not in index_list:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
    )

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

def file_hash(file):
    return hashlib.sha256(file.read()).hexdigest()

def get_vectorstore(chunks, namespace):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    return LangchainPinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace
    )

@st.cache_resource
def create_chain(namespace):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectorstore = LangchainPinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace
    )
    retriever = vectorstore.as_retriever()

    prompt_template = """
Use the following context to answer the question in detail. If the answer is not in the context, say "Answer not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

    def run_chain(inputs):
        question = inputs["question"]
        docs = retriever.get_relevant_documents(question)
        answer = qa_chain.run(input_documents=docs, question=question)
        return {"answer": answer}

    return run_chain

st.set_page_config(page_title="Gemini PDF Chat", layout="centered")
st.title("Chat with PDF using Gemini+Pinecone+Memory")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    file_id = file_hash(uploaded_file)
    uploaded_file.seek(0)  
    namespace = file_id[:12] 

    if "last_file_id" not in st.session_state or st.session_state.last_file_id != file_id:
        with st.spinner("Indexing new PDF..."):
            chunks = process_pdf(uploaded_file)
            get_vectorstore(chunks, namespace=namespace)
            st.session_state.last_file_id = file_id
            st.session_state.chat_history = []
            st.success(f"Indexed {len(chunks)} chunks.")

    st.markdown("---")
    chat_chain = create_chain(namespace)

    user_input = st.text_input("Ask a question from the uploaded PDF:")
    if user_input:
        with st.spinner("Thinking..."):
            try:
                result = chat_chain({"question": user_input})
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Gemini", result["answer"]))
            except Exception as e:
                st.error(f"Error:{str(e)}")

    for role, msg in st.session_state.chat_history:
        emoji = "🧑‍💬" if role == "You" else "🤖"
        st.markdown(f"**{emoji} {role}:** {msg}")
