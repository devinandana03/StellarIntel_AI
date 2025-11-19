
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# ------------------ Load the RAG chain (cached) ------------------
@st.cache_resource
def load_astroai():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./nasa_chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # perfect balance

    
    llm = OllamaLLM(model="llama3.2", temperature=0.4)  # ← if you prefer

    system_prompt = (
        "You are AstroAI, a NASA,Appolo,James Webb Space Telescope (JWST),and Voyager Program expert."
        "Answer using only the Wikipedia context provided. "
        "Be exciting, clear, and keep answers short (3-4 sentences max)."
        " If you don't know, say you don't know."
        "\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain

rag_chain = load_astroai()

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="StellarIntel", page_icon="rocket", layout="centered")

st.title("StellarIntel")
st.caption("A RAG-powered space mission assistant using knowledge sourced from Wikipedia.")

# Welcome message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! StellarIntel here. Ask me anything about NASA, Apollo, Voyager, JWST, or space missions!"}
    ]

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if question := st.chat_input("Ask your question here!"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": question})
            answer = response["answer"]
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})