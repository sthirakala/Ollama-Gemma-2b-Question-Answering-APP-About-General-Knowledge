import os

os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; LangChainBot/1.0; +https://yourwebsite.example)"
os.environ["LANGCHAIN_TRACING_V2"] = "false"  
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import WebBaseLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Main_Page")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstoredb = FAISS.from_documents(doc_chunks, embeddings)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Respond to the question clearly and helpfully."),
    ("user", "Question: {question}")
])

llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt_template | llm | output_parser

st.title("Gemma:2b Question Answering App About General Knowledge")
input_text = st.text_input("Ask your question:")

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)


