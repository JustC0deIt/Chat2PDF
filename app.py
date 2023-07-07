from langchain.document_loaders import PyMuPDFLoader
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
import os

OPENAI_API_KEY = st.secrets['openai_api_key']
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
collection_name = 'pdf_collection'
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type='stuff')
vectorstore = None

st.title("PDF Chatbot")
persist_index = 'chromadata'
def load_pdf(pdf_path):
    return PyMuPDFLoader(pdf_path).load()

with st.container():
    upload_file = st.file_uploader('Choose a PDF file', type='pdf')
    if upload_file is not None:
        path = os.path.join('.', upload_file.name)
        with open(path, 'wb') as f:
            f.write(upload_file.getbuffer())

        docs = load_pdf(path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_docs = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(split_docs, embeddings, collection_name=collection_name, persist_directory=persist_index)
        vectorstore.persist()

        st.write("Done")

with st.container():
    question = st.text_input("Question")
    if vectorstore is not None and question is not None and question != "":
        doc = vectorstore.similarity_search(question, 5, include_metadata=True)
        answer = chain.run(input_documents=doc, question=question)
        st.write(answer)

