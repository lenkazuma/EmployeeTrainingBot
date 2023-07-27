from dotenv import load_dotenv
import streamlit as st
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.mapreduce import MapReduceChain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from docx import Document
from docx.table import _Cell
import glob

def get_pdf_text(files):
    text = ""
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def main():
    # brief summary
    llm = OpenAI()
    chain = load_summarize_chain(llm, chain_type="stuff")
    chain_large = load_summarize_chain(llm, chain_type="map_reduce")
    chain_qa = load_qa_chain(llm, chain_type="stuff")
    chain_large_qa = load_qa_chain(llm, chain_type="map_reduce")


    load_dotenv()
    st.set_page_config(page_title="EEC Training", page_icon=":books:")
    st.title("✨ Responsible Person Introduction Internal Training ✨")
    
    # upload file
    #uploaded_file  = 'Responsible Person Introduction Internal Training Document.pdf'
    #uploaded_file  = '1903.08057.pdf'

    # returns all file paths that has .pdf as extension in the specified directory
    pdf_search = glob.glob("*.pdf")


    # Clear summary if a new file is uploaded
    if 'summary' in st.session_state and st.session_state.file_name not in pdf_search:
        st.session_state.summary = None
        
    st.session_state.file_name = pdf_search[0]
    

    # Handle PDF files
    text = get_pdf_text(pdf_search)

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    chunks = text_splitter.split_text(text)
            

    # create embeddings
    embeddings = OpenAIEmbeddings(disallowed_special=())
    knowledge_base = FAISS.from_texts(chunks, embeddings)

            
    st.header("About: ")
    pdf_summary = "Give me a concise summary"

    
    docs = knowledge_base.similarity_search(pdf_summary)
    #st.write(docs)   
            
    if 'summary' not in st.session_state or st.session_state.summary is None:
        with st.spinner('Wait for it...'):
            try:
                st.session_state.summary = chain.run(input_documents=docs, question=pdf_summary)
            except Exception as maxtoken_error:
            # Fallback to the larger model if the context length is exceeded
                print(maxtoken_error)
                print("pin0")
                st.session_state.summary = chain_large.run(input_documents=docs, question=pdf_summary)
                print("pin1")
    st.write(st.session_state.summary)


            # show user input
    user_question = st.text_input("Ask a question about the training :")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        with st.spinner('Wait for it...'):
            with get_openai_callback() as cb:
                try:
                    response = chain_qa.run(input_documents=docs, question=user_question)
                    
                except Exception as maxtoken_error:
                    print(maxtoken_error)
                    response = chain_large_qa.run(input_documents=docs, question=user_question) 
                print(cb)
            # show/hide section using st.beta_expander
            with st.expander("Used Tokens", expanded=False):
                st.write(cb)
        st.write(response)
 


if __name__ == '__main__':
    main()
