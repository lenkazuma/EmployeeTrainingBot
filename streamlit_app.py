import streamlit as st
import streamlit.components.v1 as components
from langchain.vectorstores import Chroma
from langchain.embeddings import QianfanEmbeddingsEndpoint
from langchain.llms import QianfanLLMEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

llm = QianfanLLMEndpoint(
    streaming=True, 
    model="ERNIE-Bot",
    endpoint="eb-instant"
    )


# chunk the data
def chunk_data(data, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    return chunks

# Create embeddings in chroma db
def create_embeddings(chunks):
    print("Embedding to Chroma DB...")
    embeddings = QianfanEmbeddingsEndpoint()
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print("Done")
    return vector_store

def ask_with_memory(vector_store, question, chat_history=[], document_description=""):
    retriever = vector_store.as_retriever( # the vs can return documents
    search_type='similarity', search_kwargs={'k': 3})
    general_system_template = f""" 
    You are an assistant named Ernie. You are examining a document. Use only the heading and piece of context to answer the questions at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Do not add any observations or comments. Answer only in Chinese.
    ----
    HEADING: ({document_description})
    CONTEXT: {{context}}
    ----
    """
    general_user_template = "Here is the next question, remember to only answer if you can from the provided context. Only respond in Chinese. QUESTION:```{question}```"
    messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )

    #st.write(qa_prompt)
    #st.write(retriever)
    crc = ConversationalRetrievalChain.from_llm(llm, retriever, combine_docs_chain_kwargs={'prompt': qa_prompt})
    result = crc({'question': question, 'chat_history': chat_history})
    return result

# def ask_for_document_summary(vector_store, question,document_description=""):
#     prompt_template = f""" 
#     You are an assistant named Ernie. You are examining a document. Use only the heading and piece of context to do the summary.  Answer only in Chinese.
#     ----
#     HEADING: ({document_description})
#     CONTEXT: {{context}}
#     ----
#     """

#     llm = QianfanLLMEndpoint(
#         streaming=True, 
#         model="ERNIE-Bot",
#         endpoint="eb-instant",
#         )
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain_type_kwargs = {"prompt": prompt, "verbose":True}
#     retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})
#     qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,chain_type_kwargs=chain_type_kwargs)
#     document_summary=qa.run(question)
#     st.write(document_summary)
#     return document_summary


def ask_for_summary(vector_store, chat_history=[], document_description=""):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
    llm = QianfanLLMEndpoint(
        streaming=True, 
        model="ERNIE-bot")
    retriever = vector_store.as_retriever( # the vs can return documents
    search_type='similarity', search_kwargs={'k': 3})
    
    general_system_template = f""" 
    You are an assistant named Ernie. You are examining a document and the previous chat history. Use only the heading and piece of context to answer the questions at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Do not add any observations or comments. Answer only in Chinese.
    ----
    HEADING: ({document_description})
    CONTEXT: {{context}}
    ----
    """
    general_user_template = "Here is the chat history ```{chat_history}```, do a conversation summary based on the chat history and the document. Remember to only answer if you can from the provided context. Only respond in Chinese. "
    messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )

    crc = ConversationalRetrievalChain.from_llm(llm, retriever, combine_docs_chain_kwargs={'prompt': qa_prompt})
    summary = crc({'question': "Give me a summary of the conversation", 'chat_history': chat_history})
    return summary

def clear_history():
    if "history" in st.session_state:
        del st.session_state["history"]

def format_chat_history(chat_history):
    formatted_history = ""
    for entry in chat_history:
        question, answer = entry
        # Added an extra '\n' for the blank line
        formatted_history += f"问题: {question}\n回答: {answer}\n\n"
    return formatted_history



if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
    st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    )
    #st.subheader("万科企业股份有限公司2023年第一季度报告")
    st.subheader("比亚迪ATTO3车型说明书助手")
    
    #st.session_state.document_description = "万科企业股份有限公司2023年第一季度报告"
    st.session_state.document_description = "比亚迪ATTO3车型说明书助手"
    st.session_state.chat_context_length = 10
    if "data" not in st.session_state:
        #loader = PyPDFLoader("http://static.cninfo.com.cn/finalpage/2023-04-29/1216686497.PDF")
        loader = PyPDFLoader("https://bydautomotive.com.au/brochures/BYD-ATTO-3-Owners-Handbook-2022.pdf")
        st.session_state.data = loader.load()

    
    chunks = chunk_data(st.session_state.data, 384)
    st.session_state.vector_store = create_embeddings(chunks)

    # if "summary" not in st.session_state:
    #     #st.session_state.summary = []
    #     pdf_summary = "Give me a concise summary of the document, only respond in Chinese. "
    #     st.session_state.summary = ask_for_document_summary(st.session_state["vector_store"],pdf_summary,st.session_state.document_description)
    #     st.write(st.session_state.summary)
    # else:
    #     st.write(st.session_state.summary)
    
    # Create the placeholder for chat history
    chat_history_placeholder = st.empty()


    if "history" not in st.session_state:
        st.session_state.history = []
        chat_history_placeholder.text_area(label="你好，我是文心智能助理Ernie。请问你有什么问题呢？", value="", height=400)
    else:
        chat_history_placeholder.text_area(label="你好，我是文心智能助理Ernie。请问你有什么问题呢？", value=format_chat_history(st.session_state.history)  , height=400)

    # User input for the question
    with st.form(key="myform", clear_on_submit=True):
        q = st.text_input("请输入你的问题：", key="user_question")
        submit_button = st.form_submit_button("提交问题")
    
    col1, col2, col3,col4, col5 = st.columns(5)
    with col1:
        pass
    with col2:
        pass
    with col3 :
        end_button = st.button("结束对话")
    with col4 :
        pass
    with col5 :
        pass
    # If user en
    # tered a question
    if submit_button:
        if "vector_store" in st.session_state:
            vector_store = st.session_state["vector_store"]
            result = ask_with_memory(vector_store, q, st.session_state.history, st.session_state.document_description)
            # If there are n or more messages, remove the first element of the array
            if len(st.session_state.history) >= st.session_state.chat_context_length:
                st.session_state.history = st.session_state.history[1:]

            st.session_state.history.append((q, result['answer']))

            # Create formatted string to show user, removing the inserted phrase
            chat_history_str = format_chat_history(st.session_state.history)            

            # Update the chat history in the placeholder as a text area
            chat_history_placeholder.text_area(label="你好，我是文心智能助理Ernie。请问你有什么问题呢？",value=chat_history_str, height=400)

            # JavaScript code to scroll the text area to the bottom
            js = f"""
            <script>
                function scroll(dummy_var_to_force_repeat_execution){{
                    var textAreas = parent.document.querySelectorAll('.stTextArea textarea');
                    for (let index = 0; index < textAreas.length; index++) {{
                        textAreas[index].scrollTop = textAreas[index].scrollHeight;
                    }}
                }}
                scroll({len(st.session_state.history)})
            </script>
            """

            components.html(js)
        
            # If user choose to end the conversation
    if end_button:
        if "vector_store" in st.session_state:
            vector_store = st.session_state["vector_store"]
            chat_summary = ask_for_summary(vector_store, st.session_state.history, st.session_state.document_description)
            st.write(chat_summary['answer'])
        else:
            st.write("There is nothing to be summarised")
    

