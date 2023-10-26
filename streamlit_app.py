import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import QianfanEmbeddingsEndpoint
from langchain.llms import QianfanLLMEndpoint
from langchain_wenxin.llms import Wenxin
import streamlit.components.v1 as components
import sys

from langchain.document_loaders import PyPDFLoader
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# chunk the data
def chunk_data(data, chunk_size):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

    #llm = Wenxin(model="ernie-bot-turbo")
    llm = QianfanLLMEndpoint(
        streaming=True, 
        model="ERNIE-Bot-turbo",
        endpoint="eb-instant",
        )
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


    crc = ConversationalRetrievalChain.from_llm(llm, retriever, combine_docs_chain_kwargs={'prompt': qa_prompt})
    result = crc({'question': question, 'chat_history': chat_history})
    return result


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
    st.subheader("万科企业股份有限公司2023年第一季度报告")

    st.session_state.data = ""
    st.session_state.document_description = "万科企业股份有限公司2023年第一季度报告"
    st.session_state.chat_context_length = 10
    if not st.session_state.data:
        loader = PyPDFLoader("http://static.cninfo.com.cn/finalpage/2023-04-29/1216686497.PDF")
        data = loader.load()

    
    chunks = chunk_data(st.session_state.data, 384)
    st.session_state.vector_store = create_embeddings(chunks)

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
        submit_button = st.form_submit_button("提交")

    # If user entered a question
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
