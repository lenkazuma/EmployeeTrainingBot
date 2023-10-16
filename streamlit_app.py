import os
from langchain.vectorstores import Chroma
from langchain.embeddings import QianfanEmbeddingsEndpoint
from langchain_wenxin.llms import Wenxin
import streamlit.components.v1 as components
import streamlit as st
import glob
import sys
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

llm = Wenxin(model="ernie-bot-turbo")


# def get_pdf_text(files):
#     data = ""
#     for file in files:
#         pdf_reader = PdfReader(file)
#         for page in pdf_reader.pages:
#             data += page.extract_text()
#     return data

# Loading Documents
def load_document(file):
    loader = PyPDFLoader(file)
    data = loader.load()
    return data

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

# get answer from chatGPT, increase k for more elaborate answers


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ErnieBotChat
    from langchain.prompts import PromptTemplate
    llm = Wenxin(model="ernie-bot-turbo")

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={'k': k})

    prompt_template = """You are are examining a document. Use only the following piece of context to answer the questions at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Do not add any observations or comments. Answer only in English".
    
    CONTEXT {context}

    QUESTION: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
    answer = chain.run(q)
    return answer


def ask_with_memory(vector_store, question, chat_history=[], document_description=""):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ErnieBotChat
    from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

    #llm = ErnieBotChat()
    llm = Wenxin(model="ernie-bot-turbo")

    retriever = vector_store.as_retriever( # the vs can return documents
    search_type='similarity', search_kwargs={'k': 3})
 
    general_system_template = f""" 
    You are examining a document. Use only the heading and piece of context to answer the questions at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Do not add any observations or comments. Answer only in Chinese.
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
        formatted_history += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_history


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
    st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    )
    #st.image("images/ai-document-reader.jpg")
    st.subheader("Ask questions to your documents")

    files = glob.glob("*.pdf")
    #sst.session_state.file_name = pdf_search
    #data = load_document(pdf_search)

    #with st.sidebar:

        # uploaded_file = st.file_uploader(
        #     "上传文件", type=["pdf", "doc", "txt"])
    

        # k = st.number_input("k", min_value=1, max_value=20,
        #                     value=3, on_change=clear_history)
        # st.session_state.chat_context_length = st.number_input(
        #     "Chat context length", min_value=1, max_value=30, value=10, on_change=clear_history) or 10
        # st.session_state.document_description = st.text_input("有什么补充信息吗？")
        # add_data = st.button("Add Data", on_click=clear_history)

        # if uploaded_file and add_data:
            # display a message + execute block of code
            # with st.spinner("好的人类, 我会拜读你刚传的文件..."):

    # uploaded_file = "1216686497.pdf"
    # bytes_data = uploaded_file.read()
    # file_path = os.path.join("./", uploaded_file.name)
    # with open(file_path, "wb") as f:
    #     f.write(bytes_data)

    file_path = ".\1216686497.pdf"
    data = load_document(file_path)
    #data = get_pdf_text(files)
    print(data)
    st.write(data)

    chunks = chunk_data(data, 384)
                
    st.session_state.vector_store = create_embeddings(chunks)

        #st.success("我成功读取了你给予我的文件，这下来问个问题吧。 ")

    # Create the placeholder for chat history
    chat_history_placeholder = st.empty()

    if "history" not in st.session_state:
        st.session_state.history = []

    # Create an empty text area at the start
    chat_history_placeholder.text_area(
        label="Chat History", value="", height=400)

    # User input for the question
    with st.form(key="myform", clear_on_submit=True):
        q = st.text_input("问一个问题", key="user_question")
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
            chat_history_placeholder.text_area(
                label="对话记录", value=chat_history_str, height=400)

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
