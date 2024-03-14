# EmployeevTraining Bot - Streamlit App for Document Processing and Analysis

This Streamlit application leverages the power of various libraries including LangChain, PyPDF2, and Streamlit itself to perform document analysis, embedding generation, and conversational retrieval from documents. It's designed to handle Chinese text, providing functionalities such as chunking data, creating embeddings, and facilitating interactive Q&A sessions based on the document content.

## Features

- **Document Loading and Processing**: Load documents and process them into manageable chunks.
- **Embedding Generation**: Create embeddings for document chunks using LangChain and store them in Chroma DB.
- **Interactive Q&A**: Engage in interactive Q&A sessions, with the ability to ask questions based on the document's content.
- **Conversational Summaries**: Generate summaries of conversations and documents.

## Requirements

To run this application, you will need Python 3.7 or later. The required Python packages can be installed using the provided `requirements.txt` file.

```
langchain
docx2txt
pypdf
streamlit
chromadb
tiktoken
pysqlite3-binary
qianfan==0.0.3
PyJWT>=2.3.0
bcrypt>=3.1.7
PyYAML>=5.3.1
extra-streamlit-components>=0.1.60
PyPDF2==3.0.1
```

## Installation

1. Clone this repository to your local machine.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

## Usage

Upon launching the application, you will be presented with an interface to interact with. You can upload documents, ask questions, and receive summaries based on the document content.

## How It Works

The `streamlit_app.py` script includes several key functions:

- `chunk_data`: Splits the document into manageable chunks.
- `create_embeddings`: Generates embeddings for each chunk and stores them in a database.
- `ask_with_memory`: Facilitates a Q&A session using the generated embeddings and document context.
- `ask_for_document_summary`: Provides a summary of the document based on its content.

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
