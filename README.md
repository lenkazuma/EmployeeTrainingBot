# 百度千帆大模型文件读取器 - Streamlit App for Document Processing and Analysis using Qianfan

This Streamlit application leverages the power of various libraries including LangChain, Qianfan, PyPDF, and Streamlit itself to perform document analysis, embedding generation, and conversational retrieval from documents. It's designed to handle Chinese text, providing functionalities such as chunking data, creating embeddings, and facilitating interactive Q&A sessions based on the document content.

该 Streamlit 应用利用了包括 LangChain、PyPDF2 和 Streamlit 在内的多种库的能力，执行文档分析、嵌入生成和文档的对话式检索。它旨在处理中文文本，提供如数据块切分、生成嵌入以及基于文档内容进行交互式问答等功能。

## Features 功能

- **Document Loading and Processing**: Load documents and process them into manageable chunks.
- **Embedding Generation**: Create embeddings for document chunks using LangChain and store them in Chroma DB.
- **Interactive Q&A**: Engage in interactive Q&A sessions, with the ability to ask questions based on the document's content.
- **Conversational Summaries**: Generate summaries of conversations and documents.

- **文档加载与处理**：加载文档并将其处理成可管理的块。
- **嵌入生成**：为文档块生成嵌入，并将其存储在 Chroma DB 中。
- **交互式问答**：基于文档内容进行交互式问答。
- **对话摘要**：生成基于文档和对话内容的摘要。


## Requirements 需求

To run this application, you will need Python 3.7 or later. The required Python packages can be installed using the provided `requirements.txt` file.

运行此应用，您将需要 Python 3.7 或更高版本。可以使用提供的 `requirements.txt` 文件安装所需的 Python 包。

## Installation 安装

1. Clone this repository to your local machine.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

## Usage 使用方法

Upon launching the application, you will be presented with an interface to interact with. You can change the documents to read, ask questions, and receive summaries based on the document content.

启动应用后，您将看到一个交互界面。您可以更改读取的文档、提问并根据文档内容接收摘要。

## How It Works

The `streamlit_app.py` script includes several key functions:

- `chunk_data`: Splits the document into manageable chunks.
- `create_embeddings`: Generates embeddings for each chunk and stores them in a database.
- `ask_with_memory`: Facilitates a Q&A session using the generated embeddings and document context.
- `ask_for_document_summary`: Provides a summary of the document based on its content.

## Contributing 贡献

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features.

欢迎贡献！请随时提交拉取请求、报告错误或建议功能。

## License

This project is licensed under the MIT License - see the LICENSE file for details.

此项目根据 MIT 许可证授权 - 详情请见 LICENSE 文件。
