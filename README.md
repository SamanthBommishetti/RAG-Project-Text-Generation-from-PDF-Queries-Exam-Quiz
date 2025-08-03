# RAG-Project-Text-Generation-from-PDF-Queries-Exam-Quiz

This Streamlit app enables users to upload a PDF, interact with its content through questions, and create exam questions such as multiple choice and fill-in-the-blank items. The app leverages LangChain for document processing, Hugging Face for text embeddings, and integrates ChatGroq to deliver advanced language model capabilities.

# Features
- Upload a PDF and get questions tailored to its content.
- Create multiple-choice and fill-in-the-blank questions from the uploaded PDF.
- Use a chat interface to ask questions about the PDF’s text.
- Take interactive exams with built-in answer checking.

# Requirements
The following Python packages are required to run the application:

streamlit
os
langchain_groq
langchain_community.document_loaders
langchain_community.vectorstores
langchain.text_splitter
langchain_core.prompts
langchain.chains
langchain_huggingface.embeddings
dotenv
random
time
warnings

# Key Libraries

- **Streamlit**: Powers the web app interface.
- **LangChain**: Handles loading documents, splitting text, creating prompts, and managing document retrieval chains.
- **HuggingFaceEmbeddings**: Produces text embeddings from the uploaded PDF content.
- **FAISS**: Enables fast similarity searches using vector storage.
- **ChatGroq**: Supplies advanced natural language processing through a large language model.
- **dotenv**: Loads required environment variables.
