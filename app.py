# Import libraries
import streamlit as st  # Mengimpor library Streamlit untuk membuat aplikasi web interaktif
from dotenv import load_dotenv  # Mengimpor fungsi load_dotenv untuk mengelola variabel lingkungan dari file .env
from PyPDF2 import PdfReader  # Mengimpor PdfReader dari PyPDF2 untuk membaca teks dari file PDF
from langchain.text_splitter import CharacterTextSplitter  # Mengimpor CharacterTextSplitter dari langchain untuk membagi teks
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings  # Mengimpor jenis embeddings
from langchain.vectorstores import FAISS  # Mengimpor FAISS untuk penyimpanan vektor
from langchain.chat_models import ChatOpenAI  # Mengimpor ChatOpenAI untuk model percakapan
from langchain.memory import ConversationBufferMemory  # Mengimpor ConversationBufferMemory untuk menyimpan sejarah percakapan
from langchain.chains import ConversationalRetrievalChain  # Mengimpor ConversationalRetrievalChain untuk menghubungkan komponen percakapan
from htmlTemplates import css, bot_template, user_template  # Mengimpor template HTML untuk tampilan bot dan pengguna
from langchain.llms import HuggingFaceHub  # Mengimpor HuggingFaceHub untuk pemahaman bahasa

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    """
    Extract text from PDF documents.

    Args:
    pdf_docs (List[BytesIO]): List of uploaded PDF files.

    Returns:
    str: Extracted text from PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into smaller chunks
def get_text_chunks(text):
    """
    Split text into smaller chunks.

    Args:
    text (str): Text to be split.

    Returns:
    List[str]: List of divided text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",  # Karakter untuk memisahkan teks menjadi potongan-potongan
        chunk_size=1000,  # Jumlah karakter dalam setiap potongan teks
        chunk_overlap=200,  # Jumlah tumpang tindih antara potongan-potongan teks
        length_function=len  # Fungsi untuk menghitung panjang teks
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create vector store from text chunks
def get_vectorstore(text_chunks):
    """
    Create a vector store from text chunks.

    Args:
    text_chunks (List[str]): List of text chunks.

    Returns:
    FAISS: Vector store created from the text.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Function to create conversation chain using language model
def get_conversation_chain(vectorstore):
    """
    Create a conversation chain using a language model.

    Args:
    vectorstore (FAISS): Text vector store.

    Returns:
    ConversationalRetrievalChain: Created conversation chain.
    """
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Function to handle user input and display chat history
def handle_userinput(user_question):
    """
    Handle user input and display chat history.

    Args:
    user_question (str): User's question.
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


# Main function to run the Streamlit app
def main():
    # Load environment variables and set Streamlit page configuration
    load_dotenv()
    st.set_page_config(page_title="Chatbot UDINUS", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Check and initialize conversation and chat history state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Display header and input for user question
    st.image('udinus.png', width=125)   
    st.header(":books: CHATBOT DOKUMEN AKADEMIK UDINUS")
    user_question = st.text_input("Berikan pertanyaanmu :")
    if user_question:
        handle_userinput(user_question)

    # Display sidebar with options to upload PDF documents
    with st.sidebar:
        st.image('udinus.png', width=125)   
        st.subheader("Dokumen Saya")
        pdf_docs = st.file_uploader(
            "Upload pdf lalu klik Process'", accept_multiple_files=True)
        try:
            # Process uploaded PDF documents
            if st.button("Process"):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("Processing completed successfully!")

        except Exception as e:
            st.error(f"Processing Failed")
            st.exception(e)  # Display the full traceback for detailed error analysis

# Run the app if executed as a script
if __name__ == '__main__':
    main()
