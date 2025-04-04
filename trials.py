from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gradio as gr

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Use environment variable for security

def initialize_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

def create_vector_db():
    loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)  # Relative path for deployment
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
    vector_db.persist()
    print("✅ ChromaDB created and data saved")
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot: """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

print("🚀 Initializing Chatbot...")
llm = initialize_llm()
db_path = "chroma_db"

if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

qa_chain = setup_qa_chain(vector_db, llm)

def chatbot_response(user_input, history=[]):
    if not user_input.strip():
        return "Please provide a valid input"

    try:
        response = qa_chain.invoke({"query": user_input})
        if isinstance(response, dict) and 'result' in response:
            return response['result']
        else:
            return "I'm sorry, I couldn't process that. Please try again."
    except Exception as e:
        print(f"🔥 DEBUG ERROR: {str(e)}")
        return f"⚠️ Error: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧠 Mental Health Chatbot 🤖")
    gr.Markdown("A compassionate chatbot designed to assist with mental well-being. Please note: For serious concerns, contact a professional.")
    
    chatbot = gr.ChatInterface(fn=chatbot_response, title="Mental Health Chatbot")
    
    gr.Markdown("This chatbot provides general support. For urgent issues, seek help from licensed professionals.")

app.launch(server_name="0.0.0.0", server_port=8080)
