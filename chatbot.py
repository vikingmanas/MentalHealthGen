from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
import os
import gradio as gr

# ✅ Secure API Key Handling
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Please set the GROQ_API_KEY environment variable.")

# ✅ Initialize LLM
def initialize_llm():
    return Groq(api_key=api_key)

# ✅ Create Vector Database
def create_vector_db():
    loader = DirectoryLoader("./data/", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    vector_db.persist()

    print("ChromaDB created and data saved.")
    return vector_db

# ✅ Set up the QA Chain
def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """
    You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot: 
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return qa_chain

# ✅ Initialize Chatbot
print("Initializing Chatbot...")
llm = initialize_llm()

db_path = os.path.abspath("./chroma_db")
if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

qa_chain = setup_qa_chain(vector_db, llm)

# ✅ Define Chatbot Response Function
def chatbot_response(user_input, history=[]):
    if not user_input.strip():
        return "Please provide a valid input", history
    
    response = qa_chain.invoke(user_input)  # ✅ Fixed API Call
    history.append((user_input, response["result"]))  # ✅ Fix History Format
    return response["result"], history

# ✅ Create Gradio UI
with gr.Blocks(theme="Respair/Shiki@1.2.1") as app:
    gr.Markdown("# 🧠 Mental Health Chatbot 🤖")
    gr.Markdown("A compassionate chatbot designed to assist with mental well-being. Please note: For serious concerns, contact a professional.")
    
    chatbot = gr.ChatInterface(fn=chatbot_response, title="Mental Health Chatbot")
    
    gr.Markdown("This chatbot provides general support. For urgent issues, seek help from licensed professionals.")

app.launch()
