#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install langchain_groq')


# In[2]:


from langchain_groq import ChatGroq
llm=ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile",
)
result=llm.invoke("What is the capital of France?")
print(result.content)


# In[3]:


get_ipython().system('pip install pypdf')


# In[4]:


get_ipython().system('pip install chromadb')


# In[5]:


get_ipython().system('pip install --upgrade huggingface_hub')


# In[6]:


get_ipython().system('pip install --upgrade sentence-transformers')


# In[7]:


get_ipython().system('pip install chromadb')


# In[8]:


get_ipython().system('pip install --upgrade groq gradio langchain chromadb')


# In[9]:


from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gradio as gr


GROQ_API_KEY = "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" 


def initialize_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )


def create_vector_db():
    loader = DirectoryLoader(r"C:\Users\manas\Desktop\MentalHealthGen\data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=r"C:\Users\manas\Desktop\MentalHealthGen\chroma_db")
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
db_path = r"C:\Users\manas\Desktop\MentalHealthGen\chroma_db"

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
            return response['result']  # ✅ Return only the bot response (not a tuple)
        else:
            return "I'm sorry, I couldn't process that. Please try again."
    except Exception as e:
        print(f"🔥 DEBUG ERROR: {str(e)}")  # Log error for debugging
        return f"⚠️ Error: {str(e)}"



with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧠 Mental Health Chatbot 🤖")
    gr.Markdown("A compassionate chatbot designed to assist with mental well-being. Please note: For serious concerns, contact a professional.")

    chatbot = gr.ChatInterface(fn=chatbot_response, title="Mental Health Chatbot")

    gr.Markdown("This chatbot provides general support. For urgent issues, seek help from licensed professionals.")

app.launch()


# In[10]:


create_vector_db


# In[11]:


get_ipython().system('pip install gradio')

