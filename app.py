import os
import re
import cv2
import numpy as np
import pytesseract
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import streamlit as st

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

st.set_page_config(page_title="TEXTIQ", layout="wide")

# --- CSS Styling (injected for a "similar" look) ---
st.markdown("""
<style>
body {
  background: #fff url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><circle cx="10" cy="10" r="1" fill="grey" opacity="0.2"/></svg>') repeat;
  font-family: 'Poppins', sans-serif;
}
h1 {
  font-size: 3em;
  font-weight: bold;
  background: linear-gradient(90deg, #6A5ACD, #483D8B);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: center;
  margin-top: 20px;
}
.upload-container, .analysis-container, .chat-container {
  background: rgba(106, 90, 205, 0.1);
  padding: 20px;
  margin: 20px;
  border-radius: 10px;
}
.analysis-box, .chat-box {
  background: #fff;
  border-radius: 10px;
  border: 1px solid #ddd;
  padding: 15px;
  min-height: 150px;
  box-shadow: 0 0 10px rgba(106, 90, 205, 0.3);
}
button, .stButton button {
  background: #6A5ACD !important;
  color: #fff !important;
  border: none !important;
  padding: 12px 30px !important;
  border-radius: 25px !important;
  font-size: 1.1em !important;
  cursor: pointer !important;
}
button:hover, .stButton button:hover {
  background: #483D8B !important;
}
.chat-interface {
  display: flex;
  gap: 10px;
  margin-top: 10px;
}
.chat-interface input[type="text"] {
  flex: 1;
  padding: 10px;
  border-radius: 5px;
  border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# --- System Prompt ---
SYSTEM_PROMPT = (
    "You are a helpful analyzer. When given a piece of text, analyze it and provide "
    "the following in simple language using Markdown, in under 500 words:\n"
    "- **Main Topic:** The core subject of the text.\n"
    "- **Key Insights:** Important ideas or findings, listed in bullet points.\n"
    "- **Key Words:** Important terms or phrases, each explained in one simple line (6-7 words).\n\n"
    "Ensure your analysis captures the essence and core of the content."
)

# --- Hugging Face Token ---
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("HF_TOKEN environment variable not found. Please set it before running.")
    st.stop()

# --- Global Session State ---
if "pdf_text" not in st.session_state:
    st.session_state["pdf_text"] = ""
if "ocr_text" not in st.session_state:
    st.session_state["ocr_text"] = ""
if "url_text" not in st.session_state:
    st.session_state["url_text"] = ""
if "conversation_chain" not in st.session_state:
    st.session_state["conversation_chain"] = None
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# --- Helper Functions ---
def extract_text_from_pdf(file) -> str:
    pdf_reader = PdfReader(file)
    extracted_pages = []
    for page in pdf_reader.pages:
        extracted_pages.append(page.extract_text())
    return "\n".join(extracted_pages)

def extract_text_from_image(file) -> str:
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(thresh, lang="eng")
    return text

def extract_text_from_url(url: str) -> str:
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    for elem in soup(["script", "style"]):
        elem.decompose()
    raw_text = soup.get_text(separator=" ", strip=True)
    cleaned_text = re.sub(r"\s+", " ", raw_text).strip()
    return cleaned_text

def update_combined_text():
    texts = []
    if st.session_state["pdf_text"]:
        texts.append(st.session_state["pdf_text"])
    if st.session_state["ocr_text"]:
        texts.append(st.session_state["ocr_text"])
    if st.session_state["url_text"]:
        texts.append(st.session_state["url_text"])
    return "\n".join(texts)

def analyze_text_with_gemma(text: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\nPlease analyze the following:\n{text}"
    return llm(prompt)

def update_vectorstore(combined_text: str):
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(combined_text)
    embed = OpenAIEmbeddings()
    return FAISS.from_texts(texts=chunks, embedding=embed)

# --- Load Gemma Model ---
@st.cache_resource
def load_gemma_model():
    model_name = "google/gemma-3-4b-it"
    tokenizer_os = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model_os = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", use_auth_token=hf_token)
    text_pipe = pipeline("text-generation", model=model_os, tokenizer=tokenizer_os, max_length=512, temperature=0.7, do_sample=True)
    return HuggingFacePipeline(pipeline=text_pipe)

llm = load_gemma_model()

# --- Layout ---
st.title("TEXTIQ")

st.subheader("Transform PDFs, images, and links into clear insights and interactive conversations.")

# --- Upload Section ---
with st.container():
    st.markdown("### 1. Upload / Provide Input")
    col1, col2, col3 = st.columns(3)

    with col1:
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf_file:
            st.session_state["pdf_text"] = extract_text_from_pdf(pdf_file)

    with col2:
        image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
        if image_file:
            st.session_state["ocr_text"] = extract_text_from_image(image_file)

    with col3:
        url_val = st.text_input("Enter URL")
        if url_val:
            if st.button("Scrape URL"):
                st.session_state["url_text"] = extract_text_from_url(url_val)

# --- Analysis Section ---
st.markdown("### 2. Analyze the Collected Text")
analysis_container = st.container()
with analysis_container:
    if st.button("Analyze"):
        combined = update_combined_text()
        st.session_state["vectorstore"] = update_vectorstore(combined)
        st.session_state["conversation_chain"] = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=st.session_state["memory"],
            retriever=st.session_state["vectorstore"].as_retriever()
        )
        analysis = analyze_text_with_gemma(combined)
        st.markdown("**Analysis Result:**")
        st.write(analysis)

# --- Chat Section ---
st.markdown("### 3. Chat with TEXTIQ")
chat_container = st.container()
with chat_container:
    if st.session_state["conversation_chain"] is None:
        st.info("No conversation chain is initialized yet. Please click 'Analyze' first.")
    else:
        user_query = st.text_input("Type your question here...")
        if st.button("Send"):
            combined = update_combined_text()
            st.session_state["vectorstore"] = update_vectorstore(combined)
            st.session_state["conversation_chain"] = ConversationalRetrievalChain.from_llm(
                llm=llm,
                memory=st.session_state["memory"],
                retriever=st.session_state["vectorstore"].as_retriever()
            )
            result = st.session_state["conversation_chain"].invoke({"question": user_query})
            st.markdown(f"**You:** {user_query}")
            st.markdown(f"**TEXTIQ:** {result['answer']}")

# --- Reset Section ---
if st.button("Reset All"):
    st.session_state["pdf_text"] = ""
    st.session_state["ocr_text"] = ""
    st.session_state["url_text"] = ""
    st.session_state["memory"].clear()
    st.session_state["vectorstore"] = None
    st.session_state["conversation_chain"] = None
    st.success("All inputs and conversation have been reset.")

