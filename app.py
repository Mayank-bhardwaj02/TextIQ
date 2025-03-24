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
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import anthropic
from pydantic import BaseModel


st.set_page_config(page_title="TEXTIQ", layout="wide")
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


if "pdf_text" not in st.session_state:
    st.session_state["pdf_text"] = ""
if "ocr_text" not in st.session_state:
    st.session_state["ocr_text"] = ""
if "url_text" not in st.session_state:
    st.session_state["url_text"] = ""
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "conversation_chain" not in st.session_state:
    st.session_state["conversation_chain"] = None
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

SYSTEM_PROMPT = ("You are a helpful analyzer. When given a piece of text, analyze it and provide "
                 "the following in simple language using Markdown, in under 500 words:\n"
                 "- **Main Topic:** The core subject of the text.\n"
                 "- **Key Insights:** Important ideas or findings, listed in bullet points.\n"
                 "- **Key Words:** Important terms or phrases, each explained in one simple line (6-7 words).\n\n"
                 "Ensure your analysis captures the essence and core of the content.")


class AnthropicLLM:
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-20241022", max_tokens: int = 500, temperature: float = 0.7):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    def __call__(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Please analyze the following:\n{prompt}"}]
        )
        return response.content[0].text

anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
llm = AnthropicLLM(api_key=anthropic_api_key)


def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    pages = [page.extract_text() for page in reader.pages]
    return "\n".join(pages)

def extract_text_from_image(file) -> str:
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return pytesseract.image_to_string(thresh, lang="eng")

def extract_text_from_url(url: str) -> str:
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    raw = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", raw).strip()

def update_combined_text() -> str:
    texts = []
    if st.session_state["pdf_text"]:
        texts.append(st.session_state["pdf_text"])
    if st.session_state["ocr_text"]:
        texts.append(st.session_state["ocr_text"])
    if st.session_state["url_text"]:
        texts.append(st.session_state["url_text"])
    return "\n".join(texts)

def update_vectorstore(combined_text: str):
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(combined_text)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=chunks, embedding=embed)


def update_chain():
    combined = update_combined_text()
    st.session_state["vectorstore"] = update_vectorstore(combined)
    retriever = st.session_state["vectorstore"].as_retriever()
    st.session_state["conversation_chain"] = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state["memory"],
        retriever=retriever
    )

def analyze_text(text: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\nPlease analyze the following:\n{text}"
    return llm(prompt)


class AnalyzeRequest(BaseModel):
    text: str = ""
class ChatRequest(BaseModel):
    question: str
class URLRequest(BaseModel):
    url: str


st.title("TEXTIQ")
st.subheader("Transform PDFs, images, and links into clear insights and interactive conversations.")

col1, col2, col3 = st.columns(3)
with col1:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        st.session_state["pdf_text"] = extract_text_from_pdf(pdf_file)
with col2:
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if image_file:
        st.session_state["ocr_text"] = extract_text_from_image(image_file)
with col3:
    url_input = st.text_input("Enter URL")
    if url_input and st.button("Scrape URL"):
        st.session_state["url_text"] = extract_text_from_url(url_input)

if st.button("Analyze"):
    combined_text = update_combined_text()
    update_chain()
    analysis = analyze_text(combined_text)
    st.markdown("**Analysis Result:**")
    st.write(analysis)

st.markdown("### Chat with TEXTIQ")
if st.session_state["conversation_chain"] is None:
    st.info("Please click 'Analyze' first to initialize the conversation chain.")
else:
    user_question = st.text_input("Type your question here:")
    if st.button("Send"):
        update_chain()
        result = st.session_state["conversation_chain"].invoke({"question": user_question})
        st.markdown(f"**You:** {user_question}")
        st.markdown(f"**TEXTIQ:** {result['answer']}")

if st.button("Reset All"):
    st.session_state["pdf_text"] = ""
    st.session_state["ocr_text"] = ""
    st.session_state["url_text"] = ""
    st.session_state["memory"].clear()
    st.session_state["vectorstore"] = None
    st.session_state["conversation_chain"] = None
    st.success("Reset successful.")
