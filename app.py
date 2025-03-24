import streamlit as st
import requests
import re
import cv2
import numpy as np
import pytesseract
from bs4 import BeautifulSoup
from pypdf import PdfReader


import anthropic


SYSTEM_PROMPT = (
    "You are a helpful analyzer. When given a piece of text, analyze it and provide "
    "the following in simple language using Markdown, in under 500 words:\n"
    "- **Main Topic:** The core subject of the text.\n"
    "- **Key Insights:** Important ideas or findings, listed in bullet points.\n"
    "- **Key Words:** Important terms or phrases, each explained in one simple line (6-7 words).\n\n"
    "Ensure your analysis captures the essence and core of the content."
)

def load_anthropic_client():
    
    anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
    return anthropic.Anthropic(api_key=anthropic_key)

def analyze_text_with_claude(client, text):
    
    response = client.messages.create(
        model="claude-2",  
        system=SYSTEM_PROMPT,
        max_tokens=500,
        temperature=0.7,
        messages=[
            {"role": "user", "content": f"Please analyze the following:\n{text}"}
        ]
    )
    
    return response.content[0].text

def extract_text_from_pdf(uploaded_pdf) -> str:
    
    pdf_reader = PdfReader(uploaded_pdf)
    extracted_pages = []
    for page in pdf_reader.pages:
        extracted_pages.append(page.extract_text())
    return "\n".join(extracted_pages)

def extract_text_from_image(uploaded_image) -> str:
    
    file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    
    text = pytesseract.image_to_string(thresh, lang="eng")
    return text

def extract_text_from_url(url) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    for elem in soup(["script", "style"]):
        elem.decompose()
   
    raw_text = soup.get_text(separator=" ", strip=True)
   
    cleaned_text = re.sub(r"\s+", " ", raw_text).strip()
    return cleaned_text

def main():
    st.title("📄 Document & Web Summarizer")

    st.markdown(
        "Upload a **PDF** or an **Image** (for OCR), or provide a **URL**, "
        "then get a concise summary from Anthropic Claude."
    )

    
    client = load_anthropic_client()

    
    st.subheader("1. Analyze a PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf is not None:
        if st.button("Summarize PDF"):
            with st.spinner("Extracting and analyzing PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_pdf)
                summary = analyze_text_with_claude(client, pdf_text)
            st.markdown(summary)

    
    st.subheader("2. Analyze an Image (OCR)")
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        if st.button("Summarize Image Text"):
            with st.spinner("Extracting text via OCR..."):
                img_text = extract_text_from_image(uploaded_image)
            with st.spinner("Analyzing text with Claude..."):
                summary = analyze_text_with_claude(client, img_text)
            st.markdown(summary)

    
    st.subheader("3. Analyze a Webpage")
    user_url = st.text_input("Enter a URL")
    if user_url:
        if st.button("Summarize Webpage"):
            with st.spinner("Extracting webpage text..."):
                page_text = extract_text_from_url(user_url)
            with st.spinner("Analyzing text with Claude..."):
                summary = analyze_text_with_claude(client, page_text)
            st.markdown(summary)

if __name__ == "__main__":
    main()
