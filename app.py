import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import clip
import json
import requests
import torch
import faiss
import numpy as np
from PIL import Image
from io import BytesIO
import streamlit as st

@st.cache_resource
def load_clip():
    # return model, preprocess
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    return model, preprocess

@st.cache_resource  
def load_index():
    # return index, metadata
    index = faiss.read_index("coco.index")
    with open("metadata.json") as f:
        metadata = json.load(f)
    return index, metadata

model, preprocess = load_clip()
index, metadata = load_index()

def fetch_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def find_similar(image_url, k=5):
    img = fetch_image(image_url)
    preprocessed = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(preprocessed)
    embedding_np = embedding.detach().cpu().numpy()
    faiss.normalize_L2(embedding_np) 
    D, I = index.search(embedding_np, k)
    return [metadata[i] for i in I[0]]


def find_similar_text(query_text, k=5):
    tokens = clip.tokenize([query_text]) 
    with torch.no_grad():
        text_embedding = model.encode_text(tokens)
    embedding_np = text_embedding.detach().cpu().numpy().astype('float32')
    faiss.normalize_L2(embedding_np)
    D, I = index.search(embedding_np, k)
    return [metadata[i] for i in I[0]]

st.title("Visual Image Search")
st.caption("Powered by CLIP + FAISS")

tab1, tab2 = st.tabs(["Search by image URL", "Search by text"])

with tab1:
    url_input = st.text_input("Paste a COCO image URL")
    if st.button("Search", key="img_search"):
        st.image(url_input, caption="Your query", width=300)
        results = find_similar(url_input)
        cols = st.columns(5)
        for col, result in zip(cols, results):
            with col:
                st.image(result["coco_url"], use_container_width=True)

with tab2:
    text_input = st.text_input("Describe an image (e.g. 'a dog in a park')")
    if st.button("Search", key="text_search"):
        st.write(f"Results for: *{text_input}*")
        results = find_similar_text(text_input)
        cols = st.columns(5)
        for col, result in zip(cols, results):
            with col:
                st.image(result["coco_url"], use_container_width=True)

