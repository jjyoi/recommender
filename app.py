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
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

def find_similar(image_url, k=5):
    img = fetch_image(image_url)
    preprocessed = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(preprocessed)
    embedding_np = embedding.detach().cpu().numpy()
    faiss.normalize_L2(embedding_np) 
    D, I = index.search(embedding_np, k)
    return [metadata[i] for i in I[0]], D[0]


def find_similar_text(query_text, k=5):
    tokens = clip.tokenize([query_text]) 
    with torch.no_grad():
        text_embedding = model.encode_text(tokens)
    embedding_np = text_embedding.detach().cpu().numpy().astype('float32')
    faiss.normalize_L2(embedding_np)
    D, I = index.search(embedding_np, k)
    return [metadata[i] for i in I[0]], D[0]

st.title("Visual Image Search")
st.caption("Powered by CLIP + FAISS")

tab1, tab2 = st.tabs(["Search by image URL", "Search by text"])

with tab1:
    url_input = st.text_input("Paste a COCO image URL")

    if st.button("Search", key="img_search"):
        if url_input.strip() == "":
            st.warning("Please enter an image URL.")
        else:
            st.image(url_input, caption="Your query", width=300)

            with st.spinner("Searching..."):
                results, scores = find_similar(url_input)

                cols = st.columns(5)

                for col, result, score in zip(cols, results, scores):
                    with col:
                        st.image(result["coco_url"], use_container_width=True)
                        st.caption(f"Score: {score:.3f}")


with tab2:
    text_input = st.text_input("Describe an image", placeholder="a dog in a park")

    if st.button("Search", key="text_search"):
        if text_input.strip() == "":
            st.warning("Please enter a text query.")
        else:
            st.write(f"Results for: *{text_input}*")

            with st.spinner("Searching..."):
                results, scores = find_similar_text(text_input)

                cols = st.columns(5)

                for col, result, score in zip(cols, results, scores):
                    with col:
                        st.image(result["coco_url"], use_container_width=True)
                        st.caption(f"Score: {score:.3f}")