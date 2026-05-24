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

# Load CLIP
model, preprocess = clip.load("ViT-B/32", device="cpu")

# Load FAISS index + metadata
index = faiss.read_index("coco.index")
with open("metadata.json") as f:
    metadata = json.load(f)

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


if __name__ == "__main__":
    test_url = metadata[10]["coco_url"]
    results = find_similar(test_url, k=5)
    for r in results:
        print(r["coco_url"])