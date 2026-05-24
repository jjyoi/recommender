# Visual Image Search
### Multimodal image retrieval using CLIP embeddings and FAISS

A Pinterest-style image recommender that lets you search a dataset of images using either a query image URL or plain text descriptions. You can type something like "a dog in a park" and get back visually relevant photos with no manual labels or tags involved.

---

## Demo

**Search by image** — paste any COCO image URL and the app returns the 5 most visually similar images in the dataset.
<img width="702" height="744" alt="Screenshot 2026-05-24 at 4 48 45 PM" src="https://github.com/user-attachments/assets/abb6d2a9-7ee6-4c4b-8f2d-90606ad46e08" />

**Search by text** — describe what you're looking for in plain English and the app retrieves matching images. This works because CLIP embeds images and text into the same vector space, so the two modalities can be compared directly.
<img width="1772" height="1278" alt="image" src="https://github.com/user-attachments/assets/5bb94839-5b45-4a10-864f-93659c4d1076" />

---

## How it works

```
COCO annotations JSON
        ↓
  ETL Pipeline (main.ipynb)
  ├── Extract   fetch images on-demand from COCO servers (no 47GB download)
  ├── Transform  batch-embed via CLIP ViT-B/32 into 512-dim float32 vectors
  └── Load      persist embeddings + metadata to disk
        ↓
  FAISS Index Build
  ├── Normalize vectors (L2) to enable cosine similarity via inner product
  ├── Build IndexFlatIP for exact nearest-neighbor search
  └── Save coco.index to disk
        ↓
  Serving (app.py)
  ├── Load CLIP model + FAISS index once at startup via st.cache_resource
  ├── Embed the query (image or text) into the same 512-dim space
  ├── Run faiss.search() to find top-K nearest neighbors
  └── Return matching image URLs + similarity scores
```

The core idea is that CLIP places semantically similar things close together in vector space regardless of whether they are images or text. A search for "a dog in a park" and an actual photo of a dog in a park will land near each other, which is what makes cross-modal retrieval possible without any labeled training data.

---

## Tech stack

| Component | Tool |
|---|---|
| Image embeddings | OpenAI CLIP (ViT-B/32) |
| Vector index | FAISS (IndexFlatIP) |
| Dataset | MS-COCO 2017 validation set |
| Image loading | Pillow + requests (fetched on-demand, nothing stored locally) |
| Demo UI | Streamlit |
| Deep learning | PyTorch |

---

## Project structure

```
recommender/
├── main.ipynb              # ETL pipeline: embeds images, builds FAISS index
├── serve.py                # Inference helpers: find_similar() and find_similar_text()
├── app.py                  # Streamlit demo
├── coco.index              # Persisted FAISS vector index
├── embeddings_matrix.npy   # Raw embedding vectors (512 images x 512 dims)
├── metadata.json           # Image IDs and COCO URLs
└── data/
    └── annotations/
        └── instances_val2017.json
```

---

## Getting started

**1. Clone the repo and create a virtual environment**
```bash
git clone <your-repo-url>
cd recommender
python3.11 -m venv .venv311
source .venv311/bin/activate
```

**2. Install dependencies**
```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install faiss-cpu Pillow requests streamlit tqdm numpy
```

**3. Download the COCO annotation file**
```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d data/
```

**4. Run the ETL pipeline**

Open `main.ipynb` and run all cells. This fetches and embeds 512 COCO images via CLIP (about 3 minutes on CPU), then builds and saves the FAISS index and metadata to disk.

**5. Launch the demo**
```bash
streamlit run app.py
```

---

## Design decisions

**Why no local image storage?**
COCO images are publicly hosted at `images.cocodataset.org`. The ETL pipeline fetches each image by URL at embed time and throws it away immediately after. Only the 512-dim vector gets kept. This avoids a 47GB download while still processing the full dataset, which mirrors how production pipelines handle large media assets.

**Why IndexFlatIP over IndexIVFFlat?**
At 512 vectors, exact search is plenty fast and approximate nearest-neighbor methods would just add complexity without any real latency benefit. At a million or more vectors, switching to `IndexIVFFlat` with `nprobe` tuning would be the right call.

**Why normalize vectors before indexing?**
`IndexFlatIP` computes inner product. On unit-normalized vectors, inner product equals cosine similarity, which measures the angle between two embeddings rather than the raw distance between them. Angle is the right thing to measure here because direction encodes semantic meaning and vector magnitude is not meaningful in this context.

**Offline vs online split**
The ETL loop runs once, takes a few minutes, and writes artifacts to disk. The serving layer reads those artifacts at startup and handles queries in milliseconds without ever re-running the ETL. This separation between batch processing and real-time serving reflects how production ML systems are typically structured.

---

## What I would improve next

- Scale up to the full COCO val set (5,000 images) or train set (118,000 images)
- Switch to `IndexIVFFlat` with HNSW for faster approximate search at larger scale
- Fine-tune CLIP on a domain-specific dataset like fashion or medical imaging using contrastive loss
- Add a second-stage re-ranking pass to improve precision on the top results
- Measure retrieval quality with Precision@K and NDCG
- Deploy to Hugging Face Spaces so it's publicly accessible
