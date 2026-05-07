# AI Image Captioning and Semantic Image Search System

A complete, beginner-friendly Streamlit application that lets users upload images, automatically generate captions with BLIP, and search the uploaded image library using CLIP-powered semantic text queries.

## Project overview

This project demonstrates a practical multimodal AI workflow:

1. A user uploads one or more images.
2. The app generates a descriptive caption for each image using `Salesforce/blip-image-captioning-base`.
3. The app stores each image locally and keeps its caption plus CLIP embedding in Streamlit session state.
4. A user enters a text query such as `puppy`, `sunset`, or `food on a table`.
5. The app embeds the query with `openai/clip-vit-base-patch32`, compares it to image embeddings with cosine similarity, and displays the top-k most relevant matches.

Because CLIP learns visual-text relationships, the search is semantic rather than exact keyword matching. For example, a query like `puppy` can still retrieve an image captioned as a dog playing outside.

## Features

- Multiple image upload support.
- Automatic image caption generation with BLIP.
- Local image storage in the `stored_images/` directory.
- CLIP image embeddings for every uploaded image.
- CLIP text embeddings for user search queries.
- Cosine similarity ranking with scikit-learn.
- Top-k similar image retrieval.
- Similarity score display for every result.
- Sidebar history for uploaded images.
- Streamlit session state handling to avoid duplicate processing during reruns.
- Cached model loading for faster interaction after the first run.
- Loading spinners, progress feedback, validation, and error handling.
- Responsive, modern Streamlit UI with custom styling.
- Beginner-friendly modular Python code with comments explaining BLIP, CLIP, embeddings, and semantic similarity.

## Folder structure

```text
.
├── app.py
├── requirements.txt
└── README.md
```

The app also creates `stored_images/` at runtime to save uploaded image files.

## Installation steps

### 1. Clone or open the project

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first app run downloads the BLIP and CLIP model weights from Hugging Face, so an internet connection is required initially.

## How to run

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal, usually:

```text
http://localhost:8501
```

## How to use

1. Upload one or more `.png`, `.jpg`, `.jpeg`, `.webp`, or `.bmp` images.
2. Wait while the app captions and indexes each image.
3. Review generated captions below the uploaded images.
4. Enter a search query in the search bar.
5. Adjust the top-k slider to control how many matches are returned.
6. Optionally set a minimum similarity score threshold.
7. View ranked image results with captions and similarity scores.

## Model explanation

### BLIP: `Salesforce/blip-image-captioning-base`

BLIP is used for image captioning because it is designed for vision-language tasks. It receives image pixels and generates human-readable text that describes the content of the image. In this project, BLIP captions make the app easy to understand and provide useful metadata for each upload.

### CLIP: `openai/clip-vit-base-patch32`

CLIP is used for semantic image search because it embeds images and text into the same vector space. If an image and a text query have similar meanings, their vectors should be close together.

### Embeddings and semantic similarity

An embedding is a numeric vector that represents the meaning of an image or a piece of text. This app generates:

- Image embeddings from uploaded images.
- Text embeddings from the user's search query.

The app normalizes embeddings and compares them with cosine similarity. Higher cosine similarity means the query and image are more semantically related.

## Production notes

- Uploaded images are saved locally in `stored_images/`.
- Captions and embeddings are kept in Streamlit session state for simplicity and speed.
- For a multi-user or long-running deployment, move metadata and embeddings to persistent storage.
- The app uses cached model loading with `st.cache_resource` so models are not reloaded on every Streamlit rerun.

## Optional improvements

### Voice search

Add microphone input and speech-to-text so users can search by speaking. This can be implemented with a Streamlit audio recorder component plus a speech recognition model or API.

### Database storage

Use SQLite for a lightweight local project, or PostgreSQL/MongoDB for production. Store image paths, captions, upload times, and embedding vectors so the index survives app restarts.

### FAISS vector search

For large collections, replace the in-memory cosine similarity scan with FAISS. FAISS can search thousands or millions of embeddings more efficiently than a basic NumPy matrix comparison.

### Streamlit Cloud deployment

1. Push the project to GitHub.
2. Make sure `requirements.txt` is committed.
3. Go to Streamlit Community Cloud.
4. Create a new app from the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy the app.

> Note: BLIP and CLIP model weights are large. On small cloud instances, startup may take time and memory usage can be significant.

## Future improvements

- Add persistent database-backed image history.
- Store embeddings in FAISS or a managed vector database.
- Add image deletion and re-indexing controls.
- Add voice search and multilingual queries.
- Add user authentication for private image libraries.
- Add caption editing so users can correct generated descriptions.
- Add batch export of image metadata and search results.
