"""Streamlit app for AI image captioning and semantic image search.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

# BLIP creates natural-language image descriptions, while CLIP maps text and images
# into the same embedding space. Keeping the two responsibilities separate makes the
# app easier to understand, debug, and present as a production-style project.
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
STORAGE_DIR = Path("stored_images")


st.set_page_config(
    page_title="AI Image Captioning & Semantic Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    .hero-card {
        padding: 1.4rem 1.6rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, #111827 0%, #1d4ed8 52%, #7c3aed 100%);
        color: white;
        box-shadow: 0 18px 40px rgba(17, 24, 39, 0.18);
        margin-bottom: 1.2rem;
    }
    .hero-card h1 { margin-bottom: 0.35rem; }
    .hero-card p { font-size: 1.02rem; opacity: 0.94; margin-bottom: 0; }
    .metric-card {
        padding: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 1rem;
        background: rgba(248, 250, 252, 0.72);
    }
    .caption-box {
        padding: 0.85rem 0.95rem;
        border-radius: 0.85rem;
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        min-height: 4.5rem;
    }
    .score-pill {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        background: #dbeafe;
        color: #1e40af;
        font-weight: 700;
        font-size: 0.88rem;
    }
    @media (max-width: 760px) {
        .hero-card { padding: 1rem; }
        .hero-card h1 { font-size: 1.65rem; }
    }
</style>
"""


@st.cache_resource(show_spinner="Loading BLIP and CLIP models. This happens once per app session...")
def load_models() -> tuple[Any, Any, Any, Any, torch.device]:
    """Load Hugging Face models once and reuse them across reruns.

    Streamlit reruns the script whenever users interact with widgets. Model loading is
    expensive, so cache_resource keeps the BLIP and CLIP objects in memory and makes
    the UI much faster after the first load.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(device)
    blip_model.eval()

    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    clip_model.eval()

    return blip_processor, blip_model, clip_processor, clip_model, device


def initialize_session_state() -> None:
    """Create the app-level in-memory index used by Streamlit session state."""
    defaults = {
        "images": [],
        "processed_hashes": set(),
        "last_query": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def image_to_rgb(image_bytes: bytes) -> Image.Image:
    """Open uploaded bytes as a PIL RGB image with helpful validation errors."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc


def file_fingerprint(file_name: str, image_bytes: bytes) -> str:
    """Return a stable fingerprint so reruns do not duplicate uploaded images."""
    digest = hashlib.sha256(image_bytes).hexdigest()
    return f"{file_name}:{digest}"


def save_image(image: Image.Image, original_name: str, fingerprint: str) -> Path:
    """Persist the image locally and return the saved path."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = Path(original_name).stem.replace(" ", "_")[:50] or "image"
    output_path = STORAGE_DIR / f"{safe_stem}_{fingerprint.split(':')[-1][:12]}.jpg"
    image.save(output_path, format="JPEG", quality=92)
    return output_path


def generate_caption(
    image: Image.Image,
    blip_processor: BlipProcessor,
    blip_model: BlipForConditionalGeneration,
    device: torch.device,
) -> str:
    """Generate a detailed caption for an image using BLIP.

    BLIP is used because it is trained for vision-language understanding and can turn
    raw image pixels into human-readable descriptions. The generated caption helps
    users understand what the system sees and is also useful metadata for storage.
    """
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = blip_model.generate(
            **inputs,
            max_new_tokens=55,
            num_beams=5,
            repetition_penalty=1.15,
        )
    caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption.strip().capitalize()


def get_image_embedding(
    image: Image.Image,
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
    device: torch.device,
) -> np.ndarray:
    """Convert an image into a normalized CLIP embedding vector.

    CLIP is used for semantic search because it was trained to align images and text
    in one shared vector space. Images with meanings close to a text query should be
    close to that query vector, even when the exact words differ.
    """
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)

    # Embeddings are high-dimensional numeric summaries of meaning. Normalizing makes
    # cosine similarity compare vector direction instead of raw vector length.
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype(np.float32)[0]


def get_text_embedding(
    query: str,
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
    device: torch.device,
) -> np.ndarray:
    """Convert a search query into the same normalized CLIP space as images."""
    inputs = clip_processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        embedding = clip_model.get_text_features(**inputs)
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype(np.float32)[0]


def add_uploaded_image(
    uploaded_file: Any,
    blip_processor: BlipProcessor,
    blip_model: BlipForConditionalGeneration,
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
    device: torch.device,
) -> bool:
    """Process one upload, store metadata, and return True when a new item is added."""
    image_bytes = uploaded_file.getvalue()
    fingerprint = file_fingerprint(uploaded_file.name, image_bytes)

    if fingerprint in st.session_state.processed_hashes:
        return False

    image = image_to_rgb(image_bytes)
    caption = generate_caption(image, blip_processor, blip_model, device)
    embedding = get_image_embedding(image, clip_processor, clip_model, device)
    saved_path = save_image(image, uploaded_file.name, fingerprint)

    st.session_state.images.append(
        {
            "id": fingerprint,
            "name": uploaded_file.name,
            "caption": caption,
            "embedding": embedding,
            "path": str(saved_path),
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    st.session_state.processed_hashes.add(fingerprint)
    return True


def search_images(
    query: str,
    top_k: int,
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Return the top-k images ranked by cosine similarity to the text query.

    Semantic similarity works by measuring the angle between embeddings. A high cosine
    similarity means the query and image vectors point in similar directions, so their
    meanings are likely related. For example, CLIP can rank a dog image highly for the
    query "puppy" because both concepts are nearby in its learned embedding space.
    """
    if not query.strip() or not st.session_state.images:
        return []

    query_embedding = get_text_embedding(query, clip_processor, clip_model, device)
    image_embeddings = np.vstack([item["embedding"] for item in st.session_state.images])
    scores = cosine_similarity(query_embedding.reshape(1, -1), image_embeddings)[0]
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results: list[dict[str, Any]] = []
    for index in ranked_indices:
        item = dict(st.session_state.images[index])
        item["score"] = float(scores[index])
        results.append(item)
    return results


def render_sidebar() -> None:
    """Show upload history and lightweight controls in the sidebar."""
    with st.sidebar:
        st.header("🗂️ Image history")
        st.caption("Images are stored locally in `stored_images/` and indexed in session state.")

        if st.button("Clear current session", use_container_width=True):
            st.session_state.images = []
            st.session_state.processed_hashes = set()
            st.session_state.last_query = ""
            st.rerun()

        if not st.session_state.images:
            st.info("No images uploaded yet.")
            return

        for index, item in enumerate(reversed(st.session_state.images), start=1):
            with st.expander(f"{index}. {item['name']}", expanded=index == 1):
                st.image(item["path"], use_container_width=True)
                st.caption(item["caption"])
                st.caption(f"Uploaded: {item['uploaded_at']}")


def render_uploaded_gallery() -> None:
    """Display all uploaded images with captions."""
    st.subheader("Uploaded image library")
    if not st.session_state.images:
        st.info("Upload one or more images to generate captions and build your searchable image library.")
        return

    columns = st.columns(3)
    for index, item in enumerate(st.session_state.images):
        with columns[index % 3]:
            st.image(item["path"], use_container_width=True)
            st.markdown(f"**{item['name']}**")
            st.markdown(f"<div class='caption-box'>{item['caption']}</div>", unsafe_allow_html=True)


def render_search_results(results: list[dict[str, Any]], min_score: float) -> None:
    """Render search results after applying the selected score threshold."""
    filtered_results = [item for item in results if item["score"] >= min_score]

    if not filtered_results:
        st.warning("No images matched the current query and score threshold.")
        return

    st.subheader("Most relevant images")
    for rank, item in enumerate(filtered_results, start=1):
        left, right = st.columns([1, 1.25], vertical_alignment="top")
        with left:
            st.image(item["path"], use_container_width=True)
        with right:
            st.markdown(f"### #{rank} — {item['name']}")
            st.markdown(f"<span class='score-pill'>Similarity: {item['score']:.3f}</span>", unsafe_allow_html=True)
            st.markdown("#### Generated caption")
            st.markdown(f"<div class='caption-box'>{item['caption']}</div>", unsafe_allow_html=True)
            st.caption(f"Stored at: {item['path']}")


def render_optional_improvements() -> None:
    """Document extension points directly in the app for project presentations."""
    with st.expander("🚀 Optional production improvements"):
        st.markdown(
            """
            - **Voice search:** add speech-to-text input with browser microphone components or an API.
            - **Database storage:** move metadata from session state to SQLite, PostgreSQL, or MongoDB.
            - **FAISS vector search:** replace in-memory cosine ranking with FAISS for large image collections.
            - **Deployment:** deploy the app to Streamlit Community Cloud with the included `requirements.txt`.
            """
        )


def main() -> None:
    """Run the Streamlit UI."""
    initialize_session_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero-card">
            <h1>🔎 AI Image Captioning & Semantic Image Search</h1>
            <p>Upload images, generate BLIP captions, and retrieve the most relevant images with CLIP-powered semantic search.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Preparing AI models..."):
        blip_processor, blip_model, clip_processor, clip_model, device = load_models()

    render_sidebar()

    stat_cols = st.columns(3)
    stat_cols[0].metric("Images indexed", len(st.session_state.images))
    stat_cols[1].metric("Caption model", "BLIP base")
    stat_cols[2].metric("Search model", "CLIP ViT-B/32")

    st.divider()

    upload_col, search_col = st.columns([1, 1], gap="large")

    with upload_col:
        st.subheader("1️⃣ Upload images")
        uploaded_files = st.file_uploader(
            "Choose one or more images",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=True,
            help="Each new image is captioned with BLIP and embedded with CLIP.",
        )

        if uploaded_files:
            progress = st.progress(0, text="Processing uploads...")
            added_count = 0
            for index, uploaded_file in enumerate(uploaded_files, start=1):
                try:
                    with st.spinner(f"Captioning and indexing {uploaded_file.name}..."):
                        if add_uploaded_image(
                            uploaded_file,
                            blip_processor,
                            blip_model,
                            clip_processor,
                            clip_model,
                            device,
                        ):
                            added_count += 1
                except ValueError as exc:
                    st.error(f"Could not process {uploaded_file.name}: {exc}")
                except RuntimeError as exc:
                    st.error(f"Model error while processing {uploaded_file.name}: {exc}")
                progress.progress(index / len(uploaded_files), text=f"Processed {index}/{len(uploaded_files)} files")
            progress.empty()

            if added_count:
                st.success(f"Added {added_count} new image(s) to the semantic index.")
            else:
                st.info("No new images were added because these uploads were already indexed.")

    with search_col:
        st.subheader("2️⃣ Search semantically")
        query = st.text_input(
            "Search your images",
            placeholder="Try: puppy, sunset beach, food, people playing sports...",
            help="CLIP compares your text embedding with every image embedding.",
        )
        top_k = st.slider(
            "Top-k results",
            min_value=1,
            max_value=max(1, min(10, len(st.session_state.images) or 1)),
            value=max(1, min(5, len(st.session_state.images) or 1)),
        )
        min_score = st.slider("Minimum similarity score", 0.0, 1.0, 0.0, 0.01)

        if query.strip():
            st.session_state.last_query = query
            if not st.session_state.images:
                st.warning("Upload images before searching.")
            else:
                with st.spinner("Finding semantically similar images..."):
                    results = search_images(query, top_k, clip_processor, clip_model, device)
                render_search_results(results, min_score)
        else:
            st.info("Enter a natural-language query to retrieve matching images dynamically.")

    st.divider()
    render_uploaded_gallery()
    render_optional_improvements()


if __name__ == "__main__":
    main()
