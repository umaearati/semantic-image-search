import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

from semantic_image_search.backend.config import Config
from semantic_image_search.backend.query_translator import translate_query
from semantic_image_search.backend.ingestion import IndexService
from semantic_image_search.backend.retriever import ImageSearchService
from semantic_image_search.backend.logger import GLOBAL_LOGGER as log
from semantic_image_search.backend.exception.custom_exception import SemanticImageSearchException


app = FastAPI(
    title="Semantic Image Search API",
    description="CLIP + Qdrant + LLM Query Translator",
    version="1.0",
)

# Lazy singletons
search_service = None
index_service = None

# What it means
# You are creating two global variables (placeholders):
# search_service → will later store an object of ImageSearchService()
# index_service → will later store an object of IndexService()
# Right now they are set to None because the real objects are not created yet.

#  Why do we do this?
# Because we want to create these services only once when the API starts (startup).
# This is called a lazy singleton pattern.

# Why is this important?
# “These services are expensive to set up. Connecting to Qdrant, loading the CLIP model,
# and reading configuration all take time and memory. So we create them once and reuse 
# them instead of creating them again for every request.”


@app.on_event("startup")
def init_services():
    global search_service, index_service
    search_service = ImageSearchService()
    index_service = IndexService()
    log.info("Services initialized successfully")


# ---------------------------------------------------------
# INGEST ENDPOINT
# ---------------------------------------------------------
@app.post("/ingest")
def ingest_images(
    folder_path: Optional[str] = Query(None),
):
    folder = Path(folder_path or Config.IMAGES_ROOT)

    if not folder.exists() or not folder.is_dir():
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid folder: {folder}"}
        )

    try:
        index_service.index_folder(str(folder))
        return {"message": f"Ingested images from {folder}"}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# The API tells the backend to process images that already exist in a folder.
# It checks the folder, runs ingestion, and reports success or failure.

# ---------------------------------------------------------
# TRANSLATE ENDPOINT
# ---------------------------------------------------------
@app.get("/translate")
def translate(q: str):
    log.info("Translate request received", query=q)

    try:
        translated = translate_query(q)
        log.info("Query translated", original=q, translated=translated)
        return {"input": q, "translated": translated}

    except Exception as e:
        log.error("Translation failed", query=q, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})

# The API receives a text query, translates it, and returns the result, with logging and error handling included.
# ---------------------------------------------------------
# TEXT SEARCH ENDPOINT
# ---------------------------------------------------------
@app.get("/search-text")
def search_text_endpoint(
    q: str,
    k: int = 5,
    category: Optional[str] = None,
    save_results: bool = False,
):
    log.info("Text search request received", query=q, top_k=k, category=category)

    try:
        translated = translate_query(q)
        log.info("Query translated for text search", translated=translated)

        metadata_filter = {"category": category} if category else None

        results = search_service.search_by_text(translated, k=k, metadata_filter=metadata_filter)

        log.info("Text search completed", total_results=len(results.points))

        resp = [
            {
                "filename": p.payload.get("filename"),
                "path": p.payload.get("path"),
                "category": p.payload.get("category"),
                "score": p.score,
            }
            for p in results.points
        ]

        folder = None
        if save_results and results.points:
            folder = search_service.save_results(results)
            log.info("Search results saved locally", folder=folder)

        return {"query": q, "translated": translated, "k": k, "saved_folder": folder, "results": resp}

    except Exception as e:
        log.error("Text search failed", query=q, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})

# Category is an optional backend filter; if the UI doesn’t send it, the search runs across all data without filtering.
# The API performs a multilingual text search on indexed data, returns the top results, and safely handles logging and errors.
# ---------------------------------------------------------
# IMAGE SEARCH ENDPOINT
# ---------------------------------------------------------
@app.post("/search-image")
def search_image_endpoint(
    file: UploadFile = File(...),
    k: int = 5,
    category: Optional[str] = None,
    save_results: bool = False,
):
    log.info("Image search request received", filename=file.filename)

    try:
        if not file.content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"error": "Only image files allowed"})

        Config.QUERY_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
        query_path = Config.QUERY_IMAGE_ROOT / file.filename

        with query_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        log.info("Uploaded query image saved", path=str(query_path))

        metadata_filter = {"category": category} if category else None

        results = search_service.search_by_image(str(query_path), k=k, metadata_filter=metadata_filter)

        resp = [
            {
                "filename": p.payload.get("filename"),
                "path": p.payload.get("path"),
                "category": p.payload.get("category"),
                "score": p.score,
            }
            for p in results.points
        ]

        folder = None
        if save_results and results.points:
            folder = search_service.save_results(results)
            log.info("Search results saved locally", folder=folder)

        return {"query_image": str(query_path), "k": k, "saved_folder": folder, "results": resp}

    except Exception as e:
        log.error("Image search failed", filename=file.filename, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})
    
    
    
    
    # This endpoint takes a user’s text query, translates it, and searches previously ingested data with optional category filtering. It formats the matched results, optionally saves them, and returns the response with proper logging and error handling.
    # This endpoint accepts an image uploaded by the user, validates and saves it, and performs a similarity search on previously ingested data. It returns the top matching results with optional category filtering, saving, and proper logging and error handling.
    
    # uvicorn semantic_image_search.backend.main:app --reload
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# cd /Users/nanimahi/semantic-image-search
# source env/bin/activate
# uvicorn semantic_image_search.backend.main:app --reload --port 8000
# A) Reset Qdrant (clean slate)

# 1) (Optional)Stop and remove any existing Qdrant container (ignore errors if it doesn’t exist)
# docker stop qdrant 2>/dev/null || true
# docker rm qdrant 2>/dev/null || true
# docker rm qdrant

# 2) Start Qdrant with ports
# docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 3) Confirm Qdrant is healthy
# curl http://localhost:6333/healthz

# B) Start FastAPI
# 4) Run the API (keep this terminal open)

# uvicorn semantic_image_search.backend.main:app --reload --port 8000

# C) Verify + test
# 5) In a second terminal: check docs
# curl -I http://127.0.0.1:8000/docs

# 6) Test your endpoint
# curl "http://127.0.0.1:8000/search-text?q=dog&k=5"

# 7)Ensure your images folder has images
# ls -lah /Users/nanimahi/semantic-image-search/images

# 8) Ingest (index) images into Qdrant (Terminal B)
# Use default folder (Config.IMAGES_ROOT)
# curl -X POST "http://127.0.0.1:8000/ingest"
# curl -s http://localhost:6333/collections/semantic-image-search | python -m json.tool | grep -E "points_count|indexed_vectors_count"


