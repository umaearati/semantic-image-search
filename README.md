# Multimodal Semantic Image Search System

A production-oriented semantic image retrieval system built using CLIP embeddings and Qdrant for similarity search, with lightweight LLM-based query rewriting for improved search relevance.

## Overview

This project implements a multimodal semantic search backend that supports:

Text-to-Image Retrieval Image-to-Image Retrieval Folder-based Batch Indexing Lightweight LLM-based Query Rewriting Optimised Vector Storage using Qdrant

The system is designed with production considerations including structured logging, error handling, configuration management, and cost-aware LLM invocation.

## Architecture

### 1. Embedding Layer

Uses CLIP (ViT-B-32) for image and text embeddings. Configurable model checkpoint. CPU/GPU device selection supported.

### 2. Vector Storage (Qdrant)

Named vector configuration. On-disk vector persistence enabled. Scalar quantisation (INT8) for reduced memory footprint. Minimal metadata payload (filename, path, category).

### 3. Indexing Pipeline

Single image indexing. Folder-based batch indexing. Automatic category inference from folder structure. Batch upsert to Qdrant.

### 4. Query Processing

Lightweight LLM-based query rewriting. Intent detection to avoid unnecessary LLM calls. Query length control to limit token usage. In-memory caching to reduce repeated API calls.

### 5. UI Layer

Streamlit interface for controlled testing. Supports manual evaluation of retrieval quality.

# Optimisations Implemented

## LLM Optimisation

Reduced token usage via conditional LLM invocation. Reduced API calls through query caching. Reduced inference latency by skipping rewrite for caption-style queries. Input length control to prevent excessive token cost.

## Infrastructure Optimisation

Reduced memory usage via on-disk vector storage. Reduced storage overhead using scalar quantisation (INT8). Minimal metadata storage to avoid large payload overhead.
