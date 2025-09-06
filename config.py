
"""Configuration file for RAG Pipeline"""

import os
from pathlib import Path

class Config:
    """Configuration settings"""
    
    # API Keys (set via environment variables)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    # ChromaDB settings
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "rag_documents"
    
    # Text splitting settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval settings
    SIMILARITY_TOP_K = 4
    
    # Model settings
    GEMINI_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "models/embedding-001"
    TEMPERATURE = 0.3
    
    # File upload settings
    MAX_FILE_SIZE_MB = 100
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.txt']
    UPLOAD_DIRECTORY = "./uploads"
    
    # Streamlit settings
    PAGE_TITLE = "RAG Pipeline"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
