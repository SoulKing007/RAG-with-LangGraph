
# startup.py
"""Startup script for RAG Pipeline"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements = [
        "langchain-text-splitters",
        "langchain-community", 
        "langgraph",
        "langchain-core",
        "langchain",
        "langchain-google-genai",
        "google-generativeai",
        "chromadb",
        "langchain-chroma",
        "python-pptx",
        "python-docx", 
        "PyPDF2",
        "pdfplumber",
        "beautifulsoup4",
        "requests",
        "streamlit",
        "streamlit-extras",
        "typing-extensions"
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def setup_directories():
    """Create necessary directories"""
    directories = ["./chroma_db", "./uploads", "./temp"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def check_api_key():
    """Check if Google API key is set"""
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠️  WARNING: GOOGLE_API_KEY environment variable is not set!")
        print("Please set it using one of these methods:")
        print("1. export GOOGLE_API_KEY='your-api-key-here'")
        print("2. Set it in your environment variables")
        print("3. Enter it in the Streamlit interface")
        print()

def main():
    """Main startup function"""
    print("🚀 Starting RAG Pipeline setup...")
    
    # Install requirements
    install_requirements()
    
    # Setup directories
    setup_directories()
    
    # Check API key
    check_api_key()
    
    print("✅ Setup complete!")
    print("\nTo run the application:")
    print("streamlit run rag_pipeline.py")

if __name__ == "__main__":
    main()

