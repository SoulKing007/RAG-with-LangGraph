
# run.py
"""Simple run script"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    # Set environment variables if not already set
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  GOOGLE_API_KEY not set. You'll need to enter it in the interface.")
    
    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "rag_pipeline.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    main()
