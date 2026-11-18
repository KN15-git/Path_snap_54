import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import uvicorn
from src.api.main import app

if __name__ == "__main__":
    print(f"Starting PathSnap API Server on Laptop 1 (Host: 0.0.0.0, Port: 8000)...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
