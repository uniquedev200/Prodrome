"""
Run Server for Abu Dhabi DoH Jawda Clinical Voice Navigation Agent.
Serves FastAPI, WebSockets, and the Interactive Chat/Testing Interface on port 8000.
"""

from dotenv import load_dotenv

load_dotenv()

import uvicorn

if __name__ == "__main__":
    print("Starting Abu Dhabi DoH Clinical Agent Server on http://localhost:8000 ...")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
