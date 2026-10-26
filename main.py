"""Entry point: `uvicorn main:app`, or `python main.py` (reads PORT, default 8080)."""

import os

from src.api.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
