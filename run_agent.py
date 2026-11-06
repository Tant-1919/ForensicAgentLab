"""PyCharm entry point: right-click this file and select Run."""
from pathlib import Path
import uvicorn

if __name__ == "__main__":
    # Read a local .env only if it exists; a fresh checkout needs no key.
    env = Path(__file__).with_name(".env")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, env_file=str(env) if env.exists() else None)

