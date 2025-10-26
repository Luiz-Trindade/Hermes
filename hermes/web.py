from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

async def serve_static_fastapi(port: int = 8000, directory: str = "web_interface"):
    app = FastAPI()
    app.mount("/", StaticFiles(directory=directory, html=True), name="static")

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    
    print(f"Servindo '{directory}' na porta {port}...")
    await server.serve()
