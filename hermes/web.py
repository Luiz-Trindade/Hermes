from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import shutil
import importlib.resources as pkg_resources
import hermes.web_interface  # pacote de arquivos estáticos do Vue


async def hermes_web(port: int = 8000, agent=None):
    app = FastAPI()

    # Configuração CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Em produção, especifique os domínios permitidos
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rota de chat
    @app.post("/chat")
    async def chat(request: Request):
        data = await request.json()
        message = data.get("message", "")
        chat_history = data.get("chat_history", [])

        if not message:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        if agent is None:
            return JSONResponse({"error": "Agent not available"}, status_code=500)

        response = await agent.execute(input_data=message, chat_history=chat_history)
        return response

    # Cria diretório temporário para servir arquivos estáticos
    temp_dir = tempfile.mkdtemp()

    # Copia todo o conteúdo do build Vue (dist) para o diretório temporário
    dist_path = pkg_resources.files(hermes.web_interface) / "dist"
    for item in dist_path.iterdir():
        dest_path = os.path.join(temp_dir, item.name)
        if item.is_dir():
            shutil.copytree(item, dest_path)
        else:
            shutil.copy2(item, dest_path)

    # Monta os arquivos estáticos a partir do diretório temporário
    app.mount("/", StaticFiles(directory=temp_dir, html=True), name="static")

    print(f"Serving static files on port {port}...")
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
