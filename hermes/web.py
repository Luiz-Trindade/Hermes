from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


async def serve_static_fastapi(
    port: int = 8000, directory: str = "web_interface", agent=None
):
    try:
        app = FastAPI()

        # Configuração CORS - DEVE vir ANTES das rotas
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Em produção, especifique os domínios permitidos
            allow_credentials=True,
            allow_methods=["*"],  # Permite todos os métodos (GET, POST, OPTIONS, etc.)
            allow_headers=["*"],  # Permite todos os headers
        )

        # Chat route using the agent
        @app.post("/chat")
        async def chat(request: Request):
            data = await request.json()
            message = data.get("message", "")
            chat_history = data.get("chat_history", [])

            if not message:
                return JSONResponse({"error": "Empty message"}, status_code=400)

            if agent is None:
                return JSONResponse({"error": "Agent not available"}, status_code=500)

            # Pass both input_data and chat_history to the agent
            response = await agent.execute(
                input_data=message, chat_history=chat_history
            )
            return response

        # Mount static files - DEVE vir por ÚLTIMO
        app.mount(
            "/",
            StaticFiles(directory="hermes/web_interface/dist", html=True),
            name="static",
        )

        print(f"Serving '{directory}' on port {port}...")
        config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    except KeyboardInterrupt:
        print("Server interrupted by user.")
    except Exception as e:
        print(f"Error starting FastAPI server: {str(e)}")
