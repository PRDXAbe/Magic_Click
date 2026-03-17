"""
Face Embedding Database
FastAPI application entry point.
"""
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from app.api.routes import api
from app.config import HOST, PORT, DEBUG, BASE_DIR
from app.core.camera_service import camera_service_instance
from app.core.job_worker_service import job_worker_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    camera_service_instance.start()
    job_worker_instance.start()
    yield
    # Shutdown
    camera_service_instance.stop()
    job_worker_instance.stop()

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="UltronVision Backend",
        version="3.0.0",
        description="FastAPI service for UltronVision (Face embeddings, search, live cameras, pipelines)",
        lifespan=lifespan
    )
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register API router
    app.include_router(api)
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=f"{BASE_DIR}/app/static"), name="static")
    
    # Serve frontend templates
    NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}

    @app.get("/")
    async def serve_dashboard():
        return FileResponse(f"{BASE_DIR}/templates/dashboard.html", headers=NO_CACHE)

    @app.get("/camera")
    async def serve_camera():
        # Keep original single camera page
        return FileResponse(f"{BASE_DIR}/templates/camera.html", headers=NO_CACHE)
    
    @app.get("/cameras")
    async def serve_cameras():
        # New multi-camera view
        return FileResponse(f"{BASE_DIR}/templates/cameras.html", headers=NO_CACHE)
    
    @app.get("/upload")
    async def serve_upload():
        return FileResponse(f"{BASE_DIR}/templates/upload.html", headers=NO_CACHE)
    
    @app.get("/info")
    async def api_info():
        return {
            'name': 'Face Embedding Database',
            'version': '2.0.0',
            'framework': 'FastAPI',
            'endpoints': {
                'search': 'GET /api/search?img=<base64> | ?id=<id> | ?name=<name>',
                'add': 'POST /api/add',
                'name': 'POST /api/name',
                'topn': 'GET /api/topn',
                'health': 'GET /api/health',
                'get_person': 'GET /api/person/<id>',
                'delete_person': 'DELETE /api/person/<id>'
            }
        }
        
    return app

app = create_app()

if __name__ == '__main__':
    print(f"\n🚀 Starting UltronVision Backend (FastAPI)...")
    print(f"📍 Running on http://{HOST}:{PORT}")
    print(f"📚 API docs at http://localhost:{PORT}/docs\n")
    uvicorn.run("run:app", host=HOST, port=PORT, reload=DEBUG)
