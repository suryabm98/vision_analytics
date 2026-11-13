import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from colorama import init, Fore, Style

from app.database.tenant_manager import init_common_db
from app.core.face_engine import face_engine
from app.router.stream_router import router as stream_router

init(autoreset=True)
load_dotenv(find_dotenv())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Connect to Common DB and prepare 'tenant' table
    try:
        init_common_db()
        print(Fore.GREEN + Style.BRIGHT + "\n" + "=" * 60)
        print(Fore.GREEN + Style.BRIGHT + " COMMON DATABASE CONNECTED SUCCESSFULLY")
        print(Fore.GREEN + Style.BRIGHT + "=" * 60 + "\n")
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"[CRITICAL] Common Database connection failed: {e}")
        raise e

    # 2. Startup: Load InsightFace Model & Sync Embeddings for active tenants
    face_engine.load_model()
    try:
        face_engine.refresh_embeddings()
        face_engine.start_background_sync(interval=30)
    except Exception as e:
        print(Fore.YELLOW + f"[WARN] Initial embeddings sync deferred: {e}")

    yield

    print(Fore.YELLOW + "[SHUTDOWN] Application shutting down cleanly.")

app = FastAPI(
    title="Vision Attend - Multi-Tenant Face Recognition Streaming Service",
    description="Multi-tenant face recognition attendance and streaming platform backed by Central Common DB.",
    version="2.1.0",
    lifespan=lifespan
)

# Middlewares
allowed_hosts = os.getenv("ALLOWED_HOSTS", "*").split(",")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Stream Router
app.include_router(stream_router)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Vision Attend Multi-Tenant FastAPI"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=False)


# checking gpu availability
# import torch
# print(torch.cuda.is_available())