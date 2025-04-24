from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from starlette.responses import JSONResponse
import redis.asyncio as redis
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from typing import Optional
import time
import httpx

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Code Companion API",
    description="API for generating code using local Mistral model via Ollama",
    version="1.0.0"
)

# ✅ Configure CORS middleware (fixed to support preflight properly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["http://localhost:3000"] for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis initialization for rate limiting
@app.on_event("startup")
async def startup():
    try:
        redis_connection = redis.from_url("redis://localhost:6379", encoding="utf-8", decode_responses=True)
        await FastAPILimiter.init(redis_connection)
        print("✅ Redis connected successfully")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")

# Request model
class PromptRequest(BaseModel):
    prompt: str
    model: Optional[str] = "mistral"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

# Custom 429 rate limit handler
@app.exception_handler(429)
async def rate_limit_exceeded_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down."},
    )

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the AI Code Companion Bot backend!",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# Greet endpoint
@app.get("/greet/{name}")
def greet_user(name: str):
    return {"greeting": f"Hello, {name}! How can I assist you today?"}

# Main code generation endpoint with rate limiting
@app.post("/generate")
async def generate_code(
    req: PromptRequest,
    request: Request,
    rate_limiter: None = Depends(RateLimiter(times=5, minutes=1))
):
    try:
        start_time = time.time()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url="http://localhost:11434/api/generate",
                json={
                    "model": req.model,
                    "prompt": req.prompt,
                    "temperature": req.temperature,
                    "max_tokens": req.max_tokens,
                    "stream": False
                }
            )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama error: {response.text}")

        result = response.json()
        processing_time = time.time() - start_time

        return {
            "response": result.get("response", ""),
            "model": req.model,
            "processing_time_seconds": round(processing_time, 2),
            "requests_remaining": await FastAPILimiter.get_remaining(request)
        }

    except Exception as e:
        import traceback
        print("🔥 Error Traceback:")
        print(traceback.format_exc())  # This prints the full error in the terminal

        raise HTTPException(
            status_code=500,
            detail=f"Error communicating with Ollama: {str(e)}"
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    redis_status = "active" if FastAPILimiter.redis else "inactive"
    return {
        "status": "healthy",
        "ollama_configured": True,
        "redis_status": redis_status,
        "timestamp": time.time()
    }
