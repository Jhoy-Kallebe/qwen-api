from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(
    title="Qwen Local API",
    description="API experimental para inferência local usando Ollama",
    version="1.0.0"
)

OLLAMA_URL = "http://host.minikube.internal:11434/api/chat"
MODEL = "qwen2.5:3b"


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/")
def root():
    return {
        "service": "Qwen Local API",
        "status": "running",
        "model": MODEL
    }


@app.get("/health")
def health():
    try:
        response = requests.get(
            "http://host.minikube.internal:11434/api/tags",
            timeout=5
        )

        if response.status_code != 200:
            raise Exception()

        return {
            "status": "healthy",
            "ollama": "available",
            "model": MODEL
        }

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not available"
        )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": request.message
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return {
            "response": data["message"]["content"]
        }

    except requests.exceptions.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Could not communicate with Ollama"
        )