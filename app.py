import os
import secrets
import sys
import types
import logging
from importlib.machinery import ModuleSpec

import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel


def disable_torchcodec_for_text_embeddings() -> None:
    decoders = types.ModuleType("torchcodec.decoders")

    class AudioDecoder:
        pass

    class VideoDecoder:
        pass

    decoders.AudioDecoder = AudioDecoder
    decoders.VideoDecoder = VideoDecoder

    torchcodec = types.ModuleType("torchcodec")
    torchcodec.decoders = decoders
    torchcodec.__spec__ = ModuleSpec("torchcodec", loader=None)
    decoders.__spec__ = ModuleSpec("torchcodec.decoders", loader=None)

    sys.modules["torchcodec"] = torchcodec
    sys.modules["torchcodec.decoders"] = decoders


disable_torchcodec_for_text_embeddings()

from sentence_transformers import SentenceTransformer


load_dotenv()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers_modules").setLevel(logging.ERROR)

def get_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


MODEL_NAME = get_env("EMBEDDING_MODEL")
DEVICE = os.getenv("EMBEDDING_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = get_env("EMBEDDING_DTYPE")
TASK = get_env("EMBEDDING_TASK")
MAX_LENGTH = int(get_env("EMBEDDING_MAX_LENGTH"))
BATCH_SIZE = int(get_env("EMBEDDING_BATCH_SIZE"))
TRUNCATE_DIM = int(get_env("EMBEDDING_TRUNCATE_DIM"))
NORMALIZE = get_env("EMBEDDING_NORMALIZE").lower() == "true"
ATTENTION = (os.getenv("EMBEDDING_ATTENTION") or "").strip() or None
API_TOKEN = get_env("EMBEDDING_API_TOKEN")

VALID_TASKS = {
    "retrieval.query",
    "retrieval.passage",
    "separation",
    "classification",
    "text-matching",
}

model_dtype = torch.float16 if DTYPE == "float16" else torch.float32

app = FastAPI(title="Jina Embeddings Server")
bearer_scheme = HTTPBearer(auto_error=False)

def load_embedding_model(attention: str | None) -> SentenceTransformer:
    model_kwargs = {"dtype": model_dtype}
    if attention:
        model_kwargs["attn_implementation"] = attention

    return SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        device=DEVICE,
        model_kwargs=model_kwargs,
    )


ACTIVE_ATTENTION = ATTENTION
model = load_embedding_model(ACTIVE_ATTENTION)
model.eval()
model.max_seq_length = MAX_LENGTH

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = MODEL_NAME
    task: str | None = None
    dimensions: int | None = None
    max_length: int | None = None
    batch_size: int | None = None
    normalize: bool | None = None


def verify_api_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(credentials.credentials, API_TOKEN):
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "attention": ACTIVE_ATTENTION,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest, _: None = Depends(verify_api_token)):
    texts = [request.input] if isinstance(request.input, str) else request.input
    task = request.task or TASK

    if task not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task: {task}")

    vectors = model.encode(
        texts,
        task=task,
        prompt_name=task,
        batch_size=request.batch_size or BATCH_SIZE,
        truncate_dim=request.dimensions or TRUNCATE_DIM,
        normalize_embeddings=NORMALIZE if request.normalize is None else request.normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    return {
        "object": "list",
        "model": request.model,
        "data": [
            {"object": "embedding", "index": i, "embedding": vector.tolist()}
            for i, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
