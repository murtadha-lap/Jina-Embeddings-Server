# Jina Embeddings FastAPI Server

OpenAI-compatible FastAPI server for `jinaai/jina-embeddings-v3`.

![Jina Embeddings Server](public/image.png)

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

Use only `--workers 1`, because every worker loads another copy of the model into VRAM.

## Test Health

```powershell
curl http://127.0.0.1:8000/health
```

## Test Embeddings

```powershell
curl http://127.0.0.1:8000/v1/embeddings `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer change-me" `
  -d "{\"input\":[\"hello world\",\"semantic search\"],\"task\":\"retrieval.passage\",\"dimensions\":1024}"
```

## Settings

Edit `.env`:

```env
EMBEDDING_MODEL=jinaai/jina-embeddings-v3
EMBEDDING_DEVICE=cuda
EMBEDDING_DTYPE=float16
EMBEDDING_TASK=retrieval.passage
EMBEDDING_MAX_LENGTH=2048
EMBEDDING_BATCH_SIZE=32
EMBEDDING_TRUNCATE_DIM=1024
EMBEDDING_NORMALIZE=true
EMBEDDING_API_TOKEN=change-me
```

## API Token

The embeddings endpoint requires a bearer token:

```http
Authorization: Bearer change-me
```

Set `EMBEDDING_API_TOKEN` in `.env` to your real secret token before running the server.

## Tasks

Valid Jina tasks:

```txt
retrieval.query
retrieval.passage
separation
classification
text-matching
```

Use `retrieval.query` for search queries.

Use `retrieval.passage` for documents/chunks stored in vector database.

## Dimensions

Valid dimensions:

```txt
32, 64, 128, 256, 512, 768, 1024
```

Recommended:

```env
EMBEDDING_TRUNCATE_DIM=1024
```

Use `512` if you want smaller vector storage.
