# CHALLENGERS AI Assistant

Local AI assistant for the CHALLENGERS club, combining:
- RAG over club documents (FAISS + sentence-transformers)
- Streaming LLM responses through Ollama
- Face recognition for session personalization
- Short-term conversation memory in the CLI session

## Current Project State

This is what is implemented and working in the current codebase:

1. CLI app entrypoint in `app/main.py`
2. RAG retrieval and LLM streaming in `backend/rag/rag.py`
3. Face recognition flow in `backend/vision/vision.py`
4. Prebuilt FAISS index in `backend/rag/faiss_index/`
5. Saved face embeddings in `backend/vision/embeddings/`

These files currently exist but are placeholders (empty):

1. `app/flask_app.py`
2. `backend/core/controller.py`
3. `backend/config/settings.py`

## Repository Layout

```text
project_alonee/
|- app/
|  |- main.py
|  `- flask_app.py
|- backend/
|  |- config/
|  |  `- settings.py
|  |- core/
|  |  `- controller.py
|  |- rag/
|  |  |- rag.py
|  |  |- store.py
|  |  `- faiss_index/
|  |     |- index.faiss
|  |     `- index.pkl
|  `- vision/
|     |- enroll.py
|     |- recognition.py
|     |- vision.py
|     `- embeddings/
|        |- darshan.npy
|        `- mitra.npy
|- data/
|  `- challengers_scraper.pdf
|- requirements.txt
`- README.md
```

## How the App Works Today

### 1. Chat Mode (RAG + Streaming)

When you choose menu option `1`:

1. User question is captured.
2. If a face was recognized earlier, user name is prepended to query context.
3. Last 5 chat messages are passed as conversation context.
4. Top 3 chunks are retrieved from FAISS.
5. Prompt is built with:
  - retrieved context
  - short conversation history
  - strict answer rules
6. Prompt is sent to Ollama endpoint:
  - URL: `http://localhost:11434/api/generate`
  - model: `phi3`
  - stream: `true`
7. Tokens are streamed live to terminal and also collected into final response text.

### 2. Camera Mode (Face Recognition)

When you choose menu option `2`:

1. Webcam opens.
2. Face detection uses MediaPipe.
3. Face embedding extraction uses `face_recognition`.
4. Embedding is compared against stored embeddings (`.npy` files).
5. Stable recognized name is returned and stored for current session.

### 3. Session Memory

Current memory is process-local only:

1. `user_name` stays in memory while app runs.
2. `chat_history` grows in memory.
3. Only the latest 5 lines are sent to model for continuity.
4. Restarting the program clears memory.

## Prerequisites

1. Python 3.10+ (3.11 recommended)
2. Webcam (for face mode)
3. Ollama running locally
4. Ollama model pulled locally (`phi3`)
5. Internet for first-time embedding model download

## Installation

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `face_recognition` install causes issues on Windows:

```powershell
pip install dlib-bin
pip install face_recognition --no-deps
```

## Ollama Setup

Start Ollama and ensure `phi3` is available:

```powershell
ollama serve
ollama pull phi3
```

Quick health check:

```powershell
curl http://localhost:11434/api/tags
```

## Run the Assistant

```powershell
.\.venv\Scripts\python.exe app\main.py
```

Menu options:

1. Chat
2. Camera (Face Recognition)
3. Exit

Recommended usage:

1. Run Camera first to identify user.
2. Use Chat for personalized RAG responses.

## RAG Index: Current Behavior and Rebuild

Current runtime reads index from:

- `backend/rag/faiss_index/index.faiss`
- `backend/rag/faiss_index/index.pkl`

`backend/rag/store.py` currently uses relative paths (expects PDF and output in current working directory). If you run it from another folder, paths may not match expected runtime location.

If rebuilding index, run from `backend/rag` or update `store.py` to use project-root absolute paths.

## Dependencies (Current)

Main libraries used:

- OpenCV, MediaPipe, face_recognition, dlib-bin
- NumPy, python-dotenv
- LangChain + FAISS + sentence-transformers
- PyMuPDF
- requests

Note: `google-genai` is still listed in `requirements.txt`, but the current chat flow uses Ollama (`phi3`) and does not use Gemini in runtime code.

## Known Limitations

1. No API server yet (`app/flask_app.py` is empty).
2. No orchestration/service layer yet (`backend/core/controller.py` is empty).
3. No centralized settings module yet (`backend/config/settings.py` is empty).
4. No persistent chat memory store (memory resets on restart).
5. No automated tests currently.
6. `backend/rag/store.py` path handling is not yet robust.
7. Model/provider settings are hardcoded in code (`phi3`, Ollama URL).
8. `backend/rag/rag.py` imports `requests`, but `requirements.txt` does not currently list `requests` explicitly.

## Troubleshooting

### 1. Always getting fallback answers

Check these first:

1. Ollama is running and model exists.
2. FAISS index files exist in `backend/rag/faiss_index/`.
3. Your question relates to content from indexed PDF.

### 2. LLM not responding

If CLI shows LLM error, verify:

```powershell
curl http://localhost:11434/api/tags
```

### 3. Camera not opening

1. Close other apps using webcam.
2. Try different camera index in vision code (`cv2.VideoCapture(0)` to `1` or `2`).

### 4. Face stays Unknown

1. Confirm `.npy` files exist in `backend/vision/embeddings/`.
2. Re-enroll using `backend/vision/enroll.py`.
3. Ensure enough light and frontal face angle.

## Future Implementation Roadmap

### Phase 1: Stabilization

1. Centralize config in `backend/config/settings.py`:
  - model name
  - Ollama base URL
  - retrieval parameters (`k`, chunk limits)
  - camera and recognition thresholds
2. Make all file paths project-root relative in all modules.
3. Add structured logging and better error messages.

### Phase 2: Backend Architecture

1. Implement `backend/core/controller.py` as orchestration layer for:
  - chat flow
  - retrieval flow
  - identity/session flow
2. Refactor `app/main.py` into a thin interface layer.
3. Define clear module boundaries for RAG, vision, and session.

### Phase 3: Flask API

1. Implement `app/flask_app.py` endpoints:
  - `POST /chat`
  - `POST /detect-face`
  - `GET /health`
2. Add session handling strategy for API clients.
3. Add CORS, request validation, and error schema.

### Phase 4: Data and Retrieval Quality

1. Improve indexing pipeline (`store.py`) for repeatable rebuilds.
2. Add metadata-aware retrieval (page/source reference).
3. Add retrieval diagnostics and optional score logging.
4. Add query rewriting and answer citation support.

### Phase 5: Reliability and Testing

1. Add unit tests for RAG and parsing logic.
2. Add integration tests for chat flow and face recognition mocks.
3. Add CI checks for lint, test, and dependency health.

### Phase 6: Productization

1. Add authentication and role-based access.
2. Add persistent storage for chat history and user profiles.
3. Containerize app and prepare deployment configs.
4. Add monitoring and runtime metrics.

## Suggested Next Milestones

1. Implement `backend/config/settings.py` and remove hardcoded runtime values.
2. Implement minimal Flask API with `/chat` and `/health`.
3. Refactor `store.py` to robust absolute paths and one-command index rebuild.
4. Add first test suite for `backend/rag/rag.py`.
