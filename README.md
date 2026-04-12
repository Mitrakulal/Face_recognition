# CHALLENGERS AI Assistant

RAG + Face Recognition + Session Memory assistant for CHALLENGERS club.

## What It Does

- Chat with club knowledge using FAISS retrieval + Gemini response generation.
- Detect known users from webcam and store identity in session.
- Keep short-term conversation memory (last 5 messages) for continuity.
- Inject detected user identity into chat queries when available.

## Current Architecture

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
|- data/
|  `- challengers_scraper.pdf
|- requirements.txt
`- README.md
```

## Memory Behavior

- `user_name` is stored in session after successful camera detection.
- `chat_history` stores user/assistant/system messages.
- Only the latest 5 messages are sent to the LLM:
  - `history_text = "\n".join(chat_history[-5:])`
- RAG retrieval remains fixed at `k=3`.

## Prerequisites

- Python 3.11 recommended
- Webcam for face detection/enrollment
- Gemini API key
- Internet on first run for sentence-transformer model download

## Setup (Windows PowerShell)

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `face_recognition` install fails:

```powershell
pip install dlib-bin
pip install face_recognition --no-deps
```

Create `.env` in project root:

```env
GEMINI_API_KEY=your_api_key_here
```

## Build / Refresh RAG Index

`store.py` currently loads `challengers_scraper.pdf` from the working directory.  
If you rebuild index, ensure the PDF is accessible to that script path.

Current files used by runtime:
- PDF source: `data/challengers_scraper.pdf`
- Index used by app: `backend/rag/faiss_index/`

## Run Assistant

Always use project virtual env interpreter:

```powershell
.\.venv\Scripts\python.exe app\main.py
```

Menu:
- `1` Chat mode
- `2` Camera mode (detect user and store identity)
- `3` Exit

Recommended flow:
1. Run camera mode first to detect a known face.
2. Use chat mode for memory-aware responses.

## Key Runtime Notes

- `app/main.py` now resolves FAISS index path relative to project root.
- `backend/vision/vision.py` now resolves embeddings path relative to its module folder.
- Chat continuity is session-based only (memory resets when program restarts).

## Troubleshooting

### `No module named cv2` or `No module named face_recognition`

You are likely using the wrong Python interpreter.

Check interpreter:

```powershell
python -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

Run app with:

```powershell
.\.venv\Scripts\python.exe app\main.py
```

### HuggingFace model download/network errors

On first run, embedding model may attempt download from `huggingface.co`.  
If blocked by network/firewall, app startup will fail before menu appears.

### Camera not opened

- Close other webcam apps.
- Try changing camera index in vision scripts (`cv2.VideoCapture(0)` to `1` or `2`).

### Recognition always unknown

- Ensure `.npy` files exist in `backend/vision/embeddings/`.
- Re-enroll using `backend/vision/enroll.py` if needed.

## Next Planned Work

- Flask API integration in `app/flask_app.py`
- Move orchestration into `backend/core/controller.py`
- Centralize configs in `backend/config/settings.py`
- Add auth, logging, and deployment setup
