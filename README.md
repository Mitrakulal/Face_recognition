# Face Recognition + RAG Backend Project

This project contains two working AI modules organized for future backend integration with Flask:

- Face recognition module for enrollment and realtime identity matching
- RAG module for PDF-based question answering using FAISS and Gemini

The folder structure is already prepared for future API development in Flask.

## Current Project Status

- Face module is runnable from backend/face_system
- RAG module is runnable from backend/rag_system
- Flask integration folder is ready at backend/flask_app (currently empty)

## Project Structure

```text
project_alonee/
|- backend/
|  |- face_system/
|  |  |- Face_enroll.py
|  |  |- Face_Recognition.py
|  |  `- embeddings/
|  |- rag_system/
|  |  |- app.py
|  |  |- store.py
|  |  |- challengers_scraper.pdf
|  |  `- faiss_index/
|  |     |- index.faiss
|  |     `- index.pkl
|  `- flask_app/
`- README.md
```

## What Each Module Does

### Face System

- Enrolls a person by capturing multiple stable face embeddings
- Saves embeddings into .npy files inside embeddings/
- Performs realtime recognition with distance thresholding and temporal smoothing

### RAG System

- Reads PDF source content
- Splits text into chunks
- Builds FAISS vector index for retrieval
- Uses retrieved context with Gemini for question answering

## Prerequisites

- Python 3.11 recommended
- Webcam required for face enrollment/recognition
- Gemini API key for RAG chatbot

## Environment Setup (Windows PowerShell)

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Install Dependencies

Install core dependencies:

```powershell
pip install opencv-python opencv-contrib-python mediapipe==0.10.14 numpy
pip install dlib-bin
pip install face_recognition --no-deps
pip install python-dotenv langchain langchain-community langchain-text-splitters sentence-transformers faiss-cpu pymupdf google-genai
```

## Configure Environment Variables

Create a .env file in project root:

```env
GEMINI_API_KEY=your_api_key_here
```

## Run Face Enrollment

```powershell
cd backend/face_system
python Face_enroll.py
```

Controls:

- Press e to start enrollment
- Enter person name
- Keep face stable for capture
- Press q to quit

Output:

- Embeddings are saved as embeddings/<name>.npy

## Run Face Recognition

```powershell
cd backend/face_system
python Face_Recognition.py
```

Important:

- Enroll at least one person before running recognition
- If embeddings folder is empty, recognition may fail

## Build RAG Index

```powershell
cd backend/rag_system
python store.py
```

This generates/updates:

- faiss_index/index.faiss
- faiss_index/index.pkl

## Run RAG Chatbot

```powershell
cd backend/rag_system
python app.py
```

You can then ask questions in terminal using retrieved PDF context.

## Common Issues

### No module named cv2

- Activate the correct virtual environment
- Confirm interpreter points to .venv/Scripts/python.exe

### No module named face_recognition

- Install dlib-bin first, then install face_recognition with --no-deps

### mediapipe has no attribute solutions

- Use mediapipe==0.10.14 in this project

### Camera not opened

- Close other apps using webcam
- Try changing cv2.VideoCapture(0) to 1 or 2

### ValueError in Face_Recognition argmin

- This happens when no enrolled embeddings exist
- Run Face_enroll.py first

## Future Development Direction

The repository is prepared to combine both modules under Flask backend:

- Move face and RAG logic into reusable services
- Add REST endpoints in backend/flask_app
- Add authentication, logging, and deployment configs

## Author

Mitra Kulal
