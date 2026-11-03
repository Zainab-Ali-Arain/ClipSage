"""
main.py
Step 4 - the backend API. Wraps rag.py so the frontend can call it over HTTP.

Run with:
    uvicorn main:app --reload

Then open http://localhost:8000/docs to test endpoints interactively.
"""

import os
import re
import shutil
import traceback

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import answer_question
import transcribe
import build_index

app = FastAPI(title="ClipSage API")

# allow the frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve the raw video files so the frontend <video> tag can play them directly
# (absolute path so this works no matter which folder uvicorn was started from)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)
app.mount("/videos", StaticFiles(directory=DATA_DIR), name="videos")

# In-memory job tracker: { video_name: {"stage": ..., "error": ... } }
# A dict is fine for a single-user student project - no need for a real
# task queue / database here.
PROCESSING_STATUS = {}


class Question(BaseModel):
    question: str


def _safe_filename(name: str) -> str:
    """Strip anything that isn't a safe filename character, keep the extension."""
    base, ext = os.path.splitext(name)
    base = re.sub(r"[^A-Za-z0-9_\-]", "_", base)
    return f"{base}{ext}"


def _process_uploaded_video(video_path: str, video_name: str):
    """Runs in the background after /upload returns, so the browser doesn't hang."""
    try:
        PROCESSING_STATUS[video_name] = {"stage": "transcribing", "error": None}
        _, segments = transcribe.process_video(video_path)

        PROCESSING_STATUS[video_name] = {"stage": "indexing", "error": None}
        build_index.index_transcript(video_name, video_path, segments)

        PROCESSING_STATUS[video_name] = {"stage": "done", "error": None}
    except Exception as e:
        traceback.print_exc()
        PROCESSING_STATUS[video_name] = {"stage": "error", "error": str(e)}


@app.get("/")
def root():
    return {"status": "ClipSage API is running"}


@app.post("/upload")
def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a video file from the browser, saves it into data/, and kicks off
    transcription + indexing in the background. Poll /status/{video_name}
    to know when it's ready to be asked about.
    """
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
        raise HTTPException(400, "Please upload a video file (.mp4, .mov, .mkv, .webm)")

    safe_name = _safe_filename(file.filename)
    video_name = os.path.splitext(safe_name)[0]
    dest_path = os.path.join(DATA_DIR, safe_name)

    with open(dest_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    PROCESSING_STATUS[video_name] = {"stage": "queued", "error": None}
    background_tasks.add_task(_process_uploaded_video, dest_path, video_name)

    return {"video_name": video_name, "status": "queued"}


@app.get("/status/{video_name}")
def get_status(video_name: str):
    status = PROCESSING_STATUS.get(video_name)
    if status is None:
        raise HTTPException(404, "No such video job. Has it been uploaded?")
    return {"video_name": video_name, **status}


@app.post("/ask")
def ask(payload: Question):
    """Main endpoint: student sends a question, gets back an answer + timestamp."""
    result = answer_question(payload.question)
    return result
