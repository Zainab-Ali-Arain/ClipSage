"""
transcribe.py
Step 1 of the ClipSage pipeline.

Takes a video file, extracts its audio, and runs Whisper to produce a
TIMESTAMPED transcript (list of segments, each with start_time, end_time, text).

Usage:
    python transcribe.py path/to/lecture.mp4

Output:
    Saves data/<video_name>_transcript.json
"""

import os
import sys
import json
import subprocess
import whisper

# Always write into <project_root>/data, no matter where this script is run from.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def extract_audio(video_path: str, audio_path: str):
    """Use ffmpeg to pull a mono 16kHz wav out of the video (Whisper likes this format)."""
    command = [
        "ffmpeg", "-y",              # -y = overwrite output if it exists
        "-i", video_path,
        "-ar", "16000",              # 16kHz sample rate
        "-ac", "1",                  # mono channel
        "-vn",                       # no video
        audio_path,
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


_model = None


def _get_model(model_size: str = "base"):
    global _model
    if _model is None:
        print(f"Loading Whisper model '{model_size}' (first run downloads it, then it's cached)...")
        _model = whisper.load_model(model_size)
    return _model


def transcribe_audio(audio_path: str, model_size: str = "base"):
    """Run Whisper locally (no API, no internet needed after model is downloaded once)."""
    model = _get_model(model_size)

    print("Transcribing... this can take a few minutes on CPU, be patient.")
    result = model.transcribe(audio_path, verbose=False)

    # result["segments"] already has start/end/text per chunk - exactly what we need
    segments = [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        }
        for seg in result["segments"]
    ]
    return segments


def process_video(video_path: str):
    """
    Reusable core of the pipeline (extract audio -> transcribe -> save JSON).
    Used by both the CLI (main() below) and the /upload API endpoint.

    Returns (video_name, segments).
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    os.makedirs(DATA_DIR, exist_ok=True)
    audio_path = os.path.join(DATA_DIR, f"{video_name}.wav")
    output_path = os.path.join(DATA_DIR, f"{video_name}_transcript.json")

    extract_audio(video_path, audio_path)
    segments = transcribe_audio(audio_path, model_size="base")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"video_name": video_name, "video_path": video_path, "segments": segments}, f, indent=2)

    return video_name, segments


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"[1/2] Extracting audio from {video_path} ...")
    print("[2/2] Running speech-to-text (Whisper) ...")
    video_name, segments = process_video(video_path)

    print(f"\nDone. Transcript saved for '{video_name}'.")
    print(f"Total segments: {len(segments)}")


if __name__ == "__main__":
    main()
