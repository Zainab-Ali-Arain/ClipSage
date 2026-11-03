# ClipSage - Run Instructions (Simple Version)

## Folder Location
Extract this zip anywhere you like, e.g.:
```
C:\Users\MUREED\Downloads\ClipSage
```
Just remember that exact path - you'll need it below. Everything from
here on assumes you are inside THIS folder (the one with `setup.bat` in it).

## One-Time Setup

1. Make sure **Python 3.11**, **Node.js**, **FFmpeg**, and **Ollama** are
   already installed (Day 1 setup from before).
2. Pull the local LLM model (only needed once):
   ```
   ollama pull phi3
   ```
3. Double-click **`setup.bat`** in this folder.
   - This creates the virtual environment and installs all Python packages.
   - It takes a few minutes the first time (downloads ~1-2 GB, mainly `torch`).
   - If it says Python 3.11 wasn't found, install it first, then re-run.

## Processing a Lecture Video

You now have two ways to index a video - use whichever is easier:

**Option A: Upload from the browser (recommended)**
Once the backend and frontend are both running, use the "Add a lecture
video" panel at the top of the page - pick a `.mp4` and click
**Upload & Process**. You'll see live status (transcribing → indexing →
ready) without touching the terminal.

**Option B: Drag-and-drop script (no backend needed)**
1. Find a short **.mp4** test video (2-5 minutes is fastest while testing).
2. **Drag and drop** the video file onto **`process_video.bat`**.
   - This runs transcription (Whisper) then builds the searchable index.
   - Watch the console window - it'll tell you when each step finishes.
3. Repeat for as many videos as you want indexed.

## Running the App

**If you already had the backend running from before:** stop it (Ctrl+C)
and start it again so it picks up the new `/upload` endpoint - no new
packages needed, `python-multipart` was already in requirements.txt.

1. Make sure Ollama is running (it usually runs automatically after install;
   if not, run `ollama serve` in a separate terminal).
2. Double-click **`run_backend.bat`** - leave this window open.
   You should see:
   ```
   Uvicorn running on http://127.0.0.1:8000
   ```
3. Double-click **`run_frontend.bat`** - leave this window open too.
   - First run installs npm packages automatically (needs Node.js installed).
   - It will print a local address, usually `http://localhost:5173`.
4. Open that address in your browser, type a question, and click **Ask**.
   Click any timestamp pill under an answer to jump the docked video player
   to that exact moment.

## If Something Breaks

- **`setup.bat` says Python 3.11 not found** -> install it from
  python.org, then run `setup.bat` again.
- **"site can't be reached" in browser** -> the `run_backend.bat` window
  isn't open/running, or it showed an error - check that window first and
  copy any red error text.
- **Whisper is slow** -> normal on CPU. Use a short (2-5 min) video while developing.
- **Ollama connection refused** -> run `ollama serve` in a separate terminal,
  and confirm `ollama pull phi3` finished successfully.
- **Video doesn't play in the browser** -> the video file must physically
  sit inside the `data` folder for the player to find it (process_video.bat
  does NOT copy it there automatically - copy your video into `data\` too,
  or point transcribe at a video already inside `data\`).
- **`npm` not recognized / frontend won't start** -> Node.js isn't installed
  or isn't on PATH - install it from nodejs.org, then open a new terminal.
- **Frontend loads but "Ask" does nothing / network error** -> the backend
  must be running on port 8000 first (`run_backend.bat`) before you use
  the frontend.

## Evaluation Metrics (For the ML Presentation)

This measures two things that matter for the project's core claim - that
it finds the *right* moment in a lecture, not just *a* moment:

- **Retrieval accuracy (Recall@1 / Recall@3)** - does the correct
  segment show up in the top-1 / top-3 results from semantic search?
- **Timestamp precision** - how many seconds off is the returned
  timestamp from the true moment the concept was actually taught?

### Steps

1. Make sure at least one video is already processed (transcribed +
   indexed) - see above.
2. Open `backend/eval_questions.json`. For each test question, add:
   - `question` - what you'd ask
   - `video_name` - must exactly match the indexed video's name
   - `expected_start` / `expected_end` - the true time range (in
     seconds) where that concept is actually explained. Get this by
     opening the video and noting the timestamp yourself, or checking
     `data/<video_name>_transcript.json`.
   - Aim for **at least 8-10 questions** spread across different parts
     of the video(s) - a single question isn't a credible evaluation.
3. Run it:
   ```
   cd backend
   python eval.py
   ```
4. This prints a results table to the terminal and saves:
   - `eval_results.json` - raw per-question numbers
   - `eval_report.png` - a bar chart + histogram, ready to drop into
     your slides

Be upfront in the presentation that this is a small, hand-labeled test
set (not a large public benchmark) - that's normal and expected for a
semester project, and framing it honestly is better than overstating it.

## What's Next (After Today)
- Polish loading/empty/error states further and rehearse the demo flow.
