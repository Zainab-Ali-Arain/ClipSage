import { useState, useRef, useCallback } from "react";

const API_URL = "http://localhost:8000";
const POLL_INTERVAL_MS = 2500;

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
}

const STAGE_LABEL = {
  queued: "Queued…",
  transcribing: "Transcribing audio (this can take a few minutes)…",
  indexing: "Building searchable index…",
  done: "Ready to ask questions about this video.",
  error: "Something went wrong while processing this video.",
};

export default function App() {
  const [question, setQuestion] = useState("");
  const [exchanges, setExchanges] = useState([]); // { question, answer, sources, error }
  const [loading, setLoading] = useState(false);
  const [activeSource, setActiveSource] = useState(null); // { video_name, start }
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadState, setUploadState] = useState(null); // { video_name, stage, error }
  const [uploading, setUploading] = useState(false);
  const videoRef = useRef(null);

  const playSource = useCallback((source) => {
    setActiveSource(source);
    // wait a tick for the <video> element to mount/update its src, then seek
    requestAnimationFrame(() => {
      if (videoRef.current) {
        videoRef.current.currentTime = source.start;
        videoRef.current.play().catch(() => {});
      }
    });
  }, []);

  function pollStatus(videoName) {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/status/${videoName}`);
        const data = await res.json();
        setUploadState(data);
        if (data.stage === "done" || data.stage === "error") {
          clearInterval(interval);
        }
      } catch {
        clearInterval(interval);
        setUploadState({ video_name: videoName, stage: "error", error: "Lost connection to backend." });
      }
    }, POLL_INTERVAL_MS);
  }

  async function handleUpload(e) {
    e.preventDefault();
    if (!uploadFile || uploading) return;

    setUploading(true);
    setUploadState(null);

    try {
      const formData = new FormData();
      formData.append("file", uploadFile);

      const res = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();

      setUploadState({ video_name: data.video_name, stage: "queued", error: null });
      pollStatus(data.video_name);
      setUploadFile(null);
    } catch (err) {
      setUploadState({ stage: "error", error: "Couldn't reach the backend to upload." });
    } finally {
      setUploading(false);
    }
  }

  async function handleAsk(e) {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setQuestion("");

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();

      const entry = {
        question: trimmed,
        answer: data.answer,
        sources: data.sources || [],
      };
      setExchanges((prev) => [...prev, entry]);

      if (data.best_timestamp) {
        playSource(data.best_timestamp);
      }
    } catch (err) {
      setExchanges((prev) => [
        ...prev,
        {
          question: trimmed,
          error:
            "Couldn't reach ClipSage's backend. Make sure the API server (uvicorn) is running on port 8000.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header__mark">
          <h1 className="header__title">
            Clip<span className="header__title-accent">Sage</span>
          </h1>
        </div>
        <p className="header__tagline">
          Ask a question about your lecture videos and get an answer with
          the exact moment it was taught, highlighted like a line in a
          textbook.
        </p>
      </header>

      <main className="main">
        <form className="upload-panel" onSubmit={handleUpload}>
          <span className="upload-panel__label">
            Add a lecture video to index it for questions
          </span>
          <input
            type="file"
            accept=".mp4,.mov,.mkv,.webm"
            onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
            disabled={uploading}
          />
          <button type="submit" disabled={!uploadFile || uploading}>
            {uploading ? "Uploading…" : "Upload & Process"}
          </button>

          {uploadState && (
            <span
              className={
                "upload-status" +
                (uploadState.stage === "done" ? " upload-status--done" : "") +
                (uploadState.stage === "error" ? " upload-status--error" : "")
              }
            >
              {uploadState.stage !== "done" && uploadState.stage !== "error" && (
                <span className="thinking__dot" />
              )}
              {uploadState.video_name ? `${uploadState.video_name}: ` : ""}
              {uploadState.error || STAGE_LABEL[uploadState.stage] || uploadState.stage}
            </span>
          )}
        </form>

        <form className="ask-bar" onSubmit={handleAsk}>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. what is a stack?"
            disabled={loading}
            autoFocus
          />
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? "Thinking…" : "Ask"}
          </button>
        </form>

        {exchanges.length === 0 && !loading && (
          <div className="empty-state">
            <p className="empty-state__title">Nothing asked yet</p>
            <p className="empty-state__body">
              Type a question about a topic covered in one of your indexed
              lectures. ClipSage will search the transcript, answer using
              only what was actually said, and jump the player to that exact
              moment.
            </p>
          </div>
        )}

        {exchanges.map((ex, i) => (
          <div className="exchange" key={i}>
            <p className="exchange__question">{ex.question}</p>

            {ex.error ? (
              <div className="error-banner">{ex.error}</div>
            ) : (
              <div className="exchange__answer-card">
                <p className="exchange__answer-text">{ex.answer}</p>

                {ex.sources.length > 0 ? (
                  <div className="exchange__sources">
                    {ex.sources.map((s, j) => (
                      <button
                        key={j}
                        className={
                          "source-pill" + (j > 0 ? " source-pill--secondary" : "")
                        }
                        onClick={() => playSource(s)}
                      >
                        ▸ {s.video_name} @ {formatTime(s.start)}
                      </button>
                    ))}
                  </div>
                ) : (
                  <span className="exchange__no-source">
                    No matching lecture segment found yet.
                  </span>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="thinking">
            <span className="thinking__dot" />
            <span className="thinking__dot" />
            <span className="thinking__dot" />
            Searching the lecture and drafting an answer…
          </div>
        )}
      </main>

      {activeSource && (
        <div className="player-dock">
          <div className="player-dock__inner">
            <p className="player-dock__label">
              <span>
                {activeSource.video_name} — from {formatTime(activeSource.start)}
              </span>
              <button
                className="player-dock__close"
                onClick={() => setActiveSource(null)}
              >
                Hide player
              </button>
            </p>
            <video
              ref={videoRef}
              key={activeSource.video_name}
              controls
              src={`${API_URL}/videos/${activeSource.video_name}.mp4`}
            />
          </div>
        </div>
      )}
    </div>
  );
}
