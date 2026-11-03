# ClipSage: A Study Assistant Built from YouTube Lectures

*Machine Learning semester project, BS Computer Science, Semester V*

A student types a topic, such as *overfitting*, *binary search tree* or *TCP handshake*. ClipSage searches a collection of educational YouTube lectures and returns three things:

1. **A short explanation** of 4–6 sentences, starting with a definition.
2. **References** to the exact videos the explanation came from, each a clickable link that opens the video at the right second.
3. **A visual representation**: a concept map of related terms, a keyword cloud, a timeline of the best video segments, sub-topic groups and a list of key terms for revision.

The project follows the proposal step by step. [`docs/proposal_mapping.md`](docs/proposal_mapping.md) shows where each item of the proposal is implemented.

---

## How it works (7-stage pipeline)

```
 YouTube Data API v3 ──► 1. collect.py     video metadata + timestamped transcripts  ─► data/raw/
                         2. preprocess.py  clean text, sentences, 30–60 s overlapping segments ─► data/processed/
                         3. index.py       TF-IDF (baseline) + Sentence-BERT (improved) + FAISS ─► data/index/
 student's topic ──────► 4. retrieve.py    cosine similarity → MMR re-ranking (different videos)
                         5. summarize.py   extractive explanation (top sentences / TextRank / embedding),
                                           optional BART/T5; every sentence keeps video + timestamp
                         6. keywords.py    TF-IDF / KeyBERT key terms     topics.py  K-Means sub-topics, LDA
                            visualize.py   concept map (NetworkX + PyVis), keyword cloud, timeline (Plotly)
                         7. app.py         Streamlit web page
```

| Task | Baseline | Improved method |
|---|---|---|
| Text representation | TF-IDF | Sentence-BERT embeddings (`all-MiniLM-L6-v2`) |
| Retrieval | Keyword matching with cosine similarity | Semantic search with cosine similarity (FAISS) |
| Explanation | Top-ranked sentences | TextRank / embedding-based extractive summary (optional: BART or T5) |
| Keyword extraction | TF-IDF scores | KeyBERT |
| Topic grouping (extra) | – | K-Means sub-topics per query, LDA over the corpus |
| Visualization | Keyword cloud | Concept graph (co-occurrence network) |

You can switch every baseline/improved choice from the app's sidebar, so both can be shown in the demo.

---

## Setup (Windows)

1. Install **Python 3.11** from python.org. Tick "Add to PATH" during install.
2. Double-click **`setup.bat`**. It creates `venv/`, installs `requirements.txt` and creates a `.env` file.
3. Get a free **YouTube Data API key**:
   1. Open the Google Cloud Console and create a new project.
   2. Go to APIs & Services → Library and enable **YouTube Data API v3**.
   3. Go to Credentials → Create credentials → **API key**.
4. Open `.env` in Notepad and paste the key: `YOUTUBE_API_KEY=...`

Mac / Linux:

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then paste your key into .env
```

## Build the dataset and index

1. Check `config/sources.json`. It lists the trusted channels and playlists. Open each `@handle` on YouTube to confirm it is the right channel, and add playlist IDs for lecture series you want.
2. Check `config/topics.json`. It lists the subjects and the keywords used to select videos.
3. Double-click **`run_pipeline.bat`**, or run each step yourself:

```bash
python -m clipsage.collect --max-videos 50   # Stage 1 (try a small run first; no flag = up to 500 videos)
python -m clipsage.preprocess                # Stage 2
python -m clipsage.index                     # Stage 3 (downloads the SBERT model once)
python -m clipsage.explore                   # dataset report  -> reports/dataset_report.md
python -m clipsage.topics --lda-topics 10    # LDA topics      -> reports/lda_topics.md
```

Notes on Stage 1:

- **Quota.** The free quota is 10,000 units per day. Listing a channel costs 1 unit per 50 videos, a search costs 100 units. The script counts the units it uses and stops safely before the limit. Everything is cached in `data/raw/`, so re-running only fetches what is missing.
- **Transcripts** are fetched with `youtube-transcript-api`. Human-made English captions are preferred over auto-generated ones. Videos without English captions are skipped and logged in `data/raw/collection_log.csv`.
- **Blocked requests.** If YouTube starts refusing transcript requests (`IpBlocked`), wait a few hours or switch to another connection, then continue with `python -m clipsage.collect --skip-metadata`. YouTube blocks cloud servers, so collect from a home or university connection.

## Run the app

Double-click **`run_app.bat`**, or run:

```bash
streamlit run app.py
```

Then open http://localhost:8501.

You can also query from the command line:

```bash
python -m clipsage.pipeline "overfitting" --method sbert --summarizer textrank
```

## Evaluation (proposal section 10)

| What is measured | Command | Output |
|---|---|---|
| Retrieval quality: Precision@5, Recall@10, MRR, TF-IDF vs SBERT, plus a Wilcoxon significance test | `pool`, then label by hand, then `retrieval` | `reports/retrieval_results.json`, `retrieval_comparison.png` |
| Summary quality: ROUGE-1/2/L for every explanation method, plus a manual 1–5 rating sheet | `summary` | `reports/summary_results.json`, `evaluation/summary_ratings.csv` |
| Key-term quality: manual check (target ≥ 80% relevant) | `keywords` | `evaluation/keyword_review.csv` |
| Source accuracy: does each link and timestamp point to the quoted text? | `sources` (`--online` also checks the videos still exist) | `reports/source_results.json` |
| Speed: time from query to result (target < 5 s) | `speed` | `reports/speed_results.json` |
| Usability survey with 10–15 students | fill `evaluation/survey_responses.csv` (questions in `docs/usability_survey.md`) | – |
| Everything in one report | `report` | `reports/evaluation_report.md` |

Run each with `python -m clipsage.evaluate <command>`, or `run_evaluation.bat <command>`.

Workflow:

1. Run `python -m clipsage.evaluate pool`. This writes the top-10 videos of both methods for each of the 30 test queries (`evaluation/queries.json`), shuffled so you can't tell which method found which video.
2. Open `evaluation/relevance_labels.csv` and put **1** (the video explains the topic) or **0** in the `relevant` column.
3. Run `python -m clipsage.evaluate all`. It runs the retrieval, summary, sources and speed evaluations, then writes `reports/evaluation_report.md`.
4. Fill in `summary_ratings.csv`, `keyword_review.csv` (create it with `python -m clipsage.evaluate keywords`) and `survey_responses.csv`.
5. Run `python -m clipsage.evaluate report` again.

The reference summaries for ROUGE are in `evaluation/reference_summaries.json`. Review them as a team before reporting the scores.

## Tests

```bash
pytest -q
```

The tests use a small **synthetic** corpus with fake video IDs and a stand-in embedder. They check every stage without needing the API key or the SBERT download.

## Project structure

```
app.py                     Streamlit interface (Stage 7)
clipsage/
  config.py                all settings (paths, segment length, thresholds, models)
  collect.py               Stage 1 - YouTube Data API + transcripts
  preprocess.py            Stage 2 - cleaning, sentences, 30-60 s segments
  embeddings.py, index.py  Stage 3 - TF-IDF, Sentence-BERT, FAISS
  retrieve.py              Stage 4 - search, MMR diversity re-ranking, "topic not covered"
  summarize.py             Stage 5 - extractive (3 methods) + optional BART/T5
  keywords.py              TF-IDF / KeyBERT key terms
  topics.py                K-Means sub-topics, corpus LDA
  visualize.py             Stage 6 - concept map, keyword cloud, timeline
  pipeline.py              ties everything together (also a CLI)
  evaluate.py              evaluation plan
  explore.py               dataset report
config/                    sources.json (channels/playlists), topics.json (subjects)
evaluation/                30 test queries, reference summaries, labelling sheets
docs/                      proposal mapping, usability survey
reports/                   generated results and charts
tests/                     automated tests
legacy/whisper_rag_prototype/   the earlier upload-a-video prototype (kept for reference)
```

## Ethics

- ClipSage uses only the official YouTube Data API and publicly available captions.
- Videos are **never downloaded or re-hosted**. The app only links to youtube.com, and the embedded player is YouTube's own.
- Every explanation sentence credits its video and channel.
- No user data is stored. Queries are not logged.
- ClipSage is a study aid. Students should watch the referenced lectures.

## Team roles

| Member | Modules |
|---|---|
| Member 1: data | `collect.py`, `preprocess.py`, `explore.py`, `config/` |
| Member 2: ML / NLP | `embeddings.py`, `index.py`, `retrieve.py`, `summarize.py`, `keywords.py`, `topics.py`, `evaluate.py` |
| Member 3: visuals / interface | `visualize.py`, `app.py`, `pipeline.py`, tests, demo |
