# ClipSage: A Study Assistant Built from YouTube Lectures

*Machine Learning semester project, BS Computer Science, Semester V*

A student types a topic, such as *overfitting*, *binary search tree* or *TCP handshake*. ClipSage searches a collection of educational YouTube lectures and returns three things:

1. **A short explanation** of 4–6 sentences, starting with a definition.
2. **References** to the exact videos the explanation came from, each a clickable link that opens the video at the right second.
3. **A visual representation**: a concept map of related terms, a keyword cloud, a timeline of the best video segments, sub-topic groups and a list of key terms for revision.

The project follows the proposal step by step. [`docs/proposal_mapping.md`](docs/proposal_mapping.md) shows where each item of the proposal is implemented.

### Study features (version 3)

ClipSage covers **exactly six subjects**: Operating Systems, Machine Learning, Linear Algebra, Compiler Construction, Web Engineering and Computer Architecture (45 written topics in total).

| Page | What it does |
|---|---|
| **Dashboard** | Only a search bar and the **Study Spaces** dropdown with the six subjects. A question typed in the search bar opens the matching subject's chat with the answer |
| **Study space** (one per subject) | A **chat**: ask anything about the subject in English or Roman Urdu. Each answer shows the explanation on the left and the lecture video with caption timestamps on the right, plus *Full lesson*, *Practice quiz* and *Save Topic* buttons and related topics |
| **⋮ menu** in each study space | **Practice Quiz** (built from the topics you discussed in the chat, with generated numerical problems), Topics & Lessons, Algorithm Practice, Numerical Practice, Video References, Overview & Progress, Saved, Clear chat |
| **Video Library** | Every collected lecture by subject and topic, timeline of sections, jump-to-time, *"What is taught between 10:20 and 15:30?"* and questions about a lecture |
| **Mathematics / ML Algorithm Playground** | Step-by-step maths (verified with SymPy/NumPy/SciPy) and ML algorithms with intermediate calculations and plots |
| **Saved Data / Settings** | Per-subject saved items (view and delete), response language and retrieval settings |

The sidebar (hidden by default, open it with the arrow at the top left) links to the Video Library, both playgrounds, Saved Data and Settings. The chat history is kept only in the browser session.

Practice tools per subject (all answers are computed, never typed in): CPU scheduling (FCFS, SJF, SRTF, Priority, Round Robin), page replacement (FIFO, LRU, OPT), paging address translation, Banker's algorithm; two's complement, IEEE 754, cache simulator, 5-stage pipeline with hazards, CPI and Amdahl's law; lexer, FIRST/FOLLOW and LL(1) table, SLR(1) shift-reduce parser, three-address code; Gaussian elimination and matrix tools; a live HTML/CSS/JS editor and an HTTP request builder; the ML playground. Quizzes have MCQ, true/false, conceptual short-answer and numerical questions with easy/medium/hard levels.

**Language.** The interface is always English. The *answer* follows the language of your question: ClipSage detects English or Roman Urdu with a word lexicon plus a character n-gram Naive Bayes classifier trained on the bilingual lessons (about 96% cross-validated accuracy). Settings can force English or Roman Urdu. Roman Urdu text was written by hand; lecture transcripts stay in English.

**Videos are never invented.** Every video comes from the YouTube Data API run on your computer, and every timestamp comes from that video's captions. If no reliable video or timestamp exists for a topic, ClipSage says so instead of guessing.

**Nothing is saved automatically.** Searches, answers and notes stay in the browser session. Data is written only when you click a Save button (Save Topic, Save Notes, Save Quiz Results, Save Progress, Save Learning Material, Save result), and each subject has its own file in `data/user/subjects/`.

**Source labels:** 🎥 *From lectures* (real transcript with timestamps), 📘 *ClipSage notes* (written lesson), 🧮 *Computed result*, *Sample data* (synthetic, clearly marked).

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

Double-click **`run_pipeline.bat`**, or run each step yourself:

```bash
python -m clipsage.collect                   # Stage 1: topic video library (2 videos per topic, all six subjects)
python -m clipsage.preprocess                # Stage 2
python -m clipsage.index                     # Stage 3 (downloads the SBERT model once)
python -m clipsage.explore                   # dataset report  -> reports/dataset_report.md
python -m clipsage.topics --lda-topics 10    # LDA topics      -> reports/lda_topics.md
```

Stage 1 searches YouTube with the `video_queries` of every lesson in `knowledge/*.json` and keeps only videos that have English captions, last 3 minutes to 3 hours, match the topic and are not duplicates. Useful options:

- `--per-topic 3`: more videos per topic (default 2, which gives about 90 videos)
- `--subject "Operating Systems"`: only one subject (repeatable)
- `--channels`: the old mode that scans the channels in `config/sources.json`

Notes on Stage 1:

- **Quota.** The free quota is 10,000 units per day and each search costs 100 units, so the 45 topics may need two days. The script stops safely before the limit and caches every search in `data/raw/library_searches.json`; run it again the next day and it continues where it stopped.
- **Transcripts** are fetched with `youtube-transcript-api`. Human-made English captions are preferred over auto-generated ones. Videos without English captions are skipped and logged in `data/raw/collection_log.csv`.
- **Blocked requests.** If YouTube starts refusing transcript requests (`IpBlocked`), wait a few hours or switch to another connection, then continue with `python -m clipsage.collect --skip-metadata`. YouTube blocks cloud servers, so collect from a home or university connection.

## Run the app

Double-click **`run_app.bat`**, or run:

```bash
streamlit run app.py
```

Then open http://localhost:8501.

The dashboard shows a search bar and the Study Spaces dropdown. Choose a subject to open its chat; the *Dashboard* button returns. Lessons, quizzes and all practice tools work even before you collect any data. Video references and timestamps need the library built by `run_pipeline.bat`.

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

The 178 tests use a small **synthetic** corpus with fake video IDs and a stand-in embedder, so they need neither the API key nor the SBERT download. They cover:

- every pipeline stage
- every lesson's Python code (it is executed)
- every quiz answer and practice problem
- every maths solver, checked against NumPy, SymPy and SciPy
- every playground algorithm, checked against scikit-learn
- lecture navigation and time-range explanations
- language detection, query analysis and the six subjects
- subject chat replies and the practice quiz built from the discussed topics
- every OS, architecture, compiler and linear algebra simulator, and the numerical practice generators
- the video library (relevance filter, cache, no unrelated videos in the video panel)
- explicit per-subject saving (no file is written before Save)

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
  lessons.py               lesson = written notes + what the lectures say
  lectures.py              segment references, lecture timeline, time-range explain, ask a lecture
  quiz.py                  grading (MCQ / numerical / SBERT short answers), lecture fill-in-the-blank
  mathsolve.py             step-by-step maths with SymPy/NumPy verification
  playground.py            ML algorithms with intermediate calculations and plots
  knowledge.py, i18n.py    bilingual lessons loader, Roman Urdu text helpers
  subjects.py              the six subjects (config/subjects.json) and their topics
  langdetect.py            English / Roman Urdu detection (lexicon + char n-gram Naive Bayes)
  search.py                query analysis: topic, subject, language
  chat.py                  subject chat replies, practice quiz from the discussed topics
  library.py               topic video library, video panel, caption timestamps of sub-topics
  simulators.py            OS, architecture, compiler and linear algebra algorithms
  practice_gen.py          numerical practice problems per subject
  progress.py              session activity (memory only) + per-subject saved data
  ui/                      dashboard, subject environment, search, video library, playgrounds, tools
knowledge/                 one JSON file per subject: bilingual lessons, questions, video queries
.streamlit/config.toml     light-blue theme
config/                    subjects.json (six subjects and their tools), sources.json, topics.json
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
- Nothing is saved automatically. Items saved with a Save button are stored only on the student's own computer in `data/user/subjects/<subject>.json`. They are never uploaded, and Saved Data → Manage deletes them.
- ClipSage is a study aid. Students should watch the referenced lectures.

## Team roles

| Member | Modules |
|---|---|
| Member 1: data | `collect.py`, `preprocess.py`, `explore.py`, `config/` |
| Member 2: ML / NLP | `embeddings.py`, `index.py`, `retrieve.py`, `summarize.py`, `keywords.py`, `topics.py`, `evaluate.py` |
| Member 3: visuals / interface | `visualize.py`, `app.py`, `pipeline.py`, tests, demo |
