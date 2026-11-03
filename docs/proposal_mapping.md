# Proposal → implementation

This table maps every item of the ClipSage proposal to the code that implements it.

## 3. Objectives

| # | Objective | Where |
|---|---|---|
| 1 | Build a corpus of lecture videos and transcripts with the YouTube Data API | `clipsage/collect.py` |
| 2 | Use ML/NLP to understand the query and find relevant segments | `clipsage/index.py`, `clipsage/retrieve.py` |
| 3 | Generate a short definition and explanation | `clipsage/summarize.py` |
| 4 | Show the source of every explanation (video link + start time) | `summarize.py` (each sentence keeps `video_id` + `start`), `pipeline.build_references`, `app.py` |
| 5 | Visual representation (concept map, keyword cloud, key-term list) | `clipsage/visualize.py`, `clipsage/keywords.py`, `app.py` |
| 6 | Evaluate the system and compare baseline against improved method | `clipsage/evaluate.py` |

## 4. Scope

| Item | Where |
|---|---|
| Text-based topic queries | `app.py` text box, `pipeline.py` CLI |
| English lectures with transcripts only | `collect.py` (`TRANSCRIPT_LANGUAGES`, `default_audio_language` filter) |
| Five CS subjects; more can be added later | `config/topics.json` (add a subject, re-run the pipeline) |
| Simple web interface | `app.py` (Streamlit) |
| *Out of scope:* no downloading or re-hosting of videos | Only metadata and captions are stored; the app links to youtube.com |
| *Out of scope:* content indexed in advance, not in real time | Stages 1–3 run offline; queries only read `data/index/` |
| *Out of scope:* no LLM training | Only pre-trained models are used (SBERT, KeyBERT, optional BART/T5) |

## 5. Key features

| Feature | Where |
|---|---|
| Topic search | `app.py`, `text.normalize_query` ("What is X?" becomes "X") |
| Smart retrieval (meaning, not only words) | `SbertIndex` in `index.py` |
| Text description | `summarize.extractive_summary` (definition sentence first) |
| Source references (title, channel, link, timestamp) | `pipeline.build_references`, References section in the app |
| Visual representation (concept map, keyword cloud, timeline) | `visualize.py`, tabs in the app |
| Key terms for revision | `keywords.py`, "Key terms" chips in the app |

## 6. Methodology (the 7 stages)

| Stage | Proposal | Implementation |
|---|---|---|
| 1 | Search + video-details endpoints; IDs, titles, descriptions, channels, durations; timestamped transcripts; saved as CSV/JSON | `collect.py`: `channels.list` (handles), `playlistItems.list`, `search.list`, `videos.list`. Writes `data/raw/videos.csv` and `data/raw/transcripts/<id>.json`. Tracks quota and resumes where it stopped |
| 2 | Remove filler words, repeated text and noise tags such as [Music]; split into sentences; group into overlapping 30–60 s segments; keep video ID + start time | `preprocess.py`: `clean_text`, `clean_snippets`, `split_sentences`, `make_segments` (target 45 s, 15 s overlap) |
| 3 | TF-IDF baseline, SBERT improved, stored in FAISS or scikit-learn | `index.py`: `TfidfIndex`, `SbertIndex` (FAISS `IndexFlatIP`, NumPy fallback) |
| 4 | Cosine similarity top-k, re-ranked so results come from different videos | `retrieve.py`: `SearchEngine.search` (relative score filter, MMR, at most 2 segments per video) |
| 5 | Extractive summary (TextRank / embedding ranking), optional BART/T5, each sentence linked to its video | `summarize.py`: `top_sentences` (baseline), `textrank`, `embedding`; `abstractive_summary` (BART by default, T5 supported) |
| 6 | Key terms (KeyBERT / TF-IDF); co-occurrence concept graph (NetworkX + PyVis); keyword cloud; timeline | `keywords.py`, `visualize.build_concept_graph`, `concept_map_html`, `keyword_cloud`, `timeline_figure` |
| 7 | Streamlit page with explanation, clickable references and visuals | `app.py` |

## 7. ML / NLP techniques (baseline against improved)

Every pair can be switched from the app's sidebar and is compared in `evaluate.py`.

| Task | Baseline | Improved | Compared by |
|---|---|---|---|
| Representation and retrieval | TF-IDF + cosine | SBERT + cosine (FAISS) | P@5, R@10, MRR, Wilcoxon test |
| Explanation | `top_sentences` | `textrank`, `embedding` (+ BART) | ROUGE-1/2/L, manual 1–5 rating |
| Keywords | TF-IDF | KeyBERT | manual relevance check |
| Topic grouping | – | K-Means sub-topics (k by silhouette), corpus LDA | shown in the app and in `reports/lda_topics.md` |
| Visualization | keyword cloud | co-occurrence concept graph | usability survey |

## 8. Dataset

| Item | Where |
|---|---|
| Trusted channels and playlists | `config/sources.json` |
| 200–500 videos | `max_total_videos` in `sources.json` (500), balanced across subjects |
| Several thousand segments | depends on what is collected; `python -m clipsage.explore` reports the numbers |
| 30 test queries, relevant videos marked by hand | `evaluation/queries.json` (30), `evaluate pool` → `evaluation/relevance_labels.csv` |

## 9. Tools

All tools listed in the proposal are used except these:

- **NLTK / spaCy.** Sentence splitting and stop words are handled with regular expressions and scikit-learn's English stop-word list, plus a list of filler words common in lectures (`clipsage/text.py`). Lecture captions often have no punctuation, so time-based sentence cutting worked better than an NLTK sentence splitter.
- **Pandas.** Plain `csv`/`json` files are used, and Pandas can open all of them for analysis.

## 10. Evaluation plan

| Measure | Target | Command |
|---|---|---|
| Retrieval quality (P@5, R@10, MRR) | SBERT clearly better than TF-IDF | `evaluate pool` → label → `evaluate retrieval` |
| Summary quality (ROUGE + manual 1–5) | average rating ≥ 4 | `evaluate summary`, fill `summary_ratings.csv` |
| Keyword / concept-map quality | ≥ 80% relevant | `evaluate keywords`, fill `keyword_review.csv` |
| Source accuracy | 100% valid links | `evaluate sources [--online]` |
| Usability (10–15 students) | positive feedback | `docs/usability_survey.md`, `survey_responses.csv` |
| Speed | under 5 s | `evaluate speed` |
| All in one report | – | `evaluate report` → `reports/evaluation_report.md` |

## 11. Deliverables

| Deliverable | Status |
|---|---|
| 1. Working prototype (code + Streamlit app) | this repository |
| 2. Collected dataset | produced by `run_pipeline.bat` in `data/raw` and `data/processed`; commit it once collected (`data/index` is rebuilt, not committed) |
| 3. Evaluation results with baseline vs improved comparison | `reports/` after running the evaluation |
| 4. Final report | to be written; use `reports/*.md` and the charts |
| 5. Presentation and live demo | the app plus the charts in `reports/` |

## 14. Risks: mitigations in code

| Risk | Mitigation in code |
|---|---|
| API quota | `QuotaTracker`, cheap endpoints first, search capped (`max_search_calls`), everything cached |
| Missing or poor transcripts | human captions preferred; videos without English captions skipped and logged |
| Unclear summaries | extractive only, chatter filter, redundancy filter, similarity score filter, definition sentence first |
| Topic not in dataset | "not covered" message plus related topics (`retrieve.suggest_topics`) |
| Not enough time | abstractive summary and clustering are optional and switchable |
| Team member unavailable | one module per stage, docstrings, tests, this mapping |
