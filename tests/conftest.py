"""
Test fixtures: a tiny SYNTHETIC lecture corpus (fake video IDs, invented
transcripts) so the whole pipeline can be tested without the YouTube API or
downloading the Sentence-BERT model. The real dataset is never touched.
"""

import csv
import json

import numpy as np
import pytest
from sklearn.feature_extraction.text import HashingVectorizer

from clipsage import config, embeddings, retrieve

LECTURES = {
    "fake_ml_001": ("Machine Learning", "Overfitting and Regularization Explained", "Demo ML Channel", [
        "Welcome back to the channel, today we talk about overfitting.",
        "Overfitting is when a model learns the training data too well, including the noise.",
        "An overfit model has a very low training error but a high validation error.",
        "This happens when the model is too complex for the amount of training data.",
        "For example a high degree polynomial can pass through every training point.",
        "The model then fails to generalize to new unseen data.",
        "To detect overfitting we compare training error and validation error.",
        "Cross validation gives a more reliable estimate of the validation error.",
        "Regularization reduces overfitting by adding a penalty on large weights.",
        "L2 regularization is also called ridge regression or weight decay.",
        "Dropout is a regularization method for neural networks that randomly drops units.",
        "Collecting more data is another way to reduce overfitting.",
        "Early stopping halts training when the validation error starts to increase.",
        "Underfitting is the opposite problem where the model is too simple.",
        "The bias variance tradeoff explains the balance between underfitting and overfitting.",
        "A model with high variance is usually overfitting the training data.",
        "So remember, a good model has low error on both training and validation data.",
        "Thanks for watching and please subscribe for more machine learning videos.",
    ]),
    "fake_ml_002": ("Machine Learning", "Gradient Descent Step by Step", "Demo Stats Channel", [
        "Gradient descent is an optimization algorithm that minimizes a loss function.",
        "We start with random values for the parameters of the model.",
        "Then we compute the gradient of the loss with respect to each parameter.",
        "The gradient points in the direction of steepest increase of the loss.",
        "So we take a step in the opposite direction of the gradient.",
        "The size of the step is controlled by the learning rate.",
        "If the learning rate is too large we can overshoot the minimum.",
        "If the learning rate is too small training becomes very slow.",
        "Stochastic gradient descent uses one random sample at each step.",
        "Mini batch gradient descent uses a small batch of samples instead.",
        "We repeat the steps until the loss stops decreasing.",
        "Gradient descent is used to train linear regression and neural networks.",
        "Regularization terms can also be added to the loss to avoid overfitting.",
    ]),
    "fake_ds_001": ("Data Structures", "Binary Search Trees", "Demo CS Channel", [
        "A binary search tree is a binary tree where each node has a key.",
        "Every key in the left subtree is smaller than the key of the node.",
        "Every key in the right subtree is larger than the key of the node.",
        "To search for a key we compare it with the root node.",
        "If the key is smaller we go to the left child, otherwise to the right child.",
        "Insertion follows the same path and adds the new node as a leaf.",
        "Deletion of a node with two children uses the inorder successor.",
        "In a balanced tree search insertion and deletion take log n time.",
        "In a skewed tree the height becomes n and operations take linear time.",
        "AVL trees and red black trees keep the binary search tree balanced.",
        "An inorder traversal of a binary search tree visits the keys in sorted order.",
        "Binary search trees are used to implement sets and maps.",
    ]),
    "fake_os_001": ("Operating Systems", "Deadlock in Operating Systems", "Demo OS Channel", [
        "A deadlock is a situation where processes wait for each other forever.",
        "Each process holds a resource and waits for a resource held by another process.",
        "There are four necessary conditions for a deadlock.",
        "The first condition is mutual exclusion of resources.",
        "The second condition is hold and wait.",
        "The third condition is no preemption of resources.",
        "The fourth condition is circular wait between the processes.",
        "Deadlock prevention breaks at least one of these conditions.",
        "Deadlock avoidance uses the bankers algorithm to stay in a safe state.",
        "Deadlock detection builds a resource allocation graph and looks for a cycle.",
        "Recovery can terminate a process or preempt its resources.",
        "A semaphore used incorrectly can easily cause a deadlock.",
    ]),
    "fake_net_001": ("Computer Networks", "TCP Three Way Handshake", "Demo Networks Channel", [
        "TCP is a reliable connection oriented transport protocol.",
        "Before sending data TCP sets up a connection with a three way handshake.",
        "First the client sends a SYN segment with its initial sequence number.",
        "The server answers with a SYN ACK segment that acknowledges the client sequence number.",
        "Finally the client sends an ACK segment and the connection is established.",
        "The handshake synchronizes the sequence numbers on both sides.",
        "UDP in contrast is connectionless and has no handshake.",
        "Closing a TCP connection uses FIN and ACK segments from both sides.",
        "Flow control in TCP uses a sliding window.",
        "Congestion control reduces the sending rate when packets are lost.",
    ]),
    "fake_db_001": ("Databases", "Database Normalization 1NF 2NF 3NF", "Demo DB Channel", [
        "Normalization organizes database tables to reduce redundancy.",
        "Redundant data causes insertion update and deletion anomalies.",
        "First normal form requires that every column holds atomic values.",
        "Second normal form removes partial dependency on a composite primary key.",
        "Third normal form removes transitive dependency between non key attributes.",
        "Boyce Codd normal form is a stronger version of third normal form.",
        "Normalization uses functional dependencies to decide how to split tables.",
        "A foreign key links the smaller tables back together.",
        "Too much normalization can make queries slower because of many joins.",
    ]),
}


def make_snippets(sentences, seconds_per_sentence=6.0):
    """Turn sentences into caption lines (two lines per sentence, like real captions)."""
    snippets, t = [], 0.0
    for s in sentences:
        words = s.split()
        half = len(words) // 2
        for part in (" ".join(words[:half]), " ".join(words[half:])):
            snippets.append({"text": part, "start": round(t, 2), "duration": seconds_per_sentence / 2})
            t += seconds_per_sentence / 2
    snippets.insert(1, {"text": "[Music]", "start": 0.5, "duration": 0.5})
    return snippets


class FakeEmbedder:
    """Deterministic stand-in for Sentence-BERT (hashed bag of words + bigrams)."""

    def __init__(self):
        self.vec = HashingVectorizer(n_features=2 ** 15, ngram_range=(1, 2), stop_words="english",
                                     alternate_sign=False, norm=None)

    def encode(self, texts, batch_size=64, show_progress=False):
        m = self.vec.transform(list(texts)).toarray().astype(np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    """Point every ClipSage path at a temporary folder with the synthetic corpus."""
    data = tmp_path / "data"
    raw = data / "raw"
    paths = {
        "DATA_DIR": data, "RAW_DIR": raw, "TRANSCRIPT_DIR": raw / "transcripts",
        "VIDEOS_CSV": raw / "videos.csv", "COLLECTION_LOG": raw / "collection_log.csv",
        "PROCESSED_DIR": data / "processed", "SENTENCES_FILE": data / "processed" / "sentences.jsonl",
        "SEGMENTS_FILE": data / "processed" / "segments.jsonl", "SEGMENTS_CSV": data / "processed" / "segments.csv",
        "INDEX_DIR": data / "index", "REPORTS_DIR": tmp_path / "reports",
    }
    for k, v in paths.items():
        monkeypatch.setattr(config, k, v)
    # hashed bag-of-words scores are lower than real SBERT cosine scores
    monkeypatch.setattr(config, "NOT_COVERED_THRESHOLD", {"tfidf": 0.08, "sbert": 0.08})
    paths["TRANSCRIPT_DIR"].mkdir(parents=True)

    from clipsage.collect import VIDEO_FIELDS

    with open(paths["VIDEOS_CSV"], "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VIDEO_FIELDS)
        writer.writeheader()
        for vid, (subject, title, channel, sentences) in LECTURES.items():
            snippets = make_snippets(sentences)
            duration = int(snippets[-1]["start"] + snippets[-1]["duration"]) + 1
            writer.writerow({"video_id": vid, "title": title, "channel_id": "UC_fake", "channel_title": channel,
                             "subject": subject, "source": "test", "duration_seconds": duration,
                             "published_at": "", "has_captions": True, "default_audio_language": "en",
                             "view_count": 0, "description": "", "url": ""})
            with open(paths["TRANSCRIPT_DIR"] / f"{vid}.json", "w", encoding="utf-8") as tf:
                json.dump({"video_id": vid, "language": "en", "kind": "manual", "snippets": snippets}, tf)

    embeddings.set_embedder(FakeEmbedder())
    retrieve.reset_engine()
    yield paths
    embeddings.set_embedder(None)
    retrieve.reset_engine()


@pytest.fixture()
def built(corpus):
    from clipsage import index, preprocess

    preprocess.run()
    index.build()
    return corpus
