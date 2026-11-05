"""Learning data is temporary unless the student clicks Save, and saved data stays inside its subject."""

import json

from clipsage.progress import SessionActivity, SubjectStore


def files_in(path):
    return sorted(p.name for p in path.rglob("*") if p.is_file())


def test_session_activity_never_touches_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    act = SessionActivity()
    act.add_search("paging kya hai", "ru", "Operating Systems")
    act.add_studied("Operating Systems", "paging_virtual_memory", "Paging")
    act.record_answer("Operating Systems", "paging_virtual_memory", "q1", "correct")
    act.record_answer("Operating Systems", "paging_virtual_memory", "q2", "incorrect")
    act.add_practice("Operating Systems", "os_scheduling", "FCFS avg WT 3.33")
    s = act.subject_summary("Operating Systems")
    assert s["answered"] == 2 and s["correct"] == 1 and s["accuracy"] == 0.5
    assert files_in(tmp_path) == []                     # nothing written anywhere
    assert not hasattr(act, "save")


def test_store_creates_no_file_until_save(tmp_path):
    store = SubjectStore("Operating Systems", tmp_path)
    assert store.load()["saved_topics"] == [] and not store.exists()
    store.counts()
    store.is_topic_saved("x")
    assert files_in(tmp_path) == []
    store.save_topic("cpu_scheduling", "CPU Scheduling")
    assert files_in(tmp_path) == ["operating_systems.json"]


def test_saved_data_stays_in_its_subject(tmp_path):
    os_store = SubjectStore("Operating Systems", tmp_path)
    ml_store = SubjectStore("Machine Learning", tmp_path)
    os_store.save_note("paging_virtual_memory", "Paging", "offset = log2(page size)")
    os_store.save_quiz_result("cpu_scheduling", "CPU Scheduling", 4, 3, [{"q": "q1", "correct": True}])
    ml_store.save_progress({"answered": 5, "correct": 4})
    ml_store.save_material("explanation", "Overfitting", "text", lang="ru")

    os_data, ml_data = os_store.load(), ml_store.load()
    assert list(os_data["notes"]) == ["paging_virtual_memory"] and ml_data["notes"] == {}
    assert len(os_data["quiz_results"]) == 1 and ml_data["quiz_results"] == []
    assert len(ml_data["progress"]) == 1 and os_data["progress"] == []
    assert ml_data["materials"][0]["lang"] == "ru" and os_data["materials"] == []
    assert files_in(tmp_path) == ["machine_learning.json", "operating_systems.json"]
    on_disk = json.loads((tmp_path / "operating_systems.json").read_text())
    assert on_disk["subject"] == "Operating Systems"


def test_manage_saved_items(tmp_path):
    store = SubjectStore("Linear Algebra", tmp_path)
    store.save_topic("eigenvalues", "Eigenvalues")
    store.save_topic("eigenvalues", "Eigenvalues")          # no duplicates
    store.save_note("vectors", "Vectors", "dot product")
    store.save_material("solution", "det", "5")
    assert store.counts() == {"saved_topics": 1, "notes": 1, "quiz_results": 0, "progress": 0, "materials": 1}
    store.remove("saved_topics", "eigenvalues")
    store.remove("notes", "vectors")
    store.remove("materials", store.load()["materials"][0]["id"])
    assert store.counts() == {"saved_topics": 0, "notes": 0, "quiz_results": 0, "progress": 0, "materials": 0}
    store.clear()
    assert not store.exists()
