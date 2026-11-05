import numpy as np
import pytest

from clipsage import mathsolve as m
from clipsage import playground as pg


# --------------------------------------------------------------------------- math
def test_matrix_and_linear_algebra():
    r = m.matrix_multiply(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
    assert r["answer"].tolist() == [[19, 22], [43, 50]] and r["verified"].startswith("NumPy")
    assert m.determinant(m.parse_matrix("2 0 1; 1 3 2; 1 1 1"))["answer"] == pytest.approx(0)
    inv = m.inverse(m.parse_matrix("1 2; 3 4"))
    assert np.allclose(inv["answer"], [[-2, 1], [1.5, -0.5]])
    assert m.inverse(m.parse_matrix("1 2; 2 4"))["answer"] is None
    assert sorted(z.real for z in m.eigen2(m.parse_matrix("2 1; 1 2"))["answer"]) == pytest.approx([1, 3])
    with pytest.raises(m.MathInputError):
        m.matrix_multiply(np.ones((2, 3)), np.ones((2, 3)))


def test_vectors_stats_probability():
    assert m.vectors(np.array([1, 0]), np.array([1, 1]))["answer"]["cosine"] == pytest.approx(0.70711, abs=1e-4)
    s = m.statistics(np.array([2, 4, 4, 4, 5, 5, 7, 9]))["answer"]
    assert s["std_pop"] == pytest.approx(2) and s["var_sample"] == pytest.approx(32 / 7)
    assert m.bayes(0.01, 0.99, 0.05)["answer"] == pytest.approx(0.1667, abs=1e-4)
    assert m.binomial(5, 3, 0.5)["answer"] == pytest.approx(0.3125)


@pytest.mark.parametrize("text,expected", [
    ("derivative of x^3 + 2x at x = 2", 14),
    ("integral of x^2 from 0 to 3", 9),
    ("solve 2x + 3 = 11", 4),
    ("mean of 2, 4, 9", None),
])
def test_detect_and_solve(text, expected):
    r = m.detect_and_solve(text)
    assert "MISMATCH" not in r["verified"]
    if expected is not None:
        assert float(r["answer"]) == pytest.approx(expected)


def test_optimize_and_gradient_descent():
    pts = m.optimize("x^2 - 6x + 10")["answer"]
    assert pts == [{"x": 3.0, "f": 1.0, "type": "minimum"}]
    gd = m.gradient_descent("x^2", 4, 0.1, 3)
    assert [round(h["x"], 3) for h in gd["history"]] == [4, 3.2, 2.56, 2.048]
    assert m.gradient_descent("x^2", 4, 1.1, 50)["diverged"]


@pytest.mark.parametrize("bad", ["__import__('os')", "open('x')", "x +* 2", ""])
def test_unsafe_or_bad_input_is_rejected(bad):
    with pytest.raises(m.MathInputError):
        m.parse_expression(bad)


@pytest.mark.parametrize("kind", m.PRACTICE_KINDS)
def test_practice_problems_are_self_consistent(kind):
    for seed in range(5):
        p = m.practice_problem(kind, seed)
        assert m.check_practice(p, repr(float(p["answer"])))["status"] == "correct"
        assert m.check_practice(p, str(float(p["answer"]) + 1))["status"] == "incorrect"
        assert "MISMATCH" not in p["solution"]["verified"]


# --------------------------------------------------------------------------- playground
def test_linear_regression_matches_closed_form():
    X = np.array([[1], [2], [3], [4]], float)
    y = np.array([2, 4, 5, 8], float)
    out = pg.run_linear_regression(X, y)
    assert out["metrics"]["slope w"] == pytest.approx(1.9)
    assert out["metrics"]["intercept b"] == pytest.approx(0, abs=1e-9)
    assert out["metrics"]["sklearn slope"] == pytest.approx(1.9)
    assert out["predict"]([5])["prediction"] == pytest.approx(9.5)
    assert out["figures"]["fit"].data


def test_gradient_descent_converges_and_diverges():
    X = np.array([[1], [2], [3], [4]], float)
    y = np.array([2, 4, 5, 8], float)
    ok = pg.run_gradient_descent(X, y, lr=0.05, epochs=2000)
    assert ok["metrics"]["final w"] == pytest.approx(1.9, abs=1e-3) and not ok["diverged"]
    assert pg.run_gradient_descent(X, y, lr=0.5, epochs=50)["diverged"]


@pytest.mark.parametrize("name", ["logistic_regression", "knn", "naive_bayes", "decision_tree", "svm"])
def test_classifiers_on_sample_data(name):
    d = pg.make_sample("classification")
    fn = getattr(pg, f"run_{name}")
    out = fn(d["X"], d["y"])
    assert out["metrics"]["training accuracy"] >= 0.85
    assert out["predict"]([2, 2])["prediction"] == 0
    assert out["predict"]([5, 5])["prediction"] == 1
    assert out["figures"]


def test_knn_distances_and_vote():
    X = np.array([[1, 1], [2, 1], [4, 4], [5, 4], [4, 5]], float)
    y = np.array(["red", "red", "blue", "blue", "blue"])
    out = pg.run_knn(X, y, k=3, query=[2, 2])
    table = out["tables"]["distances from the query point"]
    assert [row["distance"] for row in table[:3]] == pytest.approx([1.0, 1.4142, 2.8284], abs=1e-4)
    assert out["metrics"]["prediction"] == "red" and out["metrics"]["vote"] == {"blue": 1, "red": 2}


def test_naive_bayes_manual_matches_sklearn():
    d = pg.make_sample("classification", seed=3)
    out = pg.run_naive_bayes(d["X"], d["y"], query=[3.5, 3.4])
    assert out["metrics"]["manual prediction"] == out["metrics"]["sklearn prediction"]


def test_decision_tree_root_split_by_hand():
    d = pg.make_sample("classification")
    out = pg.run_decision_tree(d["X"], d["y"], max_depth=2)
    rows = out["tables"]["root split by hand"]
    assert rows[0]["Gini"] == pytest.approx(0.5)
    assert rows[-1]["Gini"] > 0
    assert "tree structure" in out["matplotlib"]


def test_kmeans_plus_plus_finds_the_three_groups():
    d = pg.make_sample("clustering")
    from sklearn.cluster import KMeans

    best = KMeans(n_clusters=3, n_init=10, random_state=0).fit(d["X"]).inertia_
    for seed in range(5):
        out = pg.run_kmeans(d["X"], k=3, seed=seed, init="k-means++")
        assert out["metrics"]["final inertia"] <= best * 1.05


def test_kmeans_inertia_never_increases():
    d = pg.make_sample("clustering")
    out = pg.run_kmeans(d["X"], k=3)
    inertia = [row["inertia"] for row in out["tables"]["iterations"]]
    assert all(b <= a + 1e-9 for a, b in zip(inertia, inertia[1:]))
    assert sum(out["metrics"]["cluster sizes"].values()) == len(d["X"])
    assert len(out["figures"]["iterations"].frames) == len(inertia)


def test_tfidf_cosine_and_sbert_fallback():
    t = pg.run_tfidf(["cats like milk", "dogs like bones", "cats chase dogs"])
    milk = next(r for r in t["tables"]["term statistics"] if r["term"] == "milk")
    assert milk["idf = ln(N/df)"] == pytest.approx(1.0986, abs=1e-4)
    assert milk["tf-idf D1 (plain)"] == pytest.approx(0.3662, abs=1e-4)
    c = pg.run_cosine([np.array([1, 2, 3]), np.array([2, 4, 6]), np.array([3, 0, -1])])
    assert c["tables"]["pairwise"][0]["cosine"] == pytest.approx(1.0)
    assert c["tables"]["pairwise"][1]["cosine"] == pytest.approx(0.0, abs=1e-9)
    s = pg.run_sentence_bert(["a model learns noise", "overfitting means learning noise", "tcp handshake"])
    assert s["tables"]["most similar pairs"]


def test_csv_parsing():
    d = pg.parse_csv("x1,x2,label\n1,2,a\n3,4,b\n5,6,a", "classification")
    assert d["X"].shape == (3, 2) and list(d["y"]) == ["a", "b", "a"]
    with pytest.raises(pg.DataError):
        pg.parse_csv("x,y\n1,a\n2,b", "regression")
    with pytest.raises(pg.DataError):
        pg.parse_csv("x,y\n1,2", "regression")
    assert pg.parse_csv(pg.sample_csv("clustering"), "clustering")["X"].shape[1] == 2
