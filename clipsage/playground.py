"""
ML Algorithm Playground: run real algorithms on small datasets and look at
the intermediate calculations and plots.

Every run_* function returns a dict:
    tables   {name: list of row dicts}     intermediate calculations
    metrics  {name: value}
    figures  {name: plotly Figure}         built from the computed values
    notes    [ {en, ru} ]                  how to read the result
    predict  optional callable(point) -> dict

Datasets come from the student (CSV text/upload) or from make_sample(),
which generates clearly labelled SYNTHETIC data.
"""

import io
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

PALETTE = ["#4A90E2", "#F5A623", "#7ED321", "#BD10E0", "#D0021B", "#50E3C2"]
LAYOUT = dict(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F5F9FF",
              font=dict(color="#243447"), margin=dict(l=40, r=20, t=50, b=40), height=420)

ALGORITHMS = {
    "linear_regression": {"en": "Linear Regression", "ru": "Linear Regression", "task": "regression"},
    "gradient_descent": {"en": "Gradient Descent (for linear regression)", "ru": "Gradient Descent (linear regression ke liye)", "task": "regression"},
    "logistic_regression": {"en": "Logistic Regression", "ru": "Logistic Regression", "task": "classification"},
    "knn": {"en": "K-Nearest Neighbors", "ru": "K-Nearest Neighbors", "task": "classification"},
    "naive_bayes": {"en": "Gaussian Naive Bayes", "ru": "Gaussian Naive Bayes", "task": "classification"},
    "decision_tree": {"en": "Decision Tree", "ru": "Decision Tree", "task": "classification"},
    "svm": {"en": "Support Vector Machine", "ru": "Support Vector Machine", "task": "classification"},
    "kmeans": {"en": "K-Means Clustering", "ru": "K-Means Clustering", "task": "clustering"},
    "tfidf": {"en": "TF-IDF", "ru": "TF-IDF", "task": "text"},
    "cosine_similarity": {"en": "Cosine Similarity", "ru": "Cosine Similarity", "task": "vectors"},
    "sentence_bert": {"en": "Sentence-BERT similarity", "ru": "Sentence-BERT similarity", "task": "text"},
}


class DataError(ValueError):
    pass


def _layout(fig: go.Figure, title: str, **kw) -> go.Figure:
    fig.update_layout(title=title, **{**LAYOUT, **kw})
    return fig


def r(v, d: int = 4) -> float:
    return float(np.round(float(v), d))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_sample(task: str, seed: int = 0, n: int = 40) -> Dict:
    """SYNTHETIC sample data for practice (never presented as real data)."""
    rng = np.random.default_rng(seed)
    if task == "regression":
        x = np.round(rng.uniform(1, 10, n // 2), 1)
        y = np.round(3 + 2 * x + rng.normal(0, 2, len(x)), 1)
        return {"X": x.reshape(-1, 1), "y": y, "features": ["hours_studied"], "target": "marks",
                "label": "Synthetic: marks = 3 + 2 x hours + noise"}
    if task == "classification":
        a = rng.normal([2, 2], 1.0, size=(n // 2, 2))
        b = rng.normal([5, 5], 1.1, size=(n // 2, 2))
        X = np.round(np.vstack([a, b]), 2)
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        return {"X": X, "y": y, "features": ["feature_1", "feature_2"], "target": "class",
                "label": "Synthetic: two Gaussian blobs (class 0 around (2,2), class 1 around (5,5))"}
    if task == "clustering":
        centers = np.array([[2, 2], [7, 3], [4, 7]])
        X = np.round(np.vstack([rng.normal(c, 0.9, size=(n // 3, 2)) for c in centers]), 2)
        return {"X": X, "y": None, "features": ["x", "y"], "target": None,
                "label": "Synthetic: three groups around (2,2), (7,3), (4,7)"}
    raise DataError(f"no sample for {task}")


def parse_csv(text: str, task: str) -> Dict:
    """CSV with a header row. Regression/classification: last column is the target."""
    import csv

    rows = [row for row in csv.reader(io.StringIO((text or "").strip())) if any(c.strip() for c in row)]
    if len(rows) < 3:
        raise DataError("need a header row and at least two data rows")
    header, body = [h.strip() for h in rows[0]], rows[1:]
    if any(len(row) != len(header) for row in body):
        raise DataError("every row must have the same number of columns as the header")
    if task == "clustering":
        try:
            X = np.array([[float(v) for v in row] for row in body])
        except ValueError as e:
            raise DataError("all values must be numbers") from e
        return {"X": X, "y": None, "features": header, "target": None, "label": "Your data"}
    if len(header) < 2:
        raise DataError("need at least one feature column and one target column")
    try:
        X = np.array([[float(v) for v in row[:-1]] for row in body])
    except ValueError as e:
        raise DataError("feature values must be numbers") from e
    targets = [row[-1].strip() for row in body]
    if task == "regression":
        try:
            y = np.array([float(t) for t in targets])
        except ValueError as e:
            raise DataError("the target column must be numeric for regression") from e
    else:
        classes = sorted(set(targets))
        if len(classes) < 2:
            raise DataError("classification needs at least two classes in the last column")
        y = np.array(targets)
    return {"X": X, "y": y, "features": header[:-1], "target": header[-1], "label": "Your data"}


def sample_csv(task: str) -> str:
    d = make_sample(task)
    head = d["features"] + ([d["target"]] if d["target"] else [])
    lines = [",".join(head)]
    for i, row in enumerate(d["X"]):
        vals = [f"{v:g}" for v in row] + ([f"{d['y'][i]:g}"] if d["y"] is not None else [])
        lines.append(",".join(vals))
    return "\n".join(lines)


def _grid(X: np.ndarray, steps: int = 120):
    pad = 0.08 * (X.max(axis=0) - X.min(axis=0) + 1e-9) + 0.5
    xs = np.linspace(X[:, 0].min() - pad[0], X[:, 0].max() + pad[0], steps)
    ys = np.linspace(X[:, 1].min() - pad[1], X[:, 1].max() + pad[1], steps)
    gx, gy = np.meshgrid(xs, ys)
    return xs, ys, np.c_[gx.ravel(), gy.ravel()]


def _region_figure(model, X, y, title, features, extra_traces=()):
    """Decision regions of a fitted classifier on 2-D data (computed on a grid)."""
    classes = list(np.unique(y))
    xs, ys, pts = _grid(X)
    pred = model.predict(pts)
    z = np.array([classes.index(p) for p in pred]).reshape(len(ys), len(xs))
    fig = go.Figure()
    fig.add_trace(go.Contour(x=xs, y=ys, z=z, showscale=False, opacity=0.25, hoverinfo="skip",
                             colorscale=[[i / max(1, len(classes) - 1), PALETTE[i % len(PALETTE)]] for i in range(len(classes))],
                             contours=dict(coloring="fill", showlines=False)))
    for i, c in enumerate(classes):
        m = y == c
        fig.add_trace(go.Scatter(x=X[m, 0], y=X[m, 1], mode="markers", name=f"class {c}",
                                 marker=dict(color=PALETTE[i % len(PALETTE)], size=9, line=dict(color="white", width=1))))
    for t in extra_traces:
        fig.add_trace(t)
    return _layout(fig, title, xaxis_title=features[0], yaxis_title=features[1])


def _need_2d(X):
    if X.shape[1] != 2:
        raise DataError("this visualisation needs exactly 2 feature columns")


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def run_linear_regression(X, y, features=("x",), target="y") -> Dict:
    if X.shape[1] != 1:
        raise DataError("use one feature column for the step-by-step linear regression")
    x = X[:, 0].astype(float)
    y = y.astype(float)
    xm, ym = x.mean(), y.mean()
    dx, dy = x - xm, y - ym
    w = float(np.sum(dx * dy) / np.sum(dx ** 2))
    b = float(ym - w * xm)
    pred = w * x + b
    res = y - pred
    mse = float(np.mean(res ** 2))
    r2 = float(1 - np.sum(res ** 2) / np.sum(dy ** 2))
    from sklearn.linear_model import LinearRegression

    sk = LinearRegression().fit(X, y)
    table = [{"x": r(a), "y": r(c), "x - mean_x": r(p), "y - mean_y": r(q), "(x-mx)(y-my)": r(p * q),
              "(x-mx)^2": r(p ** 2), "prediction": r(e), "residual": r(f)}
             for a, c, p, q, e, f in zip(x, y, dx, dy, pred, res)]
    xs = np.linspace(x.min(), x.max(), 50)
    fig = go.Figure()
    for xi, yi, pi in zip(x, y, pred):
        fig.add_trace(go.Scatter(x=[xi, xi], y=[yi, pi], mode="lines", line=dict(color="#F5A623", width=1),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data", marker=dict(color=PALETTE[0], size=9)))
    fig.add_trace(go.Scatter(x=xs, y=w * xs + b, mode="lines", name=f"y = {w:.3f}x + {b:.3f}",
                             line=dict(color="#243447", width=3)))
    _layout(fig, "Least-squares line (orange = residuals)", xaxis_title=features[0], yaxis_title=target)
    return {
        "tables": {"calculations": table},
        "metrics": {"mean_x": r(xm), "mean_y": r(ym), "sum (x-mx)(y-my)": r(np.sum(dx * dy)),
                    "sum (x-mx)^2": r(np.sum(dx ** 2)), "slope w": r(w), "intercept b": r(b), "MSE": r(mse),
                    "R squared": r(r2), "sklearn slope": r(sk.coef_[0]), "sklearn intercept": r(sk.intercept_)},
        "figures": {"fit": fig},
        "notes": [
            {"en": f"Slope = {np.sum(dx * dy):.3f} / {np.sum(dx ** 2):.3f} = {w:.3f}: each extra unit of "
                   f"{features[0]} changes {target} by {w:.3f}.",
             "ru": f"Slope = {np.sum(dx * dy):.3f} / {np.sum(dx ** 2):.3f} = {w:.3f}: {features[0]} mein har 1 unit "
                   f"izafe se {target} {w:.3f} badalta hai."},
            {"en": f"R squared = {r2:.3f}: the line explains {100 * r2:.1f}% of the variation in {target}.",
             "ru": f"R squared = {r2:.3f}: line {target} ki {100 * r2:.1f}% variation samjhati hai."},
            {"en": "The hand calculation matches scikit-learn's LinearRegression.",
             "ru": "Haath ka hisaab scikit-learn ke LinearRegression se milta hai."},
        ],
        "predict": lambda point: {"prediction": r(w * float(point[0]) + b),
                                  "working": f"{w:.4f} x {float(point[0]):g} + {b:.4f}"},
    }


def run_gradient_descent(X, y, lr: float = 0.01, epochs: int = 200, features=("x",), target="y") -> Dict:
    if X.shape[1] != 1:
        raise DataError("use one feature column")
    x, y = X[:, 0].astype(float), y.astype(float)
    w = b = 0.0
    history = []
    for epoch in range(epochs + 1):
        pred = w * x + b
        err = y - pred
        loss = float(np.mean(err ** 2))
        gw, gb = float(-2 * np.mean(x * err)), float(-2 * np.mean(err))
        history.append({"epoch": epoch, "w": r(w), "b": r(b), "loss (MSE)": r(loss), "dL/dw": r(gw), "dL/db": r(gb)})
        if not np.isfinite(loss) or loss > 1e12:
            break
        if epoch < epochs:
            w, b = w - lr * gw, b - lr * gb
    diverged = not np.isfinite(history[-1]["loss (MSE)"]) or history[-1]["loss (MSE)"] > history[0]["loss (MSE)"]
    exact_w = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    exact_b = y.mean() - exact_w * x.mean()
    loss_fig = go.Figure(go.Scatter(x=[h["epoch"] for h in history], y=[h["loss (MSE)"] for h in history],
                                    mode="lines", line=dict(color=PALETTE[0], width=3)))
    _layout(loss_fig, "Loss going down (MSE per epoch)", xaxis_title="epoch", yaxis_title="MSE",
            yaxis_type="log" if not diverged else "linear")
    line_fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", name="data", marker=dict(color=PALETTE[0], size=9)))
    xs = np.linspace(x.min(), x.max(), 20)
    shown = sorted(set([0, 1, 5, 20, len(history) // 2, len(history) - 1]))
    for i, e in enumerate(e for e in shown if e < len(history)):
        h = history[e]
        line_fig.add_trace(go.Scatter(x=xs, y=h["w"] * xs + h["b"], mode="lines", name=f"epoch {h['epoch']}",
                                      line=dict(width=2, color=f"rgba(36,52,71,{0.2 + 0.8 * i / max(1, len(shown) - 1)})")))
    _layout(line_fig, "How the line moves during training", xaxis_title=features[0], yaxis_title=target)
    step_rows = history[:6] + ([history[-1]] if len(history) > 6 else [])
    return {
        "tables": {"first updates": step_rows},
        "metrics": {"final w": history[-1]["w"], "final b": history[-1]["b"], "final MSE": history[-1]["loss (MSE)"],
                    "least-squares w": r(exact_w), "least-squares b": r(exact_b)},
        "figures": {"loss": loss_fig, "lines": line_fig},
        "notes": [
            {"en": "Update rule: w = w - lr x dL/dw, b = b - lr x dL/db (see the table for the first steps).",
             "ru": "Update ka qaida: w = w - lr x dL/dw, b = b - lr x dL/db (pehle steps table mein dekhein)."},
            {"en": "The loss diverged: the learning rate is too large. Try a smaller value." if diverged else
                   "The loss decreased; with enough epochs w and b approach the least-squares solution.",
             "ru": "Loss phat gaya: learning rate bohat bara hai. Chhota try karein." if diverged else
                   "Loss kam hua; kaafi epochs ke baad w aur b least-squares hal ke qareeb aa jate hain."},
        ],
        "diverged": diverged,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _accuracy(model, X, y) -> float:
    return r(np.mean(model.predict(X) == y))


def run_logistic_regression(X, y, C: float = 1.0, features=("x1", "x2"), target="class") -> Dict:
    _need_2d(X)
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y)) != 2:
        raise DataError("this demo needs exactly two classes")
    model = LogisticRegression(C=C).fit(X, y)
    w, b = model.coef_[0], model.intercept_[0]
    z = X @ w + b
    p = 1 / (1 + np.exp(-z))
    rows = [{features[0]: r(a[0]), features[1]: r(a[1]), "z = w.x + b": r(zz), "p = sigmoid(z)": r(pp),
             "predicted": model.classes_[int(pp >= 0.5)], "true": t} for a, zz, pp, t in zip(X[:10], z[:10], p[:10], y[:10])]
    extra = []
    if abs(w[1]) > 1e-9:
        xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 2)
        extra.append(go.Scatter(x=xs, y=-(w[0] * xs + b) / w[1], mode="lines", name="boundary (p = 0.5)",
                                line=dict(color="#243447", width=3, dash="dash")))
    fig = _region_figure(model, X, y, "Logistic regression decision regions", features, extra)
    zs = np.linspace(-8, 8, 200)
    sig = _layout(go.Figure([go.Scatter(x=zs, y=1 / (1 + np.exp(-zs)), mode="lines", name="sigmoid",
                                        line=dict(color=PALETTE[0], width=3)),
                             go.Scatter(x=z, y=p, mode="markers", name="your points",
                                        marker=dict(color=[PALETTE[list(model.classes_).index(t)] for t in y], size=8))]),
                  "Sigmoid: score z -> probability", xaxis_title="z", yaxis_title="p")

    def predict(point):
        zz = float(np.dot(w, point) + b)
        pp = 1 / (1 + np.exp(-zz))
        return {"z": r(zz), "probability of class " + str(model.classes_[1]): r(pp),
                "prediction": model.classes_[int(pp >= 0.5)],
                "working": f"z = {w[0]:.3f} x {point[0]:g} + {w[1]:.3f} x {point[1]:g} + {b:.3f}"}
    return {
        "tables": {"first 10 points": rows},
        "metrics": {"w1": r(w[0]), "w2": r(w[1]), "b": r(b), "training accuracy": _accuracy(model, X, y)},
        "figures": {"regions": fig, "sigmoid": sig},
        "notes": [{"en": "The dashed line is where w.x + b = 0, i.e. probability 0.5.",
                   "ru": "Dashed line wahan hai jahan w.x + b = 0, yaani probability 0.5."},
                  {"en": "Smaller C means stronger regularisation (smaller weights, softer probabilities).",
                   "ru": "Chhota C matlab zyada regularisation (chhote weights, narm probabilities)."}],
        "predict": predict,
    }


def run_knn(X, y, k: int = 3, query=None, features=("x1", "x2"), target="class") -> Dict:
    _need_2d(X)
    from sklearn.neighbors import KNeighborsClassifier

    k = int(min(k, len(X)))
    model = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    query = np.array(query if query is not None else X.mean(axis=0), float)
    d = np.sqrt(((X - query) ** 2).sum(axis=1))
    order = np.argsort(d)
    rows = [{"rank": i + 1, features[0]: r(X[j, 0]), features[1]: r(X[j, 1]), "label": y[j],
             "distance": r(d[j]), "neighbour": "yes" if i < k else ""} for i, j in enumerate(order[:max(k + 3, 6)])]
    labels, counts = np.unique(y[order[:k]], return_counts=True)
    vote = {str(lab): int(c) for lab, c in zip(labels, counts)}
    pred = model.predict([query])[0]
    lines = [go.Scatter(x=[query[0], X[j, 0]], y=[query[1], X[j, 1]], mode="lines", showlegend=False,
                        line=dict(color="#243447", width=1, dash="dot"), hoverinfo="skip") for j in order[:k]]
    q_trace = go.Scatter(x=[query[0]], y=[query[1]], mode="markers", name=f"query -> {pred}",
                         marker=dict(symbol="star", size=18, color="#243447"))
    fig = _region_figure(model, X, y, f"KNN regions (k = {k}) and the {k} nearest neighbours", features,
                         lines + [q_trace])
    return {
        "tables": {"distances from the query point": rows},
        "metrics": {"k": k, "vote": vote, "prediction": pred, "training accuracy": _accuracy(model, X, y)},
        "figures": {"regions": fig},
        "notes": [{"en": f"The {k} closest points vote {vote}; the majority class is {pred}.",
                   "ru": f"{k} sab se qareebi points ka vote {vote} hai; majority class {pred} hai."},
                  {"en": "Try k = 1 and a large k to see a jagged vs a smooth boundary.",
                   "ru": "k = 1 aur bara k try karein: tedhi vs hamwar boundary dekhein."}],
        "predict": lambda point: {"prediction": model.predict([point])[0]},
    }


def run_naive_bayes(X, y, query=None, features=("x1", "x2"), target="class") -> Dict:
    _need_2d(X)
    from scipy.stats import norm
    from sklearn.naive_bayes import GaussianNB

    model = GaussianNB().fit(X, y)
    classes = model.classes_
    query = np.array(query if query is not None else X.mean(axis=0), float)
    rows, scores = [], {}
    for i, c in enumerate(classes):
        Xc = X[y == c]
        mu, sd = Xc.mean(axis=0), Xc.std(axis=0) + 1e-9
        prior = len(Xc) / len(X)
        lik = norm.pdf(query, mu, sd)
        score = prior * np.prod(lik)
        scores[c] = score
        rows.append({"class": c, "prior P(c)": r(prior), f"mean {features[0]}": r(mu[0]), f"std {features[0]}": r(sd[0]),
                     f"mean {features[1]}": r(mu[1]), f"std {features[1]}": r(sd[1]),
                     f"P({features[0]}|c)": r(lik[0], 5), f"P({features[1]}|c)": r(lik[1], 5),
                     "prior x likelihoods": r(score, 6)})
    total = sum(scores.values())
    for row in rows:
        row["posterior P(c|x)"] = r(scores[row["class"]] / total) if total else None
    manual = max(scores, key=scores.get)
    q_trace = go.Scatter(x=[query[0]], y=[query[1]], mode="markers", name=f"query -> {manual}",
                         marker=dict(symbol="star", size=18, color="#243447"))
    fig = _region_figure(model, X, y, "Gaussian Naive Bayes regions", features, [q_trace])
    return {
        "tables": {"per-class calculation for the query point": rows},
        "metrics": {"manual prediction": manual, "sklearn prediction": model.predict([query])[0],
                    "training accuracy": _accuracy(model, X, y)},
        "figures": {"regions": fig},
        "notes": [{"en": "Each feature gets a normal distribution per class; likelihoods are multiplied with the prior "
                         "(the naive independence assumption).",
                   "ru": "Har class mein har feature ki normal distribution banti hai; likelihoods prior se zarab hoti "
                         "hain (naive independence assumption)."}],
        "predict": lambda point: {"prediction": model.predict([point])[0],
                                  "probabilities": dict(zip(map(str, classes), model.predict_proba([point])[0].round(3)))},
    }


def _gini(labels) -> float:
    if len(labels) == 0:
        return 0.0
    _, c = np.unique(labels, return_counts=True)
    p = c / c.sum()
    return float(1 - np.sum(p ** 2))


def run_decision_tree(X, y, max_depth: int = 3, criterion: str = "gini", features=("x1", "x2"), target="class") -> Dict:
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, export_text

    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=0).fit(X, y)
    # root split, recomputed by hand
    t = model.tree_
    f, thr = t.feature[0], t.threshold[0]
    rows = []
    if f >= 0:
        left, right = y[X[:, f] <= thr], y[X[:, f] > thr]
        g_root, g_l, g_r = _gini(y), _gini(left), _gini(right)
        weighted = len(left) / len(y) * g_l + len(right) / len(y) * g_r
        rows = [{"node": "root (all data)", "samples": len(y), "Gini": r(g_root)},
                {"node": f"left: {features[f]} <= {thr:.3f}", "samples": len(left), "Gini": r(g_l)},
                {"node": f"right: {features[f]} > {thr:.3f}", "samples": len(right), "Gini": r(g_r)},
                {"node": "weighted Gini after split", "samples": "", "Gini": r(weighted)},
                {"node": "impurity decrease (gain)", "samples": "", "Gini": r(g_root - weighted)}]
    # depth comparison on a held-out split (overfitting demo)
    depth_rows = []
    if len(X) >= 10 and len(np.unique(y)) > 1:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        for dep in [1, 2, 3, 5, 8, None]:
            m = DecisionTreeClassifier(max_depth=dep, criterion=criterion, random_state=0).fit(X_tr, y_tr)
            depth_rows.append({"max_depth": dep or "unlimited", "train accuracy": _accuracy(m, X_tr, y_tr),
                               "test accuracy": _accuracy(m, X_te, y_te), "leaves": int(m.get_n_leaves())})
    figures = {}
    if X.shape[1] == 2:
        figures["regions"] = _region_figure(model, X, y, f"Decision regions (max_depth = {max_depth})", features)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    tree_fig, ax = plt.subplots(figsize=(10, 5))
    plot_tree(model, feature_names=list(features), class_names=[str(c) for c in model.classes_], filled=True,
              rounded=True, ax=ax, fontsize=8, impurity=True)
    tree_fig.tight_layout()
    return {
        "tables": {"root split by hand": rows, "depth vs accuracy (30% held out)": depth_rows},
        "metrics": {"depth": int(model.get_depth()), "leaves": int(model.get_n_leaves()),
                    "training accuracy": _accuracy(model, X, y)},
        "figures": figures, "matplotlib": {"tree structure": tree_fig},
        "text": export_text(model, feature_names=list(features)),
        "notes": [{"en": "The root question is the split with the largest impurity decrease (see the table).",
                   "ru": "Root ka sawal woh split hai jis se impurity sab se zyada kam hoti hai (table dekhein)."},
                  {"en": "In the depth table, look for where test accuracy stops improving while train accuracy keeps "
                         "rising: that is overfitting.",
                   "ru": "Depth table mein dekhein kahan test accuracy behtar hona band ho jati hai jabke train accuracy "
                         "barhti rehti hai: yahi overfitting hai."}],
        "predict": lambda point: {"prediction": model.predict([point])[0]},
    }


def run_svm(X, y, kernel: str = "linear", C: float = 1.0, gamma: str = "scale", features=("x1", "x2"), target="class") -> Dict:
    _need_2d(X)
    from sklearn.svm import SVC

    if len(np.unique(y)) != 2:
        raise DataError("this demo needs exactly two classes")
    model = SVC(kernel=kernel, C=C, gamma=gamma).fit(X, y)
    xs, ys, pts = _grid(X)
    dec = model.decision_function(pts).reshape(len(ys), len(xs))
    sv = model.support_vectors_
    extra = [go.Contour(x=xs, y=ys, z=dec, showscale=False, contours=dict(start=-1, end=1, size=1, coloring="lines"),
                        line=dict(width=2), colorscale=[[0, "#243447"], [1, "#243447"]], hoverinfo="skip",
                        name="margin (-1, 0, +1)"),
             go.Scatter(x=sv[:, 0], y=sv[:, 1], mode="markers", name="support vectors",
                        marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="#243447", width=2)))]
    fig = _region_figure(model, X, y, f"SVM ({kernel} kernel, C = {C})", features, extra)
    metrics = {"support vectors": int(len(sv)), "training accuracy": _accuracy(model, X, y)}
    if kernel == "linear":
        w = model.coef_[0]
        metrics.update({"w1": r(w[0]), "w2": r(w[1]), "b": r(model.intercept_[0]),
                        "margin width 2/||w||": r(2 / np.linalg.norm(w))})
    return {
        "tables": {"support vectors": [{features[0]: r(a), features[1]: r(b)} for a, b in sv[:15]]},
        "metrics": metrics, "figures": {"boundary": fig},
        "notes": [{"en": "Solid lines show the boundary (0) and the margins (-1, +1); circled points are the support vectors.",
                   "ru": "Lines boundary (0) aur margins (-1, +1) dikhati hain; gol ghere wale points support vectors hain."},
                  {"en": "Increase C to see a narrower margin with fewer violations; try the RBF kernel for curved boundaries.",
                   "ru": "C barha kar patla margin dekhein; murhi boundary ke liye RBF kernel try karein."}],
        "predict": lambda point: {"prediction": model.predict([point])[0],
                                  "decision value": r(model.decision_function([point])[0])},
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def init_centroids(X: np.ndarray, k: int, rng, method: str = "k-means++") -> np.ndarray:
    """'random': k random points; 'k-means++': spread out, each new centre picked with probability ~ D(x)^2."""
    if method == "random":
        return X[rng.choice(len(X), size=k, replace=False)].astype(float)
    centroids = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centroids)[None]) ** 2).sum(axis=2), axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(len(X), 1 / len(X))
        centroids.append(X[rng.choice(len(X), p=probs)])
    return np.array(centroids, dtype=float)


def kmeans_steps(X: np.ndarray, k: int, max_iter: int = 20, seed: int = 0, init: str = "k-means++"):
    """Plain K-Means that records every assignment/update step."""
    rng = np.random.default_rng(seed)
    centroids = init_centroids(X, k, rng, init)
    history = []
    labels = None
    for it in range(1, max_iter + 1):
        d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = d.argmin(axis=1)
        inertia = float(np.sum((X - centroids[new_labels]) ** 2))
        history.append({"iteration": it, "centroids": centroids.copy(), "labels": new_labels.copy(), "inertia": inertia})
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centroids = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j] for j in range(k)])
    return history


def run_kmeans(X, k: int = 3, seed: int = 0, init: str = "k-means++", features=("x", "y")) -> Dict:
    _need_2d(X)
    if not 1 <= k <= len(X):
        raise DataError("k must be between 1 and the number of points")
    history = kmeans_steps(X.astype(float), k, seed=seed, init=init)
    final = history[-1]
    frames = []
    for h in history:
        data = [go.Scatter(x=X[h["labels"] == j, 0], y=X[h["labels"] == j, 1], mode="markers",
                           marker=dict(color=PALETTE[j % len(PALETTE)], size=9), name=f"cluster {j + 1}")
                for j in range(k)]
        data.append(go.Scatter(x=h["centroids"][:, 0], y=h["centroids"][:, 1], mode="markers", name="centroids",
                               marker=dict(symbol="x", size=16, color="#243447", line=dict(width=2))))
        frames.append(go.Frame(data=data, name=str(h["iteration"])))
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        updatemenus=[dict(type="buttons", x=0, y=-0.12, xanchor="left", buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=900, redraw=True), fromcurrent=True)])])],
        sliders=[dict(active=0, x=0.12, len=0.88, y=-0.05, currentvalue=dict(prefix="iteration "),
                      steps=[dict(method="animate", label=f.name,
                                  args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for f in frames])])
    _layout(fig, "K-Means: assignments and centroids at each iteration", xaxis_title=features[0],
            yaxis_title=features[1], height=480)
    inertia_fig = _layout(go.Figure(go.Scatter(x=[h["iteration"] for h in history], y=[h["inertia"] for h in history],
                                               mode="lines+markers", line=dict(color=PALETTE[0], width=3))),
                          "Inertia decreases every iteration", xaxis_title="iteration", yaxis_title="inertia")
    from sklearn.cluster import KMeans

    elbow = []
    for kk in range(1, min(8, len(X)) + 1):
        elbow.append({"k": kk, "inertia": r(KMeans(n_clusters=kk, n_init=10, random_state=0).fit(X).inertia_, 2)})
    elbow_fig = _layout(go.Figure(go.Scatter(x=[e["k"] for e in elbow], y=[e["inertia"] for e in elbow],
                                             mode="lines+markers", line=dict(color=PALETTE[1], width=3))),
                        "Elbow method", xaxis_title="k", yaxis_title="inertia")
    rows = [{"iteration": h["iteration"], "inertia": r(h["inertia"], 3),
             **{f"centroid {j + 1}": f"({h['centroids'][j, 0]:.2f}, {h['centroids'][j, 1]:.2f})" for j in range(k)}}
            for h in history]
    sizes = np.bincount(final["labels"], minlength=k)
    return {
        "tables": {"iterations": rows, "elbow": elbow},
        "metrics": {"iterations until stable": len(history), "final inertia": r(final["inertia"], 3),
                    "cluster sizes": {f"cluster {j + 1}": int(s) for j, s in enumerate(sizes)}},
        "figures": {"iterations": fig, "inertia": inertia_fig, "elbow": elbow_fig},
        "notes": [{"en": "Press Play (or move the slider) to see points re-assigned and centroids move to the mean.",
                   "ru": "Play dabayein (ya slider hilayein): points dobara assign hote aur centroids mean ki taraf jate dikhenge."},
                  {"en": "The algorithm stops when no point changes cluster.", "ru": "Jab koi point cluster na badle to algorithm ruk jata hai."},
                  {"en": f"Initialisation: {init}. With 'random' some seeds end in a poor local minimum (higher inertia) - "
                         "compare with the elbow table, which uses the best of 10 k-means++ runs.",
                   "ru": f"Shuruat: {init}. 'random' ke saath kuch seeds kharab local minimum par ruk jate hain (zyada inertia) - "
                         "elbow table se muqabla karein, jo 10 k-means++ runs ka behtareen natija hai."}],
        "predict": lambda point: {"cluster": int(np.argmin(np.linalg.norm(final["centroids"] - np.array(point, float), axis=1))) + 1},
    }


# ---------------------------------------------------------------------------
# Text and vectors
# ---------------------------------------------------------------------------
def run_tfidf(docs: List[str]) -> Dict:
    import math

    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [d.strip() for d in docs if d.strip()]
    if len(docs) < 2:
        raise DataError("enter at least two documents (one per line)")
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    M = vec.fit_transform(docs).toarray()
    vocab = list(vec.get_feature_names_out())
    tokens = [vec.build_analyzer()(d) for d in docs]
    N = len(docs)
    rows = []
    for w in vocab:
        df = sum(w in t for t in tokens)
        idf_plain = math.log(N / df)
        row = {"term": w, "df": df, "idf = ln(N/df)": r(idf_plain), "sklearn idf": r(vec.idf_[vocab.index(w)])}
        for i, t in enumerate(tokens):
            tf = t.count(w) / len(t)
            row[f"tf-idf D{i + 1} (plain)"] = r(tf * idf_plain)
        rows.append(row)
    heat = go.Figure(go.Heatmap(z=M, x=vocab, y=[f"D{i + 1}" for i in range(N)], colorscale="Blues",
                                hovertemplate="%{y}, %{x}: %{z:.3f}<extra></extra>"))
    _layout(heat, "TF-IDF weights (scikit-learn, L2-normalised rows)", height=160 + 40 * N)
    top = [{"document": f"D{i + 1}", "most characteristic terms": ", ".join(vocab[j] for j in np.argsort(-M[i])[:3])}
           for i in range(N)]
    return {
        "tables": {"term statistics": rows, "top terms": top},
        "metrics": {"documents": N, "vocabulary size": len(vocab)},
        "figures": {"heatmap": heat},
        "notes": [{"en": "Terms in every document get idf = 0 with the plain formula; scikit-learn smooths the idf "
                         "(ln((1+N)/(1+df)) + 1) and normalises each row, so its numbers differ slightly.",
                   "ru": "Har document mein aane wale terms ka plain formula mein idf = 0 hota hai; scikit-learn idf ko "
                         "smooth karta hai (ln((1+N)/(1+df)) + 1) aur har row normalise karta hai, is liye numbers thore mukhtalif hain."}],
    }


def run_cosine(vectors: List[np.ndarray], names: Optional[List[str]] = None) -> Dict:
    if len(vectors) < 2:
        raise DataError("enter at least two vectors")
    dims = {len(v) for v in vectors}
    if len(dims) != 1:
        raise DataError("all vectors must have the same length")
    V = np.array(vectors, float)
    names = names or [f"v{i + 1}" for i in range(len(V))]
    norms = np.linalg.norm(V, axis=1)
    if np.any(norms == 0):
        raise DataError("a zero vector has no direction")
    S = (V @ V.T) / np.outer(norms, norms)
    rows = []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            rows.append({"pair": f"{names[i]} . {names[j]}", "dot": r(V[i] @ V[j]), "||a||": r(norms[i]),
                         "||b||": r(norms[j]), "cosine": r(S[i, j]),
                         "angle (deg)": r(np.degrees(np.arccos(np.clip(S[i, j], -1, 1))), 2)})
    figures = {"matrix": _layout(go.Figure(go.Heatmap(z=S, x=names, y=names, colorscale="Blues", zmin=-1, zmax=1,
                                                      text=np.round(S, 3), texttemplate="%{text}")),
                                 "Cosine similarity matrix", height=150 + 50 * len(V))}
    if V.shape[1] == 2:
        fig = go.Figure()
        for i, v in enumerate(V):
            fig.add_trace(go.Scatter(x=[0, v[0]], y=[0, v[1]], mode="lines+markers+text", name=names[i],
                                     text=["", names[i]], textposition="top center",
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=4)))
        lim = float(np.abs(V).max()) * 1.2
        _layout(fig, "Vectors drawn from the origin (the angle decides the cosine)",
                xaxis=dict(range=[-lim, lim], zeroline=True), yaxis=dict(range=[-lim, lim], scaleanchor="x"))
        figures["arrows"] = fig
    return {"tables": {"pairwise": rows}, "metrics": {}, "figures": figures,
            "notes": [{"en": "cosine = dot / (||a|| ||b||); 1 = same direction, 0 = perpendicular, -1 = opposite.",
                       "ru": "cosine = dot / (||a|| ||b||); 1 = ek simt, 0 = perpendicular, -1 = ulti simt."}]}


def run_sentence_bert(sentences: List[str]) -> Dict:
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        raise DataError("enter at least two sentences (one per line)")
    labels = [f"S{i + 1}" for i in range(len(sentences))]
    try:
        from .embeddings import get_embedder

        E = get_embedder().encode(sentences)
        method, label = "sbert", "Sentence-BERT"
    except Exception as e:  # model not downloaded / offline
        from sklearn.feature_extraction.text import TfidfVectorizer

        E = TfidfVectorizer().fit_transform(sentences).toarray()
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        method, label = "tfidf", f"TF-IDF (Sentence-BERT unavailable: {type(e).__name__})"
    S = E @ E.T
    rows = [{"pair": f"{labels[i]} - {labels[j]}", "similarity": r(S[i, j], 3)}
            for i in range(len(S)) for j in range(i + 1, len(S))]
    rows.sort(key=lambda x: -x["similarity"])
    fig = _layout(go.Figure(go.Heatmap(z=S, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1,
                                       text=np.round(S, 2), texttemplate="%{text}")),
                  f"{label} cosine similarity", height=200 + 45 * len(S))
    from sklearn.feature_extraction.text import TfidfVectorizer

    T = TfidfVectorizer().fit_transform(sentences)
    T = (T @ T.T).toarray()
    compare = [{"pair": f"{labels[i]} - {labels[j]}", "embedding similarity": r(S[i, j], 3),
                "TF-IDF similarity": r(T[i, j], 3)} for i in range(len(S)) for j in range(i + 1, len(S))]
    return {
        "tables": {"sentences": [{"id": l, "sentence": s} for l, s in zip(labels, sentences)],
                   "most similar pairs": rows, "embedding vs TF-IDF": compare},
        "metrics": {"method": method, "embedding size": int(E.shape[1])},
        "figures": {"similarity": fig},
        "notes": [{"en": "Pairs with the same meaning but different words score high with Sentence-BERT and low with TF-IDF."
                         if method == "sbert" else "The Sentence-BERT model is not available on this computer yet "
                         "(it downloads on first use with internet); the table shows TF-IDF similarity instead.",
                   "ru": "Same maani lekin mukhtalif alfaaz wale pairs ka Sentence-BERT score zyada aur TF-IDF score kam hota hai."
                         if method == "sbert" else "Sentence-BERT model abhi is computer par nahi (internet ke saath pehli "
                         "dafa download hota hai); is liye table TF-IDF similarity dikha raha hai."}],
    }
