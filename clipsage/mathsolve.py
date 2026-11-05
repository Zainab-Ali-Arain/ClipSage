"""
Mathematics playground: step-by-step solutions computed (and verified) with
SymPy and NumPy - never just written as text.

Every solver returns a dict:
    kind, title {en, ru}, concept {en, ru},
    steps   [ {en, ru, latex?} ]   human-readable working,
    answer  the final value, answer_text  (string form),
    verified  how the result was double-checked (e.g. "NumPy", "SymPy substitution").

`practice_problem(kind, seed)` makes a similar random problem with a known
answer; `check_practice(problem, text)` grades the student's answer.
`detect_and_solve(text)` handles typed questions such as
"derivative of x^3 + 2x" or "solve 2x + 3 = 11".
"""

import random
import re
from fractions import Fraction
from typing import Dict, List, Optional

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

X = sp.symbols("x")
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
ALLOWED_FUNCS = {"sin", "cos", "tan", "exp", "log", "ln", "sqrt", "pi", "e", "x", "abs"}

KINDS = {
    "matrix_multiply": {"en": "Matrix multiplication", "ru": "Matrix multiplication"},
    "determinant": {"en": "Determinant", "ru": "Determinant"},
    "inverse": {"en": "Matrix inverse", "ru": "Matrix ka inverse"},
    "eigen": {"en": "Eigenvalues (2 x 2)", "ru": "Eigenvalues (2 x 2)"},
    "gaussian_elimination": {"en": "Gaussian elimination / rank", "ru": "Gaussian elimination / rank"},
    "vectors": {"en": "Vectors: dot product, norm, cosine", "ru": "Vectors: dot product, norm, cosine"},
    "statistics": {"en": "Mean, variance, standard deviation", "ru": "Mean, variance, standard deviation"},
    "bayes": {"en": "Bayes' theorem", "ru": "Bayes' theorem"},
    "binomial": {"en": "Binomial probability", "ru": "Binomial probability"},
    "derivative": {"en": "Derivative", "ru": "Derivative"},
    "integral": {"en": "Integral", "ru": "Integral"},
    "optimize": {"en": "Minimum / maximum", "ru": "Minimum / maximum"},
    "equation": {"en": "Solve an equation", "ru": "Equation hal karein"},
    "gradient_descent": {"en": "Gradient descent", "ru": "Gradient descent"},
}


class MathInputError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_expression(text: str) -> sp.Expr:
    """Safely parse a one-variable expression in x (only maths names are allowed)."""
    text = (text or "").strip().replace("−", "-").replace("·", "*")
    if not text:
        raise MathInputError("empty expression")
    if "__" in text or len(text) > 200:
        raise MathInputError("expression not allowed")
    for name in re.findall(r"[A-Za-z_]+", text):
        if name.lower() not in ALLOWED_FUNCS:
            raise MathInputError(f"unknown name '{name}' (use x as the variable)")
    text = re.sub(r"\bln\b", "log", text)
    try:
        expr = parse_expr(text, transformations=TRANSFORMS, local_dict={"x": X, "e": sp.E, "pi": sp.pi})
    except Exception as e:  # SyntaxError, TokenError, ...
        raise MathInputError(f"could not parse '{text}'") from e
    return sp.sympify(expr)


def parse_matrix(text: str) -> np.ndarray:
    """'1 2; 3 4' or '1,2\\n3,4' -> 2-D array."""
    rows = [r for r in re.split(r"[;\n]", (text or "").strip()) if r.strip()]
    if not rows:
        raise MathInputError("empty matrix")
    try:
        data = [[float(Fraction(v)) for v in re.split(r"[,\s]+", r.strip()) if v] for r in rows]
    except (ValueError, ZeroDivisionError) as e:
        raise MathInputError("matrix entries must be numbers") from e
    if len({len(r) for r in data}) != 1:
        raise MathInputError("every row must have the same number of entries")
    return np.array(data)


def parse_vector(text: str) -> np.ndarray:
    try:
        vals = [float(Fraction(v)) for v in re.split(r"[,\s;]+", (text or "").strip().strip("()[]")) if v]
    except (ValueError, ZeroDivisionError) as e:
        raise MathInputError("vector entries must be numbers") from e
    if not vals:
        raise MathInputError("empty vector")
    return np.array(vals)


def fmt(v, digits: int = 4) -> str:
    """Pretty number: integers without decimals, others rounded."""
    if isinstance(v, (sp.Basic,)):
        v = float(v) if v.is_number and v.is_real else v
        if isinstance(v, sp.Basic):
            return str(v)
    v = float(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{round(v, digits):g}"


def mat_str(m: np.ndarray) -> str:
    return "[" + ", ".join("[" + ", ".join(fmt(v) for v in row) + "]" for row in np.atleast_2d(m)) + "]"


def mat_latex(m) -> str:
    m = np.atleast_2d(m)
    body = " \\\\ ".join(" & ".join(fmt(v) for v in row) for row in m)
    return f"\\begin{{pmatrix}} {body} \\end{{pmatrix}}"


def _step(en: str, ru: str, latex: Optional[str] = None) -> dict:
    d = {"en": en, "ru": ru}
    if latex:
        d["latex"] = latex
    return d


def _result(kind, steps, answer, answer_text, verified, concept, **extra) -> dict:
    return {"kind": kind, "title": KINDS[kind], "concept": concept, "steps": steps,
            "answer": answer, "answer_text": answer_text, "verified": verified, **extra}


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------
def matrix_multiply(A: np.ndarray, B: np.ndarray) -> dict:
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    if A.shape[1] != B.shape[0]:
        raise MathInputError(f"cannot multiply {A.shape[0]}x{A.shape[1]} by {B.shape[0]}x{B.shape[1]}: "
                             "the inner sizes must match")
    m, n, p = A.shape[0], A.shape[1], B.shape[1]
    C = np.zeros((m, p))
    steps = [_step(f"A is {m}x{n} and B is {n}x{p}, so C = AB is {m}x{p}.",
                   f"A {m}x{n} aur B {n}x{p} hai, is liye C = AB {m}x{p} hoga.")]
    for i in range(m):
        for j in range(p):
            terms = [f"{fmt(A[i, k])}*{fmt(B[k, j])}" for k in range(n)]
            C[i, j] = sum(A[i, k] * B[k, j] for k in range(n))
            steps.append(_step(f"C[{i + 1},{j + 1}] = row {i + 1} of A . column {j + 1} of B = "
                               f"{' + '.join(terms)} = {fmt(C[i, j])}",
                               f"C[{i + 1},{j + 1}] = A ki row {i + 1} . B ka column {j + 1} = "
                               f"{' + '.join(terms)} = {fmt(C[i, j])}"))
    check = np.allclose(C, A @ B)
    return _result("matrix_multiply", steps, C, mat_str(C), "NumPy A @ B" if check else "MISMATCH",
                   {"en": "Each entry of AB is the dot product of a row of A with a column of B.",
                    "ru": "AB ka har element A ki ek row aur B ke ek column ka dot product hai."},
                   answer_latex=mat_latex(C))


def determinant(A: np.ndarray) -> dict:
    A = np.atleast_2d(A)
    if A.shape[0] != A.shape[1]:
        raise MathInputError("the determinant needs a square matrix")
    n = A.shape[0]
    steps = []
    if n == 1:
        det = A[0, 0]
        steps.append(_step(f"For a 1x1 matrix the determinant is the entry itself: {fmt(det)}.",
                           f"1x1 matrix ka determinant khud wahi element hai: {fmt(det)}."))
    elif n == 2:
        a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
        det = a * d - b * c
        steps.append(_step("For [[a, b], [c, d]] the determinant is ad - bc.",
                           "[[a, b], [c, d]] ka determinant ad - bc hai.", "\\det = ad - bc"))
        steps.append(_step(f"= {fmt(a)}*{fmt(d)} - {fmt(b)}*{fmt(c)} = {fmt(a * d)} - {fmt(b * c)} = {fmt(det)}",
                           f"= {fmt(a)}*{fmt(d)} - {fmt(b)}*{fmt(c)} = {fmt(a * d)} - {fmt(b * c)} = {fmt(det)}"))
    elif n == 3:
        steps.append(_step("Expand along the first row: det = a11 M11 - a12 M12 + a13 M13, where Mij is the 2x2 minor.",
                           "Pehli row ke saath phailayein: det = a11 M11 - a12 M12 + a13 M13, jahan Mij 2x2 minor hai."))
        det = 0.0
        for j in range(3):
            minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
            mdet = minor[0, 0] * minor[1, 1] - minor[0, 1] * minor[1, 0]
            sign = (-1) ** j
            det += sign * A[0, j] * mdet
            steps.append(_step(f"M1{j + 1} = det{mat_str(minor)} = {fmt(mdet)}; term = {'+' if sign > 0 else '-'}"
                               f"{fmt(A[0, j])} x {fmt(mdet)} = {fmt(sign * A[0, j] * mdet)}",
                               f"M1{j + 1} = det{mat_str(minor)} = {fmt(mdet)}; term = {'+' if sign > 0 else '-'}"
                               f"{fmt(A[0, j])} x {fmt(mdet)} = {fmt(sign * A[0, j] * mdet)}"))
        steps.append(_step(f"Add the three terms: det = {fmt(det)}", f"Teeno terms jama karein: det = {fmt(det)}"))
    else:
        det = float(sp.Matrix(A.tolist()).det())
        steps.append(_step(f"For an {n}x{n} matrix SymPy computes det = {fmt(det)} (by elimination).",
                           f"{n}x{n} matrix ke liye SymPy det = {fmt(det)} nikalta hai (elimination se)."))
    ok = np.isclose(det, np.linalg.det(A))
    note = ({"en": "The determinant is 0, so the matrix is singular (no inverse).",
             "ru": "Determinant 0 hai, yaani matrix singular hai (inverse nahi)."} if np.isclose(det, 0) else None)
    if note:
        steps.append(note)
    return _result("determinant", steps, float(det), fmt(det), "NumPy linalg.det" if ok else "MISMATCH",
                   {"en": "The determinant measures how a matrix scales area/volume; 0 means it is not invertible.",
                    "ru": "Determinant batata hai matrix area/volume ko kitna barhati hai; 0 ka matlab inverse nahi."})


def inverse(A: np.ndarray) -> dict:
    A = np.atleast_2d(A)
    if A.shape[0] != A.shape[1]:
        raise MathInputError("only square matrices have an inverse")
    det_res = determinant(A)
    det = det_res["answer"]
    steps = [_step(f"First the determinant: det(A) = {fmt(det)}.", f"Pehle determinant: det(A) = {fmt(det)}.")]
    if np.isclose(det, 0):
        steps.append(_step("det(A) = 0, so A has no inverse.", "det(A) = 0, is liye A ka inverse nahi."))
        return _result("inverse", steps, None, "no inverse (singular)", "determinant = 0",
                       {"en": "Only matrices with a non-zero determinant are invertible.",
                        "ru": "Sirf non-zero determinant wali matrices ka inverse hota hai."})
    if A.shape[0] == 2:
        a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
        adj = np.array([[d, -b], [-c, a]])
        inv = adj / det
        steps.append(_step("For a 2x2 matrix: A^-1 = (1/det) [[d, -b], [-c, a]].",
                           "2x2 matrix ke liye: A^-1 = (1/det) [[d, -b], [-c, a]].",
                           "A^{-1} = \\frac{1}{ad-bc}\\begin{pmatrix} d & -b \\\\ -c & a \\end{pmatrix}"))
        steps.append(_step(f"Swap a and d, negate b and c: {mat_str(adj)}.",
                           f"a aur d ki jagah badlein, b aur c ka sign ulta karein: {mat_str(adj)}."))
        steps.append(_step(f"Divide by {fmt(det)}: A^-1 = {mat_str(inv)}.", f"{fmt(det)} se taqseem: A^-1 = {mat_str(inv)}."))
    else:
        inv = np.array(sp.Matrix(A.tolist()).inv().evalf(), dtype=float)
        steps.append(_step(f"For a larger matrix SymPy uses Gauss-Jordan elimination: A^-1 = {mat_str(inv)}.",
                           f"Bari matrix ke liye SymPy Gauss-Jordan elimination istemal karta hai: A^-1 = {mat_str(inv)}."))
    ok = np.allclose(A @ inv, np.eye(A.shape[0]))
    steps.append(_step(f"Check: A x A^-1 = {mat_str(np.round(A @ inv, 10))} = I.",
                       f"Check: A x A^-1 = {mat_str(np.round(A @ inv, 10))} = I."))
    return _result("inverse", steps, inv, mat_str(inv), "A @ A^-1 = I (NumPy)" if ok else "MISMATCH",
                   {"en": "The inverse undoes the matrix: A A^-1 = I.", "ru": "Inverse matrix ke asar ko ulat deta hai: A A^-1 = I."},
                   answer_latex=mat_latex(inv))


def eigen2(A: np.ndarray) -> dict:
    A = np.atleast_2d(A)
    if A.shape != (2, 2):
        raise MathInputError("enter a 2x2 matrix")
    tr, det = A[0, 0] + A[1, 1], A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    lam = sp.symbols("lambda")
    roots = sp.solve(lam ** 2 - tr * lam + det, lam)
    steps = [
        _step("Eigenvalues solve det(A - lambda I) = 0. For 2x2 this is lambda^2 - trace x lambda + det = 0.",
              "Eigenvalues det(A - lambda I) = 0 se nikalti hain. 2x2 ke liye: lambda^2 - trace x lambda + det = 0.",
              "\\lambda^2 - \\text{tr}(A)\\lambda + \\det(A) = 0"),
        _step(f"trace = {fmt(A[0, 0])} + {fmt(A[1, 1])} = {fmt(tr)}, det = {fmt(det)}",
              f"trace = {fmt(A[0, 0])} + {fmt(A[1, 1])} = {fmt(tr)}, det = {fmt(det)}"),
        _step(f"lambda^2 - {fmt(tr)} lambda + {fmt(det)} = 0  ->  lambda = {', '.join(str(sp.nsimplify(r)) for r in roots)}",
              f"lambda^2 - {fmt(tr)} lambda + {fmt(det)} = 0  ->  lambda = {', '.join(str(sp.nsimplify(r)) for r in roots)}"),
    ]
    numeric = sorted(np.linalg.eigvals(A), key=lambda z: (z.real, z.imag))
    sym = sorted([complex(sp.N(r)) for r in roots], key=lambda z: (z.real, z.imag))
    ok = np.allclose(numeric, sym)
    text = ", ".join(fmt(z.real) if abs(z.imag) < 1e-12 else f"{fmt(z.real)} {'+' if z.imag > 0 else '-'} {fmt(abs(z.imag))}i"
                     for z in sym)
    return _result("eigen", steps, sym, text, "NumPy linalg.eigvals" if ok else "MISMATCH",
                   {"en": "Eigenvalues tell how much a matrix stretches its special (eigen) directions; PCA uses them.",
                    "ru": "Eigenvalues batati hain matrix apni khaas (eigen) simton ko kitna khenchti hai; PCA inhein istemal karta hai."})


def _row_str(row) -> str:
    return "[" + ", ".join(str(x) for x in row) + "]"


def gaussian_elimination(aug: np.ndarray) -> dict:
    """
    Row-reduce an augmented matrix [A | b] (last column = b) to reduced row
    echelon form with exact fractions, recording every row operation.
    """
    M = sp.Matrix(np.atleast_2d(aug).tolist()).applyfunc(lambda v: sp.nsimplify(v))
    rows, cols = M.shape
    if cols < 2:
        raise MathInputError("enter an augmented matrix with at least one variable column and the b column")
    n_vars = cols - 1
    steps = [_step(f"Augmented matrix [A | b]: {', '.join(_row_str(M.row(i)) for i in range(rows))}",
                   f"Augmented matrix [A | b]: {', '.join(_row_str(M.row(i)) for i in range(rows))}")]
    pivot_cols, r = [], 0
    for c in range(n_vars):
        piv = next((i for i in range(r, rows) if M[i, c] != 0), None)
        if piv is None:
            continue
        if piv != r:
            M.row_swap(piv, r)
            steps.append(_step(f"Swap R{piv + 1} and R{r + 1} to get a non-zero pivot.",
                               f"Non-zero pivot ke liye R{piv + 1} aur R{r + 1} ki jagah badlein."))
        if M[r, c] != 1:
            f = M[r, c]
            M[r, :] = M[r, :] / f
            steps.append(_step(f"R{r + 1} = R{r + 1} / {f}  ->  {_row_str(M.row(r))}",
                               f"R{r + 1} = R{r + 1} / {f}  ->  {_row_str(M.row(r))}"))
        for i in range(rows):
            if i != r and M[i, c] != 0:
                f = M[i, c]
                M[i, :] = M[i, :] - f * M[r, :]
                steps.append(_step(f"R{i + 1} = R{i + 1} - ({f}) R{r + 1}  ->  {_row_str(M.row(i))}",
                                   f"R{i + 1} = R{i + 1} - ({f}) R{r + 1}  ->  {_row_str(M.row(i))}"))
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
    rank = len(pivot_cols)
    inconsistent = any(all(M[i, j] == 0 for j in range(n_vars)) and M[i, n_vars] != 0 for i in range(rows))
    steps.append(_step(f"Reduced row echelon form: {', '.join(_row_str(M.row(i)) for i in range(rows))}; rank(A) = {rank}",
                       f"Reduced row echelon form: {', '.join(_row_str(M.row(i)) for i in range(rows))}; rank(A) = {rank}"))
    names = [f"x{j + 1}" for j in range(n_vars)]
    if inconsistent:
        kind, text, answer = "none", "no solution (inconsistent system)", None
        steps.append(_step("A row reads 0 = non-zero, so the system has no solution.",
                           "Ek row 0 = non-zero kehti hai, is liye system ka koi hal nahi."))
    elif rank == n_vars:
        sol = [M[pivot_cols.index(j), n_vars] for j in range(n_vars)]
        kind, answer = "unique", [float(v) for v in sol]
        text = ", ".join(f"{nm} = {v}" for nm, v in zip(names, sol))
        steps.append(_step(f"Every variable has a pivot: unique solution {text}.",
                           f"Har variable ka pivot hai: ek hi hal {text}."))
    else:
        free = [names[j] for j in range(n_vars) if j not in pivot_cols]
        kind, answer = "infinite", None
        text = f"infinitely many solutions (free variables: {', '.join(free)})"
        steps.append(_step(f"Free variables {', '.join(free)}: infinitely many solutions.",
                           f"Free variables {', '.join(free)}: be-shumar hal."))
    A = np.array(np.atleast_2d(aug)[:, :-1], dtype=float)
    ok = rank == np.linalg.matrix_rank(A)
    if kind == "unique":
        ok = ok and np.allclose(A @ np.array(answer), np.array(np.atleast_2d(aug)[:, -1], dtype=float))
    return _result("gaussian_elimination", steps, answer, text,
                   "NumPy matrix_rank + substitution" if ok else "MISMATCH",
                   {"en": "Row operations do not change the solutions; the pivots give the rank.",
                    "ru": "Row operations hal nahi badaltin; pivots rank batate hain."},
                   rank=rank, solution_kind=kind, rref=[[str(x) for x in M.row(i)] for i in range(rows)])


def vectors(u: np.ndarray, v: np.ndarray) -> dict:
    u, v = np.asarray(u, float), np.asarray(v, float)
    if u.shape != v.shape:
        raise MathInputError("the two vectors must have the same length")
    dot = float(u @ v)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    steps = [
        _step(f"Dot product: {' + '.join(f'{fmt(a)}*{fmt(b)}' for a, b in zip(u, v))} = {fmt(dot)}",
              f"Dot product: {' + '.join(f'{fmt(a)}*{fmt(b)}' for a, b in zip(u, v))} = {fmt(dot)}", "u \\cdot v = \\sum_i u_i v_i"),
        _step(f"||u|| = sqrt({' + '.join(f'{fmt(a)}^2' for a in u)}) = {fmt(nu)}",
              f"||u|| = sqrt({' + '.join(f'{fmt(a)}^2' for a in u)}) = {fmt(nu)}"),
        _step(f"||v|| = sqrt({' + '.join(f'{fmt(b)}^2' for b in v)}) = {fmt(nv)}",
              f"||v|| = sqrt({' + '.join(f'{fmt(b)}^2' for b in v)}) = {fmt(nv)}"),
    ]
    out = {"dot": dot, "norm_u": nu, "norm_v": nv}
    if nu == 0 or nv == 0:
        steps.append(_step("A zero vector has no direction, so cosine similarity is undefined.",
                           "Zero vector ki koi simt nahi, is liye cosine similarity nahi nikal sakti."))
        cos = None
    else:
        cos = dot / (nu * nv)
        angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        steps.append(_step(f"cosine = {fmt(dot)} / ({fmt(nu)} x {fmt(nv)}) = {fmt(cos)}",
                           f"cosine = {fmt(dot)} / ({fmt(nu)} x {fmt(nv)}) = {fmt(cos)}",
                           "\\cos\\theta = \\frac{u \\cdot v}{\\lVert u \\rVert \\lVert v \\rVert}"))
        steps.append(_step(f"angle = arccos({fmt(cos)}) = {fmt(angle, 2)} degrees",
                           f"angle = arccos({fmt(cos)}) = {fmt(angle, 2)} degree"))
        out.update(cosine=cos, angle=angle)
        if nv:
            proj = (dot / nv ** 2) * v
            steps.append(_step(f"Projection of u on v = (u.v / ||v||^2) v = {mat_str(proj)}",
                               f"u ka v par projection = (u.v / ||v||^2) v = {mat_str(proj)}"))
            out["projection"] = proj
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos

    ok = cos is None or np.isclose(cos, sk_cos([u], [v])[0, 0])
    return _result("vectors", steps, out, f"dot = {fmt(dot)}, cosine = {fmt(cos) if cos is not None else 'undefined'}",
                   "scikit-learn cosine_similarity" if ok else "MISMATCH",
                   {"en": "Cosine similarity compares directions; ClipSage uses it to match queries with lecture segments.",
                    "ru": "Cosine similarity simtein milati hai; ClipSage isi se queries ko lecture segments se milata hai."},
                   u=u, v=v)


# ---------------------------------------------------------------------------
# Probability and statistics
# ---------------------------------------------------------------------------
def statistics(data: np.ndarray) -> dict:
    data = np.asarray(data, float)
    n = len(data)
    if n < 2:
        raise MathInputError("enter at least two numbers")
    mean = data.mean()
    dev = data - mean
    ss = float(np.sum(dev ** 2))
    steps = [
        _step(f"n = {n}, sum = {fmt(data.sum())}, mean = {fmt(data.sum())} / {n} = {fmt(mean)}",
              f"n = {n}, sum = {fmt(data.sum())}, mean = {fmt(data.sum())} / {n} = {fmt(mean)}", "\\bar{x} = \\frac{1}{n}\\sum x_i"),
        _step("Squared deviations (x - mean)^2: " + ", ".join(fmt(d ** 2) for d in dev) + f"; their sum = {fmt(ss)}",
              "Squared farq (x - mean)^2: " + ", ".join(fmt(d ** 2) for d in dev) + f"; in ka sum = {fmt(ss)}"),
        _step(f"Population variance = {fmt(ss)} / {n} = {fmt(ss / n)}; standard deviation = {fmt(np.sqrt(ss / n))}",
              f"Population variance = {fmt(ss)} / {n} = {fmt(ss / n)}; standard deviation = {fmt(np.sqrt(ss / n))}",
              "\\sigma^2 = \\frac{1}{n}\\sum (x_i - \\bar{x})^2"),
        _step(f"Sample variance = {fmt(ss)} / {n - 1} = {fmt(ss / (n - 1))}; sample std = {fmt(np.sqrt(ss / (n - 1)))}",
              f"Sample variance = {fmt(ss)} / {n - 1} = {fmt(ss / (n - 1))}; sample std = {fmt(np.sqrt(ss / (n - 1)))}",
              "s^2 = \\frac{1}{n-1}\\sum (x_i - \\bar{x})^2"),
        _step(f"Median (middle of the sorted data {mat_str(np.sort(data))[1:-1]}) = {fmt(np.median(data))}",
              f"Median (sorted data {mat_str(np.sort(data))[1:-1]} ka beech) = {fmt(np.median(data))}"),
    ]
    out = {"mean": mean, "median": float(np.median(data)), "var_pop": ss / n, "std_pop": np.sqrt(ss / n),
           "var_sample": ss / (n - 1), "std_sample": np.sqrt(ss / (n - 1))}
    ok = np.isclose(out["std_pop"], data.std()) and np.isclose(out["var_sample"], data.var(ddof=1))
    return _result("statistics", steps, out,
                   f"mean = {fmt(mean)}, variance = {fmt(ss / n)}, std = {fmt(np.sqrt(ss / n))}",
                   "NumPy mean/var/std" if ok else "MISMATCH",
                   {"en": "The mean is the centre; variance and standard deviation measure the spread.",
                    "ru": "Mean markaz hai; variance aur standard deviation phailao naapte hain."}, data=data)


def bayes(prior: float, likelihood: float, false_positive: float) -> dict:
    for name, v in [("prior", prior), ("P(E|H)", likelihood), ("P(E|not H)", false_positive)]:
        if not 0 <= v <= 1:
            raise MathInputError(f"{name} must be between 0 and 1")
    evidence = likelihood * prior + false_positive * (1 - prior)
    if evidence == 0:
        raise MathInputError("P(E) is 0, the evidence is impossible")
    post = likelihood * prior / evidence
    steps = [
        _step("Bayes' theorem: P(H|E) = P(E|H) P(H) / P(E).", "Bayes' theorem: P(H|E) = P(E|H) P(H) / P(E).",
              "P(H \\mid E) = \\frac{P(E \\mid H)\\,P(H)}{P(E)}"),
        _step(f"Total probability: P(E) = P(E|H)P(H) + P(E|not H)P(not H) = {fmt(likelihood)} x {fmt(prior)} + "
              f"{fmt(false_positive)} x {fmt(1 - prior)} = {fmt(evidence, 6)}",
              f"Total probability: P(E) = {fmt(likelihood)} x {fmt(prior)} + {fmt(false_positive)} x {fmt(1 - prior)} = {fmt(evidence, 6)}"),
        _step(f"P(H|E) = {fmt(likelihood * prior, 6)} / {fmt(evidence, 6)} = {fmt(post)}",
              f"P(H|E) = {fmt(likelihood * prior, 6)} / {fmt(evidence, 6)} = {fmt(post)}"),
    ]
    h = sp.Rational(str(prior)) if prior else sp.Integer(0)
    exact = sp.Rational(str(likelihood)) * h / (sp.Rational(str(likelihood)) * h +
                                                  sp.Rational(str(false_positive)) * (1 - h))
    ok = np.isclose(float(exact), post)
    return _result("bayes", steps, post, fmt(post), "SymPy exact fractions" if ok else "MISMATCH",
                   {"en": "Bayes' theorem updates a prior belief with new evidence.",
                    "ru": "Bayes' theorem naye saboot se pehle wale andaze ko update karta hai."})


def binomial(n: int, k: int, p: float) -> dict:
    if not (0 <= k <= n) or not 0 <= p <= 1:
        raise MathInputError("need 0 <= k <= n and 0 <= p <= 1")
    comb = sp.binomial(n, k)
    prob = float(comb) * p ** k * (1 - p) ** (n - k)
    steps = [
        _step("P(X = k) = C(n, k) p^k (1-p)^(n-k).", "P(X = k) = C(n, k) p^k (1-p)^(n-k).",
              "P(X=k)=\\binom{n}{k}p^k(1-p)^{n-k}"),
        _step(f"C({n}, {k}) = {n}! / ({k}! {n - k}!) = {comb}", f"C({n}, {k}) = {n}! / ({k}! {n - k}!) = {comb}"),
        _step(f"P = {comb} x {fmt(p)}^{k} x {fmt(1 - p)}^{n - k} = {fmt(prob, 6)}",
              f"P = {comb} x {fmt(p)}^{k} x {fmt(1 - p)}^{n - k} = {fmt(prob, 6)}"),
    ]
    from scipy.stats import binom

    ok = np.isclose(binom.pmf(k, n, p), prob)
    return _result("binomial", steps, prob, fmt(prob, 6), "SciPy binom.pmf" if ok else "MISMATCH",
                   {"en": "The binomial distribution counts successes in n independent yes/no trials.",
                    "ru": "Binomial distribution n azad haan/nahi koshishon mein kamyabiyan ginti hai."})


# ---------------------------------------------------------------------------
# Calculus and optimisation
# ---------------------------------------------------------------------------
def _rule_name(term: sp.Expr) -> Dict[str, str]:
    if term.is_number:
        return {"en": "constant rule (derivative 0)", "ru": "constant rule (derivative 0)"}
    if term.is_Pow and term.base == X and term.exp.is_number:
        return {"en": "power rule", "ru": "power rule"}
    if term.is_Mul and any(f.is_number for f in term.args):
        return {"en": "constant multiple + inner rule", "ru": "constant multiple + andar ka qaida"}
    if term == X:
        return {"en": "d/dx x = 1", "ru": "d/dx x = 1"}
    if isinstance(term, (sp.sin, sp.cos, sp.tan, sp.exp, sp.log)) and term.args[0] != X:
        return {"en": "chain rule", "ru": "chain rule"}
    if term.is_Mul:
        return {"en": "product rule", "ru": "product rule"}
    return {"en": "standard derivative", "ru": "standard derivative"}


def derivative(expr_text: str, at: Optional[float] = None, order: int = 1) -> dict:
    f = parse_expression(expr_text)
    steps = [_step(f"f(x) = {sp.sstr(f)}", f"f(x) = {sp.sstr(f)}", f"f(x) = {sp.latex(f)}")]
    if order == 1:
        terms = sp.Add.make_args(sp.expand(f)) if f.is_polynomial(X) else sp.Add.make_args(f)
        if len(terms) > 1:
            steps.append(_step("Differentiate term by term (the derivative of a sum is the sum of the derivatives):",
                               "Har term ka alag derivative lein (sum ka derivative derivatives ka sum hai):"))
        for t in terms:
            rule = _rule_name(t)
            steps.append(_step(f"d/dx [{sp.sstr(t)}] = {sp.sstr(sp.diff(t, X))}   ({rule['en']})",
                               f"d/dx [{sp.sstr(t)}] = {sp.sstr(sp.diff(t, X))}   ({rule['ru']})",
                               f"\\frac{{d}}{{dx}}\\left[{sp.latex(t)}\\right] = {sp.latex(sp.diff(t, X))}"))
    raw = sp.diff(f, X, order)
    df = sp.expand(raw) if raw.is_polynomial(X) else sp.simplify(raw)
    label = "f'(x)" if order == 1 else f"f^({order})(x)"
    steps.append(_step(f"{label} = {sp.sstr(df)}", f"{label} = {sp.sstr(df)}",
                       f"f{chr(39) * order}(x) = {sp.latex(df)}"))
    answer, text = df, sp.sstr(df)
    if at is not None:
        val = df.subs(X, at)
        steps.append(_step(f"At x = {fmt(at)}: {label} = {fmt(sp.N(val))}", f"x = {fmt(at)} par: {label} = {fmt(sp.N(val))}"))
        answer, text = float(sp.N(val)), fmt(sp.N(val))
    # numerical verification with a central difference
    x0 = 0.7 if at is None else at
    fn = sp.lambdify(X, f, "numpy")
    h = 1e-5
    numeric = (fn(x0 + h) - fn(x0 - h)) / (2 * h) if order == 1 else None
    ok = numeric is None or np.isclose(float(sp.N(sp.diff(f, X).subs(X, x0))), numeric, rtol=1e-4, atol=1e-6)
    return _result("derivative", steps, answer, text,
                   "SymPy diff + numerical central difference" if ok else "MISMATCH",
                   {"en": "The derivative is the slope of the function: how fast f changes when x changes.",
                    "ru": "Derivative function ka slope hai: x badalne par f kitni tezi se badalta hai."},
                   expr=f, deriv=df)


def integral(expr_text: str, a: Optional[float] = None, b: Optional[float] = None) -> dict:
    f = parse_expression(expr_text)
    F = sp.integrate(f, X)
    steps = [_step(f"f(x) = {sp.sstr(f)}", f"f(x) = {sp.sstr(f)}", f"f(x) = {sp.latex(f)}"),
             _step(f"Antiderivative F(x) = {sp.sstr(F)} + C   (check: F'(x) = {sp.sstr(sp.simplify(sp.diff(F, X)))})",
                   f"Antiderivative F(x) = {sp.sstr(F)} + C   (check: F'(x) = {sp.sstr(sp.simplify(sp.diff(F, X)))})",
                   f"\\int {sp.latex(f)}\\,dx = {sp.latex(F)} + C")]
    ok = sp.simplify(sp.diff(F, X) - f) == 0
    if a is None or b is None:
        return _result("integral", steps, F, f"{sp.sstr(F)} + C", "SymPy: dF/dx = f" if ok else "MISMATCH",
                       {"en": "Integration reverses differentiation.", "ru": "Integration differentiation ka ulat hai."},
                       expr=f)
    val = sp.N(F.subs(X, b) - F.subs(X, a))
    steps.append(_step(f"Definite integral = F({fmt(b)}) - F({fmt(a)}) = {fmt(sp.N(F.subs(X, b)))} - "
                       f"{fmt(sp.N(F.subs(X, a)))} = {fmt(val)}",
                       f"Definite integral = F({fmt(b)}) - F({fmt(a)}) = {fmt(val)}"))
    from scipy.integrate import quad

    numeric, _ = quad(sp.lambdify(X, f, "numpy"), a, b)
    ok = ok and np.isclose(float(val), numeric, rtol=1e-6, atol=1e-8)
    return _result("integral", steps, float(val), fmt(val), "SymPy + SciPy numerical quadrature" if ok else "MISMATCH",
                   {"en": "The definite integral is the signed area under f between a and b.",
                    "ru": "Definite integral a aur b ke darmiyan f ke neeche ka (sign wala) area hai."},
                   expr=f, bounds=(a, b))


def optimize(expr_text: str) -> dict:
    f = parse_expression(expr_text)
    df, d2f = sp.diff(f, X), sp.diff(f, X, 2)
    steps = [_step(f"f(x) = {sp.sstr(f)}", f"f(x) = {sp.sstr(f)}", f"f(x) = {sp.latex(f)}"),
             _step(f"f'(x) = {sp.sstr(df)}; set f'(x) = 0.", f"f'(x) = {sp.sstr(df)}; f'(x) = 0 rakhein.")]
    try:
        crit = [c for c in sp.solve(sp.Eq(df, 0), X) if c.is_real]
    except NotImplementedError:
        crit = []
    if not crit:
        steps.append(_step("No real critical points were found.", "Koi real critical point nahi mila."))
        return _result("optimize", steps, [], "no critical points", "SymPy solve",
                       {"en": "Minima and maxima occur where the derivative is zero.",
                        "ru": "Minimum aur maximum wahan hote hain jahan derivative zero ho."}, expr=f)
    points = []
    for c in crit:
        second = sp.N(d2f.subs(X, c))
        kind = ("minimum", "minimum") if second > 0 else ("maximum", "maximum") if second < 0 else ("inconclusive", "ghair wazeh")
        val = sp.N(f.subs(X, c))
        points.append({"x": float(sp.N(c)), "f": float(val), "type": kind[0]})
        steps.append(_step(f"x = {sp.sstr(c)}: f''(x) = {fmt(second)} -> {kind[0]}, f(x) = {fmt(val)}",
                           f"x = {sp.sstr(c)}: f''(x) = {fmt(second)} -> {kind[1]}, f(x) = {fmt(val)}"))
    ok = all(abs(float(sp.N(df.subs(X, p["x"])))) < 1e-8 for p in points)
    text = "; ".join(f"{p['type']} at x = {fmt(p['x'])} (f = {fmt(p['f'])})" for p in points)
    return _result("optimize", steps, points, text, "f'(x*) = 0 checked by substitution" if ok else "MISMATCH",
                   {"en": "At a minimum the slope is 0 and the curve bends upwards (f'' > 0).",
                    "ru": "Minimum par slope 0 hota hai aur curve upar ki taraf murta hai (f'' > 0)."}, expr=f)


def solve_equation(eq_text: str) -> dict:
    if "=" in eq_text:
        left, right = eq_text.split("=", 1)
    else:
        left, right = eq_text, "0"
    lhs, rhs = parse_expression(left), parse_expression(right)
    expr = sp.simplify(lhs - rhs)
    steps = [_step(f"Equation: {sp.sstr(lhs)} = {sp.sstr(rhs)}", f"Equation: {sp.sstr(lhs)} = {sp.sstr(rhs)}",
                   f"{sp.latex(lhs)} = {sp.latex(rhs)}"),
             _step(f"Move everything to one side: {sp.sstr(expr)} = 0", f"Sab kuch ek taraf le jayein: {sp.sstr(expr)} = 0")]
    poly = sp.Poly(expr, X) if expr.is_polynomial(X) else None
    if poly is not None and poly.degree() == 1:
        a, b = poly.all_coeffs()
        steps.append(_step(f"Linear: {sp.sstr(a)}x + ({sp.sstr(b)}) = 0  ->  x = -({sp.sstr(b)}) / {sp.sstr(a)}",
                           f"Linear: {sp.sstr(a)}x + ({sp.sstr(b)}) = 0  ->  x = -({sp.sstr(b)}) / {sp.sstr(a)}"))
    elif poly is not None and poly.degree() == 2:
        a, b, c = poly.all_coeffs()
        disc = b ** 2 - 4 * a * c
        steps.append(_step(f"Quadratic formula x = (-b +/- sqrt(b^2 - 4ac)) / 2a with a = {a}, b = {b}, c = {c}; "
                           f"discriminant = {sp.sstr(disc)}",
                           f"Quadratic formula x = (-b +/- sqrt(b^2 - 4ac)) / 2a, a = {a}, b = {b}, c = {c}; "
                           f"discriminant = {sp.sstr(disc)}", "x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}"))
    sols = sp.solve(sp.Eq(lhs, rhs), X)
    steps.append(_step(f"Solution(s): x = {', '.join(sp.sstr(s) for s in sols) or 'none'}",
                       f"Hal: x = {', '.join(sp.sstr(s) for s in sols) or 'koi nahi'}"))
    ok = all(abs(complex(sp.N((lhs - rhs).subs(X, s)))) < 1e-9 for s in sols)
    if sols:
        steps.append(_step("Check by substituting each solution back into the equation: both sides are equal.",
                           "Har hal ko wapas equation mein rakh kar check kiya: dono taraf barabar hain."))
    numeric = [float(sp.N(s)) for s in sols if sp.N(s).is_real]
    return _result("equation", steps, numeric if len(numeric) != 1 else numeric[0],
                   ", ".join(sp.sstr(s) for s in sols) or "no solution",
                   "substitution check (SymPy)" if ok else "MISMATCH",
                   {"en": "Solving means finding the x values that make both sides equal.",
                    "ru": "Hal karne ka matlab woh x dhoondna hai jis se dono taraf barabar hon."})


def gradient_descent(expr_text: str, x0: float, lr: float, n_steps: int = 10) -> dict:
    f = parse_expression(expr_text)
    df = sp.diff(f, X)
    fn, dfn = sp.lambdify(X, f, "math"), sp.lambdify(X, df, "math")
    history = [{"step": 0, "x": float(x0), "f": float(fn(x0)), "grad": float(dfn(x0))}]
    steps = [_step(f"f(x) = {sp.sstr(f)}, f'(x) = {sp.sstr(df)}, start x0 = {fmt(x0)}, learning rate = {fmt(lr)}",
                   f"f(x) = {sp.sstr(f)}, f'(x) = {sp.sstr(df)}, shuru x0 = {fmt(x0)}, learning rate = {fmt(lr)}",
                   "x_{t+1} = x_t - \\alpha f'(x_t)")]
    x = float(x0)
    diverged = False
    for t in range(1, n_steps + 1):
        g = float(dfn(x))
        new_x = x - lr * g
        if t <= 5:
            steps.append(_step(f"Step {t}: x = {fmt(x)} - {fmt(lr)} x {fmt(g)} = {fmt(new_x)}",
                               f"Step {t}: x = {fmt(x)} - {fmt(lr)} x {fmt(g)} = {fmt(new_x)}"))
        x = new_x
        try:
            fx = float(fn(x))
        except OverflowError:
            diverged = True
            break
        history.append({"step": t, "x": x, "f": fx, "grad": float(dfn(x))})
        if abs(x) > 1e8 or not np.isfinite(fx):
            diverged = True
            break
    # growing loss over the run also means the steps overshoot the minimum
    if not diverged and len(history) > 3 and history[-1]["f"] > history[0]["f"] and \
            history[-1]["f"] > history[-2]["f"] > history[-3]["f"]:
        diverged = True
    if diverged:
        steps.append(_step("The values are exploding: the learning rate is too large (divergence).",
                           "Values phat rahi hain: learning rate bohat bara hai (divergence)."))
    else:
        steps.append(_step(f"After {len(history) - 1} steps: x = {fmt(x)}, f(x) = {fmt(history[-1]['f'])}, "
                           f"gradient = {fmt(history[-1]['grad'])}",
                           f"{len(history) - 1} steps ke baad: x = {fmt(x)}, f(x) = {fmt(history[-1]['f'])}, "
                           f"gradient = {fmt(history[-1]['grad'])}"))
    # verify the first update independently with SymPy
    first = float(sp.N(x0 - lr * df.subs(X, x0)))
    ok = len(history) < 2 or np.isclose(first, history[1]["x"])
    return _result("gradient_descent", steps, history[-1]["x"], fmt(history[-1]["x"]),
                   "first update re-checked with SymPy" if ok else "MISMATCH",
                   {"en": "Gradient descent repeatedly steps against the slope to reach a minimum.",
                    "ru": "Gradient descent baar baar slope ke khilaf qadam utha kar minimum tak pohanchta hai."},
                   history=history, expr=f, diverged=diverged)


# ---------------------------------------------------------------------------
# Typed questions
# ---------------------------------------------------------------------------
def detect_and_solve(text: str) -> dict:
    """Recognise simple typed questions and solve them."""
    q = (text or "").strip()
    low = q.lower()
    m = re.search(r"(?:derivative of|differentiate|d/dx)\s*(.+?)(?:\s+at\s+x\s*=\s*(-?[\d.]+))?\s*$", low)
    if m:
        return derivative(m.group(1), at=float(m.group(2)) if m.group(2) else None)
    m = re.search(r"(?:integral of|integrate)\s*(.+?)(?:\s+from\s+(-?[\d.]+)\s+to\s+(-?[\d.]+))?\s*$", low)
    if m:
        a, b = m.group(2), m.group(3)
        return integral(m.group(1), float(a) if a else None, float(b) if b else None)
    m = re.search(r"(?:minimum|maximum|minimi[sz]e|maximi[sz]e|optimi[sz]e|critical points?)\s*(?:of)?\s*(.+)$", low)
    if m:
        return optimize(m.group(1))
    m = re.search(r"(mean|average|variance|standard deviation|std)\s*(?:of)?\s*([-\d.,\s]+)$", low)
    if m:
        return statistics(parse_vector(m.group(2)))
    m = re.search(r"^(?:solve)?\s*(.*=.*)$", low)
    if m:
        return solve_equation(m.group(1))
    raise MathInputError("try e.g. 'derivative of x^3 + 2x', 'integral of x^2 from 0 to 3', "
                         "'minimum of x^2 - 6x + 10', 'solve 2x + 3 = 11', 'mean of 2, 4, 9'")


# ---------------------------------------------------------------------------
# Practice problems
# ---------------------------------------------------------------------------
PRACTICE_KINDS = ["matrix_multiply", "determinant", "vectors", "gaussian_elimination", "statistics", "bayes",
                  "derivative", "equation", "gradient_descent", "binomial"]


def practice_problem(kind: str, seed: Optional[int] = None) -> dict:
    rng = random.Random(seed)
    r = lambda a, b: rng.randint(a, b)  # noqa: E731
    if kind in ("matrix_multiply", "inverse", "eigen"):
        A = np.array([[r(-3, 5), r(-3, 5)], [r(-3, 5), r(-3, 5)]])
        B = np.array([[r(-3, 5), r(-3, 5)], [r(-3, 5), r(-3, 5)]])
        sol = matrix_multiply(A, B)
        return {"kind": "matrix_multiply", "q": {"en": f"A = {mat_str(A)}, B = {mat_str(B)}. What is entry (1,1) of AB?",
                                                 "ru": f"A = {mat_str(A)}, B = {mat_str(B)}. AB ka (1,1) element kya hai?"},
                "answer": float(sol["answer"][0, 0]), "tolerance": 1e-6, "solution": sol,
                "hint": {"en": "Row 1 of A dotted with column 1 of B.", "ru": "A ki row 1 ka B ke column 1 se dot product."}}
    if kind == "gaussian_elimination":
        while True:
            A = np.array([[r(-3, 4) for _ in range(3)] for _ in range(3)])
            if round(np.linalg.det(A)) != 0:
                break
        x = np.array([r(-3, 4) for _ in range(3)])
        aug = np.column_stack([A, A @ x])
        sol = gaussian_elimination(aug)
        eqs = "; ".join(" + ".join(f"{A[i, j]}x{j + 1}" for j in range(3)) + f" = {A[i] @ x}" for i in range(3))
        return {"kind": kind, "q": {"en": f"Solve by Gaussian elimination: {eqs}. What is x1?",
                                    "ru": f"Gaussian elimination se hal karein: {eqs}. x1 kya hai?"},
                "answer": float(x[0]), "tolerance": 1e-6, "solution": sol,
                "hint": {"en": "Eliminate x1 from rows 2 and 3 first.", "ru": "Pehle rows 2 aur 3 se x1 khatam karein."}}
    if kind == "determinant":
        A = np.array([[r(-4, 6), r(-4, 6)], [r(-4, 6), r(-4, 6)]])
        sol = determinant(A)
        return {"kind": kind, "q": {"en": f"Find det({mat_str(A)}).", "ru": f"det({mat_str(A)}) nikalein."},
                "answer": sol["answer"], "tolerance": 1e-6, "solution": sol,
                "hint": {"en": "ad - bc", "ru": "ad - bc"}}
    if kind == "vectors":
        u = np.array([r(-3, 4), r(-3, 4), r(0, 4)])
        v = np.array([r(-3, 4), r(1, 4), r(-2, 4)])
        sol = vectors(u, v)
        return {"kind": kind, "q": {"en": f"Dot product of u = {mat_str(u)[1:-1]} and v = {mat_str(v)[1:-1]}?",
                                    "ru": f"u = {mat_str(u)[1:-1]} aur v = {mat_str(v)[1:-1]} ka dot product?"},
                "answer": sol["answer"]["dot"], "tolerance": 1e-6, "solution": sol,
                "hint": {"en": "Multiply matching entries and add them.", "ru": "Mutabiq entries zarab kar ke jama karein."}}
    if kind == "statistics":
        data = np.array([r(1, 12) for _ in range(rng.choice([4, 5, 6]))])
        sol = statistics(data)
        return {"kind": kind, "q": {"en": f"Population variance of {', '.join(str(int(d)) for d in data)}? (2 decimals)",
                                    "ru": f"{', '.join(str(int(d)) for d in data)} ki population variance? (2 decimal)"},
                "answer": sol["answer"]["var_pop"], "tolerance": 0.01, "solution": sol,
                "hint": {"en": "Average of the squared deviations from the mean.", "ru": "Mean se squared farq ka average."}}
    if kind == "bayes":
        prior = rng.choice([0.01, 0.02, 0.05, 0.1, 0.2])
        lik = rng.choice([0.8, 0.9, 0.95, 0.99])
        fp = rng.choice([0.02, 0.05, 0.1, 0.2])
        sol = bayes(prior, lik, fp)
        return {"kind": kind, "q": {"en": f"P(H) = {prior}, P(E|H) = {lik}, P(E|not H) = {fp}. Find P(H|E) (3 decimals).",
                                    "ru": f"P(H) = {prior}, P(E|H) = {lik}, P(E|H nahi) = {fp}. P(H|E) nikalein (3 decimal)."},
                "answer": sol["answer"], "tolerance": 0.002, "solution": sol,
                "hint": {"en": "First compute P(E) with the total probability rule.", "ru": "Pehle total probability se P(E) nikalein."}}
    if kind == "derivative":
        a, b, c, at = r(1, 5), r(-6, 6), r(-5, 5), r(-3, 3)
        expr = f"{a}*x^3 + {b}*x^2 + {c}*x"
        sol = derivative(expr, at=at)
        return {"kind": kind, "q": {"en": f"f(x) = {a}x^3 + {b}x^2 + {c}x. Find f'({at}).",
                                    "ru": f"f(x) = {a}x^3 + {b}x^2 + {c}x. f'({at}) nikalein."},
                "answer": sol["answer"], "tolerance": 1e-6, "solution": sol,
                "hint": {"en": "Power rule on each term, then substitute.", "ru": "Har term par power rule, phir value rakhein."}}
    if kind == "equation":
        a, x_true = rng.choice([2, 3, 4, 5, -2, -3]), r(-6, 9)
        b = r(-10, 10)
        c = a * x_true + b
        sol = solve_equation(f"{a}*x + {b} = {c}")
        return {"kind": kind, "q": {"en": f"Solve {a}x + {b} = {c}.", "ru": f"{a}x + {b} = {c} hal karein."},
                "answer": float(x_true), "tolerance": 1e-6, "solution": sol,
                "hint": {"en": "Subtract the constant, then divide by the coefficient of x.",
                         "ru": "Constant minus karein, phir x ke coefficient se taqseem karein."}}
    if kind == "gradient_descent":
        x0, lr = r(2, 6), rng.choice([0.1, 0.2, 0.25])
        sol = gradient_descent("x^2", x0, lr, n_steps=1)
        return {"kind": kind, "q": {"en": f"f(x) = x^2, x0 = {x0}, learning rate {lr}. x after one step?",
                                    "ru": f"f(x) = x^2, x0 = {x0}, learning rate {lr}. Ek step ke baad x?"},
                "answer": sol["history"][1]["x"], "tolerance": 1e-6, "solution": sol,
                "hint": {"en": "x - lr x 2x.", "ru": "x - lr x 2x."}}
    if kind == "binomial":
        n = r(3, 7)
        k, p = r(0, n), rng.choice([0.5, 0.3, 0.2])
        sol = binomial(n, k, p)
        return {"kind": kind, "q": {"en": f"n = {n} trials, p = {p}. P(exactly {k} successes)? (4 decimals)",
                                    "ru": f"n = {n} koshishein, p = {p}. Bilkul {k} kamyabiyon ki probability? (4 decimal)"},
                "answer": sol["answer"], "tolerance": 1e-4, "solution": sol,
                "hint": {"en": "C(n, k) p^k (1-p)^(n-k).", "ru": "C(n, k) p^k (1-p)^(n-k)."}}
    raise MathInputError(f"no practice generator for {kind}")


def check_practice(problem: dict, text: str) -> dict:
    from .quiz import grade_numerical

    return grade_numerical({"answer": problem["answer"], "tolerance": problem["tolerance"]}, text)
