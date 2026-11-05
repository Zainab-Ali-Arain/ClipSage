"""
Numerical practice problems for every subject.

Each generator makes a fresh random problem and computes its answer with the
real algorithm (simulators.py / mathsolve.py), so answers are never typed in
by hand. A problem is a dict:
    kind, title, q {en, ru}, answer (float), tolerance, hint {en, ru}, steps [str]
"""

import random
from typing import Dict, List, Optional

from . import mathsolve as M
from . import simulators as S

SUBJECT_KINDS: Dict[str, List[str]] = {
    "Operating Systems": ["os_fcfs", "os_sjf", "os_rr", "os_page_faults", "os_physical_address"],
    "Machine Learning": ["statistics", "bayes", "derivative", "gradient_descent", "binomial"],
    "Linear Algebra": ["matrix_multiply", "determinant", "gaussian_elimination", "vectors"],
    "Compiler Construction": ["cc_token_count", "cc_tac_count"],
    "Web Engineering": ["web_box_width", "web_box_space"],
    "Computer Architecture": ["ca_twos_complement", "ca_cache_tag", "ca_amat", "ca_amdahl", "ca_pipeline_cycles"],
}

TITLES = {
    "os_fcfs": "FCFS average waiting time", "os_sjf": "SJF average waiting time", "os_rr": "Round Robin average waiting time",
    "os_page_faults": "Page faults", "os_physical_address": "Logical to physical address",
    "cc_token_count": "Count the tokens", "cc_tac_count": "Three-address code size",
    "web_box_width": "CSS box model width", "web_box_space": "CSS box with margins",
    "ca_twos_complement": "Two's complement value", "ca_cache_tag": "Cache tag bits", "ca_amat": "Average memory access time",
    "ca_amdahl": "Amdahl's law", "ca_pipeline_cycles": "Pipeline cycles",
}


def title(kind: str) -> str:
    return TITLES.get(kind) or M.KINDS.get(kind, {}).get("en", kind)


def _procs(rng, n):
    t, out = 0, []
    for i in range(n):
        out.append({"name": f"P{i + 1}", "arrival": t, "burst": rng.randint(1, 7), "priority": 0})
        t += rng.randint(0, 3)
    return out


def generate(kind: str, seed: Optional[int] = None) -> Dict:
    rng = random.Random(seed)
    if kind in M.KINDS or kind in M.PRACTICE_KINDS:
        p = M.practice_problem(kind, seed)
        steps = [s["en"] for s in p["solution"]["steps"]]
        return {"kind": kind, "title": title(kind), "q": p["q"], "answer": float(p["answer"]),
                "tolerance": p["tolerance"], "hint": p["hint"], "steps": steps}

    if kind in ("os_fcfs", "os_sjf", "os_rr"):
        procs = _procs(rng, rng.randint(3, 4))
        algo = {"os_fcfs": "FCFS", "os_sjf": "SJF", "os_rr": "RR"}[kind]
        r = S.schedule(procs, algo, quantum=2)
        table = "; ".join(f"{p['name']} (arrival {p['arrival']}, burst {p['burst']})" for p in procs)
        extra = " with time quantum 2" if algo == "RR" else ""
        steps = ["Gantt chart: " + " | ".join(f"{n} {a}-{b}" for n, a, b in r["gantt"])]
        steps += [f"{row['process']}: TAT = {row['completion']} - {row['arrival']} = {row['turnaround']}, "
                  f"WT = {row['turnaround']} - {row['burst']} = {row['waiting']}" for row in r["rows"]]
        steps.append(f"average waiting time = {r['avg_waiting']:.2f}")
        return {"kind": kind, "title": title(kind),
                "q": {"en": f"{algo}{extra}: {table}. What is the average waiting time? (2 decimals)",
                      "ru": f"{algo}{extra}: {table}. Average waiting time kya hai? (2 decimal)"},
                "answer": round(r["avg_waiting"], 2), "tolerance": 0.011,
                "hint": {"en": "Draw the Gantt chart, then WT = (completion - arrival) - burst.",
                         "ru": "Gantt chart banayein, phir WT = (completion - arrival) - burst."},
                "steps": steps}

    if kind == "os_page_faults":
        refs = [rng.randint(0, 5) for _ in range(rng.randint(10, 12))]
        algo, frames = rng.choice(["FIFO", "LRU"]), 3
        r = S.page_replacement(refs, frames, algo)
        steps = [f"ref {s['ref']}: {s['result']}, frames {['-' if f is None else f for f in s['frames']]}"
                 + (f", evicted {s['evicted']}" if s["evicted"] is not None else "") for s in r["steps"]]
        steps.append(f"total page faults = {r['faults']}")
        return {"kind": kind, "title": title(kind),
                "q": {"en": f"{algo} with {frames} frames, reference string {', '.join(map(str, refs))}. How many page faults?",
                      "ru": f"{frames} frames ke saath {algo}, reference string {', '.join(map(str, refs))}. Kitne page faults?"},
                "answer": r["faults"], "tolerance": 0,
                "hint": {"en": "The first loads into empty frames are faults too.", "ru": "Khali frames mein pehli dafa load bhi faults hain."},
                "steps": steps}

    if kind == "os_physical_address":
        size = rng.choice([256, 512, 1024, 2048, 4096])
        page, frame, offset = rng.randint(0, 7), rng.randint(0, 15), rng.randint(0, size - 1)
        logical = page * size + offset
        r = S.translate_addresses(size, {page: frame}, [logical])["rows"][0]
        return {"kind": kind, "title": title(kind),
                "q": {"en": f"Page size {size} bytes. Logical address {logical}; page {page} is in frame {frame}. Physical address?",
                      "ru": f"Page size {size} bytes. Logical address {logical}; page {page} frame {frame} mein hai. Physical address?"},
                "answer": r["physical"], "tolerance": 0,
                "hint": {"en": "Split into page and offset, then frame x page size + offset.",
                         "ru": "Page aur offset mein todein, phir frame x page size + offset."},
                "steps": [f"page = {logical} // {size} = {page}, offset = {logical} mod {size} = {offset}",
                          f"physical = {frame} x {size} + {offset} = {r['physical']}"]}

    if kind == "cc_token_count":
        var, other, num = rng.choice(["x", "total", "count", "sum"]), rng.choice(["y", "rate", "i", "n"]), rng.randint(1, 99)
        stmt = rng.choice([f"{var} = {other} + {num} ;", f"if ( {var} >= {num} ) {other} = {other} - 1 ;",
                           f"while ( {var} < {num} ) {var} = {var} * 2 ;", f"return {var} * ( {other} + {num} ) ;"])
        toks = S.tokenize(stmt)["tokens"]
        return {"kind": kind, "title": title(kind),
                "q": {"en": f"How many tokens are in: {stmt}", "ru": f"Is mein kitne tokens hain: {stmt}"},
                "answer": len(toks), "tolerance": 0,
                "hint": {"en": "Keywords, identifiers, numbers, operators and punctuation each count once.",
                         "ru": "Keywords, identifiers, numbers, operators aur punctuation har ek, ek token hai."},
                "steps": [f"{t['token']}({t['lexeme']})" for t in toks] + [f"total = {len(toks)}"]}

    if kind == "cc_tac_count":
        names = ["a", "b", "c", "d"]
        ops = ["+", "-", "*"]
        k = rng.randint(2, 4)
        expr = names[0]
        for i in range(1, k + 1):
            expr = f"{expr} {rng.choice(ops)} {names[i % 4]}"
        r = S.three_address_code(expr, "x")
        n = len(r["tac"]) - 1
        return {"kind": kind, "title": title(kind),
                "q": {"en": f"How many three-address instructions (excluding the final copy x = t) does x = {expr} need?",
                      "ru": f"x = {expr} ke liye kitni three-address instructions chahiyein (aakhri x = t ke ilawa)?"},
                "answer": n, "tolerance": 0,
                "hint": {"en": "One instruction per operator.", "ru": "Har operator ke liye ek instruction."},
                "steps": r["tac"]}

    if kind in ("web_box_width", "web_box_space"):
        w, p, b, m = rng.choice([100, 150, 200, 240, 300]), rng.randint(2, 20), rng.randint(1, 5), rng.randint(0, 20)
        total = w + 2 * p + 2 * b + (2 * m if kind == "web_box_space" else 0)
        what = "horizontal space including margins" if kind == "web_box_space" else "rendered width (without margin)"
        return {"kind": kind, "title": title(kind),
                "q": {"en": f"box-sizing: content-box; width {w}px; padding {p}px; border {b}px; margin {m}px. What is the {what} in px?",
                      "ru": f"box-sizing: content-box; width {w}px; padding {p}px; border {b}px; margin {m}px. {what} kitne px?"},
                "answer": total, "tolerance": 0,
                "hint": {"en": "Add padding and border on both sides.", "ru": "Dono taraf padding aur border jama karein."},
                "steps": [f"{w} + 2x{p} + 2x{b}" + (f" + 2x{m}" if kind == "web_box_space" else "") + f" = {total}"]}

    if kind == "ca_twos_complement":
        pattern = "1" + "".join(rng.choice("01") for _ in range(7))
        d = S.decode_binary(pattern)
        return {"kind": kind, "title": title(kind),
                "q": {"en": f"What decimal value does the 8-bit two's complement pattern {pattern} represent?",
                      "ru": f"8-bit two's complement pattern {pattern} kis decimal value ko dikhata hai?"},
                "answer": d["signed"], "tolerance": 0,
                "hint": {"en": "The first bit is 1, so the value is negative: unsigned value - 256.",
                         "ru": "Pehla bit 1 hai, is liye value manfi hai: unsigned value - 256."},
                "steps": [f"unsigned = {d['unsigned']}", f"signed = {d['unsigned']} - 256 = {d['signed']}"]}

    if kind == "ca_cache_tag":
        bits, size_kb, block, ways = rng.choice([32, 32, 16]), rng.choice([4, 8, 16, 32, 64]), rng.choice([16, 32, 64]), rng.choice([1, 2, 4])
        if bits == 16:
            size_kb = rng.choice([1, 2, 4])
        r = S.cache_simulate(bits, size_kb * 1024, block, ways, [0])
        org = "direct-mapped" if ways == 1 else f"{ways}-way set-associative"
        return {"kind": kind, "title": title(kind),
                "q": {"en": f"{bits}-bit addresses, {size_kb} KB {org} cache, {block}-byte blocks. How many tag bits?",
                      "ru": f"{bits}-bit addresses, {size_kb} KB {org} cache, {block}-byte blocks. Kitne tag bits?"},
                "answer": r["tag_bits"], "tolerance": 0,
                "hint": {"en": "offset = log2(block), index = log2(sets), tag = rest.", "ru": "offset = log2(block), index = log2(sets), tag = baqi."},
                "steps": [f"offset = log2({block}) = {r['offset_bits']}",
                          f"sets = {size_kb * 1024} / ({block} x {ways}) = {r['sets']}, index = {r['index_bits']}",
                          f"tag = {bits} - {r['index_bits']} - {r['offset_bits']} = {r['tag_bits']}"]}

    if kind == "ca_amat":
        h, mr, mp = rng.choice([1, 2]), rng.choice([0.02, 0.05, 0.1]), rng.choice([50, 100, 200])
        amat = h + mr * mp
        return {"kind": kind, "title": title(kind),
                "q": {"en": f"Hit time {h} ns, miss rate {mr:.0%}, miss penalty {mp} ns. AMAT in ns?",
                      "ru": f"Hit time {h} ns, miss rate {mr:.0%}, miss penalty {mp} ns. AMAT kitne ns?"},
                "answer": amat, "tolerance": 1e-6,
                "hint": {"en": "AMAT = hit time + miss rate x miss penalty.", "ru": "AMAT = hit time + miss rate x miss penalty."},
                "steps": [f"{h} + {mr} x {mp} = {amat:g}"]}

    if kind == "ca_amdahl":
        f, s = rng.choice([0.5, 0.6, 0.75, 0.8, 0.9]), rng.choice([2, 4, 5, 10])
        r = S.amdahl(f, s)
        return {"kind": kind, "title": title(kind),
                "q": {"en": f"{f:.0%} of the execution time is sped up {s} times. Overall speedup? (2 decimals)",
                      "ru": f"Chalne ke waqt ka {f:.0%} hissa {s} guna tez kiya gaya. Kul speedup? (2 decimal)"},
                "answer": round(r["speedup"], 2), "tolerance": 0.011,
                "hint": {"en": "1 / ((1 - f) + f / s).", "ru": "1 / ((1 - f) + f / s)."}, "steps": r["steps"]}

    if kind == "ca_pipeline_cycles":
        k, n = rng.choice([4, 5, 6]), rng.choice([10, 20, 50, 100])
        return {"kind": kind, "title": title(kind),
                "q": {"en": f"An ideal {k}-stage pipeline executes {n} instructions. How many cycles?",
                      "ru": f"Behtareen {k}-stage pipeline {n} instructions chalati hai. Kitne cycles?"},
                "answer": k + n - 1, "tolerance": 0,
                "hint": {"en": "k + n - 1.", "ru": "k + n - 1."}, "steps": [f"{k} + {n} - 1 = {k + n - 1}"]}

    raise ValueError(f"unknown practice kind {kind}")


def check(problem: Dict, text: str) -> Dict:
    from .quiz import grade_numerical

    return grade_numerical({"answer": problem["answer"], "tolerance": problem["tolerance"]}, text)
