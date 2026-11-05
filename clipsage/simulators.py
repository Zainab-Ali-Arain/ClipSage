"""
Algorithm practice simulators for Operating Systems, Computer Architecture
and Compiler Construction. Every function computes the real algorithm and
returns its step-by-step trace (a "dry run") so students can check their
own hand calculations.
"""

import math
import re
import struct
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple


class SimError(ValueError):
    pass


# ===========================================================================
# Operating Systems
# ===========================================================================
def parse_processes(text: str) -> List[dict]:
    """Lines 'P1, 0, 5' or 'P1 0 5 2' -> name, arrival, burst, optional priority."""
    procs = []
    for line in (text or "").strip().splitlines():
        parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
        if not parts:
            continue
        if len(parts) < 3:
            raise SimError(f"need 'name arrival burst [priority]' in: {line!r}")
        try:
            procs.append({"name": parts[0], "arrival": int(parts[1]), "burst": int(parts[2]),
                          "priority": int(parts[3]) if len(parts) > 3 else 0})
        except ValueError as e:
            raise SimError(f"arrival, burst and priority must be integers in: {line!r}") from e
        if procs[-1]["burst"] <= 0 or procs[-1]["arrival"] < 0:
            raise SimError("burst must be > 0 and arrival >= 0")
    if not procs:
        raise SimError("enter at least one process")
    return procs


def schedule(procs: List[dict], algorithm: str, quantum: int = 2) -> Dict:
    """FCFS, SJF (non-preemptive), SRTF, Priority (non-preemptive, lower = higher) and RR."""
    algorithm = algorithm.upper()
    remaining = {p["name"]: p["burst"] for p in procs}
    info = {p["name"]: p for p in procs}
    order = sorted(procs, key=lambda p: (p["arrival"], p["name"]))
    time, gantt, first_run, completion, log = 0, [], {}, {}, []

    def run(name, start, end):
        if gantt and gantt[-1][0] == name and gantt[-1][2] == start:
            gantt[-1] = (name, gantt[-1][1], end)
        else:
            gantt.append((name, start, end))
        first_run.setdefault(name, start)

    if algorithm == "RR":
        queue, arrived, i = [], set(), 0
        while len(completion) < len(procs):
            while i < len(order) and order[i]["arrival"] <= time:
                queue.append(order[i]["name"]); arrived.add(order[i]["name"]); i += 1
            if not queue:
                nxt = order[i]["arrival"]
                gantt.append(("idle", time, nxt)); time = nxt
                continue
            name = queue.pop(0)
            run_for = min(quantum, remaining[name])
            log.append(f"t={time}: {name} runs {run_for} unit(s) (remaining {remaining[name]} -> {remaining[name] - run_for})")
            run(name, time, time + run_for)
            time += run_for
            remaining[name] -= run_for
            while i < len(order) and order[i]["arrival"] <= time:  # new arrivals join before the preempted one
                queue.append(order[i]["name"]); arrived.add(order[i]["name"]); i += 1
            if remaining[name] == 0:
                completion[name] = time
                log.append(f"t={time}: {name} finishes")
            else:
                queue.append(name)
    elif algorithm == "SRTF":
        while len(completion) < len(procs):
            ready = [p for p in procs if p["arrival"] <= time and remaining[p["name"]] > 0]
            if not ready:
                nxt = min(p["arrival"] for p in procs if remaining[p["name"]] > 0)
                gantt.append(("idle", time, nxt)); time = nxt
                continue
            p = min(ready, key=lambda p: (remaining[p["name"]], p["arrival"], p["name"]))
            run(p["name"], time, time + 1)
            remaining[p["name"]] -= 1
            time += 1
            if remaining[p["name"]] == 0:
                completion[p["name"]] = time
                log.append(f"t={time}: {p['name']} finishes")
    else:
        if algorithm not in ("FCFS", "SJF", "PRIORITY"):
            raise SimError("algorithm must be FCFS, SJF, SRTF, PRIORITY or RR")
        done = set()
        while len(done) < len(procs):
            ready = [p for p in procs if p["arrival"] <= time and p["name"] not in done]
            if not ready:
                nxt = min(p["arrival"] for p in procs if p["name"] not in done)
                gantt.append(("idle", time, nxt)); time = nxt
                continue
            key = {"FCFS": lambda p: (p["arrival"], p["name"]),
                   "SJF": lambda p: (p["burst"], p["arrival"], p["name"]),
                   "PRIORITY": lambda p: (p["priority"], p["arrival"], p["name"])}[algorithm]
            p = min(ready, key=key)
            log.append(f"t={time}: ready = {[r['name'] for r in ready]} -> choose {p['name']}")
            run(p["name"], time, time + p["burst"])
            time += p["burst"]
            completion[p["name"]] = time
            done.add(p["name"])

    rows = []
    for p in procs:
        ct = completion[p["name"]]
        tat = ct - p["arrival"]
        rows.append({"process": p["name"], "arrival": p["arrival"], "burst": p["burst"], "priority": p["priority"],
                     "completion": ct, "turnaround": tat, "waiting": tat - p["burst"],
                     "response": first_run[p["name"]] - p["arrival"]})
    n = len(rows)
    return {"algorithm": algorithm, "quantum": quantum if algorithm == "RR" else None, "gantt": gantt, "rows": rows,
            "avg_waiting": sum(r["waiting"] for r in rows) / n,
            "avg_turnaround": sum(r["turnaround"] for r in rows) / n,
            "avg_response": sum(r["response"] for r in rows) / n, "log": log,
            "context_switches": max(0, len([g for g in gantt if g[0] != "idle"]) - 1)}


def page_replacement(refs: Sequence[int], frames: int, algorithm: str) -> Dict:
    """FIFO, LRU or OPT. Returns the frame contents after every reference."""
    algorithm = algorithm.upper()
    if frames < 1:
        raise SimError("frames must be at least 1")
    memory: List[int] = []
    queue: List[int] = []           # FIFO order
    last_used: Dict[int, int] = {}  # LRU
    steps, faults = [], 0
    for t, page in enumerate(refs):
        evicted = None
        hit = page in memory
        if not hit:
            faults += 1
            if len(memory) < frames:
                memory.append(page)
            else:
                if algorithm == "FIFO":
                    victim = queue.pop(0)
                elif algorithm == "LRU":
                    victim = min(memory, key=lambda p: last_used[p])
                elif algorithm == "OPT":
                    future = list(refs[t + 1:])
                    victim = max(memory, key=lambda p: future.index(p) if p in future else math.inf)
                else:
                    raise SimError("algorithm must be FIFO, LRU or OPT")
                memory[memory.index(victim)] = page
                evicted = victim
            if algorithm == "FIFO":
                queue.append(page)
        last_used[page] = t
        steps.append({"ref": page, "frames": list(memory) + [None] * (frames - len(memory)),
                      "result": "hit" if hit else "fault", "evicted": evicted})
    return {"algorithm": algorithm, "frames": frames, "steps": steps, "faults": faults,
            "hits": len(refs) - faults, "fault_rate": faults / len(refs) if refs else 0.0}


def translate_addresses(page_size: int, page_table: Dict[int, int], addresses: Sequence[int],
                        address_bits: Optional[int] = None) -> Dict:
    if page_size <= 0 or page_size & (page_size - 1):
        raise SimError("page size must be a power of two")
    offset_bits = int(math.log2(page_size))
    rows = []
    for a in addresses:
        page, offset = divmod(a, page_size)
        frame = page_table.get(page)
        rows.append({"logical": a, "page": page, "offset": offset, "frame": frame,
                     "physical": None if frame is None else frame * page_size + offset,
                     "result": "page fault" if frame is None else "ok"})
    out = {"offset_bits": offset_bits, "rows": rows}
    if address_bits:
        out["page_bits"] = address_bits - offset_bits
        out["pages"] = 2 ** (address_bits - offset_bits)
    return out


def bankers(available: List[int], max_need: List[List[int]], allocation: List[List[int]],
            request: Optional[Tuple[int, List[int]]] = None) -> Dict:
    n, m = len(allocation), len(available)
    if len(max_need) != n or any(len(r) != m for r in max_need + allocation):
        raise SimError("Max and Allocation must have one row per process and one column per resource")
    need = [[max_need[i][j] - allocation[i][j] for j in range(m)] for i in range(n)]
    if any(v < 0 for row in need for v in row):
        raise SimError("Allocation cannot exceed Max")
    notes = []
    available = list(available)
    allocation = [list(r) for r in allocation]
    if request is not None:
        pid, req = request
        if any(req[j] > need[pid][j] for j in range(m)):
            return {"need": need, "safe": False, "sequence": [], "steps": [],
                    "notes": [f"P{pid} asks for more than its declared need - request rejected."]}
        if any(req[j] > available[j] for j in range(m)):
            return {"need": need, "safe": False, "sequence": [], "steps": [],
                    "notes": [f"Not enough resources available - P{pid} must wait."]}
        for j in range(m):
            available[j] -= req[j]; allocation[pid][j] += req[j]; need[pid][j] -= req[j]
        notes.append(f"Pretend to grant P{pid}'s request {req}; check if the new state is safe.")
    work, finish, sequence, steps = list(available), [False] * n, [], []
    progressed = True
    while progressed:
        progressed = False
        for i in range(n):
            if not finish[i] and all(need[i][j] <= work[j] for j in range(m)):
                steps.append({"process": f"P{i}", "need": need[i], "work_before": list(work),
                              "work_after": [work[j] + allocation[i][j] for j in range(m)]})
                work = [work[j] + allocation[i][j] for j in range(m)]
                finish[i] = progressed = True
                sequence.append(f"P{i}")
    safe = all(finish)
    if request is not None:
        notes.append("Safe: the request can be granted." if safe else "Unsafe: the request must wait.")
    return {"need": need, "safe": safe, "sequence": sequence, "steps": steps, "notes": notes}


# ===========================================================================
# Computer Architecture
# ===========================================================================
def integer_representation(value: int, bits: int = 8) -> Dict:
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    if not lo <= value <= hi:
        raise SimError(f"{value} does not fit in {bits}-bit two's complement ({lo} .. {hi})")
    mask = 2 ** bits - 1
    pattern = format(value & mask, f"0{bits}b")
    steps = []
    if value < 0:
        mag = format(-value, f"0{bits}b")
        inv = "".join("1" if c == "0" else "0" for c in mag)
        steps = [f"|{value}| = {-value} = {mag}", f"invert the bits: {inv}", f"add 1: {pattern}"]
    else:
        steps = [f"{value} in binary = {pattern} (positive: the same in two's complement)"]
    return {"value": value, "bits": bits, "binary": pattern, "grouped": " ".join(pattern[i:i + 4] for i in range(0, bits, 4)),
            "hex": f"0x{value & mask:0{bits // 4}X}", "unsigned_value": value & mask, "range": (lo, hi), "steps": steps,
            "sign_extended_16": format(value & 0xFFFF, "016b") if bits < 16 else None}


def decode_binary(pattern: str) -> Dict:
    pattern = pattern.replace(" ", "")
    if not pattern or set(pattern) - {"0", "1"}:
        raise SimError("enter a binary pattern of 0s and 1s")
    bits = len(pattern)
    unsigned = int(pattern, 2)
    signed = unsigned - 2 ** bits if pattern[0] == "1" else unsigned
    return {"pattern": pattern, "bits": bits, "unsigned": unsigned, "signed": signed, "hex": hex(unsigned)}


def ieee754_single(x: float) -> Dict:
    (bits,) = struct.unpack(">I", struct.pack(">f", x))
    s = format(bits, "032b")
    sign, exp, frac = s[0], s[1:9], s[9:]
    e = int(exp, 2)
    steps = []
    if 0 < e < 255:
        mantissa = 1 + int(frac, 2) / 2 ** 23
        steps = [f"sign = {sign} ({'negative' if sign == '1' else 'positive'})",
                 f"exponent bits {exp} = {e}; real exponent = {e} - 127 = {e - 127}",
                 f"fraction bits give mantissa 1.{frac.rstrip('0') or '0'} (binary) = {mantissa}",
                 f"value = {'-' if sign == '1' else ''}{mantissa} x 2^{e - 127} = {struct.unpack('>f', struct.pack('>I', bits))[0]}"]
    stored = struct.unpack(">f", struct.pack(">I", bits))[0]
    return {"value": x, "sign": sign, "exponent": exp, "fraction": frac, "hex": f"0x{bits:08X}",
            "biased_exponent": e, "stored_value": stored, "rounding_error": stored - x, "steps": steps}


def cache_simulate(address_bits: int, cache_bytes: int, block_bytes: int, ways: int, addresses: Sequence[int],
                   hit_time: float = 1.0, miss_penalty: float = 100.0) -> Dict:
    for name, v in [("cache size", cache_bytes), ("block size", block_bytes), ("ways", ways)]:
        if v <= 0 or v & (v - 1):
            raise SimError(f"{name} must be a power of two")
    if cache_bytes < block_bytes * ways:
        raise SimError("cache must hold at least one set")
    offset_bits = int(math.log2(block_bytes))
    n_sets = cache_bytes // (block_bytes * ways)
    index_bits = int(math.log2(n_sets))
    tag_bits = address_bits - index_bits - offset_bits
    if tag_bits < 0:
        raise SimError("address too short for this cache")
    sets: List[OrderedDict] = [OrderedDict() for _ in range(n_sets)]  # LRU order inside each set
    rows, hits = [], 0
    for a in addresses:
        if a < 0 or a >= 2 ** address_bits:
            raise SimError(f"address {a} does not fit in {address_bits} bits")
        offset = a & (block_bytes - 1)
        index = (a >> offset_bits) & (n_sets - 1)
        tag = a >> (offset_bits + index_bits)
        s = sets[index]
        evicted = None
        if tag in s:
            hits += 1
            s.move_to_end(tag)
            result = "hit"
        else:
            if len(s) == ways:
                evicted, _ = s.popitem(last=False)
            s[tag] = True
            result = "miss"
        rows.append({"address": a, "hex": hex(a), "tag": tag, "index": index, "offset": offset, "result": result,
                     "evicted_tag": evicted})
    miss_rate = 1 - hits / len(addresses) if addresses else 0.0
    return {"offset_bits": offset_bits, "index_bits": index_bits, "tag_bits": tag_bits, "sets": n_sets, "ways": ways,
            "rows": rows, "hits": hits, "misses": len(addresses) - hits, "hit_rate": 1 - miss_rate,
            "amat": hit_time + miss_rate * miss_penalty}


REG = r"\$?[A-Za-z]+\d*"


def parse_instructions(text: str) -> List[dict]:
    """'LW R1, 0(R2)', 'ADD R3, R1, R4', 'SW R3, 4(R2)' -> dest / sources."""
    out = []
    for line in (text or "").strip().splitlines():
        line = line.strip().split("#")[0].strip()
        if not line:
            continue
        op, _, rest = line.partition(" ")
        args = [a.strip() for a in rest.split(",") if a.strip()]
        op = op.upper()
        if op in ("LW", "LD", "LOAD"):
            m = re.match(rf"(-?\d+)\(({REG})\)", args[1]) if len(args) == 2 else None
            if not m:
                raise SimError(f"use 'LW Rd, offset(Rs)': {line!r}")
            out.append({"text": line, "op": op, "dest": args[0].upper(), "src": [m.group(2).upper()], "load": True})
        elif op in ("SW", "SD", "STORE"):
            m = re.match(rf"(-?\d+)\(({REG})\)", args[1]) if len(args) == 2 else None
            if not m:
                raise SimError(f"use 'SW Rs, offset(Rb)': {line!r}")
            out.append({"text": line, "op": op, "dest": None, "src": [args[0].upper(), m.group(2).upper()], "load": False})
        else:
            if len(args) < 2:
                raise SimError(f"use 'OP Rd, Rs, Rt': {line!r}")
            srcs = [a.upper() for a in args[1:] if not re.fullmatch(r"-?\d+", a)]
            out.append({"text": line, "op": op, "dest": args[0].upper(), "src": srcs, "load": False})
    if not out:
        raise SimError("enter at least one instruction")
    return out


def pipeline_simulate(instrs: List[dict], forwarding: bool = True) -> Dict:
    """Classic 5-stage pipeline (IF ID EX MEM WB) with RAW-hazard stalls."""
    stages = ["IF", "ID", "EX", "MEM", "WB"]
    timeline, issue = [], []
    prev_id = 0            # cycle in which the previous instruction was in ID
    rows = []
    for i, ins in enumerate(instrs):
        id_cycle = prev_id + 1 if i else 2
        reason = []
        for j in range(max(0, i - 3), i):     # producers that may still be in flight
            prod = instrs[j]
            if prod["dest"] and prod["dest"] in ins["src"]:
                p_id = issue[j]
                if forwarding:
                    # ALU result is forwarded from EX/MEM, a loaded value only from MEM/WB:
                    # earliest consumer EX = producer EX + 1 (ALU) or + 2 (load)
                    ready_ex = p_id + (3 if prod["load"] else 2)
                    need_id = ready_ex - 1
                else:
                    need_id = p_id + 3                              # read in ID after producer's WB (split cycle)
                if need_id > id_cycle:
                    reason.append(f"{prod['dest']} from I{j + 1}")
                    id_cycle = need_id
        stalls = id_cycle - (prev_id + 1 if i else 2)
        issue.append(id_cycle)
        prev_id = id_cycle
        cycles = {"IF": id_cycle - 1 - stalls}
        cycles.update({"ID": id_cycle, "EX": id_cycle + 1, "MEM": id_cycle + 2, "WB": id_cycle + 3})
        rows.append({"instr": f"I{i + 1}: {ins['text']}", "stalls": stalls, "reason": ", ".join(reason) or "-",
                     "cycles": cycles})
    total = rows[-1]["cycles"]["WB"]
    grid = []
    for r in rows:
        line = ["" for _ in range(total)]
        c = r["cycles"]
        for s in stages:
            line[c[s] - 1] = s
        for k in range(c["IF"] + 1, c["ID"]):   # stalled in IF/ID
            line[k - 1] = "stall"
        grid.append(line)
    n = len(instrs)
    stall_total = sum(r["stalls"] for r in rows)
    return {"rows": rows, "grid": grid, "total_cycles": total, "stalls": stall_total,
            "ideal_cycles": 5 + n - 1, "cpi": (total - 4) / n, "forwarding": forwarding,
            "speedup_vs_unpipelined": (5 * n) / total}


def cpu_performance(instruction_count: float, cpi: float, clock_hz: float) -> Dict:
    t = instruction_count * cpi / clock_hz
    return {"time_s": t, "steps": [f"T = IC x CPI / f = {instruction_count:g} x {cpi:g} / {clock_hz:g} = {t:g} s"]}


def weighted_cpi(mix: List[Tuple[str, float, float]]) -> Dict:
    total = sum(p for _, p, _ in mix)
    if abs(total - 1) > 1e-6:
        raise SimError(f"fractions must add up to 1 (they add up to {total:g})")
    cpi = sum(p * c for _, p, c in mix)
    return {"cpi": cpi, "steps": [" + ".join(f"{p:g} x {c:g}" for _, p, c in mix) + f" = {cpi:g}"]}


def amdahl(fraction: float, factor: float) -> Dict:
    if not 0 <= fraction <= 1 or factor <= 0:
        raise SimError("fraction must be in [0, 1] and factor > 0")
    s = 1 / ((1 - fraction) + fraction / factor)
    limit = 1 / (1 - fraction) if fraction < 1 else math.inf
    return {"speedup": s, "limit": limit,
            "steps": [f"S = 1 / ((1 - {fraction:g}) + {fraction:g} / {factor:g}) = 1 / {(1 - fraction) + fraction / factor:g} = {s:.4g}",
                      f"upper limit (infinite speed-up of that part) = 1 / {1 - fraction:g} = {limit:.4g}"]}


# ===========================================================================
# Compiler Construction
# ===========================================================================
KEYWORDS = {"if", "else", "while", "for", "return", "int", "float", "char", "void", "do", "break", "continue"}
TOKEN_SPEC = [
    ("COMMENT", r"//[^\n]*|/\*.*?\*/"), ("NUMBER", r"\d+\.\d+|\d+"), ("STRING", r"\"[^\"\n]*\""),
    ("IDENT", r"[A-Za-z_]\w*"), ("RELOP", r"<=|>=|==|!=|<|>"), ("LOGIC", r"&&|\|\|"), ("ASSIGN", r"="),
    ("ARITH", r"[+\-*/%]"), ("LPAREN", r"\("), ("RPAREN", r"\)"), ("LBRACE", r"\{"), ("RBRACE", r"\}"),
    ("SEMI", r";"), ("COMMA", r","), ("NEWLINE", r"\n"), ("SKIP", r"[ \t\r]+"), ("ERROR", r"."),
]
_LEX = re.compile("|".join(f"(?P<{n}>{p})" for n, p in TOKEN_SPEC), re.DOTALL)


def tokenize(source: str) -> Dict:
    tokens, symbols, errors = [], OrderedDict(), []
    line, line_start = 1, 0
    for m in _LEX.finditer(source):
        kind, lexeme = m.lastgroup, m.group()
        col = m.start() - line_start + 1
        if kind == "NEWLINE":
            line += 1; line_start = m.end(); continue
        if kind in ("SKIP", "COMMENT"):
            line += lexeme.count("\n"); continue
        if kind == "ERROR":
            errors.append(f"line {line}, col {col}: unexpected character {lexeme!r}"); continue
        if kind == "IDENT" and lexeme in KEYWORDS:
            kind = "KEYWORD"
        if kind == "IDENT" and lexeme not in symbols:
            symbols[lexeme] = {"name": lexeme, "first_line": line, "uses": 0}
        if kind == "IDENT":
            symbols[lexeme]["uses"] += 1
        tokens.append({"token": kind, "lexeme": lexeme, "line": line, "col": col})
    return {"tokens": tokens, "symbols": list(symbols.values()), "errors": errors}


EPS = "eps"


def parse_grammar(text: str) -> Tuple[str, "OrderedDict[str, List[List[str]]]"]:
    """
    One rule per line, alternatives with |, symbols separated by spaces, eps / ε for epsilon:
        E -> T E'
        E' -> + T E' | eps
    """
    g: "OrderedDict[str, List[List[str]]]" = OrderedDict()
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if "->" not in line:
            raise SimError(f"each rule needs '->': {line!r}")
        lhs, rhs = [x.strip() for x in line.split("->", 1)]
        if not lhs or " " in lhs:
            raise SimError(f"left side must be one non-terminal: {line!r}")
        for alt in rhs.split("|"):
            syms = [s for s in alt.split() if s]
            syms = [] if syms in ([EPS], ["ε"], ["epsilon"]) else syms
            g.setdefault(lhs, []).append(syms)
    if not g:
        raise SimError("enter at least one rule")
    return next(iter(g)), g


def first_follow(start: str, g: Dict[str, List[List[str]]]) -> Dict:
    nts = set(g)
    first = {A: set() for A in g}

    def first_of(seq: List[str]) -> set:
        out = set()
        for X in seq:
            fx = first[X] if X in nts else {X}
            out |= fx - {EPS}
            if EPS not in fx:
                return out
        out.add(EPS)
        return out

    changed = True
    while changed:
        changed = False
        for A, prods in g.items():
            for p in prods:
                new = first_of(p)
                if not new <= first[A]:
                    first[A] |= new; changed = True
    follow = {A: set() for A in g}
    follow[start].add("$")
    changed = True
    while changed:
        changed = False
        for A, prods in g.items():
            for p in prods:
                for i, B in enumerate(p):
                    if B not in nts:
                        continue
                    rest = first_of(p[i + 1:])
                    add = (rest - {EPS}) | (follow[A] if EPS in rest else set())
                    if not add <= follow[B]:
                        follow[B] |= add; changed = True
    table: Dict[Tuple[str, str], List[List[str]]] = {}
    for A, prods in g.items():
        for p in prods:
            f = first_of(p)
            for a in (f - {EPS}) | (follow[A] if EPS in f else set()):
                table.setdefault((A, a), [])
                if p not in table[(A, a)]:
                    table[(A, a)].append(p)
    conflicts = {k: v for k, v in table.items() if len(v) > 1}
    terminals = sorted({s for prods in g.values() for p in prods for s in p if s not in nts} | {"$"})
    return {"first": {A: sorted(s) for A, s in first.items()}, "follow": {A: sorted(s) for A, s in follow.items()},
            "table": table, "conflicts": conflicts, "is_ll1": not conflicts, "terminals": terminals,
            "first_of": first_of}


def ll1_parse(start: str, g: Dict[str, List[List[str]]], tokens: List[str], analysis: Optional[Dict] = None) -> Dict:
    analysis = analysis or first_follow(start, g)
    if not analysis["is_ll1"]:
        raise SimError("the grammar is not LL(1) (the table has conflicts)")
    table = analysis["table"]
    stack, inp, steps = ["$", start], list(tokens) + ["$"], []
    while True:
        top, a = stack[-1], inp[0]
        row = {"stack": " ".join(stack), "input": " ".join(inp)}
        if top == "$" and a == "$":
            steps.append({**row, "action": "accept"})
            return {"steps": steps, "accepted": True}
        if top not in g:  # terminal
            if top == a:
                stack.pop(); inp.pop(0)
                steps.append({**row, "action": f"match {a}"})
            else:
                steps.append({**row, "action": f"error: expected {top}, found {a}"})
                return {"steps": steps, "accepted": False}
        else:
            prods = table.get((top, a))
            if not prods:
                steps.append({**row, "action": f"error: no entry M[{top}, {a}]"})
                return {"steps": steps, "accepted": False}
            p = prods[0]
            stack.pop()
            stack.extend(reversed(p))
            steps.append({**row, "action": f"{top} -> {' '.join(p) or EPS}"})
        if len(steps) > 500:
            raise SimError("parse did not finish (check the grammar)")


def slr_parse(start: str, g: Dict[str, List[List[str]]], tokens: List[str]) -> Dict:
    """Build the SLR(1) table (LR(0) items + FOLLOW) and run the shift-reduce parser."""
    aug = start + "'"
    prods = [(aug, [start])] + [(A, p) for A, ps in g.items() for p in ps]
    nts = set(g) | {aug}
    ff = first_follow(start, g)
    follow = ff["follow"]

    def closure(items):
        items = set(items)
        changed = True
        while changed:
            changed = False
            for (pi, dot) in list(items):
                rhs = prods[pi][1]
                if dot < len(rhs) and rhs[dot] in nts:
                    for qi, (A, _) in enumerate(prods):
                        if A == rhs[dot] and (qi, 0) not in items:
                            items.add((qi, 0)); changed = True
        return frozenset(items)

    def goto(items, X):
        return closure({(pi, dot + 1) for (pi, dot) in items if dot < len(prods[pi][1]) and prods[pi][1][dot] == X})

    states = [closure({(0, 0)})]
    transitions = {}
    symbols = sorted({s for _, r in prods for s in r})
    i = 0
    while i < len(states):
        for X in symbols:
            nxt = goto(states[i], X)
            if nxt:
                if nxt not in states:
                    states.append(nxt)
                transitions[(i, X)] = states.index(nxt)
        i += 1
    action, conflicts = {}, []

    def set_action(key, val):
        if key in action and action[key] != val:
            conflicts.append((key, action[key], val))
        else:
            action[key] = val

    for si, items in enumerate(states):
        for (pi, dot) in items:
            A, rhs = prods[pi]
            if dot < len(rhs) and rhs[dot] not in nts:
                set_action((si, rhs[dot]), ("shift", transitions[(si, rhs[dot])]))
            elif dot == len(rhs):
                if A == aug:
                    set_action((si, "$"), ("accept",))
                else:
                    for a in follow[A]:
                        set_action((si, a), ("reduce", pi))
    stack, inp, steps = [0], list(tokens) + ["$"], []
    sym_stack: List[str] = []
    while True:
        s, a = stack[-1], inp[0]
        act = action.get((s, a))
        row = {"stack": " ".join(f"{x}" for x in [0] + [y for pair in zip(sym_stack, stack[1:]) for y in pair]),
               "input": " ".join(inp)}
        if act is None:
            steps.append({**row, "action": f"error: no action for state {s} on {a}"})
            return {"steps": steps, "accepted": False, "states": len(states), "conflicts": conflicts}
        if act[0] == "shift":
            stack.append(act[1]); sym_stack.append(inp.pop(0))
            steps.append({**row, "action": f"shift, go to state {act[1]}"})
        elif act[0] == "reduce":
            A, rhs = prods[act[1]]
            for _ in rhs:
                stack.pop(); sym_stack.pop()
            stack.append(transitions[(stack[-1], A)]); sym_stack.append(A)
            steps.append({**row, "action": f"reduce {A} -> {' '.join(rhs) or EPS}"})
        else:
            steps.append({**row, "action": "accept"})
            return {"steps": steps, "accepted": True, "states": len(states), "conflicts": conflicts}
        if len(steps) > 1000:
            raise SimError("parse did not finish")


def three_address_code(expr: str, target: str = "x") -> Dict:
    """TAC, quadruples and simple optimizations (constant folding, common sub-expressions)."""
    import ast

    try:
        tree = ast.parse(expr, mode="eval").body
    except SyntaxError as e:
        raise SimError(f"not a valid arithmetic expression: {expr!r}") from e
    ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
    code, counter = [], [0]

    def walk(node):
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = walk(node.left), walk(node.right)
            counter[0] += 1
            t = f"t{counter[0]}"
            code.append((ops[type(node.op)], left, right, t))
            return t
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = walk(node.operand)
            counter[0] += 1
            t = f"t{counter[0]}"
            code.append(("minus", v, "", t))
            return t
        if isinstance(node, (ast.Name, ast.Constant)):
            return ast.unparse(node)
        raise SimError("only + - * / and parentheses are supported")

    result = walk(tree)
    tac = [f"{t} = {a} {op} {b}" if op != "minus" else f"{t} = -{a}" for op, a, b, t in code] + [f"{target} = {result}"]
    quads = [{"op": op, "arg1": a, "arg2": b, "result": t} for op, a, b, t in code] + \
            [{"op": "=", "arg1": result, "arg2": "", "result": target}]
    # optimization: constant folding + common sub-expression elimination
    values, seen, optimized, notes = {}, {}, [], []

    def val(x):
        return values.get(x, x)

    def is_num(x):
        return re.fullmatch(r"-?\d+(\.\d+)?", str(x)) is not None

    for op, a, b, t in code:
        a, b = val(a), val(b)
        if op != "minus" and is_num(a) and is_num(b):
            r = eval(f"{a}{op}{b}")  # numbers only (checked above)
            r = int(r) if float(r).is_integer() else r
            values[t] = str(r)
            notes.append(f"constant folding: {a} {op} {b} = {r}")
            continue
        key = (op, a, b) if op not in "+*" else (op, *sorted([a, b]))
        if key in seen:
            values[t] = seen[key]
            notes.append(f"common sub-expression: {a} {op} {b} already in {seen[key]}")
            continue
        seen[key] = t
        optimized.append(f"{t} = {a} {op} {b}" if op != "minus" else f"{t} = -{a}")
    optimized.append(f"{target} = {val(result)}")
    return {"tac": tac, "quadruples": quads, "optimized": optimized, "notes": notes}
