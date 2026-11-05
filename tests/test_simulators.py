"""Simulators checked against textbook examples with known answers."""

import numpy as np
import pytest

from clipsage import mathsolve as m
from clipsage import simulators as S


# ----------------------------------------------------------------- Operating Systems
@pytest.mark.parametrize("algo,gantt,avg_wait", [
    ("FCFS", [("P1", 0, 5), ("P2", 5, 8), ("P3", 8, 9)], 10 / 3),
    ("SJF", [("P1", 0, 5), ("P3", 5, 6), ("P2", 6, 9)], 8 / 3),
    ("SRTF", [("P1", 0, 1), ("P2", 1, 2), ("P3", 2, 3), ("P2", 3, 5), ("P1", 5, 9)], 5 / 3),
    ("RR", [("P1", 0, 2), ("P2", 2, 4), ("P3", 4, 5), ("P1", 5, 7), ("P2", 7, 8), ("P1", 8, 9)], 10 / 3),
])
def test_cpu_scheduling(algo, gantt, avg_wait):
    r = S.schedule(S.parse_processes("P1 0 5\nP2 1 3\nP3 2 1"), algo, quantum=2)
    assert r["gantt"] == gantt
    assert r["avg_waiting"] == pytest.approx(avg_wait)
    for row in r["rows"]:
        assert row["turnaround"] == row["completion"] - row["arrival"]
        assert row["waiting"] == row["turnaround"] - row["burst"]


def test_scheduling_idle_and_priority():
    r = S.schedule(S.parse_processes("A 0 2 3\nB 5 3 1\nC 5 1 2"), "PRIORITY")
    assert r["gantt"] == [("A", 0, 2), ("idle", 2, 5), ("B", 5, 8), ("C", 8, 9)]
    with pytest.raises(S.SimError):
        S.parse_processes("P1 zero 5")


def test_page_replacement_textbook():
    refs = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]   # Silberschatz
    assert [S.page_replacement(refs, 3, a)["faults"] for a in ("FIFO", "LRU", "OPT")] == [15, 12, 9]
    belady = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    assert S.page_replacement(belady, 3, "FIFO")["faults"] == 9
    assert S.page_replacement(belady, 4, "FIFO")["faults"] == 10      # Belady's anomaly
    assert S.page_replacement(belady, 4, "LRU")["faults"] <= S.page_replacement(belady, 3, "LRU")["faults"]


def test_bankers_textbook():
    args = ([3, 3, 2], [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]],
            [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]])
    r = S.bankers(*args)
    assert r["safe"] and r["sequence"] == ["P1", "P3", "P4", "P0", "P2"]
    assert S.bankers(*args, request=(1, [1, 0, 2]))["safe"]
    assert not S.bankers(*args, request=(4, [3, 3, 1]))["safe"]       # more than available


def test_paging_translation():
    rows = S.translate_addresses(1024, {0: 3, 1: 7, 2: 5}, [2500, 5000], address_bits=16)
    assert rows["rows"][0]["physical"] == 5572 and rows["rows"][1]["result"] == "page fault"
    assert rows["offset_bits"] == 10 and rows["page_bits"] == 6


# ----------------------------------------------------------------- Computer Architecture
def test_number_representation():
    assert S.integer_representation(-13)["binary"] == "11110011"
    assert S.decode_binary("11110011")["signed"] == -13
    assert S.ieee754_single(5.75)["hex"] == "0x40B80000"
    assert S.ieee754_single(5.75)["biased_exponent"] == 129
    with pytest.raises(S.SimError):
        S.integer_representation(200, 8)


def test_cache_direct_and_associative():
    c = S.cache_simulate(16, 64, 16, 1, [0, 4, 16, 64, 0, 128, 4])
    assert (c["tag_bits"], c["index_bits"], c["offset_bits"]) == (10, 2, 4)
    assert [r["result"] for r in c["rows"]] == ["miss", "hit", "miss", "miss", "miss", "miss", "miss"]
    two_way = S.cache_simulate(16, 64, 16, 2, [0, 64, 0, 128, 64])
    assert [r["result"] for r in two_way["rows"]] == ["miss", "miss", "hit", "miss", "miss"]  # LRU evicts 64
    assert S.cache_simulate(32, 64 * 1024, 16, 1, [0])["tag_bits"] == 16
    assert S.cache_simulate(16, 64, 16, 1, [0, 0, 0, 0], 1, 100)["amat"] == pytest.approx(1 + 0.25 * 100)


def test_pipeline_hazards():
    ins = S.parse_instructions("LW R1, 0(R2)\nADD R3, R1, R4\nSUB R5, R3, R6\nSW R5, 8(R2)")
    fwd = S.pipeline_simulate(ins, forwarding=True)
    assert [r["stalls"] for r in fwd["rows"]] == [0, 1, 0, 0] and fwd["total_cycles"] == 9
    no_fwd = S.pipeline_simulate(ins, forwarding=False)
    assert [r["stalls"] for r in no_fwd["rows"]] == [0, 2, 2, 2] and no_fwd["total_cycles"] == 14
    independent = S.pipeline_simulate(S.parse_instructions("ADD R1, R2, R3\nSUB R4, R5, R6"), False)
    assert independent["total_cycles"] == 6 and independent["stalls"] == 0


def test_performance_equations():
    assert S.cpu_performance(1e9, 2, 2e9)["time_s"] == pytest.approx(1)
    assert S.weighted_cpi([("ALU", .5, 1), ("LS", .3, 2), ("BR", .2, 3)])["cpi"] == pytest.approx(1.7)
    assert S.amdahl(.8, 4)["speedup"] == pytest.approx(2.5) and S.amdahl(.8, 4)["limit"] == pytest.approx(5)
    with pytest.raises(S.SimError):
        S.weighted_cpi([("A", .5, 1)])


# ----------------------------------------------------------------- Compiler Construction
def test_lexer():
    r = S.tokenize("if (count <= 10) total = total + 1; // comment")
    assert [t["token"] for t in r["tokens"]][:4] == ["KEYWORD", "LPAREN", "IDENT", "RELOP"]
    assert len(r["tokens"]) == 12 and [s["name"] for s in r["symbols"]] == ["count", "total"]
    assert S.tokenize("x = 1 $")["errors"]


GRAMMAR = "E -> T E'\nE' -> + T E' | eps\nT -> F T'\nT' -> * F T' | eps\nF -> ( E ) | id"


def test_first_follow_ll1():
    start, g = S.parse_grammar(GRAMMAR)
    ff = S.first_follow(start, g)
    assert ff["first"]["E"] == ["(", "id"] and ff["first"]["E'"] == ["+", "eps"]
    assert ff["follow"]["E"] == ["$", ")"] and ff["follow"]["F"] == ["$", ")", "*", "+"]
    assert ff["is_ll1"]
    assert S.ll1_parse(start, g, "id + id * id".split())["accepted"]
    assert not S.ll1_parse(start, g, "id + * id".split())["accepted"]
    left_rec = S.parse_grammar("E -> E + T | T\nT -> id")
    assert not S.first_follow(*left_rec)["is_ll1"]


def test_slr_shift_reduce():
    start, g = S.parse_grammar("E -> E + T | T\nT -> T * F | F\nF -> ( E ) | id")
    r = S.slr_parse(start, g, "id * id + id".split())
    assert r["accepted"] and r["states"] == 12 and r["conflicts"] == []
    assert not S.slr_parse(start, g, "id + + id".split())["accepted"]


def test_three_address_code():
    t = S.three_address_code("b * c + b * c - 2 * 3", "a")
    assert t["tac"][0] == "t1 = b * c" and len(t["tac"]) == 6
    assert t["optimized"] == ["t1 = b * c", "t3 = t1 + t1", "t5 = t3 - 6", "a = t5"]
    with pytest.raises(S.SimError):
        S.three_address_code("import os")


# ----------------------------------------------------------------- Linear Algebra
@pytest.mark.parametrize("aug,kind,rank", [
    ("1 1 3; 2 -1 0", "unique", 2),
    ("2 1 -1 8; -3 -1 2 -11; -2 1 2 -3", "unique", 3),
    ("1 2 3 4; 2 4 6 8; 1 0 1 2", "infinite", 2),
    ("1 1 2; 2 2 5", "none", 1),
])
def test_gaussian_elimination(aug, kind, rank):
    r = m.gaussian_elimination(m.parse_matrix(aug))
    assert r["solution_kind"] == kind and r["rank"] == rank and "MISMATCH" not in r["verified"]


def test_gaussian_textbook_solution():
    r = m.gaussian_elimination(m.parse_matrix("2 1 -1 8; -3 -1 2 -11; -2 1 2 -3"))
    assert np.allclose(r["answer"], [2, 3, -1])
