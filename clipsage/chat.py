"""
Subject chat: answers a student's message inside one subject, and builds a
practice quiz from the topics discussed in that chat.

    reply = respond("paging kya hai", "Operating Systems")
    -> {"kind": "topic", "topic_id": "paging_virtual_memory", "lang": "ru", ...}

Reply kinds
  greeting       hello / salam / thanks - a short friendly answer
  topic          a written lesson of THIS subject matches the message
  other_subject  the message matches a lesson of a DIFFERENT subject
  open           no written lesson matches; the UI answers from the lecture
                 transcripts if they cover it, otherwise says it is not covered

Nothing here writes to disk.
"""

import random
from typing import Dict, Iterable, List, Optional

from . import practice_gen as G
from .knowledge import load_topics
from .langdetect import tokens
from .search import analyze_query

GREETINGS = {"hi", "hello", "hey", "salam", "assalam", "assalamualaikum", "aoa", "slam", "thanks", "thank", "thankyou",
             "shukriya", "ok", "okay", "acha", "theek", "good", "morning", "evening", "you", "u", "alaikum", "o", "bye"}
URDU_GREETINGS = {"salam", "assalam", "assalamualaikum", "aoa", "slam", "shukriya", "acha", "theek", "alaikum"}

# numerical problem generators that belong to each lesson
TOPIC_KINDS: Dict[str, List[str]] = {
    "cpu_scheduling": ["os_fcfs", "os_sjf", "os_rr"],
    "page_replacement": ["os_page_faults"],
    "paging_virtual_memory": ["os_physical_address"],
    "vectors": ["vectors"],
    "matrices": ["matrix_multiply", "determinant"],
    "systems_of_equations": ["gaussian_elimination"],
    "lexical_analysis": ["cc_token_count"],
    "intermediate_code": ["cc_tac_count"],
    "css_layout": ["web_box_width", "web_box_space"],
    "number_representation": ["ca_twos_complement"],
    "cache_memory": ["ca_cache_tag", "ca_amat"],
    "cpu_performance": ["ca_amdahl"],
    "pipelining": ["ca_pipeline_cycles"],
    "gradient_descent": ["gradient_descent"],
    "probability_bayes": ["bayes", "binomial"],
    "naive_bayes": ["bayes"],
    "derivatives": ["derivative"],
    "statistics_basics": ["statistics"],
}


def is_greeting(text: str) -> bool:
    words = tokens(text)
    return bool(words) and len(words) <= 4 and all(w in GREETINGS for w in words)


def respond(query: str, subject: str) -> Dict:
    """Decide how to answer one chat message inside `subject`."""
    query = (query or "").strip()
    a = analyze_query(query, subject=subject)
    reply = {"query": query, "lang": a["lang"], "subject": subject, "focus": a["focus"] or query, "topic_id": None,
             "other_subject": None, "suggestions": [], "kind": "open"}
    if is_greeting(query):
        reply["kind"] = "greeting"
        if set(tokens(query)) & URDU_GREETINGS:
            reply["lang"] = "ru"
        return reply
    if a["topic"]:
        reply.update(kind="topic", topic_id=a["topic"]["id"])
    else:
        anywhere = analyze_query(query)
        if anywhere["topic"] and anywhere["topic"]["subject"] != subject:
            reply.update(kind="other_subject", other_subject=anywhere["topic"]["subject"],
                         topic_id=anywhere["topic"]["id"])
    reply["suggestions"] = [m["topic"]["id"] for m in a["matches"] if m["topic"]["id"] != reply["topic_id"]][:3]
    return reply


def discussed_topics(messages: Iterable[Dict], subject: str) -> List[str]:
    """Topic ids of this subject that came up in the chat, oldest first, without repeats."""
    topics = load_topics()
    out: List[str] = []
    for m in messages:
        tid = (m.get("reply") or {}).get("topic_id")
        if tid and tid not in out and topics.get(tid, {}).get("subject") == subject:
            out.append(tid)
    return out


def problem_question(kind: str, seed: int) -> Dict:
    """A freshly generated numerical problem in the quiz-question format (answer computed by the simulators)."""
    p = G.generate(kind, seed)
    steps = "\n".join(p["steps"])
    return {"id": f"gen_{kind}_{seed}", "type": "numerical", "kind": "problem", "difficulty": "medium",
            "q": p["q"], "answer": float(p["answer"]), "tolerance": p["tolerance"], "hints": [p["hint"]],
            "explanation": {"en": steps, "ru": steps}, "generated": True}


def practice_quiz(topic_ids: List[str], seed: int, n: int = 10, levels: Optional[Iterable[str]] = None,
                  numerical: bool = True) -> List[Dict]:
    """Questions about the given topics, spread evenly over them, plus generated numerical problems."""
    topics = load_topics()
    levels = set(levels or ("easy", "medium", "hard"))
    rng = random.Random(seed)
    pools = []
    for tid in topic_ids:
        if tid not in topics:
            continue
        pool = [q for q in topics[tid].get("questions", []) if q.get("difficulty", "medium") in levels]
        if numerical:
            pool += [problem_question(k, rng.randint(0, 10 ** 6)) for k in TOPIC_KINDS.get(tid, [])[:1]]
        rng.shuffle(pool)
        if pool:
            pools.append(pool)
    out: List[Dict] = []
    while pools and len(out) < n:                     # round robin so every topic is represented
        for pool in list(pools):
            if len(out) >= n:
                break
            out.append(pool.pop())
            if not pool:
                pools.remove(pool)
    return out
