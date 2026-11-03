"""
add_question.py
Small helper so you don't have to hand-edit eval_questions.json (which is
easy to break via copy-paste in a terminal). Appends one test question.

Usage:
    python add_question.py "what is a stack?" "video_name_here" 0 22.88

video_name_here must exactly match the indexed video's name (no extension) -
run `dir ..\\data\\*.json` to see the exact names available.
"""

import sys
import json
import os

QUESTIONS_FILE = "eval_questions.json"


def main():
    if len(sys.argv) != 5:
        print('Usage: python add_question.py "<question>" "<video_name>" <expected_start> <expected_end>')
        print('Example: python add_question.py "what is a stack?" "my_video" 0 22.88')
        sys.exit(1)

    question, video_name, start, end = sys.argv[1:5]

    try:
        start = float(start)
        end = float(end)
    except ValueError:
        print("expected_start and expected_end must be numbers (seconds).")
        sys.exit(1)

    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE, "r", encoding="utf-8-sig") as f:
            questions = json.load(f)
    else:
        questions = []

    questions.append({
        "question": question,
        "video_name": video_name,
        "expected_start": start,
        "expected_end": end,
    })

    with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)

    print(f"Added. {QUESTIONS_FILE} now has {len(questions)} question(s).")


if __name__ == "__main__":
    main()
