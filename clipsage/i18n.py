"""
Bilingual support: English, Roman Urdu (Urdu written in English letters) or both.

Bilingual text is stored as {"en": "...", "ru": "..."}.  `pick()` returns
the text for the chosen language; `t()` looks up a UI string by key.

No machine translation is used anywhere: every Roman Urdu string in ClipSage
was written by hand (here or in the knowledge/ topic files). Lecture
transcripts stay in their original English and are labelled as such.
"""

from typing import Dict, Union

LANGUAGES = {"en": "English", "ru": "Roman Urdu", "both": "Both / Dono"}
DEFAULT_LANGUAGE = "both"

Text = Union[str, Dict[str, str]]


def pick(text: Text, lang: str = DEFAULT_LANGUAGE, sep: str = "\n\n") -> str:
    """Text in the chosen language. With 'both', English first, then Roman Urdu."""
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    en, ru = text.get("en", ""), text.get("ru", "")
    if lang == "en":
        return en or ru
    if lang == "ru":
        return ru or en
    if en and ru and en != ru:
        return f"{en}{sep}{ru}"
    return en or ru


def short(text: Text, lang: str = DEFAULT_LANGUAGE) -> str:
    """Like pick(), but on one line ('English / Roman Urdu') for labels and buttons."""
    return pick(text, lang, sep=" / ")


def t(key: str, lang: str = DEFAULT_LANGUAGE, **fmt) -> str:
    """UI string by key. Unknown keys return the key itself (easy to spot)."""
    entry = UI.get(key)
    if entry is None:
        return key
    out = pick(entry, lang)
    return out.format(**fmt) if fmt else out


def tl(key: str, lang: str = DEFAULT_LANGUAGE, **fmt) -> str:
    """One-line version of t() for labels."""
    entry = UI.get(key)
    if entry is None:
        return key
    out = short(entry, lang)
    return out.format(**fmt) if fmt else out


UI: Dict[str, Dict[str, str]] = {
    # --- navigation -------------------------------------------------------
    "nav_dashboard": {"en": "Dashboard", "ru": "Dashboard"},
    "nav_explore": {"en": "Explore Topics", "ru": "Topics Dekhein"},
    "nav_videos": {"en": "Video Lectures", "ru": "Video Lectures"},
    "nav_learn": {"en": "Topic Learning", "ru": "Topic Seekhein"},
    "nav_practice": {"en": "Practice & Quizzes", "ru": "Practice aur Quiz"},
    "nav_math": {"en": "Mathematics Playground", "ru": "Math Playground"},
    "nav_ml": {"en": "ML Algorithm Playground", "ru": "ML Algorithm Playground"},
    "nav_saved": {"en": "Saved Topics", "ru": "Save Kiye Topics"},
    "nav_progress": {"en": "Learning Progress", "ru": "Seekhne ki Progress"},
    "nav_settings": {"en": "Settings", "ru": "Settings"},

    # --- general ----------------------------------------------------------
    "tagline": {"en": "Study smarter with real lecture moments, bilingual lessons and hands-on practice.",
                "ru": "Asli lecture moments, do zubanon mein lessons aur practice ke saath behtar parhain."},
    "search_placeholder": {"en": "Search a topic, e.g. overfitting, k-means, deadlock",
                           "ru": "Koi topic likhein, maslan overfitting, k-means, deadlock"},
    "search": {"en": "Search", "ru": "Search karein"},
    "language": {"en": "Language", "ru": "Zubaan"},
    "no_index": {"en": "The lecture index has not been built yet, so video references are unavailable. "
                       "Run run_pipeline.bat (see README). Lessons, quizzes and playgrounds still work.",
                 "ru": "Lecture index abhi nahi bana, is liye video references dastiyab nahi. "
                       "run_pipeline.bat chalayein (README dekhein). Lessons, quiz aur playgrounds phir bhi chalenge."},
    "not_covered": {"en": "“{q}” is not covered by the collected lectures yet.",
                    "ru": "“{q}” abhi jama ki gayi lectures mein maujood nahi."},
    "try_related": {"en": "Related topics you can try:", "ru": "Yeh milte julte topics try karein:"},
    "no_notes": {"en": "ClipSage has no written lesson for this topic yet. Everything below comes directly "
                       "from the lecture transcripts.",
                 "ru": "Is topic ka likha hua lesson abhi ClipSage mein nahi. Neeche sab kuch seedha lecture "
                       "transcripts se hai."},
    "save_topic": {"en": "Save topic", "ru": "Topic save karein"},
    "saved": {"en": "Saved", "ru": "Save ho gaya"},
    "remove": {"en": "Remove", "ru": "Hatayein"},
    "open": {"en": "Open", "ru": "Kholein"},

    # --- source labels ----------------------------------------------------
    "src_lecture": {"en": "From lectures (retrieved transcript)", "ru": "Lectures se (asli transcript)"},
    "src_notes": {"en": "ClipSage notes (written explanation)", "ru": "ClipSage notes (likhi hui wazahat)"},
    "src_example": {"en": "Additional example", "ru": "Izafi misaal"},
    "src_computed": {"en": "Computed result", "ru": "Hisaab se nikla natija"},
    "src_sample": {"en": "Sample data (synthetic, for practice)", "ru": "Sample data (banaya hua, practice ke liye)"},
    "transcript_english": {"en": "Transcript excerpts are shown in the lecture's original language (English).",
                           "ru": "Transcript ke hisse lecture ki asal zubaan (English) mein dikhaye gaye hain."},

    # --- lesson sections --------------------------------------------------
    "sec_intro": {"en": "Introduction", "ru": "Taaruf"},
    "sec_definition": {"en": "Definition", "ru": "Tareef (Definition)"},
    "sec_explanation": {"en": "Detailed explanation", "ru": "Tafseeli wazahat"},
    "sec_example": {"en": "Worked example", "ru": "Misaal"},
    "sec_concepts": {"en": "Important concepts", "ru": "Ahem concepts"},
    "sec_formulas": {"en": "Important formulas", "ru": "Ahem formulas"},
    "sec_workflow": {"en": "Algorithm workflow", "ru": "Algorithm ke steps"},
    "sec_code": {"en": "Python implementation", "ru": "Python implementation"},
    "sec_code_explain": {"en": "What the code does", "ru": "Code kya karta hai"},
    "sec_applications": {"en": "Real-world applications", "ru": "Asal zindagi mein istemal"},
    "sec_lectures": {"en": "Lecture references with timestamps", "ru": "Timestamps ke saath lecture references"},
    "sec_lecture_explain": {"en": "What the lectures say", "ru": "Lectures kya kehti hain"},
    "sec_exam": {"en": "Important exam points", "ru": "Exam ke ahem points"},
    "sec_mistakes": {"en": "Common mistakes", "ru": "Aam ghaltiyan"},
    "sec_practice": {"en": "Practice questions", "ru": "Practice sawalat"},
    "sec_try_playground": {"en": "Try it in the ML Playground", "ru": "ML Playground mein khud try karein"},

    # --- video references -------------------------------------------------
    "segment": {"en": "Relevant segment", "ru": "Mutaliqa hissa"},
    "watch_segment": {"en": "Watch from {ts}", "ru": "{ts} se dekhein"},
    "concepts_here": {"en": "Concepts discussed here", "ru": "Yahan in concepts par baat hui"},
    "excerpt": {"en": "Lecture excerpt", "ru": "Lecture ka hissa"},
    "why_useful": {"en": "Why this segment is useful", "ru": "Yeh hissa kyun mufeed hai"},
    "segment_summary_ru": {"en": "",
                           "ru": "Is video ke {start} se {end} tak ke hisse mein in cheezon par baat hui hai: {terms}. "
                                 "Neeche asli transcript ka hissa (English) diya gaya hai."},
    "why_similarity": {"en": "Matches your topic with similarity {score:.2f} (rank {rank} of {n}).",
                       "ru": "Aap ke topic se {score:.2f} similarity hai (rank {rank} / {n})."},
    "why_definition": {"en": "Contains a definition-style sentence about “{topic}”.",
                       "ru": "Is mein “{topic}” ki tareef jaisa sentence maujood hai."},
    "why_mentions": {"en": "Mentions “{topic}” {n} time(s).", "ru": "“{topic}” ka zikr {n} dafa hai."},
    "why_concepts": {"en": "Covers related concepts: {terms}.", "ru": "Milte julte concepts bhi hain: {terms}."},
    "timestamps_exact": {"en": "Timestamps come from the video's captions and are accurate to a few seconds.",
                         "ru": "Timestamps video ke captions se aaye hain aur chand seconds tak durust hain."},

    # --- navigation / video page -----------------------------------------
    "pick_video": {"en": "Choose a lecture", "ru": "Lecture chunein"},
    "video_timeline": {"en": "Lecture timeline", "ru": "Lecture ki timeline"},
    "ask_range": {"en": "Ask about a time range, e.g. “What is taught between 10:20 and 15:30?”",
                  "ru": "Kisi waqt ke baare mein poochein, maslan “10:20 aur 15:30 ke darmiyan kya parhaya gaya?”"},
    "ask_segment": {"en": "Ask a question about this lecture", "ru": "Is lecture ke baare mein sawal poochein"},
    "range_summary": {"en": "Between {start} and {end} the lecture covers: {terms}.",
                      "ru": "{start} se {end} tak lecture mein yeh parhaya gaya: {terms}."},
    "range_out": {"en": "That time range is outside the available transcript ({start}–{end}).",
                  "ru": "Yeh waqt dastiyab transcript ({start}–{end}) se bahar hai."},
    "range_parse": {"en": "Could not find two times like 10:20 and 15:30 in the question.",
                    "ru": "Sawal mein 10:20 aur 15:30 jaise do waqt nahi mile."},
    "definitions_here": {"en": "Definitions in this part", "ru": "Is hisse ki definitions"},
    "most_useful": {"en": "Most useful sections for “{topic}”", "ru": "“{topic}” ke liye sab se mufeed hisse"},
    "answer_from_lecture": {"en": "Best matching lecture sentences", "ru": "Lecture ke sab se milte julte sentences"},

    # --- quiz -------------------------------------------------------------
    "your_answer": {"en": "Your answer", "ru": "Aap ka jawab"},
    "submit": {"en": "Submit", "ru": "Jama karein"},
    "hint": {"en": "Show a hint", "ru": "Hint dekhein"},
    "show_solution": {"en": "Show solution", "ru": "Hal dekhein"},
    "retry": {"en": "Try again", "ru": "Dobara koshish karein"},
    "correct": {"en": "Correct! Well done.", "ru": "Sahi jawab! Shabash."},
    "partly": {"en": "Partly correct.", "ru": "Jawab kuch had tak sahi hai."},
    "incorrect": {"en": "Not correct yet.", "ru": "Abhi sahi nahi."},
    "quiz_instructions": {"en": "Answer each question, use hints if you are stuck, and retry as often as you like. "
                                "Your score is saved in Learning Progress.",
                          "ru": "Har sawal ka jawab dein, atak jayein to hint lein, aur jitni dafa chahein dobara "
                                "koshish karein. Aap ka score Learning Progress mein save hota hai."},
    "short_grading": {"en": "Short answers are checked by meaning (Sentence-BERT similarity) and key terms, "
                            "so your wording does not need to match exactly.",
                      "ru": "Chhote jawab maani (Sentence-BERT similarity) aur key terms se check hote hain, "
                            "is liye bilkul wohi alfaaz zaroori nahi."},
    "score": {"en": "Score", "ru": "Score"},
    "lecture_questions": {"en": "Questions from the lectures", "ru": "Lectures se sawalat"},
    "lecture_q_note": {"en": "These fill-in-the-blank questions are built from real lecture sentences. "
                             "The blanked word is a key term found by the keyword extractor.",
                       "ru": "Yeh khali jagah wale sawal asli lecture sentences se bane hain. Khali jagah wala "
                             "lafz keyword extractor ka nikala hua key term hai."},
    "fill_blank": {"en": "Fill in the blank (from {title} at {ts}):", "ru": "Khali jagah bharein ({title}, {ts} se):"},
    "choose_topic": {"en": "Choose a topic", "ru": "Topic chunein"},
    "no_questions": {"en": "No questions are available for this topic yet.", "ru": "Is topic ke liye abhi sawalat nahi."},

    # --- math -------------------------------------------------------------
    "problem_type": {"en": "Problem type", "ru": "Sawal ki qisam"},
    "solve": {"en": "Solve step by step", "ru": "Step by step hal karein"},
    "steps": {"en": "Step-by-step solution", "ru": "Step by step hal"},
    "final_answer": {"en": "Final answer", "ru": "Aakhri jawab"},
    "verified": {"en": "Verified with {lib}.", "ru": "{lib} se verify kiya gaya."},
    "practice_problem": {"en": "Similar practice problem", "ru": "Isi tarah ka practice sawal"},
    "new_problem": {"en": "New problem", "ru": "Naya sawal"},
    "check_answer": {"en": "Check my answer", "ru": "Mera jawab check karein"},
    "free_text_math": {"en": "Or type a question, e.g. “derivative of x^3 + 2x” or “solve 2x + 3 = 11”",
                       "ru": "Ya sawal likhein, maslan “derivative of x^3 + 2x” ya “solve 2x + 3 = 11”"},
    "detected_type": {"en": "Detected problem type: {kind}", "ru": "Pehchani gayi qisam: {kind}"},
    "math_parse_error": {"en": "Could not understand the input: {err}", "ru": "Input samajh nahi aaya: {err}"},

    # --- playground -------------------------------------------------------
    "algorithm": {"en": "Algorithm", "ru": "Algorithm"},
    "dataset": {"en": "Dataset", "ru": "Dataset"},
    "data_source": {"en": "Data source", "ru": "Data kahan se"},
    "use_sample": {"en": "Sample data (synthetic)", "ru": "Sample data (banaya hua)"},
    "enter_data": {"en": "Type / paste CSV", "ru": "CSV likhein / paste karein"},
    "upload_csv": {"en": "Upload CSV", "ru": "CSV upload karein"},
    "parameters": {"en": "Parameters", "ru": "Parameters"},
    "run": {"en": "Run algorithm", "ru": "Algorithm chalayein"},
    "results": {"en": "Results", "ru": "Natije"},
    "calculations": {"en": "Intermediate calculations", "ru": "Darmiyani hisaab"},
    "understand": {"en": "How to read this result", "ru": "Is natije ko kaise samjhein"},
    "predict_point": {"en": "Predict a new point", "ru": "Naye point ki prediction"},
    "data_error": {"en": "Data problem: {err}", "ru": "Data mein masla: {err}"},
    "lesson_link": {"en": "Read the lesson for this algorithm in Topic Learning.",
                    "ru": "Is algorithm ka lesson Topic Learning mein parhein."},

    # --- dashboard / progress --------------------------------------------
    "welcome": {"en": "Welcome to ClipSage", "ru": "ClipSage mein khush aamdeed"},
    "recent_searches": {"en": "Recent searches", "ru": "Haal ki searches"},
    "recommended": {"en": "Recommended topics", "ru": "Tajweez kiye gaye topics"},
    "recent_lectures": {"en": "Recently viewed lectures", "ru": "Haal hi mein dekhi lectures"},
    "your_progress": {"en": "Your learning progress", "ru": "Aap ki progress"},
    "nothing_yet": {"en": "Nothing here yet.", "ru": "Abhi yahan kuch nahi."},
    "topics_studied": {"en": "Topics studied", "ru": "Parhe gaye topics"},
    "questions_answered": {"en": "Questions answered", "ru": "Jawab diye gaye sawal"},
    "accuracy": {"en": "Accuracy", "ru": "Durusti"},
    "lectures_indexed": {"en": "Lecture segments indexed", "ru": "Index kiye gaye lecture hisse"},
    "progress_note": {"en": "Progress is stored only on this computer (data/user/progress.json).",
                      "ru": "Progress sirf isi computer par save hoti hai (data/user/progress.json)."},
    "reset_progress": {"en": "Reset all progress", "ru": "Saari progress mita dein"},
    "progress_great": {"en": "Great work — keep going!", "ru": "Zabardast — isi tarah jaari rakhein!"},
    "progress_start": {"en": "Start with a topic and a few questions to see your progress here.",
                       "ru": "Koi topic aur chand sawal karein, phir aap ki progress yahan nazar aayegi."},
    "progress_review": {"en": "Review these topics again: {topics}", "ru": "In topics ko dobara dekhein: {topics}"},

    # --- settings ---------------------------------------------------------
    "retrieval_method": {"en": "Retrieval method", "ru": "Retrieval ka tareeqa"},
    "summary_method": {"en": "Explanation method", "ru": "Explanation ka tareeqa"},
    "segments_to_use": {"en": "Lecture segments to use", "ru": "Kitne lecture hisse istemal hon"},
    "settings_saved": {"en": "Settings are kept for this browser session.",
                       "ru": "Settings is browser session ke liye yaad rehti hain."},
}
