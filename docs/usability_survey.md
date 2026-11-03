# ClipSage usability survey

Ask 10–15 students to use ClipSage for about 10 minutes. Each student should look up at least three topics they are studying. Then ask them to fill in the form below. You can copy the questions into Google Forms and export the responses as CSV.

Enter the answers in `evaluation/survey_responses.csv`, one row per student, then run `python -m clipsage.evaluate report`.

**Scale:** 1 = strongly disagree · 2 = disagree · 3 = neutral · 4 = agree · 5 = strongly agree

| Column | Question |
|---|---|
| `q1_useful` | ClipSage helped me understand the topics I searched for. |
| `q2_clear` | The explanations were short and clear. |
| `q3_accurate` | The explanations were correct and on-topic. |
| `q4_references` | The video links and timestamps took me to the right part of the lecture. |
| `q5_visual` | The concept map and other visuals helped me see how the ideas connect. |
| `q6_time` | ClipSage saved me time compared with searching YouTube myself. |
| `q7_easy` | The interface was easy to use. |
| `q8_again` | I would use ClipSage again while studying. |
| `comments` | (free text) What should be improved? |

ClipSage does not store anything a user types. The survey is anonymous: do not collect names or roll numbers.
