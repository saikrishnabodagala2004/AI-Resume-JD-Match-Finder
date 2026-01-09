"""
JD–Resume Matcher (Backend-Only, Offline, LLM-Style Analysis)

Inputs:
- resume_text (str)
- jd_text (str)

Output:
- Structured semantic match analysis
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- Skill Knowledge Base ---------------- #

SKILLS = [
    "python", "java", "javascript", "c", "c++", "c#",
    "html", "css", "react", "angular", "vue",
    "nodejs", "express", "spring", "spring boot",
    "sql", "mysql", "postgresql", "mongodb",
    "pandas", "numpy", "data analysis",
    "data visualization", "machine learning",
    "git", "github", "excel",
    "aws", "docker", "kubernetes"
]


# ---------------- Core Helpers ---------------- #

def _extract_skills(text: str) -> set:
    text = text.lower()
    found = set()
    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found.add(skill)
    return found


def _semantic_similarity(resume_text: str, jd_text: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)


# ---------------- REQUIRED PUBLIC FUNCTION ---------------- #

def run_jd_match(resume_text: str, jd_text: str) -> dict:
    """
    Input:
      - resume_text: extracted resume text
      - jd_text: selected job description text

    Output:
      - structured JD match analysis
    """

    resume_skills = _extract_skills(resume_text)
    jd_skills = _extract_skills(jd_text)

    strengths = sorted(resume_skills.intersection(jd_skills))
    weaknesses = sorted(jd_skills - resume_skills)

    if jd_skills:
        skill_coverage = (len(strengths) / len(jd_skills)) * 100
    else:
        skill_coverage = 0.0

    semantic_score = _semantic_similarity(resume_text, jd_text)

    match_percentage = round(
        (0.7 * skill_coverage) + (0.3 * semantic_score),
        2
    )

    if match_percentage >= 75:
        feedback = "Strong alignment with the job requirements."
    elif match_percentage >= 50:
        feedback = "Partial match. Some key skills are missing."
    else:
        feedback = "Low alignment. Significant skill gaps detected."

    return {
        "match_percentage": match_percentage,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "overall_feedback": feedback
    }
