"""
JD–Resume Matcher (Backend Module)

- No UI
- No file handling
- No external APIs
- Fully offline
- Flask/API ready
- JSON-serializable output
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG: SKILL KNOWLEDGE BASE
# =========================================================

SKILLS = [
    # Programming
    "python", "java", "c", "c++", "c#", "javascript",

    # Web
    "html", "css", "react", "angular", "vue",
    "nodejs", "express",

    # Backend / Frameworks
    "spring", "spring boot",

    # Databases
    "sql", "mysql", "postgresql", "mongodb",

    # Data & AI
    "pandas", "numpy", "data analysis",
    "data visualization", "machine learning",

    # Tools
    "git", "github", "excel",

    # Cloud / DevOps
    "aws", "docker", "kubernetes"
]


# =========================================================
# SKILL EXTRACTION (Rule-based, word-boundary safe)
# =========================================================

def extract_skills(text: str) -> set:
    text = text.lower()
    found = set()

    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found.add(skill)

    return found


# =========================================================
# EXPERIENCE EXTRACTION (Rule-based)
# =========================================================

def extract_experience_years(text: str) -> float:
    """
    Supports:
    - 2 years
    - 3+ years
    - 1.5 years
    """
    text = text.lower()
    matches = re.findall(r'(\d+(\.\d+)?)\s*\+?\s*years?', text)

    if not matches:
        return 0.0

    years = [float(m[0]) for m in matches]
    return max(years)


# =========================================================
# TEXT SIMILARITY (TF-IDF + Cosine Similarity)
# =========================================================

def compute_text_similarity(resume_text: str, jd_text: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity * 100, 2)


# =========================================================
# MAIN MATCHING FUNCTION (Flask-ready)
# =========================================================

def jd_resume_matcher(resume_text: str, jd_text: str) -> dict:
    """
    Core backend function
    """

    # ---------------- Skill Matching ----------------
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched_skills = sorted(resume_skills.intersection(jd_skills))
    missing_skills = sorted(jd_skills - resume_skills)

    if jd_skills:
        skill_match_percentage = (len(matched_skills) / len(jd_skills)) * 100
    else:
        skill_match_percentage = 0.0

    # ---------------- Experience Matching ----------------
    candidate_experience = extract_experience_years(resume_text)
    required_experience = extract_experience_years(jd_text)

    if required_experience == 0:
        experience_match_percentage = 100.0
    elif candidate_experience >= required_experience:
        experience_match_percentage = 100.0
    else:
        experience_match_percentage = (
            candidate_experience / required_experience
        ) * 100

    # ---------------- TF-IDF Similarity (Context) ----------------
    text_similarity_percentage = compute_text_similarity(
        resume_text, jd_text
    )

    # ---------------- Final Score (Same logic, explainable) ----------------
    final_match_percentage = round(
        (0.6 * skill_match_percentage) +
        (0.3 * experience_match_percentage) +
        (0.1 * text_similarity_percentage),
        2
    )

    # ---------------- Explainability ----------------
    explanation = (
        f"Matched {len(matched_skills)} out of {len(jd_skills)} required skills. "
        f"Candidate experience is {candidate_experience} years "
        f"against required {required_experience} years. "
        f"Text similarity score is {text_similarity_percentage}%."
    )

    return {
        "match_percentage": final_match_percentage,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "candidate_experience": round(candidate_experience, 2),
        "required_experience": round(required_experience, 2),
        "experience_match_percentage": round(experience_match_percentage, 2),
        "explanation": explanation
    }
