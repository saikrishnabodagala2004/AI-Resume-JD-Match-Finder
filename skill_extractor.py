import re

SKILLS = [
    # Programming
    "python", "java", "c", "c++", "c#", "javascript",

    # Data / Analytics
    "pandas", "numpy", "data analysis", "data visualization",
    "machine learning", "deep learning",

    # Databases
    "sql", "mysql", "postgresql", "mongodb",

    # Tools
    "excel", "power bi", "tableau",

    # Web
    "html", "css", "react",

    # Cloud / DevOps
    "aws", "docker"
]

def extract_skills(text):
    text = text.lower()
    found_skills = set()

    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found_skills.add(skill)

    return list(found_skills)
