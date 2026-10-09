
import streamlit as st
import pandas as pd
import re
import hashlib
from io import BytesIO
from datetime import date, timedelta
from difflib import SequenceMatcher
from collections import Counter

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Exam Intelligence Assistant",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#18213b; --muted:#667085; --line:#e7eaf3; --purple:#6558d9; --blue:#4776e6; }
.stApp { background: radial-gradient(circle at 8% 0%, #eef0ff 0, transparent 32%), linear-gradient(145deg,#f7f8fc 0%,#f9fbff 52%,#f2f5ff 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
.block-container { padding-top:1.35rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3,h4 { font-family:'Manrope','DM Sans',sans-serif !important; letter-spacing:-.035em; color:#18213b; }
h2 { margin-top:.25rem !important; }
p, label, .stCaption { color:#566078; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#171b3b 0%,#26245a 100%); }
section[data-testid="stSidebar"] * { color:#f5f6ff !important; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.5rem .65rem; border-radius:10px; }
div[data-testid="stMetric"] { background:rgba(255,255,255,.88); border:1px solid rgba(222,226,241,.95); border-radius:18px; padding:17px 18px; box-shadow:0 8px 24px rgba(36,48,92,.045); min-height:112px; }
div[data-testid="stMetric"] label { color:#69738d !important; font-size:.83rem !important; font-weight:600 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#20264a; font-family:'Manrope',sans-serif; font-weight:800; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size:.78rem; }
.stSelectbox > div > div, .stTextInput > div > div, .stDateInput > div > div, .stNumberInput > div > div, .stTextArea textarea { border-radius:12px !important; border-color:#dfe4f1 !important; }
.stButton > button, .stDownloadButton > button { border-radius:12px; border:0; background:linear-gradient(115deg,#6156d9,#4776e6); color:white; font-weight:700; padding:.58rem 1rem; box-shadow:0 6px 16px rgba(87,91,210,.18); transition:transform .15s ease,box-shadow .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { color:white; transform:translateY(-1px); box-shadow:0 9px 22px rgba(87,91,210,.25); }
div[data-testid="stDataFrame"] { background:white; border:1px solid #e5e8f2; border-radius:16px; overflow:hidden; }
div[data-testid="stAlert"] { border-radius:14px; border:1px solid #e4e8f5; }
hr { border-color:#e4e8f2; }
.hero-panel { background:linear-gradient(120deg,#171b3b 0%,#35317e 52%,#6558d9 100%); padding:30px 32px; border-radius:24px; color:#fff; margin:0 0 20px 0; box-shadow:0 16px 38px rgba(50,49,126,.18); position:relative; overflow:hidden; }
.hero-panel:after { content:''; position:absolute; width:250px; height:250px; right:-55px; top:-105px; border:1px solid rgba(255,255,255,.18); border-radius:50%; box-shadow:0 0 0 28px rgba(255,255,255,.045),0 0 0 58px rgba(255,255,255,.035); }
.hero-panel h1 { color:#fff !important; font-size:2.05rem !important; margin:7px 0 8px 0 !important; }
.hero-panel p { color:#e3e7ff !important; margin:0; max-width:760px; font-size:1rem; }
.eyebrow { color:#c9ceff; font-size:.72rem; font-weight:800; letter-spacing:.15em; text-transform:uppercase; }
.subject-strip { background:rgba(255,255,255,.82); border:1px solid #e4e8f4; border-radius:16px; padding:13px 17px; margin:4px 0 15px 0; display:flex; justify-content:space-between; align-items:center; gap:12px; }
.subject-strip strong { color:#25284f; font-size:1.05rem; }
.subject-strip span { color:#747d97; font-size:.84rem; }
.section-kicker { color:#7774d7; font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.13em; margin-bottom:4px; }
.insight-card { background:rgba(255,255,255,.94); border:1px solid #e4e8f4; border-radius:18px; padding:18px 18px 17px 18px; min-height:155px; box-shadow:0 8px 22px rgba(38,48,91,.04); }
.insight-card .icon { font-size:1.3rem; margin-bottom:8px; }
.insight-card .label { font-size:.75rem; color:#747d97; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
.insight-card .value { color:#20264a; font-family:'Manrope',sans-serif; font-size:1.15rem; font-weight:800; margin:5px 0; }
.insight-card .detail { color:#68718a; font-size:.85rem; line-height:1.45; }
.priority-chip { display:inline-block; padding:4px 9px; border-radius:999px; font-size:.75rem; font-weight:800; background:#f0edff; color:#5948c6; }
@media (max-width: 700px) { .block-container { padding-left:1rem; padding-right:1rem; } .hero-panel { padding:23px 20px; } .hero-panel h1 { font-size:1.6rem !important; } }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

for key, default in {
    "subjects": [],
    "next_id": 1,
    "analysis": [],
    "completed_tasks": set(),
    "active_dashboard": "Subjects & Setup"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# TEXT NORMALIZATION AND MATCHING HELPERS
# =========================================================

STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this",
    "find", "show", "prove", "calculate", "using", "given",
    "derive", "determine", "solve", "evaluate", "write",
    "explain", "define", "of", "to", "in", "on", "is",
    "are", "was", "were", "a", "an", "be", "if", "or",
    "as", "by", "at", "it", "its", "let", "where",
    "which", "then", "also", "following", "any", "all",
    "question", "questions", "marks", "attempt", "answer",
    "obtain", "hence", "using", "show", "calculate",
    "each", "every", "function", "given", "following"
}

# Common mathematical and academic variations.
SYNONYMS = {
    "asymptote": ["asymptotes", "asymptotic"],
    "curve": ["curves", "tracing", "trace", "traced"],
    "differentiate": ["differentiation", "derivative", "derivatives"],
    "successive": ["repeated", "higher order", "higher-order"],
    "taylor": ["taylor's", "taylor series"],
    "maclaurin": ["maclaurin's", "maclaurin series"],
    "expansion": ["series", "expand"],
    "curvature": ["radius of curvature", "curvature radius"],
    "polar": ["polar coordinates", "polar form"],
    "cartesian": ["cartesian coordinates", "rectangular coordinates"],
    "parametric": ["parametric equations", "parameter"],
    "beta": ["beta function"],
    "gamma": ["gamma function"],
    "leibnitz": ["leibniz", "leibnitz theorem", "leibniz theorem"],
    "theorem": ["theorem", "theorems"],
    "integration": ["integral", "integrals", "integrate"],
    "matrix": ["matrices"],
    "eigenvalue": ["eigenvalues"],
    "eigenvector": ["eigenvectors"],
    "differential": ["differential equation", "differential equations"],
    "probability": ["probability distribution", "probability distributions"],
    "electrochemistry": ["electrochemical"],
    "corrosion": ["rusting", "corrosion prevention"],
    "polymer": ["polymers", "polymeric"],
    "thermodynamics": ["thermodynamic"],
    "semiconductor": ["semiconductors"],
    "diode": ["diodes", "pn junction", "p-n junction"],
}

def normalize(text):
    text = str(text).lower()
    text = text.replace("’", "'")
    text = text.replace("–", " ").replace("—", " ")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return [
        w for w in normalize(text).split()
        if len(w) > 1 and w not in STOP_WORDS
    ]


def expand_topic(topic):
    """Expand topic terminology to improve semantic keyword coverage."""
    base = normalize(topic)
    expanded = [base]

    for key, alternatives in SYNONYMS.items():
        if re.search(r"\b" + re.escape(key) + r"\w*\b", base):
            expanded.extend(alternatives)

        for alternative in alternatives:
            alt = normalize(alternative)
            if alt and re.search(
                r"\b" + re.escape(alt) + r"\b", base
            ):
                expanded.append(key)

    return list(dict.fromkeys(expanded))


def stable_id(*parts):
    raw = "|".join(map(str, parts))
    return hashlib.sha256(raw.encode()).hexdigest()[:18]


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf(pdf_bytes):
    pieces = []

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pieces.append(page_text)
    except Exception:
        pass

    text = "\n".join(pieces)

    if len(text.strip()) >= 40:
        return text, "Selectable text"

    # Optional OCR fallback, if system packages are installed.
    try:
        from pdf2image import convert_from_bytes
        import pytesseract

        pages = convert_from_bytes(pdf_bytes, dpi=200)
        ocr_text = "\n".join(
            pytesseract.image_to_string(page)
            for page in pages
        )

        if ocr_text.strip():
            return ocr_text, "OCR"

    except Exception:
        pass

    return text, "OCR unavailable or no readable text"


# =========================================================
# QUESTION EXTRACTION
# =========================================================

def extract_questions(text):
    # Remove common paper instructions and metadata.
    lines = []

    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue

        if re.match(
            r"(?i)^(time allowed|maximum marks|max\.?\s*marks|"
            r"attempt any|attempt all|instructions|duration|"
            r"roll no|enrollment no|semester|subject\s*:|"
            r"university|department)\b",
            line
        ):
            continue

        lines.append(line)

    text = "\n".join(lines)

    # Recognize Q1, Q.1, Question 1, 1. and 1)
    pattern = re.compile(
        r"(?im)^\s*(?:question\s*|q\s*\.?\s*)?"
        r"(\d{1,2})\s*[\.\):]\s+"
    )

    matches = list(pattern.finditer(text))
    result = []

    for i, match in enumerate(matches):
        start = match.end()
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        question = re.sub(
            r"\s+", " ", text[start:end]
        ).strip()

        # Exclude empty entries and page headers.
        if len(question) < 18:
            continue

        # Exclude instruction-like text.
        if re.match(
            r"(?i)^(attempt any|answer all|time allowed|"
            r"maximum marks|instructions)\b",
            question
        ):
            continue

        result.append({
            "number": match.group(1),
            "text": question
        })

    # Fallback for papers without recognized numbering.
    if not result:
        paragraphs = re.split(r"\n+", text)
        for i, paragraph in enumerate(paragraphs, 1):
            paragraph = paragraph.strip()
            if len(paragraph) >= 35:
                result.append({
                    "number": str(i),
                    "text": paragraph
                })

    return result


# =========================================================
# IMPROVED QUESTION-TO-TOPIC MAPPING
# =========================================================

def keyword_match_score(question, topic):
    """
    Measures overlap between question concepts and syllabus concepts.
    Uses topic aliases and partial word matches.
    """
    q = normalize(question)
    q_tokens = set(tokenize(q))

    topic_variants = expand_topic(topic)
    best = 0.0

    for variant in topic_variants:
        t_tokens = set(tokenize(variant))

        if not t_tokens:
            continue

        # Full topic phrase is strong evidence.
        phrase_score = 1.0 if variant in q else 0.0

        # How much of the topic's vocabulary appears in the question?
        coverage = len(q_tokens & t_tokens) / len(t_tokens)

        # Partial matches for plural or word-ending variations.
        partial_matches = 0
        for term in t_tokens:
            if any(
                qword.startswith(term) or term.startswith(qword)
                for qword in q_tokens
                if min(len(qword), len(term)) >= 4
            ):
                partial_matches += 1

        partial_coverage = partial_matches / len(t_tokens)

        score = max(
            phrase_score,
            0.75 * coverage + 0.25 * partial_coverage
        )

        best = max(best, score)

    return best


def match_topics(questions, topics):
    if not questions or not topics:
        return []

    topic_variants = [
        " ".join(expand_topic(topic))
        for topic in topics
    ]

    questions_text = [q["text"] for q in questions]

    # Character n-grams help with spelling differences and word forms.
    # Word n-grams help with meaningful phrases.
    try:
        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode"
        )

        combined_text = topic_variants + questions_text
        word_matrix = word_vectorizer.fit_transform(combined_text)

        topic_word_matrix = word_matrix[:len(topics)]
        question_word_matrix = word_matrix[len(topics):]

        word_similarity = cosine_similarity(
            question_word_matrix, topic_word_matrix
        )

    except ValueError:
        word_similarity = [
            [0.0] * len(topics) for _ in questions
        ]

    try:
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True
        )

        char_matrix = char_vectorizer.fit_transform(
            topic_variants + questions_text
        )

        char_similarity = cosine_similarity(
            char_matrix[len(topics):],
            char_matrix[:len(topics)]
        )

    except ValueError:
        char_similarity = [
            [0.0] * len(topics) for _ in questions
        ]

    output = []

    for i, question in enumerate(questions):
        scores = []

        for j, topic in enumerate(topics):
            lexical = keyword_match_score(
                question["text"], topic
            )
            word_score = float(word_similarity[i][j])
            char_score = float(char_similarity[i][j])

            # Lexical coverage receives greater weight because a
            # question often describes a topic without naming it.
            score = (
                0.55 * lexical
                + 0.30 * word_score
                + 0.15 * char_score
            )

            scores.append(score)

        order = sorted(
            range(len(scores)),
            key=lambda j: scores[j],
            reverse=True
        )

        best_index = order[0]
        best_score = scores[best_index]
        second_score = (
            scores[order[1]] if len(order) > 1 else 0
        )

        # A match can be shown even if it is uncertain.
        # The review flag communicates uncertainty honestly.
        matched_topic = topics[best_index]

        if best_score >= 0.38:
            match_status = "Strong match"
        elif best_score >= 0.22:
            match_status = "Review suggested"
        else:
            match_status = "Low confidence — verify"

        output.append({
            "Question No.": question["number"],
            "Question": question["text"],
            "Matched Topic": matched_topic,
            "Confidence %": round(best_score * 100, 1),
            "Match Status": match_status,
            "Score Gap %": round(
                max(0, best_score - second_score) * 100, 1
            ),
            "Question Type": classify_type(question["text"]),
            "Difficulty (estimated)": estimate_difficulty(
                question["text"]
            ),
            "Marks (if detected)": extract_marks(question["text"])
        })

    return output


# =========================================================
# QUESTION CLASSIFICATION
# =========================================================

def classify_type(question):
    q = normalize(question)

    if any(x in q for x in [
        "derive", "prove that", "proof of",
        "deduce", "show that"
    ]):
        return "Derivation"

    if any(x in q for x in [
        "calculate", "compute", "find the value",
        "evaluate", "numerical", "determine the current",
        "find the voltage", "solve for"
    ]):
        return "Numerical"

    if any(x in q for x in [
        "explain", "describe", "discuss", "define",
        "differentiate between", "compare",
        "write a note", "state the"
    ]):
        return "Theory"

    return "Other / Mixed"


def estimate_difficulty(question):
    q = normalize(question)

    hard_words = [
        "hence prove", "derive and prove",
        "multi-step", "optimize", "discuss in detail"
    ]
    medium_words = [
        "derive", "calculate", "compare", "differentiate",
        "evaluate", "determine", "explain"
    ]

    score = (
        2 * sum(w in q for w in hard_words)
        + sum(w in q for w in medium_words)
    )

    if len(q) > 220:
        score += 2
    elif len(q) > 100:
        score += 1

    if score >= 4:
        return "Hard"
    if score >= 2:
        return "Medium"
    return "Easy"


def extract_marks(question):
    patterns = [
        r"\[\s*(\d{1,2})\s*marks?\s*\]",
        r"\(\s*(\d{1,2})\s*marks?\s*\)",
        r"(\d{1,2})\s*marks?\s*$",
        r"\[\s*(\d{1,2})\s*\]\s*$"
    ]

    for pattern in patterns:
        m = re.search(pattern, question, re.I)
        if m:
            marks = int(m.group(1))
            if 1 <= marks <= 30:
                return marks

    return None


def extract_year(filename):
    m = re.search(r"\b(19\d{2}|20\d{2})\b", filename)
    return int(m.group(1)) if m else None


# =========================================================
# ANALYSIS
# =========================================================

def analyze_subject(subject, files):
    topics = [
        x.strip() for x in subject["topics"].splitlines()
        if x.strip()
    ]

    all_rows = []
    notices = []

    for pdf in files:
        text, method = extract_pdf(pdf.getvalue())

        if not text.strip():
            notices.append(
                f"{pdf.name}: no readable text was extracted."
            )
            continue

        questions = extract_questions(text)

        if not questions:
            notices.append(
                f"{pdf.name}: no question boundaries were detected."
            )
            continue

        rows = match_topics(questions, topics)
        year = extract_year(pdf.name)

        for row in rows:
            row["Paper"] = pdf.name
            row["Year"] = year
            all_rows.append(row)

        notices.append(
            f"{pdf.name}: {len(rows)} questions detected ({method})."
        )

    result = pd.DataFrame(all_rows)

    if result.empty:
        return result, pd.DataFrame(), notices

    summary = []

    for topic in topics:
        subset = result[result["Matched Topic"] == topic]

        summary.append({
            "Topic": topic,
            "Questions": len(subset),
            "Papers Appearing": subset["Paper"].nunique(),
            "Theory": int((subset["Question Type"] == "Theory").sum()),
            "Numerical": int((subset["Question Type"] == "Numerical").sum()),
            "Derivation": int((subset["Question Type"] == "Derivation").sum()),
            "Easy": int((subset["Difficulty (estimated)"] == "Easy").sum()),
            "Medium": int((subset["Difficulty (estimated)"] == "Medium").sum()),
            "Hard": int((subset["Difficulty (estimated)"] == "Hard").sum()),
            "Marks Detected": subset["Marks (if detected)"].sum(min_count=1)
        })

    topic_df = pd.DataFrame(summary)

    total_questions = max(len(result), 1)
    total_papers = max(result["Paper"].nunique(), 1)

    topic_df["Frequency %"] = (
        topic_df["Questions"] / total_questions * 100
    ).round(1)

    topic_df["Paper Coverage %"] = (
        topic_df["Papers Appearing"] / total_papers * 100
    ).round(1)

    total_marks = topic_df["Marks Detected"].sum(min_count=1)

    if pd.notna(total_marks) and total_marks > 0:
        topic_df["Detected Marks %"] = (
            topic_df["Marks Detected"] / total_marks * 100
        ).round(1)
    else:
        topic_df["Detected Marks %"] = None

    topic_df["Priority Score"] = (
        0.60 * topic_df["Frequency %"]
        + 0.40 * topic_df["Paper Coverage %"]
    ).round(1)

    def priority(row):
        if row["Questions"] == 0:
            return "No mapped questions"
        if row["Priority Score"] >= 40:
            return "High"
        if row["Priority Score"] >= 20:
            return "Medium"
        return "Low"

    topic_df["Priority"] = topic_df.apply(priority, axis=1)

    topic_df["Reason"] = topic_df.apply(
        lambda r: (
            "No question mapped to this topic."
            if r["Questions"] == 0
            else (
                f"{int(r['Questions'])} question(s) across "
                f"{int(r['Papers Appearing'])} paper(s)."
            )
        ),
        axis=1
    )

    topic_df = topic_df.sort_values(
        ["Priority Score", "Questions"],
        ascending=False
    ).reset_index(drop=True)

    return result, topic_df, notices


# =========================================================
# TOPIC INTELLIGENCE + PERSONAL READINESS HELPERS
# =========================================================

def topic_difficulty_from_counts(row):
    """Estimate topic difficulty from extracted question classifications."""
    total = int(row.get("Easy", 0)) + int(row.get("Medium", 0)) + int(row.get("Hard", 0))
    if total == 0:
        return "Insufficient data"

    hard_ratio = float(row.get("Hard", 0)) / total
    medium_or_hard_ratio = (
        float(row.get("Medium", 0)) + float(row.get("Hard", 0))
    ) / total
    easy_ratio = float(row.get("Easy", 0)) / total

    if hard_ratio >= 0.40 or medium_or_hard_ratio >= 0.80:
        return "Hard"
    if easy_ratio >= 0.65:
        return "Easy"
    return "Medium"


def readiness_bonus(record):
    """Return a transparent, bounded revision boost from student self-assessment."""
    if not isinstance(record, dict):
        return 0.0

    level_bonus = {
        "Very difficult": 20.0,
        "Difficult": 15.0,
        "Okay": 7.0,
        "Confident": 0.0,
        "Not assessed": 0.0,
    }.get(record.get("readiness", "Not assessed"), 0.0)

    try:
        accuracy = max(0.0, min(100.0, float(record.get("accuracy", 100))))
        accuracy_bonus = max(0.0, min(15.0, (60.0 - accuracy) * 0.25))
    except (TypeError, ValueError):
        accuracy_bonus = 0.0

    return level_bonus + accuracy_bonus


def sync_task_completion(task_id):
    """Synchronize a checkbox value into the completed-task set on change."""
    widget_key = f"task_{task_id}"
    completed = st.session_state.setdefault("completed_tasks", set())

    if st.session_state.get(widget_key, False):
        completed.add(task_id)
    else:
        completed.discard(task_id)


def rebuild_topic_summary(question_df, topic_names):
    """Recalculate topic statistics after a topic assignment is manually corrected."""
    summary = []

    for topic in topic_names:
        subset = question_df[question_df["Matched Topic"] == topic]
        summary.append({
            "Topic": topic,
            "Questions": len(subset),
            "Papers Appearing": subset["Paper"].nunique(),
            "Theory": int((subset["Question Type"] == "Theory").sum()),
            "Numerical": int((subset["Question Type"] == "Numerical").sum()),
            "Derivation": int((subset["Question Type"] == "Derivation").sum()),
            "Easy": int((subset["Difficulty (estimated)"] == "Easy").sum()),
            "Medium": int((subset["Difficulty (estimated)"] == "Medium").sum()),
            "Hard": int((subset["Difficulty (estimated)"] == "Hard").sum()),
            "Marks Detected": subset["Marks (if detected)"].sum(min_count=1),
        })

    topic_df = pd.DataFrame(summary)
    total_questions = max(len(question_df), 1)
    total_papers = max(question_df["Paper"].nunique(), 1)

    topic_df["Frequency %"] = (
        topic_df["Questions"] / total_questions * 100
    ).round(1)
    topic_df["Paper Coverage %"] = (
        topic_df["Papers Appearing"] / total_papers * 100
    ).round(1)

    total_marks = topic_df["Marks Detected"].sum(min_count=1)
    if pd.notna(total_marks) and total_marks > 0:
        topic_df["Detected Marks %"] = (
            topic_df["Marks Detected"] / total_marks * 100
        ).round(1)
    else:
        topic_df["Detected Marks %"] = None

    topic_df["Priority Score"] = (
        0.60 * topic_df["Frequency %"]
        + 0.40 * topic_df["Paper Coverage %"]
    ).round(1)

    def priority(row):
        if row["Questions"] == 0:
            return "No mapped questions"
        if row["Priority Score"] >= 40:
            return "High"
        if row["Priority Score"] >= 20:
            return "Medium"
        return "Low"

    topic_df["Priority"] = topic_df.apply(priority, axis=1)
    topic_df["Reason"] = topic_df.apply(
        lambda row: (
            "No question mapped to this topic."
            if row["Questions"] == 0
            else (
                f"{int(row['Questions'])} question(s) across "
                f"{int(row['Papers Appearing'])} paper(s)."
            )
        ),
        axis=1,
    )
    topic_df["Difficulty (estimated)"] = topic_df.apply(
        topic_difficulty_from_counts, axis=1
    )
    return topic_df.sort_values(
        ["Priority Score", "Questions"], ascending=False
    ).reset_index(drop=True)


# =========================================================
# STUDY PLAN INCLUDING REVISION
# =========================================================

def build_timetable(analysis, daily_hours):
    today = date.today()
    tasks = []

    for subject in analysis:
        exam_date = subject["exam_date"]
        topics = subject["topics"]

        # Do not create new study tasks after the exam.
        if exam_date <= today:
            continue

        readiness_map = subject.get("readiness", {})
        difficult_topics = [
            topic_name
            for topic_name, record in readiness_map.items()
            if isinstance(record, dict)
            and record.get("readiness") in ("Difficult", "Very difficult")
        ]

        active_topics = topics[
            (topics["Questions"] > 0)
            | topics["Topic"].isin(difficult_topics)
        ]

        for _, row in active_topics.iterrows():
            record = readiness_map.get(row["Topic"], {})
            historical_score = float(row["Priority Score"])
            personalized_score = min(
                100.0, historical_score + readiness_bonus(record)
            )

            if int(row["Questions"]) == 0 and not record:
                continue

            priority = (
                "High" if personalized_score >= 40
                else "Medium" if personalized_score >= 20
                else "Low"
            )
            minutes = (
                75 if priority == "High"
                else 50 if priority == "Medium"
                else 35
            )

            tasks.append({
                "Subject": subject["name"],
                "Topic": row["Topic"],
                "Priority": priority,
                "Minutes": minutes,
                "Exam Date": exam_date,
                "Score": personalized_score,
                "Kind": "Study"
            })

        # Add revision and practice for the day before the exam.
        tasks.append({
            "Subject": subject["name"],
            "Topic": "Revise important formulas and concepts",
            "Priority": "Revision",
            "Minutes": 45,
            "Exam Date": exam_date,
            "Score": 100,
            "Kind": "Revision"
        })

        tasks.append({
            "Subject": subject["name"],
            "Topic": "Solve selected PYQs and review mistakes",
            "Priority": "Revision",
            "Minutes": 45,
            "Exam Date": exam_date,
            "Score": 95,
            "Kind": "Revision"
        })

    if not tasks:
        return pd.DataFrame()

    # Earlier exams are handled first; high-priority topics first.
    tasks.sort(
        key=lambda x: (x["Exam Date"], -x["Score"])
    )

    # Each task must finish before its subject's exam.
    # Revision tasks are reserved for the day before the exam.
    revision_tasks = [
        t for t in tasks if t["Kind"] == "Revision"
    ]
    study_tasks = [
        t for t in tasks if t["Kind"] == "Study"
    ]

    schedule = []
    capacity = daily_hours * 60

    # Reserve revision for the final day before each exam.
    for task in revision_tasks:
        revision_day = task["Exam Date"] - timedelta(days=1)

        schedule.append({
            **task,
            "Date": revision_day
        })

    # Allocate topic study sessions across the days before the exam.
    for task in study_tasks:
        exam_date = task["Exam Date"]
        deadline = exam_date - timedelta(days=2)

        available_days = max(
            (deadline - today).days + 1, 1
        )

        # Distribute study tasks over the available dates.
        placed = False

        for offset in range(available_days):
            study_day = today + timedelta(days=offset)

            day_minutes = sum(
                x["Minutes"] for x in schedule
                if x["Date"] == study_day
            )

            if (
                study_day < exam_date
                and day_minutes + task["Minutes"] <= capacity
            ):
                schedule.append({
                    **task,
                    "Date": study_day
                })
                placed = True
                break

        if not placed:
            # If overloaded, use the latest available pre-exam day.
            last_day = exam_date - timedelta(days=2)

            if last_day >= today:
                schedule.append({
                    **task,
                    "Date": last_day
                })

    df = pd.DataFrame(schedule)

    if df.empty:
        return df

    df["Task ID"] = df.apply(
        lambda r: stable_id(
            r["Date"], r["Subject"], r["Topic"], r["Kind"]
        ),
        axis=1
    )

    return df.sort_values(
        ["Date", "Exam Date", "Kind", "Subject"]
    ).reset_index(drop=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
<h1>🎓 Exam Intelligence Assistant</h1>
<p>Topic intelligence · Personalized priorities · A smarter revision plan</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Study Settings")
    daily_hours = st.slider(
        "Available study hours per day",
        1, 12, 4
    )
    st.caption("Free analysis. No paid AI API required.")


# =========================================================
# AUTOMATIC DASHBOARD NAVIGATION
# =========================================================

dashboard_names = [
    "Subjects & Setup",
    "Exam Intelligence",
    "Study Planner"
]

# Change the radio state before the widget is instantiated.
if st.session_state.pop("navigate_to_intelligence", False):
    st.session_state["active_dashboard"] = "Exam Intelligence"

if st.session_state.active_dashboard not in dashboard_names:
    st.session_state.active_dashboard = dashboard_names[0]

active_dashboard = st.radio(
    "Navigate dashboard",
    dashboard_names,
    horizontal=True,
    key="active_dashboard",
    label_visibility="collapsed"
)


# =========================================================
# DASHBOARD 1: SUBJECTS & SETUP
# =========================================================

if active_dashboard == "Subjects & Setup":
    st.header("📚 Subjects and Question Papers")
    st.write(
        "Create each subject separately. Enter one syllabus topic per line "
        "and upload all relevant previous-year papers."
    )

    if st.button("➕ Add Subject", type="primary"):
        sid = st.session_state.next_id
        st.session_state.next_id += 1

        st.session_state.subjects.append({
            "id": sid,
            "name": "",
            "exam_date": date.today() + timedelta(days=7),
            "topics": ""
        })
        st.rerun()

    if not st.session_state.subjects:
        st.info("Start by clicking Add Subject.")

    for index, subject in enumerate(st.session_state.subjects):
        sid = subject["id"]

        with st.container(border=True):
            st.subheader(f"Subject {index + 1}")

            c1, c2 = st.columns([2, 1])

            with c1:
                name = st.text_input(
                    "Subject name",
                    value=subject["name"],
                    key=f"name_{sid}"
                )

            with c2:
                exam_date = st.date_input(
                    "Exam date",
                    value=subject["exam_date"],
                    key=f"date_{sid}"
                )

            topics = st.text_area(
                "Syllabus topics (one per line)",
                value=subject["topics"],
                height=130,
                key=f"topics_{sid}"
            )

            files = st.file_uploader(
                "Upload previous-year papers (PDF)",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"files_{sid}"
            )

            if files:
                st.caption("Selected papers:")
                for file in files:
                    st.write(f"📄 {file.name}")

            c3, c4 = st.columns(2)

            with c3:
                if st.button("Save Subject", key=f"save_{sid}"):
                    if not name.strip() or not topics.strip():
                        st.error("Enter a subject name and syllabus.")
                    else:
                        duplicate = any(
                            s["id"] != sid
                            and s["name"].strip().lower() == name.strip().lower()
                            for s in st.session_state.subjects
                        )

                        if duplicate:
                            st.error("This subject name already exists.")
                        else:
                            subject["name"] = name.strip()
                            subject["exam_date"] = exam_date
                            subject["topics"] = topics
                            st.success("Subject saved.")

            with c4:
                if st.button("Remove Subject", key=f"remove_{sid}"):
                    st.session_state.subjects = [
                        s for s in st.session_state.subjects
                        if s["id"] != sid
                    ]
                    st.session_state.analysis = []
                    st.session_state.completed_tasks = set()
                    st.rerun()

    st.divider()

    if st.button(
        "🚀 Analyze All Subjects",
        type="primary",
        use_container_width=True
    ):
        analyses = []
        errors = []

        for subject in st.session_state.subjects:
            sid = subject["id"]

            name = st.session_state.get(
                f"name_{sid}", subject["name"]
            ).strip()

            exam_date = st.session_state.get(
                f"date_{sid}", subject["exam_date"]
            )

            topics = st.session_state.get(
                f"topics_{sid}", subject["topics"]
            )

            files = st.session_state.get(
                f"files_{sid}", []
            )

            if not name or not topics.strip() or not files:
                errors.append(
                    f"{name or 'Subject ' + str(sid)}: missing name, topics, or PDF."
                )
                continue

            subject["name"] = name
            subject["exam_date"] = exam_date
            subject["topics"] = topics

            with st.spinner(f"Analyzing {name}..."):
                question_df, topic_df, notices = analyze_subject(
                    subject, files
                )

            for notice in notices:
                st.write(notice)

            if question_df.empty:
                errors.append(
                    f"{name}: no readable questions were extracted."
                )
                continue

            analyses.append({
                "id": sid,
                "name": name,
                "exam_date": exam_date,
                "questions": question_df,
                "topics": topic_df,
                "topic_names": [
                    item.strip()
                    for item in topics.splitlines()
                    if item.strip()
                ],
                "readiness": st.session_state.get(
                    f"readiness_{sid}", {}
                )
            })

        for error in errors:
            st.warning(error)

        if analyses:
            st.session_state.analysis = analyses

            # Navigate before the radio widget is created on the next run.
            st.session_state["navigate_to_intelligence"] = True
            st.rerun()


# =========================================================
# DASHBOARD 2: EXAM INTELLIGENCE
# =========================================================

elif active_dashboard == "Exam Intelligence":
    st.markdown("""
    <div class="hero-panel">
      <div class="eyebrow">YOUR EXAM COMMAND CENTER</div>
      <h1>Study smarter. Walk in prepared.</h1>
      <p>Turn previous question papers into clear revision priorities, topic trends, and syllabus coverage insights.</p>
    </div>
    """, unsafe_allow_html=True)

    analyses = st.session_state.analysis

    if not analyses:
        st.info("Add your subjects and analyze them first to unlock your exam dashboard.")
    else:
        selected = st.selectbox(
            "Select subject",
            range(len(analyses)),
            format_func=lambda i: analyses[i]["name"]
        )

        subject = analyses[selected]
        questions = subject["questions"].copy()
        topics = subject["topics"].copy()
        topic_names = subject.get("topic_names", topics["Topic"].tolist())

        # Backward compatibility for analyses created before the latest summary.
        required_summary_columns = {
            "Topic", "Questions", "Papers Appearing", "Frequency %",
            "Paper Coverage %", "Priority Score", "Priority"
        }
        if not required_summary_columns.issubset(set(topics.columns)):
            topics = rebuild_topic_summary(questions, topic_names)
            subject["topics"] = topics
            st.session_state.analysis[selected] = subject

        if "Difficulty (estimated)" not in topics.columns:
            topics["Difficulty (estimated)"] = topics.apply(
                topic_difficulty_from_counts, axis=1
            )
            subject["topics"] = topics
            st.session_state.analysis[selected] = subject

        total_questions = len(questions)
        papers_count = int(questions["Paper"].nunique()) if "Paper" in questions else 0
        topics_with_questions = int((topics["Questions"] > 0).sum())
        topics_without_questions = int((topics["Questions"] == 0).sum())
        days_remaining = max((subject["exam_date"] - date.today()).days, 0)

        st.markdown(
            f"<div class='subject-strip'><div><strong>📘 {subject['name']}</strong><br><span>Historical paper analysis · Revision overview</span></div><div class='priority-chip'>{len(topic_names)} syllabus topics</div></div>",
            unsafe_allow_html=True
        )
        a, b, c, d = st.columns(4)
        a.metric("Questions analyzed", total_questions, help="Questions extracted from the uploaded papers.")
        b.metric("Papers analyzed", papers_count, help="Unique uploaded papers included in this analysis.")
        c.metric("Topics found", f"{topics_with_questions}/{len(topics)}", help="Topics with at least one question assigned in the uploaded papers.")
        d.metric("Days until exam", days_remaining, help="Calculated from the exam date saved for this subject.")

        st.markdown("<div class='section-kicker'>THE REVISION OVERVIEW</div>", unsafe_allow_html=True)
        st.subheader("🔥 Smart Revision Priority")
        st.caption(
            "Rankings use the historical patterns in your uploaded papers. They guide revision; they cannot predict or guarantee the next exam."
        )

        priority_rows = []
        for _, row in topics.iterrows():
            question_count = int(row.get("Questions", 0) or 0)
            papers_appearing = int(row.get("Papers Appearing", 0) or 0)
            score = float(row.get("Priority Score", 0) or 0)

            if question_count == 0:
                priority = "Evidence unavailable"
                action = "Revise from the syllabus and class notes"
            elif score >= 40:
                priority = "High"
                action = "Prioritize revision and practice"
            elif score >= 20:
                priority = "Medium"
                action = "Include in planned revision"
            else:
                priority = "Lower"
                action = "Review after higher-frequency topics"

            priority_rows.append({
                "Topic": row["Topic"],
                "Questions found": question_count,
                "Papers appearing in": papers_appearing,
                "Frequency (%)": row.get("Frequency %", 0),
                "Paper coverage (%)": row.get("Paper Coverage %", 0),
                "Estimated difficulty": row.get(
                    "Difficulty (estimated)", "Insufficient data"
                ),
                "Revision priority": priority,
                "Suggested action": action,
                "Priority score": round(score, 1)
            })

        priority_df = pd.DataFrame(priority_rows)
        priority_order = {
            "High": 0,
            "Medium": 1,
            "Lower": 2,
            "Evidence unavailable": 3
        }
        priority_df["_order"] = priority_df["Revision priority"].map(priority_order)
        priority_df = priority_df.sort_values(
            ["_order", "Priority score", "Questions found"],
            ascending=[True, False, False]
        ).drop(columns=["_order"]).reset_index(drop=True)
        priority_df.insert(0, "Rank", range(1, len(priority_df) + 1))

        st.markdown(
            f"<div class='section-kicker'>RANKED FROM YOUR PAPER HISTORY</div>",
            unsafe_allow_html=True
        )
        st.dataframe(priority_df, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download topic priority report (CSV)",
            data=priority_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{subject['name'].replace(' ', '_')}_topic_priority.csv",
            mime="text/csv"
        )

        st.divider()
        st.markdown("<div class='section-kicker'>WHAT THE DATA IS TELLING YOU</div>", unsafe_allow_html=True)
        st.subheader("💡 Actionable Exam Insights")

        if not topics.empty:
            mapped_topics = topics[topics["Questions"] > 0].copy()
            unmapped_topics = topics[topics["Questions"] == 0]["Topic"].tolist()

            if not mapped_topics.empty:
                top_topic = mapped_topics.sort_values(
                    ["Priority Score", "Questions"], ascending=False
                ).iloc[0]
                st.info(
                    f"**Start here:** {top_topic['Topic']} has the strongest "
                    f"historical priority score ({top_topic['Priority Score']}/100), "
                    f"with {int(top_topic['Questions'])} question(s) across "
                    f"{int(top_topic['Papers Appearing'])} paper(s)."
                )

                recurring = mapped_topics[mapped_topics["Papers Appearing"] >= 2]
                if not recurring.empty:
                    recurring_names = ", ".join(
                        recurring.sort_values(
                            ["Papers Appearing", "Questions"], ascending=False
                        )["Topic"].head(5).tolist()
                    )
                    st.success(
                        "**Recurring topics:** These topics appear in at least "
                        f"two uploaded papers: {recurring_names}."
                    )
                else:
                    st.write(
                        "No topic appears in two or more uploaded papers yet. "
                        "Upload more past papers to reveal recurring patterns."
                    )
            else:
                st.warning(
                    "No questions have been assigned to the entered topics. "
                    "Check the uploaded papers and topic list before relying on the rankings."
                )

            if unmapped_topics:
                st.warning(
                    "**Not found in uploaded papers:** "
                    + ", ".join(unmapped_topics[:10])
                    + ("…" if len(unmapped_topics) > 10 else "")
                    + ". This only means no question was assigned to these topics "
                    "in the papers analyzed; it does not mean the topics are unimportant."
                )

        st.divider()
        left, right = st.columns(2)

        with left:
            st.markdown("<div class='section-kicker'>QUESTION-LEVEL SIGNAL</div>", unsafe_allow_html=True)
            st.subheader("📊 Estimated Topic Difficulty")
            difficulty_counts = (
                topics["Difficulty (estimated)"]
                .value_counts()
                .reindex(
                    ["Easy", "Medium", "Hard", "Insufficient data"],
                    fill_value=0
                )
            )
            st.bar_chart(difficulty_counts)
            st.caption(
                "Difficulty is estimated from extracted questions. Topics without "
                "enough assigned questions are marked as insufficient data."
            )

        with right:
            st.markdown("<div class='section-kicker'>REPEATED TOPIC SIGNAL</div>", unsafe_allow_html=True)
            st.subheader("📚 Questions by Topic")
            frequency_chart = topics[topics["Questions"] > 0].set_index("Topic")[["Questions"]]
            if not frequency_chart.empty:
                st.bar_chart(frequency_chart, horizontal=True)
            else:
                st.info("Topic frequency will appear after questions are assigned to topics.")

        st.divider()
        st.markdown("<div class='section-kicker'>WHAT HAS SHOWN UP SO FAR?</div>", unsafe_allow_html=True)
        st.subheader("🧭 Syllabus Coverage Insights")
        coverage_df = topics[[
            "Topic", "Questions", "Papers Appearing", "Paper Coverage %"
        ]].copy()
        coverage_df["Coverage status"] = coverage_df["Questions"].apply(
            lambda count: (
                "Found in uploaded papers" if int(count) > 0
                else "Not found in uploaded papers"
            )
        )
        coverage_df = coverage_df.rename(columns={
            "Questions": "Questions found",
            "Papers Appearing": "Papers found in",
            "Paper Coverage %": "Paper coverage (%)"
        })
        st.dataframe(coverage_df, use_container_width=True, hide_index=True)
        st.caption(
            "Coverage reflects only the papers you uploaded and the current topic assignments. "
            "A topic not found here may still be important for the exam."
        )

        st.divider()
        st.markdown("<div class='section-kicker'>PATTERNS OVER TIME</div>", unsafe_allow_html=True)
        st.subheader("📅 Year-wise Topic Trends")
        if "Year" in questions.columns and questions["Year"].notna().any():
            year_data = questions.dropna(subset=["Year"]).copy()
            year_data["Year"] = year_data["Year"].astype(int)
            trend = year_data.groupby(
                ["Year", "Matched Topic"]
            ).size().unstack(fill_value=0)
            st.line_chart(trend)
        else:
            st.info(
                "Year-wise trends appear when a year is detected from the uploaded "
                "paper filename, e.g. Maths_2024.pdf."
            )


# =========================================================
# DASHBOARD 3: STUDY PLANNER
# =========================================================

elif active_dashboard == "Study Planner":
    st.header("📅 Smart Study Planner")

    analyses = st.session_state.analysis

    if not analyses:
        st.info("Analyze your subjects first.")
    else:
        timetable = build_timetable(analyses, daily_hours)

        if timetable.empty:
            st.warning(
                "No future study tasks found. Check the exam dates "
                "and topic mappings."
            )
        else:
            today = date.today()
            task_ids = timetable["Task ID"].tolist()

            done = sum(
                task_id in st.session_state.completed_tasks
                for task_id in task_ids
            )

            progress = round(
                done / len(task_ids) * 100
            )

            nearest_exam = min(
                s["exam_date"] for s in analyses
                if s["exam_date"] >= today
            ) if any(
                s["exam_date"] >= today for s in analyses
            ) else None

            days_left = (
                max((nearest_exam - today).days, 0)
                if nearest_exam else 0
            )

            a, b, c = st.columns(3)
            a.metric("Total tasks", len(task_ids))
            b.metric("Completed tasks", done)
            c.metric("Overall progress", f"{progress}%")

            st.progress(progress / 100)

            st.metric("Days to nearest exam", days_left)

            st.subheader("Your Daily Checklist")

            for plan_day in sorted(timetable["Date"].unique()):
                day_tasks = timetable[
                    timetable["Date"] == plan_day
                ]

                day_ids = day_tasks["Task ID"].tolist()

                completed_today = sum(
                    task_id in st.session_state.completed_tasks
                    for task_id in day_ids
                )

                with st.container(border=True):
                    st.markdown(
                        f"### {plan_day.strftime('%A, %d %B %Y')}"
                    )

                    st.caption(
                        f"{completed_today}/{len(day_ids)} tasks completed"
                    )

                    for _, task in day_tasks.iterrows():
                        task_id = task["Task ID"]

                        label = (
                            f"{task['Subject']} — {task['Topic']} "
                            f"({task['Minutes']} min; {task['Kind']})"
                        )

                        widget_key = f"task_{task_id}"

                        # Initialize each checkbox from the saved task set.
                        if widget_key not in st.session_state:
                            st.session_state[widget_key] = (
                                task_id in st.session_state.completed_tasks
                            )

                        st.checkbox(
                            label,
                            key=widget_key,
                            on_change=sync_task_completion,
                            args=(task_id,)
                        )

                    current_done = sum(
                        task_id in st.session_state.completed_tasks
                        for task_id in day_ids
                    )

                    st.progress(
                        current_done / len(day_ids)
                        if day_ids else 0
                    )

            st.subheader("Preparation Progress")

            latest_done = sum(
                task_id in st.session_state.completed_tasks
                for task_id in task_ids
            )

            latest_percent = round(
                latest_done / len(task_ids) * 100
            )

            st.metric("Study plan completed", f"{latest_percent}%")
            st.progress(latest_percent / 100)

            if latest_percent == 100:
                st.success(
                    "All scheduled tasks are complete. "
                    "Use your remaining time for a calm final review."
                )
            elif latest_percent >= 60:
                st.success("Good progress. Keep completing your daily goals.")
            else:
                st.info(
                    "Start with the earliest exam and the highest-priority topics."
                )

            export = timetable.copy()
            export["Completed"] = export["Task ID"].apply(
                lambda x: x in st.session_state.completed_tasks
            )
            export["Date"] = export["Date"].apply(
                lambda x: x.strftime("%d %b %Y")
            )
            export["Exam Date"] = export["Exam Date"].apply(
                lambda x: x.strftime("%d %b %Y")
            )

            st.download_button(
                "📥 Download timetable CSV",
                data=export.to_csv(index=False).encode("utf-8"),
                file_name="study_timetable.csv",
                mime="text/csv"
            )

st.divider()
st.caption(
    "Exam Intelligence Assistant | Free text analysis | "
    "Always verify low-confidence topic matches."
)


