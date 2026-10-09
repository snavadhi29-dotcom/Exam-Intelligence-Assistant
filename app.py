
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;450;500;600;700&family=Libre+Baskerville:wght@400;700&display=swap" rel="stylesheet">
<style>
:root {
 --background:#f5f7fb; --surface:#ffffff; --ink:#20283f;
 --muted:#64708a; --line:#e1e6ef; --primary:#4263eb;
 --primary-hover:#3451cf; --teal:#0b8f8a; --purple:#7957c6;
 --midnight:#11192e; --sidebar-text:#e6ecfa; --sidebar-muted:#aab7d2;
 --tint:#eef2ff; --success:#e9f7f4; --shadow:0 4px 16px #18264708;
 --font-body:'IBM Plex Sans',sans-serif;
 --font-heading:'Libre Baskerville',Georgia,serif;
}
.stApp {background:var(--background);color:var(--ink)}
html,body,[class*="css"],.stApp,p,label,input,textarea,button {font-family:var(--font-body)}
.block-container {max-width:1360px;padding:2.4rem 3rem 3rem}
h1,h2,h3,h4 {color:var(--ink);letter-spacing:0!important}
h1,h2 {font-family:var(--font-heading)!important}
h1 {font-size:2rem!important;line-height:1.4!important;font-weight:400!important}
h2 {font-size:1.5rem!important;font-weight:400!important}
h3 {font-family:var(--font-body)!important;font-size:1.08rem!important;font-weight:600!important}
p,label,.stCaption {color:var(--muted)}
[data-testid="stHeader"] {background:var(--background)}
section[data-testid="stSidebar"] {background:var(--midnight);border-right:1px solid var(--line)}
section[data-testid="stSidebar"]>div {padding-top:1.8rem}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] span {color:var(--sidebar-text)!important}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:var(--sidebar-muted)!important}
.sidebar-brand {display:flex;gap:12px;align-items:center;margin-bottom:32px}
.brand-mark {display:grid;place-items:center;width:40px;height:40px;background:var(--primary);color:var(--surface);border-radius:8px;font-weight:700;font-size:22px}
.brand-name {font-weight:600;font-size:18px;color:var(--sidebar-text);line-height:1.4}
.brand-name small {display:block;font-size:11px;font-weight:400;color:var(--sidebar-muted)}
.sidebar-label {font-size:11px;font-weight:600;color:var(--sidebar-muted);margin:24px 0 10px}
section[data-testid="stSidebar"] [role="radiogroup"] {gap:6px}
section[data-testid="stSidebar"] [role="radiogroup"] label {width:100%;border:1px solid transparent;padding:12px 14px;border-radius:6px;transition:background .16s}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:color-mix(in srgb,var(--primary) 15%,var(--midnight))}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:color-mix(in srgb,var(--primary) 23%,var(--midnight));border-color:color-mix(in srgb,var(--primary) 40%,var(--midnight))}
.workspace-top {display:flex;justify-content:space-between;align-items:center;gap:12px;border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:28px;color:var(--muted);font-size:12px}
.workspace-status {display:flex;align-items:center;gap:8px;white-space:nowrap}
.status-dot {width:6px;height:6px;border-radius:50%;background:var(--teal)}
.hero {padding:0 0 22px;margin:0;background:none;border-radius:0;box-shadow:none}
.hero h1 {margin:8px 0!important;color:var(--ink)!important}
.hero p {max-width:720px;font-size:14px;color:var(--muted);line-height:1.7;margin:0}
.eyebrow {color:var(--teal);font-size:11px;font-weight:600}
.section-label {font-size:11px;font-weight:600;color:var(--primary);margin:20px 0 8px}
.summary-strip {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:20px 0;margin:4px 0 24px}
.summary-item {padding-left:16px;border-left:3px solid var(--primary)}
.summary-item:nth-child(2) {border-color:var(--teal)}
.summary-item:nth-child(3) {border-color:var(--purple)}
.summary-item span {display:block;font-size:12px;color:var(--muted);margin-bottom:4px}
.summary-item strong {font-weight:600;font-size:24px;color:var(--ink)}
[data-testid="stMetric"] {background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:18px 20px;box-shadow:var(--shadow);min-height:112px}
[data-testid="stMetric"] label {font-size:12px!important;font-weight:500!important}
[data-testid="stMetricValue"] {font-family:var(--font-body);font-size:28px;font-weight:600;color:var(--ink)}
[data-testid="stVerticalBlockBorderWrapper"]>div {background:var(--surface);border-radius:8px}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:8px!important;border-color:var(--line)!important}
.stTextInput input,.stTextArea textarea,.stDateInput input {color:var(--ink)!important;line-height:1.6}
.stTextInput>div>div,.stDateInput>div>div,.stNumberInput>div>div,.stSelectbox>div>div,.stTextArea textarea {background:var(--surface)!important;border-color:var(--line)!important;border-radius:6px!important}
.stTextInput>div>div:focus-within,.stTextArea textarea:focus {border-color:var(--primary)!important;box-shadow:0 0 0 3px var(--tint)!important}
.stButton>button,.stDownloadButton>button {min-height:42px;border-radius:6px!important;font-size:13px;font-weight:600;border:1px solid var(--line);background:var(--surface);color:var(--ink);transition:background .16s,border-color .16s}
.stButton>button p,.stDownloadButton>button p {color:inherit!important}
.stButton>button:hover,.stDownloadButton>button:hover {border-color:var(--primary);color:var(--primary);background:var(--tint)}
.stButton>button[kind="primary"] {background:var(--primary);border-color:var(--primary);color:var(--surface)}
.stButton>button[kind="primary"]:hover {background:var(--primary-hover);color:var(--surface)}
[data-testid="stFileUploaderDropzone"] {background:var(--background);border:1px dashed var(--line);border-radius:6px;padding:24px 16px}
[data-testid="stFileUploaderDropzone"] button {background:var(--surface);border:1px solid var(--line);border-radius:6px}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:8px;overflow:hidden}
[data-testid="stAlert"] {border-radius:6px;border:1px solid var(--line)}
[data-testid="stProgress"]>div>div>div>div {background:var(--teal)!important;border-radius:4px}
[data-testid="stCheckbox"] {padding:6px 0}
[data-testid="stCheckbox"] label p {color:var(--ink)}
hr {border-color:var(--line)!important;margin:24px 0}
.empty-workspace {text-align:center;padding:40px 20px 32px;border-top:1px solid var(--line)}
.empty-symbol {display:inline-grid;place-items:center;width:52px;height:52px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--primary);font-size:26px;margin-bottom:12px}
.empty-workspace h3 {margin:8px 0!important}
.empty-workspace p {font-size:13px}
.footer-note {border-top:1px solid var(--line);margin-top:32px;padding-top:18px;display:flex;justify-content:space-between;gap:16px;font-size:11px;color:var(--muted)}
@media(max-width:900px) {.block-container{padding:1.8rem 1.3rem}h1{font-size:1.6rem!important}.summary-strip{gap:12px}.summary-item{padding-left:10px}.workspace-status{display:none}}
@media(max-width:500px) {.summary-strip{grid-template-columns:1fr;gap:16px}.footer-note{flex-direction:column}.workspace-top{font-size:11px}}
@media(prefers-reduced-motion:reduce) {*{transition:none!important;animation:none!important}}
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

        active_topics = topics[topics["Questions"] > 0]

        for _, row in active_topics.iterrows():
            priority = row["Priority"]

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
                "Score": float(row["Priority Score"]),
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
# WORKSPACE SHELL AND NAVIGATION
# =========================================================

# Apply automatic navigation before the radio widget is instantiated.
if "pending_dashboard" in st.session_state:
    st.session_state.active_dashboard = st.session_state.pop("pending_dashboard")

dashboard_names = ["Subjects & Setup", "Exam Intelligence", "Study Planner"]
if st.session_state.active_dashboard not in dashboard_names:
    st.session_state.active_dashboard = dashboard_names[0]

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand"><div class="brand-mark">E</div>
    <div class="brand-name">Exam Intelligence<small>YOUR STUDY WORKSPACE</small></div></div>
    <div class="sidebar-label">WORKSPACE</div>
    """, unsafe_allow_html=True)
    active_dashboard = st.radio(
        "Navigate dashboard", dashboard_names,
        key="active_dashboard", label_visibility="collapsed"
    )
    st.divider()
    st.markdown('<div class="sidebar-label">DAILY STUDY TARGET</div>', unsafe_allow_html=True)
    daily_hours = st.slider("Available study hours per day", 1, 12, 4)
    st.caption(f"{daily_hours} hours available each day")
    st.divider()
    st.caption("Local text analysis · No AI API required")
    st.caption("Your workspace lasts for this browser session.")

st.markdown("""
<div class="workspace-top"><span>WORKSPACE / EXAM INTELLIGENCE ASSISTANT</span>
<span class="workspace-status"><span class="status-dot"></span>Local analysis</span></div>
<div class="hero">
  <div class="eyebrow">A LITTLE CLARITY. A BETTER STUDY PLAN.</div>
  <h1>Exam Intelligence Assistant</h1>
  <p>Your subjects, question patterns, and study priorities — together in one workspace.</p>
</div>
""", unsafe_allow_html=True)

subject_count = len(st.session_state.subjects)
paper_count = sum(len(st.session_state.get(f"files_{s['id']}", []) or []) for s in st.session_state.subjects)
question_count = sum(len(s["questions"]) for s in st.session_state.analysis)
st.markdown(f"""
<div class="summary-strip">
<div class="summary-item"><span>Subjects in workspace</span><strong>{subject_count:02d}</strong></div>
<div class="summary-item"><span>Papers uploaded</span><strong>{paper_count:02d}</strong></div>
<div class="summary-item"><span>Questions analyzed</span><strong>{question_count:02d}</strong></div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# DASHBOARD 1: SUBJECTS & SETUP
# =========================================================

if active_dashboard == "Subjects & Setup":
    st.markdown("<div class='section-label'>WORKSPACE / SETUP</div>", unsafe_allow_html=True)
    st.header("Subjects & Question Papers")
    st.write(
        "Create each subject separately. Enter one syllabus topic per line "
        "and upload all relevant previous-year papers."
    )

    if st.button("Add subject", type="primary", icon=":material/add:"):
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
        st.markdown("""
        <div class="empty-workspace"><div class="empty-symbol">＋</div>
        <h3>Your workspace starts here</h3>
        <p>No subjects or question papers yet.</p></div>
        """, unsafe_allow_html=True)

    for index, subject in enumerate(st.session_state.subjects):
        sid = subject["id"]

        with st.container(border=True):
            st.subheader(f"{index + 1:02d} / Subject details")

            c1, c2 = st.columns([2, 1])

            with c1:
                name = st.text_input(
                    "Subject name",
                    value=subject["name"],
                    key=f"name_{sid}",
                    placeholder="e.g. Engineering Mathematics"
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
                height=160,
                placeholder="Successive differentiation\nTaylor and Maclaurin series\nMatrices and eigenvalues",
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
                if st.button("Save subject", key=f"save_{sid}", icon=":material/check:"):
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
                if st.button("Remove subject", key=f"remove_{sid}", icon=":material/delete:"):
                    st.session_state.subjects = [
                        s for s in st.session_state.subjects
                        if s["id"] != sid
                    ]
                    st.session_state.analysis = []
                    st.session_state.completed_tasks = set()
                    st.rerun()

    st.divider()

    if st.button(
        "Analyze all subjects",
        icon=":material/analytics:",
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
                "topics": topic_df
            })

        for error in errors:
            st.warning(error)

        if analyses:
            st.session_state.analysis = analyses

            # Automatically switch to the analysis dashboard.
            st.session_state.pending_dashboard = "Exam Intelligence"
            st.rerun()


# =========================================================
# DASHBOARD 2: EXAM INTELLIGENCE
# =========================================================

elif active_dashboard == "Exam Intelligence":
    st.markdown("<div class='section-label'>WORKSPACE / ANALYTICS</div>", unsafe_allow_html=True)
    st.header("Exam Intelligence")
    st.caption("A clear view of question patterns, topic frequency, estimated difficulty, and revision priorities.")

    analyses = st.session_state.analysis

    if not analyses:
        st.info("Add your subjects and analyze them first.")
    else:
        total_questions = sum(
            len(s["questions"]) for s in analyses
        )

        high_topics = sum(
            int((s["topics"]["Priority"] == "High").sum())
            for s in analyses
        )

        low_confidence = sum(
            int(
                s["questions"]["Match Status"].isin(
                    ["Review suggested", "Low confidence — verify"]
                ).sum()
            )
            for s in analyses
        )

        a, b, c = st.columns(3)
        a.metric("Subjects analyzed", len(analyses))
        b.metric("Questions extracted", total_questions)
        c.metric("Matches to review", low_confidence)

        selected = st.selectbox(
            "Select subject",
            range(len(analyses)),
            format_func=lambda i: analyses[i]["name"]
        )

        subject = analyses[selected]
        questions = subject["questions"]
        topics = subject["topics"]

        st.subheader(subject["name"])

        x, y, z = st.columns(3)
        x.metric("Exam date", subject["exam_date"].strftime("%d %b %Y"))
        y.metric(
            "Days remaining",
            max((subject["exam_date"] - date.today()).days, 0)
        )
        z.metric("Papers analyzed", questions["Paper"].nunique())

        st.subheader("Topic frequency")
        chart = topics[topics["Questions"] > 0].set_index("Topic")[
            ["Questions"]
        ]

        if not chart.empty:
            st.bar_chart(chart, horizontal=True, color="#4263eb")
        else:
            st.warning("No topics have been matched yet.")

        st.subheader("Topic Priority and Weightage")

        st.dataframe(
            topics,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Question Type Pattern")
        st.bar_chart(questions["Question Type"].value_counts(), color="#0b8f8a")

        st.subheader("Difficulty Distribution")
        st.bar_chart(
            questions["Difficulty (estimated)"].value_counts(), color="#7957c6"
        )

        st.subheader("Question Mapping")

        status_filter = st.selectbox(
            "Filter mapping results",
            [
                "All questions",
                "Strong match",
                "Review suggested",
                "Low confidence — verify"
            ]
        )

        shown = questions.copy()

        if status_filter != "All questions":
            shown = shown[
                shown["Match Status"] == status_filter
            ]

        st.dataframe(
            shown[
                [
                    "Paper", "Year", "Question No.", "Question",
                    "Matched Topic", "Confidence %", "Match Status",
                    "Score Gap %", "Question Type",
                    "Difficulty (estimated)", "Marks (if detected)"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Export question mapping",
            icon=":material/download:",
            data=questions.to_csv(index=False).encode("utf-8"),
            file_name="question_mapping.csv",
            mime="text/csv"
        )

        st.caption(
            "Confidence is a similarity score, not a probability. "
            "Review uncertain matches before relying on topic priorities."
        )

        st.subheader("Year-wise Trends")
        year_data = questions.dropna(subset=["Year"]).copy()

        if not year_data.empty:
            year_data["Year"] = year_data["Year"].astype(int)

            trend = year_data.groupby(
                ["Year", "Matched Topic"]
            ).size().unstack(fill_value=0)

            st.line_chart(trend)
        else:
            st.info(
                "Include a year in each filename, e.g. Maths_2024.pdf, "
                "to enable year-wise trends."
            )


# =========================================================
# DASHBOARD 3: STUDY PLANNER
# =========================================================

elif active_dashboard == "Study Planner":
    st.markdown("<div class='section-label'>WORKSPACE / EXECUTION</div>", unsafe_allow_html=True)
    st.header("Smart Study Planner")
    st.caption("Turn your topic priorities into daily actions and track your progress as you go.")

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

                        checked = st.checkbox(
                            label,
                            value=(
                                task_id in st.session_state.completed_tasks
                            ),
                            key=f"task_{task_id}"
                        )

                        if checked:
                            st.session_state.completed_tasks.add(task_id)
                        else:
                            st.session_state.completed_tasks.discard(task_id)

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
                "Export study timetable",
                icon=":material/download:",
                data=export.to_csv(index=False).encode("utf-8"),
                file_name="study_timetable.csv",
                mime="text/csv"
            )

st.markdown("""
<div class="footer-note"><span>Exam Intelligence Assistant</span>
<span>Similarity scores are estimates. Verify uncertain matches.</span></div>
""", unsafe_allow_html=True)
