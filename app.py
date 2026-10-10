import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
import hashlib
from io import BytesIO
from datetime import date, timedelta
from difflib import SequenceMatcher
from collections import Counter

# Safe import for PdfReader
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

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

# Enhanced Styling: Palette-driven Highlighted Heading Boxes & Subtle Accents
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;450;500;600;700&family=Libre+Baskerville:wght@400;700&display=swap" rel="stylesheet">
<style>
:root {
  --background: #f5f7fb;
  --surface: #ffffff;
  --ink: #1e263d;
  --muted: #5e6b85;
  --line: #dfe5f0;
  --primary: #4263eb;
  --primary-hover: #3451cf;
  --primary-tint: #edf2ff;
  --primary-border: #c7d6fc;
  --teal: #0b8f8a;
  --teal-tint: #e6f7f6;
  --teal-border: #a4e5e2;
  --purple: #7957c6;
  --purple-tint: #f3eefa;
  --purple-border: #d7c7f5;
  --pink: #d66ca6;
  --pink-tint: #fdf2f8;
  --pink-border: #f9cbe3;
  --amber: #e8ac45;
  --amber-tint: #fef8ed;
  --amber-border: #fce1b1;
  --midnight: #11192e;
  --sidebar-text: #e6ecfa;
  --sidebar-muted: #aab7d2;
  --grid: #edf0f6;
  --shadow-sm: 0 2px 8px rgba(24, 38, 71, 0.05);
  --shadow-md: 0 4px 16px rgba(24, 38, 71, 0.08);
  --font-body: 'IBM Plex Sans', sans-serif;
  --font-heading: 'Libre Baskerville', Georgia, serif;
}

.stApp {
  background: var(--background);
  color: var(--ink);
}

html, body, [class*="css"], .stApp, p, label, input, textarea, button {
  font-family: var(--font-body);
}

.block-container {
  max-width: 1360px;
  padding: 2.2rem 3rem 3rem;
}

/* =========================================================
   HIGHLIGHTED HEADING BOXES (PALETTE-DRIVEN)
   Automatically styles ALL Streamlit headers & subheaders
   ========================================================= */

/* H1 Main Heading */
[data-testid="stHeadingWithAction"] h1,
[data-testid="stHeading"] h1,
.stMarkdown h1,
h1 {
  font-family: var(--font-heading) !important;
  font-size: 1.95rem !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em !important;
  line-height: 1.35 !important;
  margin: 0.4rem 0 0.8rem !important;
}

/* H2 Section Headings - In prominent palette color box */
[data-testid="stHeadingWithAction"] h2,
[data-testid="stHeading"] h2,
.stMarkdown h2,
h2 {
  font-family: var(--font-heading) !important;
  font-size: 1.35rem !important;
  font-weight: 700 !important;
  color: #1a329c !important;
  background: linear-gradient(90deg, #edf2ff 0%, #f4f7ff 100%) !important;
  border: 1px solid #c7d6fc !important;
  border-left: 5px solid #4263eb !important;
  padding: 10px 18px !important;
  border-radius: 8px !important;
  margin: 1.6rem 0 0.9rem !important;
  box-shadow: 0 2px 6px rgba(66, 99, 235, 0.08) !important;
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
}

/* H3 Subsection Headings - In highlighted teal palette badge box */
[data-testid="stHeadingWithAction"] h3,
[data-testid="stHeading"] h3,
.stMarkdown h3,
h3 {
  font-family: var(--font-body) !important;
  font-size: 1.05rem !important;
  font-weight: 600 !important;
  color: #075f5c !important;
  background: #e6f7f6 !important;
  border: 1px solid #a4e5e2 !important;
  border-left: 4px solid #0b8f8a !important;
  padding: 7px 14px !important;
  border-radius: 6px !important;
  margin: 1.2rem 0 0.7rem !important;
  display: inline-flex !important;
  align-items: center !important;
  box-shadow: 0 1px 3px rgba(11, 143, 138, 0.06) !important;
}

/* Custom Highlight Boxes for Key Headers */
.heading-box-primary {
  display: flex;
  align-items: center;
  gap: 10px;
  background: #edf2ff;
  border: 1px solid #c7d6fc;
  border-left: 5px solid #4263eb;
  padding: 12px 18px;
  border-radius: 8px;
  margin: 1.4rem 0 0.9rem;
  box-shadow: 0 2px 6px rgba(66, 99, 235, 0.07);
}
.heading-box-primary h2, .heading-box-primary span {
  margin: 0 !important;
  padding: 0 !important;
  background: none !important;
  border: none !important;
  box-shadow: none !important;
  color: #1a329c !important;
  font-size: 1.25rem !important;
  font-weight: 700 !important;
}

.heading-box-teal {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: #e6f7f6;
  border: 1px solid #a4e5e2;
  border-left: 4px solid #0b8f8a;
  padding: 7px 14px;
  border-radius: 6px;
  margin: 1rem 0 0.6rem;
  font-size: 0.98rem;
  font-weight: 600;
  color: #075f5c;
  box-shadow: 0 1px 3px rgba(11, 143, 138, 0.06);
}

.heading-box-purple {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: #f3eefa;
  border: 1px solid #d7c7f5;
  border-left: 4px solid #7957c6;
  padding: 7px 14px;
  border-radius: 6px;
  margin: 1rem 0 0.6rem;
  font-size: 0.98rem;
  font-weight: 600;
  color: #4c3285;
  box-shadow: 0 1px 3px rgba(121, 87, 198, 0.06);
}

.heading-box-amber {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: #fef8ed;
  border: 1px solid #fce1b1;
  border-left: 4px solid #e8ac45;
  padding: 7px 14px;
  border-radius: 6px;
  margin: 1rem 0 0.6rem;
  font-size: 0.98rem;
  font-weight: 600;
  color: #925f0a;
  box-shadow: 0 1px 3px rgba(232, 172, 69, 0.06);
}

p, label, .stCaption {
  color: var(--muted);
}

/* Sidebar styling */
[data-testid="stHeader"] {
  background: var(--background);
}

section[data-testid="stSidebar"] {
  background: var(--midnight);
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}

section[data-testid="stSidebar"] > div {
  padding-top: 1.8rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
  color: var(--sidebar-text) !important;
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  color: var(--sidebar-muted) !important;
}

.sidebar-brand {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 26px;
}

.brand-mark {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  background: linear-gradient(135deg, var(--primary) 0%, #3451cf 100%);
  color: #ffffff;
  border-radius: 10px;
  font-weight: 700;
  font-size: 22px;
  box-shadow: 0 4px 12px rgba(66, 99, 235, 0.35);
}

.brand-name {
  font-weight: 600;
  font-size: 18px;
  color: var(--sidebar-text);
  line-height: 1.35;
}

.brand-name small {
  display: block;
  font-size: 11px;
  font-weight: 400;
  color: var(--sidebar-muted);
  letter-spacing: 0.05em;
}

.sidebar-label {
  font-size: 11px;
  font-weight: 700;
  color: var(--sidebar-muted);
  letter-spacing: 0.08em;
  margin: 20px 0 10px;
  text-transform: uppercase;
}

section[data-testid="stSidebar"] [role="radiogroup"] {
  gap: 8px;
}

section[data-testid="stSidebar"] [role="radiogroup"] label {
  width: 100%;
  border: 1px solid transparent;
  padding: 11px 14px;
  border-radius: 8px;
  transition: all 0.16s ease;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(66, 99, 235, 0.16);
}

section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: rgba(66, 99, 235, 0.28);
  border-color: rgba(66, 99, 235, 0.6);
  box-shadow: inset 3px 0 0 var(--primary);
}

/* Top bar and Hero */
.workspace-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid var(--line);
  padding-bottom: 16px;
  margin-bottom: 24px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 500;
}

.workspace-status {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--teal-tint);
  border: 1px solid var(--teal-border);
  color: #075f5c;
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--teal);
  box-shadow: 0 0 6px rgba(11, 143, 138, 0.5);
}

.hero {
  padding: 4px 0 20px;
  margin: 0;
}

.hero h1 {
  margin: 8px 0 10px !important;
}

.hero p {
  max-width: 720px;
  font-size: 14.5px;
  color: var(--muted);
  line-height: 1.65;
  margin: 0;
}

.eyebrow {
  display: inline-block;
  color: var(--teal);
  background: var(--teal-tint);
  border: 1px solid var(--teal-border);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  padding: 3px 10px;
  border-radius: 4px;
}

.section-label {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--primary);
  background: var(--primary-tint);
  border: 1px solid var(--primary-border);
  padding: 4px 10px;
  border-radius: 4px;
  margin-bottom: 6px;
}

/* Metric summary strip with subtle rich colors */
.summary-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  padding: 16px 0;
  margin: 4px 0 24px;
}

.summary-item {
  padding: 12px 18px;
  background: var(--surface);
  border-radius: 8px;
  border: 1px solid var(--line);
  border-left: 4px solid var(--primary);
  box-shadow: var(--shadow-sm);
}

.summary-item:nth-child(2) {
  border-left-color: var(--teal);
}

.summary-item:nth-child(3) {
  border-left-color: var(--purple);
}

.summary-item span {
  display: block;
  font-size: 12px;
  font-weight: 500;
  color: var(--muted);
  margin-bottom: 4px;
}

.summary-item strong {
  font-weight: 700;
  font-size: 26px;
  color: var(--ink);
  font-family: var(--font-body);
}

/* Streamlit Native Components Enhanced */
[data-testid="stPlotlyChart"] {
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--surface);
  box-shadow: var(--shadow-sm);
  overflow: hidden;
}

[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-top: 3px solid var(--primary);
  border-radius: 8px;
  padding: 16px 20px;
  box-shadow: var(--shadow-sm);
}

[data-testid="stMetric"] label {
  font-size: 12px !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
  font-size: 28px !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
  background: var(--surface);
  border-radius: 8px;
  padding: 18px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 10px !important;
  border-color: var(--line) !important;
  box-shadow: var(--shadow-sm);
}

.stTextInput input, .stTextArea textarea, .stDateInput input {
  color: var(--ink) !important;
  border-radius: 6px !important;
}

.stTextInput > div > div,
.stDateInput > div > div,
.stNumberInput > div > div,
.stSelectbox > div > div,
.stTextArea textarea {
  background: var(--surface) !important;
  border-color: var(--line) !important;
  border-radius: 6px !important;
}

.stTextInput > div > div:focus-within,
.stTextArea textarea:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px var(--primary-tint) !important;
}

.stButton > button, .stDownloadButton > button {
  min-height: 42px;
  border-radius: 8px !important;
  font-size: 13.5px;
  font-weight: 600;
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--ink);
  transition: all 0.16s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--primary);
  color: var(--primary);
  background: var(--primary-tint);
}

.stButton > button[kind="primary"] {
  background: var(--primary) !important;
  border-color: var(--primary) !important;
  color: #ffffff !important;
  box-shadow: 0 2px 8px rgba(66, 99, 235, 0.25);
}

.stButton > button[kind="primary"]:hover {
  background: var(--primary-hover) !important;
  border-color: var(--primary-hover) !important;
}

[data-testid="stFileUploaderDropzone"] {
  background: #fbfcfe;
  border: 1.5px dashed var(--line);
  border-radius: 8px;
  padding: 24px 16px;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}

[data-testid="stProgress"] > div > div > div > div {
  background: linear-gradient(90deg, var(--teal) 0%, #20b8b2 100%) !important;
  border-radius: 4px;
}

.footer-note {
  border-top: 1px solid var(--line);
  margin-top: 36px;
  padding-top: 18px;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  font-size: 12px;
  color: var(--muted);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SAMPLE DATA SEEDING (Ensures instant prototype display)
# =========================================================

SAMPLE_RAW_SUBJECTS = [
    {
        "id": 1,
        "name": "Engineering Mathematics",
        "exam_date": date.today() + timedelta(days=7),
        "topics": """Successive differentiation and Leibnitz theorem
Taylor and Maclaurin series expansion
Asymptotes and curve tracing
Curvature and radius of curvature
Matrices and eigenvalues
Beta and Gamma functions
Partial differentiation and Euler theorem""",
        "demo_papers": [
            {
                "name": "Mathematics_2023.pdf",
                "year": 2023,
                "text": """
1. State and prove Leibnitz theorem for successive differentiation of the product of two functions. [7 marks]
2. Find the nth derivative of y = (x^2 + 1) * sin(2x) using Leibnitz rule. [7 marks]
3. Expand f(x) = log(1 + x) up to fourth degree terms using Maclaurin series. [6 marks]
4. Obtain the Taylor series expansion of f(x, y) = e^x * cos(y) in powers of (x - 1) and (y - pi/2). [8 marks]
5. Find all the asymptotes of the algebraic curve x^3 + 2x^2y - xy^2 - 2y^3 + 4x^2 - y^2 + 1 = 0. [8 marks]
6. Find the radius of curvature for the cycloid x = a(theta - sin(theta)), y = a(1 - cos(theta)) at theta = pi. [6 marks]
7. Determine the eigenvalues and corresponding eigenvectors for the matrix A = [[2, 1, 1], [1, 2, 1], [0, 0, 1]]. [10 marks]
8. Evaluate the definite integral using Beta and Gamma functions: integral from 0 to pi/2 of sin^5(x) * cos^4(x) dx. [6 marks]
"""
            },
            {
                "name": "Mathematics_2024.pdf",
                "year": 2024,
                "text": """
1. If y = (sin^-1 x)^2, prove that (1 - x^2) y_(n+2) - (2n + 1)x y_(n+1) - n^2 y_n = 0 using Leibnitz theorem. [10 marks]
2. Using Taylor's theorem, calculate the approximate value of sqrt(26) correct to four decimal places. [5 marks]
3. Trace the cartesian curve y^2(a - x) = x^3 (Cissoid) showing all asymptotes and symmetry. [8 marks]
4. Show that the radius of curvature of the catenary y = c*cosh(x/c) is y^2/c. [7 marks]
5. Find the characteristic equation and verify Cayley-Hamilton theorem for matrix M = [[1, 2], [3, 4]]. Hence find M^-1. [8 marks]
6. Prove the relationship between Beta and Gamma functions: B(m, n) = (Gamma(m) * Gamma(n)) / Gamma(m + n). [8 marks]
7. Verify Euler's theorem for homogeneous function u = x^2 * y / (x + y). [6 marks]
"""
            }
        ]
    },
    {
        "id": 2,
        "name": "Basic Electrical Engineering",
        "exam_date": date.today() + timedelta(days=12),
        "topics": """DC Circuits and Kirchhoff laws
Thevenin and Norton theorems
Single phase AC series RLC resonance
Three phase balanced star delta connections
Single phase transformer equivalent circuit
DC Motor working principle and back EMF""",
        "demo_papers": [
            {
                "name": "Electrical_2024.pdf",
                "year": 2024,
                "text": """
1. State Kirchhoff's Current Law (KCL) and Kirchhoff's Voltage Law (KVL) with suitable circuit diagrams. [5 marks]
2. Calculate the load current passing through 10 ohm resistor using Thevenin theorem for the given bridge circuit. [8 marks]
3. Derive the expression for resonant frequency and quality factor Q for a series RLC AC circuit. [8 marks]
4. Explain the relationship between line and phase voltages and currents in a balanced star connected 3-phase load. [7 marks]
5. Draw and explain the approximate equivalent circuit of a single phase transformer referred to primary side. [8 marks]
6. Explain the working principle of a DC motor and derive the back EMF equation. What is its role? [8 marks]
"""
            }
        ]
    }
]


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
    base = normalize(topic)
    expanded = [base]
    for key, alternatives in SYNONYMS.items():
        if re.search(r"\b" + re.escape(key) + r"\w*\b", base):
            expanded.extend(alternatives)
        for alternative in alternatives:
            alt = normalize(alternative)
            if alt and re.search(r"\b" + re.escape(alt) + r"\b", base):
                expanded.append(key)
    return list(dict.fromkeys(expanded))

def stable_id(*parts):
    raw = "|".join(map(str, parts))
    return hashlib.sha256(raw.encode()).hexdigest()[:18]


# =========================================================
# PDF & TEXT EXTRACTION
# =========================================================

def extract_pdf(pdf_bytes):
    if PdfReader is None:
        return "", "pypdf not installed"
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

    return text, "OCR unavailable or no readable text"


# =========================================================
# QUESTION EXTRACTION
# =========================================================

def extract_questions(text):
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
    pattern = re.compile(
        r"(?im)^\s*(?:question\s*|q\s*\.?\s*)?(\d{1,2})\s*[\.\):]\s+"
    )
    matches = list(pattern.finditer(text))
    result = []

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        question = re.sub(r"\s+", " ", text[start:end]).strip()

        if len(question) < 18:
            continue
        if re.match(
            r"(?i)^(attempt any|answer all|time allowed|maximum marks|instructions)\b",
            question
        ):
            continue

        result.append({
            "number": match.group(1),
            "text": question
        })

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
# QUESTION-TO-TOPIC MAPPING
# =========================================================

def keyword_match_score(question, topic):
    q = normalize(question)
    q_tokens = set(tokenize(q))
    topic_variants = expand_topic(topic)
    best = 0.0

    for variant in topic_variants:
        t_tokens = set(tokenize(variant))
        if not t_tokens:
            continue

        phrase_score = 1.0 if variant in q else 0.0
        coverage = len(q_tokens & t_tokens) / len(t_tokens)
        partial_matches = sum(
            1 for term in t_tokens
            if any(qword.startswith(term) or term.startswith(qword) for qword in q_tokens if min(len(qword), len(term)) >= 4)
        )
        partial_coverage = partial_matches / len(t_tokens)
        score = max(phrase_score, 0.75 * coverage + 0.25 * partial_coverage)
        best = max(best, score)

    return best


def match_topics(questions, topics):
    if not questions or not topics:
        return []

    topic_variants = [" ".join(expand_topic(topic)) for topic in topics]
    questions_text = [q["text"] for q in questions]

    try:
        word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode"
        )
        combined_text = topic_variants + questions_text
        word_matrix = word_vectorizer.fit_transform(combined_text)
        topic_word_matrix = word_matrix[:len(topics)]
        question_word_matrix = word_matrix[len(topics):]
        word_similarity = cosine_similarity(question_word_matrix, topic_word_matrix)
    except ValueError:
        word_similarity = [[0.0] * len(topics) for _ in questions]

    try:
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
        )
        char_matrix = char_vectorizer.fit_transform(topic_variants + questions_text)
        char_similarity = cosine_similarity(char_matrix[len(topics):], char_matrix[:len(topics)])
    except ValueError:
        char_similarity = [[0.0] * len(topics) for _ in questions]

    output = []
    for i, question in enumerate(questions):
        scores = []
        for j, topic in enumerate(topics):
            lexical = keyword_match_score(question["text"], topic)
            word_score = float(word_similarity[i][j])
            char_score = float(char_similarity[i][j])
            score = 0.55 * lexical + 0.30 * word_score + 0.15 * char_score
            scores.append(score)

        order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
        best_index = order[0]
        best_score = scores[best_index]
        second_score = scores[order[1]] if len(order) > 1 else 0
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
            "Score Gap %": round(max(0, best_score - second_score) * 100, 1),
            "Question Type": classify_type(question["text"]),
            "Difficulty (estimated)": estimate_difficulty(question["text"]),
            "Marks (if detected)": extract_marks(question["text"])
        })

    return output


def classify_type(question):
    q = normalize(question)
    if any(x in q for x in ["derive", "prove that", "proof of", "deduce", "show that"]):
        return "Derivation"
    if any(x in q for x in ["calculate", "compute", "find the value", "evaluate", "numerical", "determine the current", "find the voltage", "solve for"]):
        return "Numerical"
    if any(x in q for x in ["explain", "describe", "discuss", "define", "differentiate between", "compare", "write a note", "state the"]):
        return "Theory"
    return "Other / Mixed"


def estimate_difficulty(question):
    q = normalize(question)
    hard_words = ["hence prove", "derive and prove", "multi-step", "optimize", "discuss in detail"]
    medium_words = ["derive", "calculate", "compare", "differentiate", "evaluate", "determine", "explain"]
    score = 2 * sum(w in q for w in hard_words) + sum(w in q for w in medium_words)
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

def analyze_subject(subject, files, demo_papers=None):
    topics = [x.strip() for x in subject["topics"].splitlines() if x.strip()]
    all_rows = []
    notices = []

    # Process uploaded files
    for pdf in (files or []):
        text, method = extract_pdf(pdf.getvalue())
        if not text.strip():
            notices.append(f"{pdf.name}: no readable text was extracted.")
            continue
        questions = extract_questions(text)
        if not questions:
            notices.append(f"{pdf.name}: no question boundaries were detected.")
            continue
        rows = match_topics(questions, topics)
        year = extract_year(pdf.name)
        for row in rows:
            row["Paper"] = pdf.name
            row["Year"] = year
            all_rows.append(row)
        notices.append(f"{pdf.name}: {len(rows)} questions detected ({method}).")

    # Process embedded demo papers
    for paper in (demo_papers or []):
        questions = extract_questions(paper["text"])
        rows = match_topics(questions, topics)
        for row in rows:
            row["Paper"] = paper["name"]
            row["Year"] = paper.get("year")
            all_rows.append(row)
        notices.append(f"{paper['name']}: {len(rows)} questions extracted.")

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

    topic_df["Frequency %"] = (topic_df["Questions"] / total_questions * 100).round(1)
    topic_df["Paper Coverage %"] = (topic_df["Papers Appearing"] / total_papers * 100).round(1)

    total_marks = topic_df["Marks Detected"].sum(min_count=1)
    if pd.notna(total_marks) and total_marks > 0:
        topic_df["Detected Marks %"] = (topic_df["Marks Detected"] / total_marks * 100).round(1)
    else:
        topic_df["Detected Marks %"] = None

    topic_df["Priority Score"] = (
        0.60 * topic_df["Frequency %"] + 0.40 * topic_df["Paper Coverage %"]
    ).round(1)

    def priority(row):
        if row["Questions"] == 0:
            return "No mapped questions"
        if row["Priority Score"] >= 35:
            return "High"
        if row["Priority Score"] >= 18:
            return "Medium"
        return "Low"

    topic_df["Priority"] = topic_df.apply(priority, axis=1)
    topic_df["Reason"] = topic_df.apply(
        lambda r: "No question mapped to this topic." if r["Questions"] == 0
        else f"{int(r['Questions'])} question(s) across {int(r['Papers Appearing'])} paper(s).",
        axis=1
    )

    topic_df = topic_df.sort_values(["Priority Score", "Questions"], ascending=False).reset_index(drop=True)
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
        if exam_date <= today:
            continue
        active_topics = topics[topics["Questions"] > 0]

        for _, row in active_topics.iterrows():
            priority = row["Priority"]
            minutes = 75 if priority == "High" else 50 if priority == "Medium" else 35
            tasks.append({
                "Subject": subject["name"],
                "Topic": row["Topic"],
                "Priority": priority,
                "Minutes": minutes,
                "Exam Date": exam_date,
                "Score": float(row["Priority Score"]),
                "Kind": "Study"
            })

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

    tasks.sort(key=lambda x: (x["Exam Date"], -x["Score"]))
    revision_tasks = [t for t in tasks if t["Kind"] == "Revision"]
    study_tasks = [t for t in tasks if t["Kind"] == "Study"]
    schedule = []
    capacity = daily_hours * 60

    for task in revision_tasks:
        revision_day = task["Exam Date"] - timedelta(days=1)
        schedule.append({**task, "Date": revision_day})

    for task in study_tasks:
        exam_date = task["Exam Date"]
        deadline = exam_date - timedelta(days=2)
        available_days = max((deadline - today).days + 1, 1)
        placed = False

        for offset in range(available_days):
            study_day = today + timedelta(days=offset)
            day_minutes = sum(x["Minutes"] for x in schedule if x["Date"] == study_day)
            if study_day < exam_date and day_minutes + task["Minutes"] <= capacity:
                schedule.append({**task, "Date": study_day})
                placed = True
                break

        if not placed:
            last_day = exam_date - timedelta(days=2)
            if last_day >= today:
                schedule.append({**task, "Date": last_day})

    df = pd.DataFrame(schedule)
    if df.empty:
        return df

    df["Task ID"] = df.apply(
        lambda r: stable_id(r["Date"], r["Subject"], r["Topic"], r["Kind"]),
        axis=1
    )
    return df.sort_values(["Date", "Exam Date", "Kind", "Subject"]).reset_index(drop=True)


# =========================================================
# SEED INITIAL DATA (Auto-populates so it matches the preview!)
# =========================================================

if "initialized" not in st.session_state:
    st.session_state.subjects = [
        {
            "id": s["id"],
            "name": s["name"],
            "exam_date": s["exam_date"],
            "topics": s["topics"],
            "demo_papers": s["demo_papers"]
        }
        for s in SAMPLE_RAW_SUBJECTS
    ]
    st.session_state.next_id = 3
    st.session_state.completed_tasks = set()
    st.session_state.active_dashboard = "Subjects & Setup"

    # Pre-run analysis so Exam Intelligence and Study Planner work immediately
    initial_analysis = []
    for s in st.session_state.subjects:
        q_df, t_df, _ = analyze_subject(s, files=[], demo_papers=s["demo_papers"])
        initial_analysis.append({
            "id": s["id"],
            "name": s["name"],
            "exam_date": s["exam_date"],
            "questions": q_df,
            "topics": t_df
        })
    st.session_state.analysis = initial_analysis
    st.session_state.initialized = True


# =========================================================
# VISUAL ANALYTICS (Rich Palette)
# =========================================================

CHART_THEME = {
    "ink": "#1e263d", "muted": "#5e6b85", "surface": "#ffffff",
    "grid": "#edf0f6", "blue": "#4263eb", "teal": "#0b8f8a",
    "purple": "#7957c6", "pink": "#d66ca6", "amber": "#e8ac45",
}
CHART_COLORS = [CHART_THEME[k] for k in ("blue", "teal", "purple", "pink", "amber")]
PRIORITY_COLORS = dict(zip(
    ["High", "Medium", "Low", "No mapped questions", "Revision"],
    [CHART_THEME["blue"], CHART_THEME["amber"], CHART_THEME["teal"],
     CHART_THEME["grid"], CHART_THEME["purple"]]
))

def style_figure(fig, height=310):
    fig.update_layout(
        template="plotly_white", height=height,
        paper_bgcolor=CHART_THEME["surface"], plot_bgcolor=CHART_THEME["surface"],
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color=CHART_THEME["muted"]),
        margin=dict(l=24, r=24, t=24, b=50),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor=CHART_THEME["surface"], font_size=12),
        colorway=CHART_COLORS,
    )
    fig.update_xaxes(gridcolor=CHART_THEME["grid"], zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=CHART_THEME["grid"], zeroline=False, automargin=True)
    return fig

def show_figure(fig, key):
    st.plotly_chart(fig, use_container_width=True, key=key,
                    config={"displayModeBar": False, "scrollZoom": False, "responsive": True})

def donut_figure(labels, values, center, caption, colors=None):
    values = list(values)
    labels = list(labels)
    empty = sum(values) == 0
    fig = go.Figure(go.Pie(
        labels=["No data yet"] if empty else labels,
        values=[1] if empty else values,
        hole=0.76, sort=False, direction="clockwise", textinfo="none",
        marker=dict(colors=[CHART_THEME["grid"]] if empty else (colors or CHART_COLORS),
                    line=dict(color=CHART_THEME["surface"], width=3)),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>" if not empty else "No data yet<extra></extra>",
        showlegend=not empty,
    ))
    style_figure(fig)
    fig.update_layout(annotations=[
        dict(text=str(center), x=0.5, y=0.54, showarrow=False,
             font=dict(size=32, color=CHART_THEME["ink"])),
        dict(text=caption, x=0.5, y=0.40, showarrow=False,
             font=dict(size=11, color=CHART_THEME["muted"]))
    ])
    return fig

def horizontal_figure(labels, values, color=None):
    fig = go.Figure(go.Bar(
        x=list(values), y=list(labels), orientation="h",
        marker_color=color or CHART_THEME["blue"],
        text=list(values), textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    style_figure(fig, max(290, 42 * len(labels) + 70))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(rangemode="tozero", dtick=1)
    fig.update_layout(showlegend=False, bargap=0.42, margin=dict(r=55))
    return fig

def task_progress_data(timetable, completed_tasks):
    data = timetable.copy()
    if data.empty:
        return data, 0, 0, 0
    data["Completed"] = data["Task ID"].isin(completed_tasks)
    total = len(data)
    done = int(data["Completed"].sum())
    return data, done, total, round(done / total * 100)

def sync_task_completion(task_id):
    if st.session_state.get(f"task_{task_id}", False):
        st.session_state.completed_tasks.add(task_id)
    else:
        st.session_state.completed_tasks.discard(task_id)

def render_plan_charts(timetable):
    data, done, total, percent = task_progress_data(timetable, st.session_state.completed_tasks)
    if not total:
        return
    left, right = st.columns([1, 1.6])
    with left:
        st.markdown('<div class="heading-box-teal"><span>Task completion</span></div>', unsafe_allow_html=True)
        show_figure(donut_figure(["Completed", "Remaining"], [done, total-done],
                    f"{percent}%", f"{done} of {total} tasks", [CHART_THEME["teal"], CHART_THEME["grid"]]), "task_completion")
    with right:
        st.markdown('<div class="heading-box-primary" style="margin: 0.5rem 0 0.8rem; padding: 6px 14px;">'
                    '<span style="font-weight:600; color:#1a329c; font-size:1rem;">Subject progress</span></div>', unsafe_allow_html=True)
        by_subject = data.groupby("Subject", sort=False)["Completed"].agg(["sum", "count"])
        fig = go.Figure()
        for label, vals, color in [
            ("Completed", by_subject["sum"], CHART_THEME["teal"]),
            ("Remaining", by_subject["count"]-by_subject["sum"], CHART_THEME["blue"]),
        ]:
            fig.add_bar(name=label, x=by_subject.index, y=vals, marker_color=color,
                        hovertemplate="%{x}: %{y} tasks<extra>"+label+"</extra>")
        style_figure(fig)
        fig.update_layout(barmode="stack", bargap=0.55)
        fig.update_yaxes(title="Tasks", dtick=1)
        show_figure(fig, "subject_completion")

    st.markdown('<div class="heading-box-purple"><span>Study workload</span></div>', unsafe_allow_html=True)
    fig = go.Figure()
    dates = sorted(data["Date"].unique())
    for label, mask, color in [
        ("Completed", data["Completed"], CHART_THEME["teal"]),
        ("Remaining", ~data["Completed"], CHART_THEME["blue"]),
    ]:
        minutes = data[mask].groupby("Date")["Minutes"].sum().reindex(dates, fill_value=0)
        fig.add_bar(name=label, x=[d.isoformat() for d in dates], y=minutes, marker_color=color,
                    hovertemplate="%{x}: %{y} minutes<extra>"+label+"</extra>")
    style_figure(fig, 270)
    fig.update_layout(barmode="stack", bargap=0.45)
    fig.update_xaxes(type="category")
    fig.update_yaxes(title="Minutes")
    show_figure(fig, "daily_workload")


# =========================================================
# WORKSPACE SHELL AND NAVIGATION
# =========================================================

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

    c_rst1, c_rst2 = st.columns(2)
    with c_rst1:
        if st.button("Reload Demo", use_container_width=True):
            st.session_state.subjects = [
                {
                    "id": s["id"],
                    "name": s["name"],
                    "exam_date": s["exam_date"],
                    "topics": s["topics"],
                    "demo_papers": s["demo_papers"]
                }
                for s in SAMPLE_RAW_SUBJECTS
            ]
            initial_analysis = []
            for s in st.session_state.subjects:
                q_df, t_df, _ = analyze_subject(s, files=[], demo_papers=s["demo_papers"])
                initial_analysis.append({
                    "id": s["id"],
                    "name": s["name"],
                    "exam_date": s["exam_date"],
                    "questions": q_df,
                    "topics": t_df
                })
            st.session_state.analysis = initial_analysis
            st.session_state.completed_tasks = set()
            st.rerun()

    with c_rst2:
        if st.button("Clear All", use_container_width=True):
            st.session_state.subjects = []
            st.session_state.analysis = []
            st.session_state.completed_tasks = set()
            st.rerun()

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
paper_count = sum(
    len(st.session_state.get(f"files_{s['id']}", []) or []) + len(s.get("demo_papers", []))
    for s in st.session_state.subjects
)
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
            "topics": "",
            "demo_papers": []
        })
        st.rerun()

    if not st.session_state.subjects:
        st.markdown("""
        <div class="empty-workspace"><div class="empty-symbol">＋</div>
        <div class="heading-box-primary" style="margin: 0 auto; max-width: 300px;">
          <h2 style="font-size: 1.1rem !important;">Your workspace starts here</h2>
        </div>
        <p style="margin-top: 10px;">No subjects or question papers yet.</p></div>
        """, unsafe_allow_html=True)

    for index, subject in enumerate(st.session_state.subjects):
        sid = subject["id"]
        with st.container(border=True):
            st.markdown(f"""
            <div class="heading-box-primary" style="margin: 0.2rem 0 1rem; padding: 8px 14px;">
              <span style="font-weight:700; color:#1a329c; font-size:1.1rem;">
                {index + 1:02d} / Subject Details
              </span>
            </div>
            """, unsafe_allow_html=True)

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

            # Show either newly uploaded files or existing demo papers
            total_papers = list(files or []) + subject.get("demo_papers", [])
            if total_papers:
                st.caption("Selected papers:")
                for p in total_papers:
                    p_name = p.name if hasattr(p, "name") else p.get("name", "Paper")
                    st.write(f"📄 {p_name}")

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
                    st.session_state.subjects = [s for s in st.session_state.subjects if s["id"] != sid]
                    st.session_state.analysis = [a for a in st.session_state.analysis if a["id"] != sid]
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
            name = st.session_state.get(f"name_{sid}", subject["name"]).strip()
            exam_date = st.session_state.get(f"date_{sid}", subject["exam_date"])
            topics = st.session_state.get(f"topics_{sid}", subject["topics"])
            files = st.session_state.get(f"files_{sid}", [])
            demo_papers = subject.get("demo_papers", [])

            if not name or not topics.strip() or (not files and not demo_papers):
                errors.append(f"{name or 'Subject ' + str(sid)}: missing name, topics, or PDF.")
                continue

            subject["name"] = name
            subject["exam_date"] = exam_date
            subject["topics"] = topics

            with st.spinner(f"Analyzing {name}..."):
                question_df, topic_df, notices = analyze_subject(subject, files, demo_papers)

            for notice in notices:
                st.write(notice)

            if question_df.empty:
                errors.append(f"{name}: no readable questions were extracted.")
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
        total_questions = sum(len(s["questions"]) for s in analyses)
        high_topics = sum(int((s["topics"]["Priority"] == "High").sum()) for s in analyses)
        low_confidence = sum(
            int(s["questions"]["Match Status"].isin(["Review suggested", "Low confidence — verify"]).sum())
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

        st.markdown(f"""
        <div class="heading-box-primary">
          <h2>Subject Focus: {subject['name']}</h2>
        </div>
        """, unsafe_allow_html=True)

        x, y, z = st.columns(3)
        x.metric("Exam date", subject["exam_date"].strftime("%d %b %Y"))
        y.metric("Days remaining", max((subject["exam_date"] - date.today()).days, 0))
        z.metric("Papers analyzed", questions["Paper"].nunique())

        left, right = st.columns([1.6, 1])
        with left:
            st.markdown('<div class="heading-box-teal"><span>Topic frequency</span></div>', unsafe_allow_html=True)
            chart = topics[topics["Questions"] > 0].head(12)
            if not chart.empty:
                show_figure(horizontal_figure(chart["Topic"], chart["Questions"],
                            [PRIORITY_COLORS[p] for p in chart["Priority"]]), "topic_frequency")
            else:
                st.info("No mapped questions yet.")
        with right:
            st.markdown('<div class="heading-box-purple"><span>Priority distribution</span></div>', unsafe_allow_html=True)
            priorities = topics["Priority"].value_counts()
            show_figure(donut_figure(priorities.index, priorities.values, len(topics), "syllabus topics",
                        [PRIORITY_COLORS[p] for p in priorities.index]), "priority_distribution")

        st.markdown('<div class="heading-box-primary" style="margin: 1.5rem 0 0.8rem; padding: 8px 16px;">'
                    '<span style="font-weight:700; color:#1a329c; font-size:1.15rem;">Topic Priority and Weightage</span></div>', unsafe_allow_html=True)

        st.dataframe(topics, use_container_width=True, hide_index=True)

        left, right = st.columns([1.6, 1])
        with left:
            st.markdown('<div class="heading-box-teal"><span>Question type pattern</span></div>', unsafe_allow_html=True)
            kinds = questions["Question Type"].value_counts()
            show_figure(horizontal_figure(kinds.index, kinds.values, CHART_THEME["teal"]), "question_types")
        with right:
            st.markdown('<div class="heading-box-amber"><span>Estimated difficulty</span></div>', unsafe_allow_html=True)
            difficulty = questions["Difficulty (estimated)"].value_counts().reindex(["Easy", "Medium", "Hard"], fill_value=0)
            show_figure(donut_figure(difficulty.index, difficulty.values, len(questions), "questions",
                        [CHART_THEME["teal"], CHART_THEME["amber"], CHART_THEME["purple"]]), "difficulty")

        st.markdown('<div class="heading-box-primary" style="margin: 1.5rem 0 0.8rem; padding: 8px 16px;">'
                    '<span style="font-weight:700; color:#1a329c; font-size:1.15rem;">Question Mapping</span></div>', unsafe_allow_html=True)

        status_filter = st.selectbox(
            "Filter mapping results",
            ["All questions", "Strong match", "Review suggested", "Low confidence — verify"]
        )

        shown = questions.copy()
        if status_filter != "All questions":
            shown = shown[shown["Match Status"] == status_filter]

        st.dataframe(
            shown[[
                "Paper", "Year", "Question No.", "Question",
                "Matched Topic", "Confidence %", "Match Status",
                "Score Gap %", "Question Type",
                "Difficulty (estimated)", "Marks (if detected)"
            ]],
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

        st.caption("Confidence is a similarity score, not a probability. Review uncertain matches before relying on topic priorities.")

        st.markdown('<div class="heading-box-purple"><span>Year-wise Trends</span></div>', unsafe_allow_html=True)
        year_data = questions.dropna(subset=["Year"]).copy()

        if not year_data.empty:
            year_data["Year"] = year_data["Year"].astype(int)
            trend = year_data.groupby(["Year", "Matched Topic"]).size().unstack(fill_value=0)
            fig = go.Figure()
            for topic in trend.columns:
                fig.add_scatter(x=trend.index.tolist(), y=trend[topic].tolist(),
                                mode="lines+markers", name=topic,
                                hovertemplate="%{x}: %{y} questions<extra>%{fullData.name}</extra>")
            style_figure(fig, 340)
            fig.update_xaxes(title="Paper year", dtick=1, tickformat="d")
            fig.update_yaxes(title="Questions", dtick=1, rangemode="tozero")
            show_figure(fig, "year_trends")
        else:
            st.info("Include a year in each filename, e.g. Maths_2024.pdf, to enable year-wise trends.")


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
            st.warning("No future study tasks found. Check the exam dates and topic mappings.")
        else:
            today = date.today()
            task_ids = timetable["Task ID"].tolist()
            done = sum(task_id in st.session_state.completed_tasks for task_id in task_ids)
            progress = round(done / len(task_ids) * 100)

            nearest_exam = min(
                s["exam_date"] for s in analyses if s["exam_date"] >= today
            ) if any(s["exam_date"] >= today for s in analyses) else None

            days_left = max((nearest_exam - today).days, 0) if nearest_exam else 0

            a, b, c = st.columns(3)
            a.metric("Total tasks", len(task_ids))
            b.metric("Completed tasks", done)
            c.metric("Overall progress", f"{progress}%")

            st.metric("Days to nearest exam", days_left)
            render_plan_charts(timetable)

            st.markdown('<div class="heading-box-primary" style="margin: 1.8rem 0 1rem; padding: 10px 18px;">'
                        '<span style="font-weight:700; color:#1a329c; font-size:1.25rem;">Your Daily Checklist</span></div>', unsafe_allow_html=True)

            for plan_day in sorted(timetable["Date"].unique()):
                day_tasks = timetable[timetable["Date"] == plan_day]
                day_ids = day_tasks["Task ID"].tolist()
                completed_today = sum(task_id in st.session_state.completed_tasks for task_id in day_ids)

                with st.container(border=True):
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                      <div class="heading-box-teal" style="margin:0; padding:6px 12px; font-size:0.95rem;">
                        📅 {plan_day.strftime('%A, %d %B %Y')}
                      </div>
                      <span style="font-size:12px; font-weight:600; color:var(--muted); background:var(--background); padding:4px 10px; border-radius:12px;">
                        {completed_today}/{len(day_ids)} tasks done
                      </span>
                    </div>
                    """, unsafe_allow_html=True)

                    for _, task in day_tasks.iterrows():
                        task_id = task["Task ID"]
                        label = f"{task['Subject']} — {task['Topic']} ({task['Minutes']} min; {task['Kind']})"
                        checked = st.checkbox(
                            label,
                            value=(task_id in st.session_state.completed_tasks),
                            key=f"task_{task_id}",
                            on_change=sync_task_completion,
                            args=(task_id,)
                        )
                        if checked:
                            st.session_state.completed_tasks.add(task_id)
                        else:
                            st.session_state.completed_tasks.discard(task_id)

                    current_done = sum(task_id in st.session_state.completed_tasks for task_id in day_ids)
                    st.progress(current_done / len(day_ids) if day_ids else 0)

            st.markdown('<div class="heading-box-purple" style="margin: 1.6rem 0 0.8rem; font-size: 1.1rem;">'
                        '<span>Preparation Progress</span></div>', unsafe_allow_html=True)

            latest_done = sum(task_id in st.session_state.completed_tasks for task_id in task_ids)
            latest_percent = round(latest_done / len(task_ids) * 100)

            st.metric("Study plan completed", f"{latest_percent}%")
            st.progress(latest_percent / 100)

            if latest_percent == 100:
                st.success("All scheduled tasks are complete. Use your remaining time for a calm final review.")
            elif latest_percent >= 60:
                st.success("Good progress. Keep completing your daily goals.")
            else:
                st.info("Start with the earliest exam and the highest-priority topics.")

            export = timetable.copy()
            export["Completed"] = export["Task ID"].apply(lambda x: x in st.session_state.completed_tasks)
            export["Date"] = export["Date"].apply(lambda x: x.strftime("%d %b %Y"))
            export["Exam Date"] = export["Exam Date"].apply(lambda x: x.strftime("%d %b %Y"))

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
