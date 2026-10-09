
import streamlit as st
import pandas as pd
import re
import hashlib
from io import BytesIO
from datetime import date, timedelta
from difflib import SequenceMatcher

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIGURATION AND STYLE
# =========================================================

st.set_page_config(
    page_title="Exam Intelligence Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #f5f7ff 0%, #ffffff 65%);
}
.block-container {
    padding-top: 1.7rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
.hero {
    padding: 28px;
    border-radius: 20px;
    color: white;
    background: linear-gradient(115deg, #172554, #4338ca, #7c3aed);
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    font-size: 2.2rem;
    margin-bottom: 8px;
}
.hero p {
    color: #e0e7ff;
    margin-bottom: 0;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e4e7f0;
    padding: 16px;
    border-radius: 15px;
    box-shadow: 0 3px 12px rgba(35, 45, 90, 0.04);
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
}
.small-note {
    color: #64748b;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "subjects": [],
    "next_id": 1,
    "analysis": [],
    "completed_tasks": set(),
    "analysis_signature": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# TEXT UTILITIES
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
    "using", "obtain", "hence", "show", "calculate"
}

def normalize(text):
    text = str(text).lower()
    text = text.replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def words(text):
    result = re.findall(r"[a-zA-Z]+", normalize(text))
    result = [
        w for w in result
        if len(w) > 2 and w not in STOP_WORDS
    ]
    return result


def stable_key(*parts):
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:18]


# =========================================================
# PDF EXTRACTION WITH OPTIONAL OCR
# =========================================================

def extract_pdf(pdf_bytes):
    text_parts = []

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                text_parts.append(text)
    except Exception:
        pass

    text = "\n".join(text_parts)

    if len(text.strip()) >= 60:
        return text, "Selectable PDF text"

    # Optional OCR fallback. This needs system packages too.
    try:
        from pdf2image import convert_from_bytes
        import pytesseract

        images = convert_from_bytes(pdf_bytes, dpi=180)
        ocr_parts = []

        for image in images:
            ocr_parts.append(
                pytesseract.image_to_string(image, config="--psm 6")
            )

        ocr_text = "\n".join(ocr_parts)

        if ocr_text.strip():
            return ocr_text, "OCR"

    except Exception:
        pass

    return text, "Text extraction only; OCR unavailable or unsuccessful"


# =========================================================
# QUESTION EXTRACTION
# =========================================================

def clean_header_lines(text):
    patterns = [
        r"^\s*time\s*(allowed)?\s*[:\-]?",
        r"^\s*maximum marks",
        r"^\s*max\.?\s*marks",
        r"^\s*instructions?\s*[:\-]?",
        r"^\s*attempt\s+(any|all)",
        r"^\s*duration\s*[:\-]?",
        r"^\s*roll\s*no",
        r"^\s*enrollment",
        r"^\s*university",
        r"^\s*department",
        r"^\s*semester\s*[:\-]?",
        r"^\s*branch\s*[:\-]?",
        r"^\s*subject\s*[:\-]?",
        r"^\s*end semester",
        r"^\s*mid semester"
    ]

    output = []
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(re.search(p, line, re.I) for p in patterns):
            continue
        output.append(line)

    return "\n".join(output)


def extract_questions(text):
    text = clean_header_lines(text)

    # Recognizes formats such as:
    # 1. ...  2. ...
    # Q.1 ... Q2 ...
    # Question 1 ...
    pattern = re.compile(
        r"(?m)^\s*(?:Q(?:uestion)?\s*\.?\s*)?"
        r"(\d{1,2})\s*[\.\):\-]\s+"
    )

    matches = list(pattern.finditer(text))
    questions = []

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        q = re.sub(r"\s+", " ", text[start:end]).strip()

        if len(q) < 20:
            continue

        if re.fullmatch(r"[\W\d]+", q):
            continue

        questions.append({
            "number": match.group(1),
            "text": q
        })

    # Fallback for papers without clear numbering.
    if not questions:
        paragraphs = re.split(r"\n+", text)
        for i, p in enumerate(paragraphs, start=1):
            p = re.sub(r"\s+", " ", p).strip()
            if len(p) >= 35:
                questions.append({
                    "number": str(i),
                    "text": p
                })

    return questions


# =========================================================
# QUESTION TYPE, DIFFICULTY AND MARKS ESTIMATION
# =========================================================

def classify_type(question):
    q = normalize(question)

    derivation_words = [
        "derive", "derivation", "prove that", "proof of",
        "deduce", "show that"
    ]

    numerical_words = [
        "calculate", "compute", "determine the value",
        "find the current", "find the voltage", "evaluate",
        "solve for", "numerical", "resistance", "velocity",
        "temperature", "pressure", "concentration"
    ]

    theory_words = [
        "explain", "describe", "discuss", "define",
        "differentiate between", "compare", "write a note",
        "state the", "list the"
    ]

    if any(x in q for x in derivation_words):
        return "Derivation"

    if any(x in q for x in numerical_words):
        return "Numerical"

    if any(x in q for x in theory_words):
        return "Theory"

    if re.search(r"\d+\s*(?:ohm|volt|ampere|kg|mol|°c)", q):
        return "Numerical"

    return "Other / Mixed"


def estimate_difficulty(question):
    q = normalize(question)
    score = 0

    medium_words = [
        "explain", "compare", "differentiate", "calculate",
        "derive", "prove", "evaluate", "determine"
    ]

    hard_words = [
        "hence prove", "derive and prove",
        "using the above result", "optimize",
        "multi-step", "discuss in detail"
    ]

    score += sum(1 for w in medium_words if w in q)
    score += 2 * sum(1 for w in hard_words if w in q)

    if len(q) > 220:
        score += 2
    elif len(q) > 100:
        score += 1

    if re.search(r"\([a-d]\)", q) or re.search(r"\([ivx]+\)", q):
        score += 1

    if score >= 4:
        return "Hard"
    if score >= 2:
        return "Medium"
    return "Easy"


def extract_marks(question):
    # Only accepts marks explicitly attached to the question.
    patterns = [
        r"\[\s*(\d{1,2})\s*marks?\s*\]",
        r"\(\s*(\d{1,2})\s*marks?\s*\)",
        r"(\d{1,2})\s*marks?\s*$",
        r"\[\s*(\d{1,2})\s*\]\s*$"
    ]

    for pattern in patterns:
        match = re.search(pattern, question, re.I)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 30:
                return value

    return None


def extract_year(filename):
    match = re.search(r"\b(20\d{2}|19\d{2})\b", filename)
    return int(match.group(1)) if match else None


# =========================================================
# SYLLABUS TOPIC MATCHING
# =========================================================

def match_topics(questions, topics):
    if not questions or not topics:
        return []

    question_texts = [q["text"] for q in questions]

    # TF-IDF works without a paid API or network access.
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )

        all_texts = topics + question_texts
        matrix = vectorizer.fit_transform(all_texts)

        topic_matrix = matrix[:len(topics)]
        question_matrix = matrix[len(topics):]

        similarities = cosine_similarity(
            question_matrix,
            topic_matrix
        )

    except ValueError:
        similarities = [[0] * len(topics) for _ in questions]

    results = []

    for i, question in enumerate(questions):
        qtext = normalize(question["text"])
        scores = []

        for j, topic in enumerate(topics):
            ttext = normalize(topic)

            if ttext and ttext in qtext:
                score = 1.0
            else:
                tfidf_score = float(similarities[i][j])

                qwords = set(words(qtext))
                twords = set(words(ttext))
                overlap = (
                    len(qwords & twords) / len(twords)
                    if twords else 0
                )

                fuzzy = SequenceMatcher(
                    None, qtext, ttext
                ).ratio()

                score = (
                    0.65 * tfidf_score
                    + 0.25 * overlap
                    + 0.10 * fuzzy
                )

            scores.append(score)

        best_index = max(range(len(scores)), key=lambda x: scores[x])
        best_score = scores[best_index]

        # Conservative threshold prevents forcing unrelated matches.
        if best_score < 0.10:
            matched_topic = "Unclassified"
        else:
            matched_topic = topics[best_index]

        confidence = round(min(best_score, 1.0) * 100, 1)

        results.append({
            "Question No.": question["number"],
            "Question": question["text"],
            "Matched Topic": matched_topic,
            "Confidence %": confidence,
            "Question Type": classify_type(question["text"]),
            "Difficulty (estimated)": estimate_difficulty(question["text"]),
            "Marks (if detected)": extract_marks(question["text"])
        })

    return results


# =========================================================
# TOPIC ANALYSIS
# =========================================================

def analyze_subject(subject, paper_files):
    topics = [
        t.strip()
        for t in subject["topics"].splitlines()
        if t.strip()
    ]

    all_rows = []
    notices = []
    paper_years = {}

    for paper in paper_files:
        text, method = extract_pdf(paper.getvalue())

        if not text.strip():
            notices.append(f"{paper.name}: no readable text found.")
            continue

        questions = extract_questions(text)

        if not questions:
            notices.append(f"{paper.name}: no questions were detected.")
            continue

        rows = match_topics(questions, topics)
        year = extract_year(paper.name)

        for row in rows:
            row["Paper"] = paper.name
            row["Year"] = year
            all_rows.append(row)

        paper_years[paper.name] = year
        notices.append(
            f"{paper.name}: {len(rows)} questions detected using {method}."
        )

    result = pd.DataFrame(all_rows)

    if result.empty:
        return result, pd.DataFrame(), notices

    topic_rows = []

    for topic in topics:
        subset = result[result["Matched Topic"] == topic]

        count = len(subset)
        papers_count = subset["Paper"].nunique()

        detected_marks = subset["Marks (if detected)"].dropna()
        marks_total = (
            int(detected_marks.sum())
            if not detected_marks.empty else None
        )

        topic_rows.append({
            "Topic": topic,
            "Questions": count,
            "Papers Appearing": papers_count,
            "Theory": int((subset["Question Type"] == "Theory").sum()),
            "Numerical": int((subset["Question Type"] == "Numerical").sum()),
            "Derivation": int((subset["Question Type"] == "Derivation").sum()),
            "Easy": int((subset["Difficulty (estimated)"] == "Easy").sum()),
            "Medium": int((subset["Difficulty (estimated)"] == "Medium").sum()),
            "Hard": int((subset["Difficulty (estimated)"] == "Hard").sum()),
            "Marks Detected": marks_total
        })

    topic_df = pd.DataFrame(topic_rows)

    total_questions = max(len(result), 1)
    total_papers = max(result["Paper"].nunique(), 1)

    topic_df["Frequency %"] = (
        topic_df["Questions"] / total_questions * 100
    ).round(1)

    topic_df["Paper Coverage %"] = (
        topic_df["Papers Appearing"] / total_papers * 100
    ).round(1)

    # Marks weightage is shown only when marks were detected.
    total_detected_marks = topic_df["Marks Detected"].sum(min_count=1)

    if pd.notna(total_detected_marks) and total_detected_marks > 0:
        topic_df["Detected Marks %"] = (
            topic_df["Marks Detected"] / total_detected_marks * 100
        ).round(1)
    else:
        topic_df["Detected Marks %"] = None

    # Explainable heuristic score; not a prediction of exam questions.
    topic_df["Priority Score"] = (
        0.60 * topic_df["Frequency %"]
        + 0.40 * topic_df["Paper Coverage %"]
    ).round(1)

    def priority(row):
        if row["Questions"] == 0:
            return "Insufficient evidence"
        if row["Priority Score"] >= 45:
            return "High"
        if row["Priority Score"] >= 20:
            return "Medium"
        return "Low"

    topic_df["Priority"] = topic_df.apply(priority, axis=1)

    def reason(row):
        if row["Questions"] == 0:
            return "No question was confidently mapped to this topic."
        return (
            f"{int(row['Questions'])} mapped question(s) across "
            f"{int(row['Papers Appearing'])} paper(s); "
            f"frequency {row['Frequency %']}%."
        )

    topic_df["Reason"] = topic_df.apply(reason, axis=1)

    topic_df = topic_df.sort_values(
        ["Priority Score", "Questions"],
        ascending=False
    ).reset_index(drop=True)

    # Trend table only where paper filenames contain years.
    result["Year"] = pd.to_numeric(result["Year"], errors="coerce")

    return result, topic_df, notices


# =========================================================
# TIMETABLE GENERATOR
# =========================================================

def build_timetable(analysis, daily_hours):
    today = date.today()
    tasks = []

    for subject in analysis:
        exam_date = subject["exam_date"]

        if exam_date < today:
            continue

        for _, row in subject["topics"].iterrows():
            if row["Questions"] <= 0:
                continue

            priority = row["Priority"]
            minutes = 90 if priority == "High" else 60 if priority == "Medium" else 40

            tasks.append({
                "Subject": subject["name"],
                "Topic": row["Topic"],
                "Priority": priority,
                "Score": float(row["Priority Score"]),
                "Minutes": minutes,
                "Exam Date": exam_date
            })

    if not tasks:
        return pd.DataFrame()

    # Schedule nearer exams first, then higher-priority topics.
    tasks.sort(key=lambda t: (t["Exam Date"], -t["Score"]))

    plan = []
    remaining = tasks.copy()
    max_minutes = daily_hours * 60
    last_exam = max(t["Exam Date"] for t in tasks)

    day = today

    while day <= last_exam and remaining:
        available = max_minutes
        candidates = sorted(
            [t for t in remaining if t["Exam Date"] >= day],
            key=lambda t: (
                t["Exam Date"],
                -t["Score"]
            )
        )

        selected = []

        for task in candidates:
            if task["Minutes"] <= available:
                selected.append(task)
                available -= task["Minutes"]

            if available < 30:
                break

        for task in selected:
            plan.append({
                "Date": day,
                "Subject": task["Subject"],
                "Topic": task["Topic"],
                "Priority": task["Priority"],
                "Minutes": task["Minutes"],
                "Exam Date": task["Exam Date"]
            })
            remaining.remove(task)

        day += timedelta(days=1)

    df = pd.DataFrame(plan)

    if not df.empty:
        df["Task ID"] = df.apply(
            lambda r: stable_key(
                r["Date"], r["Subject"], r["Topic"]
            ),
            axis=1
        )

    return df


# =========================================================
# HEADER AND SIDEBAR
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🎓 Exam Intelligence Assistant</h1>
    <p>Understand your PYQs. Prioritize your syllabus. Track every study session.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Study Settings")
    daily_hours = st.slider(
        "Available study hours per day",
        min_value=1,
        max_value=12,
        value=4
    )

    st.caption("Analysis runs locally using free text-processing tools.")
    st.divider()
    st.markdown("**Your workflow**")
    st.write("1. Add subjects and papers")
    st.write("2. Analyze questions")
    st.write("3. Follow and check off the timetable")


# =========================================================
# DASHBOARD NAVIGATION
# =========================================================

setup_tab, analysis_tab, planner_tab = st.tabs([
    "📚 1. Subjects & Setup",
    "📊 2. Exam Intelligence",
    "📅 3. Study Planner"
])


# =========================================================
# DASHBOARD 1: SUBJECT SETUP
# =========================================================

with setup_tab:
    st.header("Your Subjects")
    st.write("Each subject has its own syllabus, exam date and previous-year papers.")

    if st.button("➕ Add Subject", type="primary", key="add_subject"):
        new_id = st.session_state.next_id
        st.session_state.next_id += 1

        st.session_state.subjects.append({
            "id": new_id,
            "name": "",
            "exam_date": date.today() + timedelta(days=7),
            "topics": ""
        })

        st.rerun()

    if not st.session_state.subjects:
        st.info("Click Add Subject to create your first subject.")

    for index, subject in enumerate(st.session_state.subjects):
        sid = subject["id"]

        with st.container(border=True):
            st.subheader(f"Subject {index + 1}")

            c1, c2 = st.columns([2, 1])

            with c1:
                name = st.text_input(
                    "Subject name",
                    value=subject["name"],
                    key=f"subject_name_{sid}",
                    placeholder="e.g. Applied Mathematics"
                )

            with c2:
                exam_date = st.date_input(
                    "Exam date",
                    value=subject["exam_date"],
                    key=f"subject_date_{sid}"
                )

            topics = st.text_area(
                "Syllabus topics — one topic per line",
                value=subject["topics"],
                key=f"subject_topics_{sid}",
                height=130,
                placeholder="Taylor and Maclaurin theorems\nBeta and Gamma functions\nTracing of curves"
            )

            files = st.file_uploader(
                "Upload previous-year question papers (PDF)",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"subject_papers_{sid}"
            )

            if files:
                st.caption("Uploaded papers:")
                for f in files:
                    st.write(f"📄 {f.name} · {f.size / 1024:.0f} KB")

            b1, b2 = st.columns(2)

            with b1:
                if st.button("💾 Save Subject", key=f"save_subject_{sid}"):
                    if not name.strip():
                        st.error("Enter a subject name.")
                    elif not topics.strip():
                        st.error("Enter at least one syllabus topic.")
                    else:
                        duplicate = any(
                            other["id"] != sid
                            and other["name"].strip().lower() == name.strip().lower()
                            for other in st.session_state.subjects
                        )

                        if duplicate:
                            st.error("Another subject already has this name.")
                        else:
                            subject["name"] = name.strip()
                            subject["exam_date"] = exam_date
                            subject["topics"] = topics
                            st.success("Subject details saved.")

            with b2:
                if st.button("🗑️ Remove Subject", key=f"remove_subject_{sid}"):
                    st.session_state.subjects = [
                        s for s in st.session_state.subjects if s["id"] != sid
                    ]
                    st.session_state.analysis = []
                    st.session_state.analysis_signature = None
                    st.rerun()

    st.divider()

    if st.button("🚀 Analyze All Subjects", type="primary", use_container_width=True):
        all_analysis = []
        errors = []

        for subject in st.session_state.subjects:
            sid = subject["id"]

            # Read current widget values, even if Save Subject wasn't clicked.
            name = st.session_state.get(f"subject_name_{sid}", subject["name"]).strip()
            exam_date = st.session_state.get(f"subject_date_{sid}", subject["exam_date"])
            topics = st.session_state.get(f"subject_topics_{sid}", subject["topics"])
            files = st.session_state.get(f"subject_papers_{sid}", [])

            if not name or not topics.strip() or not files:
                errors.append(
                    f"{name or 'Subject ' + str(sid)}: provide a name, syllabus and at least one PDF."
                )
                continue

            current_subject = {
                "id": sid,
                "name": name,
                "exam_date": exam_date,
                "topics": topics
            }

            with st.spinner(f"Analyzing {name}..."):
                result, topic_df, notices = analyze_subject(current_subject, files)

            for notice in notices:
                st.write(notice)

            if result.empty:
                errors.append(f"{name}: no questions could be extracted.")
                continue

            all_analysis.append({
                "id": sid,
                "name": name,
                "exam_date": exam_date,
                "questions": result,
                "topics": topic_df,
                "notices": notices
            })

            # Update saved subject details.
            subject["name"] = name
            subject["exam_date"] = exam_date
            subject["topics"] = topics

        if errors:
            for error in errors:
                st.warning(error)

        if all_analysis:
            st.session_state.analysis = all_analysis
            st.session_state.analysis_signature = stable_key(
                date.today(),
                *[
                    f"{s['id']}:{s['name']}:{s['exam_date']}:{s['topics']}"
                    for s in all_analysis
                ]
            )
            st.success("Analysis completed. Open Dashboard 2 to review the results.")


# =========================================================
# DASHBOARD 2: EXAM INTELLIGENCE
# =========================================================

with analysis_tab:
    st.header("Exam Intelligence Dashboard")

    analysis = st.session_state.analysis

    if not analysis:
        st.info("Add subjects and click Analyze All Subjects in Dashboard 1.")
    else:
        total_questions = sum(len(s["questions"]) for s in analysis)
        total_topics = sum(len(s["topics"]) for s in analysis)
        high_count = sum(
            int((s["topics"]["Priority"] == "High").sum())
            for s in analysis
        )
        unmapped = sum(
            int((s["questions"]["Matched Topic"] == "Unclassified").sum())
            for s in analysis
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Subjects analyzed", len(analysis))
        c2.metric("Questions extracted", total_questions)
        c3.metric("High-priority topics", high_count)
        c4.metric("Unclassified questions", unmapped)

        st.caption(
            "Priority is based on the uploaded papers, not a guarantee of what will appear in your exam."
        )

        selected_subject = st.selectbox(
            "Choose a subject to inspect",
            options=list(range(len(analysis))),
            format_func=lambda i: analysis[i]["name"]
        )

        subject = analysis[selected_subject]
        questions = subject["questions"]
        topics = subject["topics"]

        st.subheader(f"📘 {subject['name']}")

        a, b, c = st.columns(3)
        a.metric("Exam date", subject["exam_date"].strftime("%d %b %Y"))
        a.metric(
            "Days remaining",
            max((subject["exam_date"] - date.today()).days, 0)
        )
        b.metric("Question papers", questions["Paper"].nunique())
        c.metric("Syllabus topics", len(topics))

        st.subheader("🔥 Topic frequency and priority")

        chart_data = topics[topics["Questions"] > 0].set_index("Topic")[
            ["Questions"]
        ]

        if not chart_data.empty:
            st.bar_chart(chart_data, horizontal=True)
        else:
            st.info("No syllabus topics have been confidently matched yet.")

        st.dataframe(
            topics[
                [
                    "Topic", "Questions", "Frequency %",
                    "Papers Appearing", "Paper Coverage %",
                    "Theory", "Numerical", "Derivation",
                    "Easy", "Medium", "Hard",
                    "Marks Detected", "Detected Marks %",
                    "Priority Score", "Priority", "Reason"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("🧠 Question type distribution")

        type_counts = questions["Question Type"].value_counts()
        st.bar_chart(type_counts)

        st.subheader("🎚️ Estimated difficulty distribution")

        difficulty_counts = questions["Difficulty (estimated)"].value_counts()
        st.bar_chart(difficulty_counts)

        st.subheader("📝 Question-to-syllabus mapping")

        st.dataframe(
            questions[
                [
                    "Paper", "Year", "Question No.", "Question",
                    "Matched Topic", "Confidence %",
                    "Question Type", "Difficulty (estimated)",
                    "Marks (if detected)"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download question mapping (CSV)",
            data=questions.to_csv(index=False).encode("utf-8"),
            file_name=f"{subject['name'].replace(' ', '_')}_mapping.csv",
            mime="text/csv"
        )

        st.subheader("📈 Trends across years")

        year_data = questions.dropna(subset=["Year"]).copy()

        if not year_data.empty:
            year_data["Year"] = year_data["Year"].astype(int)
            trend = year_data.groupby(
                ["Year", "Matched Topic"]
            ).size().unstack(fill_value=0)

            st.line_chart(trend)
            st.caption(
                "Years are inferred from PDF filenames. Rename files with the year, "
                "for example Applied_Maths_2024.pdf, to enable this chart."
            )
        else:
            st.info(
                "Year-wise trends need the year in each PDF filename, such as Maths_2023.pdf."
            )

        st.subheader("🎯 Why should I study this topic?")

        studied = topics[topics["Questions"] > 0]

        if not studied.empty:
            first = studied.iloc[0]

            st.success(
                f"**{first['Topic']}** has {int(first['Questions'])} mapped question(s) "
                f"and appeared in {int(first['Papers Appearing'])} uploaded paper(s). "
                f"Its calculated priority is **{first['Priority']}**. "
                f"{first['Reason']}"
            )

        st.caption(
            "Confidence is a text-similarity estimate, not a calibrated probability. "
            "Check low-confidence matches manually."
        )


# =========================================================
# DASHBOARD 3: STUDY PLANNER
# =========================================================

with planner_tab:
    st.header("📅 Smart Study Planner")

    analysis = st.session_state.analysis

    if not analysis:
        st.info("Analyze your subjects in Dashboard 1 first.")
    else:
        today = date.today()

        timetable = build_timetable(analysis, daily_hours)

        if timetable.empty:
            st.warning(
                "No future study tasks could be generated. Check your exam dates and topic mappings."
            )
        else:
            # Stable checklist keys preserve progress through normal reruns.
            task_ids = list(timetable["Task ID"])
            completed = st.session_state.completed_tasks

            completed_ids = [
                task_id for task_id in task_ids if task_id in completed
            ]

            total_tasks = len(task_ids)
            done_tasks = len(completed_ids)
            percent = round(done_tasks / total_tasks * 100) if total_tasks else 0

            days_left = min(
                max((s["exam_date"] - today).days, 0)
                for s in analysis
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tasks planned", total_tasks)
            m2.metric("Tasks completed", done_tasks)
            m3.metric("Overall progress", f"{percent}%")
            m4.metric("Nearest exam countdown", f"{days_left} days")

            st.progress(percent / 100)

            st.subheader("📈 Your progress")

            if total_tasks:
                progress_df = pd.DataFrame({
                    "Status": ["Completed", "Remaining"],
                    "Tasks": [done_tasks, total_tasks - done_tasks]
                }).set_index("Status")

                st.bar_chart(progress_df)

            st.caption(
                "Progress is calculated from the study tasks you check off below. "
                "It is not a prediction of your exam marks."
            )

            st.subheader("✅ Daily checklist")

            plan_dates = sorted(timetable["Date"].unique())

            for plan_day in plan_dates:
                day_rows = timetable[timetable["Date"] == plan_day]

                day_total = len(day_rows)
                day_done = sum(
                    1 for task_id in day_rows["Task ID"]
                    if task_id in st.session_state.completed_tasks
                )

                with st.container(border=True):
                    st.markdown(
                        f"### {plan_day.strftime('%A, %d %B %Y')}"
                    )
                    st.caption(
                        f"{day_done}/{day_total} tasks completed"
                    )

                    for _, task in day_rows.iterrows():
                        task_id = task["Task ID"]
                        label = (
                            f"{task['Subject']} — {task['Topic']} "
                            f"({task['Minutes']} min · {task['Priority']})"
                        )

                        is_done = task_id in st.session_state.completed_tasks

                        checked = st.checkbox(
                            label,
                            value=is_done,
                            key=f"check_{task_id}"
                        )

                        if checked:
                            st.session_state.completed_tasks.add(task_id)
                        else:
                            st.session_state.completed_tasks.discard(task_id)

                    # Update visible day progress after checking tasks.
                    latest_done = sum(
                        1 for task_id in day_rows["Task ID"]
                        if task_id in st.session_state.completed_tasks
                    )
                    st.progress(latest_done / day_total if day_total else 0)

            st.divider()

            completed_now = sum(
                1 for task_id in task_ids
                if task_id in st.session_state.completed_tasks
            )

            current_percent = round(
                completed_now / total_tasks * 100
            ) if total_tasks else 0

            st.subheader("🏁 Preparation summary")
            st.metric("Current completion", f"{current_percent}%")
            st.progress(current_percent / 100)

            if current_percent == 100:
                st.success("All generated study tasks are complete. Review weak topics before the exam.")
            elif current_percent >= 70:
                st.success("Good progress. Keep going and revisit difficult questions.")
            elif current_percent >= 35:
                st.info("You're making progress. Complete today's remaining tasks when possible.")
            else:
                st.info("Start with the nearest exam and the highest-priority mapped topics.")

            export_plan = timetable.copy()
            export_plan["Completed"] = export_plan["Task ID"].apply(
                lambda x: x in st.session_state.completed_tasks
            )
            export_plan["Date"] = export_plan["Date"].apply(
                lambda x: x.strftime("%d %b %Y")
            )
            export_plan["Exam Date"] = export_plan["Exam Date"].apply(
                lambda x: x.strftime("%d %b %Y")
            )

            st.download_button(
                "📥 Download study timetable (CSV)",
                data=export_plan.to_csv(index=False).encode("utf-8"),
                file_name="exam_study_timetable.csv",
                mime="text/csv"
            )

st.divider()
st.caption(
    "Exam Intelligence Assistant · Free local text analysis · "
    "Always verify extracted questions and estimated priorities."
)

