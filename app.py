import streamlit as st
import pandas as pd
import re
from io import BytesIO
from datetime import date, timedelta
from difflib import SequenceMatcher
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Exam Intelligence Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM FRONTEND
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.hero {
    padding: 30px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        #eef2ff 0%,
        #f8fafc 50%,
        #ecfeff 100%
    );
    border: 1px solid #e2e8f0;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    color: #64748b;
}

.section-title {
    font-size: 27px;
    font-weight: 750;
    margin-top: 10px;
    margin-bottom: 15px;
}

.subject-card {
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
    margin-bottom: 15px;
}

.priority-high {
    border-left: 6px solid #ef4444;
    padding: 12px 15px;
    border-radius: 10px;
    background: #fef2f2;
    margin-bottom: 8px;
}

.priority-medium {
    border-left: 6px solid #f59e0b;
    padding: 12px 15px;
    border-radius: 10px;
    background: #fffbeb;
    margin-bottom: 8px;
}

.priority-low {
    border-left: 6px solid #22c55e;
    padding: 12px 15px;
    border-radius: 10px;
    background: #f0fdf4;
    margin-bottom: 8px;
}

.dashboard-card {
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    background: white;
}

.small-muted {
    color: #64748b;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "subjects" not in st.session_state:
    st.session_state.subjects = []

if "next_subject_id" not in st.session_state:
    st.session_state.next_subject_id = 1

if "analyzed" not in st.session_state:
    st.session_state.analyzed = []

if "completed_topics" not in st.session_state:
    st.session_state.completed_topics = set()

if "selected_subject" not in st.session_state:
    st.session_state.selected_subject = None

if "dashboard" not in st.session_state:
    st.session_state.dashboard = "Exam Intelligence"


# ============================================================
# TEXT FUNCTIONS
# ============================================================

STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this",
    "find", "show", "prove", "calculate", "using", "given",
    "derive", "determine", "solve", "evaluate", "write",
    "explain", "define", "of", "to", "in", "on", "is",
    "are", "a", "an", "be", "if", "or", "as", "by",
    "at", "it", "its", "let", "where", "which", "then",
    "also", "following", "any", "all", "question"
}


def normalize_text(text):
    text = text.lower()
    text = text.replace("’", "'")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_words(text):

    words = re.findall(
        r"[a-zA-Z]+",
        normalize_text(text)
    )

    result = []

    for word in words:

        if word in STOP_WORDS:
            continue

        if len(word) <= 2:
            continue

        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"

        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]

        result.append(word)

    return result


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_bytes):

    try:

        from pypdf import PdfReader

        reader = PdfReader(
            BytesIO(pdf_bytes)
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        normal_text = "\n".join(pages)

        if len(normal_text.strip()) >= 80:
            return normal_text, "Text extraction"

    except Exception:
        pass

    # OCR FALLBACK

    try:

        from pdf2image import convert_from_bytes
        import pytesseract

        images = convert_from_bytes(
            pdf_bytes,
            dpi=180,
            fmt="jpeg"
        )

        all_text = []

        for image in images:

            text = pytesseract.image_to_string(
                image,
                config="--psm 6"
            )

            if text:
                all_text.append(text)

        return "\n".join(all_text), "OCR"

    except Exception:

        return "", "Failed"


# ============================================================
# SEMESTER SYLLABUS PDF → UNIT DETECTION
# ============================================================

def clean_unit_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def detect_units_from_syllabus_pdf(text):

    """
    Detects headings such as:

    UNIT 1
    UNIT I
    Unit-2
    Unit II
    UNIT - III

    and collects the text until the next unit.
    """

    pattern = re.compile(
        r"(?:^|\n)"
        r"\s*"
        r"(?:unit|module)"
        r"\s*[-.:]?\s*"
        r"(1|2|3|4|5|6|7|8|9|10|i|ii|iii|iv|v|vi|vii|viii|ix|x)"
        r"\s*[-.:]?",
        re.IGNORECASE
    )

    matches = list(
        pattern.finditer(text)
    )

    units = {}

    roman_to_number = {
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5,
        "vi": 6,
        "vii": 7,
        "viii": 8,
        "ix": 9,
        "x": 10
    }

    for i, match in enumerate(matches):

        raw_number = match.group(1).lower()

        if raw_number.isdigit():
            unit_number = int(raw_number)
        else:
            unit_number = roman_to_number.get(
                raw_number,
                None
            )

        if unit_number is None:
            continue

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end]

        content = clean_unit_text(
            content
        )

        if len(content) > 10:

            units[
                f"Unit {unit_number}"
            ] = content

    return units


# ============================================================
# EXPAND USER SYLLABUS ENTRIES
# ============================================================

def is_unit_entry(text):

    pattern = re.compile(
        r"^\s*(?:unit|module)"
        r"\s*[-.:]?\s*"
        r"(?:\d+|i|ii|iii|iv|v|vi|vii|viii|ix|x)"
        r"\s*$",
        re.IGNORECASE
    )

    return bool(
        pattern.match(text)
    )


def get_unit_number(text):

    match = re.search(
        r"(?:unit|module)"
        r"\s*[-.:]?\s*"
        r"(1|2|3|4|5|6|7|8|9|10|i|ii|iii|iv|v|vi|vii|viii|ix|x)",
        text,
        re.IGNORECASE
    )

    if not match:
        return None

    value = match.group(1).lower()

    roman = {
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5,
        "vi": 6,
        "vii": 7,
        "viii": 8,
        "ix": 9,
        "x": 10
    }

    if value.isdigit():
        return int(value)

    return roman.get(value)


def expand_syllabus_entries(
    syllabus_text,
    detected_units
):

    entries = [
        x.strip()
        for x in syllabus_text.split("\n")
        if x.strip()
    ]

    analysis_topics = []
    display_mapping = []

    for entry in entries:

        if is_unit_entry(entry):

            number = get_unit_number(
                entry
            )

            unit_key = (
                f"Unit {number}"
            )

            if unit_key in detected_units:

                unit_content = (
                    detected_units[unit_key]
                )

                # Try to split unit content
                # into smaller academic topics.
                pieces = re.split(
                    r";|\n|(?<=\.)\s+(?=[A-Z])",
                    unit_content
                )

                cleaned = []

                for piece in pieces:

                    piece = clean_unit_text(
                        piece
                    )

                    if len(piece) >= 8:
                        cleaned.append(
                            piece
                        )

                # If splitting didn't work,
                # use the entire unit.
                if not cleaned:

                    cleaned = [
                        unit_content
                    ]

                for topic in cleaned:

                    analysis_topics.append(
                        topic
                    )

                display_mapping.append({

                    "Selected":
                        entry,

                    "Expanded Into":
                        ", ".join(cleaned)
                })

            else:

                # If PDF did not expose the unit,
                # keep the unit itself.
                analysis_topics.append(
                    entry
                )

                display_mapping.append({

                    "Selected":
                        entry,

                    "Expanded Into":
                        "Unit details could not be extracted"
                })

        else:

            analysis_topics.append(
                entry
            )

            display_mapping.append({

                "Selected":
                    entry,

                "Expanded Into":
                    entry
            })

    return (
        analysis_topics,
        pd.DataFrame(
            display_mapping
        )
    )


# ============================================================
# PAPER CLEANING
# ============================================================

def remove_instruction_lines(text):

    lines = text.split("\n")

    ignored_patterns = [

        r"^\s*time\s*[:\-]?",
        r"^\s*time allowed",
        r"^\s*maximum marks",
        r"^\s*max marks",
        r"^\s*max\.\s*marks",
        r"^\s*note\s*[:\-]?",
        r"^\s*instructions?\s*[:\-]?",
        r"^\s*attempt\s+any",
        r"^\s*attempt\s+all",
        r"^\s*duration\s*[:\-]?",
        r"^\s*date\s*[:\-]?",
        r"^\s*roll\s*no",
        r"^\s*enrollment",
        r"^\s*semester\s*[:\-]?",
        r"^\s*branch\s*[:\-]?",
        r"^\s*university",
        r"^\s*department",
        r"^\s*b\.?tech",
        r"^\s*end semester",
        r"^\s*mid semester"
    ]

    output = []

    for line in lines:

        clean = line.strip()

        if len(clean) < 3:
            continue

        ignored = False

        for pattern in ignored_patterns:

            if re.search(
                pattern,
                clean,
                re.IGNORECASE
            ):

                ignored = True
                break

        if not ignored:
            output.append(line)

    return "\n".join(output)


# ============================================================
# QUESTION EXTRACTION
# ============================================================

def extract_questions(text):

    text = text.replace(
        "\r",
        "\n"
    )

    text = remove_instruction_lines(
        text
    )

    pattern = re.compile(
        r"(?:^|\n)"
        r"\s*"
        r"(?:Q(?:uestion)?\s*)?"
        r"(\d{1,2})"
        r"\s*[\.\):\-]"
        r"\s*",
        re.IGNORECASE
    )

    matches = list(
        pattern.finditer(text)
    )

    questions = []

    for i, match in enumerate(matches):

        start = match.end()

        if i + 1 < len(matches):
            end = matches[
                i + 1
            ].start()
        else:
            end = len(text)

        question = text[
            start:end
        ].strip()

        question = re.sub(
            r"\s+",
            " ",
            question
        )

        if len(question) < 25:
            continue

        lower = question.lower()

        bad_phrases = [
            "attempt any",
            "attempt all",
            "maximum marks",
            "time allowed",
            "instructions"
        ]

        if any(
            phrase in lower
            for phrase in bad_phrases
        ):
            continue

        questions.append(
            question
        )

    if not questions:

        paragraphs = re.split(
            r"\n+",
            text
        )

        for paragraph in paragraphs:

            paragraph = re.sub(
                r"\s+",
                " ",
                paragraph
            ).strip()

            if len(paragraph) >= 35:

                questions.append(
                    paragraph
                )

    return questions


# ============================================================
# TOPIC MATCHING
# ============================================================

def word_overlap(
    question,
    topic
):

    q_words = set(
        get_words(question)
    )

    t_words = set(
        get_words(topic)
    )

    if not t_words:
        return 0

    common = q_words & t_words

    return (
        len(common)
        /
        len(t_words)
    ) * 100


def fuzzy_score(
    question,
    topic
):

    q_words = set(
        get_words(question)
    )

    t_words = set(
        get_words(topic)
    )

    if not t_words:
        return 0

    total = 0

    for topic_word in t_words:

        best = 0

        for question_word in q_words:

            score = SequenceMatcher(
                None,
                topic_word,
                question_word
            ).ratio()

            best = max(
                best,
                score
            )

        total += best

    return (
        total
        /
        len(t_words)
    ) * 100


def topic_similarity(
    question,
    topic
):

    q = normalize_text(
        question
    )

    t = normalize_text(
        topic
    )

    if t in q:
        return 100

    overlap = word_overlap(
        question,
        topic
    )

    fuzzy = fuzzy_score(
        question,
        topic
    )

    return round(
        overlap * 0.65
        +
        fuzzy * 0.35,
        2
    )


def find_best_topic(
    question,
    topics
):

    scores = []

    for topic in topics:

        score = topic_similarity(
            question,
            topic
        )

        scores.append(
            (
                topic,
                score
            )
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    if not scores:

        return (
            "Unclassified",
            0
        )

    best_topic, best_score = (
        scores[0]
    )

    if best_score < 12:

        return (
            "Unclassified",
            best_score
        )

    return (
        best_topic,
        best_score
    )


# ============================================================
# ANALYZE SUBJECT
# ============================================================

def analyze_subject(
    subject_name,
    topics,
    uploaded_files
):

    rows = []
    messages = []

    for uploaded_file in uploaded_files:

        pdf_bytes = (
            uploaded_file.getvalue()
        )

        text, method = (
            extract_text_from_pdf(
                pdf_bytes
            )
        )

        if not text.strip():

            messages.append(
                f"⚠️ {uploaded_file.name}: "
                "No readable text found."
            )

            continue

        questions = (
            extract_questions(
                text
            )
        )

        messages.append(
            f"📄 {uploaded_file.name}: "
            f"{len(questions)} questions "
            f"detected ({method})."
        )

        for number, question in enumerate(
            questions,
            start=1
        ):

            topic, score = (
                find_best_topic(
                    question,
                    topics
                )
            )

            rows.append({

                "Paper":
                    uploaded_file.name,

                "Question No.":
                    number,

                "Question":
                    question,

                "Matched Topic":
                    topic,

                "Match Score":
                    score
            })

    return (
        pd.DataFrame(rows),
        messages
    )


# ============================================================
# PRIORITY
# ============================================================

def calculate_priority(
    results,
    topics,
    paper_count
):

    rows = []

    for topic in topics:

        matched = results[
            results["Matched Topic"]
            == topic
        ]

        question_count = len(
            matched
        )

        papers_with_topic = (
            matched["Paper"]
            .nunique()
        )

        coverage = (
            papers_with_topic
            /
            max(paper_count, 1)
        ) * 100

        frequency = (
            question_count
            /
            max(len(results), 1)
        ) * 100

        score = (
            coverage * 0.65
            +
            frequency * 0.35
        )

        rows.append({

            "Topic":
                topic,

            "Questions":
                question_count,

            "Papers":
                papers_with_topic,

            "Frequency %":
                round(
                    frequency,
                    1
                ),

            "Priority Score":
                round(
                    score,
                    1
                )
        })

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        return df

    df = df.sort_values(
        "Priority Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    priorities = []

    for _, row in df.iterrows():

        if row["Questions"] == 0:

            priorities.append(
                "Low"
            )

        elif row["Priority Score"] >= 35:

            priorities.append(
                "High"
            )

        elif row["Priority Score"] >= 15:

            priorities.append(
                "Medium"
            )

        else:

            priorities.append(
                "Low"
            )

    df["Priority"] = priorities

    return df


# ============================================================
# URGENCY
# ============================================================

def get_urgency(
    exam_date
):

    days = (
        exam_date
        - date.today()
    ).days

    if days < 0:
        return "Exam Passed"

    if days <= 2:
        return "🔴 Critical"

    if days <= 7:
        return "🟠 Urgent"

    if days <= 14:
        return "🟡 Soon"

    return "🟢 Comfortable"


# ============================================================
# TIMETABLE
# ============================================================

def generate_timetable(
    analyzed_subjects,
    daily_hours
):

    today = date.today()

    tasks = []

    for subject in analyzed_subjects:

        for _, row in subject[
            "priority"
        ].iterrows():

            if row["Questions"] <= 0:
                continue

            if row["Priority"] == "High":

                minutes = 90
                priority_value = 3

            elif row["Priority"] == "Medium":

                minutes = 60
                priority_value = 2

            else:

                minutes = 40
                priority_value = 1

            tasks.append({

                "Subject":
                    subject["name"],

                "Topic":
                    row["Topic"],

                "Priority":
                    row["Priority"],

                "Priority Score":
                    row["Priority Score"],

                "Minutes":
                    minutes,

                "Exam Date":
                    subject["exam_date"],

                "Priority Value":
                    priority_value
            })

    if not tasks:
        return pd.DataFrame()

    latest_exam = max(
        x["Exam Date"]
        for x in tasks
    )

    remaining = tasks.copy()

    timetable = []

    current_day = today

    while current_day < latest_exam:

        available = daily_hours * 60

        candidates = []

        for task in remaining:

            if current_day >= task[
                "Exam Date"
            ]:
                continue

            days_left = (
                task["Exam Date"]
                - current_day
            ).days

            urgency = (
                task["Priority Value"]
                * 100
                +
                task["Priority Score"]
                +
                max(
                    0,
                    50 - days_left * 3
                )
            )

            candidates.append(
                (
                    urgency,
                    task
                )
            )

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        selected = []

        for _, task in candidates:

            if task["Minutes"] <= available:

                selected.append(task)

                available -= (
                    task["Minutes"]
                )

            if available < 30:
                break

        for task in selected:

            timetable.append({

                "Date":
                    current_day,

                "Day":
                    current_day.strftime(
                        "%A"
                    ),

                "Subject":
                    task["Subject"],

                "Topic":
                    task["Topic"],

                "Priority":
                    task["Priority"],

                "Study Time":
                    f"{task['Minutes']} min",

                "Exam Date":
                    task["Exam Date"]
            })

            remaining.remove(
                task
            )

        current_day += timedelta(
            days=1
        )

    return pd.DataFrame(
        timetable
    )


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<h1>🎓 Exam Intelligence Assistant</h1>

<p>
Turn your semester syllabus and previous-year papers
into a personalized exam preparation system.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🎓 Exam Intelligence"
    )

    dashboard = st.radio(
        "Navigate",
        [
            "📊 Exam Intelligence",
            "📅 My Study Plan"
        ],
        key="dashboard"
    )

    st.divider()

    daily_hours = st.slider(
        "Daily study hours",
        1,
        12,
        4
    )

    st.divider()

    st.caption(
        "Add subjects → upload syllabus → "
        "upload PYQs → analyze → study."
    )


# ============================================================
# PAGE 1 — EXAM INTELLIGENCE
# ============================================================

if dashboard == "📊 Exam Intelligence":

    st.markdown(
        '<div class="section-title">'
        '📚 Your Subjects'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # ADD SUBJECT
    # --------------------------------------------------------

    if st.button(
        "➕ Add New Subject",
        type="primary"
    ):

        new_id = (
            st.session_state.next_subject_id
        )

        st.session_state.next_subject_id += 1

        st.session_state.subjects.append({

            "id":
                new_id,

            "name":
                "",

            "exam_date":
                date.today()
                + timedelta(days=7),

            "topics":
                ""
        })

        st.rerun()


    if not st.session_state.subjects:

        st.info(
            "Start by clicking **Add New Subject**."
        )


    # --------------------------------------------------------
    # SUBJECT INPUT CARDS
    # --------------------------------------------------------

    for index, subject in enumerate(
        st.session_state.subjects
    ):

        sid = subject["id"]

        with st.container(
            border=True
        ):

            st.subheader(
                f"📘 Subject {index + 1}"
            )

            c1, c2 = st.columns(
                [3, 2]
            )

            with c1:

                name = st.text_input(
                    "Subject Name",
                    value=subject["name"],
                    key=f"name_{sid}",
                    placeholder=(
                        "Example: Applied Mathematics"
                    )
                )

            with c2:

                exam_date = st.date_input(
                    "Exam Date",
                    value=subject["exam_date"],
                    key=f"exam_{sid}"
                )


            # ------------------------------------------------
            # SEMESTER SYLLABUS PDF
            # ------------------------------------------------

            st.markdown(
                "**📖 Semester Syllabus PDF**"
            )

            syllabus_pdf = st.file_uploader(
                "Upload your complete semester syllabus PDF",
                type=["pdf"],
                key=f"semester_pdf_{sid}",
                help=(
                    "Upload the complete syllabus PDF "
                    "provided at the beginning of the semester."
                )
            )


            # ------------------------------------------------
            # SYLLABUS ENTRIES
            # ------------------------------------------------

            topics_text = st.text_area(
                "What is coming in this exam?",
                value=subject["topics"],
                key=f"topics_{sid}",
                height=150,
                placeholder=(
                    "You can mix units and individual topics.\n\n"
                    "Example:\n"
                    "Unit 2\n"
                    "Unit 3\n"
                    "Beta and Gamma Functions\n"
                    "Tracing of Curves"
                )
            )

            st.caption(
                "💡 You can write **Unit 2** if the whole unit "
                "is coming, or write individual topic names "
                "if only selected topics are coming."
            )


            # ------------------------------------------------
            # QUESTION PAPERS
            # ------------------------------------------------

            papers = st.file_uploader(
                "📄 Previous-Year Question Papers",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"papers_{sid}"
            )


            # ------------------------------------------------
            # SAVE / REMOVE
            # ------------------------------------------------

            b1, b2 = st.columns(
                2
            )

            with b1:

                if st.button(
                    "💾 Save Subject",
                    key=f"save_{sid}"
                ):

                    if not name.strip():

                        st.error(
                            "Enter the subject name."
                        )

                    elif not topics_text.strip():

                        st.error(
                            "Enter at least one syllabus "
                            "unit/topic."
                        )

                    else:

                        duplicate = False

                        for other in (
                            st.session_state.subjects
                        ):

                            if (
                                other["id"] != sid
                                and
                                other["name"]
                                .strip()
                                .lower()
                                ==
                                name.strip().lower()
                            ):

                                duplicate = True

                        if duplicate:

                            st.error(
                                "This subject already exists."
                            )

                        else:

                            subject["name"] = (
                                name.strip()
                            )

                            subject["exam_date"] = (
                                exam_date
                            )

                            subject["topics"] = (
                                topics_text
                            )

                            st.success(
                                f"✅ {name} saved."
                            )


            with b2:

                if st.button(
                    "🗑️ Remove Subject",
                    key=f"remove_{sid}"
                ):

                    st.session_state.subjects = [
                        x
                        for x in st.session_state.subjects
                        if x["id"] != sid
                    ]

                    st.session_state.analyzed = []

                    st.rerun()


    # --------------------------------------------------------
    # ANALYZE ALL
    # --------------------------------------------------------

    if st.session_state.subjects:

        st.divider()

        if st.button(
            "🚀 Analyze All Subjects",
            type="primary",
            use_container_width=True
        ):

            analyzed_subjects = []

            failed = False

            for index, subject in enumerate(
                st.session_state.subjects
            ):

                sid = subject["id"]

                name = st.session_state.get(
                    f"name_{sid}",
                    subject["name"]
                ).strip()

                exam_date = st.session_state.get(
                    f"exam_{sid}",
                    subject["exam_date"]
                )

                syllabus = st.session_state.get(
                    f"topics_{sid}",
                    subject["topics"]
                )

                syllabus_pdf = st.session_state.get(
                    f"semester_pdf_{sid}",
                    None
                )

                papers = st.session_state.get(
                    f"papers_{sid}",
                    []
                )


                # ------------------------------------------
                # VALIDATION
                # ------------------------------------------

                if not name:

                    st.error(
                        f"Subject {index + 1}: "
                        "Enter subject name."
                    )

                    failed = True
                    continue

                if not syllabus.strip():

                    st.error(
                        f"{name}: Enter syllabus "
                        "units/topics."
                    )

                    failed = True
                    continue

                if not papers:

                    st.error(
                        f"{name}: Upload at least "
                        "one previous-year paper."
                    )

                    failed = True
                    continue


                # ------------------------------------------
                # READ SEMESTER SYLLABUS
                # ------------------------------------------

                detected_units = {}

                if syllabus_pdf:

                    pdf_bytes = (
                        syllabus_pdf.getvalue()
                    )

                    syllabus_pdf_text, method = (
                        extract_text_from_pdf(
                            pdf_bytes
                        )
                    )

                    detected_units = (
                        detect_units_from_syllabus_pdf(
                            syllabus_pdf_text
                        )
                    )

                    if detected_units:

                        st.success(
                            f"📖 {name}: "
                            f"{len(detected_units)} units "
                            f"detected from semester syllabus."
                        )

                    else:

                        st.warning(
                            f"⚠️ {name}: "
                            "No unit headings were detected "
                            "in the syllabus PDF. "
                            "Individual topics will still work."
                        )


                # ------------------------------------------
                # EXPAND UNITS
                # ------------------------------------------

                analysis_topics, unit_mapping = (
                    expand_syllabus_entries(
                        syllabus,
                        detected_units
                    )
                )


                # Remove duplicates
                analysis_topics = list(
                    dict.fromkeys(
                        analysis_topics
                    )
                )


                # ------------------------------------------
                # ANALYZE PAPERS
                # ------------------------------------------

                with st.spinner(
                    f"Analyzing {name}..."
                ):

                    results, messages = (
                        analyze_subject(
                            name,
                            analysis_topics,
                            papers
                        )
                    )

                for message in messages:

                    st.write(message)


                if results.empty:

                    st.warning(
                        f"{name}: "
                        "No readable questions found."
                    )

                    continue


                priority = calculate_priority(
                    results,
                    analysis_topics,
                    len(papers)
                )


                analyzed_subjects.append({

                    "name":
                        name,

                    "exam_date":
                        exam_date,

                    "results":
                        results,

                    "priority":
                        priority,

                    "unit_mapping":
                        unit_mapping,

                    "selected_syllabus":
                        syllabus
                })


            if analyzed_subjects:

                st.session_state.analyzed = (
                    analyzed_subjects
                )

                st.success(
                    "🎉 All available subjects analyzed successfully!"
                )

            elif failed:

                st.warning(
                    "Fix the missing information "
                    "and analyze again."
                )


    # ========================================================
    # ANALYSIS DASHBOARD
    # ========================================================

    if st.session_state.analyzed:

        analyzed = (
            st.session_state.analyzed
        )

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '📊 Exam Intelligence'
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # SELECT SUBJECT
        # ----------------------------------------------------

        subject_names = [
            x["name"]
            for x in analyzed
        ]

        selected_name = st.selectbox(
            "Select subject to analyze",
            subject_names
        )

        selected = next(
            x
            for x in analyzed
            if x["name"] == selected_name
        )


        # ----------------------------------------------------
        # SUBJECT HEADER
        # ----------------------------------------------------

        days_left = (
            selected["exam_date"]
            - date.today()
        ).days

        a1, a2, a3, a4 = st.columns(
            4
        )

        a1.metric(
            "📚 Subject",
            selected["name"]
        )

        a2.metric(
            "⏳ Days Left",
            max(days_left, 0)
        )

        a3.metric(
            "❓ Questions",
            len(selected["results"])
        )

        a4.metric(
            "🔥 High Priority",
            len(
                selected["priority"][
                    selected["priority"]["Priority"]
                    == "High"
                ]
            )
        )

        st.caption(
            get_urgency(
                selected["exam_date"]
            )
        )


        # ----------------------------------------------------
        # TOP TOPICS
        # ----------------------------------------------------

        st.subheader(
            "🔥 Most Important Topics"
        )

        top_topics = selected[
            "priority"
        ].head(5)

        for _, row in top_topics.iterrows():

            if row["Priority"] == "High":

                st.markdown(
                    f"""
                    <div class="priority-high">
                    🔴 <b>{row['Topic']}</b>
                    — {row['Questions']} questions
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif row["Priority"] == "Medium":

                st.markdown(
                    f"""
                    <div class="priority-medium">
                    🟡 <b>{row['Topic']}</b>
                    — {row['Questions']} questions
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="priority-low">
                    🟢 <b>{row['Topic']}</b>
                    — {row['Questions']} questions
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        st.subheader(
            "📈 Previous-Year Question Frequency"
        )

        chart_df = selected[
            "priority"
        ][
            [
                "Topic",
                "Questions"
            ]
        ].copy()

        chart_df = chart_df[
            chart_df["Questions"] > 0
        ]

        if not chart_df.empty:

            fig = px.bar(
                chart_df,
                x="Questions",
                y="Topic",
                orientation="h",
                title="Questions Asked by Topic"
            )

            fig.update_layout(
                height=max(
                    400,
                    len(chart_df) * 45
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ----------------------------------------------------
        # PRIORITY TABLE
        # ----------------------------------------------------

        st.subheader(
            "🚦 Topic Priority"
        )

        st.dataframe(
            selected["priority"],
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # UNIT MAPPING
        # ----------------------------------------------------

        if not selected[
            "unit_mapping"
        ].empty:

            st.subheader(
                "📖 Syllabus Interpretation"
            )

            st.caption(
                "This shows how the app interpreted "
                "your Unit/topic entries."
            )

            st.dataframe(
                selected[
                    "unit_mapping"
                ],
                use_container_width=True,
                hide_index=True
            )


        # ----------------------------------------------------
        # QUESTION MAPPING
        # ----------------------------------------------------

        st.subheader(
            "🧠 Question → Topic Mapping"
        )

        st.dataframe(
            selected[
                "results"
            ][
                [
                    "Paper",
                    "Question No.",
                    "Question",
                    "Matched Topic",
                    "Match Score"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # DOWNLOAD CSV
        # ----------------------------------------------------

        csv = (
            selected["results"]
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "📥 Download Analysis CSV",
            data=csv,
            file_name=(
                selected["name"]
                .replace(" ", "_")
                +
                "_analysis.csv"
            ),
            mime="text/csv"
        )


# ============================================================
# PAGE 2 — MY STUDY PLAN
# ============================================================

else:

    st.markdown(
        '<div class="section-title">'
        '📅 My Study Plan'
        '</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.analyzed:

        st.info(
            "First go to **📊 Exam Intelligence**, "
            "add your subjects and analyze your papers."
        )

    else:

        analyzed = (
            st.session_state.analyzed
        )


        # ====================================================
        # EXAM OVERVIEW
        # ====================================================

        st.subheader(
            "⏳ Exam Countdown"
        )

        countdown = []

        for subject in analyzed:

            days = (
                subject["exam_date"]
                - date.today()
            ).days

            countdown.append({

                "Subject":
                    subject["name"],

                "Exam Date":
                    subject["exam_date"].strftime(
                        "%d %b %Y"
                    ),

                "Days Left":
                    max(days, 0),

                "Status":
                    get_urgency(
                        subject["exam_date"]
                    )
            })

        countdown_df = pd.DataFrame(
            countdown
        ).sort_values(
            "Days Left"
        )

        st.dataframe(
            countdown_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # WHAT SHOULD I STUDY FIRST?
        # ====================================================

        st.header(
            "🎯 What Should I Study First?"
        )

        recommendations = []

        for subject in analyzed:

            days_left = (
                subject["exam_date"]
                - date.today()
            ).days

            for _, row in subject[
                "priority"
            ].iterrows():

                if row["Questions"] <= 0:
                    continue

                priority_value = {

                    "High": 3,

                    "Medium": 2,

                    "Low": 1

                }[
                    row["Priority"]
                ]

                urgency_bonus = max(
                    0,
                    50 - days_left * 3
                )

                final_score = (
                    priority_value * 100
                    +
                    row["Priority Score"]
                    +
                    urgency_bonus
                )

                recommendations.append({

                    "Score":
                        final_score,

                    "Subject":
                        subject["name"],

                    "Topic":
                        row["Topic"],

                    "Priority":
                        row["Priority"],

                    "Questions":
                        row["Questions"],

                    "Exam Date":
                        subject["exam_date"]
                })


        recommendations.sort(
            key=lambda x: x["Score"],
            reverse=True
        )


        if recommendations:

            first = recommendations[0]

            st.success(
                f"📖 **Study first:** "
                f"{first['Topic']} "
                f"({first['Subject']})\n\n"
                f"🔥 Priority: "
                f"{first['Priority']}  |  "
                f"❓ PYQ Questions: "
                f"{first['Questions']}  |  "
                f"📅 Exam: "
                f"{first['Exam Date'].strftime('%d %b %Y')}"
            )


            rec_df = pd.DataFrame(
                recommendations[:10]
            )

            rec_df["Exam Date"] = (
                rec_df["Exam Date"]
                .apply(
                    lambda x:
                    x.strftime(
                        "%d %b %Y"
                    )
                )
            )

            st.dataframe(
                rec_df[
                    [
                        "Subject",
                        "Topic",
                        "Priority",
                        "Questions",
                        "Exam Date"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # SMART TIMETABLE
        # ====================================================

        st.header(
            "🗓️ Smart Timetable"
        )

        timetable = generate_timetable(
            analyzed,
            daily_hours
        )


        if timetable.empty:

            st.warning(
                "No timetable can be generated "
                "from the current analysis."
            )

        else:

            st.caption(
                f"Timetable generated using "
                f"{daily_hours} study hours/day, "
                f"exam urgency and PYQ priority."
            )


            # ------------------------------------------------
            # DAILY CHECKLIST
            # ------------------------------------------------

            st.subheader(
                "✅ Your Study Checklist"
            )

            unique_dates = (
                timetable["Date"]
                .drop_duplicates()
            )

            for current_date in unique_dates:

                day_tasks = timetable[
                    timetable["Date"]
                    == current_date
                ]

                st.markdown(
                    f"### 📅 "
                    f"{current_date.strftime('%A, %d %B')}"
                )

                for _, task in day_tasks.iterrows():

                    task_id = (
                        f"{current_date}_"
                        f"{task['Subject']}_"
                        f"{task['Topic']}"
                    )

                    task_id = (
                        task_id
                        .replace(" ", "_")
                    )

                    completed = st.checkbox(
                        f"{task['Subject']} — "
                        f"{task['Topic']} "
                        f"({task['Study Time']}) "
                        f"[{task['Priority']}]",
                        key=f"check_{task_id}",
                        value=(
                            task_id
                            in
                            st.session_state.completed_topics
                        )
                    )

                    if completed:

                        st.session_state.completed_topics.add(
                            task_id
                        )

                    else:

                        st.session_state.completed_topics.discard(
                            task_id
                        )

                st.divider()


            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            total_tasks = len(
                timetable
            )

            completed_count = len([
                x
                for x in timetable.itertuples()
                if (
                    f"{x.Date}_"
                    f"{x.Subject}_"
                    f"{x.Topic}"
                )
                .replace(" ", "_")
                in
                st.session_state.completed_topics
            ])

            progress = (
                completed_count
                /
                max(total_tasks, 1)
            )

            st.subheader(
                "📊 Study Progress"
            )

            st.progress(
                progress
            )

            st.write(
                f"**{completed_count} / "
                f"{total_tasks} tasks completed**"
            )

            if progress == 1:

                st.success(
                    "🎉 Excellent! "
                    "You completed your current study plan."
                )

            elif progress >= 0.5:

                st.info(
                    "💪 More than half completed. "
                    "Keep going!"
                )

            else:

                st.caption(
                    "Complete the checklist as you study."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Exam Intelligence Assistant • "
    "Syllabus + PYQs → Priorities → Study Plan"
)
