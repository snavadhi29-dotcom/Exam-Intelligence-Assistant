import streamlit as st
import pandas as pd
import plotly.express as px
import re
import io
from datetime import date, timedelta
from difflib import SequenceMatcher

from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Exam Intelligence Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f7f9fc 0%, #eef3ff 100%);
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 4px;
}

.subtitle {
    color: #667085;
    font-size: 17px;
    margin-bottom: 25px;
}

.hero {
    padding: 28px;
    border-radius: 22px;
    background: linear-gradient(135deg, #ffffff, #eef4ff);
    border: 1px solid #dbe4f0;
    margin-bottom: 25px;
}

.section-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e4e7ec;
    margin-bottom: 18px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e4e7ec;
    text-align: center;
}

.priority-high {
    border-left: 6px solid #ef4444;
}

.priority-medium {
    border-left: 6px solid #f59e0b;
}

.priority-low {
    border-left: 6px solid #22c55e;
}

.small-muted {
    color: #667085;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "subjects" not in st.session_state:
    st.session_state.subjects = []

if "semester_syllabus_text" not in st.session_state:
    st.session_state.semester_syllabus_text = ""

if "semester_syllabus_name" not in st.session_state:
    st.session_state.semester_syllabus_name = ""

if "syllabus_subjects" not in st.session_state:
    st.session_state.syllabus_subjects = {}

if "analyzed" not in st.session_state:
    st.session_state.analyzed = {}

if "dashboard" not in st.session_state:
    st.session_state.dashboard = "Exam Intelligence"

if "completed_topics" not in st.session_state:
    st.session_state.completed_topics = set()


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):
    """
    First try normal PDF text extraction.
    If the PDF is scanned/image-based, use OCR.
    """

    pdf_bytes = uploaded_file.getvalue()

    text_parts = []

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))

        for page in reader.pages:
            txt = page.extract_text()

            if txt:
                text_parts.append(txt)

    except Exception:
        pass

    text = "\n".join(text_parts).strip()

    # OCR fallback
    if len(text) < 300:

        try:
            images = convert_from_bytes(
                pdf_bytes,
                dpi=180
            )

            ocr_parts = []

            for img in images:
                ocr_text = pytesseract.image_to_string(img)

                if ocr_text:
                    ocr_parts.append(ocr_text)

            text = "\n".join(ocr_parts)

        except Exception:
            pass

    return text


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_line(line):
    line = line.replace("\xa0", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# UNIT DETECTION
# =========================================================

def is_unit_heading(line):
    """
    Strict unit detection.

    Accepts things like:
    UNIT 1
    UNIT-I
    UNIT II
    Unit 3:
    MODULE 2

    Does NOT treat arbitrary text as a unit.
    """

    line = clean_line(line)

    patterns = [
        r"^\s*unit\s*[-:]?\s*(\d+|[ivxlcdm]+)\s*$",
        r"^\s*unit\s*[-:]?\s*(\d+|[ivxlcdm]+)\s*[:\-]\s*$",
        r"^\s*module\s*[-:]?\s*(\d+|[ivxlcdm]+)\s*$",
        r"^\s*module\s*[-:]?\s*(\d+|[ivxlcdm]+)\s*[:\-]\s*$",
    ]

    for pattern in patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True

    return False


def extract_unit_number(line):
    line = clean_line(line)

    match = re.search(
        r"(?:unit|module)\s*[-:]?\s*(\d+|[ivxlcdm]+)",
        line,
        re.IGNORECASE
    )

    if not match:
        return None

    value = match.group(1).upper()

    roman_map = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8
    }

    if value.isdigit():
        return int(value)

    return roman_map.get(value)


# =========================================================
# SUBJECT DETECTION
# =========================================================

def looks_like_subject_heading(line):
    """
    Conservative subject-heading detector.

    We do NOT treat every capitalized line as a subject.
    """

    line = clean_line(line)

    if not line:
        return False

    if is_unit_heading(line):
        return False

    # Ignore common document headings
    ignored = [
        "syllabus",
        "curriculum",
        "semester",
        "course",
        "contents",
        "index",
        "b.tech",
        "btech",
        "department",
        "university",
        "davv",
        "institute"
    ]

    lower = line.lower()

    if any(x in lower for x in ignored):
        return False

    # Subject codes often appear with words
    if re.search(r"\b[A-Z]{2,6}\s*[-:]?\s*\d{2,4}\b", line):
        return True

    # Common subject indicators
    subject_words = [
        "mathematics",
        "electronics",
        "chemistry",
        "physics",
        "mechanics",
        "programming",
        "computer",
        "engineering",
        "environment",
        "communication",
        "electrical",
        "thermodynamics",
        "graphics",
        "workshop"
    ]

    if any(word in lower for word in subject_words):
        return True

    return False


def detect_subject_sections(text):
    """
    Detect likely subject headings and capture the text
    until the next subject heading.

    This is intentionally conservative.
    """

    lines = [
        clean_line(x)
        for x in text.splitlines()
        if clean_line(x)
    ]

    candidates = []

    for i, line in enumerate(lines):

        if looks_like_subject_heading(line):

            # Avoid tiny random headings
            if len(line) <= 100:
                candidates.append((i, line))

    # Remove nearby duplicate headings
    filtered = []

    for item in candidates:

        if not filtered:
            filtered.append(item)
            continue

        previous_index, previous_name = filtered[-1]

        if item[0] - previous_index <= 2:
            continue

        filtered.append(item)

    sections = {}

    for idx, (start, subject_name) in enumerate(filtered):

        end = (
            filtered[idx + 1][0]
            if idx + 1 < len(filtered)
            else len(lines)
        )

        section_text = "\n".join(lines[start:end])

        # Require at least one UNIT/MODULE
        if re.search(
            r"\b(unit|module)\s*[-:]?\s*(\d+|i{1,3}|iv|v)\b",
            section_text,
            re.IGNORECASE
        ):
            sections[subject_name] = section_text

    return sections


# =========================================================
# EXTRACT UNITS FROM A SUBJECT SECTION
# =========================================================

def extract_units_from_subject_section(section_text):

    lines = [
        clean_line(x)
        for x in section_text.splitlines()
        if clean_line(x)
    ]

    units = {}

    current_unit = None
    current_lines = []

    for line in lines:

        number = extract_unit_number(line)

        if number is not None and is_unit_heading(line):

            if current_unit is not None:

                units[current_unit] = "\n".join(
                    current_lines
                ).strip()

            current_unit = number
            current_lines = []

        elif current_unit is not None:

            current_lines.append(line)

    if current_unit is not None:
        units[current_unit] = "\n".join(
            current_lines
        ).strip()

    return units


# =========================================================
# SAFE TOPIC EXTRACTION
# =========================================================

def extract_topics_from_unit(unit_text):

    """
    Only extract reasonably topic-like lines.

    We intentionally avoid blindly splitting every sentence,
    because that was causing unrelated words such as
    'democracy' or 'private sector' to become topics.
    """

    lines = [
        clean_line(x)
        for x in unit_text.splitlines()
        if clean_line(x)
    ]

    topics = []

    for line in lines:

        # Remove page-number-only lines
        if re.fullmatch(r"\d+", line):
            continue

        # Ignore obvious metadata
        if len(line) < 5:
            continue

        lower = line.lower()

        if any(
            x in lower
            for x in [
                "text book",
                "reference book",
                "textbook",
                "course outcome",
                "learning outcome",
                "total hours",
                "credits",
                "marks",
                "lecture",
                "tutorial"
            ]
        ):
            continue

        # Avoid extremely long paragraphs
        if len(line) > 180:
            continue

        # Remove leading bullets / numbering
        cleaned = re.sub(
            r"^[\-\•\●\*\d\.\)\(]+\s*",
            "",
            line
        ).strip()

        if len(cleaned) >= 5:
            topics.append(cleaned)

    # Remove duplicates while preserving order
    unique_topics = []

    for topic in topics:

        normalized = normalize_text(topic)

        if not normalized:
            continue

        if normalized not in [
            normalize_text(x)
            for x in unique_topics
        ]:
            unique_topics.append(topic)

    return unique_topics


# =========================================================
# BUILD SYLLABUS DATABASE
# =========================================================

def build_syllabus_database(text):

    sections = detect_subject_sections(text)

    database = {}

    for subject_name, section_text in sections.items():

        units = extract_units_from_subject_section(
            section_text
        )

        database[subject_name] = {
            "raw_text": section_text,
            "units": units,
            "topics": {
                unit_number: extract_topics_from_unit(
                    unit_text
                )
                for unit_number, unit_text in units.items()
            }
        }

    return database


# =========================================================
# FIND BEST SUBJECT MATCH
# =========================================================

def find_best_subject_match(subject_name, syllabus_subjects):

    if not syllabus_subjects:
        return None

    target = normalize_text(subject_name)

    best_name = None
    best_score = 0

    for candidate in syllabus_subjects.keys():

        candidate_norm = normalize_text(candidate)

        score = SequenceMatcher(
            None,
            target,
            candidate_norm
        ).ratio()

        # Word overlap bonus
        target_words = set(target.split())
        candidate_words = set(candidate_norm.split())

        if target_words and candidate_words:

            overlap = len(
                target_words & candidate_words
            ) / len(target_words)

            score = (score * 0.7) + (overlap * 0.3)

        if score > best_score:

            best_score = score
            best_name = candidate

    if best_score >= 0.45:
        return best_name

    return None


# =========================================================
# MATCH QUESTION TO TOPIC
# =========================================================

def tokenize(text):

    return set(
        re.findall(
            r"[a-zA-Z]{3,}",
            normalize_text(text)
        )
    )


def topic_similarity(question, topic):

    q_words = tokenize(question)
    t_words = tokenize(topic)

    if not q_words or not t_words:
        return 0

    overlap = len(q_words & t_words)

    overlap_score = overlap / max(
        1,
        len(t_words)
    )

    fuzzy = SequenceMatcher(
        None,
        normalize_text(question),
        normalize_text(topic)
    ).ratio()

    return (
        overlap_score * 0.65
        +
        fuzzy * 0.35
    )


def extract_questions_from_pdf(uploaded_file):

    text = extract_pdf_text(uploaded_file)

    lines = [
        clean_line(x)
        for x in text.splitlines()
        if clean_line(x)
    ]

    questions = []

    current = ""

    for line in lines:

        # Detect common question numbering
        starts_question = bool(
            re.match(
                r"^(\d+[\.\)]|Q\.?\s*\d+|\d+\s*[\-\:])",
                line,
                re.IGNORECASE
            )
        )

        if starts_question:

            if current:
                questions.append(current.strip())

            current = line

        else:

            if current:
                current += " " + line

    if current:
        questions.append(current.strip())

    # Keep useful questions only
    questions = [
        q for q in questions
        if len(q) > 10
    ]

    return questions


# =========================================================
# ANALYZE SUBJECT
# =========================================================

def analyze_subject(
    subject_name,
    exam_date,
    selected_units,
    syllabus_subject
    ,
    pyq_files
):

    # Collect topics only from selected units
    topics = []

    for unit in selected_units:

        unit_topics = syllabus_subject[
            "topics"
        ].get(unit, [])

        for topic in unit_topics:

            if topic not in topics:
                topics.append(topic)

    # If parser couldn't identify topics,
    # use the unit label rather than inventing topics.
    if not topics:

        for unit in selected_units:
            topics.append(
                f"Unit {unit}"
            )

    # Read PYQs
    all_questions = []

    for file in pyq_files:

        questions = extract_questions_from_pdf(file)

        all_questions.extend(questions)

    results = []

    for topic in topics:

        frequency = 0
        matched_questions = []

        for question in all_questions:

            score = topic_similarity(
                question,
                topic
            )

            if score >= 0.35:

                frequency += 1
                matched_questions.append(
                    question
                )

        results.append({
            "Topic": topic,
            "Frequency": frequency,
            "Questions": matched_questions
        })

    df = pd.DataFrame(results)

    if df.empty:
        return df

    max_frequency = max(
        df["Frequency"].max(),
        1
    )

    today = date.today()

    days_left = max(
        1,
        (exam_date - today).days
    )

    for_index = []

    for _, row in df.iterrows():

        frequency_score = (
            row["Frequency"] /
            max_frequency
        )

        urgency_score = min(
            1,
            7 / days_left
        )

        priority_score = (
            frequency_score * 0.75
            +
            urgency_score * 0.25
        )

        for_index.append(
            round(priority_score * 100, 1)
        )

    df["Priority"] = for_index

    def priority_label(value):

        if value >= 70:
            return "HIGH"

        if value >= 40:
            return "MEDIUM"

        return "LOW"

    df["Priority Level"] = df[
        "Priority"
    ].apply(priority_label)

    df = df.sort_values(
        by=["Priority", "Frequency"],
        ascending=False
    ).reset_index(drop=True)

    return df


# =========================================================
# SMART TIMETABLE
# =========================================================

def create_timetable(analyzed_data):

    today = date.today()

    all_tasks = []

    for subject_name, data in analyzed_data.items():

        exam_date = data["exam_date"]
        df = data["results"]

        if df.empty:
            continue

        for _, row in df.iterrows():

            all_tasks.append({
                "Subject": subject_name,
                "Topic": row["Topic"],
                "Priority": row["Priority"],
                "Exam Date": exam_date
            })

    all_tasks.sort(
        key=lambda x: (
            x["Exam Date"],
            -x["Priority"]
        )
    )

    if not all_tasks:
        return pd.DataFrame()

    max_days = max(
        1,
        max(
            (task["Exam Date"] - today).days
            for task in all_tasks
        )
    )

    timetable = []

    for i in range(max_days + 1):

        current_date = today + timedelta(days=i)

        day_tasks = [
            task
            for task in all_tasks
            if task["Exam Date"] >= current_date
        ]

        if not day_tasks:
            continue

        # Give 3 tasks/day maximum
        selected = day_tasks[:3]

        for task in selected:

            timetable.append({
                "Date": current_date,
                "Subject": task["Subject"],
                "Topic": task["Topic"],
                "Priority": task["Priority"]
            })

    return pd.DataFrame(timetable)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🎓 Exam Intelligence")

    st.caption(
        "Your personal syllabus + PYQ study assistant"
    )

    page = st.radio(
        "Navigation",
        [
            "📊 Exam Intelligence",
            "📅 My Study Plan"
        ],
        index=(
            0
            if st.session_state.dashboard
            == "Exam Intelligence"
            else 1
        )
    )

    st.session_state.dashboard = (
        "Exam Intelligence"
        if "Exam Intelligence" in page
        else "My Study Plan"
    )

    st.divider()

    if st.session_state.semester_syllabus_name:

        st.success(
            "Semester syllabus loaded"
        )

        st.caption(
            st.session_state.semester_syllabus_name
        )

    else:

        st.warning(
            "Upload your semester syllabus first."
        )


# =========================================================
# PAGE 1
# =========================================================

if st.session_state.dashboard == "Exam Intelligence":

    st.markdown(
        '<div class="hero">'
        '<div class="main-title">🎓 Exam Intelligence Assistant</div>'
        '<div class="subtitle">'
        'Turn your syllabus and previous-year papers into a smart study strategy.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # SEMESTER SYLLABUS — UPLOAD ONCE
    # =====================================================

    st.markdown("## 📚 Semester Syllabus")

    st.info(
        "Upload your combined semester syllabus PDF once. "
        "You will NOT need to upload the syllabus separately for every subject."
    )

    syllabus_file = st.file_uploader(
        "Upload Combined Semester Syllabus PDF",
        type=["pdf"],
        key="combined_semester_syllabus"
    )

    if syllabus_file is not None:

        if (
            st.session_state.semester_syllabus_name
            != syllabus_file.name
        ):

            with st.spinner(
                "Reading your combined semester syllabus..."
            ):

                syllabus_text = extract_pdf_text(
                    syllabus_file
                )

                syllabus_database = build_syllabus_database(
                    syllabus_text
                )

                st.session_state.semester_syllabus_text = (
                    syllabus_text
                )

                st.session_state.semester_syllabus_name = (
                    syllabus_file.name
                )

                st.session_state.syllabus_subjects = (
                    syllabus_database
                )

            st.success(
                f"Syllabus loaded successfully. "
                f"Detected {len(syllabus_database)} subject sections."
            )

    if st.session_state.syllabus_subjects:

        with st.expander(
            "🔎 View subjects detected from syllabus"
        ):

            for subject_name, data in (
                st.session_state.syllabus_subjects.items()
            ):

                unit_count = len(
                    data["units"]
                )

                st.write(
                    f"**{subject_name}** — "
                    f"{unit_count} units detected"
                )

    st.divider()

    # =====================================================
    # ADD SUBJECT
    # =====================================================

    st.markdown("## ➕ Add Exam Subject")

    if not st.session_state.syllabus_subjects:

        st.warning(
            "Please upload the combined semester syllabus PDF first."
        )

    else:

        if st.button(
            "➕ Add New Subject",
            use_container_width=True
        ):

            new_id = (
                max(
                    [
                        s["id"]
                        for s in st.session_state.subjects
                    ],
                    default=0
                )
                + 1
            )

            st.session_state.subjects.append({
                "id": new_id,
                "name": "",
                "exam_date": date.today()
                + timedelta(days=7),
                "units": "",
                "pyqs": []
            })

            st.rerun()

    # =====================================================
    # SUBJECT CARDS
    # =====================================================

    for index, subject in enumerate(
        st.session_state.subjects
    ):

        subject_id = subject["id"]

        st.markdown(
            f"### 📘 Subject {index + 1}"
        )

        with st.container(border=True):

            col1, col2 = st.columns([2, 1])

            with col1:

                subject_names = list(
                    st.session_state
                    .syllabus_subjects
                    .keys()
                )

                current_name = subject.get(
                    "name",
                    ""
                )

                default_index = 0

                if current_name in subject_names:
                    default_index = (
                        subject_names.index(
                            current_name
                        )
                        + 1
                    )

                selected_subject = st.selectbox(
                    "Select Subject",
                    ["Select subject..."]
                    + subject_names,
                    index=default_index,
                    key=f"subject_{subject_id}"
                )

            with col2:

                exam_date = st.date_input(
                    "Exam Date",
                    value=subject.get(
                        "exam_date",
                        date.today()
                        + timedelta(days=7)
                    ),
                    key=f"date_{subject_id}"
                )

            if selected_subject != "Select subject...":

                matched_subject = (
                    find_best_subject_match(
                        selected_subject,
                        st.session_state
                        .syllabus_subjects
                    )
                )

                if matched_subject:

                    subject_data = (
                        st.session_state
                        .syllabus_subjects[
                            matched_subject
                        ]
                    )

                    available_units = sorted(
                        subject_data["units"].keys()
                    )

                    st.markdown(
                        "#### 🧾 What is coming in this exam?"
                    )

                    st.caption(
                        "Enter unit numbers only, for example: "
                        "Unit 2, Unit 3. "
                        "You can also enter individual topic names if needed."
                    )

                    units_text = st.text_area(
                        "Units / Topics",
                        value=subject.get(
                            "units",
                            ""
                        ),
                        placeholder=(
                            "Unit 2\n"
                            "Unit 3\n"
                            "or\n"
                            "Beta and Gamma Functions"
                        ),
                        key=f"units_{subject_id}",
                        height=120
                    )

                    if available_units:

                        st.caption(
                            "Units available in this subject: "
                            + ", ".join(
                                [
                                    f"Unit {u}"
                                    for u in available_units
                                ]
                            )
                        )

                    st.markdown(
                        "#### 📄 Previous Year Question Papers"
                    )

                    pyq_files = st.file_uploader(
                        "Upload one or more PYQ PDFs",
                        type=["pdf"],
                        accept_multiple_files=True,
                        key=f"pyq_{subject_id}"
                    )

                    col_save, col_remove = st.columns(
                        [1, 1]
                    )

                    with col_save:

                        if st.button(
                            "💾 Save Subject",
                            key=f"save_{subject_id}",
                            use_container_width=True
                        ):

                            subject["name"] = (
                                selected_subject
                            )

                            subject["exam_date"] = (
                                exam_date
                            )

                            subject["units"] = (
                                units_text
                            )

                            subject["pyqs"] = (
                                pyq_files
                            )

                            st.success(
                                f"{selected_subject} saved."
                            )

                    with col_remove:

                        if st.button(
                            "🗑️ Remove",
                            key=f"remove_{subject_id}",
                            use_container_width=True
                        ):

                            st.session_state.subjects = [
                                s
                                for s in
                                st.session_state.subjects
                                if s["id"] != subject_id
                            ]

                            st.rerun()

    st.divider()

    # =====================================================
    # ANALYZE ALL
    # =====================================================

    if st.session_state.subjects:

        if st.button(
            "🚀 Analyze All Subjects",
            type="primary",
            use_container_width=True
        ):

            if not st.session_state.syllabus_subjects:

                st.error(
                    "Please upload the combined syllabus PDF."
                )

            else:

                analyzed = {}

                for subject in (
                    st.session_state.subjects
                ):

                    name = subject.get(
                        "name"
                    )

                    if not name:
                        continue

                    matched_subject = (
                        find_best_subject_match(
                            name,
                            st.session_state
                            .syllabus_subjects
                        )
                    )

                    if not matched_subject:
                        continue

                    subject_data = (
                        st.session_state
                        .syllabus_subjects[
                            matched_subject
                        ]
                    )

                    # -------------------------------------
                    # Parse user-entered units/topics
                    # -------------------------------------

                    entries = [
                        clean_line(x)
                        for x in subject[
                            "units"
                        ].splitlines()
                        if clean_line(x)
                    ]

                    selected_units = []

                    individual_topics = []

                    for entry in entries:

                        unit_number = (
                            extract_unit_number(
                                entry
                            )
                        )

                        if unit_number is not None:

                            selected_units.append(
                                unit_number
                            )

                        else:

                            individual_topics.append(
                                entry
                            )

                    # -------------------------------------
                    # Build results
                    # -------------------------------------

                    df = analyze_subject(
                        name,
                        subject[
                            "exam_date"
                        ],
                        selected_units,
                        subject_data,
                        subject[
                            "pyqs"
                        ]
                    )

                    # Add explicitly entered topics
                    # if they are not already present
                    if individual_topics:

                        extra_rows = []

                        all_questions = []

                        for file in subject[
                            "pyqs"
                        ]:

                            all_questions.extend(
                                extract_questions_from_pdf(
                                    file
                                )
                            )

                        for topic in individual_topics:

                            frequency = 0
                            matched_questions = []

                            for question in all_questions:

                                score = topic_similarity(
                                    question,
                                    topic
                                )

                                if score >= 0.35:

                                    frequency += 1
                                    matched_questions.append(
                                        question
                                    )

                            extra_rows.append({
                                "Topic": topic,
                                "Frequency": frequency,
                                "Questions": matched_questions,
                                "Priority": 0,
                                "Priority Level": "LOW"
                            })

                        if extra_rows:

                            extra_df = pd.DataFrame(
                                extra_rows
                            )

                            if df.empty:
                                df = extra_df

                            else:
                                df = pd.concat(
                                    [
                                        df,
                                        extra_df
                                    ],
                                    ignore_index=True
                                )

                    analyzed[name] = {
                        "exam_date": subject[
                            "exam_date"
                        ],
                        "results": df,
                        "matched_syllabus_subject":
                            matched_subject
                    }

                st.session_state.analyzed = analyzed

                st.success(
                    "All subjects analyzed successfully."
                )

    # =====================================================
    # SUBJECT ANALYSIS
    # =====================================================

    if st.session_state.analyzed:

        st.divider()

        st.markdown(
            "## 📊 Subject Analysis"
        )

        subject_names = list(
            st.session_state
            .analyzed
            .keys()
        )

        selected_analysis_subject = st.selectbox(
            "Select a subject to analyze",
            subject_names
        )

        data = (
            st.session_state
            .analyzed[
                selected_analysis_subject
            ]
        )

        df = data["results"]

        if df.empty:

            st.warning(
                "No syllabus topics were detected for this subject."
            )

        else:

            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Topics",
                    len(df)
                )

            with col2:
                st.metric(
                    "PYQ Matches",
                    int(df["Frequency"].sum())
                )

            with col3:
                high_count = len(
                    df[
                        df[
                            "Priority Level"
                        ] == "HIGH"
                    ]
                )

                st.metric(
                    "High Priority",
                    high_count
                )

            with col4:

                days_left = max(
                    0,
                    (
                        data["exam_date"]
                        - date.today()
                    ).days
                )

                st.metric(
                    "Days Left",
                    days_left
                )

            st.markdown(
                "### 🎯 Topic Priority"
            )

            display_df = df[
                [
                    "Topic",
                    "Frequency",
                    "Priority",
                    "Priority Level"
                ]
            ].copy()

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                "### 📈 PYQ Frequency"
            )

            chart = px.bar(
                df,
                x="Topic",
                y="Frequency",
                color="Priority Level",
                title="Previous-Year Question Frequency"
            )

            chart.update_layout(
                xaxis_tickangle=-45
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

            st.markdown(
                "### 🔍 Question → Topic Mapping"
            )

            for _, row in df.iterrows():

                if row["Questions"]:

                    with st.expander(
                        f"{row['Topic']} "
                        f"({len(row['Questions'])} matches)"
                    ):

                        for q in row[
                            "Questions"
                        ][:10]:

                            st.write(
                                "• " + q
                            )

            csv = df[
                [
                    "Topic",
                    "Frequency",
                    "Priority",
                    "Priority Level"
                ]
            ].to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Analysis CSV",
                csv,
                f"{selected_analysis_subject}_analysis.csv",
                "text/csv"
            )


# =========================================================
# PAGE 2 — STUDY PLAN
# =========================================================

else:

    st.markdown(
        '<div class="hero">'
        '<div class="main-title">📅 My Study Plan</div>'
        '<div class="subtitle">'
        'Your priority-based study plan and interactive progress tracker.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.analyzed:

        st.info(
            "First analyze your subjects from the Exam Intelligence dashboard."
        )

    else:

        # =================================================
        # COUNTDOWN
        # =================================================

        st.markdown(
            "## ⏳ Exam Countdown"
        )

        countdown_cols = st.columns(
            len(
                st.session_state.analyzed
            )
        )

        for col, (
            subject_name,
            data
        ) in zip(
            countdown_cols,
            st.session_state.analyzed.items()
        ):

            days_left = max(
                0,
                (
                    data["exam_date"]
                    - date.today()
                ).days
            )

            with col:

                st.metric(
                    subject_name,
                    f"{days_left} days"
                )

        # =================================================
        # WHAT TO STUDY FIRST
        # =================================================

        st.divider()

        st.markdown(
            "## 🎯 What Should I Study First?"
        )

        all_priority_rows = []

        for subject_name, data in (
            st.session_state.analyzed.items()
        ):

            df = data["results"]

            for _, row in df.iterrows():

                all_priority_rows.append({
                    "Subject": subject_name,
                    "Topic": row["Topic"],
                    "Priority": row["Priority"],
                    "Frequency": row["Frequency"],
                    "Exam Date": data["exam_date"]
                })

        priority_df = pd.DataFrame(
            all_priority_rows
        )

        if not priority_df.empty:

            priority_df = priority_df.sort_values(
                by=[
                    "Exam Date",
                    "Priority"
                ],
                ascending=[
                    True,
                    False
                ]
            )

            st.dataframe(
                priority_df[
                    [
                        "Subject",
                        "Topic",
                        "Priority",
                        "Frequency",
                        "Exam Date"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        # =================================================
        # TIMETABLE
        # =================================================

        st.divider()

        st.markdown(
            "## 🗓️ Smart Timetable"
        )

        timetable = create_timetable(
            st.session_state.analyzed
        )

        if timetable.empty:

            st.warning(
                "No timetable could be generated."
            )

        else:

            # =================================================
            # PROGRESS
            # =================================================

            total_tasks = len(
                timetable
            )

            completed_count = 0

            for _, task in timetable.iterrows():

                task_id = (
                    f"{task['Date']}_"
                    f"{task['Subject']}_"
                    f"{task['Topic']}"
                ).replace(
                    " ",
                    "_"
                )

                if task_id in (
                    st.session_state
                    .completed_topics
                ):
                    completed_count += 1

            progress = (
                completed_count /
                total_tasks
                if total_tasks
                else 0
            )

            st.markdown(
                "### 📈 Your Progress"
            )

            st.progress(
                progress
            )

            st.write(
                f"**{completed_count} / "
                f"{total_tasks} tasks completed**"
            )

            # =================================================
            # INTERACTIVE CHECKLIST
            # =================================================

            current_date = None

            for _, task in timetable.iterrows():

                task_date = task["Date"]

                if task_date != current_date:

                    current_date = task_date

                    st.markdown(
                        f"### 📅 {task_date.strftime('%d %b %Y')}"
                    )

                task_id = (
                    f"{task_date}_"
                    f"{task['Subject']}_"
                    f"{task['Topic']}"
                ).replace(
                    " ",
                    "_"
                )

                is_done = task_id in (
                    st.session_state
                    .completed_topics
                )

                checked = st.checkbox(
                    f"{task['Subject']} — "
                    f"{task['Topic']}",
                    value=is_done,
                    key=f"check_{task_id}"
                )

                if checked:
                    st.session_state.completed_topics.add(
                        task_id
                    )
                else:
                    st.session_state.completed_topics.discard(
                        task_id
                    )

            st.divider()

            st.success(
                "Tick a topic after completing it. "
                "Your progress is tracked on this dashboard."
            )
