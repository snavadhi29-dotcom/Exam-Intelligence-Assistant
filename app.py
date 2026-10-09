import io
import re
from datetime import date, timedelta
from difflib import SequenceMatcher

import pandas as pd
import plotly.express as px
import streamlit as st
from pypdf import PdfReader

# OCR is optional
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    Image = None
    OCR_AVAILABLE = False

st.set_page_config(
    page_title="Exam Intelligence Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f7f9fc 0%, #eef3ff 100%);
}
.hero {
    padding: 26px;
    border-radius: 20px;
    background: linear-gradient(135deg, #ffffff, #eef4ff);
    border: 1px solid #dbe4f0;
    margin-bottom: 22px;
}
.hero h1 {
    margin: 0;
    font-size: 2.25rem;
}
.muted { color: #667085; }
div[data-testid="stMetric"] {
    background: white;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #e4e7ec;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------

DEFAULTS = {
    "subjects": [],
    "syllabus_text": "",
    "syllabus_name": "",
    "syllabus_db": {},
    "analyzed": {},
    "completed_topics": set(),
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- TEXT AND PDF HELPERS ----------------

def clean_line(line):
    line = str(line).replace("\xa0", " ").replace("\u200b", "")
    line = re.sub(r"\s+", " ", line)
    return line.strip(" \t|")


def normalize(text):
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s.+-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_pdf_bytes(data):
    parts = []

    try:
        reader = PdfReader(io.BytesIO(data))

        for page in reader.pages:
            text = page.extract_text() or ""

            if text.strip():
                parts.append(text)

    except Exception:
        pass

    return "\n".join(parts).strip()


def extract_uploaded_text(uploaded_file):
    """Extract text from PDFs or images."""

    data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        text = read_pdf_bytes(data)

        # Try OCR if the PDF appears to be scanned.
        if len(text) < 80 and OCR_AVAILABLE:
            try:
                import fitz

                doc = fitz.open(
                    stream=data,
                    filetype="pdf"
                )

                ocr_parts = []

                for page in doc:
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(1.5, 1.5)
                    )

                    img = Image.open(
                        io.BytesIO(pix.tobytes("png"))
                    )

                    ocr_parts.append(
                        pytesseract.image_to_string(img)
                    )

                text = "\n".join(ocr_parts).strip()

            except Exception:
                pass

        return text

    if name.endswith(
        (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    ):
        if OCR_AVAILABLE:
            try:
                image = Image.open(io.BytesIO(data))
                return pytesseract.image_to_string(image).strip()
            except Exception:
                return ""

        return ""

    return ""


def is_unit_heading(line):
    return bool(
        re.match(
            r"^\s*(?:unit|module)\s*[-:]?\s*"
            r"(?:\d+|[ivxlcdm]+)\s*[:\-]?\s*$",
            clean_line(line),
            re.I,
        )
    )


def unit_number(line):
    match = re.search(
        r"\b(?:unit|module)\s*[-:]?\s*(\d+|[ivxlcdm]+)",
        clean_line(line),
        re.I,
    )

    if not match:
        return None

    token = match.group(1).upper()

    roman = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8,
        "IX": 9,
        "X": 10,
    }

    return int(token) if token.isdigit() else roman.get(token)


def is_noise(line):
    low = normalize(line)

    return (
        len(line) < 4
        or len(line) > 260
        or re.fullmatch(r"[\d\s./-]+", line) is not None
        or any(
            x in low
            for x in [
                "text book",
                "textbook",
                "reference book",
                "course outcome",
                "learning outcome",
                "total hours",
                "credits",
                "lecture hours",
                "tutorial hours",
                "practical hours",
                "course code",
                "page no",
            ]
        )
    )


def topic_lines(unit_text):
    """Extract candidate topic lines from a unit."""

    topics = []

    for raw in unit_text.splitlines():
        line = clean_line(raw)

        if is_noise(line) or is_unit_heading(line):
            continue

        line = re.sub(r"^[\s•●▪◦*\-–—]+", "", line)

        line = re.sub(
            r"^\(?\d+(?:\.\d+)*[.)]?\s+",
            "",
            line,
        )

        if len(line) >= 4:
            existing = {normalize(x) for x in topics}

            if normalize(line) not in existing:
                topics.append(line)

    expanded = []

    for topic in topics:
        pieces = re.split(
            r"\s+[;•]\s+|\s+\|\s+",
            topic,
        )

        expanded.extend(
            p.strip()
            for p in pieces
            if len(p.strip()) >= 4
        )

    return list(dict.fromkeys(expanded))


def build_syllabus_database(text):
    """
    Detect subject headings and their unit/topic sections.
    Always review the extracted topics in the app.
    """

    lines = [
        clean_line(x)
        for x in text.splitlines()
        if clean_line(x)
    ]

    subject_words = (
        "mathematics",
        "maths",
        "electronics",
        "chemistry",
        "physics",
        "mechanical",
        "programming",
        "computer",
        "engineering",
        "environment",
        "english",
        "humanities",
        "graphics",
        "design thinking",
        "electrical",
    )

    candidates = []

    for i, line in enumerate(lines):
        low = line.lower()

        if (
            len(line) <= 120
            and not is_unit_heading(line)
            and any(word in low for word in subject_words)
        ):
            if not any(
                word in low
                for word in [
                    "syllabus",
                    "curriculum",
                    "department",
                    "university",
                ]
            ):
                candidates.append((i, line))

    filtered = []

    for item in candidates:
        if filtered and item[0] - filtered[-1][0] <= 2:
            continue

        filtered.append(item)

    sections = {}

    for idx, (start, name) in enumerate(filtered):
        if idx + 1 < len(filtered):
            end = filtered[idx + 1][0]
        else:
            end = len(lines)

        section = lines[start:end]

        if any(is_unit_heading(x) for x in section):
            sections[name] = "\n".join(section)

    database = {}

    for name, section in sections.items():
        units = {}
        current = None
        buffer = []

        for line in section.splitlines():
            number = unit_number(line)

            if number is not None and is_unit_heading(line):
                if current is not None:
                    units[current] = "\n".join(buffer).strip()

                current = number
                buffer = []

            elif current is not None:
                buffer.append(line)

        if current is not None:
            units[current] = "\n".join(buffer).strip()

        if units:
            database[name] = {
                "raw_text": section,
                "units": units,
                "topics": {
                    number: topic_lines(body)
                    for number, body in units.items()
                },
            }

    return database


def best_subject_match(name, database):
    if not database:
        return None, 0.0

    target = normalize(name)
    best = None
    best_score = 0.0

    for candidate in database:
        cand = normalize(candidate)

        sequence_score = SequenceMatcher(
            None,
            target,
            cand,
        ).ratio()

        target_words = set(target.split())
        candidate_words = set(cand.split())

        overlap = (
            len(target_words & candidate_words)
            / max(1, len(target_words))
        )

        score = (
            0.65 * sequence_score
            + 0.35 * overlap
        )

        if score > best_score:
            best = candidate
            best_score = score

    if best_score >= 0.38:
        return best, best_score

    return None, best_score
    # ---------------- QUESTION EXTRACTION ----------------

QUESTION_START = re.compile(
    r"^\s*(?:(?:Q(?:uestion)?\.?\s*)?\d{1,2}\s*[\).:]|"
    r"(?:[a-hA-H]\s*[\).])|"
    r"(?:SECTION\s+[A-Z0-9]+)|"
    r"(?:PART\s+[A-Z0-9]+))",
    re.I,
)


def split_questions(text):
    """Group numbered question lines and their continuation lines."""

    lines = [
        clean_line(x)
        for x in text.splitlines()
        if clean_line(x)
    ]

    questions = []
    current = ""

    for line in lines:
        starts = bool(QUESTION_START.match(line))

        if starts:
            if current and len(current) >= 12:
                questions.append(current.strip())

            current = line

        elif current:
            current += " " + line

        elif (
            len(line) > 30
            and (
                "?" in line
                or re.search(
                    r"\b(explain|derive|calculate|define|discuss|"
                    r"write|find|state)\b",
                    line,
                    re.I,
                )
            )
        ):
            current = line

    if current and len(current) >= 12:
        questions.append(current.strip())

    # Fallback for papers without numbered questions.
    if not questions:
        questions = [
            clean_line(x)
            for x in re.split(r"(?<=[?])\s+|\n+", text)
            if len(clean_line(x)) >= 18
        ]

    unique = []
    seen = set()

    for question in questions:
        key = normalize(question)

        if key and key not in seen:
            seen.add(key)
            unique.append(question)

    return unique


def marks_from_question(question):
    patterns = [
        r"\[\s*(\d{1,2})\s*marks?\s*\]",
        r"\(\s*(\d{1,2})\s*marks?\s*\)",
        r"\b(\d{1,2})\s*marks?\b",
        r"\b(\d{1,2})\s*M\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, re.I)

        if match:
            value = int(match.group(1))

            if 1 <= value <= 30:
                return value

    return None


def classify_question(question):
    q = question.lower()

    if re.search(
        r"\b(derive|derivation|prove that|proof of|show that)\b",
        q,
    ):
        return "Derivation"

    if (
        re.search(
            r"\b(calculate|compute|solve|find the value|"
            r"numerical|determine the value)\b",
            q,
        )
        or re.search(
            r"\b\d+(?:\.\d+)?"\s*(
            # ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## 🎓 Exam Intelligence")
    st.caption("Syllabus + previous-paper analysis")

    page = st.radio(
        "Navigation",
        [
            "📊 Exam Intelligence",
            "📅 My Study Plan",
        ],
    )

    st.divider()

    if st.session_state.syllabus_name:
        st.success("Syllabus loaded")
        st.caption(st.session_state.syllabus_name)
    else:
        st.info("Upload your semester syllabus PDF to begin.")


# ---------------- MAIN DASHBOARD ----------------

if page == "📊 Exam Intelligence":

    st.markdown("""
    <div class="hero">
        <h1>🎓 Exam Intelligence Assistant</h1>
        <p class="muted">
            Use syllabus topics and past papers to build
            an evidence-based revision plan.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📚 Semester Syllabus")

    st.caption(
        "Upload your combined syllabus PDF once. "
        "No semester-selection option is needed."
    )

    syllabus_file = st.file_uploader(
        "Upload combined syllabus PDF",
        type=["pdf"],
        key="syllabus_pdf",
    )

    if syllabus_file is not None:
        if syllabus_file.name != st.session_state.syllabus_name:

            with st.spinner(
                "Extracting subjects, units and topics..."
            ):
                syllabus_text = extract_uploaded_text(
                    syllabus_file
                )

                database = build_syllabus_database(
                    syllabus_text
                )

                st.session_state.syllabus_text = syllabus_text
                st.session_state.syllabus_name = syllabus_file.name
                st.session_state.syllabus_db = database
                st.session_state.analyzed = {}

            if not syllabus_text:
                st.error(
                    "No readable text found. If this is a scanned "
                    "PDF, OCR may not be available."
                )

            elif not database:
                st.warning(
                    "Text was extracted, but subject/unit headings "
                    "were not detected reliably. Review the extracted "
                    "text and try a clearer syllabus PDF."
                )

            else:
                st.success(
                    f"Loaded syllabus. Detected "
                    f"{len(database)} subject section(s). "
                    "Review the extracted topics before analysis."
                )

    if st.session_state.syllabus_text:
        with st.expander("Preview extracted syllabus text"):
            st.text_area(
                "Extracted text",
                st.session_state.syllabus_text[:12000],
                height=220,
                disabled=True,
            )

    database = st.session_state.syllabus_db

    # Review the automatically detected syllabus.
    if database:

        with st.expander(
            "Review detected subjects, units and topics"
        ):
            for subject_name, data in database.items():
                st.markdown(f"**{subject_name}**")

                for unit in sorted(data["units"]):
                    topics = data["topics"].get(unit, [])

                    st.write(
                        f"Unit {unit}: {len(topics)} topic line(s)"
                    )

                    if topics:
                        st.caption(
                            " • ".join(topics[:12])
                        )
                    else:
                        st.warning(
                            f"Unit {unit}: no topics detected. "
                            "Inspect the extracted PDF text."
                        )

        st.divider()
        st.markdown("## ➕ Add Exam Subject")

        if st.button(
            "➕ Add New Subject",
            use_container_width=True,
        ):
            new_id = max(
                [
                    subject["id"]
                    for subject in st.session_state.subjects
                ],
                default=0,
            ) + 1

            st.session_state.subjects.append({
                "id": new_id,
                "name": "",
                "exam_date": date.today() + timedelta(days=7),
                "selection_mode": "Whole unit(s)",
                "selected_units": [],
                "selected_topics": [],
                "selected_topic_pairs": [],
                "pyqs": [],
                "manual_topics": "",
            })

            st.rerun()

        # Configure each subject.
        for index, subject in enumerate(
            st.session_state.subjects
        ):
            subject_id = subject["id"]

            with st.container(border=True):

                st.markdown(f"### 📘 Subject {index + 1}")

                names = list(database.keys())

                if subject.get("name") in names:
                    default_index = names.index(
                        subject["name"]
                    ) + 1
                else:
                    default_index = 0

                selected_name = st.selectbox(
                    "Choose subject",
                    ["Select subject..."] + names,
                    index=default_index,
                    key=f"name_{subject_id}",
                )

                exam_date = st.date_input(
                    "Exam date",
                    value=subject.get(
                        "exam_date",
                        date.today() + timedelta(days=7),
                    ),
                    key=f"examdate_{subject_id}",
                    min_value=date.today(),
                )

                if selected_name != "Select subject...":

                    subject_data = database[selected_name]

                    available_units = sorted(
                        subject_data["units"]
                    )

                    modes = [
                        "Whole unit(s)",
                        "Individual topics",
                        "Mix of units and topics",
                    ]

                    current_mode = subject.get(
                        "selection_mode",
                        "Whole unit(s)",
                    )

                    mode = st.radio(
                        "What is included in the exam?",
                        modes,
                        index=modes.index(current_mode),
                        horizontal=True,
                        key=f"mode_{subject_id}",
                    )

                    chosen_units = []
                    chosen_topics = []

                    # Whole units or a mix.
                    if mode in [
                        "Whole unit(s)",
                        "Mix of units and topics",
                    ]:
                        chosen_units = st.multiselect(
                            "Select complete units",
                            available_units,
                            default=[
                                unit
                                for unit in subject.get(
                                    "selected_units", []
                                )
                                if unit in available_units
                            ],
                            format_func=lambda unit:
                                f"Unit {unit}",
                            key=f"units_{subject_id}",
                        )

                    # Individual topics or a mix.
                    if mode in [
                        "Individual topics",
                        "Mix of units and topics",
                    ]:
                        topic_pairs = [
                            (unit, topic)
                            for unit in available_units
                            for topic in subject_data[
                                "topics"
                            ].get(unit, [])
                        ]

                        topic_labels = {
                            f"Unit {unit} — {topic}":
                                (unit, topic)
                            for unit, topic in topic_pairs
                        }

                        previous_pairs = subject.get(
                            "selected_topic_pairs", []
                        )

                        default_labels = [
                            f"Unit {unit} — {topic}"
                            for unit, topic in previous_pairs
                            if f"Unit {unit} — {topic}"
                            in topic_labels
                        ]

                        chosen_labels = st.multiselect(
                            "Select specific topics",
                            list(topic_labels.keys()),
                            default=default_labels,
                            key=f"topics_{subject_id}",
                        )

                        chosen_topics = [
                            topic_labels[label]
                            for label in chosen_labels
                        ]

                    manual_topics = st.text_area(
                        "Optional extra topics (one per line)",
                        value=subject.get(
                            "manual_topics", ""
                        ),
                        placeholder=(
                            "Add a topic if it was not detected "
                            "from the syllabus"
                        ),
                        key=f"manual_{subject_id}",
                    )

                    st.markdown(
                        "#### 📄 Previous-year question papers"
                    )

                    pyq_files = st.file_uploader(
                        "Upload PYQ PDFs or question-paper images",
                        type=[
                            "pdf", "png", "jpg", "jpeg",
                            "tif", "tiff", "bmp",
                        ],
                        accept_multiple_files=True,
                        key=f"pyqs_{subject_id}",
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        save_subject = st.button(
                            "💾 Save Subject",
                            key=f"save_{subject_id}",
                            use_container_width=True,
                        )

                    with col2:
                        remove_subject = st.button(
                            "🗑️ Remove",
                            key=f"remove_{subject_id}",
                            use_container_width=True,
                        )

                    if save_subject:
                        subject.update({
                            "name": selected_name,
                            "exam_date": exam_date,
                            "selection_mode": mode,
                            "selected_units": chosen_units,
                            "selected_topic_pairs": chosen_topics,
                            "selected_topics": [
                                topic
                                for _, topic in chosen_topics
                            ],
                            "manual_topics": manual_topics,
                            "pyqs": (
                                pyq_files
                                if pyq_files
                                else subject.get("pyqs", [])
                            ),
                        })

                        st.success(
                            f"Saved {selected_name}."
                        )

                    if remove_subject:
                        st.session_state.subjects = [
                            item
                            for item in st.session_state.subjects
                            if item["id"] != subject_id
                        ]

                        st.rerun()

        st.divider()

        # Run analysis for saved subjects.
        if (
            st.session_state.subjects
            and st.button(
                "🚀 Analyze all saved subjects",
                type="primary",
                use_container_width=True,
            )
        ):
            analyzed = {}

            with st.spinner(
                "Extracting questions, mapping topics "
                "and calculating priorities..."
            ):

                for subject in st.session_state.subjects:

                    name = subject.get("name")

                    if not name or name not in database:
                        continue

                    subject_data = database[name]

                    selected_units = subject.get(
                        "selected_units", []
                    )

                    topic_unit = {}
                    topics = []

                    # Add every topic from selected units.
                    for unit in selected_units:
                        for topic in subject_data[
                            "topics"
                        ].get(unit, []):

                            if topic not in topics:
                                topics.append(topic)
                                topic_unit[topic] = (
                                    f"Unit {unit}"
                                )

                    # Add individually selected topics.
                    for unit, topic in subject.get(
                        "selected_topic_pairs", []
                    ):
                        if topic not in topics:
                            topics.append(topic)
                            topic_unit[topic] = (
                                f"Unit {unit}"
                            )

                    # Add any manually entered topics.
                    for topic in subject.get(
                        "manual_topics", ""
                    ).splitlines():

                        topic = clean_line(topic)

                        if topic and topic not in topics:
                            topics.append(topic)
                            topic_unit[topic] = (
                                "Manually added"
                            )

                    if not topics:
                        st.warning(
                            f"No topics selected for {name}. "
                            "Select at least one unit/topic and save."
                        )
                        continue

                    question_records = []

                    for file in subject.get("pyqs", []):
                        raw_text = extract_uploaded_text(file)
                        questions = split_questions(raw_text)

                        for question in questions:
                            question_records.append({
                                "Question": question,
                                "Paper": file.name,
                            })

                    question_df = pd.DataFrame(
                        question_records
                    )

                    if question_df.empty:
                        mapped = pd.DataFrame(
                            columns=[
                                "Question",
                                "Paper",
                                "Topic",
                                "Match confidence (%)",
                                "Type",
                                "Difficulty",
                                "Marks",
                            ]
                        )

                    else:
                        mapped_parts = []

                        for paper_name, group in (
                            question_df.groupby("Paper")
                        ):
                            part = map_questions_to_topics(
                                group["Question"].tolist(),
                                topics,
                            )

                            part["Paper"] = paper_name
                            mapped_parts.append(part)

                        if mapped_parts:
                            mapped = pd.concat(
                                mapped_parts,
                                ignore_index=True,
                            )
                        else:
                            mapped = pd.DataFrame()

                    results = analyze_topics(
                        topics,
                        mapped,
                        subject["exam_date"],
                        topic_unit,
                    )

                    analyzed[name] = {
                        "exam_date": subject["exam_date"],
                        "results": results,
                        "mapping": mapped,
                        "matched_syllabus_subject": name,
                    }

            st.session_state.analyzed = analyzed

            if analyzed:
                st.success(
                    "Analysis complete. Review the rankings, "
                    "question mapping, trends and evidence confidence."
                )
            else:
                st.error(
                    "No subjects were analyzed. Save a subject "
                    "and select at least one unit or topic."
                )

    # ---------------- DISPLAY ANALYSIS RESULTS ----------------

    if st.session_state.analyzed:

        st.divider()
        st.markdown("## 📊 Analysis Results")

        subject_name = st.selectbox(
            "View results for",
            list(st.session_state.analyzed.keys()),
            key="results_subject",
        )

        data = st.session_state.analyzed[subject_name]

        df = data["results"]
        mapped = data["mapping"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Topics ranked", len(df))
        col2.metric("Questions extracted", len(mapped))

        col3.metric(
            "High priority topics",
            int((df["Priority"] == "HIGH").sum()),
        )

        col4.metric(
            "Days until exam",
            max(
                0,
                (data["exam_date"] - date.today()).days,
            ),
        )

        st.markdown("### 🎯 Ranked study priority")

        st.caption(
            "Priority is a heuristic based on observed question "
            "frequency, labelled marks, hard-question count and "
            "exam urgency. It does not guarantee what will appear."
        )

        display_columns = [
            "Topic",
            "Unit(s)",
            "Question Frequency",
            "Known Marks",
            "Theory",
            "Numerical",
            "Derivation",
            "Hard Questions",
            "Priority Score",
            "Priority",
            "Evidence Confidence (%)",
            "Reason",
        ]

        st.dataframe(
            df[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        if not df.empty:

            st.plotly_chart(
                px.bar(
                    df,
                    x="Topic",
                    y="Question Frequency",
                    color="Priority",
                    title="Topic-wise frequency in uploaded papers",
                ),
                use_container_width=True,
            )

            if not mapped.empty:

                type_totals = (
                    mapped["Type"]
                    .value_counts()
                    .rename_axis("Question type")
                    .reset_index(name="Count")
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.plotly_chart(
                        px.pie(
                            type_totals,
                            names="Question type",
                            values="Count",
                            title="Question-type distribution",
                        ),
                        use_container_width=True,
                    )

                with col2:
                    difficulty_totals = (
                        mapped["Difficulty"]
                        .value_counts()
                        .rename_axis("Difficulty")
                        .reset_index(name="Count")
                    )

                    st.plotly_chart(
                        px.bar(
                            difficulty_totals,
                            x="Difficulty",
                            y="Count",
                            title="Difficulty distribution",
                        ),
                        use_container_width=True,
                    )

        # Question-to-topic mapping.
        st.markdown("### 🔍 Question → topic mapping")

        if mapped.empty:
            st.info(
                "No questions were extracted. Check whether "
                "the papers contain selectable text or whether "
                "OCR is available for scanned files."
            )

        else:
            st.dataframe(
                mapped[
                    [
                        "Paper",
                        "Question",
                        "Topic",
                        "Match confidence (%)",
                        "Type",
                        "Difficulty",
                        "Marks",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            with st.expander(
                "Show questions that need manual review"
            ):
                uncertain = mapped[
                    (
                        mapped["Topic"]
                        == "Unmapped / review manually"
                    )
                    | (
                        mapped["Match confidence (%)"] < 25
                    )
                ]

                if uncertain.empty:
                    st.success(
                        "No low-confidence mappings were flagged "
                        "by this threshold."
                    )
                else:
                    st.dataframe(
                        uncertain,
                        use_container_width=True,
                        hide_index=True,
                    )

        # Trends by uploaded paper.
        st.markdown("### 📈 Trends across uploaded papers")

        if not mapped.empty:
            trend = (
                mapped.groupby(["Paper", "Topic"])
                .size()
                .reset_index(name="Question count")
            )

            st.plotly_chart(
                px.bar(
                    trend,
                    x="Paper",
                    y="Question count",
                    color="Topic",
                    barmode="group",
                    title="Question frequency by uploaded paper",
                ),
                use_container_width=True,
            )

            st.caption(
                "Paper filenames are used as year labels because "
                "the app cannot reliably infer the year from every PDF. "
                "Use filenames such as 2024_MST1.pdf for clearer trends."
            )

        st.download_button(
            "⬇️ Download topic priority CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=(
                f"{subject_name.replace(' ', '_')}_priority.csv"
            ),
            mime="text/csv",
        )

        if not mapped.empty:
            st.download_button(
                "⬇️ Download question mapping CSV",
                mapped.to_csv(index=False).encode("utf-8"),
                file_name=(
                    f"{subject_name.replace(' ', '_')}_question_mapping.csv"
                ),
                mime="text/csv",
            )


# ---------------- MY STUDY PLAN ----------------

else:

    st.markdown("""
    <div class="hero">
        <h1>📅 My Study Plan</h1>
        <p class="muted">
            Revision priorities and progress checklist.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.analyzed:

        st.info(
            "First upload the syllabus, save a subject "
            "and run the analysis."
        )

    else:

        for name, data in st.session_state.analyzed.items():
            days_left = max(
                0,
                (data["exam_date"] - date.today()).days,
            )

            st.metric(
                f"{name} exam countdown",
                f"{days_left} days",
            )

        rows = []

        for subject_name, data in (
            st.session_state.analyzed.items()
        ):
            for _, row in data["results"].iterrows():
                rows.append({
                    "Subject": subject_name,
                    "Topic": row["Topic"],
                    "Priority Score": row["Priority Score"],
                    "Exam Date": data["exam_date"],
                })

        plan = pd.DataFrame(rows)

        if not plan.empty:

            plan = plan.sort_values(
                ["Exam Date", "Priority Score"],
                ascending=[True, False],
            )

            st.markdown("## 🎯 What should I study first?")

            st.dataframe(
                plan,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("## 🗓️ Revision checklist")

            done = 0

            for _, task in plan.iterrows():

                task_id = (
                    f"{task['Subject']}|{task['Topic']}"
                )

                checked = st.checkbox(
                    (
                        f"{task['Exam Date'].strftime('%d %b')} — "
                        f"{task['Subject']}: {task['Topic']}"
                    ),
                    value=(
                        task_id
                        in st.session_state.completed_topics
                    ),
                    key=(
                        "done_"
                        + re.sub(
                            r"[^a-zA-Z0-9]",
                            "_",
                            task_id,
                        )
                    ),
                )

                if checked:
                    st.session_state.completed_topics.add(
                        task_id
                    )
                else:
                    st.session_state.completed_topics.discard(
                        task_id
                    )

                done += int(checked)

            st.progress(done / len(plan))

            st.caption(
                f"{done} of {len(plan)} listed topic tasks checked off."
            )

            st.download_button(
                "⬇️ Download revision plan CSV",
                plan.to_csv(index=False).encode("utf-8"),
                file_name="revision_plan.csv",
                mime="text/csv",
            )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Important: PDF extraction and topic matching are heuristic. "
    "Review extracted topics and low-confidence mappings before "
    "using the results to decide what to study."
)
