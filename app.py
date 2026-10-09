
import io
import re
from datetime import date, timedelta
from difflib import SequenceMatcher

import pandas as pd
import plotly.express as px
import streamlit as st
from pypdf import PdfReader


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Exam Intelligence Assistant",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff, #edf3ff);
}
.hero {
    background: white;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #dce4f2;
    margin-bottom: 20px;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #e4e7ec;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================
DEFAULTS = {
    "syllabus_name": "",
    "syllabus_text": "",
    "subject_headings": [],
    "syllabus_db": {},
    "exam_configs": {},
    "analysis_results": {},
    "completed_tasks": set()
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# GENERAL HELPERS
# =========================================================
def clean_line(value):
    value = str(value).replace("\xa0", " ")
    value = value.replace("\u200b", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t|")


def normalize(value):
    value = str(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def unique_preserve(items):
    output = []
    seen = set()

    for item in items:
        item = clean_line(item)
        key = normalize(item)

        if key and key not in seen:
            output.append(item)
            seen.add(key)

    return output


def extract_pdf(uploaded_file):
    try:
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        page_text = []

        for page in reader.pages:
            page_text.append(page.extract_text() or "")

        return "\n".join(page_text).strip()

    except Exception as exc:
        st.error(f"Unable to read PDF: {exc}")
        return ""


# =========================================================
# SUBJECT DETECTION
# =========================================================
KNOWN_SUBJECTS = [
    "Applied Mathematics",
    "Applied Chemistry and Environmental Science",
    "Applied Chemistry",
    "Engineering Physics",
    "Basic Electrical Engineering",
    "Basic Electronics Engineering",
    "Basic Mechanical Engineering",
    "Engineering Mechanics",
    "Engineering Graphics",
    "Engineering Materials",
    "Programming for Problem Solving",
    "Computer Programming",
    "Computer Fundamentals",
    "Communication Skills",
    "Professional Communication",
    "Environmental Science",
    "Engineering Mathematics",
    "Design Thinking",
    "Workshop Practice",
    "Engineering Chemistry",
    "Engineering Physics and Materials Science"
]


def detect_subject_headings(text):
    """
    Suggest subject headings from the PDF.
    Suggestions must be reviewed before analysis.
    """
    lines = [clean_line(x) for x in text.splitlines() if clean_line(x)]
    detected = []

    for line in lines:
        if len(line) > 150:
            continue

        low = normalize(line)

        for subject in KNOWN_SUBJECTS:
            subject_normalized = normalize(subject)

            if subject_normalized in low:
                detected.append(subject)
                break

    return unique_preserve(detected)


def heading_matches(line, heading):
    """
    Match a heading only at a line boundary, rather than matching
    ordinary topic descriptions containing the same words.
    """
    line_n = normalize(line)
    heading_n = normalize(heading)

    if not line_n or not heading_n:
        return False

    if line_n == heading_n:
        return True

    if line_n.startswith(heading_n + " "):
        remainder = line_n[len(heading_n):].strip()

        # Allow course codes or title suffixes following the heading.
        return len(remainder.split()) <= 8

    return False


def locate_subject_sections(text, headings):
    """
    Divide the syllabus into independent subject sections.
    No topic is automatically shared between subject sections.
    """
    lines = [clean_line(x) for x in text.splitlines()]

    positions = []

    for index, line in enumerate(lines):
        for heading in headings:
            if heading_matches(line, heading):
                positions.append((index, heading))
                break

    # Retain the first occurrence of each heading.
    selected = []
    seen = set()

    for index, heading in positions:
        key = normalize(heading)

        if key not in seen:
            selected.append((index, heading))
            seen.add(key)

    selected.sort(key=lambda item: item[0])

    sections = {}

    for i, (start, heading) in enumerate(selected):
        end = (
            selected[i + 1][0]
            if i + 1 < len(selected)
            else len(lines)
        )

        sections[heading] = lines[start + 1:end]

    return sections


# =========================================================
# UNIT AND TOPIC EXTRACTION
# =========================================================
def get_unit_number(line):
    """
    Recognise headings such as UNIT-I, UNIT 1, Unit: 2,
    MODULE III, and Module 4.
    """
    text = clean_line(line)

    match = re.match(
        r"^\s*(?:unit|module)\s*[-:]?\s*"
        r"(VIII|VII|VI|IV|V|III|II|I|[1-9]\d*)"
        r"\b\s*[:.\-–—]?",
        text,
        re.I
    )

    if not match:
        return None

    value = match.group(1).upper()

    if value.isdigit():
        return int(value)

    roman = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8
    }

    return roman.get(value)


def is_noise_line(line):
    text = normalize(line)

    if len(text) < 3:
        return True

    if re.fullmatch(r"[\d\s./-]+", line or ""):
        return True

    ignored = [
        "textbook",
        "reference books",
        "reference book",
        "course outcomes",
        "learning outcomes",
        "total credits",
        "lecture hours",
        "tutorial hours",
        "practical hours",
        "course code",
        "page number",
        "contents",
        "table of contents"
    ]

    return any(word in text for word in ignored)


def split_topic_line(line):
    """
    Split clear syllabus separators.
    Avoid splitting ordinary commas, because commas often join
    closely related concepts.
    """
    line = clean_line(line)

    line = re.sub(r"^[•●▪◦*–—-]+\s*", "", line)

    # Remove numbered list prefixes such as 1.1 or 2).
    line = re.sub(
        r"^\(?\d+(?:\.\d+)*[.)]?\s+",
        "",
        line
    )

    parts = re.split(r"\s+[;|]\s+|\s+[•●▪◦]\s+", line)

    return [
        clean_line(part)
        for part in parts
        if len(clean_line(part)) >= 3
    ]


def extract_units(section_lines):
    """
    Extract units within one subject only.
    A unit belongs to the subject section in which it was found.
    """
    units = {}
    current_unit = None

    for raw_line in section_lines:
        line = clean_line(raw_line)

        if not line:
            continue

        number = get_unit_number(line)

        if number is not None:
            current_unit = number

            if current_unit not in units:
                units[current_unit] = []

            # Preserve descriptive text appearing on the same line
            # after the UNIT heading.
            remainder = re.sub(
                r"^\s*(?:unit|module)\s*[-:]?\s*"
                r"(?:VIII|VII|VI|IV|V|III|II|I|[1-9]\d*)"
                r"\b\s*[:.\-–—]?\s*",
                "",
                line,
                flags=re.I
            ).strip()

            if remainder and not is_noise_line(remainder):
                units[current_unit].extend(
                    split_topic_line(remainder)
                )

            continue

        if current_unit is None:
            continue

        if is_noise_line(line):
            continue

        units[current_unit].extend(split_topic_line(line))

    cleaned_units = {}

    for number, topics in units.items():
        cleaned_units[number] = unique_preserve(topics)

    return cleaned_units


def build_syllabus_database(text, headings):
    sections = locate_subject_sections(text, headings)
    database = {}

    for subject, section_lines in sections.items():
        units = extract_units(section_lines)

        if units:
            database[subject] = {
                "units": units
            }

    return database


# =========================================================
# QUESTION EXTRACTION
# =========================================================
QUESTION_START = re.compile(
    r"^\s*(?:(?:Q(?:uestion)?\.?\s*)?\d{1,2}\s*[\).:]|"
    r"[a-hA-H]\s*[\).])"
)


def extract_questions(uploaded_file):
    text = extract_pdf(uploaded_file)

    if not text:
        return []

    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    questions = []
    current = []

    for line in lines:
        if QUESTION_START.match(line):
            if current:
                questions.append(" ".join(current))

            current = [line]

        elif current:
            current.append(line)

        elif "?" in line and len(line) > 20:
            questions.append(line)

    if current:
        questions.append(" ".join(current))

    if not questions:
        paragraphs = re.split(r"\n{2,}", text)

        questions = [
            clean_line(paragraph)
            for paragraph in paragraphs
            if len(clean_line(paragraph)) >= 20
        ]

    # Avoid returning duplicate extracted questions.
    return unique_preserve(questions)


def get_marks(question):
    patterns = [
        r"\[\s*(\d{1,2})\s*marks?\s*\]",
        r"\(\s*(\d{1,2})\s*marks?\s*\)",
        r"\b(\d{1,2})\s*marks?\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, question, re.I)

        if match:
            marks = int(match.group(1))

            if 1 <= marks <= 30:
                return marks

    return None


def get_question_type(question):
    q = question.lower()

    if re.search(
        r"\b(derive|derivation|prove that|proof of|show that)\b",
        q
    ):
        return "Derivation"

    if re.search(
        r"\b(calculate|compute|solve|numerical|"
        r"find the value|determine the value)\b",
        q
    ):
        return "Numerical"

    return "Theory"


def get_difficulty(question):
    q = question.lower()

    if any(word in q for word in [
        "derive", "prove", "analyse", "analyze",
        "evaluate", "justify", "design"
    ]):
        return "Hard"

    if any(word in q for word in [
        "define", "state", "list", "name"
    ]):
        return "Easy"

    return "Medium"


# =========================================================
# SUBJECT-SPECIFIC TOPIC MATCHING
# =========================================================
STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this",
    "what", "when", "where", "which", "into", "are", "was",
    "were", "how", "why", "its", "their", "then", "than",
    "can", "will", "you", "your", "using", "based", "explain",
    "write", "discuss", "describe", "define", "state", "give",
    "show", "find", "calculate", "derive", "prove", "following",
    "short", "note", "notes", "question", "questions",
    "with", "without", "about", "also", "used", "use"
}


def meaningful_tokens(text):
    words = re.findall(r"[a-z0-9]+", normalize(text))

    return {
        word for word in words
        if len(word) >= 2 and word not in STOP_WORDS
    }


def topic_similarity(question, topic):
    """
    Combine word overlap, sequence similarity and phrase containment.
    The score is used only to suggest a match, not as a probability.
    """
    q_norm = normalize(question)
    t_norm = normalize(topic)

    q_words = meaningful_tokens(question)
    t_words = meaningful_tokens(topic)

    if not t_words:
        return 0.0

    overlap = len(q_words & t_words) / len(t_words)

    sequence = SequenceMatcher(
        None,
        q_norm,
        t_norm
    ).ratio()

    phrase_score = 1.0 if (
        t_norm in q_norm and len(t_norm) >= 5
    ) else 0.0

    return min(
        1.0,
        0.60 * overlap
        + 0.25 * sequence
        + 0.15 * phrase_score
    )


def map_questions_to_topics(questions, topics, paper_name):
    records = []

    for question in questions:
        ranked = sorted(
            [
                (topic_similarity(question, topic), topic)
                for topic in topics
            ],
            key=lambda item: item[0],
            reverse=True
        )

        if not ranked:
            best_score, best_topic = 0.0, None
            second_score = 0.0
        else:
            best_score, best_topic = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else 0.0

        # Require a reasonable score and separation from the next
        # candidate. Generic/ambiguous questions are sent for review.
        confident = (
            best_topic is not None
            and best_score >= 0.28
            and (
                best_score - second_score >= 0.06
                or best_score >= 0.68
                or len(ranked) == 1
            )
        )

        if not confident:
            assigned_topic = "Unmapped / review manually"
            confidence = round(best_score * 100)
        else:
            assigned_topic = best_topic
            confidence = round(best_score * 100)

        records.append({
            "Paper": paper_name,
            "Question": question,
            "Topic": assigned_topic,
            "Match Confidence (%)": confidence,
            "Type": get_question_type(question),
            "Difficulty": get_difficulty(question),
            "Marks": get_marks(question)
        })

    return records


# =========================================================
# PRIORITY ANALYSIS
# =========================================================
def analyse_topics(topics, mapped, exam_date, topic_units):
    rows = []

    for topic in topics:
        matches = [
            row for row in mapped
            if row["Topic"] == topic
        ]

        frequency = len(matches)

        known_marks = sum(
            row["Marks"] or 0
            for row in matches
        )

        hard_count = sum(
            row["Difficulty"] == "Hard"
            for row in matches
        )

        rows.append({
            "Topic": topic,
            "Unit": topic_units.get(topic, "Not specified"),
            "Question Frequency": frequency,
            "Known Marks": known_marks,
            "Theory": sum(
                row["Type"] == "Theory" for row in matches
            ),
            "Numerical": sum(
                row["Type"] == "Numerical" for row in matches
            ),
            "Derivation": sum(
                row["Type"] == "Derivation" for row in matches
            ),
            "Hard Questions": hard_count
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    max_frequency = max(1, df["Question Frequency"].max())
    max_marks = max(1, df["Known Marks"].max())

    days_left = max(1, (exam_date - date.today()).days)
    urgency = min(1.0, 7 / days_left)

    history = df["Question Frequency"] / max_frequency
    marks_score = df["Known Marks"] / max_marks
    hard_score = (
        df["Hard Questions"]
        / df["Question Frequency"].clip(lower=1)
    )

    df["Priority Score"] = (
        55 * history
        + 25 * marks_score
        + 10 * hard_score
        + 10 * urgency
    ).round(1)

    df["Priority"] = df["Priority Score"].apply(
        lambda score:
            "HIGH" if score >= 65
            else "MEDIUM" if score >= 35
            else "LOW"
    )

    def explain(row):
        if row["Question Frequency"] == 0:
            return (
                "No confident match in the uploaded papers. "
                "This does not mean the topic is unimportant."
            )

        reason = (
            f"{int(row['Question Frequency'])} confidently matched "
            f"question(s)"
        )

        if row["Known Marks"]:
            reason += (
                f"; {int(row['Known Marks'])} labelled marks identified"
            )

        if row["Hard Questions"]:
            reason += (
                f"; {int(row['Hard Questions'])} hard question(s)"
            )

        return reason + "."

    df["Reason"] = df.apply(explain, axis=1)

    return df.sort_values(
        ["Priority Score", "Question Frequency"],
        ascending=False
    ).reset_index(drop=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("🎓 Exam Intelligence")
    st.caption("Syllabus analysis and revision planning")

    page = st.radio(
        "Navigation",
        [
            "📊 Exam Intelligence",
            "📅 My Study Plan"
        ]
    )

    st.divider()

    if st.session_state.syllabus_name:
        st.success("Syllabus loaded")
        st.caption(st.session_state.syllabus_name)
    else:
        st.info("Upload your complete syllabus to begin.")


# =========================================================
# MAIN DASHBOARD
# =========================================================
if page == "📊 Exam Intelligence":

    st.markdown("""
    <div class="hero">
        <h1>🎓 Exam Intelligence Assistant</h1>
        <p>Analyse your syllabus and previous-year papers
        to organise exam preparation.</p>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # STEP 1: UPLOAD FULL SYLLABUS
    # -----------------------------------------------------
    st.header("1. Upload Complete Syllabus")

    syllabus_file = st.file_uploader(
        "Upload your semester syllabus PDF",
        type=["pdf"],
        key="full_syllabus_upload"
    )

    if syllabus_file is not None:
        if syllabus_file.name != st.session_state.syllabus_name:
            with st.spinner("Extracting syllabus text..."):
                text = extract_pdf(syllabus_file)

                st.session_state.syllabus_name = syllabus_file.name
                st.session_state.syllabus_text = text
                st.session_state.subject_headings = (
                    detect_subject_headings(text)
                )
                st.session_state.syllabus_db = {}
                st.session_state.exam_configs = {}
                st.session_state.analysis_results = {}

            if text:
                st.success("Syllabus text extracted.")
            else:
                st.error(
                    "No readable text was extracted. "
                    "The PDF may be scanned or image-based."
                )

    if st.session_state.syllabus_text:
        with st.expander("Preview extracted syllabus text"):
            st.text_area(
                "Extracted PDF text",
                st.session_state.syllabus_text[:20000],
                height=250,
                disabled=True
            )

    # -----------------------------------------------------
    # STEP 2: SUBJECT HEADINGS
    # -----------------------------------------------------
    if st.session_state.syllabus_text:

        st.header("2. Review Subject Names")

        st.caption(
            "One subject per line. These names define the boundaries "
            "between subjects. Check spelling and make sure every "
            "subject heading matches the PDF."
        )

        default_headings = "\n".join(
            st.session_state.subject_headings
        )

        headings_text = st.text_area(
            "Subject headings",
            value=default_headings,
            height=180,
            placeholder=(
                "Applied Mathematics\n"
                "Applied Chemistry and Environmental Science\n"
                "Basic Electrical Engineering"
            ),
            key="subject_headings_editor"
        )

        headings = unique_preserve(
            headings_text.splitlines()
        )

        if st.button(
            "Detect and Separate Subjects",
            type="primary",
            use_container_width=True
        ):
            if not headings:
                st.error("Enter at least one subject heading.")
            else:
                database = build_syllabus_database(
                    st.session_state.syllabus_text,
                    headings
                )

                st.session_state.subject_headings = headings
                st.session_state.syllabus_db = database
                st.session_state.exam_configs = {}
                st.session_state.analysis_results = {}

                if database:
                    st.success(
                        f"Detected {len(database)} subject section(s)."
                    )
                else:
                    st.error(
                        "No subject sections with unit headings were "
                        "detected. Check that each subject heading "
                        "matches a separate heading line in the PDF, "
                        "and that its units are labelled Unit 1, "
                        "Unit I, Module 1, etc."
                    )

    # -----------------------------------------------------
    # STEP 3: REVIEW SUBJECTS AND TOPICS
    # -----------------------------------------------------
    database = st.session_state.syllabus_db

    if database:

        st.header("3. Review Extracted Subjects and Units")

        for subject_name, subject_data in database.items():
            with st.expander(
                f"{subject_name} — "
                f"{len(subject_data['units'])} unit(s)"
            ):
                for unit_number, topics in sorted(
                    subject_data["units"].items()
                ):
                    st.markdown(f"**Unit {unit_number}**")

                    if topics:
                        st.write(", ".join(topics))
                    else:
                        st.warning(
                            "No topics detected in this unit. "
                            "Check the original syllabus PDF."
                        )

        st.info(
            "If a topic appears under the wrong subject, correct the "
            "subject headings or PDF extraction before analysing PYQs."
        )

        # -------------------------------------------------
        # STEP 4: CONFIGURE MULTIPLE EXAM SUBJECTS
        # -------------------------------------------------
        st.header("4. Select Multiple Exam Subjects")

        available_subjects = list(database.keys())

        selected_subjects = st.multiselect(
            "Choose all subjects you want to analyse",
            options=available_subjects,
            default=[
                name for name in available_subjects
                if name in st.session_state.exam_configs
            ],
            key="selected_exam_subjects"
        )

        # Remove configs for subjects no longer selected.
        for old_name in list(st.session_state.exam_configs):
            if old_name not in selected_subjects:
                del st.session_state.exam_configs[old_name]

        for subject_name in selected_subjects:

            subject_data = database[subject_name]
            units = subject_data["units"]

            if subject_name not in st.session_state.exam_configs:
                st.session_state.exam_configs[subject_name] = {
                    "exam_date": date.today() + timedelta(days=7),
                    "mode": "Whole unit(s)",
                    "units": [],
                    "topics": [],
                    "manual_topics": ""
                }

            config = st.session_state.exam_configs[subject_name]

            with st.container(border=True):
                st.subheader(f"📘 {subject_name}")

                exam_date = st.date_input(
                    "Exam date",
                    value=config["exam_date"],
                    min_value=date.today(),
                    key=f"exam_date_{subject_name}"
                )

                mode = st.radio(
                    "Exam syllabus scope",
                    [
                        "Whole unit(s)",
                        "Individual topics",
                        "Mix of units and topics"
                    ],
                    index=[
                        "Whole unit(s)",
                        "Individual topics",
                        "Mix of units and topics"
                    ].index(config["mode"]),
                    horizontal=True,
                    key=f"mode_{subject_name}"
                )

                chosen_units = []

                if mode in [
                    "Whole unit(s)",
                    "Mix of units and topics"
                ]:
                    chosen_units = st.multiselect(
                        "Select complete units",
                        options=list(units.keys()),
                        default=[
                            unit for unit in config["units"]
                            if unit in units
                        ],
                        format_func=lambda unit: f"Unit {unit}",
                        key=f"units_{subject_name}"
                    )

                topic_options = []

                for unit_number, topic_list in units.items():
                    for topic in topic_list:
                        topic_options.append(
                            f"Unit {unit_number} — {topic}"
                        )

                chosen_topic_labels = []

                if mode in [
                    "Individual topics",
                    "Mix of units and topics"
                ]:
                    chosen_topic_labels = st.multiselect(
                        "Select individual topics",
                        options=topic_options,
                        default=[
                            label for label in config["topics"]
                            if label in topic_options
                        ],
                        key=f"topics_{subject_name}"
                    )

                manual_topics = st.text_area(
                    "Additional topics (one per line)",
                    value=config["manual_topics"],
                    key=f"manual_{subject_name}",
                    placeholder="Optional: add missing topics"
                )

                config["exam_date"] = exam_date
                config["mode"] = mode
                config["units"] = chosen_units
                config["topics"] = chosen_topic_labels
                config["manual_topics"] = manual_topics

        # -------------------------------------------------
        # STEP 5: UPLOAD SUBJECT-SPECIFIC PYQS
        # -------------------------------------------------
        if selected_subjects:

            st.header("5. Upload Previous-Year Papers")

            st.caption(
                "Upload papers separately for each subject. "
                "A Mathematics paper will only be matched against "
                "Mathematics topics, not Chemistry topics."
            )

            uploaded_papers = {}

            for subject_name in selected_subjects:
                uploaded_papers[subject_name] = st.file_uploader(
                    f"PYQ PDFs — {subject_name}",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key=f"papers_{subject_name}"
                )

            # ---------------------------------------------
            # STEP 6: RUN ANALYSIS
            # ---------------------------------------------
            if st.button(
                "🚀 Analyse All Selected Subjects",
                type="primary",
                use_container_width=True
            ):

                prepared = {}
                validation_errors = []

                for subject_name in selected_subjects:
                    config = st.session_state.exam_configs[subject_name]
                    units = database[subject_name]["units"]

                    topics = []
                    topic_units = {}

                    # Whole units.
                    for unit_number in config["units"]:
                        for topic in units[unit_number]:
                            if topic not in topics:
                                topics.append(topic)
                                topic_units[topic] = f"Unit {unit_number}"

                    # Individual topics.
                    for label in config["topics"]:
                        if " — " not in label:
                            continue

                        unit_label, topic = label.split(" — ", 1)

                        if topic not in topics:
                            topics.append(topic)
                            topic_units[topic] = unit_label

                    # Manually added topics.
                    for topic in config["manual_topics"].splitlines():
                        topic = clean_line(topic)

                        if topic and topic not in topics:
                            topics.append(topic)
                            topic_units[topic] = "Manually added"

                    if not topics:
                        validation_errors.append(
                            f"{subject_name}: select at least one "
                            "unit or topic."
                        )
                        continue

                    prepared[subject_name] = {
                        "exam_date": config["exam_date"],
                        "topics": topics,
                        "topic_units": topic_units,
                        "papers": uploaded_papers[subject_name]
                    }

                if validation_errors:
                    for message in validation_errors:
                        st.error(message)

                elif not prepared:
                    st.error("Select at least one subject to analyse.")

                else:
                    all_results = {}

                    with st.spinner(
                        "Analysing each subject independently..."
                    ):
                        for subject_name, config in prepared.items():
                            topics = config["topics"]
                            mapped = []

                            for paper in config["papers"] or []:
                                questions = extract_questions(paper)

                                mapped.extend(
                                    map_questions_to_topics(
                                        questions,
                                        topics,
                                        paper.name
                                    )
                                )

                            priority = analyse_topics(
                                topics,
                                mapped,
                                config["exam_date"],
                                config["topic_units"]
                            )

                            all_results[subject_name] = {
                                "exam_date": config["exam_date"],
                                "priority": priority,
                                "mapping": mapped,
                                "topics": topics,
                                "topic_units": config["topic_units"]
                            }

                    st.session_state.analysis_results = all_results

                    st.success(
                        f"Analysis completed for "
                        f"{len(all_results)} subject(s)."
                    )

        # -------------------------------------------------
        # STEP 7: RESULTS
        # -------------------------------------------------
        if st.session_state.analysis_results:

            st.divider()
            st.header("6. Analysis Results")

            result_names = list(
                st.session_state.analysis_results.keys()
            )

            result_subject = st.selectbox(
                "Choose subject to view results",
                result_names,
                key="view_result_subject"
            )

            result = st.session_state.analysis_results[result_subject]

            df = result["priority"]
            mapped = result["mapping"]

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Topics analysed", len(df))
            c2.metric("Questions extracted", len(mapped))
            c3.metric(
                "High-priority topics",
                int((df["Priority"] == "HIGH").sum())
            )
            c4.metric(
                "Days until exam",
                max(
                    0,
                    (result["exam_date"] - date.today()).days
                )
            )

            st.subheader("Topic Priority Ranking")

            st.caption(
                "Priority is a heuristic based on confident matches, "
                "identified marks, difficulty and exam urgency. "
                "It is not a guarantee of exam appearance."
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            st.plotly_chart(
                px.bar(
                    df,
                    x="Topic",
                    y="Priority Score",
                    color="Priority",
                    title=f"{result_subject}: Topic Priority"
                ),
                use_container_width=True
            )

            st.plotly_chart(
                px.bar(
                    df,
                    x="Topic",
                    y="Question Frequency",
                    title=f"{result_subject}: PYQ Topic Frequency"
                ),
                use_container_width=True
            )

            st.subheader("Question-to-Topic Mapping")

            if mapped:
                mapped_df = pd.DataFrame(mapped)

                st.dataframe(
                    mapped_df,
                    use_container_width=True,
                    hide_index=True
                )

                with st.expander("Review uncertain question matches"):
                    uncertain = mapped_df[
                        (
                            mapped_df["Topic"]
                            == "Unmapped / review manually"
                        )
                        | (mapped_df["Match Confidence (%)"] < 35)
                    ]

                    if uncertain.empty:
                        st.success(
                            "No uncertain matches were flagged "
                            "by the current threshold."
                        )
                    else:
                        st.dataframe(
                            uncertain,
                            use_container_width=True,
                            hide_index=True
                        )

                st.download_button(
                    "Download Question Mapping CSV",
                    mapped_df.to_csv(index=False).encode("utf-8"),
                    file_name=(
                        normalize(result_subject).replace(" ", "_")
                        + "_question_mapping.csv"
                    ),
                    mime="text/csv"
                )

            else:
                st.info(
                    "No questions were extracted from this subject's "
                    "uploaded papers. Check the PDF text extraction."
                )

            st.download_button(
                "Download Topic Priority CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name=(
                    normalize(result_subject).replace(" ", "_")
                    + "_priority.csv"
                ),
                mime="text/csv"
            )

    else:
        if st.session_state.syllabus_text:
            st.info(
                "Review the subject headings and click "
                "'Detect and Separate Subjects' to continue."
            )


# =========================================================
# STUDY PLAN
# =========================================================
else:

    st.markdown("""
    <div class="hero">
        <h1>📅 My Study Plan</h1>
        <p>Track revision across all your selected exam subjects.</p>
    </div>
    """, unsafe_allow_html=True)

    results = st.session_state.analysis_results

    if not results:
        st.info(
            "Upload your syllabus, select multiple subjects, "
            "and run the analysis first."
        )

    else:
        tasks = []

        for subject_name, result in results.items():
            for _, row in result["priority"].iterrows():
                tasks.append({
                    "Subject": subject_name,
                    "Topic": row["Topic"],
                    "Unit": row["Unit"],
                    "Priority": row["Priority"],
                    "Priority Score": row["Priority Score"],
                    "Exam Date": result["exam_date"]
                })

        plan = pd.DataFrame(tasks)

        if not plan.empty:
            plan = plan.sort_values(
                ["Exam Date", "Priority Score"],
                ascending=[True, False]
            ).reset_index(drop=True)

            st.subheader("Recommended Revision Order")

            st.dataframe(
                plan,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Revision Checklist")

            completed_count = 0

            for _, row in plan.iterrows():
                task_id = f"{row['Subject']}|{row['Topic']}"

                checked = st.checkbox(
                    f"{row['Exam Date'].strftime('%d %b')} — "
                    f"{row['Subject']}: {row['Topic']} "
                    f"({row['Priority']})",
                    value=(
                        task_id in st.session_state.completed_tasks
                    ),
                    key=(
                        "check_"
                        + str(abs(hash(task_id)))
                    )
                )

                if checked:
                    st.session_state.completed_tasks.add(task_id)
                else:
                    st.session_state.completed_tasks.discard(task_id)

                completed_count += int(checked)

            st.progress(completed_count / len(plan))

            st.caption(
                f"{completed_count} of {len(plan)} revision tasks completed."
            )

            st.download_button(
                "Download Complete Study Plan CSV",
                plan.to_csv(index=False).encode("utf-8"),
                file_name="complete_study_plan.csv",
                mime="text/csv"
            )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "Exam Intelligence Assistant | Always verify extracted subject "
    "boundaries and uncertain matches. Scanned PDFs may need OCR. "
    "Question extraction and topic matching depend on PDF quality."
)
