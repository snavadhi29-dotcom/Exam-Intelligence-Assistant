import streamlit as st
import pandas as pd
import re
from io import BytesIO
from datetime import date, timedelta
from difflib import SequenceMatcher

from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_bytes


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Exam Intelligence Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "subjects" not in st.session_state:
    st.session_state.subjects = []

if "next_subject_id" not in st.session_state:
    st.session_state.next_subject_id = 1


# ============================================================
# BASIC TEXT FUNCTIONS
# ============================================================

STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this",
    "find", "show", "prove", "calculate", "using", "given",
    "derive", "determine", "solve", "evaluate", "write",
    "explain", "define", "of", "to", "in", "on", "is",
    "are", "a", "an", "be", "if", "or", "as", "by",
    "at", "it", "its", "let", "where", "which", "then",
    "also", "following", "any", "all"
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

    # First try normal selectable text
    try:
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

    # If that fails, use OCR
    try:
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
# REMOVE EXAM PAPER INSTRUCTIONS
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
        r"^\s*subject\s*[:\-]?",
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

        ignore = False

        for pattern in ignored_patterns:

            if re.search(
                pattern,
                clean,
                re.IGNORECASE
            ):
                ignore = True
                break

        if not ignore:
            output.append(line)

    return "\n".join(output)


# ============================================================
# QUESTION EXTRACTION
# ============================================================

def extract_questions(text):

    text = text.replace("\r", "\n")

    text = remove_instruction_lines(text)

    # Try to identify numbered questions
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
            end = matches[i + 1].start()
        else:
            end = len(text)

        question = text[start:end].strip()

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

        questions.append(question)

    # Fallback if numbering wasn't detected
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

def word_overlap(question, topic):

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
        len(common) /
        len(t_words)
    ) * 100


def fuzzy_score(question, topic):

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
        total /
        len(t_words)
    ) * 100


def contains_topic(question, topic):

    q = normalize_text(question)
    t = normalize_text(topic)

    return t in q


def topic_similarity(question, topic):

    if contains_topic(
        question,
        topic
    ):
        return 100

    overlap = word_overlap(
        question,
        topic
    )

    fuzzy = fuzzy_score(
        question,
        topic
    )

    # Weighted score
    score = (
        overlap * 0.60
        +
        fuzzy * 0.40
    )

    return round(
        score,
        2
    )


def find_best_topic(
    question,
    topics
):

    if not topics:
        return "Unclassified", 0

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

    best_topic, best_score = scores[0]

    # Avoid forcing completely unrelated
    # questions into a syllabus topic
    if best_score < 12:
        return "Unclassified", best_score

    return best_topic, best_score


# ============================================================
# ANALYZE ONE SUBJECT
# ============================================================

def analyze_subject(
    subject_name,
    topics,
    uploaded_files
):

    rows = []
    messages = []

    for uploaded_file in uploaded_files:

        pdf_bytes = uploaded_file.getvalue()

        text, method = extract_text_from_pdf(
            pdf_bytes
        )

        if not text.strip():

            messages.append(
                f"⚠️ {uploaded_file.name}: "
                f"Could not extract text."
            )

            continue

        questions = extract_questions(
            text
        )

        messages.append(
            f"📄 {uploaded_file.name}: "
            f"{len(questions)} questions detected "
            f"({method})."
        )

        for number, question in enumerate(
            questions,
            start=1
        ):

            topic, score = find_best_topic(
                question,
                topics
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
# TOPIC PRIORITY
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

        paper_count_for_topic = (
            matched["Paper"]
            .nunique()
        )

        coverage = (
            paper_count_for_topic /
            max(paper_count, 1)
        ) * 100

        frequency = (
            question_count /
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
                paper_count_for_topic,

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

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values(
        "Priority Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    # Priority based on actual question presence
    for index in df.index:

        questions = df.loc[
            index,
            "Questions"
        ]

        score = df.loc[
            index,
            "Priority Score"
        ]

        if questions == 0:

            priority = "Low"

        elif score >= 35:

            priority = "High"

        elif score >= 15:

            priority = "Medium"

        else:

            priority = "Low"

        df.loc[
            index,
            "Priority"
        ] = priority

    return df


# ============================================================
# GENERATE TIMETABLE
# ============================================================

def generate_timetable(
    analyzed_subjects
):

    if not analyzed_subjects:
        return pd.DataFrame()

    today = date.today()

    tasks = []

    for subject in analyzed_subjects:

        subject_name = subject["name"]
        exam_date = subject["exam_date"]
        priority_df = subject["priority"]

        for _, row in priority_df.iterrows():

            if row["Questions"] == 0:
                continue

            priority = row["Priority"]

            if priority == "High":
                minutes = 90

            elif priority == "Medium":
                minutes = 60

            else:
                minutes = 40

            tasks.append({

                "Subject":
                    subject_name,

                "Topic":
                    row["Topic"],

                "Priority":
                    priority,

                "Score":
                    row["Priority Score"],

                "Minutes":
                    minutes,

                "Exam Date":
                    exam_date
            })

    if not tasks:
        return pd.DataFrame()

    latest_exam = max(
        s["exam_date"]
        for s in analyzed_subjects
    )

    if today > latest_exam:
        return pd.DataFrame()

    days = []

    current = today

    while current <= latest_exam:

        days.append(current)

        current += timedelta(
            days=1
        )

    timetable = []

    remaining = tasks.copy()

    for current_day in days:

        if not remaining:
            break

        candidates = []

        for task in remaining:

            if current_day > task["Exam Date"]:
                continue

            days_left = (
                task["Exam Date"]
                - current_day
            ).days

            priority_value = {
                "High": 3,
                "Medium": 2,
                "Low": 1
            }[
                task["Priority"]
            ]

            urgency = (
                priority_value * 100
                +
                task["Score"]
                +
                max(
                    0,
                    40 - days_left * 3
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

        # Maximum two focused tasks per day
        for _, task in candidates[:2]:

            timetable.append({

                "Date":
                    current_day.strftime(
                        "%d %b %Y"
                    ),

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

                "Exam":
                    task["Exam Date"].strftime(
                        "%d %b %Y"
                    )
            })

            remaining.remove(task)

    return pd.DataFrame(
        timetable
    )


# ============================================================
# ADD SUBJECT
# ============================================================

st.title(
    "📚 Exam Intelligence Assistant"
)

st.write(
    "Analyze your previous-year papers, "
    "identify important topics, and create "
    "a personalized multi-subject study plan."
)

st.divider()


st.header(
    "📚 My Subjects"
)


if st.button(
    "➕ Add New Subject",
    type="primary"
):

    new_id = st.session_state.next_subject_id

    st.session_state.next_subject_id += 1

    st.session_state.subjects.append({

        "id":
            new_id,

        "name":
            "",

        "exam_date":
            date.today() + timedelta(days=7),

        "topics":
            "",

    })

    st.rerun()


# ============================================================
# SUBJECT CARDS
# ============================================================

if not st.session_state.subjects:

    st.info(
        "Start by clicking **Add New Subject**."
    )


for index, subject in enumerate(
    st.session_state.subjects
):

    sid = subject["id"]

    with st.container(border=True):

        st.subheader(
            f"📘 Subject {index + 1}"
        )

        col1, col2 = st.columns(
            [3, 2]
        )

        with col1:

            name = st.text_input(
                "Subject Name",
                value=subject["name"],
                key=f"name_{sid}",
                placeholder="Example: Applied Mathematics"
            )

        with col2:

            exam = st.date_input(
                "Exam Date",
                value=subject["exam_date"],
                key=f"date_{sid}"
            )

        topics = st.text_area(
            "Syllabus Topics — one topic per line",
            value=subject["topics"],
            key=f"syllabus_{sid}",
            height=150,
            placeholder=(
                "Review of successive differentiation\n"
                "Leibnitz theorem and problems\n"
                "Taylor's and Maclaurin's theorem\n"
                "Beta and gamma functions\n"
                "Tracing curves"
            )
        )

        uploaded_files = st.file_uploader(
            "Upload Previous-Year Question Papers",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"papers_{sid}",
            help="You can select multiple PDFs for this subject."
        )

        col_a, col_b = st.columns(
            [1, 1]
        )

        with col_a:

            if name.strip():

                st.caption(
                    f"📘 {name} • "
                    f"Exam: {exam.strftime('%d %b %Y')}"
                )

        with col_b:

            if st.button(
                "🗑️ Remove Subject",
                key=f"remove_{sid}"
            ):

                st.session_state.subjects = [
                    s
                    for s in st.session_state.subjects
                    if s["id"] != sid
                ]

                st.rerun()


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

if st.session_state.subjects:

    if st.button(
        "🚀 Analyze All Subjects",
        type="primary",
        use_container_width=True
    ):

        analyzed_subjects = []

        errors = False

        for index, subject in enumerate(
            st.session_state.subjects
        ):

            sid = subject["id"]

            name = st.session_state.get(
                f"name_{sid}",
                ""
            ).strip()

            exam = st.session_state.get(
                f"date_{sid}",
                date.today()
            )

            syllabus = st.session_state.get(
                f"syllabus_{sid}",
                ""
            )

            papers = st.session_state.get(
                f"papers_{sid}",
                []
            )

            if not name:

                st.error(
                    f"Subject {index + 1}: "
                    f"Please enter the subject name."
                )

                errors = True
                continue

            topics = [
                t.strip()
                for t in syllabus.split("\n")
                if t.strip()
            ]

            if not topics:

                st.error(
                    f"{name}: Please enter "
                    f"at least one syllabus topic."
                )

                errors = True
                continue

            if not papers:

                st.error(
                    f"{name}: Please upload "
                    f"at least one question paper."
                )

                errors = True
                continue

            with st.spinner(
                f"Analyzing {name}..."
            ):

                results, messages = (
                    analyze_subject(
                        name,
                        topics,
                        papers
                    )
                )

            for message in messages:
                st.write(message)

            if results.empty:

                st.warning(
                    f"{name}: No readable "
                    f"questions were found."
                )

                continue

            priority = calculate_priority(
                results,
                topics,
                len(papers)
            )

            analyzed_subjects.append({

                "name":
                    name,

                "exam_date":
                    exam,

                "results":
                    results,

                "priority":
                    priority
            })

        if analyzed_subjects:

            st.session_state.analyzed = (
                analyzed_subjects
            )

        if errors:
            st.warning(
                "Please fix the highlighted "
                "subject information."
            )


# ============================================================
# RESULTS
# ============================================================

if "analyzed" in st.session_state:

    analyzed = st.session_state.analyzed

    st.divider()

    st.header(
        "📊 Exam Intelligence Dashboard"
    )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    total_questions = sum(
        len(s["results"])
        for s in analyzed
    )

    total_topics = sum(
        len(s["priority"])
        for s in analyzed
    )

    high_topics = sum(
        len(
            s["priority"][
                s["priority"]["Priority"]
                == "High"
            ]
        )
        for s in analyzed
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Subjects",
        len(analyzed)
    )

    c2.metric(
        "Questions Analyzed",
        total_questions
    )

    c3.metric(
        "Topics",
        total_topics
    )

    c4.metric(
        "High Priority",
        high_topics
    )


    # --------------------------------------------------------
    # SUBJECT OVERVIEW
    # --------------------------------------------------------

    st.subheader(
        "📚 Subject Overview"
    )

    overview = []

    for s in analyzed:

        priority = s["priority"]

        high = len(
            priority[
                priority["Priority"]
                == "High"
            ]
        )

        medium = len(
            priority[
                priority["Priority"]
                == "Medium"
            ]
        )

        days_left = (
            s["exam_date"]
            - date.today()
        ).days

        overview.append({

            "Subject":
                s["name"],

            "Exam Date":
                s["exam_date"].strftime(
                    "%d %b %Y"
                ),

            "Days Left":
                max(days_left, 0),

            "High":
                high,

            "Medium":
                medium
        })

    overview_df = pd.DataFrame(
        overview
    )

    st.dataframe(
        overview_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # SUBJECT DETAILS
    # --------------------------------------------------------

    st.header(
        "🔎 Detailed Subject Analysis"
    )

    for s in analyzed:

        with st.expander(
            f"📘 {s['name']} — "
            f"Exam: {s['exam_date'].strftime('%d %b %Y')}"
        ):

            priority = s["priority"]

            results = s["results"]

            st.subheader(
                "🔥 Topic Priority"
            )

            st.dataframe(
                priority,
                use_container_width=True,
                hide_index=True
            )

            st.subheader(
                "📈 Topic Frequency"
            )

            chart_data = priority[
                [
                    "Topic",
                    "Questions"
                ]
            ].copy()

            st.bar_chart(
                chart_data.set_index(
                    "Topic"
                )
            )

            st.subheader(
                "📝 Question → Topic Mapping"
            )

            st.dataframe(
                results[
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


    # --------------------------------------------------------
    # SMART TIMETABLE
    # --------------------------------------------------------

    st.divider()

    st.header(
        "📅 Smart Study Timetable"
    )

    timetable = generate_timetable(
        analyzed
    )

    if timetable.empty:

        st.warning(
            "There is not enough information "
            "to generate the timetable."
        )

    else:

        st.success(
            "Your timetable is based on "
            "exam dates + previous-paper "
            "frequency + topic priority."
        )

        st.dataframe(
            timetable,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "🗓️ Daily Plan"
        )

        timetable_dates = (
            timetable["Date"]
            .unique()
        )

        for timetable_date in timetable_dates:

            st.markdown(
                f"### 📅 {timetable_date}"
            )

            day_data = timetable[
                timetable["Date"]
                == timetable_date
            ]

            for _, row in day_data.iterrows():

                if row["Priority"] == "High":
                    icon = "🔴"
                elif row["Priority"] == "Medium":
                    icon = "🟡"
                else:
                    icon = "🟢"

                st.write(
                    f"{icon} **{row['Subject']}** — "
                    f"{row['Topic']} — "
                    f"{row['Study Time']}"
                )


    # --------------------------------------------------------
    # WHAT TO STUDY FIRST
    # --------------------------------------------------------

    st.divider()

    st.header(
        "🎯 What Should I Study First?"
    )

    candidates = []

    for s in analyzed:

        for _, row in s["priority"].iterrows():

            if row["Questions"] == 0:
                continue

            days_left = (
                s["exam_date"]
                - date.today()
            ).days

            priority_value = {
                "High": 3,
                "Medium": 2,
                "Low": 1
            }[
                row["Priority"]
            ]

            score = (
                priority_value * 100
                +
                row["Priority Score"]
                +
                max(
                    0,
                    40 - days_left * 3
                )
            )

            candidates.append({

                "Score":
                    score,

                "Subject":
                    s["name"],

                "Topic":
                    row["Topic"],

                "Priority":
                    row["Priority"],

                "Exam":
                    s["exam_date"],

                "Questions":
                    row["Questions"]
            })

    candidates.sort(
        key=lambda x: x["Score"],
        reverse=True
    )

    if candidates:

        first = candidates[0]

        st.success(
            f"### 📖 {first['Topic']}\n\n"
            f"**Subject:** {first['Subject']}\n\n"
            f"**Priority:** {first['Priority']}\n\n"
            f"**Previous-paper questions:** "
            f"{first['Questions']}\n\n"
            f"**Exam:** "
            f"{first['Exam'].strftime('%d %B %Y')}"
        )
