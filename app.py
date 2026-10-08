import streamlit as st
import pandas as pd
import plotly.express as px
import re
from io import BytesIO
from difflib import SequenceMatcher
from datetime import date, timedelta

from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_bytes

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Exam Intelligence Assistant",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "subjects" not in st.session_state:
    st.session_state.subjects = []


# =========================================================
# TEXT PROCESSING
# =========================================================

STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this",
    "find", "show", "prove", "calculate", "using", "given",
    "derive", "determine", "solve", "evaluate", "write",
    "explain", "define", "of", "to", "in", "on", "is",
    "are", "a", "an", "be", "if", "or", "as", "by",
    "at", "it", "its", "let", "where", "which", "then",
    "also", "following"
}


def normalize_text(text):

    text = text.lower()

    text = text.replace("’", "'")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_words(text):

    text = normalize_text(text)

    words = re.findall(
        r"[a-zA-Z]+",
        text
    )

    cleaned = []

    for word in words:

        if word in STOP_WORDS:
            continue

        if len(word) <= 2:
            continue

        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"

        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]

        cleaned.append(word)

    return cleaned


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_normal_pdf_text(pdf_bytes):

    try:

        reader = PdfReader(
            BytesIO(pdf_bytes)
        )

        page_texts = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                page_texts.append(text)

        return "\n".join(page_texts)

    except Exception:

        return ""


def extract_ocr_text(pdf_bytes):

    try:

        images = convert_from_bytes(
            pdf_bytes,
            dpi=180,
            fmt="jpeg"
        )

        all_text = []

        progress = st.progress(0)

        total = len(images)

        for index, image in enumerate(images):

            text = pytesseract.image_to_string(
                image,
                config="--psm 6"
            )

            if text:
                all_text.append(text)

            progress.progress(
                (index + 1) / total
            )

        progress.empty()

        return "\n".join(all_text)

    except Exception as e:

        st.error(
            "OCR could not process this PDF."
        )

        st.exception(e)

        return ""


def extract_pdf_text(pdf_bytes):

    normal_text = extract_normal_pdf_text(
        pdf_bytes
    )

    if len(normal_text.strip()) >= 80:

        return normal_text, "Text extraction"

    return extract_ocr_text(
        pdf_bytes
    ), "OCR"


# =========================================================
# REMOVE PAPER INSTRUCTIONS
# =========================================================

def remove_instruction_lines(text):

    lines = text.split("\n")

    useful_lines = []

    ignore_patterns = [

        r"^\s*time\s*[:\-]?",
        r"^\s*time allowed",
        r"^\s*maximum marks",
        r"^\s*max marks",
        r"^\s*max\.\s*marks",
        r"^\s*marks\s*[:\-]?",
        r"^\s*note\s*[:\-]?",
        r"^\s*instructions?\s*[:\-]?",
        r"^\s*attempt\s+any",
        r"^\s*attempt\s+all",
        r"^\s*attempt\s+four",
        r"^\s*attempt\s+five",
        r"^\s*duration\s*[:\-]?",
        r"^\s*date\s*[:\-]?",
        r"^\s*roll\s*no",
        r"^\s*enrollment",
        r"^\s*semester\s*[:\-]?",
        r"^\s*branch\s*[:\-]?",
        r"^\s*subject\s*[:\-]?",
        r"^\s*paper\s*[:\-]?",
        r"^\s*university",
        r"^\s*department",
        r"^\s*b\.?tech",
        r"^\s*end semester",
        r"^\s*mid semester"
    ]

    for line in lines:

        clean = line.strip()

        if len(clean) < 3:
            continue

        should_ignore = False

        for pattern in ignore_patterns:

            if re.search(
                pattern,
                clean,
                re.IGNORECASE
            ):

                should_ignore = True
                break

        if not should_ignore:

            useful_lines.append(line)

    return "\n".join(useful_lines)


# =========================================================
# QUESTION EXTRACTION
# =========================================================

def extract_questions(text):

    text = text.replace(
        "\r",
        "\n"
    )

    text = remove_instruction_lines(
        text
    )

    text = re.sub(
        r"\n{2,}",
        "\n",
        text
    )

    question_pattern = re.compile(
        r"(?:^|\n)"
        r"\s*"
        r"(?:"
        r"Q(?:uestion)?\s*\.?\s*"
        r"|"
        r"Q\s*\.?\s*"
        r")?"
        r"(\d{1,2})"
        r"\s*[\.\):\-]"
        r"\s*",
        re.IGNORECASE
    )

    matches = list(
        question_pattern.finditer(text)
    )

    questions = []

    if matches:

        for i, match in enumerate(matches):

            start = match.end()

            if i + 1 < len(matches):

                end = matches[i + 1].start()

            else:

                end = len(text)

            question_text = text[
                start:end
            ].strip()

            question_text = re.sub(
                r"\s+",
                " ",
                question_text
            )

            if len(question_text) < 25:
                continue

            lower = question_text.lower()

            instruction_words = [
                "attempt any",
                "attempt all",
                "maximum marks",
                "time allowed",
                "instructions"
            ]

            if any(
                word in lower
                for word in instruction_words
            ):
                continue

            questions.append(
                question_text
            )

    if len(questions) == 0:

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

            if len(paragraph) < 35:
                continue

            questions.append(
                paragraph
            )

    return questions


# =========================================================
# TOPIC MATCHING
# =========================================================

def fuzzy_word_score(question, topic):

    question_words = set(
        get_words(question)
    )

    topic_words = set(
        get_words(topic)
    )

    if not topic_words:
        return 0

    total = 0

    for topic_word in topic_words:

        best = 0

        for question_word in question_words:

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
        len(topic_words)
    ) * 100


def word_overlap_score(question, topic):

    question_words = set(
        get_words(question)
    )

    topic_words = set(
        get_words(topic)
    )

    if not topic_words:
        return 0

    common = (
        question_words &
        topic_words
    )

    return (
        len(common) /
        len(topic_words)
    ) * 100


def tfidf_similarity(question, topic):

    try:

        documents = [
            question,
            topic
        ]

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5)
        )

        matrix = vectorizer.fit_transform(
            documents
        )

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return score * 100

    except Exception:

        return 0


def topic_similarity(question, topic):

    question_clean = normalize_text(
        question
    )

    topic_clean = normalize_text(
        topic
    )

    if topic_clean in question_clean:

        return 100

    word_score = word_overlap_score(
        question,
        topic
    )

    fuzzy_score = fuzzy_word_score(
        question,
        topic
    )

    tfidf_score = tfidf_similarity(
        question,
        topic
    )

    final_score = (
        word_score * 0.40
        +
        fuzzy_score * 0.25
        +
        tfidf_score * 0.35
    )

    return round(
        final_score,
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
            (topic, score)
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    if not scores:

        return "Unclassified", 0

    best_topic, best_score = scores[0]

    if best_score < 15:

        return "Unclassified", best_score

    return best_topic, best_score


# =========================================================
# ANALYZE A SUBJECT
# =========================================================

def analyze_subject(
    subject_name,
    topics,
    uploaded_files
):

    all_results = []

    processing_messages = []

    for file in uploaded_files:

        pdf_bytes = file.getvalue()

        extracted_text, method = (
            extract_pdf_text(
                pdf_bytes
            )
        )

        if not extracted_text.strip():

            processing_messages.append(
                f"⚠️ {file.name}: no text found."
            )

            continue

        questions = extract_questions(
            extracted_text
        )

        processing_messages.append(
            f"📄 {file.name}: "
            f"{len(questions)} questions detected "
            f"using {method}."
        )

        for number, question in enumerate(
            questions,
            start=1
        ):

            topic, score = find_best_topic(
                question,
                topics
            )

            all_results.append({

                "Subject": subject_name,

                "Paper": file.name,

                "Question No.": number,

                "Question": question,

                "Matched Topic": topic,

                "Match Score": score
            })

    return (
        pd.DataFrame(all_results),
        processing_messages
    )


# =========================================================
# TOPIC PRIORITY
# =========================================================

def calculate_topic_priority(
    results_df,
    topics,
    number_of_papers
):

    rows = []

    for topic in topics:

        matched = results_df[
            results_df["Matched Topic"]
            == topic
        ]

        question_count = len(
            matched
        )

        paper_count = (
            matched["Paper"]
            .nunique()
        )

        paper_coverage = (
            paper_count /
            max(number_of_papers, 1)
        ) * 100

        question_frequency = (
            question_count /
            max(len(results_df), 1)
        ) * 100

        priority_score = (
            paper_coverage * 0.60
            +
            question_frequency * 0.40
        )

        rows.append({

            "Topic": topic,

            "Questions": question_count,

            "Papers": paper_count,

            "Paper Coverage %":
                round(
                    paper_coverage,
                    1
                ),

            "Question Frequency %":
                round(
                    question_frequency,
                    1
                ),

            "Priority Score":
                round(
                    priority_score,
                    1
                )
        })

    df = pd.DataFrame(
        rows
    )

    df = df.sort_values(
        by="Priority Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    total_topics = len(df)

    for index in range(
        total_topics
    ):

        questions = df.loc[
            index,
            "Questions"
        ]

        if questions == 0:

            priority = "Low"

        elif index < max(
            1,
            round(
                total_topics * 0.30
            )
        ):

            priority = "High"

        elif index < max(
            2,
            round(
                total_topics * 0.70
            )
        ):

            priority = "Medium"

        else:

            priority = "Low"

        df.loc[
            index,
            "Priority"
        ] = priority

    return df


# =========================================================
# CREATE STUDY TASKS
# =========================================================

def create_tasks(
    all_subject_data
):

    tasks = []

    for subject in all_subject_data:

        subject_name = subject["name"]

        exam_date = subject["exam_date"]

        priority_df = subject[
            "priority_df"
        ]

        for _, row in priority_df.iterrows():

            priority = row[
                "Priority"
            ]

            questions = row[
                "Questions"
            ]

            if questions == 0:
                continue

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

                "Minutes":
                    minutes,

                "Exam Date":
                    exam_date,

                "Priority Score":
                    row[
                        "Priority Score"
                    ]
            })

    return tasks


# =========================================================
# GENERATE SMART TIMETABLE
# =========================================================

def generate_timetable(
    all_subject_data
):

    if not all_subject_data:

        return pd.DataFrame()

    today = date.today()

    tasks = create_tasks(
        all_subject_data
    )

    if not tasks:

        return pd.DataFrame()

    # Sort subjects by exam date
    tasks.sort(
        key=lambda x: x["Exam Date"]
    )

    earliest_exam = min(
        subject["exam_date"]
        for subject in all_subject_data
    )

    # We plan until the last exam date
    latest_exam = max(
        subject["exam_date"]
        for subject in all_subject_data
    )

    start_date = today

    if start_date > latest_exam:

        return pd.DataFrame()

    # -----------------------------------------------------
    # Create available study days
    # -----------------------------------------------------

    available_days = []

    current = start_date

    while current <= latest_exam:

        available_days.append(
            current
        )

        current += timedelta(
            days=1
        )

    # -----------------------------------------------------
    # Schedule tasks
    # -----------------------------------------------------

    timetable = []

    remaining_tasks = tasks.copy()

    for study_day in available_days:

        if not remaining_tasks:
            break

        # Tasks whose exam has not passed
        valid_tasks = []

        for task in remaining_tasks:

            if study_day <= task[
                "Exam Date"
            ]:

                valid_tasks.append(
                    task
                )

        if not valid_tasks:
            continue

        # Calculate urgency
        scored_tasks = []

        for task in valid_tasks:

            days_left = (
                task["Exam Date"]
                - study_day
            ).days

            priority_weight = {

                "High": 3,

                "Medium": 2,

                "Low": 1
            }[
                task["Priority"]
            ]

            urgency = (
                priority_weight * 100
                +
                task["Priority Score"]
                +
                max(
                    0,
                    30 - days_left * 2
                )
            )

            scored_tasks.append(
                (
                    urgency,
                    task
                )
            )

        scored_tasks.sort(
            key=lambda x: x[0],
            reverse=True
        )

        # Give 2 study sessions per day
        daily_tasks = scored_tasks[:2]

        for _, task in daily_tasks:

            timetable.append({

                "Date":
                    study_day,

                "Day":
                    study_day.strftime(
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

            remaining_tasks.remove(
                task
            )

    return pd.DataFrame(
        timetable
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "📚 Exam Intelligence"
)

st.sidebar.write(
    "Plan your preparation across "
    "multiple subjects."
)

st.sidebar.divider()

st.sidebar.metric(
    "Subjects Added",
    len(
        st.session_state.subjects
    )
)


# =========================================================
# HEADER
# =========================================================

st.title(
    "📚 Exam Intelligence Assistant"
)

st.write(
    "Analyze previous-year papers across multiple "
    "subjects and create a personalized study plan."
)


# =========================================================
# SUBJECT CREATION
# =========================================================

st.header(
    "📖 Add Your Subjects"
)

with st.expander(
    "➕ Add a new subject",
    expanded=True
):

    subject_name = st.text_input(
        "Subject Name",
        placeholder="Example: Applied Mathematics"
    )

    exam_date = st.date_input(
        "Exam Date",
        value=date.today() + timedelta(days=7)
    )

    syllabus = st.text_area(
        "Syllabus — one topic per line",
        height=160,
        placeholder=(
            "Review of successive differentiation\n"
            "Leibnitz theorem and problems\n"
            "Taylor's and Maclaurin's theorem\n"
            "Beta and gamma functions\n"
            "Tracing curves"
        )
    )

    papers = st.file_uploader(
        "Upload previous-year question papers",
        type=["pdf"],
        accept_multiple_files=True,
        key="new_subject_papers"
    )

    if st.button(
        "➕ Add Subject"
    ):

        if not subject_name.strip():

            st.warning(
                "Please enter the subject name."
            )

        elif not syllabus.strip():

            st.warning(
                "Please enter the syllabus."
            )

        elif not papers:

            st.warning(
                "Please upload at least one "
                "question paper."
            )

        else:

            topics = [

                topic.strip()

                for topic in
                syllabus.split("\n")

                if topic.strip()
            ]

            new_subject = {

                "name":
                    subject_name.strip(),

                "exam_date":
                    exam_date,

                "topics":
                    topics,

                "papers":
                    papers
            }

            st.session_state.subjects.append(
                new_subject
            )

            st.success(
                f"{subject_name} added successfully!"
            )

            st.rerun()


# =========================================================
# DISPLAY ADDED SUBJECTS
# =========================================================

if st.session_state.subjects:

    st.subheader(
        "📚 Your Subjects"
    )

    for index, subject in enumerate(
        st.session_state.subjects
    ):

        col1, col2 = st.columns(
            [5, 1]
        )

        with col1:

            st.info(
                f"📘 **{subject['name']}**\n\n"
                f"📅 Exam: "
                f"{subject['exam_date'].strftime('%d %B %Y')}\n\n"
                f"📚 Topics: "
                f"{len(subject['topics'])}\n\n"
                f"📄 Previous papers: "
                f"{len(subject['papers'])}"
            )

        with col2:

            if st.button(
                "🗑️ Remove",
                key=f"remove_{index}"
            ):

                st.session_state.subjects.pop(
                    index
                )

                st.rerun()


# =========================================================
# ANALYZE EVERYTHING
# =========================================================

st.divider()

if st.session_state.subjects:

    if st.button(
        "🚀 Analyze All Subjects & Create Study Plan",
        type="primary"
    ):

        all_subject_data = []

        for subject in st.session_state.subjects:

            st.subheader(
                f"🔍 Analyzing {subject['name']}"
            )

            results_df, messages = (
                analyze_subject(
                    subject["name"],
                    subject["topics"],
                    subject["papers"]
                )
            )

            for message in messages:

                st.write(message)

            if results_df.empty:

                st.warning(
                    f"No questions could be "
                    f"extracted from "
                    f"{subject['name']}."
                )

                continue

            priority_df = (
                calculate_topic_priority(
                    results_df,
                    subject["topics"],
                    len(subject["papers"])
                )
            )

            all_subject_data.append({

                "name":
                    subject["name"],

                "exam_date":
                    subject["exam_date"],

                "results_df":
                    results_df,

                "priority_df":
                    priority_df
            })

        if not all_subject_data:

            st.error(
                "No subject could be analyzed."
            )

            st.stop()

        # =================================================
        # OVERALL DASHBOARD
        # =================================================

        st.divider()

        st.header(
            "📊 Overall Exam Dashboard"
        )

        total_questions = sum(
            len(
                subject["results_df"]
            )
            for subject in all_subject_data
        )

        total_topics = sum(
            len(
                subject["priority_df"]
            )
            for subject in all_subject_data
        )

        high_topics = sum(
            len(
                subject["priority_df"][
                    subject["priority_df"][
                        "Priority"
                    ] == "High"
                ]
            )
            for subject in all_subject_data
        )

        col1, col2, col3, col4 = st.columns(
            4
        )

        with col1:

            st.metric(
                "Subjects",
                len(all_subject_data)
            )

        with col2:

            st.metric(
                "Questions Analyzed",
                total_questions
            )

        with col3:

            st.metric(
                "Syllabus Topics",
                total_topics
            )

        with col4:

            st.metric(
                "High Priority Topics",
                high_topics
            )


        # =================================================
        # SUBJECT PRIORITY
        # =================================================

        st.subheader(
            "🔥 Subject Priority"
        )

        subject_rows = []

        for subject in all_subject_data:

            priority_df = subject[
                "priority_df"
            ]

            high_count = len(
                priority_df[
                    priority_df["Priority"]
                    == "High"
                ]
            )

            total_score = (
                priority_df[
                    "Priority Score"
                ].sum()
            )

            days_left = (
                subject["exam_date"]
                - date.today()
            ).days

            subject_rows.append({

                "Subject":
                    subject["name"],

                "Exam Date":
                    subject["exam_date"],

                "Days Left":
                    max(
                        days_left,
                        0
                    ),

                "High Priority Topics":
                    high_count,

                "Priority Score":
                    round(
                        total_score,
                        1
                    )
            })

        subject_df = pd.DataFrame(
            subject_rows
        )

        st.dataframe(
            subject_df,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # SUBJECT CHART
        # =================================================

        fig = px.bar(
            subject_df,
            x="Subject",
            y="Priority Score",
            text="Priority Score",
            title="Overall Subject Priority"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # =================================================
        # SUBJECT-WISE ANALYSIS
        # =================================================

        st.header(
            "📚 Subject-wise Analysis"
        )

        for subject in all_subject_data:

            with st.expander(
                f"📘 {subject['name']}"
            ):

                priority_df = subject[
                    "priority_df"
                ]

                results_df = subject[
                    "results_df"
                ]

                st.write(
                    f"**Exam Date:** "
                    f"{subject['exam_date'].strftime('%d %B %Y')}"
                )

                st.dataframe(
                    priority_df,
                    use_container_width=True,
                    hide_index=True
                )

                chart_df = priority_df[
                    [
                        "Topic",
                        "Questions"
                    ]
                ]

                fig = px.bar(
                    chart_df,
                    x="Topic",
                    y="Questions",
                    text="Questions",
                    title=(
                        f"Topic Frequency — "
                        f"{subject['name']}"
                    )
                )

                fig.update_layout(
                    xaxis_tickangle=-35
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.write(
                    "**Question → Topic Mapping**"
                )

                st.dataframe(
                    results_df[
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


        # =================================================
        # SMART STUDY TIMETABLE
        # =================================================

        st.divider()

        st.header(
            "📅 Your Personalized Study Timetable"
        )

        timetable = generate_timetable(
            all_subject_data
        )

        if timetable.empty:

            st.warning(
                "A timetable could not be generated."
            )

        else:

            st.success(
                "Your study plan has been generated "
                "based on exam dates and topic priorities."
            )

            st.dataframe(
                timetable,
                use_container_width=True,
                hide_index=True
            )

            # -------------------------------------------------
            # DAILY PLAN
            # -------------------------------------------------

            st.subheader(
                "🗓️ Daily Study Plan"
            )

            unique_dates = (
                timetable["Date"]
                .unique()
            )

            for study_date in unique_dates:

                day_tasks = timetable[
                    timetable["Date"]
                    == study_date
                ]

                st.markdown(
                    f"### 📅 "
                    f"{study_date.strftime('%A, %d %B')}"
                )

                for _, task in day_tasks.iterrows():

                    priority_icon = {

                        "High": "🔴",

                        "Medium": "🟡",

                        "Low": "🟢"

                    }.get(
                        task["Priority"],
                        "⚪"
                    )

                    st.write(
                        f"{priority_icon} "
                        f"**{task['Subject']}** — "
                        f"{task['Topic']} "
                        f"({task['Study Time']})"
                    )


        # =================================================
        # STUDY NOW
        # =================================================

        st.divider()

        st.header(
            "🎯 What Should You Study First?"
        )

        candidates = []

        for subject in all_subject_data:

            priority_df = subject[
                "priority_df"
            ]

            for _, row in priority_df.iterrows():

                if row["Questions"] == 0:
                    continue

                days_left = (
                    subject["exam_date"]
                    - date.today()
                ).days

                priority_weight = {

                    "High": 3,

                    "Medium": 2,

                    "Low": 1

                }[
                    row["Priority"]
                ]

                score = (
                    priority_weight * 100
                    +
                    row["Priority Score"]
                    +
                    max(
                        0,
                        30 - days_left * 2
                    )
                )

                candidates.append({

                    "score":
                        score,

                    "Subject":
                        subject["name"],

                    "Topic":
                        row["Topic"],

                    "Priority":
                        row["Priority"],

                    "Exam Date":
                        subject["exam_date"],

                    "Questions":
                        row["Questions"]
                })

        candidates.sort(
            key=lambda x: x["score"],
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
                f"{first['Exam Date'].strftime('%d %B %Y')}"
            )

else:

    st.info(
        "👆 Add your subjects above to begin."
    )
