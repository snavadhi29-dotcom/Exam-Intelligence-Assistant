import streamlit as st
import pandas as pd
import plotly.express as px
import re
from difflib import SequenceMatcher
from io import BytesIO

from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_bytes


# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Exam Intelligence Assistant",
    page_icon="📚",
    layout="wide"
)


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-\+\=\(\)\/\.]", " ", text)
    return text.strip()


def get_words(text):
    text = clean_text(text)

    words = re.findall(r"[a-zA-Z]+", text)

    stop_words = {
        "the", "and", "for", "with", "from", "that", "this",
        "find", "show", "prove", "calculate", "using", "given",
        "derive", "determine", "solve", "evaluate", "write",
        "explain", "define", "of", "to", "in", "on", "is", "are",
        "a", "an", "be", "if", "or", "as", "by", "at", "it",
        "its", "let", "where", "which", "then", "also"
    }

    return [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]


# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------

def extract_normal_pdf_text(pdf_bytes):
    """
    Try extracting selectable text from a normal PDF.
    """

    try:
        reader = PdfReader(BytesIO(pdf_bytes))

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    except Exception:
        return ""


# ---------------------------------------------------------
# OCR FOR SCANNED PDF
# ---------------------------------------------------------

def extract_ocr_text(pdf_bytes):
    """
    Convert PDF pages into images and use Tesseract OCR.
    This is completely free and does not require an API key.
    """

    try:
        images = convert_from_bytes(
            pdf_bytes,
            dpi=180,
            fmt="jpeg"
        )

        all_text = []

        progress = st.progress(0)

        total_pages = len(images)

        for i, image in enumerate(images):

            text = pytesseract.image_to_string(
                image,
                config="--psm 6"
            )

            if text:
                all_text.append(text)

            progress.progress((i + 1) / total_pages)

        progress.empty()

        return "\n".join(all_text)

    except Exception as e:

        st.error(
            "OCR could not process this PDF. "
            "Please check that the OCR dependencies are installed."
        )

        st.exception(e)

        return ""


# ---------------------------------------------------------
# MAIN PDF EXTRACTION FUNCTION
# ---------------------------------------------------------

def extract_pdf_text(pdf_file):

    pdf_bytes = pdf_file.getvalue()

    # First try normal PDF text extraction
    normal_text = extract_normal_pdf_text(pdf_bytes)

    if len(normal_text.strip()) >= 50:

        st.success("Readable text found in the PDF.")

        return normal_text

    # If normal extraction fails, use OCR
    st.info(
        "This looks like a scanned/image-based PDF. "
        "Using free OCR to read it..."
    )

    ocr_text = extract_ocr_text(pdf_bytes)

    return ocr_text


# ---------------------------------------------------------
# QUESTION EXTRACTION
# ---------------------------------------------------------

def extract_questions(text):

    text = re.sub(r"\r", "\n", text)

    # Try to detect numbered questions
    pattern = r"(?:^|\n)\s*(?:Q(?:uestion)?\s*)?\d+\s*[\.\):\-]"

    parts = re.split(
        pattern,
        text,
        flags=re.IGNORECASE
    )

    questions = []

    for part in parts:

        part = part.strip()

        if len(part) < 20:
            continue

        part = re.sub(r"\s+", " ", part)

        questions.append(part)

    # If numbered-question detection did not work,
    # use longer lines as possible questions.
    if len(questions) < 2:

        lines = re.split(r"\n+", text)

        questions = []

        for line in lines:

            line = line.strip()

            if len(line) > 25:
                questions.append(line)

    return questions


# ---------------------------------------------------------
# TOPIC SIMILARITY
# ---------------------------------------------------------

def topic_similarity(question, topic):

    question_clean = clean_text(question)
    topic_clean = clean_text(topic)

    # Exact phrase match
    if topic_clean in question_clean:
        return 100

    question_words = set(get_words(question))
    topic_words = set(get_words(topic))

    if not topic_words:
        return 0

    # Word overlap
    common_words = question_words.intersection(topic_words)

    overlap_score = (
        len(common_words) / len(topic_words)
    ) * 100

    # Fuzzy matching
    fuzzy_total = 0

    for topic_word in topic_words:

        best_match = 0

        for question_word in question_words:

            similarity = SequenceMatcher(
                None,
                topic_word,
                question_word
            ).ratio()

            if similarity > best_match:
                best_match = similarity

        fuzzy_total += best_match

    fuzzy_score = (
        fuzzy_total / len(topic_words)
    ) * 100

    # Final score
    final_score = (
        overlap_score * 0.65
        +
        fuzzy_score * 0.35
    )

    return round(final_score, 2)


# ---------------------------------------------------------
# FIND BEST MATCHING SYLLABUS TOPIC
# ---------------------------------------------------------

def find_best_topic(question, topics):

    topic_scores = []

    for topic in topics:

        score = topic_similarity(
            question,
            topic
        )

        topic_scores.append(
            (topic, score)
        )

    topic_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    if not topic_scores:
        return "Unclassified", 0

    best_topic, best_score = topic_scores[0]

    # Minimum confidence threshold
    if best_score < 18:
        return "Unclassified", best_score

    return best_topic, best_score


# ---------------------------------------------------------
# ANALYZE QUESTIONS
# ---------------------------------------------------------

def analyze_questions(questions, topics):

    results = []

    for question in questions:

        topic, score = find_best_topic(
            question,
            topics
        )

        results.append({
            "Question": question,
            "Matched Topic": topic,
            "Match Score": score
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# GENERATE PRIORITY DATA
# ---------------------------------------------------------

def generate_priority_data(results_df, topics):

    topic_counts = {
        topic: 0
        for topic in topics
    }

    # Count matched questions
    for topic in results_df["Matched Topic"]:

        if topic in topic_counts:
            topic_counts[topic] += 1

    data = []

    total_questions = max(
        len(results_df),
        1
    )

    for topic in topics:

        frequency = topic_counts[topic]

        percentage = (
            frequency / total_questions
        ) * 100

        # Priority score
        priority_score = (
            frequency * 10
        )

        if priority_score >= 30:
            priority = "High"

        elif priority_score >= 15:
            priority = "Medium"

        else:
            priority = "Low"

        data.append({
            "Topic": topic,
            "Questions": frequency,
            "Frequency %": round(percentage, 1),
            "Priority Score": priority_score,
            "Priority": priority
        })

    return pd.DataFrame(data)


# ---------------------------------------------------------
# REVISION PLAN
# ---------------------------------------------------------

def generate_revision_plan(priority_df):

    plan = []

    # High priority first
    priority_order = {
        "High": 1,
        "Medium": 2,
        "Low": 3
    }

    sorted_df = priority_df.copy()

    sorted_df["Order"] = sorted_df[
        "Priority"
    ].map(priority_order)

    sorted_df = sorted_df.sort_values(
        by=["Order", "Questions"],
        ascending=[True, False]
    )

    for index, row in sorted_df.iterrows():

        topic = row["Topic"]
        priority = row["Priority"]

        if priority == "High":

            action = (
                "Revise thoroughly + solve previous questions"
            )

        elif priority == "Medium":

            action = (
                "Revise concepts + practice important questions"
            )

        else:

            action = (
                "Quick revision after completing high-priority topics"
            )

        plan.append({
            "Topic": topic,
            "Priority": priority,
            "Recommended Action": action
        })

    return pd.DataFrame(plan)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("⚙️ Exam Setup")

subject = st.sidebar.text_input(
    "Subject",
    placeholder="Example: Applied Mathematics"
)

exam_date = st.sidebar.date_input(
    "Exam Date"
)


# ---------------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------------

st.title("📚 Exam Intelligence Assistant")

st.write(
    "Analyze past exam papers, identify important topics "
    "and generate a smart revision plan."
)


# ---------------------------------------------------------
# SYLLABUS INPUT
# ---------------------------------------------------------

st.subheader("1️⃣ Enter Your Syllabus")

syllabus_text = st.text_area(
    "Enter syllabus topics — one topic per line",
    height=180,
    placeholder=(
        "Example:\n"
        "Thermodynamics\n"
        "First Law of Thermodynamics\n"
        "Entropy\n"
        "Carnot Cycle"
    )
)


# ---------------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------------

st.subheader("2️⃣ Upload Past Exam Papers")

uploaded_file = st.file_uploader(
    "Upload a PDF question paper",
    type=["pdf"]
)


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

if st.button(
    "🔍 Analyze Exam Papers",
    type="primary"
):

    # Check syllabus
    if not syllabus_text.strip():

        st.warning(
            "Please enter your syllabus topics first."
        )

        st.stop()

    # Check PDF
    if uploaded_file is None:

        st.warning(
            "Please upload a past exam paper."
        )

        st.stop()

    # Convert syllabus into list
    topics = [
        topic.strip()
        for topic in syllabus_text.split("\n")
        if topic.strip()
    ]

    if len(topics) == 0:

        st.warning(
            "Please enter at least one syllabus topic."
        )

        st.stop()

    # Extract text
    with st.spinner(
        "Reading your question paper..."
    ):

        extracted_text = extract_pdf_text(
            uploaded_file
        )

    # Check extracted text
    if not extracted_text.strip():

        st.error(
            "No readable text could be extracted from "
            "this PDF."
        )

        st.stop()

    # Extract questions
    questions = extract_questions(
        extracted_text
    )

    if len(questions) == 0:

        st.error(
            "Questions could not be detected from this PDF."
        )

        st.stop()

    st.success(
        f"Successfully extracted approximately "
        f"{len(questions)} questions."
    )

    # Analyze
    with st.spinner(
        "Analyzing questions and matching them with your syllabus..."
    ):

        results_df = analyze_questions(
            questions,
            topics
        )

        priority_df = generate_priority_data(
            results_df,
            topics
        )

        revision_df = generate_revision_plan(
            priority_df
        )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    st.divider()

    st.header("📊 Exam Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Questions Found",
            len(questions)
        )

    with col2:
        st.metric(
            "Syllabus Topics",
            len(topics)
        )

    with col3:
        high_topics = len(
            priority_df[
                priority_df["Priority"] == "High"
            ]
        )

        st.metric(
            "High Priority Topics",
            high_topics
        )


    # -----------------------------------------------------
    # TOPIC FREQUENCY
    # -----------------------------------------------------

    st.subheader("📈 Topic Frequency")

    chart_df = priority_df[
        ["Topic", "Questions"]
    ].copy()

    fig = px.bar(
        chart_df,
        x="Topic",
        y="Questions",
        title="How Frequently Each Topic Appears",
        text="Questions"
    )

    fig.update_layout(
        xaxis_tickangle=-35
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------------------
    # PRIORITY TABLE
    # -----------------------------------------------------

    st.subheader("🎯 Topic Priority")

    display_priority = priority_df[
        [
            "Topic",
            "Questions",
            "Frequency %",
            "Priority"
        ]
    ]

    st.dataframe(
        display_priority,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # REVISION PLAN
    # -----------------------------------------------------

    st.subheader("📝 Smart Revision Plan")

    st.dataframe(
        revision_df,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # QUESTION-TO-TOPIC MAPPING
    # -----------------------------------------------------

    st.subheader("🔗 Question → Topic Mapping")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # IMPORTANT TOPICS
    # -----------------------------------------------------

    high_priority_topics = priority_df[
        priority_df["Priority"] == "High"
    ]["Topic"].tolist()

    if high_priority_topics:

        st.subheader("🔥 Focus on These Topics First")

        for topic in high_priority_topics:

            st.write(
                f"• **{topic}**"
            )

    else:

        st.info(
            "No high-priority topics were detected. "
            "Try uploading more previous papers for better analysis."
        )


    # -----------------------------------------------------
    # EXAM INFORMATION
    # -----------------------------------------------------

    st.divider()

    st.caption(
        f"Subject: {subject if subject else 'Not specified'}"
    )

    st.caption(
        f"Exam Date: {exam_date}"
    )
