import streamlit as st
import pandas as pd
import plotly.express as px
import re
from pypdf import PdfReader
from difflib import SequenceMatcher
from collections import Counter

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Exam Intelligence Assistant",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🎓 Exam Intelligence Assistant")

st.write(
    "Analyze past exam papers, identify important topics, "
    "and generate a smart revision plan."
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📚 Exam Setup")

subject = st.sidebar.text_input(
    "Subject",
    placeholder="Example: Applied Mathematics"
)

exam_date = st.sidebar.date_input("Exam Date")

# =========================================================
# SYLLABUS INPUT
# =========================================================

st.header("1️⃣ Enter Your Syllabus")

syllabus = st.text_area(
    "Enter your syllabus topics — one topic per line:",
    placeholder="""Example:
Differential Equations
Matrices
Laplace Transform
Fourier Series
Probability"""
)

# =========================================================
# PDF UPLOAD
# =========================================================

st.header("2️⃣ Upload Past Papers")

uploaded_files = st.file_uploader(
    "Upload one or more previous question papers",
    type=["pdf"],
    accept_multiple_files=True
)

# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove unusual characters
    text = re.sub(r"[^\w\s\-\+\=\(\)\/\.]", " ", text)

    return text.strip()


# =========================================================
# EXTRACT TEXT FROM PDF
# =========================================================

def extract_pdf_text(file):

    reader = PdfReader(file)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


# =========================================================
# SPLIT PAPER INTO QUESTIONS
# =========================================================

def extract_questions(text):

    # Normalize line breaks
    text = re.sub(r"\r", "\n", text)

    # Try to detect question numbers
    pattern = r"(?:^|\n)\s*(?:Q(?:uestion)?\s*)?\d+\s*[\.\):\-]"

    parts = re.split(pattern, text, flags=re.IGNORECASE)

    questions = []

    for part in parts:

        part = part.strip()

        # Ignore very small fragments
        if len(part) < 20:
            continue

        # Remove excessive spaces
        part = re.sub(r"\s+", " ", part)

        questions.append(part)

    # If question numbering wasn't detected,
    # use lines/sentences as fallback
    if len(questions) < 2:

        lines = re.split(r"\n+", text)

        questions = [
            line.strip()
            for line in lines
            if len(line.strip()) > 25
        ]

    return questions


# =========================================================
# WORD TOKENIZATION
# =========================================================

def get_words(text):

    text = clean_text(text)

    words = re.findall(r"[a-zA-Z]+", text)

    # Common words which do not help topic matching
    stop_words = {
        "the", "a", "an", "and", "or", "of",
        "to", "in", "on", "for", "with",
        "find", "calculate", "solve", "show",
        "prove", "using", "given", "evaluate",
        "determine", "explain", "derive",
        "is", "are", "be", "from", "by",
        "following", "following"
    }

    return [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]


# =========================================================
# TOPIC MATCHING
# =========================================================

def topic_similarity(question, topic):

    question_clean = clean_text(question)
    topic_clean = clean_text(topic)

    # -----------------------------------------------------
    # 1. Direct phrase match
    # -----------------------------------------------------

    if topic_clean in question_clean:

        return 100

    # -----------------------------------------------------
    # 2. Word overlap
    # -----------------------------------------------------

    question_words = set(get_words(question))
    topic_words = set(get_words(topic))

    if not topic_words:

        return 0

    common_words = question_words.intersection(topic_words)

    overlap_score = (
        len(common_words) / len(topic_words)
    ) * 100

    # -----------------------------------------------------
    # 3. Individual word similarity
    # -----------------------------------------------------

    fuzzy_score = 0

    for topic_word in topic_words:

        best_match = 0

        for question_word in question_words:

            similarity = SequenceMatcher(
                None,
                topic_word,
                question_word
            ).ratio()

            best_match = max(
                best_match,
                similarity
            )

        fuzzy_score += best_match

    fuzzy_score = (
        fuzzy_score / len(topic_words)
    ) * 100

    # -----------------------------------------------------
    # Combine scores
    # -----------------------------------------------------

    final_score = (
        overlap_score * 0.65
        +
        fuzzy_score * 0.35
    )

    return round(final_score, 2)


# =========================================================
# FIND BEST TOPIC
# =========================================================

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

    best_topic, best_score = topic_scores[0]

    # -----------------------------------------------------
    # Confidence threshold
    # -----------------------------------------------------

    if best_score < 18:

        return "Unclassified", best_score

    return best_topic, best_score


# =========================================================
# ANALYZE QUESTIONS
# =========================================================

def analyze_questions(questions, topics):

    results = []

    for question in questions:

        topic, score = find_best_topic(
            question,
            topics
        )

        results.append({
            "Question": question,
            "Mapped Topic": topic,
            "Match Score": score
        })

    return pd.DataFrame(results)


# =========================================================
# GENERATE REVISION PLAN
# =========================================================

def generate_revision_plan(priority_df):

    plan = []

    ranked_topics = priority_df[
        priority_df["Topic"] != "Unclassified"
    ].sort_values(
        "Priority Score",
        ascending=False
    )

    for index, row in ranked_topics.iterrows():

        score = row["Priority Score"]
        topic = row["Topic"]

        if score >= 75:

            level = "🔥 Very High Priority"

        elif score >= 50:

            level = "🟠 High Priority"

        elif score >= 25:

            level = "🟡 Medium Priority"

        else:

            level = "🟢 Low Priority"

        plan.append({
            "Topic": topic,
            "Priority": level,
            "Score": score
        })

    return pd.DataFrame(plan)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🔍 Analyze Exam Papers",
    type="primary"
):

    # -----------------------------------------------------
    # Check syllabus
    # -----------------------------------------------------

    if not syllabus.strip():

        st.error(
            "Please enter your syllabus topics first."
        )

        st.stop()

    # -----------------------------------------------------
    # Convert syllabus into topic list
    # -----------------------------------------------------

    topics = [
        topic.strip()
        for topic in syllabus.split("\n")
        if topic.strip()
    ]

    # -----------------------------------------------------
    # Check PDF
    # -----------------------------------------------------

    if not uploaded_files:

        st.error(
            "Please upload at least one question paper PDF."
        )

        st.stop()

    # -----------------------------------------------------
    # Extract questions
    # -----------------------------------------------------

    all_questions = []

    extraction_status = []

    for uploaded_file in uploaded_files:

        try:

            text = extract_pdf_text(
                uploaded_file
            )

            questions = extract_questions(text)

            all_questions.extend(
                questions
            )

            extraction_status.append(
                f"✅ {uploaded_file.name}: "
                f"{len(questions)} questions extracted"
            )

        except Exception as e:

            extraction_status.append(
                f"❌ {uploaded_file.name}: "
                f"Could not read PDF"
            )

    # -----------------------------------------------------
    # Extraction status
    # -----------------------------------------------------

    st.subheader("📄 PDF Processing")

    for status in extraction_status:

        st.write(status)

    # -----------------------------------------------------
    # Check extraction
    # -----------------------------------------------------

    if not all_questions:

        st.error(
            "No readable text was found in the PDF. "
            "This may be a scanned/image-only PDF."
        )

        st.info(
            "For this version, please try a PDF containing "
            "selectable text. OCR for scanned papers can "
            "be added next."
        )

        st.stop()

    # -----------------------------------------------------
    # Analyze
    # -----------------------------------------------------

    df = analyze_questions(
        all_questions,
        topics
    )

    # =====================================================
    # DASHBOARD
    # =====================================================

    st.divider()

    st.header("📊 Exam Intelligence Dashboard")

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    total_questions = len(df)

    identified_questions = len(
        df[df["Mapped Topic"] != "Unclassified"]
    )

    topic_counts = df[
        df["Mapped Topic"] != "Unclassified"
    ]["Mapped Topic"].value_counts()

    if len(topic_counts) > 0:

        most_frequent_topic = (
            topic_counts.index[0]
        )

    else:

        most_frequent_topic = "None"

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Questions Analyzed",
            total_questions
        )

    with col2:

        st.metric(
            "Topics Identified",
            identified_questions
        )

    with col3:

        st.metric(
            "Most Frequent Topic",
            most_frequent_topic
        )

    # =====================================================
    # TOPIC FREQUENCY
    # =====================================================

    st.subheader("🔥 Topic Frequency")

    if len(topic_counts) > 0:

        frequency_df = pd.DataFrame({
            "Topic": topic_counts.index,
            "Question Frequency": topic_counts.values
        })

        st.dataframe(
            frequency_df,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # GRAPH
        # -------------------------------------------------

        fig = px.bar(
            frequency_df,
            x="Topic",
            y="Question Frequency",
            title="Most Frequently Asked Topics"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "No syllabus topics could be matched."
        )

    # =====================================================
    # PRIORITY SCORE
    # =====================================================

    st.subheader("⭐ Smart Topic Priority")

    if len(topic_counts) > 0:

        max_frequency = max(
            topic_counts.values
        )

        priority_data = []

        for topic, frequency in topic_counts.items():

            score = round(
                (frequency / max_frequency) * 100
            )

            if score >= 75:

                priority = "🔥 Very High"

            elif score >= 50:

                priority = "🟠 High"

            elif score >= 25:

                priority = "🟡 Medium"

            else:

                priority = "🟢 Low"

            priority_data.append({
                "Topic": topic,
                "Frequency": frequency,
                "Priority Score": score,
                "Priority": priority
            })

        priority_df = pd.DataFrame(
            priority_data
        )

        priority_df = priority_df.sort_values(
            "Priority Score",
            ascending=False
        )

        st.dataframe(
            priority_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        priority_df = pd.DataFrame()

    # =====================================================
    # QUESTION MAPPING
    # =====================================================

    st.subheader("📝 Question-to-Topic Mapping")

    display_df = df.copy()

    display_df["Match Score"] = (
        display_df["Match Score"].astype(str)
        + "%"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # REVISION PLAN
    # =====================================================

    st.subheader("🗓️ Personalized Revision Plan")

    if not priority_df.empty:

        revision_plan = generate_revision_plan(
            priority_df
        )

        for i, row in revision_plan.iterrows():

            st.write(
                f"**{i + 1}. {row['Topic']}**  \n"
                f"{row['Priority']} — "
                f"Priority Score: {row['Score']}/100"
            )

        st.success(
            "Revision plan generated from "
            "past-paper frequency."
        )

    else:

        st.info(
            "A revision plan will appear after "
            "topics are successfully identified."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Exam Intelligence Assistant • GDG Prototype"
)
