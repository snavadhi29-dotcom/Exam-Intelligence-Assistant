import streamlit as st
import pandas as pd
import plotly.express as px
import re
from io import BytesIO
from difflib import SequenceMatcher

from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_bytes

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Exam Intelligence Assistant",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# GENERAL TEXT CLEANING
# =========================================================

STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this",
    "find", "show", "prove", "calculate", "using", "given",
    "derive", "determine", "solve", "evaluate", "write",
    "explain", "define", "of", "to", "in", "on", "is",
    "are", "a", "an", "be", "if", "or", "as", "by",
    "at", "it", "its", "let", "where", "which", "then",
    "also", "following", "following", "following"
}


def normalize_text(text):
    """Clean text for matching."""

    text = text.lower()

    # Common OCR corrections
    text = text.replace("’", "'")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_words(text):
    """Return useful words from text."""

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

        # Simple normalization for plurals
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"

        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]

        cleaned.append(word)

    return cleaned


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_normal_pdf_text(pdf_bytes):
    """Extract selectable text from a normal PDF."""

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


# =========================================================
# OCR
# =========================================================

def extract_ocr_text(pdf_bytes):

    """Read scanned/image-based PDF using Tesseract OCR."""

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
            "OCR could not read this PDF."
        )

        st.exception(e)

        return ""


def extract_pdf_text(pdf_bytes):

    """
    First try normal text extraction.
    If insufficient text is found, automatically use OCR.
    """

    normal_text = extract_normal_pdf_text(
        pdf_bytes
    )

    if len(normal_text.strip()) >= 80:

        return normal_text, "Text extraction"

    return extract_ocr_text(
        pdf_bytes
    ), "OCR"


# =========================================================
# REMOVE PAPER INSTRUCTIONS / HEADERS
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

    """
    Detect main numbered questions.
    Avoid treating paper instructions as questions.
    """

    text = text.replace("\r", "\n")

    text = remove_instruction_lines(text)

    # Normalize multiple blank lines
    text = re.sub(
        r"\n{2,}",
        "\n",
        text
    )

    # Question starts:
    #
    # Q1.
    # Q1)
    # Q.1
    # Question 1
    # 1.
    # 1)
    # 1:
    #
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

            # Remove very short false detections
            if len(question_text) < 25:
                continue

            # Ignore obvious instruction blocks
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

    # If no numbered questions were found,
    # use paragraph-based extraction.
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
        total / len(topic_words)
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

    # Exact phrase = very strong match
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

    # Combined score
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


def find_best_topic(question, topics):

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

    # More forgiving threshold for OCR
    if best_score < 15:

        return "Unclassified", best_score

    return best_topic, best_score


# =========================================================
# ANALYZE ONE PAPER
# =========================================================

def analyze_paper(
    questions,
    topics,
    paper_name
):

    results = []

    for number, question in enumerate(
        questions,
        start=1
    ):

        topic, score = find_best_topic(
            question,
            topics
        )

        results.append({

            "Paper": paper_name,

            "Question No.": number,

            "Question": question,

            "Matched Topic": topic,

            "Match Score": score
        })

    return pd.DataFrame(
        results
    )


# =========================================================
# BUILD TOPIC STATISTICS
# =========================================================

def build_topic_statistics(
    results_df,
    topics,
    number_of_papers
):

    rows = []

    for topic in topics:

        matched = results_df[
            results_df["Matched Topic"] == topic
        ]

        question_count = len(
            matched
        )

        paper_count = (
            matched["Paper"]
            .nunique()
        )

        # Percentage of papers in which topic appeared
        paper_coverage = (
            paper_count /
            max(number_of_papers, 1)
        ) * 100

        # Percentage of all detected questions
        question_frequency = (
            question_count /
            max(len(results_df), 1)
        ) * 100

        # Combined score
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

    df = pd.DataFrame(rows)

    # Rank topics
    df = df.sort_values(
        by="Priority Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    # Assign priority based on ranking
    # rather than requiring an arbitrary
    # large number of questions.

    total_topics = len(df)

    for index in range(total_topics):

        score = df.loc[
            index,
            "Priority Score"
        ]

        questions = df.loc[
            index,
            "Questions"
        ]

        if questions == 0:

            priority = "Low"

        elif index < max(
            1,
            round(total_topics * 0.30)
        ):

            priority = "High"

        elif index < max(
            2,
            round(total_topics * 0.70)
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
# REVISION PLAN
# =========================================================

def build_revision_plan(
    priority_df
):

    plans = []

    for _, row in priority_df.iterrows():

        topic = row["Topic"]

        priority = row["Priority"]

        questions = row["Questions"]

        if priority == "High":

            action = (
                "Study thoroughly, revise concepts "
                "and solve previous-year questions."
            )

        elif priority == "Medium":

            action = (
                "Revise concepts and practice "
                "important questions."
            )

        else:

            action = (
                "Keep for quick revision after "
                "high and medium priority topics."
            )

        plans.append({

            "Topic": topic,

            "Priority": priority,

            "Past Questions": questions,

            "Recommended Action": action
        })

    return pd.DataFrame(
        plans
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "⚙️ Exam Setup"
)

subject = st.sidebar.text_input(
    "Subject",
    placeholder="Example: Applied Mathematics"
)

exam_date = st.sidebar.date_input(
    "Exam Date"
)


# =========================================================
# MAIN PAGE
# =========================================================

st.title(
    "📚 Exam Intelligence Assistant"
)

st.write(
    "Analyze multiple past exam papers, "
    "identify repeated syllabus topics and "
    "generate a smart revision plan."
)


# =========================================================
# SYLLABUS
# =========================================================

st.subheader(
    "1️⃣ Enter Your Syllabus"
)

syllabus_text = st.text_area(
    "Enter one syllabus topic per line",
    height=180,
    placeholder=(
        "Example:\n"
        "Review of successive differentiation\n"
        "Leibnitz theorem and problems\n"
        "Taylor's and Maclaurin's theorem\n"
        "Asymptotes\n"
        "Beta and gamma functions\n"
        "Tracing curves"
    )
)


# =========================================================
# MULTIPLE PDF UPLOAD
# =========================================================

st.subheader(
    "2️⃣ Upload Previous Exam Papers"
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF question papers",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} PDF(s) uploaded successfully."
    )

    st.write("**Uploaded papers:**")

    for file in uploaded_files:

        st.write(
            f"📄 {file.name}"
        )


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🔍 Analyze Exam Papers",
    type="primary"
):

    # -----------------------------
    # Validate syllabus
    # -----------------------------

    if not syllabus_text.strip():

        st.warning(
            "Please enter your syllabus topics."
        )

        st.stop()

    # -----------------------------
    # Validate files
    # -----------------------------

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF."
        )

        st.stop()

    # -----------------------------
    # Convert syllabus to list
    # -----------------------------

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

    # -----------------------------
    # Analyze all papers
    # -----------------------------

    all_results = []

    st.header(
        "📄 Paper Processing"
    )

    for file in uploaded_files:

        st.write(
            f"**Processing:** {file.name}"
        )

        pdf_bytes = file.getvalue()

        extracted_text, method = (
            extract_pdf_text(
                pdf_bytes
            )
        )

        if not extracted_text.strip():

            st.warning(
                f"No readable text found in {file.name}."
            )

            continue

        questions = extract_questions(
            extracted_text
        )

        st.write(
            f"Detected {len(questions)} "
            f"main questions using {method}."
        )

        if len(questions) == 0:

            st.warning(
                f"Could not detect questions in "
                f"{file.name}."
            )

            continue

        paper_results = analyze_paper(
            questions,
            topics,
            file.name
        )

        all_results.append(
            paper_results
        )

    # -----------------------------
    # Check results
    # -----------------------------

    if not all_results:

        st.error(
            "No questions could be extracted "
            "from the uploaded papers."
        )

        st.stop()

    results_df = pd.concat(
        all_results,
        ignore_index=True
    )

    # -----------------------------
    # Topic statistics
    # -----------------------------

    priority_df = build_topic_statistics(
        results_df,
        topics,
        len(uploaded_files)
    )

    revision_df = build_revision_plan(
        priority_df
    )

    # =====================================================
    # DASHBOARD
    # =====================================================

    st.divider()

    st.header(
        "📊 Exam Intelligence Dashboard"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Papers Analyzed",
            len(uploaded_files)
        )

    with col2:

        st.metric(
            "Questions Detected",
            len(results_df)
        )

    with col3:

        st.metric(
            "Syllabus Topics",
            len(topics)
        )

    with col4:

        high_count = len(
            priority_df[
                priority_df["Priority"]
                == "High"
            ]
        )

        st.metric(
            "High Priority Topics",
            high_count
        )

    # =====================================================
    # TOPIC FREQUENCY
    # =====================================================

    st.subheader(
        "📈 Topic Frequency Across Papers"
    )

    chart_df = priority_df[
        [
            "Topic",
            "Questions"
        ]
    ].copy()

    fig = px.bar(
        chart_df,
        x="Topic",
        y="Questions",
        text="Questions",
        title=(
            "How Often Each Syllabus Topic "
            "Appeared in Previous Papers"
        )
    )

    fig.update_layout(
        xaxis_tickangle=-35
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # PRIORITY TABLE
    # =====================================================

    st.subheader(
        "🎯 Topic Priority"
    )

    st.dataframe(
        priority_df[
            [
                "Topic",
                "Questions",
                "Papers",
                "Paper Coverage %",
                "Question Frequency %",
                "Priority Score",
                "Priority"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # REVISION PLAN
    # =====================================================

    st.subheader(
        "📝 Smart Revision Plan"
    )

    st.dataframe(
        revision_df,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # HIGH PRIORITY TOPICS
    # =====================================================

    high_topics = priority_df[
        priority_df["Priority"] == "High"
    ]

    if len(high_topics) > 0:

        st.subheader(
            "🔥 Focus on These Topics First"
        )

        for _, row in high_topics.iterrows():

            st.write(
                f"🔴 **{row['Topic']}** — "
                f"appeared in "
                f"{row['Papers']} paper(s) "
                f"with {row['Questions']} "
                f"detected question(s)."
            )

    # =====================================================
    # MEDIUM PRIORITY
    # =====================================================

    medium_topics = priority_df[
        priority_df["Priority"] == "Medium"
    ]

    if len(medium_topics) > 0:

        st.subheader(
            "🟡 Next Priority"
        )

        for _, row in medium_topics.iterrows():

            st.write(
                f"🟡 **{row['Topic']}**"
            )

    # =====================================================
    # QUESTION MAPPING
    # =====================================================

    st.subheader(
        "🔗 Question → Topic Mapping"
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

    # =====================================================
    # UNCLASSIFIED QUESTIONS
    # =====================================================

    unclassified = results_df[
        results_df["Matched Topic"]
        == "Unclassified"
    ]

    if len(unclassified) > 0:

        st.subheader(
            "⚠️ Questions That Need Better Matching"
        )

        st.info(
            f"{len(unclassified)} question(s) "
            "could not be confidently matched "
            "to your syllabus."
        )

        st.dataframe(
            unclassified[
                [
                    "Paper",
                    "Question No.",
                    "Question",
                    "Match Score"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # EXAM INFORMATION
    # =====================================================

    st.divider()

    st.caption(
        f"Subject: "
        f"{subject if subject else 'Not specified'}"
    )

    st.caption(
        f"Exam Date: {exam_date}"
    )
