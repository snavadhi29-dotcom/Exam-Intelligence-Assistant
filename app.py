import streamlit as st
import re
from collections import Counter
import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Exam Intelligence Assistant",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🎓 Exam Intelligence Assistant")
st.write(
    "Analyze past exam papers, identify important topics, "
    "and generate a smart revision plan."
)

st.divider()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("📚 Exam Setup")

subject = st.sidebar.text_input(
    "Subject",
    placeholder="Example: Engineering Mathematics"
)

exam_date = st.sidebar.date_input(
    "Exam Date"
)

# ---------------------------------------------------
# SYLLABUS
# ---------------------------------------------------

st.header("1️⃣ Enter Your Syllabus")

syllabus = st.text_area(
    "Enter your syllabus topics (one topic per line):",
    placeholder="""Matrices
Differential Equations
Laplace Transform
Fourier Series
Probability
Complex Numbers"""
)

# ---------------------------------------------------
# PAST PAPER UPLOAD
# ---------------------------------------------------

st.header("2️⃣ Upload Past Papers")

uploaded_files = st.file_uploader(
    "Upload past question papers",
    type=["txt", "pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ---------------------------------------------------
# DEMO QUESTIONS
# ---------------------------------------------------

demo_questions = [
    "Solve the differential equation using Laplace Transform.",
    "Find the eigenvalues and eigenvectors of the given matrix.",
    "Explain the Fourier series representation of a function.",
    "Solve a problem based on probability distribution.",
    "Find the Laplace transform of the given function.",
    "Calculate eigenvalues of the matrix.",
    "Explain Fourier series and its applications.",
    "Solve the differential equation.",
    "Find the probability of the given event.",
    "Solve a matrix using eigenvectors."
]

# ---------------------------------------------------
# ANALYSIS FUNCTION
# ---------------------------------------------------

def analyze_questions(questions, topics):

    results = []

    for question in questions:

        question_lower = question.lower()

        matched_topic = "Other"

        for topic in topics:
            if topic.lower() in question_lower:
                matched_topic = topic
                break

        results.append({
            "Question": question,
            "Topic": matched_topic
        })

    return pd.DataFrame(results)


# ---------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------

if st.button("🔍 Analyze Exam Papers", type="primary"):

    # Use demo questions if files are not uploaded
    if not uploaded_files:
        questions = demo_questions

        st.info(
            "Demo mode is active. Upload your own papers for "
            "real exam-paper analysis."
        )

    else:
        questions = []

        for file in uploaded_files:

            # TXT files can be read directly
            if file.name.endswith(".txt"):
                text = file.read().decode("utf-8")

                extracted = re.split(
                    r'\n+|\d+\.\s+',
                    text
                )

                questions.extend(
                    [q.strip() for q in extracted if len(q.strip()) > 15]
                )

            else:
                st.warning(
                    f"{file.name}: PDF/image extraction will be "
                    "added in the next version. Demo analysis is "
                    "being used for now."
                )

        if not questions:
            questions = demo_questions

    # Get topics
    if syllabus.strip():

        topics = [
            topic.strip()
            for topic in syllabus.split("\n")
            if topic.strip()
        ]

    else:

        topics = [
            "Matrices",
            "Differential Equations",
            "Laplace Transform",
            "Fourier Series",
            "Probability",
            "Complex Numbers"
        ]

        st.info(
            "No syllabus entered. Demo syllabus topics are being used."
        )

    # ---------------------------------------------------
    # ANALYZE QUESTIONS
    # ---------------------------------------------------

    df = analyze_questions(questions, topics)

    st.divider()

    st.header("📊 Exam Intelligence Dashboard")

    # ---------------------------------------------------
    # TOPIC FREQUENCY
    # ---------------------------------------------------

    topic_counts = df["Topic"].value_counts()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Questions Analyzed",
            len(df)
        )

    with col2:
        st.metric(
            "Topics Identified",
            len(topic_counts)
        )

    with col3:
        most_common = topic_counts.index[0]
        st.metric(
            "Most Frequent Topic",
            most_common
        )

    # ---------------------------------------------------
    # FREQUENCY TABLE
    # ---------------------------------------------------

    st.subheader("🔥 Topic Frequency")

    frequency_df = pd.DataFrame({
        "Topic": topic_counts.index,
        "Question Frequency": topic_counts.values
    })

    st.dataframe(
        frequency_df,
        use_container_width=True,
        hide_index=True
    )

    # ---------------------------------------------------
    # GRAPH
    # ---------------------------------------------------

    fig = px.bar(
        frequency_df,
        x="Topic",
        y="Question Frequency",
        title="Topics Asked Most Frequently"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------------------------
    # PRIORITY SCORE
    # ---------------------------------------------------

    st.subheader("⭐ Smart Topic Priority")

    max_frequency = max(topic_counts.values)

    priority_data = []

    for topic, frequency in topic_counts.items():

        priority_score = round(
            (frequency / max_frequency) * 100
        )

        if priority_score >= 75:
            priority = "🔥 Very High"

        elif priority_score >= 50:
            priority = "🟠 High"

        elif priority_score >= 25:
            priority = "🟡 Medium"

        else:
            priority = "🟢 Low"

        priority_data.append({
            "Topic": topic,
            "Frequency": frequency,
            "Priority Score": priority_score,
            "Priority": priority
        })

    priority_df = pd.DataFrame(priority_data)

    priority_df = priority_df.sort_values(
        "Priority Score",
        ascending=False
    )

    st.dataframe(
        priority_df,
        use_container_width=True,
        hide_index=True
    )

    # ---------------------------------------------------
    # QUESTIONS
    # ---------------------------------------------------

    st.subheader("📝 Question Analysis")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # ---------------------------------------------------
    # REVISION PLAN
    # ---------------------------------------------------

    st.subheader("🗓️ Personalized Revision Plan")

    top_topics = priority_df.head(5)["Topic"].tolist()

    st.write(
        "Based on past-paper frequency, focus on these topics first:"
    )

    for i, topic in enumerate(top_topics, 1):

        score = priority_df[
            priority_df["Topic"] == topic
        ]["Priority Score"].iloc[0]

        st.write(
            f"**Day {i}: {topic}** — "
            f"Priority Score: {score}/100"
        )

    st.success(
        "Revision strategy generated successfully! "
        "Start with the highest-priority topics."
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "Exam Intelligence Assistant • GDG Prototype"
)
