
import io
import re
from datetime import date, timedelta
from difflib import SequenceMatcher

import pandas as pd
import plotly.express as px
import streamlit as st
from pypdf import PdfReader

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Exam Intelligence Assistant",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff, #edf4ff);
}
.hero {
    padding: 25px;
    border-radius: 18px;
    background: white;
    border: 1px solid #dce4f2;
    margin-bottom: 20px;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
defaults = {
    "syllabus_text": "",
    "syllabus_name": "",
    "syllabus_db": {},
    "subjects": [],
    "results": {},
    "completed": set()
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# PDF AND TEXT UTILITIES
# --------------------------------------------------
def clean_line(text):
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t|")


def normalize(text):
    text = str(text).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf(uploaded_file):
    try:
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages).strip()

    except Exception as error:
        st.error(f"Could not read PDF: {error}")
        return ""


def extract_questions(uploaded_file):
    text = extract_pdf(uploaded_file)

    if not text:
        return []

    lines = [clean_line(x) for x in text.splitlines()]
    lines = [x for x in lines if x]

    questions = []
    current = []

    pattern = re.compile(
        r"^(?:(?:question\s*)?\d{1,2}\s*[\).:]|"
        r"[a-h]\s*[\).])",
        re.I
    )

    for line in lines:
        if pattern.match(line):
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
        questions = [
            clean_line(x)
            for x in re.split(r"\n+|(?<=[?])\s+", text)
            if len(clean_line(x)) >= 20
        ]

    unique = []
    seen = set()

    for question in questions:
        key = normalize(question)

        if key and key not in seen:
            unique.append(question)
            seen.add(key)

    return unique


# --------------------------------------------------
# SYLLABUS ANALYSER
# --------------------------------------------------
def get_unit_number(line):
    match = re.match(
        r"^\s*(?:unit|module)\s*[-:]?\s*"
        r"(\d+|[IVXLCDM]+)\b",
        clean_line(line),
        re.I
    )

    if not match:
        return None

    value = match.group(1).upper()

    if value.isdigit():
        return int(value)

    roman = {
        "I": 1, "II": 2, "III": 3, "IV": 4,
        "V": 5, "VI": 6, "VII": 7, "VIII": 8
    }

    return roman.get(value)


def is_unit_heading(line):
    return get_unit_number(line) is not None


def extract_topics(lines):
    topics = []

    ignored = [
        "textbook", "reference book", "course outcome",
        "learning outcome", "credits", "lecture hours",
        "course code", "total marks", "practical hours"
    ]

    for line in lines:
        line = clean_line(line)

        if not line or is_unit_heading(line):
            continue

        if any(word in line.lower() for word in ignored):
            continue

        if len(line) < 4 or len(line) > 300:
            continue

        line = re.sub(r"^[•●▪◦*\-–—]+\s*", "", line)
        line = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", line)

        # Split clear separators while preserving phrases.
        pieces = re.split(r"\s+[;|]\s+", line)

        for topic in pieces:
            topic = clean_line(topic)

            if len(topic) >= 4:
                topics.append(topic)

    result = []
    seen = set()

    for topic in topics:
        key = normalize(topic)

        if key and key not in seen:
            result.append(topic)
            seen.add(key)

    return result


def build_syllabus_database(text):
    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    if not lines:
        return {}

    database = {}
    current_unit = None
    unit_lines = {}

    # Detect units throughout the uploaded syllabus.
    for line in lines:
        number = get_unit_number(line)

        if number is not None:
            current_unit = number

            if current_unit not in unit_lines:
                unit_lines[current_unit] = []

        elif current_unit is not None:
            unit_lines[current_unit].append(line)

    if unit_lines:
        database["Uploaded Syllabus"] = {
            "units": {
                number: extract_topics(content)
                for number, content in unit_lines.items()
            }
        }

    return database


def topic_similarity(question, topic):
    question_words = set(normalize(question).split())
    topic_words = set(normalize(topic).split())

    if not topic_words:
        return 0.0

    overlap = len(question_words & topic_words) / len(topic_words)

    sequence = SequenceMatcher(
        None,
        normalize(question),
        normalize(topic)
    ).ratio()

    return min(1.0, 0.75 * overlap + 0.25 * sequence)


# --------------------------------------------------
# QUESTION ANALYSIS
# --------------------------------------------------
def question_type(question):
    q = question.lower()

    if re.search(r"\b(derive|prove|show that)\b", q):
        return "Derivation"

    if re.search(
        r"\b(calculate|compute|solve|find the value|"
        r"determine|numerical)\b",
        q
    ):
        return "Numerical"

    return "Theory"


def question_difficulty(question):
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


def question_marks(question):
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


def map_questions(questions, topics, paper_name):
    records = []

    for question in questions:
        ranked = sorted(
            [
                (topic_similarity(question, topic), topic)
                for topic in topics
            ],
            reverse=True
        )

        score, topic = ranked[0] if ranked else (0, "Unmapped")

        if score < 0.12:
            topic = "Unmapped / review manually"
            score = 0

        records.append({
            "Paper": paper_name,
            "Question": question,
            "Topic": topic,
            "Match Confidence (%)": round(score * 100),
            "Type": question_type(question),
            "Difficulty": question_difficulty(question),
            "Marks": question_marks(question)
        })

    return records


def analyse_topics(topics, mapped, exam_date):
    rows = []

    for topic in topics:
        matches = [
            row for row in mapped
            if row["Topic"] == topic
        ]

        frequency = len(matches)

        marks = sum(
            row["Marks"] or 0
            for row in matches
        )

        hard = sum(
            row["Difficulty"] == "Hard"
            for row in matches
        )

        rows.append({
            "Topic": topic,
            "Question Frequency": frequency,
            "Known Marks": marks,
            "Theory": sum(
                row["Type"] == "Theory" for row in matches
            ),
            "Numerical": sum(
                row["Type"] == "Numerical" for row in matches
            ),
            "Derivation": sum(
                row["Type"] == "Derivation" for row in matches
            ),
            "Hard Questions": hard
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    max_frequency = max(1, df["Question Frequency"].max())
    max_marks = max(1, df["Known Marks"].max())

    days_left = max(1, (exam_date - date.today()).days)
    urgency = min(1, 7 / days_left)

    history_score = df["Question Frequency"] / max_frequency
    marks_score = df["Known Marks"] / max_marks
    hard_score = (
        df["Hard Questions"] /
        df["Question Frequency"].clip(lower=1)
    )

    df["Priority Score"] = (
        55 * history_score
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

    df["Reason"] = df.apply(
        lambda row:
            f"{int(row['Question Frequency'])} matching question(s); "
            f"{int(row['Known Marks'])} identified mark(s)."
            if row["Question Frequency"] > 0
            else "No clear PYQ match. This topic may still be important.",
        axis=1
    )

    return df.sort_values(
        ["Priority Score", "Question Frequency"],
        ascending=False
    ).reset_index(drop=True)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.title("🎓 Exam Intelligence")

    page = st.radio(
        "Navigation",
        ["Exam Intelligence", "My Study Plan"]
    )

    st.divider()

    if st.session_state.syllabus_name:
        st.success("Syllabus uploaded")
        st.caption(st.session_state.syllabus_name)
    else:
        st.info("Upload your syllabus PDF to begin.")


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------
if page == "Exam Intelligence":

    st.markdown("""
    <div class="hero">
        <h1>🎓 Exam Intelligence Assistant</h1>
        <p>Analyse your syllabus and previous papers to plan revision.</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Upload the combined syllabus.
    st.header("1. Upload Semester Syllabus")

    syllabus_file = st.file_uploader(
        "Upload the complete syllabus PDF",
        type=["pdf"],
        key="syllabus_upload"
    )

    if syllabus_file is not None:
        if syllabus_file.name != st.session_state.syllabus_name:
            with st.spinner("Reading syllabus PDF..."):
                text = extract_pdf(syllabus_file)

                st.session_state.syllabus_text = text
                st.session_state.syllabus_name = syllabus_file.name
                st.session_state.syllabus_db = (
                    build_syllabus_database(text)
                )
                st.session_state.subjects = []
                st.session_state.results = {}

            if text:
                st.success("Syllabus text extracted.")
            else:
                st.error(
                    "No readable text found. This may be a scanned PDF."
                )

    if st.session_state.syllabus_text:
        with st.expander("Preview extracted syllabus"):
            st.text_area(
                "Extracted text",
                st.session_state.syllabus_text[:15000],
                height=250,
                disabled=True
            )

    # 2. Select exam scope.
    st.header("2. Configure Your Exam")

    if not st.session_state.syllabus_db:
        st.info("Upload a readable syllabus PDF first.")

    else:
        database = st.session_state.syllabus_db
        detected_subject = list(database.keys())[0]
        units = database[detected_subject]["units"]

        exam_subject = st.text_input(
            "Subject name",
            value=detected_subject
        )

        exam_date = st.date_input(
            "Exam date",
            value=date.today() + timedelta(days=7),
            min_value=date.today()
        )

        mode = st.radio(
            "What is included in the exam?",
            [
                "Whole unit(s)",
                "Individual topics",
                "Mix of units and topics"
            ],
            horizontal=True
        )

        chosen_units = []
        chosen_topics = []

        if mode in ["Whole unit(s)", "Mix of units and topics"]:
            chosen_units = st.multiselect(
                "Select complete units",
                list(units.keys()),
                format_func=lambda number: f"Unit {number}"
            )

        all_topics = []

        for number, topic_list in units.items():
            for topic in topic_list:
                all_topics.append(f"Unit {number} — {topic}")

        if mode in ["Individual topics", "Mix of units and topics"]:
            selected_labels = st.multiselect(
                "Select individual topics",
                all_topics
            )

            chosen_topics = [
                label.split(" — ", 1)[1]
                for label in selected_labels
            ]

        manual_topics = st.text_area(
            "Additional topics (optional, one per line)",
            placeholder="Add topics not detected in the syllabus"
        )

        # Combine selected unit and topic content.
        topics = []
        topic_units = {}

        for number in chosen_units:
            for topic in units[number]:
                if topic not in topics:
                    topics.append(topic)
                    topic_units[topic] = f"Unit {number}"

        for label in (
            selected_labels
            if mode in ["Individual topics", "Mix of units and topics"]
            else []
        ):
            unit_label, topic = label.split(" — ", 1)

            if topic not in topics:
                topics.append(topic)
                topic_units[topic] = unit_label

        for topic in manual_topics.splitlines():
            topic = clean_line(topic)

            if topic and topic not in topics:
                topics.append(topic)
                topic_units[topic] = "Manually added"

        st.caption(f"{len(topics)} topic(s) selected.")

        # 3. Upload previous question papers.
        st.header("3. Upload Previous-Year Papers")

        pyq_files = st.file_uploader(
            "Upload PYQ PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="pyq_upload"
        )

        if st.button(
            "Analyse Syllabus and Question Papers",
            type="primary",
            use_container_width=True
        ):
            if not topics:
                st.error("Select at least one unit or topic.")
            else:
                with st.spinner("Analysing topics and question papers..."):
                    mapped = []

                    for paper in pyq_files or []:
                        questions = extract_questions(paper)

                        mapped.extend(
                            map_questions(
                                questions,
                                topics,
                                paper.name
                            )
                        )

                    priority_df = analyse_topics(
                        topics,
                        mapped,
                        exam_date
                    )

                    st.session_state.results[exam_subject] = {
                        "exam_date": exam_date,
                        "topics": topics,
                        "topic_units": topic_units,
                        "mapping": mapped,
                        "priority": priority_df
                    }

                st.success("Analysis completed.")

    # 4. Display analysis.
    if st.session_state.results:
        st.divider()
        st.header("4. Analysis Results")

        selected_result = st.selectbox(
            "View subject",
            list(st.session_state.results.keys())
        )

        data = st.session_state.results[selected_result]
        df = data["priority"]
        mapped = data["mapping"]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Topics", len(df))
        c2.metric("Questions extracted", len(mapped))
        c3.metric(
            "High-priority topics",
            int((df["Priority"] == "HIGH").sum())
        )
        c4.metric(
            "Days until exam",
            max(0, (data["exam_date"] - date.today()).days)
        )

        st.subheader("Topic Priority Ranking")

        st.caption(
            "Scores are heuristic estimates based on uploaded papers, "
            "identified marks, question difficulty and exam urgency. "
            "They do not predict which questions will appear."
        )

        display_df = df.copy()
        display_df.insert(
            1,
            "Unit",
            display_df["Topic"].map(data["topic_units"]).fillna("")
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.plotly_chart(
            px.bar(
                df,
                x="Topic",
                y="Priority Score",
                color="Priority",
                title="Topic Priority Scores"
            ),
            use_container_width=True
        )

        st.plotly_chart(
            px.bar(
                df,
                x="Topic",
                y="Question Frequency",
                title="Topic Frequency in Uploaded Papers"
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

            with st.expander("Questions requiring manual review"):
                uncertain = mapped_df[
                    (mapped_df["Topic"] == "Unmapped / review manually")
                    | (mapped_df["Match Confidence (%)"] < 25)
                ]

                if uncertain.empty:
                    st.success("No low-confidence matches flagged.")
                else:
                    st.dataframe(
                        uncertain,
                        use_container_width=True,
                        hide_index=True
                    )

            st.download_button(
                "Download question mapping CSV",
                mapped_df.to_csv(index=False).encode("utf-8"),
                file_name="question_mapping.csv",
                mime="text/csv"
            )
        else:
            st.info(
                "No questions were extracted. Check that your PDFs "
                "contain readable text."
            )

        st.download_button(
            "Download topic priority CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="topic_priority.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# STUDY PLAN
# --------------------------------------------------
else:
    st.markdown("""
    <div class="hero">
        <h1>📅 My Study Plan</h1>
        <p>Track revision tasks and focus on important topics.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.results:
        st.info(
            "Upload your syllabus, select exam topics and run the analysis first."
        )

    else:
        tasks = []

        for subject, data in st.session_state.results.items():
            for _, row in data["priority"].iterrows():
                tasks.append({
                    "Subject": subject,
                    "Topic": row["Topic"],
                    "Priority": row["Priority"],
                    "Priority Score": row["Priority Score"],
                    "Exam Date": data["exam_date"]
                })

        plan = pd.DataFrame(tasks)

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

        done = 0

        for _, row in plan.iterrows():
            task_id = f"{row['Subject']}|{row['Topic']}"

            checked = st.checkbox(
                f"{row['Subject']} — {row['Topic']} "
                f"({row['Priority']})",
                value=task_id in st.session_state.completed,
                key="task_" + str(abs(hash(task_id)))
            )

            if checked:
                st.session_state.completed.add(task_id)
            else:
                st.session_state.completed.discard(task_id)

            done += int(checked)

        if len(plan):
            st.progress(done / len(plan))

        st.caption(f"{done} of {len(plan)} revision tasks completed.")

        st.download_button(
            "Download Study Plan CSV",
            plan.to_csv(index=False).encode("utf-8"),
            file_name="study_plan.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Exam Intelligence Assistant | Review extracted syllabus topics "
    "and question matches before making study decisions. "
    "Scanned PDFs may require OCR support."
)
