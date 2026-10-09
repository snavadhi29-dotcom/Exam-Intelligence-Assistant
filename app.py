
import io
import math
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

st.set_page_config(page_title="Exam Intelligence Assistant", page_icon="📚", layout="wide")

# ----------------------------------------------------------------------
# 1. TEXT EXTRACTION (PDF / images, with optional OCR for messy scans)
# ----------------------------------------------------------------------


def ocr_image(img):
    try:
        import pytesseract
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def extract_text(uploaded):
    """Return raw text from a PDF or image upload."""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        if fitz is None:
            return ""
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            txt = page.get_text()
            if len(txt.strip()) < 50:  # probably scanned -> OCR fallback
                try:
                    from PIL import Image
                    pix = page.get_pixmap(dpi=200)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    txt = ocr_image(img) or txt
                except Exception:
                    pass
            pages.append(txt)
        return "\n".join(pages)
    try:
        from PIL import Image
        return ocr_image(Image.open(io.BytesIO(data)))
    except Exception:
        return ""


# ----------------------------------------------------------------------
# 2. SYLLABUS PARSING (units -> topics)
# ----------------------------------------------------------------------

UNIT_RE = re.compile(r"(?im)^\s*(?:unit|module)\s*[-–:.]?\s*([ivx]+|\d+)\b[\s:.\-–]*")


def parse_syllabus(text):
    """Return list of (unit, topic). Falls back to one 'General' unit."""
    text = text.strip()
    if not text:
        return []
    matches = list(UNIT_RE.finditer(text))
    blocks = []
    if matches:
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            blocks.append((f"Unit {m.group(1).upper()}", text[m.end():end]))
    else:
        blocks.append(("General", text))

    out, seen = [], set()
    for unit, body in blocks:
        parts = re.split(r"[,;•●▪\n]|(?<=[a-z])\.\s", body)
        for p in parts:
            p = re.sub(r"^[\s\d\.\)\-–]+", "", p).strip(" .:-–")
            if 4 <= len(p) <= 90 and p.lower() not in seen:
                seen.add(p.lower())
                out.append((unit, p))
    return out


# ----------------------------------------------------------------------
# 3. QUESTION EXTRACTION + CLEANING
# ----------------------------------------------------------------------

Q_SPLIT = re.compile(r"(?m)^\s*(?:Q\.?\s*\d+[\).:]?|\d{1,2}[\).])\s+")
MARKS_1 = re.compile(r"[\[\(]\s*(\d{1,2})\s*(?:marks?|m)?\s*[\]\)]", re.I)
MARKS_2 = re.compile(r"(\d{1,2})\s*marks?", re.I)


def clean(s):
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_questions(text):
    chunks = Q_SPLIT.split(text)
    if len(chunks) < 3:  # numbering not detected -> split by blank lines
        chunks = re.split(r"\n\s*\n", text)
    questions = []
    for c in chunks:
        c = clean(c)
        if len(c) < 20:
            continue
        marks = None
        m = MARKS_1.findall(c) or MARKS_2.findall(c)
        if m:
            marks = int(m[-1])
        questions.append({"question": c, "marks": marks if marks else 5})
    return questions


def question_type(q):
    q = q.lower()
    if re.search(r"\b(derive|derivation|prove|show that|obtain the expression)\b", q):
        return "Derivation"
    if re.search(r"\b(calculate|compute|find|solve|evaluate|determine|numerical|how many|what is the value)\b", q) and re.search(r"\d", q):
        return "Numerical"
    return "Theory"


def difficulty(q, marks):
    ql = q.lower()
    score = 0
    score += 2 if marks >= 10 else 1 if marks >= 5 else 0
    score += 1 if re.search(r"\b(derive|prove|design|analy[sz]e|evaluate|compare|critically|implement)\b", ql) else 0
    score += 1 if len(q) > 250 else 0
    score -= 1 if re.search(r"\b(define|list|state|name|what is|write short)\b", ql) else 0
    return "Hard" if score >= 3 else "Medium" if score >= 1 else "Easy"


# ----------------------------------------------------------------------
# 4. TOPIC MAPPING (TF-IDF cosine similarity -> explainable)
# ----------------------------------------------------------------------


def map_questions(questions, topics):
    """topics: list of (unit, topic). Adds topic/unit/sim to each question."""
    if not questions or not topics:
        return questions
    topic_texts = [f"{t} {t} {u}" for u, t in topics]
    q_texts = [q["question"] for q in questions]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(topic_texts + q_texts)
    sims = cosine_similarity(vec.transform(q_texts), vec.transform(topic_texts))
    for i, q in enumerate(questions):
        j = int(np.argmax(sims[i]))
        s = float(sims[i][j])
        if s < 0.05:
            q["unit"], q["topic"] = "-", "Unmapped"
        else:
            q["unit"], q["topic"] = topics[j]
        q["sim"] = s
    return questions


# ----------------------------------------------------------------------
# 5. TOPIC STATS + PRIORITY SCORE
# ----------------------------------------------------------------------


def topic_stats(qdf, topics):
    years = sorted(qdf["year"].unique()) if not qdf.empty else []
    total_papers = max(len(years), 1)
    latest = years[-1] if years else None
    rows = []
    for unit, topic in topics:
        sub = qdf[qdf["topic"] == topic] if not qdf.empty else qdf
        marks = int(sub["marks"].sum()) if len(sub) else 0
        papers = sub["year"].nunique() if len(sub) else 0
        per_year = [int(sub[sub["year"] == y]["marks"].sum()) for y in years]
        slope = float(np.polyfit(range(len(years)), per_year, 1)[0]) if len(years) >= 2 else 0.0
        top_type = sub["type"].mode().iloc[0] if len(sub) else "-"
        rows.append(dict(Unit=unit, Topic=topic, Questions=len(sub), Marks=marks,
                         Papers=papers, LatestMarks=int(sub[sub["year"] == latest]["marks"].sum()) if len(sub) else 0,
                         Slope=slope, MainType=top_type,
                         MeanSim=float(sub["sim"].mean()) if len(sub) else 0.0))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    mx_m = max(df["Marks"].max(), 1)
    mx_l = max(df["LatestMarks"].max(), 1)
    mx_s = max(df["Slope"].abs().max(), 1e-9)
    df["Score"] = 100 * (
        0.4 * df["Marks"] / mx_m
        + 0.3 * df["Papers"] / total_papers
        + 0.2 * df["LatestMarks"] / mx_l
        + 0.1 * ((df["Slope"] / mx_s + 1) / 2)
    )
    df.loc[df["Questions"] == 0, "Score"] = 0
    df["Confidence"] = (100 * (
        0.5 * np.minimum(1, total_papers / 3)
        + 0.3 * np.minimum(1, df["MeanSim"] * 3)
        + 0.2 * np.minimum(1, df["Questions"] / 3)
    )).round()
    df.loc[df["Questions"] == 0, "Confidence"] = 20

    def why(r):
        if r["Questions"] == 0:
            return "Not found in past papers - study after the high-priority topics."
        trend = "rising" if r["Slope"] > 0.5 else "falling" if r["Slope"] < -0.5 else "steady"
        return (f"Appeared in {r['Papers']}/{total_papers} papers, {r['Marks']} marks total, "
                f"mostly {r['MainType'].lower()} questions, trend {trend}.")

    df["Reasoning"] = df.apply(why, axis=1)
    df["Score"] = df["Score"].round(1)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df.drop(columns=["Slope", "MeanSim", "LatestMarks"])


# ----------------------------------------------------------------------
# 6. STUDY PLAN GENERATOR
# ----------------------------------------------------------------------


def build_plan(subjects, results, start, hours_per_day):
    rows = []
    last_exam = max(s["exam_date"] for s in subjects.values())
    d = start
    while d < last_exam:
        active = {n: s for n, s in subjects.items() if d < s["exam_date"] and n in results}
        if active:
            w = {n: 1 / max(1, (s["exam_date"] - d).days) for n, s in active.items()}
            tot = sum(w.values())
            for n, s in active.items():
                hrs = round(hours_per_day * w[n] / tot * 2) / 2 or 0.5
                topics = results[n]["Topic"].tolist()
                days_left = (s["exam_date"] - d).days
                study_days = max((s["exam_date"] - start).days - 1, 1)
                if days_left == 1:
                    task = "Revision: " + ", ".join(topics[:5]) + " + solve past papers"
                else:
                    per_day = math.ceil(len(topics) / study_days)
                    idx = (d - start).days
                    chunk = topics[idx * per_day:(idx + 1) * per_day]
                    task = ("Study: " + ", ".join(chunk)) if chunk else "Extra practice: weak topics + past papers"
                rows.append({"Done": False, "Date": d, "Subject": n, "Hours": hrs, "Task": task,
                             "Exam on": s["exam_date"]})
        d += timedelta(days=1)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

st.title("📚 Exam Intelligence Assistant")
st.caption("Upload syllabus + past papers → see what matters most → get a day-wise study plan.")

ss = st.session_state
ss.setdefault("subjects", {})
ss.setdefault("syllabus_full", "")
ss.setdefault("results", {})
ss.setdefault("qdfs", {})
ss.setdefault("plan", None)

tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Syllabus", "2️⃣ Subjects & Papers", "3️⃣ Analysis", "4️⃣ Study Plan"])

# ---- TAB 1 ----
with tab1:
    st.subheader("Upload your semester syllabus (optional)")
    up = st.file_uploader("Semester syllabus PDF / image", type=["pdf", "png", "jpg", "jpeg"], key="syl")
    if up and st.button("Extract syllabus text"):
        with st.spinner("Reading..."):
            ss["syllabus_full"] = extract_text(up)
    if ss["syllabus_full"]:
        st.success(f"Extracted {len(ss['syllabus_full'])} characters.")
        with st.expander("Preview extracted text"):
            st.text(ss["syllabus_full"][:3000])
    st.info("Next: add each subject in tab 2. If its name appears in the syllabus PDF, its syllabus is auto-filled.")

# ---- TAB 2 ----
with tab2:
    c1, c2 = st.columns([3, 1])
    new_name = c1.text_input("Subject name (as written in the syllabus)", key="newsub")
    if c2.button("➕ Add subject", use_container_width=True) and new_name.strip():
        name = new_name.strip()
        if name not in ss["subjects"]:
            full = ss["syllabus_full"]
            pos = full.lower().find(name.lower()) if full else -1
            guess = full[pos:pos + 6000] if pos >= 0 else ""
            ss["subjects"][name] = {"exam_date": date.today() + timedelta(days=6), "files": []}
            ss[f"syl_{name}"] = guess

    for name, s in ss["subjects"].items():
        with st.expander(f"📘 {name}", expanded=True):
            s["exam_date"] = st.date_input("Exam date", s["exam_date"], key=f"d_{name}")
            st.text_area("Syllabus / units & topics for this exam (edit freely; use 'Unit 1' headings, separate topics with commas)",
                         key=f"syl_{name}", height=160)
            topics = parse_syllabus(ss.get(f"syl_{name}", ""))
            st.caption(f"Detected {len(topics)} topics across {len(set(u for u, _ in topics))} unit(s).")
            files = st.file_uploader("Previous year papers (PDF/images)", type=["pdf", "png", "jpg", "jpeg"],
                                     accept_multiple_files=True, key=f"f_{name}")
            for f in files or []:
                m = re.search(r"(20\d{2})", f.name)
                st.number_input(f"Year for {f.name}", 2000, 2100, int(m.group(1)) if m else date.today().year - 1,
                                key=f"y_{name}_{f.name}")
            s["files"] = files or []
            if st.button("🗑 Remove subject", key=f"rm_{name}"):
                del ss["subjects"][name]
                st.rerun()

# ---- TAB 3 ----
with tab3:
    if not ss["subjects"]:
        st.warning("Add at least one subject first.")
    elif st.button("🔍 Analyze all subjects", type="primary"):
        ss["results"], ss["qdfs"] = {}, {}
        for name, s in ss["subjects"].items():
            topics = parse_syllabus(ss.get(f"syl_{name}", ""))
            if not topics:
                st.error(f"{name}: no topics found - fill the syllabus box.")
                continue
            allq = []
            with st.spinner(f"Analyzing {name}..."):
                for f in s["files"]:
                    year = ss.get(f"y_{name}_{f.name}", date.today().year - 1)
                    qs = extract_questions(extract_text(f))
                    for q in qs:
                        q["year"] = int(year)
                    allq += qs
                allq = map_questions(allq, topics)
                for q in allq:
                    q["type"] = question_type(q["question"])
                    q["difficulty"] = difficulty(q["question"], q["marks"])
                qdf = pd.DataFrame(allq) if allq else pd.DataFrame(
                    columns=["question", "marks", "year", "unit", "topic", "sim", "type", "difficulty"])
                ss["qdfs"][name] = qdf
                ss["results"][name] = topic_stats(qdf, topics)
        ss["plan"] = None

    for name, res in ss["results"].items():
        qdf = ss["qdfs"][name]
        st.markdown(f"## 📘 {name}")
        if qdf.empty:
            st.warning("No questions extracted - upload papers (text-based PDFs work best).")
        a, b, c = st.columns(3)
        a.metric("Questions found", len(qdf))
        b.metric("Papers", qdf["year"].nunique() if len(qdf) else 0)
        c.metric("Unmapped questions", int((qdf["topic"] == "Unmapped").sum()) if len(qdf) else 0)

        st.markdown("**🎯 Study priority (highest first)**")
        st.dataframe(res, use_container_width=True, hide_index=True)

        if len(qdf):
            c1, c2, c3 = st.columns(3)
            c1.markdown("**Question types**")
            c1.bar_chart(qdf["type"].value_counts())
            c2.markdown("**Difficulty distribution**")
            c2.bar_chart(qdf["difficulty"].value_counts())
            c3.markdown("**Marks by topic**")
            c3.bar_chart(res.set_index("Topic")["Marks"])
            st.markdown("**📈 Topic-wise marks across years**")
            st.dataframe(qdf.pivot_table(index="topic", columns="year", values="marks", aggfunc="sum", fill_value=0),
                         use_container_width=True)
            with st.expander("See extracted questions & their mapped topic (to verify)"):
                st.dataframe(qdf[["year", "marks", "type", "difficulty", "topic", "question"]],
                             use_container_width=True, hide_index=True)

# ---- TAB 4 ----
with tab4:
    if not ss["results"]:
        st.warning("Run the analysis first.")
    else:
        c1, c2 = st.columns(2)
        start = c1.date_input("Start studying from", date.today())
        hrs = c2.slider("Study hours per day", 1.0, 14.0, 6.0, 0.5)
        if st.button("🗓 Generate study plan", type="primary"):
            valid = {n: s for n, s in ss["subjects"].items() if s["exam_date"] > start}
            if not valid:
                st.error("Exam dates must be after the start date.")
            else:
                ss["plan"] = build_plan(valid, ss["results"], start, hrs)
        if ss["plan"] is not None and len(ss["plan"]):
            st.caption("Tick the checkbox as you finish each task.")
            edited = st.data_editor(
                ss["plan"], hide_index=True, use_container_width=True,
                disabled=["Date", "Subject", "Hours", "Task", "Exam on"],
                column_config={"Done": st.column_config.CheckboxColumn("✅")},
                key="plan_editor",
            )
            ss["plan"] = edited
            st.progress(float(edited["Done"].mean()), text=f"{int(edited['Done'].sum())}/{len(edited)} tasks done")
            st.download_button("⬇ Download plan (CSV)", edited.to_csv(index=False), "study_plan.csv", "text/csv")
