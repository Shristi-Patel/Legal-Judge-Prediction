import io
import json

import requests
import streamlit as st

try:
    from pdf2image import convert_from_bytes

    _PREVIEW_AVAILABLE = True
except ImportError:
    _PREVIEW_AVAILABLE = False

try:
    from docx import Document

    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False

st.set_page_config(page_title="Legal-J", page_icon="⚖️", layout="wide")

# Full-width buttons/images: newer Streamlit uses width="stretch", older
# versions use use_container_width=True. Pick whichever this version supports.
import inspect

_STRETCH = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.button).parameters
    else {"use_container_width": True}
)

# -----------------------------------------------------------------------------
# Shared styling. Custom HTML blocks below hardcode their own colors (rather
# than inheriting Streamlit's light/dark theme text color) so they stay
# readable regardless of which theme the user has selected.
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .lj-card {
        background-color: #f7f8fa;
        border: 1px solid #e3e5e9;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .lj-card-label { font-size: 0.8rem; color: #5f6368; margin-bottom: 2px; }
    .lj-card-value { font-size: 1.15rem; font-weight: 700; color: #111318; }
    .lj-chip {
        background: #e8eefc; color: #1a3d7c; border: 1px solid #cfdcf7;
        border-radius: 14px; padding: 4px 12px; margin: 3px; display: inline-block;
        font-size: 0.85rem; font-weight: 500;
    }
    .lj-hero {
        background: linear-gradient(135deg, #1a3d7c 0%, #2c5aa0 100%);
        border-radius: 14px; padding: 22px 26px; margin-bottom: 18px; color: #fff;
    }
    .lj-hero-title { font-size: 1.35rem; font-weight: 700; margin-bottom: 4px; }
    .lj-hero-sub { font-size: 0.92rem; opacity: 0.9; }
    .lj-bar-wrap {
        display: flex; height: 34px; border-radius: 8px; overflow: hidden;
        border: 1px solid #d7d9dc; margin: 10px 0;
    }
    .lj-bar-accept { background: #1f9d55; display: flex; align-items: center;
        justify-content: center; color: #fff; font-weight: 600; font-size: 0.85rem; }
    .lj-bar-reject { background: #d64545; display: flex; align-items: center;
        justify-content: center; color: #fff; font-weight: 600; font-size: 0.85rem; }
    .lj-point {
        background-color: #f7f8fa; border: 1px solid #e3e5e9; border-left: 4px solid #1a3d7c;
        border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; color: #111318;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "analyses" not in st.session_state:
    st.session_state["analyses"] = {}       # filename -> analysis result dict
if "file_bytes" not in st.session_state:
    st.session_state["file_bytes"] = {}     # filename -> raw uploaded PDF bytes
if "active_file" not in st.session_state:
    st.session_state["active_file"] = None

with st.sidebar:
    st.header("Settings")
    backend_url = st.text_input("Backend URL", value="http://127.0.0.1:8000")
    st.divider()
    st.caption(
        "Legal-J reads a case PDF and gives you three things: a judgment "
        "prediction, structured case details, and a detailed, plain-English "
        "breakdown of what the document actually says."
    )

    if st.session_state["analyses"]:
        st.divider()
        st.subheader("📚 Analyzed this session")
        for fname in st.session_state["analyses"]:
            is_active = fname == st.session_state["active_file"]
            if st.button(("● " if is_active else "○ ") + fname, key=f"hist_{fname}", **_STRETCH):
                st.session_state["active_file"] = fname
                st.rerun()
        st.caption("This history is kept only for the current browser session.")

st.title("⚖️ Legal-J")
st.caption("Upload one or more case PDFs to get a judgment prediction and a detailed, structured explanation of each.")

uploaded_files = st.file_uploader("Upload case PDF(s)", type=["pdf"], accept_multiple_files=True)


def _metric_card(label: str, value: str):
    st.markdown(
        f"""<div class="lj-card">
            <div class="lj-card-label">{label}</div>
            <div class="lj-card-value">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _tag_list(items):
    if not items:
        return "<span style='color:#888;'>None detected</span>"
    return "".join(f"<span class='lj-chip'>{i}</span>" for i in items)


@st.cache_data(show_spinner=False)
def _render_pdf_pages(pdf_bytes: bytes, dpi: int = 110):
    return convert_from_bytes(pdf_bytes, dpi=dpi)


def _build_docx_report(result: dict, filename: str) -> bytes:
    doc = Document()
    doc.add_heading(result.get("case_number") or filename, level=1)

    court = result.get("court")
    judges = result.get("judges") or []
    if court:
        doc.add_paragraph(court)
    if judges:
        doc.add_paragraph("Judge(s): " + ", ".join(judges))

    doc.add_heading("Case details", level=2)
    parties = result.get("parties") or {}
    area = result.get("area_of_law") or {}
    details = [
        ("Case type", result.get("case_type") or "Not detected"),
        ("Petitioner / Appellant", parties.get("appellant") or "Not detected"),
        ("Respondent", parties.get("respondent") or "Not detected"),
        ("Area of law", area.get("primary_area") or "Unclassified"),
        ("Dates mentioned", ", ".join(result.get("dates_mentioned") or []) or "None detected"),
        ("Acts & sections cited", ", ".join(result.get("acts_and_sections") or []) or "None detected"),
    ]
    table = doc.add_table(rows=0, cols=2)
    table.style = "Light Grid Accent 1"
    for label, value in details:
        row = table.add_row().cells
        row[0].text = label
        row[1].text = str(value)

    doc.add_heading("Judgment prediction", level=2)
    doc.add_paragraph(
        f"Accepted probability: {result['accepted_prob'] * 100:.1f}%  |  "
        f"Rejected probability: {result['rejected_prob'] * 100:.1f}%"
    )
    doc.add_paragraph(
        ("High" if result.get("high_certainty") else "Low") + " certainty. "
        + result.get("confidence_note", "")
    )

    doc.add_heading("Key points", level=2)
    for i, point in enumerate(result.get("key_points") or [], start=1):
        doc.add_paragraph(f"{i}. {point}")

    doc.add_heading("Narrative summary", level=2)
    doc.add_paragraph(result.get("narrative_summary", ""))

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Analyze any newly uploaded files that aren't in the session cache yet.
# -----------------------------------------------------------------------------
if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state["analyses"]]
    if new_files and st.button(f"Analyze {len(new_files)} new file(s)", type="primary"):
        progress = st.progress(0.0, text="Starting...")
        for i, uf in enumerate(new_files):
            progress.progress(i / len(new_files), text=f"Analyzing {uf.name} ...")
            file_bytes = uf.getvalue()
            try:
                files = {"file": (uf.name, file_bytes, "application/pdf")}
                response = requests.post(f"{backend_url}/analyze", files=files, timeout=300)
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Couldn't reach the backend at {backend_url}. "
                    "Make sure `uvicorn main:app --reload` is running."
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.error(f"The backend took too long to respond on {uf.name}. Try again.")
                st.stop()

            if response.status_code != 200:
                detail = response.json().get("detail", response.text)
                st.error(f"Backend error on {uf.name} ({response.status_code}): {detail}")
                continue

            st.session_state["analyses"][uf.name] = response.json()
            st.session_state["file_bytes"][uf.name] = file_bytes
            st.session_state["active_file"] = uf.name

        progress.progress(1.0, text="Done.")
        st.rerun()

active_file = st.session_state["active_file"]

if not st.session_state["analyses"]:
    st.info("Upload one or more PDFs and click Analyze to get started.")
else:
    if active_file not in st.session_state["analyses"]:
        active_file = next(iter(st.session_state["analyses"]))
        st.session_state["active_file"] = active_file

    if len(st.session_state["analyses"]) > 1:
        active_file = st.selectbox(
            "Viewing analysis for:",
            options=list(st.session_state["analyses"].keys()),
            index=list(st.session_state["analyses"].keys()).index(active_file),
        )
        st.session_state["active_file"] = active_file

    result = st.session_state["analyses"][active_file]
    pdf_bytes = st.session_state["file_bytes"].get(active_file)

    parties = result.get("parties") or {}
    area = result.get("area_of_law") or {}
    judges = result.get("judges") or []
    dates = result.get("dates_mentioned") or []
    acts_sections = result.get("acts_and_sections") or []
    key_points = result.get("key_points") or []
    word_count = result.get("word_count")
    reading_time = result.get("estimated_reading_time_min")
    accepted_pct = result["accepted_prob"] * 100
    rejected_pct = result["rejected_prob"] * 100

    st.success(f"Showing analysis for **{active_file}**")

    # -------------------------------------------------------------------
    # Hero: case title strip
    # -------------------------------------------------------------------
    st.markdown(
        f"""<div class="lj-hero">
            <div class="lj-hero-title">{result.get('case_number') or 'Case number not detected'}</div>
            <div class="lj-hero-sub">{result.get('court') or 'Court not detected'}
                {' · ' + ', '.join(judges) if judges else ''}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    tab_overview, tab_prediction, tab_details, tab_points, tab_summary, tab_doc, tab_raw = st.tabs(
        ["📄 Overview", "🔮 Prediction", "🧾 Case details", "📌 Key points",
         "📝 Summary", "📑 Document", "🗂️ Raw & export"]
    )

    with tab_overview:
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            _metric_card("Case type", result.get("case_type") or "Unclassified")
        with o2:
            _metric_card("Area of law", area.get("primary_area", "Unclassified"))
        with o3:
            _metric_card("Word count", f"{word_count:,}" if word_count else "—")
        with o4:
            _metric_card("Est. reading time", f"{reading_time} min" if reading_time else "—")

        st.markdown("**Predicted outcome at a glance:**")
        lean = "Accepted" if accepted_pct >= rejected_pct else "Rejected"
        certainty_label = "high certainty" if result.get("high_certainty") else "low certainty"
        st.markdown(
            f"The model leans toward **{lean}** ({max(accepted_pct, rejected_pct):.1f}%), "
            f"flagged as *{certainty_label}*. See the Prediction tab for details."
        )

    with tab_prediction:
        st.markdown(
            f"""<div class="lj-bar-wrap">
                <div class="lj-bar-accept" style="width:{accepted_pct:.1f}%;">
                    {accepted_pct:.1f}% Accepted</div>
                <div class="lj-bar-reject" style="width:{rejected_pct:.1f}%;">
                    {rejected_pct:.1f}% Rejected</div>
            </div>""",
            unsafe_allow_html=True,
        )

        p1, p2 = st.columns(2)
        p1.metric("Accepted probability", f"{accepted_pct:.1f}%")
        p2.metric("Rejected probability", f"{rejected_pct:.1f}%")

        certainty_badge = "🟢 High certainty" if result.get("high_certainty") else "🟡 Low certainty"
        st.markdown(f"**Model confidence:** {certainty_badge}")
        st.caption(result.get("confidence_note", ""))
        if result.get("temperature") and result["temperature"] != 1.0:
            st.caption(f"Temperature scaling applied: T = {result['temperature']:.3f}")

    with tab_details:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Case number:** {result.get('case_number') or 'Not detected'}")
            st.markdown(f"**Court:** {result.get('court') or 'Not detected'}")
            st.markdown(f"**Judge(s):** {', '.join(judges) if judges else 'Not detected'}")
            st.markdown(f"**Dates mentioned:** {', '.join(dates) if dates else 'None detected'}")
        with c2:
            st.markdown(f"**Petitioner / Appellant:** {parties.get('appellant') or 'Not detected'}")
            st.markdown(f"**Respondent:** {parties.get('respondent') or 'Not detected'}")
            st.markdown("**All matched areas of law:**")
            st.markdown(_tag_list(area.get("all_matched_areas")), unsafe_allow_html=True)

        st.markdown("**Acts & sections cited:**")
        st.markdown(_tag_list(acts_sections), unsafe_allow_html=True)

    with tab_points:
        st.caption(
            "The court's operative directions where detected, otherwise the "
            "sentences most likely to state the outcome."
        )
        if key_points:
            for i, point in enumerate(key_points, start=1):
                st.markdown(f"""<div class="lj-point"><b>{i}.</b> {point}</div>""", unsafe_allow_html=True)
        else:
            st.info("No clear operative directions or holding statements were detected in this document.")

    with tab_summary:
        st.write(result.get("narrative_summary", ""))

    with tab_doc:
        if pdf_bytes is None:
            st.info("Original file bytes aren't available for this document (re-upload to preview it).")
        elif not _PREVIEW_AVAILABLE:
            st.warning(
                "PDF preview needs `pdf2image` (pip) + `poppler` (system package) installed on "
                "the machine running this Streamlit app."
            )
        else:
            with st.spinner("Rendering pages..."):
                pages = _render_pdf_pages(pdf_bytes)
            st.caption(f"{len(pages)} page(s)")
            page_num = st.number_input(
                "Page", min_value=1, max_value=len(pages), value=1, step=1, key=f"page_{active_file}"
            )
            st.image(pages[page_num - 1], **_STRETCH)

    with tab_raw:
        st.json(result)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(result, indent=2, ensure_ascii=False),
                file_name=f"{active_file}.analysis.json",
                mime="application/json",
                **_STRETCH,
            )
        with dl2:
            if _DOCX_AVAILABLE:
                docx_bytes = _build_docx_report(result, active_file)
                st.download_button(
                    "⬇️ Download Word report",
                    data=docx_bytes,
                    file_name=f"{active_file}.report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    **_STRETCH,
                )
            else:
                st.caption("Install `python-docx` to enable the Word report export.")
