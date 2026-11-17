"""
Structured metadata extraction for legal case PDFs.

This is deliberately rule-based (regex + keyword matching) rather than a
second ML model — it's fast, needs no extra download, and its behavior is
easy to explain/debug. Swap in a proper NER model later (see bottom of
this file) if you need it to generalize beyond Indian court judgment
formatting conventions.
"""
import re
from collections import Counter

PAGE_NUMBER_LINE_PATTERN = re.compile(r"^-?\s*\d{1,4}\s*-?$")


def clean_boilerplate(text: str, min_repeats: int = 3, max_len: int = 60) -> str:
    lines = text.split("\n")
    stripped = [l.strip() for l in lines]
    counts = Counter(l for l in stripped if l and len(l) <= max_len)
    boilerplate = {l for l, c in counts.items() if c >= min_repeats}
    kept = [
        line
        for line, s in zip(lines, stripped)
        if s not in boilerplate and not PAGE_NUMBER_LINE_PATTERN.match(s)
    ]
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Case number (e.g. "Civil Appeal No. 1123 of 2024", "Criminal Appeal No. 415
# of 2021", "Appeal No. 5567 of 2024")
# ---------------------------------------------------------------------------
CASE_NUMBER_PATTERN = re.compile(
    r"((?:Civil|Criminal|Writ|Special)?\s*(?:Appeal|Petition|Suit|Case)\s+No\.?\s*[\w/]+\s+of\s+\d{4})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Court name — first line containing "COURT" or "TRIBUNAL"
# ---------------------------------------------------------------------------
COURT_PATTERN = re.compile(r"^.*\b(COURT|TRIBUNAL)\b.*$", re.IGNORECASE | re.MULTILINE)

# ---------------------------------------------------------------------------
# Parties — "<Name> ... Appellant" / "<Name> ... Respondent"
#
# IMPORTANT: this is a single combined pattern, not two independent ones.
# PDF text extraction (pdfplumber) frequently merges the appellant/Versus/
# respondent lines into ONE line of text when they're rendered close
# together (confirmed empirically — this isn't a hypothetical edge case).
# An independent "respondent" regex with no anchor has nothing stopping it
# from greedily matching from the *appellant's* name all the way to the
# *next* occurrence of "Respondent", swallowing "... Appellant Versus" in
# the middle. Anchoring both captures to a single "Appellant ... Versus ...
# Respondent" pattern forces the respondent capture to start only after the
# literal "Versus" separator, regardless of whether the source had line
# breaks there or not.
# ---------------------------------------------------------------------------
PARTIES_PATTERN = re.compile(
    r"([A-Z][\w.&' ]{2,80}?)\s*\.{2,}\s*Appellant\s*Versus\s*([A-Z][\w.&' ]{2,80}?)\s*\.{2,}\s*Respondent",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Fallback parties pattern for Writ Petitions, which are captioned
# "BETWEEN: <petitioner block> ...PETITIONER AND: <respondent block>
# ...RESPONDENT(S)" instead of the Appellant/Versus/Respondent style above.
# The petitioner/respondent blocks are multi-line (name, address, age,
# description) so we only pull the first line of each block as a readable
# label rather than the whole block.
# ---------------------------------------------------------------------------
PETITIONER_RESPONDENT_PATTERN = re.compile(
    r"BETWEEN\s*:?\s*(.+?)\s*[.…]+\s*PETITIONERS?\b.*?"
    r"AND\s*:?\s*(.+?)\s*[.…]+\s*RESPONDENTS?\b",
    re.IGNORECASE | re.DOTALL,
)

# ---------------------------------------------------------------------------
# Judge(s) — "HON'BLE MR./MRS./MS. JUSTICE <NAME>" (appears in the coram /
# "BEFORE" block, and often again next to "CORAM:").
# ---------------------------------------------------------------------------
JUDGE_PATTERN = re.compile(
    r"HON['’]BLE\s+(?:MR\.?|MRS\.?|MS\.?|DR\.?)?\s*JUSTICE\s+([A-Z][A-Z.\s]{2,60}?)(?=\n|,|$)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Acts and sections cited — e.g. "Section 25 of the Karnataka Societies
# Registration Act, 1960", "Section 302 IPC".
# ---------------------------------------------------------------------------
SECTION_PATTERN = re.compile(r"Section\s+\d+[A-Za-z()]*", re.IGNORECASE)
ACT_PATTERN = re.compile(r"[A-Z][A-Za-z,'&\s]{3,80}?\bAct,?\s*\d{4}", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Operative "ORDER" block — the final directions a court issues, usually a
# standalone "ORDER" heading followed by numbered/roman-numeral clauses
# ending near the judge's signature ("Sd/-").
#
# The heading is matched letter-by-letter with optional whitespace between
# each character (\s* between O-R-D-E-R) because pdfplumber sometimes
# extracts a bold/underlined "ORDER" heading as letter-spaced text —
# "O R D E R" — rather than the plain word. A plain "ORDER" still matches
# fine since \s* also accepts zero spaces. Confirmed empirically: without
# this, the whole ORDER block match fails silently and extraction falls
# back to the (much weaker) holding-keyword sentence scan below, which
# only picks up whichever single clause happens to contain a matching
# keyword instead of every numbered clause in the order.
# ---------------------------------------------------------------------------
ORDER_BLOCK_PATTERN = re.compile(
    r"\n\s*O\s*R\s*D\s*E\s*R\s*\n(.*?)(?:\n\s*Sd/-|\Z)", re.IGNORECASE | re.DOTALL
)
ORDER_ITEM_SPLIT_PATTERN = re.compile(r"\(\s*[ivxlcdm]+\s*\)\s*", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Generic "holding" sentences — used as a fallback key-points source when a
# document doesn't have a clean standalone ORDER block (e.g. some appeal
# judgments fold the disposal into the last paragraph).
#
# Includes both "is/are/stands <verb>" phrasing and the bare past-participle
# ("quashed", "allowed", ...) since courts phrase operative clauses
# inconsistently — e.g. "are hereby quashed" doesn't contain the literal
# substring "is quashed". The bare forms are enough on their own to catch
# any of the "is/are/stands/hereby" variants, so the longer phrases below
# are kept mainly for readability/intent rather than being load-bearing.
# ---------------------------------------------------------------------------
HOLDING_KEYWORDS = [
    "is allowed", "is dismissed", "is disposed of", "is set aside", "is quashed",
    "is directed", "are directed", "stands convicted", "stands acquitted",
    "is remitted", "is upheld", "shall be", "is liable to be",
    "hereby quashed", "are quashed", "stands quashed",
    "hereby allowed", "are allowed", "stands allowed",
    "hereby dismissed", "are dismissed",
    "hereby set aside", "are set aside",
    "hereby directed", "hereby convicted", "hereby acquitted",
]

# ---------------------------------------------------------------------------
# Dates — common Indian judgment date formats: "5th June 2019", "14 March
# 2021", "9th November, 2023"
# ---------------------------------------------------------------------------
DATE_PATTERN = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:(?:DAY\s+OF\s+)?)(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December),?\s+\d{4}\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Area of law — keyword buckets, checked in order; first match wins for the
# "primary" label, but all matches are returned so you can see overlap.
# Extend this dict as you encounter new case types.
# ---------------------------------------------------------------------------
AREA_OF_LAW_KEYWORDS = {
    "Criminal Law": [
        "indian penal code", "ipc", "section 302", "section 379", "conviction",
        "acquittal", "sessions court", "murder", "theft", "rigorous imprisonment",
        "prosecution", "accused",
    ],
    "Motor Accident Claims": [
        "motor accident", "claims tribunal", "mact", "motorcycle", "road accident",
    ],
    "Insurance Law": [
        "insurance", "insurer", "policy", "repudiat", "premium", "uberrima fides",
    ],
    "Property / Contract Law": [
        "specific performance", "agreement to sell", "sale deed", "possession of a flat",
        "builder-buyer agreement", "property tax", "conveyance",
    ],
    "Consumer Law": [
        "consumer forum", "consumer complaint", "deficiency in service",
    ],
    "Tax / Regulatory Law": [
        "tax schedule", "regulatory authority", "penalty order", "statutory returns",
        "assessing authority",
    ],
    "Banking Law": [
        "bank", "savings bank account", "fixed deposit", "debit freeze",
        "freezing of accounts", "nationalised bank", "branch manager",
    ],
    "Societies / Co-operative Law": [
        "societies registration act", "registered society", "deputy registrar",
        "co-operative society", "society",
    ],
    "Writ Jurisdiction / Constitutional Law": [
        "writ petition", "writ of mandamus", "article 226", "article 227",
        "constitution of india", "mandamus", "certiorari", "fundamental right",
        "habeas corpus", "quo warranto",
    ],
    "Family Law": [
        "divorce", "claim for maintenance", "maintenance petition",
        "custody of the child", "hindu marriage act", "matrimonial",
        "restitution of conjugal rights", "domestic violence", "alimony",
        "guardianship",
    ],
    "Labour / Employment Law": [
        "termination of service", "industrial dispute", "workman", "retrenchment",
        "gratuity", "provident fund", "labour court", "wrongful dismissal",
        "reinstatement with back wages",
    ],
    "Land Acquisition / Revenue Law": [
        "land acquisition act", "compensation for acquisition", "khata", "mutation",
        "revenue records", "survey number", "land tribunal",
    ],
    "Arbitration Law": [
        "arbitration and conciliation act", "arbitral tribunal", "arbitral award",
        "section 34", "section 11 of the arbitration",
    ],
    "Service / Administrative Law": [
        "departmental enquiry", "suspension order", "compulsory retirement",
        "seniority list", "promotion policy", "disciplinary proceedings",
    ],
    "Bail / Criminal Procedure": [
        "anticipatory bail", "regular bail", "section 438", "section 439",
        "bail application", "custodial interrogation",
    ],
}


def _first_match(pattern: re.Pattern, text: str) -> str | None:
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def extract_case_number(text: str) -> str | None:
    return _first_match(CASE_NUMBER_PATTERN, text)


def extract_court(text: str) -> str | None:
    m = COURT_PATTERN.search(text)
    return m.group(0).strip() if m else None


def _first_meaningful_line(block: str) -> str | None:
    """Given a multi-line party block, return a short readable label: the
    first non-empty line, with a leading list number/dot stripped."""
    for line in block.splitlines():
        line = line.strip().strip(",")
        if not line:
            continue
        line = re.sub(r"^\d+\.\s*", "", line)
        if line:
            return line[:120]
    return None


def extract_parties(text: str) -> dict:
    m = PARTIES_PATTERN.search(text)
    if m:
        return {"appellant": m.group(1).strip(), "respondent": m.group(2).strip()}

    # Fallback for writ-petition style captions (Petitioner/Respondent).
    m = PETITIONER_RESPONDENT_PATTERN.search(text)
    if m:
        petitioner = _first_meaningful_line(m.group(1))
        respondent = _first_meaningful_line(m.group(2))
        if petitioner or respondent:
            return {"appellant": petitioner, "respondent": respondent}

    return {"appellant": None, "respondent": None}


def extract_judges(text: str, limit: int = 5) -> list[str]:
    seen = []
    for m in JUDGE_PATTERN.finditer(text):
        name = re.sub(r"\s+", " ", m.group(1)).strip(" .")
        if name and name not in seen:
            seen.append(name)
        if len(seen) >= limit:
            break
    return seen


def extract_case_type(case_number: str | None) -> str | None:
    if not case_number:
        return None
    prefix = re.split(r"\bNo\.?\b", case_number, flags=re.IGNORECASE)[0]
    return prefix.strip().title() or None


def extract_acts_and_sections(text: str, limit: int = 10) -> list[str]:
    raw = []
    for pattern in (ACT_PATTERN, SECTION_PATTERN):
        for m in pattern.finditer(text):
            item = re.sub(r"\s+", " ", m.group(0)).strip()
            if item not in raw:
                raw.append(item)

    # Drop items that are just a shorter substring of a longer match already
    # kept (e.g. "Karnataka Societies Registration Act, 1960" is a substring
    # of "of the Karnataka Societies Registration Act, 1960") — keep the
    # longer, more descriptive form.
    deduped = []
    for item in sorted(raw, key=len, reverse=True):
        if not any(item in longer for longer in deduped):
            deduped.append(item)

    # Restore original order of first appearance.
    deduped_set = set(deduped)
    ordered = [item for item in raw if item in deduped_set]
    return ordered[:limit]


def extract_key_points(text: str, limit: int = 10) -> list[str]:
    """Pull out the document's most important, skimmable points: prefer the
    court's final operative ORDER (its numbered directions are already the
    clearest possible summary), falling back to sentences that contain
    strong holding/disposal language when no clean ORDER block exists."""
    order_match = ORDER_BLOCK_PATTERN.search(text)
    if order_match:
        block = order_match.group(1).strip()
        items = [
            re.sub(r"\s+", " ", part).strip(" ;:.")
            for part in ORDER_ITEM_SPLIT_PATTERN.split(block)
        ]
        items = [i for i in items if len(i) > 5]
        if items:
            return items[:limit]

    # Fallback: scan sentences for strong holding/disposal keywords.
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    points = []
    for sent in sentences:
        sent_clean = sent.strip()
        if not sent_clean or len(sent_clean) > 400:
            continue
        lowered = sent_clean.lower()
        if any(kw in lowered for kw in HOLDING_KEYWORDS):
            if sent_clean not in points:
                points.append(sent_clean)
        if len(points) >= limit:
            break
    return points


def extract_dates(text: str, limit: int = 8) -> list[str]:
    # Dedupe while preserving order of first appearance
    seen = []
    for m in DATE_PATTERN.finditer(text):
        d = m.group(0)
        if d not in seen:
            seen.append(d)
        if len(seen) >= limit:
            break
    return seen


def classify_area_of_law(text: str) -> dict:
    lowered = text.lower()
    matches = []
    for area, keywords in AREA_OF_LAW_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in lowered]
        if hits:
            matches.append((area, len(hits)))

    matches.sort(key=lambda x: x[1], reverse=True)
    primary = matches[0][0] if matches else "Unclassified / Other"
    all_areas = [area for area, _ in matches] or ["Unclassified / Other"]
    return {"primary_area": primary, "all_matched_areas": all_areas}


def extract_metadata(text: str) -> dict:
    text = clean_boilerplate(text)
    case_number = extract_case_number(text)
    word_count = len(text.split())
    return {
        "case_number": case_number,
        "case_type": extract_case_type(case_number),
        "court": extract_court(text),
        "judges": extract_judges(text),
        "parties": extract_parties(text),
        "dates_mentioned": extract_dates(text),
        "area_of_law": classify_area_of_law(text),
        "acts_and_sections": extract_acts_and_sections(text),
        "key_points": extract_key_points(text),
        "word_count": word_count,
        "estimated_reading_time_min": max(1, round(word_count / 200)),
    }

