"""
PDF summarization using a local HuggingFace summarization model, combined
with rule-based structured metadata extraction (case_metadata.py) so the
output isn't just a paragraph — it surfaces case number, court, parties,
dates, and area of law alongside the narrative summary.
"""
from transformers import pipeline, AutoTokenizer

from case_metadata import extract_metadata, clean_boilerplate

SUMMARY_MODEL_NAME = "facebook/bart-large-cnn"
CHUNK_TOKEN_LIMIT = 900  # leave headroom under BART's 1024 limit
MIN_SUMMARY_LEN = 60
MAX_SUMMARY_LEN = 180

_summarizer = None
_tokenizer = None


def load_summarizer():
    global _summarizer, _tokenizer
    if _summarizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_NAME)
        _summarizer = pipeline("summarization", model=SUMMARY_MODEL_NAME, tokenizer=_tokenizer)
        print("[summarizer] loaded facebook/bart-large-cnn")
    return _summarizer, _tokenizer


def _chunk_text(text: str, tokenizer, max_tokens: int = CHUNK_TOKEN_LIMIT) -> list[str]:
    """Split text into chunks under max_tokens, breaking on sentence boundaries
    where possible so we don't cut mid-sentence."""
    sentences = text.replace("\n", " ").split(". ")
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_len = len(tokenizer.encode(sent, add_special_tokens=False))
        if current_len + sent_len > max_tokens and current:
            chunks.append(". ".join(current) + ".")
            current, current_len = [], 0
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(". ".join(current) + ".")
    return chunks


def _generate_narrative_summary(text: str) -> dict:
    summarizer, tokenizer = load_summarizer()
    chunks = _chunk_text(text, tokenizer)

    chunk_summaries = []
    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=MAX_SUMMARY_LEN,
            min_length=MIN_SUMMARY_LEN,
            do_sample=False,
        )
        chunk_summaries.append(result[0]["summary_text"])

    combined = " ".join(chunk_summaries)

    final_summary = combined
    if len(chunks) > 1:
        combined_tokens = len(tokenizer.encode(combined, add_special_tokens=False))
        if combined_tokens > MAX_SUMMARY_LEN:
            final_result = summarizer(
                combined,
                max_length=MAX_SUMMARY_LEN + 60,
                min_length=MIN_SUMMARY_LEN,
                do_sample=False,
            )
            final_summary = final_result[0]["summary_text"]

    return {
        "summary": final_summary,
        "num_chunks": len(chunks),
        "chunk_summaries": chunk_summaries,
    }


def summarize_text(text: str) -> dict:
    """
    Returns a structured summary:
    {
        "case_number": "Criminal Appeal No. 415 of 2021" | None,
        "case_type": "Criminal Appeal" | None,
        "court": "IN THE HIGH COURT OF ..." | None,
        "judges": ["SACHIN SHANKAR MAGADUM"],
        "parties": {"appellant": "...", "respondent": "..."},
        "dates_mentioned": ["5th June 2019", ...],
        "area_of_law": {"primary_area": "...", "all_matched_areas": [...]},
        "acts_and_sections": ["Section 25", "Karnataka Societies Registration Act, 1960"],
        "key_points": ["The writ petition is allowed", ...],
        "word_count": 1850,
        "estimated_reading_time_min": 9,
        "narrative_summary": "...",   # the BART-generated prose summary
        "num_chunks": 1,
    }
    """
    text = clean_boilerplate(text)
    metadata = extract_metadata(text)
    narrative = _generate_narrative_summary(text)

    return {
        "case_number": metadata["case_number"],
        "case_type": metadata["case_type"],
        "court": metadata["court"],
        "judges": metadata["judges"],
        "parties": metadata["parties"],
        "dates_mentioned": metadata["dates_mentioned"],
        "area_of_law": metadata["area_of_law"],
        "acts_and_sections": metadata["acts_and_sections"],
        "key_points": metadata["key_points"],
        "word_count": metadata["word_count"],
        "estimated_reading_time_min": metadata["estimated_reading_time_min"],
        "narrative_summary": narrative["summary"],
        "num_chunks": narrative["num_chunks"],
    }