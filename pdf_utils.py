import io
import logging

import pdfplumber

logger = logging.getLogger(__name__)

# OCR is optional at runtime: pytesseract needs the `tesseract` binary and
# pdf2image needs `poppler` installed on the host (both are system packages,
# not pip installs). If either is missing we degrade gracefully to the old
# behavior (raise a clear error on scanned PDFs) rather than crashing at
# import time.
try:
    import pytesseract
    from pdf2image import convert_from_bytes

    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

MAX_OCR_PAGES = 40  # sanity cap so a huge scanned PDF can't hang a request for minutes


def _ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    """Rasterize each page and run Tesseract OCR over it. Used only as a
    fallback when a PDF has no extractable text layer (i.e. it's a scanned
    image, not real text)."""
    images = convert_from_bytes(pdf_bytes, dpi=300)
    if len(images) > MAX_OCR_PAGES:
        logger.warning(
            "PDF has %d pages, OCR-ing only the first %d to bound request time.",
            len(images), MAX_OCR_PAGES,
        )
        images = images[:MAX_OCR_PAGES]

    text_parts = []
    for page_image in images:
        page_text = pytesseract.image_to_string(page_image)
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    full_text = "\n".join(text_parts)

    if full_text.strip():
        return full_text

    # No text layer at all -> likely a scanned/image-based PDF. Fall back to
    # OCR if the system dependencies (tesseract + poppler) are available.
    if not _OCR_AVAILABLE:
        raise ValueError(
            "No extractable text found. This PDF is likely scanned/image-based "
            "and needs OCR — install `pytesseract` + `pdf2image` (pip) and the "
            "`tesseract-ocr` + `poppler-utils` system packages to enable it."
        )

    logger.info("No text layer found — falling back to OCR.")
    ocr_text = _ocr_pdf_bytes(pdf_bytes)
    if not ocr_text.strip():
        raise ValueError(
            "No extractable text found, and OCR produced no text either. "
            "The PDF may be blank, corrupted, or too low-quality to read."
        )
    return ocr_text
