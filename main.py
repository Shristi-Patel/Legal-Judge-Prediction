import os
import secrets

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from pdf_utils import extract_text_from_pdf_bytes
from judgment_model import load_judgment_model, predict_judgment
from summarizer import load_summarizer, summarize_text

app = FastAPI(title="Legal-J API", version="1.1.0")

# ---------------------------------------------------------------------------
# CORS. Defaults to "*" so local frontend dev (Streamlit on a different port)
# works out of the box. Before deploying publicly, set ALLOWED_ORIGINS to a
# comma-separated allowlist, e.g.:
#   ALLOWED_ORIGINS="https://your-frontend.example.com" uvicorn main:app
# ---------------------------------------------------------------------------
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional HTTP Basic Auth. Off by default (nothing to configure for local
# dev). Set both env vars to require a username/password on every request:
#   LEGALJ_AUTH_USER=admin LEGALJ_AUTH_PASS=change-me uvicorn main:app
# This is a minimal deterrent, not a full auth system — put a real reverse
# proxy / auth layer in front for anything beyond casual protection.
# ---------------------------------------------------------------------------
_AUTH_USER = os.environ.get("LEGALJ_AUTH_USER")
_AUTH_PASS = os.environ.get("LEGALJ_AUTH_PASS")
_AUTH_ENABLED = bool(_AUTH_USER and _AUTH_PASS)
_security = HTTPBasic(auto_error=False)


def require_auth(credentials: HTTPBasicCredentials = Depends(_security)):
    if not _AUTH_ENABLED:
        return
    valid = bool(credentials) and secrets.compare_digest(
        credentials.username, _AUTH_USER
    ) and secrets.compare_digest(credentials.password, _AUTH_PASS)
    if not valid:
        raise HTTPException(
            status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"}
        )


# ---------------------------------------------------------------------------
# Request size limit. Rejects oversized uploads before they're read into
# memory (a request with no Content-Length, e.g. chunked transfer, is let
# through this check and still bounded downstream by MAX_OCR_PAGES / model
# truncation, but this catches the common case cheaply).
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "25"))


class MaxUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_UPLOAD_MB * 1024 * 1024:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"File too large — limit is {MAX_UPLOAD_MB:.0f} MB."},
                    )
            except ValueError:
                pass
        return await call_next(request)


app.add_middleware(MaxUploadSizeMiddleware)


@app.on_event("startup")
def startup():
    load_judgment_model()
    load_summarizer()


@app.get("/health")
def health():
    return {"status": "ok"}


async def _read_pdf_upload(file: UploadFile) -> str:
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    pdf_bytes = await file.read()
    try:
        return extract_text_from_pdf_bytes(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict-judgment", dependencies=[Depends(require_auth)])
async def predict_judgment_endpoint(file: UploadFile = File(...)):
    text = await _read_pdf_upload(file)
    result = predict_judgment(text)
    return {
        "filename": file.filename,
        "accepted_prob": round(result["accepted_prob"], 4),
        "rejected_prob": round(result["rejected_prob"], 4),
        "high_certainty": result["high_certainty"],
        "temperature": result["temperature"],
        "confidence_note": (
            "This reflects how strongly the model leaned one way, not a "
            "calibrated probability of correctness — the model can output "
            "extreme confidence even on genuinely ambiguous cases. Set "
            "LEGALJ_TEMPERATURE after fitting it with calibrate_temperature.py "
            "to soften this."
        ),
    }


@app.post("/summarize", dependencies=[Depends(require_auth)])
async def summarize_endpoint(file: UploadFile = File(...)):
    """Upload a legal PDF -> get a structured summary: case number, court,
    parties, dates, area of law, and a narrative summary."""
    text = await _read_pdf_upload(file)
    result = summarize_text(text)
    return {
        "filename": file.filename,
        **result,
    }


@app.post("/analyze", dependencies=[Depends(require_auth)])
async def analyze_endpoint(file: UploadFile = File(...)):
    """Upload a case PDF -> get the judgment prediction AND the structured
    summary in a single call."""
    text = await _read_pdf_upload(file)
    judgment = predict_judgment(text)
    summary = summarize_text(text)
    return {
        "filename": file.filename,
        "accepted_prob": round(judgment["accepted_prob"], 4),
        "rejected_prob": round(judgment["rejected_prob"], 4),
        "high_certainty": judgment["high_certainty"],
        "temperature": judgment["temperature"],
        "confidence_note": (
            "This reflects how strongly the model leaned one way, not a "
            "calibrated probability of correctness — the model can output "
            "extreme confidence even on genuinely ambiguous cases. Set "
            "LEGALJ_TEMPERATURE after fitting it with calibrate_temperature.py "
            "to soften this."
        ),
        **summary,
    }