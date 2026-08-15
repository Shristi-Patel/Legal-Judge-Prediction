# Legal-J

FastAPI backend around your InLegalBERT judgment-prediction model, plus a
new PDF summarization feature.

## 1. Get your trained model in place

1. In your Kaggle notebook, add this cell at the end and re-run just that cell
   (no need to retrain — the model is already saved to disk from training):
   ```python
   import shutil
   shutil.make_archive("/kaggle/working/mvp_inlegalbert_best", "zip", "/kaggle/working/mvp_inlegalbert_best")
   ```
2. On the notebook page, open the **Output** tab (right sidebar) and download
   `mvp_inlegalbert_best.zip`.
3. Unzip it so the folder structure looks like this:
   ```
   legal-j-backend/
     models/
       mvp_inlegalbert_best/
         config.json
         model.safetensors  (or pytorch_model.bin)
         tokenizer_config.json
         vocab.txt / tokenizer.json
         ...
   ```
Note: Model files (e.g. `model.safetensors`) are excluded from this repository due to GitHub’s file size limits. 
Please download `mvp_inlegalbert_best.zip` from Kaggle (see steps below) and unzip it into `models/` before running the backend.

## 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want CUDA acceleration, install the matching
`torch` build from https://pytorch.org/get-started/locally/ instead of the
plain `torch` in requirements.txt.

## 3. Run the server

```bash
uvicorn main:app --reload
```

Server runs at `http://127.0.0.1:8000`. Interactive docs (auto-generated,
good for testing without a frontend): `http://127.0.0.1:8000/docs`

First request will be slow — that's the summarization model
(`facebook/bart-large-cnn`, ~1.6GB) downloading once and caching to
`~/.cache/huggingface`. Every request after that, on any future run, loads
instantly from cache — no re-download.

## 4. Endpoints

- `GET  /health` — sanity check (never requires auth)
- `POST /predict-judgment` — upload a case PDF, get accepted/rejected probabilities
- `POST /summarize` — upload a legal PDF, get a detailed structured summary
  (case number, case type, court, judges, parties, dates, area of law, acts/
  sections cited, key points, word count, reading time, narrative summary)
- `POST /analyze` — both of the above in one call

Example with curl:
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F "file=@/path/to/case.pdf"
```

## 5. Frontend (Streamlit)

```bash
streamlit run app.py
```

Supports uploading multiple PDFs at once, keeps an in-session history you
can switch between, shows a page-by-page PDF preview, and lets you export
each analysis as JSON or a formatted Word (`.docx`) report. The PDF preview
needs `pdf2image` (pip, already in requirements.txt) + the `poppler` system
package (see OCR section below — same dependency).

## 6. OCR for scanned PDFs

If a PDF has no text layer (a scanned image rather than real text),
`pdf_utils.py` now falls back to OCR automatically instead of just failing.
It needs two system packages in addition to the `pytesseract`/`pdf2image`
pip packages already in `requirements.txt`:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils
# macOS
brew install tesseract poppler
# Windows: install Tesseract-OCR and Poppler for Windows, add both to PATH
```

If those system packages aren't installed, OCR is skipped and you get the
original clear error message instead of a crash.

## 7. Confidence calibration (optional)

`high_certainty` and the raw accepted/rejected percentages are **not**
calibrated probabilities — see the note in `judgment_model.py`. A
`LEGALJ_TEMPERATURE` env var (default `1.0`, i.e. no change) lets you apply
temperature scaling. To fit a real value you need a labeled validation CSV
(`text,label` columns, `label` 0=rejected/1=accepted) that wasn't used in
training:

```bash
python calibrate_temperature.py path/to/validation.csv
# prints a fitted temperature, e.g.:
LEGALJ_TEMPERATURE=2.3 uvicorn main:app --reload
```

Without a labeled validation set this stays at the default 1.0 (no-op).

## Notes / things you'll likely want to tune

- **`/analyze` loads two models into memory** (InLegalBERT + BART-large-CNN,
  roughly 1.5–2GB combined). Fine on a laptop with 8GB+ RAM; if you deploy to
  a small server, watch memory.
- **CORS** defaults to `*` (open) for easy local frontend dev. Restrict it
  before deploying publicly:
  ```bash
  ALLOWED_ORIGINS="https://your-frontend.example.com" uvicorn main:app
  ```
- **Auth** is off by default. Set both env vars to require HTTP Basic auth
  on every endpoint except `/health`:
  ```bash
  LEGALJ_AUTH_USER=admin LEGALJ_AUTH_PASS=change-me uvicorn main:app
  ```
  This is a minimal deterrent, not a full auth system — put a real reverse
  proxy / auth layer in front of it for anything beyond casual protection.
- **Upload size** is capped at 25MB by default; change with
  `MAX_UPLOAD_MB=50 uvicorn main:app`.
- **Summarization model choice**: `facebook/bart-large-cnn` is a solid
  general-purpose default and needs no fine-tuning. If summary quality on
  legal text specifically isn't good enough, a legal-domain summarizer like
  `nsi319/legal-pegasus` is a drop-in swap — just change `SUMMARY_MODEL_NAME`
  in `summarizer.py`.
- **Area-of-law / entity extraction** is still deliberately rule-based
  (regex + keyword buckets in `case_metadata.py`) rather than a legal-NER
  model — a prior attempt at a spaCy-based NER model hit an unresolved
  dependency conflict with modern Python (see the note at the bottom of
  `case_metadata.py`). This round widened the keyword buckets instead
  (banking, societies, writ/constitutional, family, labour, land
  acquisition, arbitration, service, bail). A transformers-based (non-spaCy)
  legal NER model is the more promising next step if the keyword approach
  starts missing too much.
