"""
Judgment prediction (InLegalBERT) — loads the model you trained on Kaggle
and downloaded into ./models/mvp_inlegalbert_best/
This is a straight port of the inference logic from your training notebook.
"""
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/mvp_inlegalbert_best"
MAX_LEN = 512
HEAD_TOKENS = 128
TAIL_TOKENS = 382

# ---------------------------------------------------------------------------
# Confidence calibration (temperature scaling).
#
# IMPORTANT — this is scaffolding, not a finished calibration. Temperature
# scaling (Guo et al. 2017) divides the logits by a learned scalar T > 1
# before softmax, which spreads out over-confident probabilities without
# changing which class wins. But T can only be *fit* correctly against a
# held-out, labeled validation set (minimize NLL over that set) — there is
# no such labeled set available in this repo, so T defaults to 1.0 (a
# no-op: behavior is unchanged from before).
#
# To actually calibrate this model:
#   1. Get a labeled validation CSV with columns "text,label" (label 0 =
#      rejected, 1 = accepted) that the model has NOT been trained on.
#   2. Run `python calibrate_temperature.py path/to/validation.csv`
#   3. It prints a fitted T — set LEGALJ_TEMPERATURE to that value (env
#      var) when starting the server, e.g.
#      `LEGALJ_TEMPERATURE=2.3 uvicorn main:app`
# ---------------------------------------------------------------------------
TEMPERATURE = float(os.environ.get("LEGALJ_TEMPERATURE", "1.0"))

# Confidence flag threshold: the minimum gap between the two class
# probabilities required to call a prediction "high certainty". Below this,
# it's flagged as low certainty instead. See README for why this matters —
# testing showed the model outputs near-100% confidence even on genuinely
# balanced/ambiguous cases, so this flag is a heuristic signal about how hard
# the model committed to one side, not a calibrated confidence guarantee.
CERTAINTY_MARGIN_THRESHOLD = 0.80

_tokenizer = None
_model = None
_device = None


def _get_device() -> torch.device:
    """Computed lazily (and cached) rather than once at import time, so the
    right device is picked based on CUDA availability at actual load time
    rather than whatever was true when this module was first imported."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_judgment_model():
    """Loads once and caches in module-level globals. Call at app startup."""
    global _tokenizer, _model
    if _model is None:
        device = _get_device()
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
        _model.eval()
        print(f"[judgment_model] loaded on {device}")
    return _tokenizer, _model


def _prepare_input(text, tokenizer):
    device = _get_device()
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    budget = HEAD_TOKENS + TAIL_TOKENS
    if len(ids) > budget:
        ids = ids[:HEAD_TOKENS] + ids[-TAIL_TOKENS:]
    ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
    input_ids = torch.full((1, MAX_LEN), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((1, MAX_LEN), dtype=torch.long)
    input_ids[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    attention_mask[0, :len(ids)] = 1
    return input_ids.to(device), attention_mask.to(device)


def predict_judgment(text: str) -> dict:
    tokenizer, model = load_judgment_model()
    input_ids, attention_mask = _prepare_input(text, tokenizer)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        scaled_logits = logits / TEMPERATURE
        probs = F.softmax(scaled_logits, dim=1).squeeze(0).cpu()

    rejected_prob = probs[0].item()
    accepted_prob = probs[1].item()
    margin = abs(accepted_prob - rejected_prob)

    return {
        "rejected_prob": rejected_prob,
        "accepted_prob": accepted_prob,
        "high_certainty": margin > CERTAINTY_MARGIN_THRESHOLD,
        "temperature": TEMPERATURE,
        # NOTE: "high_certainty" reflects how far apart the two raw softmax
        # probabilities are, NOT genuine calibrated confidence. Testing
        # (see README) showed this model outputs near-100% "confidence"
        # even on deliberately balanced/ambiguous cases, so this flag only
        # tells you the model committed hard to one side — not that it's
        # actually more likely to be right. TEMPERATURE defaults to 1.0
        # (no-op); see the module docstring above for how to fit a real
        # value against a labeled validation set.
    }

