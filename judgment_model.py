"""
Judgment prediction (InLegalBERT) - loads the model trained on Kaggle and
placed in ./models/mvp_inlegalbert_best/

The model is trained on overlapping 512-token windows counted backwards
from the END of a judgment. At prediction time the same windows are cut
from the document, each is scored, and the scores of the last few windows
are averaged. How many windows to average and the decision threshold were
chosen on the dev set during training and are read from
inference_config.json inside the model folder.

If that file is missing, the folder holds the old model (one head+tail
snippet per document) and the old behaviour is used, so the app keeps
working until the new model is copied in.
"""
import json
import math
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/mvp_inlegalbert_best"
MAX_LEN = 512
BATCH_SIZE = 8  # windows scored per forward pass


HEAD_TOKENS = 128
TAIL_TOKENS = 382

# Confidence softening. 1.0 = no change; fit it with calibrate_temperature.py
TEMPERATURE = float(os.environ.get("LEGALJ_TEMPERATURE", "1.0"))

CERTAINTY_MARGIN_THRESHOLD = 0.80

_tokenizer = None
_model = None
_device = None
_settings = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _load_settings() -> dict:
    path = os.path.join(MODEL_DIR, "inference_config.json")
    if not os.path.isfile(path):
        print("[judgment_model] no inference_config.json - using the old head+tail input")
        return {"legacy": True, "threshold": 0.5}
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    max_len = int(cfg.get("max_len", MAX_LEN))
    windows = int(cfg["windows_to_average"])
    return {
        "legacy": False,
        "content": max_len - 2,                      # tokens per window without [CLS], [SEP]
        "stride": int(cfg["stride"]),
        "max_windows": int(cfg.get("max_windows", windows)),
        "windows_to_average": windows,
        "threshold": float(cfg["threshold"]),
    }


def load_judgment_model():
    """Loads once and caches in module-level globals. Call at app startup."""
    global _tokenizer, _model, _settings
    if _model is None:
        device = _get_device()
        _settings = _load_settings()
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
        _model.eval()
        print(f"[judgment_model] loaded on {device} | settings: {_settings}")
    return _tokenizer, _model


def _make_windows(n_tokens: int, content: int, stride: int, max_windows: int) -> list:
    """(start, end) token windows anchored at the END of the document.
    Must stay identical to make_windows() in the training script."""
    if n_tokens <= content:
        return [(0, n_tokens)]
    windows = []
    end = n_tokens
    while len(windows) < max_windows:
        if end <= content:
            if windows[-1][0] >= 64:
                windows.append((0, content))
            break
        windows.append((end - content, end))
        end -= stride
    return windows


def _token_pieces(ids: list) -> list:
    """The token-id pieces of the document that the model will score."""
    if _settings["legacy"]:
        if len(ids) > HEAD_TOKENS + TAIL_TOKENS:
            ids = ids[:HEAD_TOKENS] + ids[-TAIL_TOKENS:]
        return [ids]
    windows = _make_windows(len(ids), _settings["content"], _settings["stride"], _settings["max_windows"])
    return [ids[s:e] for s, e in windows[:_settings["windows_to_average"]]]


def document_score(text: str) -> tuple:
   
    tokenizer, model = load_judgment_model()
    device = _get_device()
    ids = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    pieces = _token_pieces(ids)

    input_ids = torch.full((len(pieces), MAX_LEN), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(pieces), MAX_LEN), dtype=torch.long)
    for row, piece in enumerate(pieces):
        seq = [tokenizer.cls_token_id] + list(piece) + [tokenizer.sep_token_id]
        input_ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, :len(seq)] = 1

    diffs = []
    with torch.no_grad():
        for i in range(0, len(pieces), BATCH_SIZE):
            logits = model(
                input_ids=input_ids[i:i + BATCH_SIZE].to(device),
                attention_mask=attention_mask[i:i + BATCH_SIZE].to(device),
            ).logits.float()
            diffs.append((logits[:, 1] - logits[:, 0]).cpu())
    mean_diff = torch.cat(diffs).mean().item()

    threshold = _settings["threshold"]
    boundary = math.log(threshold / (1.0 - threshold))  # 0 when the threshold is 0.5
    return mean_diff - boundary, len(pieces)


def predict_judgment(text: str) -> dict:
    score, windows_used = document_score(text)
    accepted_prob = 1.0 / (1.0 + math.exp(-score / TEMPERATURE))
    rejected_prob = 1.0 - accepted_prob
    margin = abs(accepted_prob - rejected_prob)

    return {
        "rejected_prob": rejected_prob,
        "accepted_prob": accepted_prob,
        "high_certainty": margin > CERTAINTY_MARGIN_THRESHOLD,
        "temperature": TEMPERATURE,
        "prediction": "accepted" if accepted_prob >= 0.5 else "rejected",
        "windows_used": windows_used,
    }
