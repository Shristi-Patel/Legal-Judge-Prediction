"""
Fits a temperature-scaling value for the InLegalBERT judgment model against
a labeled validation set, using the standard approach from Guo et al. 2017
("On Calibration of Modern Neural Networks"): find the scalar T > 0 that
minimizes negative log-likelihood of the *held-out* labels when the model's
logits are divided by T before softmax.

This does NOT change which class the model predicts — only how confidently
it reports it. It only helps if the validation set is representative of
real usage and wasn't used in training.

Usage:
    python calibrate_temperature.py path/to/validation.csv

The CSV must have two columns: "text" (the case text) and "label"
(0 = rejected, 1 = accepted). Prints the fitted temperature — set it via
the LEGALJ_TEMPERATURE environment variable when starting the server:

    LEGALJ_TEMPERATURE=2.3 uvicorn main:app
"""
import csv
import sys

import torch
import torch.nn.functional as F

from judgment_model import _prepare_input, load_judgment_model


def _collect_logits(csv_path: str):
    tokenizer, model = load_judgment_model()
    all_logits, all_labels = [], []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text, label = row["text"], int(row["label"])
            input_ids, attention_mask = _prepare_input(text, tokenizer)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            all_logits.append(logits.squeeze(0).cpu())
            all_labels.append(label)

    if not all_logits:
        raise ValueError(f"No rows found in {csv_path} — expected columns 'text,label'.")

    return torch.stack(all_logits), torch.tensor(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Grid + gradient search for the T that minimizes NLL on (logits, labels)."""
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return max(temperature.item(), 1e-3)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]
    print(f"Loading validation set from {csv_path} ...")
    logits, labels = _collect_logits(csv_path)
    print(f"Collected {len(labels)} labeled examples. Fitting temperature ...")

    before_nll = F.cross_entropy(logits, labels).item()
    temperature = fit_temperature(logits, labels)
    after_nll = F.cross_entropy(logits / temperature, labels).item()

    print(f"\nFitted temperature: {temperature:.4f}")
    print(f"NLL before calibration: {before_nll:.4f}")
    print(f"NLL after calibration:  {after_nll:.4f}")
    print(f"\nSet it with:  LEGALJ_TEMPERATURE={temperature:.4f} uvicorn main:app --reload")


if __name__ == "__main__":
    main()
