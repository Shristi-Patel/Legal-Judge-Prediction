"""
Calibrate the temperature of the InLegalBERT judgment model using a labeled validation set.

Usage:
    python calibrate_temperature.py path/to/validation.csv

The CSV needs the columns 'text' and 'label' (0 = rejected, 1 = accepted)
and must not contain judgments used for training.
"""
import csv
import sys

import torch
import torch.nn.functional as F

from judgment_model import document_score


def _collect_scores(csv_path: str):
    csv.field_size_limit(10 ** 9)  # judgments are long
    scores, labels = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            score, _ = document_score(row["text"])
            scores.append(score)
            labels.append(float(int(row["label"])))

    if not scores:
        raise ValueError(f"No rows found in {csv_path} - expected columns 'text,label'.")

    return torch.tensor(scores), torch.tensor(labels)


def fit_temperature(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Finds the T that minimizes NLL of sigmoid(score / T) against the labels."""
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(scores / temperature, labels)
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
    scores, labels = _collect_scores(csv_path)
    print(f"Collected {len(labels)} labeled examples. Fitting temperature ...")

    before_nll = F.binary_cross_entropy_with_logits(scores, labels).item()
    temperature = fit_temperature(scores, labels)
    after_nll = F.binary_cross_entropy_with_logits(scores / temperature, labels).item()
    accuracy = ((scores > 0).float() == labels).float().mean().item()

    print(f"\nAccuracy on this file: {accuracy:.4f}")
    print(f"Fitted temperature: {temperature:.4f}")
    print(f"NLL before calibration: {before_nll:.4f}")
    print(f"NLL after calibration:  {after_nll:.4f}")
    print(f"\nSet it with:  LEGALJ_TEMPERATURE={temperature:.4f} uvicorn main:app --reload")


if __name__ == "__main__":
    main()
