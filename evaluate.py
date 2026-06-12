"""Compute pairwise Balanced Accuracy between a prediction CSV and a golden CSV.

Balanced Accuracy = (TPR + TNR) / 2, treating every unordered pair (i, j) as
positive if both cases share the same label, negative otherwise.
"""
import argparse
import csv


def load_labels(path):
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                labels[row[0].strip()] = row[1].strip()
    return labels


def balanced_accuracy(pred, gold):
    cases = sorted(set(pred) & set(gold), key=lambda c: int(c) if c.isdigit() else c)
    tp = tn = fp = fn = 0
    n = len(cases)
    for a in range(n):
        for b in range(a + 1, n):
            i, j = cases[a], cases[b]
            same_gold = gold[i] == gold[j]
            same_pred = pred[i] == pred[j]
            if same_gold and same_pred:
                tp += 1
            elif same_gold and not same_pred:
                fn += 1
            elif not same_gold and not same_pred:
                tn += 1
            else:
                fp += 1
    tpr = tp / (tp + fn) if (tp + fn) else 1.0
    tnr = tn / (tn + fp) if (tn + fp) else 1.0
    return (tpr + tnr) / 2, tpr, tnr, (tp, tn, fp, fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    args = ap.parse_args()

    pred = load_labels(args.pred)
    gold = load_labels(args.gold)
    ba, tpr, tnr, (tp, tn, fp, fn) = balanced_accuracy(pred, gold)
    print(f"Cases compared: {len(set(pred) & set(gold))}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"TPR={tpr:.4f}  TNR={tnr:.4f}")
    print(f"Balanced Accuracy = {ba:.4f}")
