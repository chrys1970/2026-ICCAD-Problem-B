import argparse
import csv
import os
import numpy as np

def load_labels(path):
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳過表頭 (Case, bucket)
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                labels[row[0].strip()] = row[1].strip()
    return labels

def balanced_accuracy(pred, gold):
    cases = sorted(set(pred) & set(gold), key=lambda c: int(c) if c.isdigit() else c)
    if not cases:
        return 0.0, 0.0, 0.0, (0, 0, 0, 0)
        
    tp = tn = fp = fn = 0
    n = len(cases)
    
    # 窮舉所有 Case 對 (i, j)
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

def main():
    ap = argparse.ArgumentParser(description="ICCAD 2026 Problem B Multi-Benchmark Evaluator")
    ap.add_argument("--pred", help="Path to a single prediction CSV file")
    ap.add_argument("--gold", help="Path to a single golden CSV file")
    
    ap.add_argument("--pred_dir", help="Directory containing your pipeline output CSVs (e.g. bk1.csv, test1.csv)")
    ap.add_argument("--gold_dir", help="Directory containing golden truth CSVs")
    args = ap.parse_args()

    if args.pred and args.gold:
        pred = load_labels(args.pred)
        gold = load_labels(args.gold)
        ba, tpr, tnr, (tp, tn, fp, fn) = balanced_accuracy(pred, gold)
        print(f"=== Single Evaluation ===")
        print(f"Cases Compared: {len(set(pred) & set(gold))}")
        print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"TPR (Sensitivity) = {tpr:.4f} | TNR (Specificity) = {tnr:.4f}")
        print(f"\033[1;32mBalanced Accuracy = {ba:.4f}\033[0m")

    elif args.pred_dir and args.gold_dir:
        print(f"=== ICCAD Batch Evaluation ===")
        print(f"Comparing predictions in '{args.pred_dir}' against golden answers in '{args.gold_dir}'...\n")
        
        scores = []
        for gold_file in sorted(os.listdir(args.gold_dir)):
            if not gold_file.endswith(".csv"):
                continue
                
            gold_path = os.path.join(args.gold_dir, gold_file)
            pred_path = os.path.join(args.pred_dir, gold_file) # 假設預測檔名與正解檔名相同
            

            bench_name = os.path.splitext(gold_file)[0]
            
            if os.path.exists(pred_path):
                pred = load_labels(pred_path)
                gold = load_labels(gold_path)
                ba, _, _, _ = balanced_accuracy(pred, gold)
                print(f"Benchmark [{bench_name:7s}]: Balanced Accuracy = {ba:.4f}")
                scores.append(ba)
            else:
                print(f"Benchmark [{bench_name:7s}]: \033[1;31mMissing Prediction! (Score = 0.0000)\033[0m")
                scores.append(0.0)
                
        if scores:
            final_score = np.mean(scores)
            print("-" * 50)
            print(f"\033[1;33mFinal Averaged Score (ICCAD Metric): {final_score:.4f}\033[0m")

    else:
        ap.print_help()

if __name__ == "__main__":
    main()
