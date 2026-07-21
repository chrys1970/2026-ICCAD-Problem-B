import argparse
import csv
import os
import sys
import time
import subprocess
import numpy as np

def load_labels(path):
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
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
    ap.add_argument("--pred_dir", help="Directory containing your pipeline output CSVs")
    ap.add_argument("--gold_dir", help="Directory containing golden truth CSVs")
    
    # 執行模式：直接呼叫主程式跑指定資料夾內的考卷並評分
    ap.add_argument("--run_pipeline", help="Directory containing input_*.csv and golden_*.csv (e.g., set1/)")
    args = ap.parse_args()

    # mode one
    if args.run_pipeline:
        print(f"=== ICCAD End-to-End Pipeline Evaluation ===")
        print(f"Reading benchmarks from '{args.run_pipeline}'...\n")
        
        benchmarks = []
        for file in sorted(os.listdir(args.run_pipeline)):
            if file.startswith("input_") and file.endswith(".csv"):
                name = file.replace("input_", "").replace(".csv", "")
                benchmarks.append(name)

        if not benchmarks:
            print("not found the file")
            return

        scores = []
        for name in benchmarks:
            input_csv = os.path.join(args.run_pipeline, f"input_{name}.csv")
            gold_csv = os.path.join(args.run_pipeline, f"golden_{name}.csv")
            output_csv = os.path.join(args.run_pipeline, f"output_{name}.csv")

            if not os.path.exists(gold_csv):
                continue

            print(f"[ Evaluating {name.upper()} ]")
            
            start_wall = time.perf_counter()
            
            cmd = [sys.executable, "regr_fail_bucketing", "--input", input_csv, "--output", output_csv]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            end_wall = time.perf_counter()
            external_runtime = end_wall - start_wall

            if result.returncode != 0:
                print(f"  ERROR：\n{result.stderr.strip()}")
                scores.append(0.0)
            else:
                pred = load_labels(output_csv)
                gold = load_labels(gold_csv)
                ba, _, _, _ = balanced_accuracy(pred, gold)
                
                internal_time_str = "N/A"
                for line in result.stdout.splitlines():
                    if "Total Execution Time:" in line:
                        internal_time_str = line.split(":")[-1].strip()

                print(f"  sucessful")
                print(f"     -> total time {external_runtime:.4f} 秒")
                print(f"     -> Balanced Accuracy   : \033[1;32m{ba:.4f}\033[0m\n")
                scores.append(ba)

        if scores:
            print(f"\033[1;33mFinal Averaged Score (ICCAD Metric): {np.mean(scores):.4f}\033[0m")

    elif args.pred and args.gold:
        pred = load_labels(args.pred)
        gold = load_labels(args.gold)
        ba, tpr, tnr, (tp, tn, fp, fn) = balanced_accuracy(pred, gold)
        print(f"=== Single Evaluation ===")
        print(f"Balanced Accuracy = {ba:.4f}")

    elif args.pred_dir and args.gold_dir:
        print(f"=== ICCAD Batch Evaluation ===")
        scores = []
        for gold_file in sorted(os.listdir(args.gold_dir)):
            if not gold_file.endswith(".csv"):
                continue
            gold_path = os.path.join(args.gold_dir, gold_file)
            pred_path = os.path.join(args.pred_dir, gold_file)
            bench_name = os.path.splitext(gold_file)[0]
            
            if os.path.exists(pred_path):
                pred = load_labels(pred_path)
                gold = load_labels(gold_path)
                ba, _, _, _ = balanced_accuracy(pred, gold)
                print(f"Benchmark [{bench_name:7s}]: Balanced Accuracy = {ba:.4f}")
                scores.append(ba)
        if scores:
            print(f"Final Averaged Score: {np.mean(scores):.4f}")
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
