#!/usr/bin/env python3
import os
import csv
import gzip
import argparse

def create_dummy_log(file_path, content_type, num_lines):
    if os.path.exists(file_path):
        print(f" the file has existed: {file_path}")
        return
    
    if content_type == "regr":
        base = ["core_ibex_test_lib.sv : [FAILED]\n", "Mismatch[0]: ibex[0] : pc[8000c1a2]\n", "100 matched, 2 mismatch\n"]
    elif content_type == "sim":
        base = ["UVM_INFO @ 0: reporter [RNTST] Running test...\n", "UVM_ERROR env.sv(45) @ 12: No dret detected\n"]
    else:
        base = ["1000\t50\t8000c1a2\t00000000\taddi\tx1, x1, 1\tx1=1\n"]

    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        written = 0
        while written < num_lines:
            f.writelines(base)
            written += len(base)

def main():
    ap = argparse.ArgumentParser(description="ICCAD 2026 Problem B Dummy Data Generator")
    ap.add_argument("--out_dir", default="set1", help="the place of document")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print(" data generator")
    print("=" * 60)
    print("choose the level of data：")
    print(" [1] small")
    print(" [2] big")
    print(" [3] huge")
    print(" [4] giant")
    choice = input("choose (1-4): ").strip()

    scale_map = {
        "1": {"regr": 1000, "sim": 5000, "trace": 10000, "label": "Small"},
        "2": {"regr": 10000, "sim": 200000, "trace": 1000000, "label": "Medium"},
        "3": {"regr": 50000, "sim": 1000000, "trace": 10000000, "label": "Large_Hidden"},
        "4": {"regr": 100000, "sim": 5000000, "trace": 100000000, "label": "Extreme_Hidden"}
    }

    if choice not in scale_map:
        print("inavaliable input")
        return

    scale = scale_map[choice]
    
    for name in ["test1", "test2"]:
        case_dir = os.path.join(args.out_dir, f"case_{name}")
        os.makedirs(case_dir, exist_ok=True)
        
        print(f"\n[ generating {name.upper()} {scale['label']}  ]")
        create_dummy_log(os.path.join(case_dir, "regr.log"), "regr", scale["regr"])
        create_dummy_log(os.path.join(case_dir, "sim.log.gz"), "sim", scale["sim"])
        create_dummy_log(os.path.join(case_dir, "trace.log.gz"), "trace", scale["trace"])

        input_csv = os.path.join(args.out_dir, f"input_{name}.csv")
        with open(input_csv, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["Case", "Regr Log", "Sim Log", "Trace Log"])
            writer.writerow(["1", f"case_{name}/regr.log", f"case_{name}/sim.log.gz", f"case_{name}/trace.log.gz"])
            
        golden_csv = os.path.join(args.out_dir, f"golden_{name}.csv")
        with open(golden_csv, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["Case", "bucket"])
            writer.writerow(["1", "bucket_0"])
            
    print(f"\n generate sucessfully, the place of data: '{args.out_dir}/' ")

if __name__ == "__main__":
    main()
