#!/usr/bin/env python3
import os
import csv
import gzip
import argparse
import random

def create_specific_bug_log(file_path, content_type, bug_id):
    if os.path.exists(file_path):
        return

    # 定義 4 種不同的 Bug 形態特徵
    if bug_id == 0:  #PC Mismatch
        regr = ["core_ibex_test_lib.sv : [FAILED]\n", "Mismatch[0]: ibex[0] : pc[8000c1a2]\n", "100 matched, 2 mismatch\n"]
        sim  = ["UVM_INFO @ 0: reporter Running test...\n", "UVM_INFO my_env.sv @ 500: Normal operation\n"]
        trace = ["1000\t50\t8000c1a2\t00000000\taddi\tx1, x1, 1\n"]
    elif bug_id == 1: #Timeout
        regr = ["core_ibex_test_lib.sv : [FAILED]\n", "0 matched, 0 mismatch\n"]
        sim  = ["UVM_FATAL my_driver.sv(89) @ 50000: Simulation timeout expired\n"]
        trace = ["2000\t80\t8000f00c\t00000000\tjal\tx0, 0\n"] # 死迴圈
    elif bug_id == 2: #UVM_ERROR
        regr = ["core_ibex_test_lib.sv : [FAILED]\n", "0 matched, 0 mismatch\n"]
        sim  = ["UVM_ERROR env.sv(45) @ 1200: No dret detected within window\n"]
        trace = ["1500\t60\t8000a3b4\t00000000\tunknown_insn\n"]
    else:            #Stall
        regr = ["core_ibex_test_lib.sv : [FAILED]\n", "0 matched, 0 mismatch\n"]
        sim  = ["UVM_FATAL my_env.sv(22) @ 9999: Environment breakdown\n"]
        trace = ["3000\t90\t80001122\t00000000\tnop\n"]

    if content_type == "regr":
        lines = regr
    elif content_type == "sim":
        lines = sim
    else:
        lines = trace

    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        f.writelines(lines)

def main():
    ap = argparse.ArgumentParser(description="ICCAD 2026 Problem B Multi-Case Data Generator")
    ap.add_argument("--out_dir", default="set1", help="產出測資的目標資料夾")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print("choose scale：")
    print(" [1] small")
    print(" [2] huge")
    size_choice = input("choose (1-2): ").strip()
    
    try:
        num_cases = int(input("\n choose the number of cases: ").strip())
    except ValueError:
        print("invaliable input")
        return

    scale_lines = 100 if size_choice == "1" else 1000000
    random.seed(42)

    for b_name in ["test1", "test2"]:
        print(f"\n generating...")
        
        input_csv_path = os.path.join(args.out_dir, f"input_{b_name}.csv")
        golden_csv_path = os.path.join(args.out_dir, f"golden_{b_name}.csv")
        
        with open(input_csv_path, "w", newline="", encoding="utf-8") as cf, \
             open(golden_csv_path, "w", newline="", encoding="utf-8") as gf:
            
            in_writer = csv.writer(cf)
            gold_writer = csv.writer(gf)
            
            in_writer.writerow(["Case", "Regr Log", "Sim Log", "Trace Log"])
            gold_writer.writerow(["Case", "bucket"])
        
            for c_idx in range(1, num_cases + 1):
                case_name = f"case_{b_name}_c{c_idx}"
                case_dir = os.path.join(args.out_dir, case_name)
                os.makedirs(case_dir, exist_ok=True)
                
                actual_bug_id = random.randint(0, 3)
                
                regr_path = os.path.join(case_dir, "regr.log")
                sim_path = os.path.join(case_dir, "sim.log.gz")
                trace_path = os.path.join(case_dir, "trace.log.gz")
                
                create_specific_bug_log(regr_path, "regr", actual_bug_id)
                create_specific_bug_log(sim_path, "sim", actual_bug_id)
                create_specific_bug_log(trace_path, "trace", actual_bug_id)
                
                in_writer.writerow([str(c_idx), f"{case_name}/regr.log", f"{case_name}/sim.log.gz", f"{case_name}/trace.log.gz"])
                gold_writer.writerow([str(c_idx), f"bucket_{actual_bug_id}"])
                    
    print(f"\n generating finish, the path: '{args.out_dir}/' ")

if __name__ == "__main__":
    main()
