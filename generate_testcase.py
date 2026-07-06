#!/usr/bin/env python3
import os
import csv
import gzip
import argparse
import random

def create_specific_bug_log(file_path, content_type, bug_id):
    """根據不同的 Bug ID，生成截然不同的錯誤特徵，讓演算法有辦法分類"""
    if os.path.exists(file_path):
        return

    # 定義 4 種不同的 Bug 形態特徵
    if bug_id == 0:  # 形態 A: 記憶體 PC Mismatch
        regr = ["core_ibex_test_lib.sv : [FAILED]\n", "Mismatch[0]: ibex[0] : pc[8000c1a2]\n", "100 matched, 2 mismatch\n"]
        sim  = ["UVM_INFO @ 0: reporter Running test...\n", "UVM_INFO my_env.sv @ 500: Normal operation\n"]
        trace = ["1000\t50\t8000c1a2\t00000000\taddi\tx1, x1, 1\n"]
    elif bug_id == 1: # 形態 B: UVM 驗證超時 (Timeout)
        regr = ["core_ibex_test_lib.sv : [FAILED]\n", "0 matched, 0 mismatch\n"]
        sim  = ["UVM_FATAL my_driver.sv(89) @ 50000: Simulation timeout expired\n"]
        trace = ["2000\t80\t8000f00c\t00000000\tjal\tx0, 0\n"] # 死迴圈
    elif bug_id == 2: # 形態 C: 指令不支援造成的 UVM_ERROR
        regr = ["core_ibex_test_lib.sv : [FAILED]\n", "0 matched, 0 mismatch\n"]
        sim  = ["UVM_ERROR env.sv(45) @ 1200: No dret detected within window\n"]
        trace = ["1500\t60\t8000a3b4\t00000000\tunknown_insn\n"]
    else:            # 形態 D: 隨機產生的硬體未知掛起 (Stall)
        regr = ["core_ibex_test_lib.sv : [FAILED]\n", "0 matched, 0 mismatch\n"]
        sim  = ["UVM_FATAL my_env.sv(22) @ 9999: Environment breakdown\n"]
        trace = ["3000\t90\t80001122\t00000000\tnop\n"]

    if content_type == "regr":
        lines = regr
    elif content_type == "sim":
        lines = sim
    else:
        lines = trace

    # 模擬巨量行數 (複寫填滿)
    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        f.writelines(lines)

def main():
    ap = argparse.ArgumentParser(description="ICCAD 2026 Problem B Multi-Case Data Generator")
    ap.add_argument("--out_dir", default="set1", help="產出測資的目標資料夾")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print(" 🛠️  ICCAD Problem B 升級版：多 Case 隨機考卷生成器")
    print("=" * 60)
    
    # 1. 讓使用者決定考卷要塞多少行 (壓測規模)
    print("請選擇單一 Log 檔案的行數規模（壓測強度）：")
    print(" [1] 輕量級 (每個檔案 100 行，純看分類分數)")
    print(" [2] 大魔王級 (每個檔案 100 萬行，壓測 deque 效能)")
    size_choice = input("請選擇規模 (1-2): ").strip()
    
    # 2. 讓使用者決定要生出多少個 Case (幾十到上百列)
    try:
        num_cases = int(input("\n請輸入您想在這個 Set 裡面建立多少個 Case (例如 50 或 100): ").strip())
    except ValueError:
        print("❌ 輸入必須是數字！")
        return

    scale_lines = 100 if size_choice == "1" else 1000000
    random.seed(42) # 固定隨機，確保每次生出來的考卷題目一致

    # 本地測試建立 test1 與 test2 兩套不同的綜合大考卷
    for b_name in ["test1", "test2"]:
        print(f"\n🚀 正在建構 {b_name.upper()} 綜合考卷（內含 {num_cases} 個 Case）...")
        
        input_csv_path = os.path.join(args.out_dir, f"input_{b_name}.csv")
        golden_csv_path = os.path.join(args.out_dir, f"golden_{b_name}.csv")
        
        with open(input_csv_path, "w", newline="", encoding="utf-8") as cf, \
             open(golden_csv_path, "w", newline="", encoding="utf-8") as gf:
            
            in_writer = csv.writer(cf)
            gold_writer = csv.writer(gf)
            
            # 寫入大會標準表頭
            in_writer.writerow(["Case", "Regr Log", "Sim Log", "Trace Log"])
            gold_writer.writerow(["Case", "bucket"])
            
            # 開始瘋狂生出幾十、上百個 Case 的資料夾與日誌
            for c_idx in range(1, num_cases + 1):
                case_name = f"case_{b_name}_c{c_idx}"
                case_dir = os.path.join(args.out_dir, case_name)
                os.makedirs(case_dir, exist_ok=True)
                
                # 隨機分配一個 Bug ID (0~3)，這就是這題的「正解答案」
                actual_bug_id = random.randint(0, 3)
                
                # 在各自資料夾下建立獨立的三類 log
                regr_path = os.path.join(case_dir, "regr.log")
                sim_path = os.path.join(case_dir, "sim.log.gz")
                trace_path = os.path.join(case_dir, "trace.log.gz")
                
                create_specific_bug_log(regr_path, "regr", actual_bug_id)
                create_specific_bug_log(sim_path, "sim", actual_bug_id)
                create_specific_bug_log(trace_path, "trace", actual_bug_id)
                
                # 將相對路徑寫入 input.csv
                in_writer.writerow([str(c_idx), f"{case_name}/regr.log", f"{case_name}/sim.log.gz", f"{case_name}/trace.log.gz"])
                # 將正確分群寫入 golden.csv
                gold_writer.writerow([str(c_idx), f"bucket_{actual_bug_id}"])
                
                if c_idx % 10 == 0 or c_idx == num_cases:
                    print(f"   -> 已完成 {c_idx} / {num_cases} 個 Case 資料建構...")
                    
    print(f"\n🎉 高規格綜合考卷生成完畢！檔案已完整保存在 '{args.out_dir}/' 中。")

if __name__ == "__main__":
    main()
