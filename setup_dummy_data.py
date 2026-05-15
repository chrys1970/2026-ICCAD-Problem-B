import os
import csv
import random

# 1. Create the base logs folder
os.makedirs("logs", exist_ok=True)

csv_data = [["Trace Log", "Sim Log", "Regr Log", "Case"]]

# 2. Define 3 distinct "Bugs" to simulate a real regression run
bugs = [
    {
        "type": "mismatch_bug",
        "regr": "Mismatch[1]: ibex[75] : pc[800001d6] c.lui x25,0xd: s9:0000d000\n[FAILED]: 77 matched, 4262 mismatch",
        "sim": "UVM_INFO @ 0: reporter [UVM/RELNOTES]\n--- RISC-V UVM TEST FAILED ---",
        "trace": "140505000\t70152\t800086f6\t903f2723\tsw x3,-1778(x30)\n140509000\t70154\t800086fa\tbfe5\tc.j 800086f2\n" * 20
    },
    {
        "type": "timeout_bug",
        "regr": "riscv_debug_basic_test.0 : [FAILED]",
        "sim": "UVM_FATAL ibex/dv/uvm/tests/core_ibex_test_lib.sv(335) @ 93625000: uvm_test_top [uvm_test_top] No dret detected, or incorrect privilege mode switch in timeout period of 20000 cycles",
        "trace": "93617000\t46708\t80004fac\t018b1693\tslli x13,x22,0x18"
    },
    {
        "type": "memory_corruption_bug",
        "regr": "Mismatch[1]: ibex[102] : pc[800004a2] c.lw x15,0x4(x2)\n[FAILED]: 101 matched, 1 mismatch",
        "sim": "UVM_ERROR @ 55000: reporter [MEM_CORRUPT] Memory read returned X",
        "trace": "55000\t1200\t800004a2\t439c\tc.lw x15,0x4(x2)"
    }
]

# 3. Generate 150 random failure cases
for i in range(1, 151):
    case_dir = f"logs/case_{i}"
    os.makedirs(case_dir, exist_ok=True)
    
    # Randomly assign one of the 3 bugs to this case
    selected_bug = random.choice(bugs)
    
    regr_path = f"{case_dir}/regr.log"
    sim_path = f"{case_dir}/sim.log"
    trace_path = f"{case_dir}/trace.log"
    
    with open(regr_path, "w") as f: f.write(selected_bug["regr"])
    with open(sim_path, "w") as f: f.write(selected_bug["sim"])
    with open(trace_path, "w") as f: f.write(selected_bug["trace"])
    
    csv_data.append([trace_path, sim_path, regr_path, str(i)])

# 4. Write the massive CSV
with open("input.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print("Successfully generated 150 failure cases across 3 hidden bug clusters!")