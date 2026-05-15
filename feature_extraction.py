import re
import os

def extract_regr_features(file_path):
    """
    Extracts mismatch PCs, failed test names, and match/mismatch statistics using purely Regex.
    """
    features = {
        "mismatch_pc": None,
        "failed_test_name": None,
        "matched_count": 0,
        "mismatch_count": 0
    }
    
    if not os.path.exists(file_path): # safety check
        return features

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 1. Isolate the exact point of divergence (the Program Counter)
            # look for digits + ibex + pc[hex] pattern; capture group
            mismatch_match = re.search(r"Mismatch\[\d+\]:\s*ibex\[\d+\]\s*:\s*pc\[([0-9a-fA-F]+)\]", line)
            if mismatch_match and not features["mismatch_pc"]:
                features["mismatch_pc"] = mismatch_match.group(1)  # save first capture group
            
            # 2. Capture the testbench verdict
            failed_match = re.search(r"^([a-zA-Z0-9_.]+)\s*:\s*\[FAILED\]", line)
            if failed_match:
                features["failed_test_name"] = failed_match.group(1)

            # 3. Capture the final match/mismatch statistics
            stat_pattern = r"(\d+)\s+matched,\s+(\d+)\s+mismatch"
            stats = re.search(stat_pattern, line)
            if stats:
                features["matched_count"] = int(stats.group(1))
                features["mismatch_count"] = int(stats.group(2)) # type cast to int for future math

    return features # fully filled out dictionary with all 4 keys, some may be None or 0


def extract_sim_features(file_path):
    """
    Scans for UVM report macros and timeouts in the VCS simulator console log using Regex.
    """
    features = {
        "uvm_fatal": False,
        "uvm_error": False,
        "is_timeout": False
    }
    
    if not os.path.exists(file_path):
        return features

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if "UVM_ERROR" in line:
                features["uvm_error"] = True
            
            if "UVM_FATAL" in line:
                features["uvm_fatal"] = True
                if "timeout" in line.lower():
                    features["is_timeout"] = True

    return features


def extract_trace_features(file_path):
    """
    Analyzes the tail lines of the CPU trace to detect repeating loopy patterns.
    """
    features = {
        "unique_pc_ratio": 1.0, # default to 1.0 (all unique) and will be reduced if we find repeats
        "is_stuck_in_loop": False
    }
    
    if not os.path.exists(file_path):
        return features

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Focus only on the tail behavior to save time
            tail_lines = lines[-100:] if len(lines) > 100 else lines # analyze last 100 lines or fewer if file is small
            
            pcs = []
            # extract the Program Counter from each line using regex; we assume it's the 3rd column in the trace
            for line in tail_lines:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    pc = parts[2] 
                    if pc.isalnum(): 
                        pcs.append(pc)
            
            if pcs:
                unique_pcs = set(pcs)
                ratio = len(unique_pcs) / len(pcs)
                features["unique_pc_ratio"] = round(ratio, 3)
                
                # Flag if the CPU is stuck executing the same few instructions
                if ratio < 0.2:
                    features["is_stuck_in_loop"] = True
                    
    except Exception:
        pass 

    return features

"""
# --- INDEPENDENT TESTING BLOCK ---
if __name__ == "__main__":
    print("--- Testing Feature Extraction (Regex Only) ---")
    
    # Note: Using the folder names with underscores to match the large dummy generator
    case1_dir = "logs/case_1"
    case2_dir = "logs/case_2"
    
    print(f"\n[ Scanning {case1_dir} ]")
    print("Regr Features: ", extract_regr_features(f"{case1_dir}/regr.log"))
    print("Sim Features:  ", extract_sim_features(f"{case1_dir}/sim.log"))
    print("Trace Features:", extract_trace_features(f"{case1_dir}/trace.log"))

    print(f"\n[ Scanning {case2_dir} ]")
    print("Regr Features: ", extract_regr_features(f"{case2_dir}/regr.log"))
    print("Sim Features:  ", extract_sim_features(f"{case2_dir}/sim.log"))
    print("Trace Features:", extract_trace_features(f"{case2_dir}/trace.log"))
"""