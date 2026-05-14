import re

def extract_regr_features(file_path):
# define the variable
    features = {
        "is_mismatch": False,      
        "mismatch_index": None,    
        "rtl_pc": "None",          
        "expected_pc": "None",     
        "matched_count": 0,      
        "mismatch_count": 0,
        "verdict_failed": False
    }

    try:
        with open(file_path, 'r') as f:
            content = f.read()

            if "[FAILED]" in content:
                features["verdict_failed"] = True
              
            mismatch_pattern = r"Mismatch \[(\d+)\].*?ibex \[\d+\]: pc \[([0-9a-fA-F]+)\].*?spike \[\d+\]: pc\[([0-9a-fA-F]+)\]"
            match = re.search(mismatch_pattern, content, re.DOTALL)
            
            if match:
                features["is_mismatch"] = True
                features["mismatch_index"] = int(match.group(1)) 
                features["rtl_pc"] = match.group(2)             
                features["expected_pc"] = match.group(3)
            
            stat_pattern = r"(\d+) matched, (\d+) mismatch"
            stats = re.search(stat_pattern, content)
            if stats:
                features["matched_count"] = int(stats.group(1))
                features["mismatch_count"] = int(stats.group(2))

#sim.log feature
# UVM_<LEVEL> <FILE>(<LINE>) @ <TIME>: <ID> [<INSTANCE>]
# UVM_LEVEL:
# INFO - process tracking
# WARNING - might have problems
# ERROR - it usually plus the error number while uncertainly immediately stop
# FATAL - stop immediately, usually is time-out or protocol error.

def extract_sim_features(file_path, features):
    features.update({
        "max_uvm_level": "UVM_INFO", 
        "uvm_fatal_msg": "None",       
        "uvm_error_msg": "None",      
        "is_timeout": False,          
        "uvm_component": "None"       
    })

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # check time-out
                if "timeout" in line.lower():
                    features["is_timeout"] = True # 

                # catch the level
                if "UVM_FATAL" in line:
                    features["max_uvm_level"] = "UVM_FATAL"
                    msg_match = re.search(r"UVM_FATAL.*?@\s*\d+:\s*(.*?)\s*\[(.*?)\]\s*(.*)", line)
                    if msg_match:
                        features["uvm_component"] = msg_match.group(1)
                        features["uvm_fatal_msg"] = msg_match.group(3)
                    else:
                        features["uvm_fatal_msg"] = line.strip()

                # 3. catch UVM_ERROR
                elif "UVM_ERROR" in line and features["max_uvm_level"] != "UVM_FATAL":
                    features["max_uvm_level"] = "UVM_ERROR"
                    msg_match = re.search(r"UVM_ERROR.*?@\s*\d+:\s*(.*?)\s*\[(.*?)\]\s*(.*)", line)
                    if msg_match:
                        features["uvm_error_msg"] = msg_match.group(3)

                # 4. catch UVM_WARNING
                elif "UVM_WARNING" in line and features["max_uvm_level"] not in ["UVM_FATAL", "UVM_ERROR"]:
                    features["max_uvm_level"] = "UVM_WARNING"

    except FileNotFoundError:
    print(f"FAILED : non-founding the file {file_path}")
    except Exception as e:
    print(f"ERROR within seeking data：{e}")

    return features


# --- 測試執行 ---
# result = extract_regr_features("regr.log")
# print(result)
