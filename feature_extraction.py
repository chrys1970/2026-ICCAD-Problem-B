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

    except FileNotFoundError:
        print(f"FAILED : non-founding the file {file_path}")
    except Exception as e:
        print(f"ERROR within seeking data：{e}")

    return features

# --- 測試執行 ---
# result = extract_regr_features("regr.log")
# print(result)
