import hashlib

def encode_features_to_array(regr_features, sim_features, trace_features):
    """
    Converts the extracted feature dictionaries into a single flat numerical array.
    """
    vector = []

    # --- 1. Encode Simulation Features (Booleans to 1.0 / 0.0) ---
    vector.append(float(sim_features.get("uvm_fatal", False)))
    vector.append(float(sim_features.get("uvm_error", False)))
    vector.append(float(sim_features.get("is_timeout", False)))

    # --- 2. Encode Trace Features (Floats and Booleans) ---
    vector.append(float(trace_features.get("unique_pc_ratio", 1.0)))
    vector.append(float(trace_features.get("is_stuck_in_loop", False)))

    # --- 3. Encode Regression Features (Counts and Strings) ---
    vector.append(float(regr_features.get("matched_count", 0)))
    vector.append(float(regr_features.get("mismatch_count", 0)))

    # For text strings (like the Program Counter or Test Name), we use a hashing 
    # function to turn the text into a consistent number. If two bugs have the 
    # exact same mismatch PC, they will get the exact same hashed number!
    
    mismatch_pc = regr_features.get("mismatch_pc")
    if mismatch_pc:
        # Convert the hex string into a massive integer, then scale it down safely
        pc_num = int(hashlib.md5(mismatch_pc.encode()).hexdigest()[:8], 16)
        vector.append(float(pc_num))
    else:
        vector.append(-1.0) # -1 means no mismatch happened

    failed_test = regr_features.get("failed_test_name")
    if failed_test:
        test_num = int(hashlib.md5(failed_test.encode()).hexdigest()[:8], 16)
        vector.append(float(test_num))
    else:
        vector.append(-1.0)

    return vector

# --- INDEPENDENT TESTING BLOCK ---
if __name__ == "__main__":
    from feature_extraction import extract_regr_features, extract_sim_features, extract_trace_features
    
    # Let's test it on Case 1 from our generated dummy data
    case_dir = "logs/case_1"
    
    regr = extract_regr_features(f"{case_dir}/regr.log")
    sim = extract_sim_features(f"{case_dir}/sim.log")
    trace = extract_trace_features(f"{case_dir}/trace.log")
    
    # Convert the 3 dictionaries into 1 array!
    numerical_array = encode_features_to_array(regr, sim, trace)
    
    print("\n[ Raw Dictionaries ]")
    print(f"Regr: {regr}")
    print(f"Sim:  {sim}")
    
    print("\n[ Final Machine Learning Array ]")
    print(numerical_array)