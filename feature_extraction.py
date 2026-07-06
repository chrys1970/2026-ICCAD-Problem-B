import re
import os
import gzip
from collections import deque

def _open_log(file_path):
    """Open a log file transparently, supporting both plain and gzip (.gz) files."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None
    if file_path.endswith(".gz"):
        return gzip.open(file_path, "rt", encoding="utf-8", errors="replace")
    return open(file_path, "r", encoding="utf-8", errors="replace")

def extract_regr_features(file_path):
    """
    Extracts mismatch PCs, failed test names, and match/mismatch statistics using purely Regex.
    Safe against empty or corrupted files.
    """
    features = {
        "mismatch_pc": None,
        "failed_test_name": None,
        "matched_count": 0,
        "mismatch_count": 0
    }
    
    f = _open_log(file_path)
    if f is None:
        return features

    with f:
        for line in f:
            # 1. Isolate the exact point of divergence (the Program Counter)
            mismatch_match = re.search(r"Mismatch\[\d+\]:\s*ibex\[\d+\]\s*:\s*pc\[([0-9a-fA-F]+)\]", line)
            if mismatch_match and not features["mismatch_pc"]:
                features["mismatch_pc"] = mismatch_match.group(1)
            
            # 2. Capture the testbench verdict
            failed_match = re.search(r"^([a-zA-Z0-9_.]+)\s*:\s*\[FAILED\]", line)
            if failed_match:
                features["failed_test_name"] = failed_match.group(1)

            # 3. Capture the final match/mismatch statistics
            stat_pattern = r"(\d+)\s+matched,\s+(\d+)\s+mismatch"
            stats = re.search(stat_pattern, line)
            if stats:
                features["matched_count"] = int(stats.group(1))
                features["mismatch_count"] = int(stats.group(2))

    return features


def extract_sim_features(file_path):
    """
    Scans for UVM report macros and timeouts in the VCS simulator console log using Regex.
    Safe against empty or corrupted files.
    """
    features = {
        "uvm_fatal": False,
        "uvm_error": False,
        "is_timeout": False
    }

    f = _open_log(file_path)
    if f is None: 
        return features

    macro_re = re.compile(r"\bUVM_(FATAL|ERROR)\b\s+\S+\(\d+\)\s*@")
    summary_re = re.compile(r"^\s*UVM_(FATAL|ERROR)\s*:\s*(\d+)\s*$")

    with f:
        for line in f:
            m = macro_re.search(line)
            if m:
                if m.group(1) == "FATAL":
                    features["uvm_fatal"] = True
                    if "timeout" in line.lower():
                        features["is_timeout"] = True
                else:
                    features["uvm_error"] = True
                continue

            s = summary_re.match(line)
            if s:
                count = int(s.group(2))
                if count > 0:
                    if s.group(1) == "FATAL":
                        features["uvm_fatal"] = True
                    else:
                        features["uvm_error"] = True

    return features


def extract_trace_features(file_path):
    """
    Analyzes the tail lines of the CPU trace to detect repeating loopy patterns.
    OPTIMIZED: Completely avoids f.readlines() to prevent OOM on 100M-line logs.
    """
    features = {
        "unique_pc_ratio": 1.0, 
        "is_stuck_in_loop": False
    }
    
    f = _open_log(file_path)
    if f is None:
        return features

    try:
        tail_lines = deque(maxlen=100)
        with f:
            for line in f:
                tail_lines.append(line)
            
        pcs = []
        for line in tail_lines:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pc = parts[2] 
                if pc.isalnum(): 
                    pcs.append(pc)
            
        if pcs:
            unique_pcs = set(pcs)
            ratio = len(unique_pcs) / len(pcs) if len(pcs) > 0 else 1.0
            features["unique_pc_ratio"] = round(ratio, 3)
            
            if ratio < 0.2:
                features["is_stuck_in_loop"] = True
                    
    except Exception:
        pass 

    return features


_HEX_RE = re.compile(r"\b(?:0x)?[0-9a-fA-F]{4,}\b")
_NUM_RE = re.compile(r"\b\d+\b")
_TRACE_TAIL_LINES = 200      
_SIM_MAX_LINES = 2_000_000   


def _normalize(text):
    """Replace addresses and bare numbers with placeholders to reduce noise."""
    text = _HEX_RE.sub(" ADDR ", text)
    text = _NUM_RE.sub(" N ", text)
    return text


def _is_noise_token(tok):
    """Drop file paths, source files and placeholder tokens — keep real words."""
    if not tok or tok in ("N", "ADDR"):
        return True
    if "/" in tok or tok.endswith(".sv") or tok.endswith(".svh"):
        return True
    return False


def _sim_tokens(file_path):
    """Extract the human-readable reason from UVM_FATAL/ERROR/WARNING lines."""
    f = _open_log(file_path)
    if f is None:
        return ""
        
    words = []
    try:
        with f:
            for i, line in enumerate(f):
                if i > _SIM_MAX_LINES: 
                    break
                if ("UVM_FATAL" in line or "UVM_ERROR" in line
                        or "UVM_WARNING" in line or "TEST FAILED" in line):
                    for tok in _normalize(line).split():
                        if not _is_noise_token(tok):
                            words.append(tok)
    except Exception:
        return ""
    return " ".join(words)


def _trace_tokens(file_path):
    """Summarize the trace tail by its decoded-instruction mnemonics."""
    f = _open_log(file_path)
    if f is None:
        return ""
        
    tail = deque(maxlen=_TRACE_TAIL_LINES)
    try:
        with f:
            for line in f:
                tail.append(line)
    except Exception:
        return ""
        
    mnemonics = []
    for line in tail:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 5:
            mnem = parts[4].strip().split()[0] if parts[4].strip() else ""
            if mnem and not mnem[0].isdigit():
                mnemonics.append(mnem.replace(".", "_"))
    return " ".join(mnemonics)


def build_case_representation(regr_path, sim_path, trace_path):
    """Build the representation for one failure case.
    Returns a tuple ``(document, flags)``.
    """
    regr_feat = extract_regr_features(regr_path)
    sim_feat = extract_sim_features(sim_path)

    parts = []

    has_mismatch = (regr_feat.get("mismatch_count") or 0) > 0
    if regr_feat.get("failed_test_name"):
        name = regr_feat["failed_test_name"].replace(".", "_")
        parts += [f"TEST_{name}"] * 2
        for word in re.split(r"[^a-zA-Z]+", name.lower()):
            if word and word not in ("riscv", "test", "core", "ibex"):
                parts += [f"W_{word}"] * 2
                
    if has_mismatch:
        parts += ["HAS_MISMATCH"] * 3
    if sim_feat.get("uvm_fatal"):
        parts += ["UVM_FATAL_FLAG"] * 2
    if sim_feat.get("uvm_error"):
        parts += ["UVM_ERROR_FLAG"] * 2
    if sim_feat.get("is_timeout"):
        parts += ["TIMEOUT_FLAG"] * 2

    parts.append(_sim_tokens(sim_path))

    document = " ".join(p for p in parts if p)
    flags = [
        1.0 if has_mismatch else 0.0,
        float(sim_feat.get("uvm_fatal", False)),
        float(sim_feat.get("uvm_error", False)),
        float(sim_feat.get("is_timeout", False)),
    ]
    return document, flags


def build_case_document(regr_path, sim_path, trace_path):
    """Backward-compatible helper returning only the text document."""
    return build_case_representation(regr_path, sim_path, trace_path)[0]


if __name__ == "__main__":
    print("--- Testing Feature Extraction (Regex & Safe Streaming) ---")
    case1_dir = "logs/case_1"
    print(f"\n[ Scanning {case1_dir} ]")
    print("Regr Features: ", extract_regr_features(f"{case1_dir}/regr.log"))
    print("Sim Features:  ", extract_sim_features(f"{case1_dir}/sim.log"))
    print("Trace Features:", extract_trace_features(f"{case1_dir}/trace.log"))
