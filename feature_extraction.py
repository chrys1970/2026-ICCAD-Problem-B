import re
import os
import gzip


def _open_log(file_path):
    """Open a log file transparently, supporting both plain and gzip (.gz) files."""
    if file_path.endswith(".gz"):
        return gzip.open(file_path, "rt", encoding="utf-8", errors="replace") # check compression
    return open(file_path, "r", encoding="utf-8", errors="replace") # safe placeholder

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

    with _open_log(file_path) as f:
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

    # An actual report macro looks like:  UVM_FATAL <path>(<line>) @ <time>: ...
    # We must NOT trip on the end-of-run summary lines such as
    #   "UVM_FATAL :    0"  or  "Number of demoted UVM_FATAL reports :    0".
    macro_re = re.compile(r"\bUVM_(FATAL|ERROR)\b\s+\S+\(\d+\)\s*@")
    summary_re = re.compile(r"^\s*UVM_(FATAL|ERROR)\s*:\s*(\d+)\s*$")

    with _open_log(file_path) as f:
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
    """
    features = {
        "unique_pc_ratio": 1.0, # default to 1.0 (all unique) and will be reduced if we find repeats
        "is_stuck_in_loop": False
    }
    
    if not os.path.exists(file_path):
        return features

    try:
        with _open_log(file_path) as f:
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


# =====================================================================
#  Semantic document builder (text representation for TF-IDF / embeddings)
# =====================================================================
#
# Instead of a handful of scalar features, we turn each case into a short
# normalized "document" capturing the *meaning* of the failure:
#   - the failed test name
#   - the regr.log mismatch lines (instruction mnemonics, registers)
#   - the UVM_FATAL / UVM_ERROR *message text* from sim.log (the root-cause hint)
#   - the instruction mnemonics executing in the trace tail (near the crash)
#   - synthetic marker tokens for the structured signals
# Specific addresses / numbers are normalized away so cases cluster on the
# shape of the failure, not on random seeds or addresses.

_HEX_RE = re.compile(r"\b(?:0x)?[0-9a-fA-F]{4,}\b")
_NUM_RE = re.compile(r"\b\d+\b")
_TRACE_TAIL_LINES = 200      # how many trailing trace lines to summarize
_SIM_MAX_LINES = 2_000_000   # cap sim scan so huge logs stay within runtime


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
    """Extract the human-readable reason from UVM_FATAL/ERROR/WARNING lines.

    These messages (e.g. "Did not receive core_status HANDLING_IRQ within
    timeout", "No dret detected", assertion names) are the strongest root-cause
    signal, so we keep their words and drop file paths / line numbers / values.
    """
    if not os.path.exists(file_path):
        return ""
    words = []
    try:
        with _open_log(file_path) as f:
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
    if not os.path.exists(file_path):
        return ""
    from collections import deque
    tail = deque(maxlen=_TRACE_TAIL_LINES)
    try:
        with _open_log(file_path) as f:
            for line in f:
                tail.append(line)
    except Exception:
        return ""
    mnemonics = []
    for line in tail:
        parts = line.rstrip("\n").split("\t")
        # columns: Time | Cycle | PC | Insn | Decoded instruction | reg/mem
        if len(parts) >= 5:
            mnem = parts[4].strip().split()[0] if parts[4].strip() else ""
            if mnem and not mnem[0].isdigit():
                mnemonics.append(mnem.replace(".", "_"))
    return " ".join(mnemonics)


def build_case_representation(regr_path, sim_path, trace_path):
    """Build the representation for one failure case.

    Returns a tuple ``(document, flags)``:
      - ``document``: a normalized text document (test-name words, marker tokens,
        and the UVM failure-reason words) for TF-IDF / embedding.
      - ``flags``: the strong structured binary signals
        ``[has_mismatch, uvm_fatal, uvm_error, is_timeout]``, kept as an explicit
        weighted axis so the dominant split (e.g. mismatch vs not) is reliable.
    """
    regr_feat = extract_regr_features(regr_path)
    sim_feat = extract_sim_features(sim_path)

    parts = []

    # Strong, explicit signals as marker tokens (repeated to give them weight).
    has_mismatch = (regr_feat.get("mismatch_count") or 0) > 0
    if regr_feat.get("failed_test_name"):
        name = regr_feat["failed_test_name"].replace(".", "_")
        parts += [f"TEST_{name}"] * 2
        # Also emit the individual descriptive words (e.g. "interrupt", "csr",
        # "debug", "mem") so cases with related tests share features.
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

    # The UVM message words are high signal; the raw mismatch operands and trace
    # mnemonics are mostly per-seed noise, so they are intentionally excluded.
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
