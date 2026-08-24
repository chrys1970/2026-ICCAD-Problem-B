#!/usr/bin/env python3
"""
ICCAD 2026 Problem B -- realistic benchmark-set generator.

REWRITTEN (2026-07-19): the previous generator produced 4 fixed toy log
snippets that did not match the real benchmark formats at all (one-line
mismatches, no VCS header, no test name in sim.log, wrong trace columns),
so the pipeline's parsers could not even extract a test name from them.
This version models every log on the OFFICIAL formats from the problem PDF
(B_20260601, sections 2.1.1-2.1.3) and the two public benchmark sets:

  * regr.log     -- either a two-line "Mismatch[1]: / ibex[i] : pc[..]" block
                    ending in "[FAILED]: N matched, M mismatch", or the short
                    "riscv_xxx_test.0 : [FAILED]" verdict line.
  * sim.log.gz   -- VCS header ("Command: ... +UVM_TESTNAME=... +bin=...bin"),
                    UVM_INFO filler, optional UVM_FATAL / UVM_ERROR report
                    macros with realistic source paths, the
                    "--- RISC-V UVM TEST FAILED ---" banner and the UVM
                    report summary block.
  * trace.log.gz -- tab-separated "Time Cycle PC Insn Decoded Regs" rows with
                    monotonically advancing PCs, optionally ending in a tight
                    loop (stuck core).

Bug model (mirrors what the public goldens show): each injected bug has a
CATEGORY (debug / irq / mem / csr / decode) and produces SEVERAL different
symptoms across cases -- e.g. one debug bug can appear as an architectural
mismatch in one seed and as a testbench "No dret detected" UVM_FATAL in
another. This cross-symptom structure is exactly what makes the real
problem hard, so synthetic sets without it grade far too easy.

Usage examples:
  # benchmark_2-like set: 30 cases, 4 bugs
  python generate_testcase.py --out_dir sets/gen1 --cases 30 --bugs 4 --seed 1

  # benchmark_3-like set (private-tier size): 100 cases, 8 bugs
  python generate_testcase.py --out_dir sets/gen2 --cases 100 --bugs 8 --seed 2

  # stress test: long logs to measure runtime scaling
  python generate_testcase.py --out_dir sets/stress --cases 50 --bugs 8 \
      --sim-lines 200000 --trace-lines 500000 --seed 3

Output layout matches the official sets (input.csv + golden.csv + case_N/
directories), so score it exactly like the real ones:
  ./regr_fail_bucketing --input sets/gen1/input.csv --output /tmp/out.csv
  python evaluate.py --pred /tmp/out.csv --gold sets/gen1/golden.csv
"""
import argparse
import csv
import gzip
import os
import random

# ---------------------------------------------------------------------------
# Test pools per category -- names taken from the two public benchmark sets.
# ---------------------------------------------------------------------------
TESTS_BY_CAT = {
    "debug": ["riscv_debug_basic_test", "riscv_debug_stress_test",
              "riscv_dret_test", "riscv_ebreak_test",
              "riscv_debug_ebreak_test", "riscv_debug_ebreakmu_test"],
    "irq": ["riscv_single_interrupt_test", "riscv_multiple_interrupt_test",
            "riscv_interrupt_csr_test", "riscv_irq_csr_test"],
    "mem": ["riscv_mem_error_test", "riscv_unaligned_load_store_test",
            "riscv_loop_test"],
    "csr": ["riscv_csr_test", "riscv_invalid_csr_test",
            "riscv_machine_mode_rand_test"],
    "decode": ["riscv_rand_instr_test", "riscv_rand_jump_test",
               "riscv_mmu_stress_test", "riscv_hint_instr_test",
               "riscv_perf_counter_test", "riscv_user_mode_rand_test"],
}

# Testbench-side UVM_FATAL flavours (file, line, message) per category --
# copied from the real logs / the problem PDF.
TB_FATALS = {
    "debug": ("core_ibex_test_lib.sv", 335,
              "No dret detected, or incorrect privilege mode switch in"
              "                                     timeout period of 20000 cycles"),
    "irq": ("core_ibex_base_test.sv", 194,
            "Did not receive core_status HANDLING_IRQ within 750 cycle timeout period"),
    "mem": ("core_ibex_base_test.sv", 190,
            "Simulation timeout: memory response not observed within expected window"),
    "csr": ("core_ibex_base_test.sv", 194,
            "Did not receive core_status HANDLING_IRQ within 750 cycle timeout period"),
    "decode": ("core_ibex_base_test.sv", 190,
               "Simulation timeout: core made no forward progress"),
}

# DUT-RTL assertion identities (module path leaf, assertion name). A bug that
# is assigned one of these will always report THE SAME assertion -- assertion
# identity is a bug fingerprint in the real data.
RTL_ASSERTS = [
    ("load_store_unit_i", "IbexDataOffsetKnown"),
    ("id_stage_i", "IbexBranchDecisionValid"),
    ("cs_registers_i", "IbexCsrOpEnValid"),
    ("if_stage_i", "IbexInstrValidKnown"),
    ("ex_block_i", "IbexAluOpValid"),
    ("controller_i", "IbexCtrlStateValid"),
]

MNEMONICS = ["addi", "c.li", "lui", "auipc", "sw", "lw", "jal", "beq",
             "sltu", "andi", "c.j", "mulh", "sra", "slli", "remu", "c.and",
             "xori", "bne", "c.addi", "sub"]
REGS = ["x1", "x5", "x10", "x12", "x22", "x25", "x29", "t0", "gp", "s9"]


def _rand_pc(rng, base=0x80000000):
    return base + rng.randrange(0, 0x20000, 2)


def _trace_row(rng, time_ps, cycle, pc):
    mnem = rng.choice(MNEMONICS)
    r1, r2 = rng.choice(REGS), rng.choice(REGS)
    val = rng.randrange(0, 2**32)
    return (f"{time_ps}\t{cycle}\t{pc:08x}\t{rng.randrange(2**32):08x}\t"
            f"{mnem}\t{r1},{r2},{val & 0xfff}\t{r1}:0x{val:08x}\n")


def write_trace(path, rng, n_lines, stuck=False):
    """Trace log: advancing PCs; if `stuck`, the tail repeats a 4-PC loop
    (what a hung core looks like -- detected via unique-PC ratio)."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("Time\tCycle\tPC\tInsn\tDecoded instruction\tRegister and memory contents\n")
        pc = _rand_pc(rng)
        time_ps, cycle = 229000, 14
        loop_pcs = [pc + 0x400 + i * 4 for i in range(4)]
        for i in range(n_lines):
            if stuck and i > n_lines * 0.9:
                pc = loop_pcs[i % 4]          # tight loop at the tail
            else:
                pc += rng.choice((2, 4, 4))   # normal forward progress
            f.write(_trace_row(rng, time_ps, cycle, pc))
            time_ps += rng.choice((2000, 4000))
            cycle += rng.choice((1, 2))


def write_sim(path, rng, test_name, n_lines, fatal=None, rtl_assert=None):
    """Sim log: VCS header (carries the REAL test name in the +bin= path),
    UVM_INFO filler, optional fatal/assert report, failure banner, summary."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        seed = rng.randrange(1, 10000)
        f.write(
            "Command: ibex/dv/uvm/out/rtl_sim/vcs_simv +vcs+lic+wait "
            "+signature_addr=0x8ffffffc -ucli -do ibex/dv/uvm/vcs.tcl "
            f"+ntb_random_seed={seed} +UVM_TESTNAME=core_ibex_base_test "
            f"+bin=ibex/dv/uvm/out/instr_gen/asm_tests/{test_name}_0.bin "
            "-l sim.log +UVM_VERBOSITY=UVM_HIGH\n"
            "Chronologic VCS simulator copyright 1991-2023\n"
            "Contains Synopsys proprietary information.\n"
            "Compiler version U-2023.03-SP1_Full64; Runtime version "
            "U-2023.03-SP1_Full64;  Mar  8 06:05 2025\n"
            "UVM_INFO /global/apps/vcs_2023.03-SP1/etc/uvm-1.2/base/uvm_root.svh(402) "
            "@ 0: reporter [UVM/RELNOTES].\n"
            "UVM_INFO @ 0: reporter [RNTST] Running test core_ibex_base_test...\n"
            "UVM_INFO /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(98) @ 199000: "
            "uvm_test_top [uvm_test_top] Running test : /workspace/ibex/dv/uvm/out/"
            f"instr_gen/asm_tests/{test_name}_0.bin\n"
        )
        t = 200000
        for _ in range(max(0, n_lines - 20)):
            t += rng.randrange(1000, 9000)
            f.write(f"UVM_INFO /workspace/ibex/dv/uvm/env/core_ibex_env.sv"
                    f"({rng.randrange(40, 400)}) @ {t}: uvm_test_top "
                    "[uvm_test_top] Normal operation status update\n")
        if rtl_assert is not None:
            inst, name = rtl_assert
            f.write(f"UVM_ERROR /workspace/ibex/shared/rtl/prim_assert.sv(21) @ {t + 5000}: "
                    f"reporter [ASSERT FAILED] [core_ibex_tb_top.dut.u_ibex_core.{inst}.{name}]\n")
        if fatal is not None:
            file, line, msg = fatal
            f.write(f"UVM_FATAL /workspace/ibex/dv/uvm/tests/{file}({line}) @ {t + 9000}: "
                    f"uvm_test_top [uvm_test_top] {msg}\n")
        f.write("\n--- RISC-V UVM TEST FAILED ---\n\n")
        # end-of-run report summary (the pipeline must NOT trip on these).
        f.write("Number of demoted UVM_FATAL reports  :    0\n"
                "Number of demoted UVM_ERROR reports  :    0\n"
                "Number of demoted UVM_WARNING reports:    0\n"
                "Number of caught UVM_FATAL reports   :    0\n")


def write_regr(path, rng, test_name, mismatch):
    """Regr log: two-line mismatch block + stats line, or the short verdict."""
    with open(path, "w", encoding="utf-8") as f:
        if mismatch:
            idx = rng.randrange(20, 500)
            pc_i, pc_s = _rand_pc(rng), 0xffffffff80000000 + rng.randrange(0, 0x20000, 2)
            m1, m2 = rng.choice(MNEMONICS), rng.choice(MNEMONICS)
            r = rng.choice(REGS)
            f.write(f"Mismatch[1]:\n"
                    f"ibex[{idx}] : pc[{pc_i:08x}] {m1}\t{r},{r},"
                    f"-{rng.randrange(4, 200)}: {r}:{rng.randrange(2**32):08x}\n"
                    f"spike[{idx}] : pc[{pc_s:x}] {m2}    {r}, {r}, "
                    f"-{rng.randrange(4, 200)}: {r}:{rng.randrange(2**32):08x}\n"
                    f"[FAILED]: {rng.randrange(50, 30000)} matched, "
                    f"{rng.randrange(1000, 70000)} mismatch")
        else:
            f.write(f"{test_name}.0 : [FAILED]")


class Bug:
    """One injected bug: a category + a fixed set of symptom flavours.

    Real-data observations this reproduces:
      * a bug reports the SAME rtl assertion / tb fatal every time it shows
        that flavour (bug fingerprint),
      * but a single bug can show 2-3 DIFFERENT flavours across cases
        (mismatch vs fatal vs assert) -- the cross-symptom challenge,
      * tests come from the bug's category pool (a debug bug is tripped by
        debug/dret/ebreak tests etc.), with occasional off-category tests
        for the 'decode'-style bugs that anything random can trigger.
    """

    def __init__(self, rng, bug_id):
        self.bug_id = bug_id
        self.category = rng.choice(list(TESTS_BY_CAT))
        self.rtl_assert = rng.choice(RTL_ASSERTS) if rng.random() < 0.4 else None
        # each bug supports 1-3 symptom flavours, mismatch being most common
        flavours = ["mismatch"]
        if rng.random() < 0.6:
            flavours.append("tb_fatal")
        if self.rtl_assert is not None:
            flavours.append("rtl_assert")
        self.flavours = flavours

    def make_case(self, rng, case_dir, sim_lines, trace_lines):
        flavour = rng.choice(self.flavours)
        pool = TESTS_BY_CAT[self.category]
        # decode-ish bugs get tripped by anything, mirror that looseness
        if self.category == "decode" and rng.random() < 0.5:
            pool = rng.choice(list(TESTS_BY_CAT.values()))
        test = rng.choice(pool)

        fatal = rtl_assert = None
        mismatch = flavour == "mismatch"
        stuck = False
        if flavour == "tb_fatal":
            fatal = TB_FATALS[self.category]
            stuck = rng.random() < 0.5
        elif flavour == "rtl_assert":
            rtl_assert = self.rtl_assert

        os.makedirs(case_dir, exist_ok=True)
        write_regr(os.path.join(case_dir, "regr.log"), rng, test, mismatch)
        write_sim(os.path.join(case_dir, "sim.log.gz"), rng, test,
                  sim_lines, fatal=fatal, rtl_assert=rtl_assert)
        write_trace(os.path.join(case_dir, "trace.log.gz"), rng,
                    trace_lines, stuck=stuck)


def main():
    ap = argparse.ArgumentParser(
        description="Generate a realistic ICCAD Problem B benchmark set")
    ap.add_argument("--out_dir", required=True, help="Output set directory")
    ap.add_argument("--cases", type=int, default=30, help="Number of failure cases")
    ap.add_argument("--bugs", type=int, default=4, help="Number of injected bugs")
    ap.add_argument("--sim-lines", type=int, default=300,
                    help="Approx. lines per sim.log.gz (raise for stress tests)")
    ap.add_argument("--trace-lines", type=int, default=2000,
                    help="Approx. lines per trace.log.gz (raise for stress tests)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible sets)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    bugs = [Bug(rng, i) for i in range(args.bugs)]
    # every bug appears at least once; remaining cases drawn with a skewed
    # distribution (real sets have one dominant bug, cf. bug_107 = 15/25)
    assignment = list(range(args.bugs))
    weights = [rng.random() ** 2 + 0.1 for _ in bugs]
    while len(assignment) < args.cases:
        assignment.append(rng.choices(range(args.bugs), weights=weights)[0])
    rng.shuffle(assignment)

    with open(os.path.join(args.out_dir, "input.csv"), "w", newline="",
              encoding="utf-8") as cf, \
         open(os.path.join(args.out_dir, "golden.csv"), "w", newline="",
              encoding="utf-8") as gf:
        iw, gw = csv.writer(cf), csv.writer(gf)
        iw.writerow(["Case", "Regr Log", "Sim Log", "Trace Log"])
        gw.writerow(["Case", "Bug"])
        for i, bug_idx in enumerate(assignment, start=1):
            case = f"case_{i}"
            bugs[bug_idx].make_case(rng, os.path.join(args.out_dir, case),
                                    args.sim_lines, args.trace_lines)
            iw.writerow([i, f"{case}/regr.log", f"{case}/sim.log.gz",
                         f"{case}/trace.log.gz"])
            gw.writerow([i, f"bug_{bugs[bug_idx].bug_id}"])

    print(f"Generated {args.cases} cases / {args.bugs} bugs in '{args.out_dir}/'")
    print(f"Score with:\n  ./regr_fail_bucketing --input {args.out_dir}/input.csv "
          f"--output /tmp/out.csv\n  python evaluate.py --pred /tmp/out.csv "
          f"--gold {args.out_dir}/golden.csv")


if __name__ == "__main__":
    main()
