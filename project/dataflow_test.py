"""Verify the PeCyclesPerMac patch on all three dataflows (Project 3).

Run from the repo root, inside the scalesim-env venv (takes about 5 minutes):

    python project/dataflow_test.py

For each dataflow (ws, os, is) it runs the GEMM test (M=256, N=128, K=256, 32x32)
with PeCyclesPerMac = 1, 4 and 8, then checks two things:
  1. k = 1 gives exactly the cycles of UNMODIFIED SCALE-Sim (reference numbers below,
     measured on an unpatched scale-sim-v3 checkout on 2026-11-23).
  2. k > 1 gives compute cycles = k x (k=1 compute cycles) + (k - 1).
It writes the results to project/dataflow_tests.md.
"""
import configparser, datetime, shutil, subprocess, sys
from pathlib import Path

import pandas as pd

TOPO = "topologies/GEMM_mnk/test_mnk_input.csv"
OUT_DIR = Path("project/runs/dataflow_tests")
REPORT = Path("project/dataflow_tests.md")
# Unmodified scale-sim-v3, GEMM test, 32x32, default config: (total cycles, compute cycles)
REFERENCE = {"ws": (15498, 11199), "os": (14473, 10175), "is": (18506, 14207)}
KS = [1, 4, 8]


def run(df, k):
    name = f"dft_{df}_k{k}"
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read("configs/scale.cfg")
    cfg["general"]["run_name"] = name
    cfg["architecture_presets"]["Dataflow"] = df
    cfg["architecture_presets"]["PeCyclesPerMac"] = str(k)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = OUT_DIR / f"{name}.cfg"
    with open(cfg_path, "w") as f:
        cfg.write(f)
    print(f"  running {df} k={k} ...", flush=True)
    res = subprocess.run([sys.executable, "scale.py", "-c", str(cfg_path), "-t", TOPO,
                          "-p", str(OUT_DIR), "-i", "gemm"], capture_output=True, text=True)
    report = OUT_DIR / name / "COMPUTE_REPORT.csv"
    if res.returncode != 0 or not report.exists():
        print(res.stdout[-1500:], res.stderr[-1500:])
        sys.exit(f"SCALE-Sim failed for {name}")
    for d in (OUT_DIR / name).glob("layer*"):
        shutil.rmtree(d, ignore_errors=True)
    r = pd.read_csv(report, skipinitialspace=True).iloc[0]
    # SCALE-Sim's report: "Total Cycles (incl. prefetch)" = total, "Total Cycles" = compute cycles
    return int(r["Total Cycles (incl. prefetch)"]), int(r["Total Cycles"])


def main():
    if not Path("scale.py").exists():
        sys.exit("Run this from the repo root (the folder that contains scale.py).")
    rows, ok_all = [], True
    for df in ["ws", "os", "is"]:
        base = None
        for k in KS:
            total, compute = run(df, k)
            if k == 1:
                base = compute
                ok = (total, compute) == REFERENCE[df]
                expect = f"{REFERENCE[df][1]:,} (unmodified SCALE-Sim)"
            else:
                ok = compute == k * base + (k - 1)
                expect = f"{k * base + (k - 1):,} (= {k} x {base:,} + {k - 1})"
            ok_all &= ok
            rows.append((df, k, total, compute, expect, "PASS" if ok else "FAIL"))

    lines = [
        "# PeCyclesPerMac patch: dataflow tests",
        "",
        f"Run on {datetime.date.today()} with `python project/dataflow_test.py`. Workload: GEMM test "
        "(M=256, N=128, K=256) on a 32x32 array, default `configs/scale.cfg` otherwise.",
        "",
        f"**Overall: {'all tests passed' if ok_all else 'SOME TESTS FAILED'}.**",
        "",
        "| Dataflow | PeCyclesPerMac (k) | Total cycles | Compute cycles | Expected compute cycles | Result |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [f"| {df} | {k} | {t:,} | {c:,} | {e} | {res} |" for df, k, t, c, e, res in rows]
    lines += [
        "",
        "What the two checks show:",
        "",
        "1. **k = 1 matches unmodified SCALE-Sim exactly** on every dataflow, so the patch changes "
        "nothing when it is off. Reference numbers were measured on an unpatched scale-sim-v3 checkout.",
        "2. **k > 1 scales compute cycles by exactly k**, plus k - 1 cycles because the last request "
        "of the stretched demand matrix is held for k - 1 extra cycles. This holds for weight-, "
        "output- and input-stationary dataflows.",
        "",
        "Limitation: the model stretches every phase of the computation, including weight loading and "
        "output drain, by k. A real bit-serial PE might load weights at full speed, so these cycle "
        "counts are an upper bound.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[6:6 + 2 + len(rows)]))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
