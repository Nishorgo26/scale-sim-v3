"""Energy sensitivity analysis for Project 3. No new simulations: it re-prices runs you already have.

Run from the repo root, inside the scalesim-env venv (takes well under a minute):

    python project/sensitivity.py
    python project/sensitivity.py --results project/results/resnet18_ws_sram64.csv

For every result row in the given CSVs it recomputes energy under a grid of cost scenarios:
  SRAM energy      5, 10, 20 pJ per 32-bit access   (costs.md uses 5)
  DRAM energy      10, 20, 30 pJ per bit            (costs.md uses 20)
  shifter-PE MAC   0.5x, 1x, 2x the costs.md value  (pot, apot, mitchell, bitserial)
That is 27 scenarios. Each row is compared with the baseline PE on 32x32 at the same buffer
size, under the same scenario. Output: project/results/sensitivity.csv plus a summary.
"""
import argparse, glob, itertools, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import energy  # noqa: E402
import sweep   # noqa: E402

SRAM_PJ_PER_32B = [5, 10, 20]
DRAM_PJ_PER_BIT = [10, 20, 30]
SHIFTER_SCALE = [0.5, 1.0, 2.0]
OUT = Path("project/results/sensitivity.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", default=sorted(glob.glob("project/results/resnet18_*.csv")),
                    help="sweep result CSVs whose runs to re-price (default: all ResNet-18 results)")
    args = ap.parse_args()
    if not args.results:
        sys.exit("No result CSVs found. Pass --results <file.csv>.")

    points = pd.concat([pd.read_csv(f) for f in args.results], ignore_index=True)
    if "sram_kb" not in points.columns:
        points["sram_kb"] = sweep.DEFAULT_SRAM
    points = points.drop_duplicates(subset=sweep.KEY)
    pe_cost = dict(energy.PE)                                    # costs.md values
    rows = []
    for sram, dram, scale in itertools.product(SRAM_PJ_PER_32B, DRAM_PJ_PER_BIT, SHIFTER_SCALE):
        energy.E_SRAM_BIT, energy.E_DRAM_BIT = sram / 32, dram
        energy.PE = {pe: ((e * scale if pe != "baseline" else e), bits)
                     for pe, (e, bits) in pe_cost.items()}
        for p in points.itertuples():
            run_dir = sweep.RUNS_DIR / sweep.run_name(p.side, p.dataflow, sweep.PE_CYCLES[p.pe],
                                                      p.workload, p.sram_kb)
            if not (run_dir / "COMPUTE_REPORT.csv").exists():
                sys.exit(f"Missing simulation output {run_dir}. Run this on the machine that ran the sweep.")
            topo, itype = sweep.WORKLOADS[p.workload]
            t = energy.compute(str(run_dir), topo, itype, p.pe).sum(numeric_only=True)
            rows.append(dict(sram_pj_32b=sram, dram_pj_bit=dram, shifter_scale=scale,
                             workload=p.workload, dataflow=p.dataflow, sram_kb=p.sram_kb,
                             array_mode=p.array_mode, pe=p.pe, side=p.side, cycles=int(t.cycles),
                             E_total_uJ=t.E_total_pJ / 1e6, mac_share=t.E_mac_pJ / t.E_total_pJ))
    energy.PE = pe_cost
    df = pd.DataFrame(rows)

    # Normalise each row to the 32x32 baseline in the same scenario and hardware context
    ctx = ["sram_pj_32b", "dram_pj_bit", "shifter_scale", "workload", "dataflow", "sram_kb"]
    base = (df[(df.pe == "baseline") & (df.array_mode == "fixed")]
            .set_index(ctx)[["cycles", "E_total_uJ"]]
            .rename(columns={"cycles": "base_cycles", "E_total_uJ": "base_E"}))
    df = df.join(base, on=ctx)
    df["energy_vs_base"] = df.E_total_uJ / df.base_E
    df["edp_vs_base"] = df.energy_vs_base * df.cycles / df.base_cycles
    df = df.drop(columns=["base_cycles", "base_E"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # Summary: range of each PE's EDP over all 27 scenarios, and who wins each scenario
    print(f"{len(df)} rows ({df.groupby(ctx).ngroups} scenario x context groups) -> {OUT}\n")
    rng = (df.groupby(["workload", "dataflow", "sram_kb", "array_mode", "pe"])
             .agg(edp_min=("edp_vs_base", "min"), edp_max=("edp_vs_base", "max"),
                  energy_min=("energy_vs_base", "min"), energy_max=("energy_vs_base", "max"))
             .reset_index())
    print("EDP and energy vs. baseline, min-max over all 27 cost scenarios:")
    print(rng.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    winners = df.loc[df.groupby(ctx)["edp_vs_base"].idxmin(), ctx + ["pe", "array_mode"]]
    tally = winners.groupby(["pe", "array_mode"]).size().rename("scenarios_won").reset_index()
    print("\nLowest-EDP design per scenario (count):")
    print(tally.to_string(index=False))


if __name__ == "__main__":
    main()
