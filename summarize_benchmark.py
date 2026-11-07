"""Turn the raw benchmark CSV file(s) into one table.   python3 summarize_benchmark.py results/classical_raw*.csv
Means are taken per seed first; the +- is the standard deviation across seeds (shown only with 2 or more seeds)."""
import sys, glob, numpy as np
import csv, os
OUT = os.environ.get("OUT", "results/classical_benchmark.md")
files = [f for p in (sys.argv[1:] or ["results/classical_raw*.csv"]) for f in glob.glob(p)]
rows = [r for f in files for r in csv.DictReader(open(f))]
num = lambda r, k: float(r[k]) if r[k] != "" else np.nan
conds = [c for c in ["nominal", "varied", "stress"] if any(r["cond"] == c for r in rows)]
ctrls = [c for c in ["pd", "pdg", "pid", "ct", "cti", "mpc", "ctnn", "ctinn", "asl"] if any(r["ctrl"] == c for r in rows)]
NAMES = {"pd": "PD", "pdg": "PD + gravity", "pid": "PID", "ct": "Computed torque", "cti": "Computed torque + integral", "mpc": "MPC", "ctnn": "Computed torque + network", "ctinn": "Computed torque + integral + network", "asl": "Adaptive (Slotine-Li)"}
out = []
for cond in conds:
    out += [f"### Condition: {cond}", "", "| Controller | Episodes | Success within 20 mm | Within 5 mm | Median place error (mm) | RMS joint error (mrad) | Settling time (s) | Never settled | Torque effort | Chatter (x1e-3) | Clipped (%) | Wall time per episode (s) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in ctrls:
        R = [r for r in rows if r["cond"] == cond and r["ctrl"] == c]
        if not R: continue
        seeds = sorted({int(r["seed"]) for r in R}); n = len(R)
        def mean_pm(key, scale=1.0, fmt="{:.2f}"):
            per = [np.nanmean([num(r, key) for r in R if int(r["seed"]) == s] or [np.nan]) * scale for s in seeds]
            m = np.nanmean(per); sd = np.nanstd(per, ddof=1) if len(seeds) > 1 else None
            return fmt.format(m) + (f" ± {fmt.format(sd)}" if sd is not None else "")
        succ = np.mean([int(r["success"]) for r in R]) * 100
        e = np.array([num(r, "err_mm") for r in R]); within5 = np.mean([(not np.isnan(x)) and x < 5 and int(r["success"]) for x, r in zip(e, R)]) * 100
        unset = np.nanmean([num(r, "unsettled") for r in R]) / 2 * 100
        out.append(f"| {NAMES[c]} | {n} | {succ:.0f}% | {within5:.0f}% | {np.nanmedian(e):.2f} | {mean_pm('rms_mrad')} | {mean_pm('settle_s')} | {unset:.0f}% | "
                   f"{mean_pm('effort', 1, '{:.3f}')} | {mean_pm('chatter', 1000, '{:.2f}')} | {mean_pm('clipped', 1, '{:.1f}')} | {mean_pm('wall_s', 1, '{:.1f}')} |")
    out.append("")
txt = "\n".join(out); print(txt)
open(OUT, "w").write(txt + "\n")
print("\nwrote", OUT, "  (files read:", ", ".join(files), ")")