"""Overnight cross-vintage validation.

Runs 6 training jobs (2 variants × 3 vintages) sequentially, followed by
portfolio analysis for each, and a final combined results table.

Variants:
  A — Grade-Blind:  PD + ANR models on origination features only
  B — Grade-Aware:  PD + ANR models on origination + int_rate + sub_grade

Vintages: 2012, 2013, 2014   (2015 already done)
"""

import subprocess, sys, time, json, textwrap
from datetime import datetime
from pathlib import Path

ROOT   = Path("/Users/manoja/Documents/GitHub/Credit-Profit-Risk-Analysis/rebuild")
PY     = str(ROOT / ".venv/bin/python")
LOG    = ROOT / "reports/overnight_log.txt"
REPORT = ROOT / "reports/cross_vintage_results.txt"

JOBS = [
    # (label,                   test_year, include_grade, suffix)
    ("2012 Variant-A Grade-Blind", 2012, False, "_2012"),
    ("2012 Variant-B Grade-Aware", 2012, True,  "_with_grade_2012"),
    ("2013 Variant-A Grade-Blind", 2013, False, "_2013"),
    ("2013 Variant-B Grade-Aware", 2013, True,  "_with_grade_2013"),
    ("2014 Variant-A Grade-Blind", 2014, False, "_2014"),
    ("2014 Variant-B Grade-Aware", 2014, True,  "_with_grade_2014"),
]

# 2015 results already exist — include in final report under these suffixes
EXISTING_2015 = [
    ("2015 Variant-A Grade-Blind", "_2015_existing",   "_"),
    ("2015 Variant-B Grade-Aware", "_2015_wg_existing","_with_grade"),
]


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd: list[str], label: str) -> bool:
    log(f"START  {label}")
    log(f"CMD    {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0
    with open(LOG, "a") as f:
        f.write(result.stdout[-3000:] if result.stdout else "")
        if result.stderr:
            f.write("\nSTDERR:\n" + result.stderr[-1000:])
    if result.returncode == 0:
        log(f"OK     {label}  ({elapsed/60:.1f} min)")
        return True
    else:
        log(f"FAIL   {label}  (exit {result.returncode}) — skipping portfolio step")
        log(f"       Last stdout: {result.stdout[-300:]!r}")
        return False


def portfolio(suffix: str, label: str):
    for mod in ["returns", "portfolio"]:
        ok = run([PY, "-m", f"src.{mod}", "--suffix", suffix], f"{label} → src.{mod}")
        if not ok:
            log(f"WARNING portfolio step failed for {label}")
            break


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("OVERNIGHT CROSS-VINTAGE VALIDATION  START")
    log(f"Jobs: {len(JOBS)}  |  Estimated time: 4-5 hours")
    log("=" * 60)

    completed = []
    failed    = []

    for label, test_year, include_grade, suffix in JOBS:
        log("-" * 60)
        # Build training command
        train_cmd = [
            PY, "-m", "src.train",
            "--train-sample", "0",
            "--no-zoo",
            "--trials", "20",
            "--return-arch", "lightgbm",
            "--return-trials", "20",
            "--split-mode", "oot",
            "--test-year", str(test_year),
        ]
        if include_grade:
            train_cmd.append("--include-grade")

        ok = run(train_cmd, f"TRAIN  {label}")
        if ok:
            portfolio(suffix, label)
            completed.append((label, suffix))
        else:
            failed.append(label)

    log("=" * 60)
    log(f"TRAINING COMPLETE — {len(completed)} succeeded, {len(failed)} failed")
    if failed:
        log(f"FAILED: {failed}")
    log("=" * 60)

    # ── Compile final results table ──────────────────────────────────────────
    import pandas as pd, numpy as np
    from src import returns as R, config as C
    from src.returns import return_curve, POLICIES, optimal_fraction

    def load_book(suffix):
        path = C.PROCESSED_DIR / f"test_with_pd{suffix}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["_anr"] = R.realized_anr(df)
        mat = R.is_matured(df)
        return df[mat].copy()

    def curves_for(book):
        out = []
        for name, (col, asc) in POLICIES.items():
            if name == "invest_all" or col not in book.columns:
                continue
            c = return_curve(book, score_col=col, ascending=asc)
            c["policy"] = name
            out.append(c)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def get_at(fc, policy, frac):
        grp = fc[fc["policy"] == policy]
        if grp.empty: return float("nan")
        return grp.loc[(grp["fraction_invested"] - frac).abs().idxmin(), "port_anr"]

    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-VINTAGE RESULTS  — Full curves by vintage and variant")
    lines.append("=" * 80)

    fracs = [0.01, 0.03, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75]

    # include 2015 existing results
    all_runs = list(JOBS) + [
        ("2015 Variant-A Grade-Blind", 2015, False, ""),
        ("2015 Variant-B Grade-Aware", 2015, True,  "_with_grade"),
    ]

    for label, test_year, include_grade, suffix in all_runs:
        book = load_book(suffix)
        if book is None:
            lines.append(f"\n[{label}]  — data not found, skipped")
            continue
        fc = curves_for(book)
        if fc.empty:
            lines.append(f"\n[{label}]  — no curves")
            continue

        lines.append(f"\n{'─'*80}")
        lines.append(f"[{label}]  test_year={test_year}  matured={len(book):,} loans  "
                     f"capital=${book['funded_amnt'].sum()/1e6:.0f}M")
        lines.append(f"{'─'*80}")

        hdr = f"  {'Budget':>7}  {'invest_all':>11}  {'grade_only':>11}  "
        hdr += f"{'default_scr':>12}  {'profit_scr':>11}  {'ps vs ds (bp)':>14}  {'ps vs grade (bp)':>17}"
        lines.append(hdr)
        lines.append("  " + "-"*7 + "  " + "  ".join(["-"*11, "-"*11, "-"*12, "-"*11, "-"*14, "-"*17]))

        ia = book["_anr"].mul(book["funded_amnt"]).sum() / book["funded_amnt"].sum()
        for f in fracs:
            go = get_at(fc, "grade_only", f)
            ds = get_at(fc, "default_scoring", f)
            ps = get_at(fc, "profit_scoring", f)
            ps_vs_ds    = (ps - ds) * 10000 if not (np.isnan(ps) or np.isnan(ds)) else float("nan")
            ps_vs_grade = (ps - go) * 10000 if not (np.isnan(ps) or np.isnan(go)) else float("nan")
            lines.append(
                f"  {f:>6.0%}  {ia:>+10.2%}  {go:>+10.2%}  "
                f"{ds:>+11.2%}  {ps:>+10.2%}  "
                f"{ps_vs_ds:>+13.0f}bp  {ps_vs_grade:>+16.0f}bp"
            )

        # Optimal points
        lines.append("")
        for policy in ["profit_scoring", "default_scoring", "grade_only"]:
            grp = fc[fc["policy"] == policy]
            if grp.empty: continue
            opt = optimal_fraction(grp)
            lines.append(f"  Optimal {policy:20s}  "
                         f"budget={opt['fraction_invested']:.0%}  "
                         f"port_ANR={opt['port_anr']:+.2%}  "
                         f"n_loans={int(opt['n_invested']):,}")
        lines.append(f"  invest_all baseline:                    port_ANR={ia:+.2%}")

    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY — profit_scoring optimal ANR across vintages")
    lines.append("=" * 80)
    lines.append(f"  {'Vintage':>8}  {'Variant':>22}  {'Best budget':>12}  {'Best ANR':>10}  {'vs invest_all':>14}  {'vs grade_only':>14}")
    lines.append("  " + "-"*8 + "  " + "  ".join(["-"*22, "-"*12, "-"*10, "-"*14, "-"*14]))

    for label, test_year, include_grade, suffix in all_runs:
        book = load_book(suffix)
        if book is None: continue
        fc = curves_for(book)
        if fc.empty: continue
        ia = book["_anr"].mul(book["funded_amnt"]).sum() / book["funded_amnt"].sum()
        grp_ps    = fc[fc["policy"] == "profit_scoring"]
        grp_grade = fc[fc["policy"] == "grade_only"]
        if grp_ps.empty: continue
        opt = optimal_fraction(grp_ps)
        grade_at_opt = get_at(fc, "grade_only", opt["fraction_invested"])
        variant_short = "Grade-Blind" if not include_grade else "Grade-Aware"
        lines.append(
            f"  {test_year:>8}  {variant_short:>22}  "
            f"{opt['fraction_invested']:>11.0%}  "
            f"{opt['port_anr']:>+9.2%}  "
            f"{(opt['port_anr']-ia)*10000:>+13.0f}bp  "
            f"{(opt['port_anr']-grade_at_opt)*10000:>+13.0f}bp"
        )

    report_text = "\n".join(lines)
    REPORT.write_text(report_text)
    log(f"\nFull results saved to: {REPORT}")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
