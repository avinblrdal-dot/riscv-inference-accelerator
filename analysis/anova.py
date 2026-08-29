#!/usr/bin/env python3
"""Multi-way ANOVA over the factorial design, INCLUDING interaction terms.

WHY INTERACTIONS ARE THE WHOLE POINT
------------------------------------
RQ3 asks: "does optimising data movement reduce energy more than adding
compute units?" Stated precisely, the hypothesis is that the effect of
buffer depth DEPENDS ON array width -- buffering should matter enormously for
a wide array (which is bandwidth-starved) and hardly at all for a 1-wide array
(which never was). That "depends on" is exactly what a statistical INTERACTION
term is.

A study that varied one factor at a time could not detect this at all. It
would report two main effects and miss the actual finding. That is why
sweep_config.yaml specifies a full factorial.

WHAT THIS REPORTS
-----------------
For each response variable:
  * sum of squares, df, mean square, F, p for every main effect and interaction
  * PARTIAL ETA SQUARED as an effect size
  * a multiple-comparison correction across the family of tests

EFFECT SIZE MATTERS MORE THAN p HERE. With 320 rows almost everything reaches
p < 0.05, including effects far too small to care about. Partial eta squared
says how much variance a factor actually explains, which is the question an
engineer is really asking. Report both; lead with the effect size.

MULTIPLE COMPARISONS
--------------------
We test several effects across several response variables, so some "significant"
results would appear by chance alone. Benjamini-Hochberg (controlling the false
discovery rate) is applied across the whole family and reported alongside the
raw p-values. FDR rather than Bonferroni because this is exploratory: we would
rather tolerate a few false positives than miss a real effect.

IMPLEMENTATION
--------------
Uses statsmodels when available. When it is not, a self-contained numpy
implementation of balanced factorial ANOVA runs instead, so the analysis is
never blocked by a missing package. The two agree on balanced designs (which
ours is); the fallback prints a note saying which path ran.

Usage:
    python3 analysis/anova.py --csv sweep/results/synthetic_results.csv
    python3 analysis/anova.py --csv ... --response cycles_total
    python3 analysis/anova.py --validate     # check it finds a known effect
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from load_results import load_csv, synthesize_results  # noqa: E402

FACTORS = ["array_w", "wbuf_depth", "precision"]
RESPONSES = ["cycles_total", "stall_fraction", "utilisation", "luts", "fmax_mhz"]


# ---------------------------------------------------------------------------
# F distribution survival function, so p-values work without scipy.
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def f_sf(f: float, df1: int, df2: int) -> float:
    """P(F > f). Upper-tail p-value for the F distribution."""
    if f <= 0 or df1 <= 0 or df2 <= 0 or not math.isfinite(f):
        return 1.0
    return _betai(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f))


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05
                       ) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg FDR control. Returns (rejected, adjusted p-values)."""
    n = len(pvals)
    if n == 0:
        return [], []
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        val = min(prev, pvals[i] * n / (rank + 1))
        adj[i] = val
        prev = val
    return [adj[i] <= alpha for i in range(n)], adj


# ---------------------------------------------------------------------------
# Balanced factorial ANOVA, numpy only
# ---------------------------------------------------------------------------

def anova_numpy(rows: list[dict], factors: list[str], response: str) -> dict:
    """Type-III-equivalent ANOVA for a BALANCED full-factorial design.

    For a balanced design the factor effects are orthogonal, so each term's
    sum of squares can be computed independently as the sum of squared
    deviations of its cell means from the grand mean. That orthogonality is
    what makes this short implementation correct here -- it would NOT be
    correct for an unbalanced design, which is why balance is checked and
    reported below.
    """
    usable = [r for r in rows
              if r.get(response) is not None
              and all(r.get(f) is not None for f in factors)]
    if len(usable) < 2:
        return {"error": f"not enough usable rows for '{response}'"}

    y = np.array([float(r[response]) for r in usable])
    levels = {f: sorted({r[f] for r in usable}) for f in factors}
    codes = {f: np.array([levels[f].index(r[f]) for r in usable]) for f in factors}

    n_total = len(y)
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())

    # Balance check: every cell must have the same number of observations.
    cell_counts: dict = {}
    for i in range(n_total):
        key = tuple(codes[f][i] for f in factors)
        cell_counts[key] = cell_counts.get(key, 0) + 1
    balanced = len(set(cell_counts.values())) == 1

    terms = {}
    for order in range(1, len(factors) + 1):
        for combo in itertools.combinations(factors, order):
            # Cell means for this combination of factors.
            groups: dict = {}
            for i in range(n_total):
                key = tuple(codes[f][i] for f in combo)
                groups.setdefault(key, []).append(y[i])

            # Raw between-cell sum of squares for this combination.
            ss_cells = sum(len(v) * (np.mean(v) - grand) ** 2
                           for v in groups.values())

            # Subtract every lower-order term contained in this one, which is
            # what isolates the pure interaction from its constituent main
            # effects.
            ss = ss_cells
            for lower in range(1, order):
                for sub in itertools.combinations(combo, lower):
                    ss -= terms[" x ".join(sub)]["ss"]

            df = 1
            for f in combo:
                df *= (len(levels[f]) - 1)

            terms[" x ".join(combo)] = {"ss": float(ss), "df": int(df),
                                        "order": order}

    ss_model = sum(t["ss"] for t in terms.values())
    ss_error = ss_total - ss_model
    df_model = sum(t["df"] for t in terms.values())
    df_error = n_total - df_model - 1
    ms_error = ss_error / df_error if df_error > 0 else float("nan")

    # ------------------------------------------------------------------
    # Is this response DETERMINISTIC?
    # ------------------------------------------------------------------
    # RTL simulation is deterministic: identical parameters give identical
    # cycle counts, so replicates are exact duplicates and the residual
    # variance is ZERO. That is not a defect in the data -- it is a true
    # property of the measurement.
    #
    # But it means the classical F test is UNDEFINED: F = MS_effect/MS_error
    # divides by zero. Reporting F=inf or p=0 here would be a statistical
    # fiction, and adding artificial noise to make the test "work" would be
    # worse -- it would manufacture uncertainty that does not exist.
    #
    # The honest treatment: for a deterministic response the effects are
    # EXACT. There is no sampling variability to test against, so we report
    # the variance decomposition (what share of the total variation each term
    # explains) and omit F and p entirely.
    #
    # Responses that ARE stochastic -- anything from Vivado, whose placer is
    # seeded and heuristic -- get the ordinary F test.
    deterministic = ss_total > 0 and (ss_error / ss_total) < 1e-9

    for name, t in terms.items():
        t["ms"] = t["ss"] / t["df"] if t["df"] else float("nan")
        # Share of total variance explained. Always defined, and it is the
        # quantity an engineer actually wants: "how much of the spread in
        # cycle count is attributable to array width?"
        t["ss_fraction"] = t["ss"] / ss_total if ss_total > 0 else float("nan")

        if deterministic:
            t["F"] = float("nan")
            t["p"] = float("nan")
            t["partial_eta_sq"] = t["ss_fraction"]
        else:
            t["F"] = t["ms"] / ms_error if ms_error and ms_error > 0 else float("nan")
            t["p"] = f_sf(t["F"], t["df"], df_error) if df_error > 0 else float("nan")
            denom = t["ss"] + ss_error
            t["partial_eta_sq"] = t["ss"] / denom if denom > 0 else float("nan")

    return {
        "terms": terms, "ss_total": ss_total, "ss_error": ss_error,
        "df_error": df_error, "ms_error": ms_error, "n": n_total,
        "balanced": balanced, "response": response, "backend": "numpy",
        "deterministic": deterministic,
    }


def anova_statsmodels(rows: list[dict], factors: list[str],
                      response: str) -> dict | None:
    """Preferred path. Returns None if statsmodels is unavailable."""
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        return None

    df = pd.DataFrame([r for r in rows if r.get(response) is not None])
    if df.empty:
        return None
    for f in factors:
        df[f] = df[f].astype("category")

    formula = f"Q('{response}') ~ " + " * ".join(f"C({f})" for f in factors)
    model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)

    ss_error = float(table.loc["Residual", "sum_sq"])
    terms = {}
    for name, row in table.iterrows():
        if name == "Residual":
            continue
        pretty = name.replace("C(", "").replace(")", "").replace(":", " x ")
        ss = float(row["sum_sq"])
        terms[pretty] = {
            "ss": ss, "df": int(row["df"]),
            "ms": ss / row["df"] if row["df"] else float("nan"),
            "F": float(row["F"]), "p": float(row["PR(>F)"]),
            "partial_eta_sq": ss / (ss + ss_error) if (ss + ss_error) > 0 else float("nan"),
            "order": pretty.count(" x ") + 1,
        }
    return {
        "terms": terms, "ss_error": ss_error,
        "df_error": int(table.loc["Residual", "df"]),
        "n": len(df), "balanced": True, "response": response,
        "backend": "statsmodels",
    }


def run_anova(rows: list[dict], factors: list[str], response: str) -> dict:
    return anova_statsmodels(rows, factors, response) or \
           anova_numpy(rows, factors, response)


def describe_effect(eta: float) -> str:
    """Cohen-style labels, so effect sizes are interpretable at a glance."""
    if not math.isfinite(eta):
        return "?"
    if eta < 0.01:
        return "negligible"
    if eta < 0.06:
        return "small"
    if eta < 0.14:
        return "medium"
    return "LARGE"


def print_anova(res: dict, adj_p: dict | None = None) -> None:
    if "error" in res:
        print(f"  {res['error']}")
        return

    print(f"  n={res['n']}  backend={res['backend']}  "
          f"balanced={'yes' if res['balanced'] else 'NO'}")

    if res.get("deterministic"):
        print()
        print("  DETERMINISTIC RESPONSE -- residual variance is exactly zero.")
        print("  RTL simulation gives identical results for identical")
        print("  parameters, so there is no sampling error to test against.")
        print("  F and p are therefore UNDEFINED and are not reported: an F")
        print("  test here would be dividing by zero and dressing the result")
        print("  up as significance.")
        print("  The effects below are EXACT, reported as the share of total")
        print("  variance each term explains. Interpret them directly.")
    if not res["balanced"]:
        print("  WARNING: the design is UNBALANCED. The numpy backend assumes")
        print("  orthogonal effects, which only holds for balanced designs.")
        print("  Install statsmodels for a correct Type-II analysis, or find")
        print("  out why some cells have fewer replicates (usually failed runs).")

    det = res.get("deterministic", False)
    print()
    if det:
        print(f"  {'term':<34} {'df':>3} {'variance explained':>19}  effect")
        print("  " + "-" * 70)
    else:
        print(f"  {'term':<34} {'df':>3} {'F':>10} {'p':>11} "
              f"{'p(FDR)':>10} {'eta^2p':>8}  effect")
        print("  " + "-" * 88)

    for name in sorted(res["terms"], key=lambda k: (res["terms"][k]["order"],
                                                    -res["terms"][k]["partial_eta_sq"])):
        t = res["terms"][name]
        marker = ("  <-- INTERACTION" if t["order"] > 1
                  and t["partial_eta_sq"] >= 0.06 else "")
        if det:
            print(f"  {name:<34} {t['df']:>3} "
                  f"{100*t['ss_fraction']:>17.2f}%  "
                  f"{describe_effect(t['partial_eta_sq'])}{marker}")
        else:
            padj = adj_p.get((res["response"], name), float("nan")) if adj_p else float("nan")
            pstr = f"{t['p']:.3e}" if t["p"] >= 1e-300 else "<1e-300"
            pastr = f"{padj:.3e}" if math.isfinite(padj) else "  --"
            print(f"  {name:<34} {t['df']:>3} {t['F']:>10.2f} {pstr:>11} "
                  f"{pastr:>10} {t['partial_eta_sq']:>8.4f}  "
                  f"{describe_effect(t['partial_eta_sq'])}{marker}")


def validate() -> int:
    """Prove the analysis can recover an effect we deliberately planted.

    An ANOVA script that has never been checked against known ground truth is
    not evidence of anything. load_results.synthesize_results() injects a
    strong array_w x wbuf_depth interaction; if this test fails, the model is
    mis-specified and would miss the same effect in the real sweep.
    """
    print("=" * 88)
    print(" VALIDATION -- can the ANOVA recover a KNOWN interaction?")
    print("=" * 88)
    print(" Synthetic data is generated with a deliberate array_w x wbuf_depth")
    print(" interaction on stall_fraction: buffering helps a lot at width 8 and")
    print(" essentially not at all at width 1.")
    print()

    rows = synthesize_results(seed=1234, replicates=5)
    rows = [r for r in rows if r["workload"] == "workload_a"]

    res = run_anova(rows, FACTORS, "stall_fraction")
    print_anova(res)

    key = "array_w x wbuf_depth"
    t = res["terms"].get(key)
    print()
    if t is None:
        print(" FAIL: the interaction term is not even in the model.")
        return 1
    det = res.get("deterministic", False)
    ok_size = t["partial_eta_sq"] >= 0.06
    ok_p = det or (math.isfinite(t["p"]) and t["p"] <= 0.05)

    if not (ok_size and ok_p):
        print(f" FAIL: the planted interaction was NOT recovered "
              f"(variance explained={100*t['ss_fraction']:.2f}%, p={t['p']}).")
        print(" The ANOVA model is mis-specified -- it would miss this in the")
        print(" real sweep too. Fix this before trusting any analysis output.")
        return 1

    print(f" PASS: recovered the planted interaction -- it explains "
          f"{100*t['ss_fraction']:.2f}% of the variance in stall_fraction "
          f"({describe_effect(t['partial_eta_sq'])}).")
    if det:
        print(" (Reported without a p-value: the response is deterministic,")
        print("  so the effect is exact rather than statistically inferred.)")
    print()

    # Also exercise the stochastic path, which is what real synthesis data
    # will use -- otherwise the F/p code would never be tested at all.
    print(" Checking the stochastic path on fmax_mhz (Vivado placer noise)...")
    res2 = run_anova(rows, FACTORS, "fmax_mhz")
    t2 = res2["terms"].get("array_w")
    if t2 and math.isfinite(t2["p"]) and t2["p"] <= 0.05:
        print(f" PASS: array_w effect on fmax detected via the F test "
              f"(F={t2['F']:.1f}, p={t2['p']:.3g}, "
              f"eta^2p={t2['partial_eta_sq']:.4f}).")
    else:
        print(" FAIL: the F-test path did not detect the array_w effect on Fmax.")
        return 1

    print()
    print(" The analysis can detect the kind of effect RQ3 is about, on both")
    print(" deterministic and stochastic responses.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(ROOT, "sweep", "results",
                                                  "synthetic_results.csv"))
    ap.add_argument("--response", action="append", default=[],
                    help="response variable (repeatable); default: a standard set")
    ap.add_argument("--workload", default=None,
                    help="analyse one workload only (recommended -- pooling "
                         "two structurally different models hides effects)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--validate", action="store_true",
                    help="check the analysis against known ground truth")
    args = ap.parse_args()

    if args.validate:
        return validate()

    rows = load_csv(args.csv)
    if not rows:
        print("No usable rows.", file=sys.stderr)
        return 1

    workloads = ([args.workload] if args.workload
                 else sorted({r["workload"] for r in rows if r.get("workload")}))
    responses = args.response or RESPONSES

    # Collect every p-value first so the FDR correction can be applied across
    # the WHOLE family of tests, not per-table. Correcting per table would
    # under-correct and inflate the false discovery rate.
    all_results: dict = {}
    pvals, keys = [], []
    for wl in workloads:
        subset = [r for r in rows if r.get("workload") == wl]
        for resp in responses:
            if not any(r.get(resp) is not None for r in subset):
                continue
            res = run_anova(subset, FACTORS, resp)
            if "error" in res:
                continue
            all_results[(wl, resp)] = res
            if res.get("deterministic"):
                # No p-values exist for a deterministic response, so it
                # contributes nothing to the multiple-comparison family.
                continue
            for name, t in res["terms"].items():
                if math.isfinite(t["p"]):
                    pvals.append(t["p"])
                    keys.append((resp, name))

    _, adj = benjamini_hochberg(pvals, args.alpha)
    adj_map = {k: a for k, a in zip(keys, adj)}

    print("=" * 88)
    print(" MULTI-WAY ANOVA")
    print("=" * 88)
    print(f" factors: {', '.join(FACTORS)} (full factorial, with interactions)")
    print(f" FDR correction: Benjamini-Hochberg across "
          f"{len(pvals)} tests, alpha={args.alpha}")
    print()

    for (wl, resp), res in all_results.items():
        print("-" * 88)
        print(f" workload={wl}   response={resp}")
        print("-" * 88)
        print_anova(res, adj_map)
        print()

    # The RQ3 headline, pulled out explicitly so it is impossible to miss.
    print("=" * 88)
    print(" RQ3: does buffer depth interact with array width?")
    print("=" * 88)
    for (wl, resp), res in all_results.items():
        t = res["terms"].get("array_w x wbuf_depth")
        if t:
            verdict = ("SUPPORTED" if t["partial_eta_sq"] >= 0.06
                       and adj_map.get((resp, "array_w x wbuf_depth"), 1) <= args.alpha
                       else "not supported")
            print(f"  {wl:12s} {resp:16s} eta^2p={t['partial_eta_sq']:.4f} "
                  f"({describe_effect(t['partial_eta_sq'])})  -> {verdict}")

    synth = [r for r in rows if r.get("accuracy_is_synthetic")]
    if synth:
        print()
        print(" REMINDER: this data is tagged synthetic. These are not results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
