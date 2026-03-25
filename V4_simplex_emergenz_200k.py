"""
V4_simplex_emergenz_200k.py
Dynamisches P_eff = d_Frontier+1  vs  festes P=4
Lauf A: P=4 fest (Kontrollgruppe, weiches Modell V51+)
Lauf B: P_eff(N) = d_Frontier(N) + 1, dynamisch

Frage: Konvergieren beide Läufe bei N* gegen dasselbe d?
       Erreicht P_eff in Lauf B die 4.00 bei N*?
       Fällt Δd = |d_A - d_B| → 0?

6 Seeds, N=200k, alpha=0.037
d = log(N) / log(Lmax)  [IMMER diese Formel, kein Fit]
"""

import numpy as np
import math
from collections import defaultdict

ALPHA = 0.037
N_MAX = 200_000
CHECKPOINTS = {1000, 5000, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000}
SEEDS = [42, 137, 314, 271, 1618, 2718]
P_FIXED = 4


def compute_d(n, lmax):
    if lmax <= 1 or n <= 1:
        return 0.0
    return math.log(n) / math.log(lmax)


def compute_d_frontier(level_counts, lmax, sigma):
    """
    d_Frontier = log(n_frontier) / log(span+1)
    Frontier = alle Knoten mit L >= lmax - 3*sigma
    """
    lmin_f = max(0, lmax - int(math.ceil(3 * sigma)))
    n_f = sum(level_counts.get(l, 0) for l in range(lmin_f, lmax + 1))
    span = lmax - lmin_f
    if n_f < 2 or span < 1:
        return 1.0
    return math.log(n_f) / math.log(span + 1)


def run(seed, dynamic_p=False):
    rng = np.random.default_rng(seed)
    lc = defaultdict(int)
    lc[0] = 1
    lmax = 0
    results = []
    p_sum = 0
    p_cnt = 0

    for step in range(N_MAX):
        sigma = max(1.0, ALPHA * lmax)
        cutoff = max(0, lmax - int(5 * sigma + 1))

        lvls = np.arange(cutoff, lmax + 1, dtype=np.int32)
        cnts = np.array([lc[int(l)] for l in lvls], dtype=np.float64)
        mask = cnts > 0
        lvls = lvls[mask]
        cnts = cnts[mask]

        if len(lvls) == 0:
            new_l = lmax + 1
            p_use = P_FIXED
        else:
            w = cnts * np.exp(-(lmax - lvls) / sigma)
            w /= w.sum()

            if dynamic_p:
                d_f = compute_d_frontier(lc, lmax, sigma)
                p_use = max(2, min(8, int(round(d_f + 1))))
            else:
                p_use = P_FIXED

            p_sum += p_use
            p_cnt += 1
            new_l = int(np.max(rng.choice(lvls, size=p_use, replace=True, p=w))) + 1

        lc[new_l] += 1
        if new_l > lmax:
            lmax = new_l

        n_check = step + 2  # +1 Wurzel
        if n_check in CHECKPOINTS:
            s2 = max(1.0, ALPHA * lmax)
            d_f2 = compute_d_frontier(lc, lmax, s2)
            p_mean = p_sum / p_cnt if p_cnt > 0 else P_FIXED
            results.append({
                'N':     n_check,
                'd':     compute_d(n_check, lmax),
                'd_f':   d_f2,
                'P':     p_mean,
                'lmax':  lmax,
            })

    return results


def print_table(label, all_results):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"{'N':>10}  {'d_mean':>7}  {'±std':>6}  {'d_f':>7}  {'P_eff':>7}  {'Lmax':>8}")
    for cp in sorted(CHECKPOINTS):
        ds, dfs, ps, lms = [], [], [], []
        for sr in all_results:
            r = next((x for x in sr if x['N'] == cp), None)
            if r:
                ds.append(r['d']); dfs.append(r['d_f'])
                ps.append(r['P']); lms.append(r['lmax'])
        if ds:
            print(f"{cp:>10}  {np.mean(ds):>7.4f}  {np.std(ds):>6.4f}  "
                  f"{np.mean(dfs):>7.4f}  {np.mean(ps):>7.3f}  {np.mean(lms):>8.1f}")


def main():
    print("=" * 72)
    print("  V4 — Dynamisches P_eff = d_Frontier+1  vs  festes P=4")
    print(f"  alpha={ALPHA}, N_max={N_MAX:,}, Seeds={SEEDS}")
    print("=" * 72)

    print("\nLauf A: P=4 fest ...")
    res_A = []
    for s in SEEDS:
        print(f"  Seed {s}", flush=True)
        res_A.append(run(s, dynamic_p=False))

    print("\nLauf B: P_eff = d_Frontier+1 (dynamisch) ...")
    res_B = []
    for s in SEEDS:
        print(f"  Seed {s}", flush=True)
        res_B.append(run(s, dynamic_p=True))

    print_table("LAUF A  —  P=4 fest", res_A)
    print_table("LAUF B  —  P_eff = d_Frontier+1 dynamisch", res_B)

    # Direktvergleich pro Seed bei N=100k und N=200k
    for target in [100000, 200000]:
        print(f"\n{'='*72}")
        print(f"  DIREKTVERGLEICH A vs B  bei N={target:,}")
        print(f"{'='*72}")
        print(f"  {'Seed':>6}  {'A: d':>7}  {'A: P':>5}  |  {'B: d':>7}  {'B: P_eff':>9}  {'Δd':>7}")
        diffs = []
        for i, s in enumerate(SEEDS):
            rA = next((r for r in res_A[i] if r['N'] == target), None)
            rB = next((r for r in res_B[i] if r['N'] == target), None)
            if rA and rB:
                dd = abs(rA['d'] - rB['d'])
                diffs.append(dd)
                print(f"  {s:>6}  {rA['d']:>7.4f}  {rA['P']:>5.1f}  |  "
                      f"{rB['d']:>7.4f}  {rB['P']:>9.3f}  {dd:>7.4f}")
        if diffs:
            print(f"  {'MITTEL':>6}  {'':>7}  {'':>5}     {'':>7}  {'':>9}  {np.mean(diffs):>7.4f}")

    # P_eff-Trajektorie in Lauf B — Kernbefund
    print(f"\n{'='*72}")
    print("  P_eff-TRAJEKTORIE  Lauf B")
    print("  Erwartung: P_eff wächst von ~2 auf ~4 bei N*")
    print("  Simplex-Bedingung: P_eff = d_Frontier + 1")
    print(f"{'='*72}")
    print(f"  {'N':>10}  {'P_eff':>8}  {'±std':>6}  {'d_f':>8}  {'P_eff - d_f':>12}  {'→ soll ≈ 1.0':>14}")
    for cp in sorted(CHECKPOINTS):
        ps, dfs = [], []
        for sr in res_B:
            r = next((x for x in sr if x['N'] == cp), None)
            if r:
                ps.append(r['P']); dfs.append(r['d_f'])
        if ps:
            diff = np.mean(ps) - np.mean(dfs)
            print(f"  {cp:>10}  {np.mean(ps):>8.4f}  {np.std(ps):>6.4f}  "
                  f"{np.mean(dfs):>8.4f}  {diff:>12.4f}  ← soll → 1.0")

    # Δd-Konvergenz über alle N
    print(f"\n{'='*72}")
    print("  Δd = |d_A - d_B|  über alle N  —  Konvergenzbeweis")
    print("  Erwartung: Δd → 0 bei N*")
    print(f"{'='*72}")
    print(f"  {'N':>10}  {'Δd_mean':>10}  {'Δd_std':>8}")
    for cp in sorted(CHECKPOINTS):
        diffs = []
        for i in range(len(SEEDS)):
            rA = next((r for r in res_A[i] if r['N'] == cp), None)
            rB = next((r for r in res_B[i] if r['N'] == cp), None)
            if rA and rB:
                diffs.append(abs(rA['d'] - rB['d']))
        if diffs:
            print(f"  {cp:>10}  {np.mean(diffs):>10.4f}  {np.std(diffs):>8.4f}")


if __name__ == "__main__":
    main()
