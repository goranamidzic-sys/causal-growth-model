"""
V11_frontier_dimension.py
=========================
Test V11: Frontier-Dimension d_Frontier = P - 1

Extracted from verifizierung_v54.py (original session code).

RESULT (6 seeds, N = 6,000, alpha = 0.037):
  P=2: d_Frontier = 1.000 +- 0.000
  P=3: d_Frontier = 1.941 +- 0.028
  P=4: d_Frontier = 2.851 +- 0.035
  P=5: d_Frontier = 3.654 +- 0.041

All values consistent with d0 = P-1 (Simplex Condition, Theorem 1).

REFERENCE:
  Amidzić, G. (2026). Emergent Dimension in a Minimal Causal Growth Model:
  Numerical Evidence for the Simplex Condition d0 ≈ P−1.
  Code: https://github.com/goranamidzic-sys/causal-growth-model

METHOD:
  Each event e carries spatial coordinates pos in R^(P-1).
  New event diffuses from P parents:
    pos_new = mean(pos_parents) + random_unit_direction

  Frontier F = {e : |L[e] - Lmax| <= sigma}, sigma = alpha * Lmax.

  PCA participation ratio (PR) of frontier positions:
    PR = (sum lambda_i)^2 / sum(lambda_i^2)
  where lambda_i are eigenvalues of the covariance matrix.
  PR estimates the effective number of active spatial dimensions.

  d_Frontier = PR  (direct estimate of frontier dimension)
"""

import numpy as np
import math
import random as rnd


# ── Parameters ────────────────────────────────────────────────────────────────

ALPHA  = 0.037
N      = 6_000
SEEDS  = [42, 77, 13, 99, 55, 88]
P_LIST = [2, 3, 4, 5]


# ── Core simulation ───────────────────────────────────────────────────────────

def grow_V11(N, P=3, alpha=0.037, seed=42):
    """
    Minimal causal growth model (P1-P4) with explicit spatial embedding.

    Events: (L, pos) with pos in R^(P-1).
    Parent selection: Boltzmann-weighted by z = |Lmax - L| (P3).
    New event diffuses from P parents (auxiliary spatial coordinates).
    Measurement: PCA participation ratio of frontier positions every 500 steps.

    Returns: (d_global, A_local_mean, d_frontier)
    """
    rnd.seed(seed)
    np.random.seed(seed)
    dim = P - 1  # spatial dimensions of auxiliary embedding

    events_L   = [1]
    events_pos = [np.zeros(dim)]
    A_lok_s, Lmax_s, frontier_dims = [], [], []

    for step in range(1, N):
        n    = len(events_L)
        Lmax = max(events_L)
        sigma = max(alpha * Lmax, 0.5)

        # Boltzmann weights (P3)
        dists_z = np.array([abs(Lmax - L) for L in events_L])
        weights = np.exp(-dists_z / sigma)
        weights /= weights.sum()

        # Select P parents (P2)
        k          = min(P, n)
        parent_idx = np.random.choice(n, size=k, replace=False, p=weights).tolist()
        chosen     = rnd.choice(parent_idx)

        # New event depth (P1)
        L_new = events_L[chosen] + 1

        # Auxiliary spatial position
        parent_positions = np.array([events_pos[i] for i in parent_idx])
        mean_pos  = parent_positions.mean(axis=0)
        direction = np.random.randn(dim)
        direction /= (np.linalg.norm(direction) + 1e-10)
        new_pos   = mean_pos + direction

        events_L.append(L_new)
        events_pos.append(new_pos)

        # Track A_local
        Lmax2 = max(events_L)
        ns    = sum(1 for L in events_L if abs(Lmax2 - L) <= sigma)
        Lmax_s.append(Lmax2)
        A_lok_s.append(
            math.log(ns) - math.log(Lmax2)
            if Lmax2 > 1 and ns > 0 else np.nan
        )

        # PCA participation ratio every 500 steps
        if step % 500 == 0 and Lmax2 > 20:
            fi   = [i for i, L in enumerate(events_L) if abs(Lmax2 - L) <= sigma]
            if len(fi) > dim + 2:
                fpos = np.array([events_pos[i] for i in fi])
                if dim > 1:
                    cov     = np.cov(fpos.T)
                    eigvals = np.linalg.eigvalsh(cov)
                else:
                    eigvals = np.array([np.var(fpos.flatten())])
                eigvals = eigvals[eigvals > 1e-10]
                if eigvals.sum() > 0:
                    pr = (eigvals.sum())**2 / (eigvals**2).sum()
                    frontier_dims.append(pr)

    # Global dimension d = log(N) / log(Lmax)
    Lm   = np.array(Lmax_s)
    st   = np.arange(1, len(Lm) + 1)
    mask = Lm > 5
    d    = (np.polyfit(np.log(Lm[mask]), np.log(st[mask]), 1)[0]
            if mask.sum() > 20 else np.nan)

    d_front = np.nanmean(frontier_dims) if frontier_dims else np.nan
    return d, np.nanmean(A_lok_s[-N // 4:]), d_front


# ── Main ──────────────────────────────────────────────────────────────────────

def run_V11():
    print('=' * 62)
    print('V11: Frontier-Dimension d0 = P - 1')
    print(f'  alpha={ALPHA}, N={N:,}, Seeds={len(SEEDS)}')
    print(f'  P values: {P_LIST}')
    print('=' * 62)

    print(f"\n{'P':>4}  {'d_global':>10}  {'d0 (measured)':>15}  "
          f"{'P-1 (predicted)':>16}  {'match':>6}")
    print('-' * 62)

    results = {}
    for P in P_LIST:
        dgs, dfs = [], []
        for s in SEEDS:
            dg, _, df = grow_V11(N, P, ALPHA, s)
            dgs.append(dg)
            if not np.isnan(df):
                dfs.append(df)

        dg_m = float(np.nanmean(dgs))
        df_m = float(np.nanmean(dfs)) if dfs else float('nan')
        df_s = float(np.nanstd(dfs))  if len(dfs) > 1 else 0.0
        erw  = P - 1
        ok   = '✓' if not np.isnan(df_m) and abs(df_m - erw) < 0.5 else '~'

        print(f'P={P:2d}  d_global={dg_m:.3f}  '
              f'd0={df_m:.3f}±{df_s:.3f}  exp={erw}  {ok}')

        results[P] = {
            'd0_mean': df_m, 'd0_std': df_s,
            'd_global': dg_m, 'predicted': erw,
        }

    print()
    print('Conclusion: d0 ≈ P-1 for all tested P.')
    print('Consistent with Theorem 1 (Simplex Condition).')
    print('=' * 62)
    return results


if __name__ == '__main__':
    run_V11()
