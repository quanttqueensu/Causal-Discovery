"""
Causal Discovery POC for Equity Portfolio
=========================================
End-to-end demo:
1) Generate synthetic stock + macro data with KNOWN planted causal edges
2) Run PCMCI (Tigramite) with ParCorr + BH-FDR correction on a rolling window
3) Print discovered causal edges (source -> target @ lag, MCI stat, p)
4) Compute a simple "causal momentum" signal per stock from its parents
5) Verify the algorithm recovers the planted edges
"""

import warnings
import numpy as np

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr  # NOTE: must be the .parcorr submodule

warnings.filterwarnings("ignore")

# ---- CONFIG ----------------------------------------------------------------
RNG_SEED = 42
N_STOCKS = 10
N_MACRO = 2
T_TOTAL = 500
WINDOW = 250          # PCMCI is fit on the most recent WINDOW observations
TAU_MIN = 1
TAU_MAX = 5
ALPHA = 0.05          # BH-FDR acceptance level
PC_ALPHA = 0.05       # PC-phase screening threshold

rng = np.random.default_rng(RNG_SEED)

# ---- 1) Generate synthetic data --------------------------------------------
N_VARS = N_MACRO + N_STOCKS
MACRO_1, MACRO_2 = 0, 1
STOCK_IDX = list(range(N_MACRO, N_VARS))
STOCK_A = STOCK_IDX[0]
STOCK_B = STOCK_IDX[1]

# (source, target, lag, coefficient)
PLANTED_EDGES = [
    (MACRO_1, STOCK_A, 2, 0.55),
    (MACRO_2, STOCK_B, 3, -0.50),
]


def generate_data():
    X = rng.normal(0, 1.0, size=(T_TOTAL, N_VARS)) * 0.01

    # Macro nodes: AR(1) persistence
    for t in range(1, T_TOTAL):
        X[t, MACRO_1] = 0.30 * X[t - 1, MACRO_1] + rng.normal(0, 0.02)
        X[t, MACRO_2] = 0.30 * X[t - 1, MACRO_2] + rng.normal(0, 0.02)

    # Stocks: mild own-autocorrelation + planted causal drives from macro
    for t in range(1, T_TOTAL):
        for stock in STOCK_IDX:
            X[t, stock] = 0.05 * X[t - 1, stock] + rng.normal(0, 0.015)
        for src, tgt, lag, coef in PLANTED_EDGES:
            if t - lag >= 0:
                X[t, tgt] += coef * X[t - lag, src]
    return X


data = generate_data()
var_names = ["macro_1", "macro_2"] + [f"stock_{i}" for i in range(N_STOCKS)]
print(f"Generated data: shape={data.shape}")
print("Planted edges (source -> target @ lag, coef):")
for src, tgt, lag, coef in PLANTED_EDGES:
    print(f"  {var_names[src]}(t-{lag}) --[{coef:+.2f}]--> {var_names[tgt]}(t)")

# ---- 2) Run PCMCI on the most recent WINDOW observations -------------------
window_data = data[-WINDOW:]
dataframe = pp.DataFrame(window_data, var_names=var_names)
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)

results = pcmci.run_pcmci(tau_min=TAU_MIN, tau_max=TAU_MAX, pc_alpha=PC_ALPHA)

q_matrix = pcmci.get_corrected_pvalues(
    p_matrix=results["p_matrix"],
    fdr_method="fdr_bh",
    tau_min=TAU_MIN,
    tau_max=TAU_MAX,
)
p_matrix = results["p_matrix"]
val_matrix = results["val_matrix"]   # indexed [source i, target j, lag tau]

# ---- 3) Collect discovered edges -------------------------------------------
print("\nDiscovered causal edges (after BH-FDR correction):")
discovered = []
for i in range(N_VARS):          # source
    for j in range(N_VARS):      # target
        for tau in range(TAU_MIN, TAU_MAX + 1):
            if q_matrix[i, j, tau] < ALPHA:
                mci = val_matrix[i, j, tau]   # scalar partial-correlation stat
                p = p_matrix[i, j, tau]
                discovered.append((i, j, tau, mci, p))
                kind = "self" if i == j else "cross"
                print(f"  [{kind:>5}] {var_names[i]}(t-{tau}) "
                      f"--MCI={mci:+.3f}, p={p:.4f}--> {var_names[j]}(t)")
if not discovered:
    print("  No significant edges discovered at alpha =", ALPHA)

# ---- 4) Causal-momentum signal (cross-causal parents only) -----------------
# Exclude self-edges: a stock's own lag is plain autocorrelation, not the
# external causal drive this signal is meant to capture.
print("\nCausal-momentum scores (per stock, latest bar):")
print(f"{'stock':<12} {'score':>10}   parents(used)")
print("-" * 50)

parents_by_target = {s: [] for s in STOCK_IDX}
for i, j, tau, mci, p in discovered:
    if j in parents_by_target and i != j:
        parents_by_target[j].append((i, tau, mci))

latest = data.shape[0] - 1
for s in STOCK_IDX:
    parents = parents_by_target[s]
    if not parents:
        print(f"{var_names[s]:<12} {'n/a':>10}   (no cross-causal parent found)")
        continue
    score = 0.0
    parts = []
    for src, lag, mci in parents:
        # realized return of the parent over the past `lag` bars
        parent_ret = data[latest - lag + 1: latest + 1, src].sum()
        score += mci * parent_ret
        parts.append(f"{var_names[src]}(t-{lag})")
    print(f"{var_names[s]:<12} {score:>10.4f}   ({', '.join(parts)})")

# ---- 5) Recovery check vs planted ground truth -----------------------------
print("\nRecovery check vs planted ground truth:")
discovered_set = {(i, j, tau) for (i, j, tau, _, _) in discovered}
planted_set = {(src, tgt, lag) for (src, tgt, lag, _) in PLANTED_EDGES}

hits = 0
for src, tgt, lag, coef in PLANTED_EDGES:
    found = (src, tgt, lag) in discovered_set
    hits += int(found)
    status = "FOUND " if found else "MISSED"
    print(f"  {status} {var_names[src]}(t-{lag}) --> {var_names[tgt]}(t) [coef={coef:+.2f}]")

self_edges = {(i, j, tau) for (i, j, tau) in discovered_set if i == j}
spurious_cross = discovered_set - planted_set - self_edges

print(f"\nSummary: planted={len(planted_set)}, hits={hits}, "
      f"spurious cross-edges={len(spurious_cross)}, "
      f"self-edges (expected from AR)={len(self_edges)}")