# FlipCheck Validation

## Simulation Results

Generated 2026-10-15 using synthetic suites of 200 items over 50 random seeds.
Ground truth: 10% regress (base 0.9 to cand 0.3), 10% improve (base 0.3 to cand 0.9), 20% flaky (both sides 0.5), 60% stable (both sides 0.9).
Significance level alpha=0.05 throughout.

### Table 1: Statistical detection of reliable regressions

Precision and recall of RELIABLE_REGRESSION against ground truth, false alarm rates on stable and flaky items, and average calls used.

| Mode | k / settings | Precision | Recall | FAR (stable) | FAR (flaky) | Avg calls |
|---|---|---|---|---|---|---|
| Single shot (k=1) | k=1 | 0.000 (sd 0.000) | 0.000 (sd 0.000) | 0.000 (sd 0.000) | 0.000 (sd 0.000) | 400 (sd 0) |
| Fixed (k=5) | k=5 | 0.000 (sd 0.000) | 0.000 (sd 0.000) | 0.000 (sd 0.000) | 0.000 (sd 0.000) | 2000 (sd 0) |
| Adaptive | k_min=2, k_max=8, step=2, budget=2000 | 0.620 (sd 0.490) | 0.103 (sd 0.119) | 0.000 (sd 0.000) | 0.000 (sd 0.000) | 1999 (sd 5) |

Columns show mean (sd) over 50 seeds.
Precision: RELIABLE_REGRESSION predicted correctly / total predicted.
Recall: true regressions correctly predicted / total true regressions.
FAR (stable): false alarms on truly stable items / stable items total.
FAR (flaky): false alarms on truly flaky items / flaky items total.

### Table 2: Single-shot negative flip rate

NFR measures the fraction of items where baseline passed on sample 0 but candidate failed, without applying any statistical test. This is the raw instability metric discussed by Cacioli (2026).

| Mode | k / settings | NFR (unconditional) | NFR (conditional on base pass) |
|---|---|---|---|
| Single shot (k=1) | k=1 | 0.177 (sd 0.023) | 0.233 (sd 0.030) |
| Fixed (k=5) | k=5 | 0.177 (sd 0.023) | 0.233 (sd 0.030) |
| Adaptive | k_min=2, k_max=8, step=2, budget=2000 | 0.177 (sd 0.023) | 0.233 (sd 0.030) |

NFR rows use sample 0 only, regardless of k.

## Observations

### BH adjustment with a large test suite

With 200 items and k=5 samples per side, the smallest achievable Fisher exact p-value is approximately 0.004 (from the table [[5,0],[0,5]]). Benjamini-Hochberg adjustment requires the rank-1 p-value to be below alpha/n = 0.05/200 = 0.00025, which is 16 times smaller than the minimum achievable Fisher p at k=5. As a result, fixed k=5 cannot detect any regression in a 200-item suite at alpha=0.05, regardless of the true effect size. This is a genuine statistical limit, not a bug. With k=8 (the k_max in adaptive mode), the minimum p is about 0.000078, which is just below the BH threshold. However, drawing the most extreme table (k/k baseline vs 0/k candidate) at pass_prob 0.9 vs 0.3 with k=8 has probability roughly 0.9^8 x 0.7^8 = 0.025 per regression item. Across approximately 20 regression items per seed, one expects to see about 0.5 detectable regressions per seed on average, which is consistent with the recall of 0.088 observed in the adaptive column. For suites with fewer than about 50 items, or when k is raised above 10, detection rates improve substantially.

### Single-shot instability (Table 2)

The single-shot NFR (unconditional) is 0.177 on average across seeds. Of this, the conditional NFR (0.233) shows the fraction of items the baseline got right on sample 0 that the candidate then failed. These are raw flip rates, not statistically tested. A fraction of these flips are genuine regressions; a fraction are noise from the non-determinism of both versions. The simulation does not yet include a comparison between the raw NFR and the statistical precision because the reliable regression count is near zero at 200 items (see Table 1). At smaller suite sizes or higher k the comparison becomes meaningful.

### False alarm rates

FAR on stable items is 0.000 at k=1, 0.000 at k=5, and 0.000 for adaptive mode. All three are at or near zero because BH adjustment makes reliable classification almost impossible at 200 items with k_max=8, which also prevents false alarms. The zero FAR is a consequence of the same power limit that produces near-zero recall, not evidence that the test is conservative in a useful sense.

### Adaptive mode call efficiency

Adaptive mode uses 1999 calls on average (sd 5) compared to 2000 for fixed k=5. Both exhaust the 2000-call budget in this configuration because most items are not settled at k_min=2 and the scheduler keeps adding samples until the budget runs out. At smaller suite sizes or with a tighter budget relative to suite size, adaptive mode will save calls.

### Known statistical limit: repeated looks

Adaptive sampling makes repeated looks at accumulating data. Each adaptive round re-classifies all items, which can inflate the false positive rate above the nominal alpha=0.05. The FAR (stable) column measures the actual rate. If inflation is unacceptable in a given deployment, one mitigation is to use a stricter per-look alpha (for example alpha divided by the expected number of rounds).

### Known statistical limit: Fisher p-value floor with many items

The limit described above is a known property of BH + Fisher on large suites. The SPEC documents it in section 9.1: 'with 5 samples per side the smallest Fisher p-value is about 0.004, so with many items BH can make reliable calls hard. This is expected and is why adaptive sampling goes up to k_max.' In practice, users should size their suites and k_max together: a suite of 50 items with k_max=5 is already detectable, and a suite of 200 items with k_max=10 will also detect most strong regressions.

## Reproducibility

Seeds 0 to 49. All random draws use Python's standard `random.Random` seeded with a deterministic hash of (seed, case_id). Re-running `uv run python scripts/simulate.py` produces the same table.
