# On-device adaptation experiment (exploratory, D027)

Leave-one-pump-out on MIMII pump (DCASE 2020 Task 2 split). Each row: a model trained on the OTHER three pumps, installed on a pump it never heard, then adapted on N of that pump's normal clips (10 s each). Scores are on that pump's test clips, which are never used for adaptation. Mean ± standard deviation over 3 seed(s). AUC in %, int8 = the chip's arithmetic.

**These are industrial pumps, not handpumps** (D023). The learning is simulated in float on a laptop; only the scoring uses chip arithmetic.

| Method | Clips | id_00 | id_02 | id_04 | id_06 | Mean |
|---|---|---|---|---|---|---|
| unadapted | 0 | 78.5 ± 0.2 | 81.0 ± 1.0 | 63.5 ± 2.3 | 76.8 ± 0.6 | **74.9** ± 0.9 |
| renorm | 10 | 79.6 ± 1.2 | 80.5 ± 1.0 | 61.8 ± 3.0 | 76.8 ± 0.5 | **74.7** ± 1.0 |
| last | 10 | 80.0 ± 0.8 | 77.8 ± 2.4 | 70.1 ± 5.4 | 81.8 ± 1.9 | **77.4** ± 1.2 |
| full | 10 | 80.0 ± 0.7 | 76.2 ± 2.6 | 69.6 ± 5.6 | 81.8 ± 2.4 | **76.9** ± 1.3 |
| renorm | 30 | 79.6 ± 1.2 | 80.5 ± 1.0 | 61.6 ± 2.9 | 76.8 ± 0.5 | **74.6** ± 1.0 |
| last | 30 | 79.9 ± 0.9 | 78.0 ± 1.3 | 71.0 ± 3.4 | 82.9 ± 1.3 | **77.9** ± 1.0 |
| full | 30 | 79.8 ± 0.7 | 77.7 ± 1.6 | 71.5 ± 3.6 | 83.5 ± 0.8 | **78.1** ± 0.7 |
| renorm | 100 | 79.6 ± 1.2 | 80.5 ± 0.9 | 61.5 ± 3.1 | 76.7 ± 0.5 | **74.6** ± 1.1 |
| last | 100 | 79.7 ± 1.0 | 77.1 ± 0.9 | 71.7 ± 2.4 | 83.3 ± 1.0 | **78.0** ± 0.7 |
| full | 100 | 79.6 ± 0.6 | 75.5 ± 2.1 | 72.3 ± 2.2 | 85.0 ± 1.2 | **78.1** ± 1.0 |

Method key: `unadapted` = straight out of the box. `renorm` = re-measure only the input's average level and spread on this pump (2 numbers). `last` = retrain only the final layer (4,224 weights). `full` = retrain every layer (8,904 weights). Full per-row data, including float and pAUC: `train/runs/adapt_experiment/results.json` (regenerable).
