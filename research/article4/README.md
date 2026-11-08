# Article 4: independently specified Fiala feasibility study

This directory contains a **new quasi-steady bicycle-equilibrium implementation**, **not** the missing historical solver that produced `Article4_stage3_multivehicle_full.csv`.

## Contents
- `article4_reproducible_fiala.py`: saturated Fiala-type tire forces, quasi-steady lateral/yaw equilibrium, four initial guesses, root screening and CSV export.
- `test_fiala.py`: lightweight unit/regression tests. The tests check mathematical invariants and basic solver behavior; they are **not** experimental validation.

## Requirements
Python 3.10+, numpy, pandas, scipy.

```bash
python -m pip install numpy pandas scipy
cd research/article4
python -m unittest -v test_fiala.py
python article4_reproducible_fiala.py --input /path/to/real_vs_Test.csv --output results.csv
```

The input CSV must have **3,000 rows** and columns:
`route_id, frame_index, real_steering_deg, real_speed_kmh, test_steering_deg, test_speed_kmh`.

The script produces 3 vehicle configurations × 2 friction levels × 2 sides × 3,000 samples = **36,000 evaluations**. Speeds in the input are km/h; steering commands are degrees at the steering wheel, converted by an **assumed** steering ratio. The three vehicle parameter sets are hypothetical; they are **not measured vehicle parameters**.

## Scientific limitations
- The model is quasi-steady, not a time-domain transient vehicle simulator.
- Accepted numerical roots do not prove dynamic stability or uniqueness.
- `low_speed`, `invalid_input` and `no_verified_root` are not equivalent to observed unsafe driving.
- `yaw_relative_deviation` is a model discrepancy measure, not a calibrated crash-risk score.
- Results from this implementation must not be presented as an exact reproduction of the historical 36,000-row archive.
- Input provenance, historical solver provenance and empirical validation should be documented separately before making strong safety claims.

## Reproducibility status
Source code published; archived input CSV and output dataset are not yet committed to this directory. The unit tests have not been run in GitHub CI as part of this commit.
