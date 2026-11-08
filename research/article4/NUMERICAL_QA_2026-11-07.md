# Article 4 numerical archive QA (2026-11-07)

## Files examined
- `article4_reproducible_fiala.py` (previously authored independent quasi-steady Fiala solver)
- `Article4_new_fiala_full_36000.csv` (existing 36,000-row archived output)
- `Article4_v94_Editorial_QA.docx` (manuscript)

## Verified by independent local inspection
- CSV has 36,000 rows, 12 vehicle/friction/side groups of 3,000 rows each, and no duplicated (vehicle,mu,side,route_id,frame_index) keys.
- Status distribution: 31,329 `ok`, 4,656 `low_speed`, 15 `no_verified_root`.
- All 31,329 accepted rows have finite/nonmissing beta, nonlinear yaw, utilization and residual.
- Maximum accepted residual in archive: 1.517577785e-11, below specified 1e-7 acceptance threshold.
- All accepted rows meet |beta|<=0.3, |yaw rate|<=1.5 and max axle utilization <=1.
- 300 rows were sampled deterministically (25 per each of 12 groups; pandas random_state=7). The saved inputs were re-evaluated with the exact solver source. **Zero status mismatches**; maximum absolute differences: beta 7.92e-12; yaw 5.82e-11; utilization 3.28e-10; residual 1.61e-16; kinematic yaw 5.04e-11; relative deviation 1.67e-10.
- Both-side verified paired scenario records: 15,339 out of 18,000 (count across scenario/record combinations).

## Manuscript checks
- The v94 manuscript states 31,329 accepted roots; this matches the archived CSV.
- Its methods describe 3,000 paired records, three assumed vehicle configurations and two friction levels; the archive has the corresponding 36,000 side evaluations.
- The manuscript explicitly limits conclusions to hypothetical vehicle parameters, quasi-steady modeling, and numerical rather than experimental validation.

## Outstanding before claiming final scientific validation
1. This is **300-row numerical spot-reproduction**, not an independent recomputation of all 36,000 evaluations.
2. Original `real_vs_Test.csv` is not yet present in this GitHub directory; full raw-input-to-output reproducibility is not verified.
3. The manuscript refers to separate sensitivity, local stability, root-branch, and ReVStED audits; those claims have **not** been independently re-run or cross-checked in this QA.
4. No measured vehicle calibration, closed-loop tests, or transient maneuver-level physical safety validation has been established.
5. A full manuscript editorial/reference audit and repository release snapshot remain outstanding.

**Conclusion:** Archived independent Fiala output is internally consistent and matches the published solver on a stratified 300-row spot check. Do **not** label this a completed final scientific validation or exact recovery of the historical solver.
