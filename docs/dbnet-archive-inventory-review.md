# DBNet archive-inventory review (historical local audit)

**Status:** documentary review only; not a fresh byte-level verification of all archives.

A user-provided local `archive_inventory.csv` contains 96 rows (56 train, 20 validation, 20 test). The inventory reports the following counts:

| Split | Low-resolution camera | LiDAR LAS | Full-HD camera | Behavior CSV files |
| --- | ---: | ---: | ---: | ---: |
| Train | 6,569 | 6,569 | 6,625 | 56 |
| Validation | 2,349 | 2,349 | 2,400 | 20 |
| Test | 2,376 | 2,376 | 2,400 | 0 |
| **Total** | **11,294** | **11,294** | **11,425** | **76** |

The inventory reports equal camera-small/LiDAR counts and exact frame-index overlap for each of the 96 drives. Its `small_only`, `lidar_only`, `non_contiguous_small`, and `other_files` fields contain no flagged entries. Twelve drives have more Full-HD frames than paired low-resolution/LiDAR frames:

- Train: 11 (24 paired/25 HD), 21 (112/120), 26 (110/120), 36 (104/120), 51 (107/120), 56 (112/120).
- Validation: 5 (111/120), 10 (109/120), 20 (89/120).
- Test: 10 (116/120), 15 (106/120), 20 (114/120).

There are **131 extra Full-HD images** relative to the low-resolution/LiDAR pairs (56 train, 51 validation, 24 test). These must **not** be interpreted as additional labeled or synchronized multimodal examples.

## Evidence boundary

These are transcribed, recomputed aggregate observations from the local inventory, **not independently reproduced experimental results**. The original RAR archives and CSV are intentionally not committed. File-list overlap does **not** establish image/LAS decode validity, LiDAR numerical validity, camera–LiDAR timestamp synchronization, steering/speed channel units, licensing, or model performance. A complete, independently repeatable byte-level audit and reviewed target provenance remain prerequisites to scientific claims.
