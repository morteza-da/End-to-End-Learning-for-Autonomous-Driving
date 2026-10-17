# DBNet label channel evidence

## Primary implementation evidence

Upstream repository: https://github.com/driving-behavior/DBNet

- `provider.py`, `Provider.read_from`: `Y_train2.append(float(line.split(",")[0]) * scipy.pi / 180)`. Raw `behavior.csv` channel **0** is interpreted as a steering angle in **degrees** before conversion to radians.
- `provider.py`, `Provider.read_from`: `Y_train1.append((float(line.split(",")[1]) - 20) / 20)`. Raw channel **1** is vehicle speed before the original model's normalization.
- `data/README.md`: the *ith* CSV line corresponds to frame `i-1.jpg` and point cloud `i-1.las`. Frames within each drive are continuous and time-ordered. These statements do not establish physical sensor timestamp synchronization.

## Unit corroboration and limitations

The original upstream provider code does **not explicitly label the physical unit of speed**. A subsequent publication using DBNet explicitly describes steering in degrees and speed in km/h: https://pmc.ncbi.nlm.nih.gov/articles/PMC9525277/ . Therefore `steering_channel=0`, `speed_channel=1`, `steering_unit=deg`, `speed_unit=km/h` is a source-supported mapping, with speed units corroborated by a secondary publication rather than an explicit upstream code comment. Inspect actual data and authoritative dataset documentation before running training.

## Source and scope

Primary source: https://github.com/driving-behavior/DBNet/blob/master/provider.py
Dataset structure: https://github.com/driving-behavior/DBNet/blob/master/data/README.md
Secondary unit corroboration: https://pmc.ncbi.nlm.nih.gov/articles/PMC9525277/

No raw DBNet files, actual experiment outcomes, or physical timestamp calibration are provided by these references.
