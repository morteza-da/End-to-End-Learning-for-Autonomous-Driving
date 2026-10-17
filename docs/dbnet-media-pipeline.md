# Decoding the original DBNet sensor formats

**[SOURCE]** DBNet-2018's original layout provides camera JPEG files and LAS LiDAR point clouds, with matching numeric stems. Earlier exploratory checks decoded one 200×66 JPEG and one LAS 1.2 point-format-3 file containing 16,384 points per split. These observations are not a complete integrity audit.

**[implementation]** The new `dbnet_media.py` decodes JPEG as RGB and resizes it to 299×299 (the researcher's confirmed backbone input dimensions). It uses Pillow's bilinear interpolation, which is a newly chosen implementation detail. LiDAR decoding uses `laspy`, which applies LAS scale and offset to obtain real-world XYZ. The subsequent spherical range-image step is implemented in [LiDAR range-image processing](lidar-range-image.md). This angular projection is not a camera-view projection.

Install optional data-reading packages: `python -m pip install numpy Pillow laspy`. Then, with the project installed editable:

```bash
python scripts/inspect_dbnet_pair.py data/dbnet-2018/train/1 0
```

The command verifies one actual JPEG/LAS pair and prints image dimensions, point count and **raw** behavior CSV fields. It does not publish images or point clouds, assume label units, or infer sensor calibration.

**[OPEN]** Exact projection from 3D LiDAR coordinates into 299×299 image pixel locations; exact label semantics; full integrity audit; previous camera interpolation/normalization; and full real-data training.

**Citation:** Chen et al. (2018), *LiDAR-Video Driving Dataset: Learning Driving Policies Effectively*, CVPR. https://github.com/driving-behavior/DBNet; `CITATION_DBNet.bib`.
