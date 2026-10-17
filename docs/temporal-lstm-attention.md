# LSTM → Attention → Dense implementation

**[RESEARCHER]** Per-frame camera and LiDAR backbone feature maps are Flattened, Concatenated without an intermediate Dense layer, and assembled into consecutive temporal sequences. LSTM is followed by Attention and Dense prediction of steering angle and speed simultaneously.

**[implementation]** `src/autonomous_driving/temporal_model.py` implements batch-first PyTorch LSTM, **single-score additive temporal attention**, and one 2-unit regression Dense layer. This attention and head design is an explicit implementation choice, **not verified as the original research implementation**. LSTM hidden size and sequence length are user-configurable; the notebook uses synthetic example values only.

**Tensor shapes:**
- Camera/LiDAR aligned image sequences: `[B,T,3,299,299]`.
- Each backbone produces spatial maps, Flatten each branch, concatenate: `[B,T,F]`.
- LSTM output: `[B,T,H]`.
- Additive attention weights: `[B,T]`, summing to 1 across T.
- Context: `[B,H]`.
- Dense output: `[B,2]` for steering degrees and speed km/h.

**DBNet citation:** Chen et al. (2018), *LiDAR-Video Driving Dataset: Learning Driving Policies Effectively*, CVPR. Original: https://github.com/driving-behavior/DBNet. See `CITATION_DBNet.bib`.

**Testing:** `python -m unittest discover -s tests -v`. Torch-dependent tests skip when PyTorch is not installed. Real-data training and full integration validation remain pending.

**Important resource constraint:** Direct Flatten without pooling may create enormous input size F; a large LSTM weight matrix is expected. Do not silently insert a Dense bottleneck or global pooling to solve memory pressure and still call it a faithful implementation. Log and disclose any experimental modifications.
