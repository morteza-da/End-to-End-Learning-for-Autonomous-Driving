# Runnable backbone extraction: reconstruction protocol

**Implementation:** `src/autonomous_driving/backbone_extraction.py` and `notebooks/03b_Runnable_Backbone_Features.ipynb`.

This stage uses **PyTorch + timm inside Jupyter Notebook**, not PyTorch instead of notebooks. The framework is a **new reconstruction decision**; the original research framework is not confirmed.

## Install

```bash
python -m pip install -e .
python -m pip install torch timm notebook
jupyter notebook
```

Package versions and hashes must be pinned after a verified run. timm model registry entries and pretrained checkpoints can differ across releases. The notebook prints installed candidates; the default requested names are `xception` and `resnext50_32x4d`. If a name is unavailable, explicitly inspect the installed registry and record the chosen replacement; do not silently substitute another architecture.

## Experimental structure

- A: camera → ResNeXt-50, LiDAR → Xception.
- B: camera → Xception, LiDAR → ResNeXt-50.
- Each branch is configured to return a **spatial feature map**, not classifier logits or globally pooled vectors.
- `torch.flatten(..., start_dim=1)` is applied independently to each branch.
- `torch.cat(..., dim=1)` concatenates per-frame feature vectors.
- Example freeze policy unfreezes the final top-level child only. **This is not a verified historical freeze boundary.**
- `pretrained=False` is a quick synthetic shape test, **not** the dissertation transfer-learning configuration. Use `pretrained=True` and confirm ImageNet checkpoint provenance before a real experiment.

## Verification status

Files committed to GitHub; pretrained weight loading, forward-pass execution, memory use and numerical training have **not** been verified in the target environment. Synthetic inputs cannot establish scientific performance. The full historical LiDAR 3D→2D projection remains unknown.

## Next steps

1. Verify compatible model names and download provenance.
2. Run synthetic feature extraction for both assignments and log shapes.
3. Pin versions and record early/upper layer boundaries.
4. Add temporal batching and memory-safe LSTM design without inserting an undocumented Dense bottleneck.
5. Complete the master archive `docs/RESEARCH_RECONSTRUCTION_MANUAL.md`.
