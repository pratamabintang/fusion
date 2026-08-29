# 08: YAML Configuration System & Shared CLI Utilities

**What to build:** The YAML experiment configuration files for the pure baseline and all three innovative models (`ms2fusion_baseline.yaml`, `ms_ssm.yaml`, `ic_ssm.yaml`, `combined.yaml`, `default.yaml`), along with configuration parsing/merging utilities and multispectral side-by-side visualization helpers.

**Blocked by:** None

**Status:** done

- [x] Create `configs/default.yaml` with base hyperparameters, dataset paths, and optimizer settings.
- [x] Create `configs/ms2fusion_baseline.yaml` configured with `fusion_type: 'ms2fusion'`.
- [x] Create `configs/ms_ssm.yaml` configured with `fusion_type: 'ms_ssm'`.
- [x] Create `configs/ic_ssm.yaml` configured with `fusion_type: 'ic_ssm'`.
- [x] Create `configs/combined.yaml` configured with `fusion_type: 'combined'`.
- [x] Implement `src/fusion/utils/config.py` with `load_config` that parses YAML and applies dictionary/CLI overrides.
- [x] Implement `src/fusion/utils/visualization.py` with `draw_detections` and `create_side_by_side_vis` rendering annotated bounding boxes, confidence scores, and modality tags.
- [x] Unit tests verify config loading, parameter overrides, and visualization canvas generation without errors.
