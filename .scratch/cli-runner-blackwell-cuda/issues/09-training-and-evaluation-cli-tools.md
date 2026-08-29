# 09: Training & Evaluation CLI Tools (train.py & eval.py)

**What to build:** The high-level command-line entrypoints for multi-epoch multispectral detector training (`scripts/train.py`) and standalone evaluation (`scripts/eval.py`) supporting independent model execution across baseline and innovative architectures.

**Blocked by:** 08 (YAML Configuration System & Shared CLI Utilities)

**Status:** done

- [x] Implement `scripts/train.py` with CLI arguments (`--config`, `--fusion-type`, `--epochs`, `--batch-size`, `--lr`, `--device`, `--amp`, `--data-dir`, `--name`, `--resume`).
- [x] Support independent training of pure baseline and innovative models via `--fusion-type {ms2fusion, ms_ssm, ic_ssm, combined}`.
- [x] Save best and last checkpoints to `runs/train/<name>/weights/{best.pt, last.pt}` along with `results.csv` and `summary.json`.
- [x] Implement `scripts/eval.py` with CLI arguments (`--weights`, `--fusion-type`, `--data-dir`, `--split`, `--batch-size`, `--device`, `--conf-thres`, `--iou-thres`).
- [x] Compute and display mAP@0.5, mAP@0.5:0.95, Precision, Recall, and inference latency in formatted CLI output.
- [x] Integration tests verify CLI execution of `train.py` and `eval.py` in test/dry-run mode without crashing.
