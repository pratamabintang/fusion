# 10: Visual Demo & Multi-Architecture Ablation Benchmark Runner (demo.py & benchmark.py)

**What to build:** The interactive visual demonstration tool (`scripts/demo.py`) generating side-by-side visible and thermal annotated detection images, and the automated ablation benchmark runner (`scripts/benchmark.py`) comparing all 4 fusion architectures and synthesizing a Markdown comparison report.

**Blocked by:** 08 (YAML Configuration System & Shared CLI Utilities), 09 (Training & Evaluation CLI Tools)

**Status:** done

- [x] Implement `scripts/demo.py` accepting `--weights`, `--visible`, `--thermal`, `--data-dir`, `--split`, `--num-samples`, `--fusion-type`, `--conf-thres`, `--save-dir`.
- [x] Render side-by-side annotated images with visible and thermal bounding boxes saved to `runs/demo/<name>/`.
- [x] Implement `scripts/benchmark.py` running automated comparative evaluations across `ms2fusion`, `ms_ssm`, `ic_ssm`, and `combined`.
- [x] Measure and log parameter count, model size (MB), latency (ms/image), mAP@0.5, and mAP@0.5:0.95.
- [x] Generate structured Markdown summary report at `runs/benchmark/ablation_report.md`.
- [x] Integration tests verify CLI execution of `demo.py` producing output images and `benchmark.py` generating markdown reports.
