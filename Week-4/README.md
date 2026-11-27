# Week 4: Frozen ViT Segmentation Heads on DESI DR10 Galaxies

Pixel-level 3-class galaxy segmentation (`Background`, `Target Galaxy`, `Other Sources`) using frozen self-supervised Vision Transformer backbones (**DINOv2 ViT-S/14+reg** and **DINO ViT-S/8**) trained on `photutils` pseudo-labels across 3,000 DESI DR10 cutouts (`2,100` train / `300` val / `600` test; 1 broken mask auto-excluded).

## Test-Set Results (N = 600 Galaxies)

| Setup | Backbone | Input | Grid | mIoU | Target IoU | All-Sources IoU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive attention (Wk 3) | DINO ViT-S/8 | 224 | 28x28 | — | 0.0383 | 0.1451 |
| Linear | DINOv2 ViT-S/14+reg | 224 | 16x16 | 0.6136 | 0.4795 | — |
| Linear | DINOv2 ViT-S/14+reg | 448 | 32x32 | 0.7273 | 0.5995 | — |
| Linear | DINO ViT-S/8 | 224 | 28x28 | 0.7498 | 0.6482 | — |
| +ms (no TTA) | DINOv2 ViT-S/14+reg | 448 | 32x32 | 0.7902 | 0.6994 | — |
| **+ms (+TTA)** | **DINOv2 ViT-S/14+reg** | **448** | **32x32** | **0.8105** | **0.7292** | — |

## Directory Structure
- `Week4_Pseudo_Labels.ipynb` — DESI DR10 download, S/N normalization, `photutils` pseudo-labeling, QA, and hard-set selection.
- `Week4_Segmentation_Heads.ipynb` — Frozen ViT feature extraction, Linear & `+ms` training (FP16), 6-view TTA, and evaluation.
- `Week4_Report.pdf` — Full Week 4 report with ablation analysis and discussion.
- `data/splits.json` — Fixed 70/10/20 train/validation/test split.
- `results/pseudo_labels/` — 50-mask visual QA review sheets (`review_page_1.png`–`review_page_5.png`).
- `results/predictions/` — 20-galaxy crowded hard-set comparison galleries (`hard_set_1.png`–`hard_set_4.png`).
- `results/metrics/` — `comparison.csv` and per-epoch training histories.
