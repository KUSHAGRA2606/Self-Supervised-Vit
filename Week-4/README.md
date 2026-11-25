# Week 4 - First Segmentation Models on Frozen DINOv2

This folder contains the Week 4 pseudo-label pipeline, segmentation experiments, measured results, and final report. The completed evaluation uses 3,000 DESI/Legacy Survey cutouts and a fixed seed-42 70/10/20 split.

## Deliverables

- `Week4_Pseudo_Labels.ipynb` - normalization, photutils source detection/deblending, 3-class masks, 50-image QC gallery, and the reviewed 20-image hard set.
- `Week4_Segmentation_Heads.ipynb` - frozen-backbone linear and +ms heads, ablations, TTA, test metrics, confusion matrix, and hard-set gallery.
- `Week4_Report.md` and `Week4_Report.pdf` - methods, test results, class confusion, ablation interpretation, limitations, and answers to all five report questions.
- `data/splits.json` - fixed seed-42 split indices and image-order digest.
- `results/pseudo_labels/` - saved masks, QC galleries, class summary, and reviewed hard-set IDs.
- `results/predictions/` - test predictions, model checkpoints, and five hard-set gallery pages.
- `results/metrics/` - per-class IoUs, comparison table, confusion matrix, and final run metadata.
- `src/` - reusable implementation used by the notebooks.

## Final run

`results/metrics/run_status.json` reports `complete`: 3,000 valid masks and 600 held-out test cutouts. Every comparison row reports 600 test images. The completed DINOv2 ViT-S/14 feature-cache metadata supplied the SHA-256 of the ordered image IDs; the exact split indices were reconstructed from the notebook's seed-42 `default_rng` permutation and checked against the 20 hard-set test indices.

The hard-set candidate gallery was visually reviewed. The 20 retained cases show blends, neighboring sources, and contamination. Pseudo-label failure patterns include fragmented extended sources, missed faint outskirts, and source blending. The stored visual-review notes and report describe these observations.

## Key results

| Setup | mIoU | Target-galaxy IoU |
|---|---:|---:|
| Week 3 naive attention, DINO ViT-S/8 | N/A | 5.52% |
| Linear DINOv2 ViT-S/14, 224 px | 61.24% | 54.58% |
| Linear DINOv2 ViT-S/14, 448 px | 64.86% | 58.45% |
| Linear DINO ViT-S/8, 224 px | 66.56% | 59.95% |
| +ms DINOv2 ViT-S/14, 448 px + TTA | 68.30% | 59.28% |

Scores are agreement with photutils pseudo-labels, not human-ground-truth accuracy. The naive attention baseline is classless, so its all-source IoU (24.48%) and target-only IoU are reported separately.

## Run instructions

1. Open both notebooks from this folder in Colab or Kaggle. Use a T4 GPU for `Week4_Segmentation_Heads.ipynb`.
2. The pseudo-label notebook streams data and writes `results/pseudo_labels/masks.npy`, galleries, image IDs, RGB cache, QC CSV, and splits. Review the 50-mask gallery before accepting or changing the saved exclusions/hard set.
3. The segmentation notebook trains/evaluates the frozen-backbone heads and writes metrics, predictions, and galleries. Its final cells create both reports. The stored final run already completed; rerunning training is unnecessary unless reproducing the experiments.

Large reproducible caches (`data/rgb_images.npy`, `data/image_ids.npy`, and the 224-px feature cache) are excluded by `.gitignore`. Model checkpoints are also ignored. The compact split, pseudo-label masks, test predictions, metrics, and figures are retained for the assignment.

## Repository layout

```text
Week-4/
  README.md
  Week4_Pseudo_Labels.ipynb
  Week4_Segmentation_Heads.ipynb
  Week4_Report.md
  Week4_Report.pdf
  data/
    README.md
    splits.json
  results/
    pseudo_labels/
    predictions/
    metrics/
  src/
```
