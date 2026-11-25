# Week 4 Report: First Segmentation Models on Frozen DINOv2

## Goal and method

This experiment tests whether frozen DINO features can produce pixel-level masks for DESI galaxies. Each 160 x 160 cutout receives three labels: background (0), the central or nearest detected galaxy (1), and other sources (2). These are photutils pseudo-labels, not expert ground truth, so all scores measure agreement with the source-detection pipeline.

We streamed 3,000 g/r/z cutouts, estimated background and RMS, converted to the Week 1 asinh RGB representation, and detected/deblended sources after Gaussian smoothing at 1.5 sigma with a 10-pixel minimum area. All 3,000 had status `ok`. The fixed seed-42 split contains 2,100 train, 300 validation, and 600 test images.

The backbones were frozen and kept in evaluation mode. A trainable BatchNorm plus 1 x 1 convolution head predicted the three classes from patch tokens. The linear setup used the final layer; the +ms setup concatenated the final four DINOv2 layers and used test-time augmentation at 336, 448, and 560 px, with horizontal flips. Checkpoints were chosen on validation foreground mean IoU. Final metrics below are on the held-out test split.

## Test results

| Setup | Backbone / input / patch grid | mIoU | All-source IoU | Target-galaxy IoU |
|---|---|---:|---:|---:|
| Week 3 naive attention | DINO ViT-S/8 / 224 px / 28x28 | N/A | 24.48% | 5.52% |
| Linear DINOv2 ViT-S/14, 224 px | dinov2_s14 / 224 px / 16x16 | 61.24% | N/A | 54.58% |
| Linear DINOv2 ViT-S/14, 448 px | dinov2_s14 / 448 px / 32x32 | 64.86% | N/A | 58.45% |
| Linear DINO ViT-S/8, 224 px | dino_s8 / 224 px / 28x28 | 66.56% | N/A | 59.95% |
| +ms DINOv2 ViT-S/14, 448 px + TTA | DINOv2 ViT-S/14 / 448 px / 32x32 | 68.30% | N/A | 59.28% |

The attention baseline has no class output or three-class mIoU; its two binary IoUs are reported separately. The best supervised mIoU is 68.30% from +ms with TTA. The best target-galaxy IoU is 59.95% from the finer-patch DINO ViT-S/8 linear head.

### Class-wise results for the validation-selected +ms/TTA model

| Class | IoU |
|---|---:|
| Background | 93.10% |
| Target Galaxy | 59.28% |
| Other Sources | 52.52% |

### Confusion matrix

Pixel counts for the validation-selected +ms/TTA model (rows are pseudo-label classes; columns are predicted classes):

| Actual \ Predicted | Background | Target | Other |
|---|---:|---:|---:|
| Background | 13,045,027 | 109,461 | 228,342 |
| Target | 8,461 | 269,174 | 7,587 |
| Other | 620,016 | 59,369 | 1,012,563 |

The main class confusion is other sources predicted as background (620,016 pixels). Target pixels are usually retained as target (269,174 correct, 16,048 assigned to another class). This is consistent with the smaller IoU for other sources than for target galaxies.

## Ablations and interpretation

- Increasing DINOv2 input from 224 to 448 px raises mIoU from 61.24% to 64.86% (+3.62 percentage points) and target IoU from 54.58% to 58.45% (+3.87 points). A denser 32 x 32 patch grid helps resolve small structures.
- The DINO ViT-S/8 linear head reaches 66.56% mIoU and 59.95% target IoU, beating DINOv2 at 224 px by 5.32 and 5.37 points respectively. This comparison changes both patch size and backbone, so it is not a pure patch-size ablation.
- +ms with TTA reaches the highest mIoU, 68.30%, improving 3.44 points over DINOv2 linear at 448 px. Its target IoU (59.28%) is slightly below the DINO ViT-S/8 result; the larger gain is for the other-sources class. The +ms comparison combines multiple layers and TTA.
- The trained heads improve target isolation over Week 3 naive attention: the best target IoU is 59.95% versus 5.52%, a gain of 54.43 percentage points. Attention reaches 24.48% IoU against all sources, but cannot distinguish the target galaxy from neighbors.

## Answers to the required questions

1. **How much better is a trained linear head than naive attention at isolating the target?** The best trained setup reaches 59.95% target IoU versus 5.52% for naive attention. The attention mask often activates unrelated bright sources and has no semantic class assignment.
2. **Which matters more: multiscale features or a finer patch grid?** Both help in these runs. Larger input improves the DINOv2 linear result; the +ms/TTA setup has the best mIoU; and DINO ViT-S/8 obtains the best target IoU. Because these comparisons also change scale, backbone, or TTA, they do not isolate one factor completely.
3. **Does the head fix fragmentation or reproduce pseudo-label errors?** The heads learn the pseudo-label boundaries. The gallery shows smoother maps than the coarse Week 3 attention threshold, but the target and other-source boundaries remain constrained by photutils labels. A head trained on fragmented labels can reproduce those fragments; these metrics cannot show that it corrected them astronomically.
4. **What are the limits of pseudo-labels, and what labels would help?** Source detection can miss faint outskirts, fragment extended galaxies, or merge close objects. Expert-reviewed masks with consistent target identity and explicit blend boundaries would provide stronger supervision.
5. **What should improve next: labels, features, or the head?** Improve and validate labels first, especially for blends and faint outskirts. The measured heads already separate target and background reasonably well against the current labels; better labels are needed to tell whether the remaining error is from features or from inherited pseudo-label flaws.

## Qualitative review and limitations

The 20 held-out hard examples were reviewed across the saved gallery pages and retained as a challenge set because they show neighboring detections, blends, or substantial other-source contamination. The gallery contains DESI RGB, photutils labels, Week 3 attention, linear DINOv2, and +ms/TTA predictions. Reviewed failure patterns include fragmented extended detections, faint outskirts omitted from labels, and blends or bright neighbors that complicate source assignment.

DINO/DINOv2 natural-image pretraining transfers useful spatial information: frozen patch tokens support a lightweight segmentation head with substantially better target IoU than naive attention. Transfer remains imperfect because astronomical sources and noise differ from ImageNet images, the RGB rendering is noisy, and pseudo-labels are not ground truth. Results should be interpreted as agreement with photutils, not absolute segmentation accuracy.

## Reproducibility and deliverables

The split in `data/splits.json` records the exact seed-42 index partition and ID-order digest recovered from the completed feature-cache metadata. Large reproducible RGB/features caches and model checkpoints are excluded from Git; masks, test predictions, metrics, confusion matrix, and galleries are retained.
