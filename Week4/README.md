# Week 4 — First segmentation models on frozen DINOv2

Week 4 turns DESI g/r/z flux cutouts into three-class segmentation maps. It compares the Week 3 attention union with Linear heads and a multi-layer +ms head trained on photutils pseudo-labels.

## Run in order

1. [Pseudo-label notebook](Week4_Pseudo_Labels.ipynb) downloads six pinned DESI Parquet shards, generates the fixed-stretch RGB and `.npy` masks, and displays a review gallery. [The split](data/splits.json) fixes 2,100 train, 300 validation, and 600 test object IDs. [The review exclusions](data/excluded_ids.json) and automatically rejected masks are excluded from head evaluation without reassigning any object. [Twenty hard cases](data/hard_set.json) are drawn from test.
2. [Segmentation notebook](Week4_Segmentation_Heads.ipynb) fits each head and selects checkpoints on validation mIoU. It writes the test comparison, predictions, learning curves and the hard-set gallery. Use a Colab GPU runtime, run from the repository clone, and keep caches and checkpoints on persistent storage.
3. [Week 4 report](Week4_Report.pdf) describes the label pipeline, architecture, and measured checks.

Background is 0, target is 1, and the other sources are 2. IoU uses pixel counts pooled over the relevant split. The binary attention union gets target and all-source IoUs. Its three-class mIoU is undefined. All test metrics in this week describe agreement with photutils pseudo-labels.

The [mask archive](results/pseudo_labels/desi_pseudo_labels.zip) contains all 3,000 arrays. Generated native cutouts, features, and model checkpoints stay local. The pseudo-label notebook recreates them and retrieves data from the pinned [DESI source](https://huggingface.co/datasets/MultimodalUniverse/legacysurvey/tree/112bddb46da031de2807ced8fe28fb79ed13b515/dr10_south_21).

The CPU pilot uses a small training and validation subset and scores the 20 hard cases. Its results, split, and gallery are separate from the full segmentation notebook outputs. The base DINOv2 checkpoint matches the Week 3 morphology experiment. Week 3's attention examples provide no segmentation labels for selecting a registers variant.
