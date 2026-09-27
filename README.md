# Self-Supervised Vision Transformers for Unsupervised Object and Part Discovery

This project explores how **self-supervised Vision Transformers (ViTs)** can learn meaningful visual representations from unlabeled images and whether their learned features and attention can emerge as useful **object and part detectors**.

The project uses **DINOv2** for self-supervised visual representation learning and the **DESI Legacy Survey** galaxy images as the primary dataset. The overall work is divided into weekly stages, progressing from dataset and representation understanding to attention analysis, unsupervised discovery, and downstream evaluation.

## Week 1 — DINOv2 & DESI Exploration

Week 1 establishes the foundation for the project by exploring the DESI dataset and understanding how a pretrained DINOv2 Vision Transformer represents an astronomical image.

### Covered

- Explored the **DESI Legacy Survey** dataset and its available fields
- Visualized the **g/r/i/z** optical bands
- Loaded the pretrained **DINOv2 ViT-S/14** model
- Preprocessed DESI galaxy images for DINOv2
- Ran DINOv2 inference on sample images
- Extracted **CLS tokens** for global image representations
- Extracted **patch tokens** for local spatial representations
- Inspected feature dimensions and embedding statistics
- Visualized the **16 × 16 patch grid** corresponding to the image tokens
- Performed an initial exploration of the learned embeddings

### Notebook

`Week1_DINOv2_DESI.ipynb`

### Output

The Week 1 pipeline establishes a basic feature-extraction workflow from **DESI galaxy images through DINOv2 to CLS and patch-level representations**. These representations provide the foundation for the subsequent weeks of the project.
