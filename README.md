# Self-Supervised Vision Transformers for Unsupervised Object and Part Discovery

This project explores how **self-supervised Vision Transformers (ViTs)** can learn meaningful visual representations from unlabeled images and whether their learned features and attention can emerge as useful **object and part detectors**.

The project uses **DINOv2** for pretrained visual representation extraction and the **DESI Legacy Survey** galaxy images as the primary application dataset. Alongside this, a small DINO-style model is implemented from first principles to understand how self-supervised visual representations can be learned.

The overall work is divided into weekly stages, progressing from dataset and representation understanding to self-supervised training, attention analysis, unsupervised discovery, and downstream evaluation.

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

The Week 1 pipeline establishes a feature-extraction workflow from **DESI galaxy images through DINOv2 to CLS and patch-level representations**. These representations provide the foundation for the subsequent stages of the project.

## Week 2 — DINO from First Principles

Week 2 moves from using a pretrained model to implementing a small DINO-style self-supervised learning pipeline. The goal is to understand how a student–teacher architecture can learn visual representations without relying on class labels during training.

### Covered

- Loaded a subset of **5,000 CIFAR-10 training images**
- Implemented **multi-crop augmentation** using global and local image views
- Built a compact **Tiny Vision Transformer**
- Implemented the **student–teacher architecture**
- Implemented the DINO cross-view loss with teacher centering and sharpening
- Updated the teacher using an **exponential moving average (EMA)** of the student parameters
- Trained the model for **10 epochs**
- Extracted CLS embeddings from the trained teacher model
- Visualized learned representations using **PCA and t-SNE**
- Generated an attention map from a trained Transformer layer
- Included an exploratory ablation comparing EMA configurations

### Notebook

`DINO_Mini_Implementation_Refined.ipynb`

### Output

The Week 2 pipeline provides a compact implementation for studying how self-supervised representations can be learned from different views of the same image. Training-loss plots, embedding visualizations, attention maps, and the EMA comparison support an initial analysis of the learned features.

**Note:** This is a small educational implementation, not a reproduction of the full-scale DINO or DINOv2 training setup.

## Project Direction

The weekly stages build toward the broader objective of investigating whether self-supervised Vision Transformer representations and attention patterns can support **unsupervised object and part discovery** in astronomical images.

The current stages establish the foundation through pretrained feature extraction on DESI images and an educational implementation of DINO-style training. Subsequent stages will investigate attention behaviour, unsupervised discovery, and downstream evaluation.
