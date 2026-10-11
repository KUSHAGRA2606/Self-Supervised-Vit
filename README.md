# Self-Supervised Vision Transformers for Unsupervised Object and Part Discovery

Investigating the potential of self-supervised Vision Transformers to automatically detect objects and structural features in unlabeled astronomical data.

**Core Tech:** DINO / DINOv2 | **Data Source:** DESI Legacy Survey & CIFAR-10 (Mini-DINO Benchmark)

---

## Week 1 — DINOv2 & DESI Exploration
The first week establishes the baseline pipeline, focusing on data ingestion and extracting initial representations.

### Milestones Achieved
* **Data Processing:** Loaded DESI galaxy images and visualized the multi-band (g/r/i/z) optical data.
* **Model Deployment:** Initialized the pretrained DINOv2 ViT-S/14 and established the image preprocessing workflow.
* **Token Extraction:** Successfully generated `[CLS]` tokens (global context) and patch tokens (local features) via model inference.
* **Mapping & Analysis:** Validated embedding dimensions and mapped the $16 \times 16$ spatial patch grid back to the source images.

### Notebook
* `Week1_DINOv2_DESI.ipynb`

### Weekly Outcome
Delivered an end-to-end extraction pipeline that transforms raw DESI imagery into structured DINOv2 embeddings, setting the stage for advanced attention analysis.

---

## Week 2 — DINO from First Principles (Mini-DINO)
The second week focuses on building a complete self-supervised DINO training pipeline from scratch to understand how self-distillation learns semantic visual representations and object boundaries without labeled data.

### Milestones Achieved
* **Multi-Crop Augmentation Pipeline:** Implemented DINO multi-crop transformations on a 5,000-image CIFAR-10 subset, generating 2 global crops ($32 \times 32$) and 4 local crops ($16 \times 16$) with color jitter, random grayscale, Gaussian blur, and solarization to enforce local-to-global correspondence.
* **Student-Teacher TinyViT Architecture:** Built a custom `TinyViT` backbone ($4 \times 4$ patch size, 192 embedding dim, 6 layers, 3 attention heads) with bicubic positional encoding interpolation and a 3-layer weight-normalized projection head (1024-dim output).
* **Collapse Prevention & EMA Updates:** Implemented `DINOLoss` with teacher centering ($m = 0.9$) and softmax temperature sharpening ($\tau_t = 0.04, \tau_s = 0.1$), paired with an Exponential Moving Average (EMA) momentum teacher ($\lambda = 0.996$).
* **Ablation Study (With vs. Without EMA):** Demonstrated that EMA teacher updates drive steady convergence (loss dropping from `6.8147` to `6.5793` over 10 epochs), whereas freezing the teacher without EMA causes training to stagnate at `~6.7263`.
* **Embedding & Attention Visualization:** Projected 192-dim `[CLS]` embeddings via 2D t-SNE to verify unsupervised class clustering and extracted multi-head `[CLS]` attention heatmaps showing automatic foreground object segmentation.

### Deliverables & Directory Structure
* **Notebook:** `Week-2/DINO_Mini_Implementation.ipynb`
* **Report:** `Week-2/Week2_Report.pdf`
* **Visualizations:** `Week-2/results/` (Training & Ablation Loss Curve, t-SNE Embeddings, Multi-Head ViT Attention Maps)

### Weekly Outcome
Successfully trained a self-supervised Vision Transformer from scratch without labels, validating the mathematical mechanisms (EMA, centering, sharpening, and multi-crop views) that enable emergent object and part segmentation in ViT attention heads.