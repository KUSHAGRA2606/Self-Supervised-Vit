# Self-Supervised Vision Transformers for Unsupervised Object and Part Discovery

Investigating the potential of self-supervised Vision Transformers to automatically detect objects and structural features in unlabeled astronomical data.

Core Tech: DINOv2 | Data Source: DESI Legacy Survey

## Week 1 — DINOv2 & DESI Exploration
The first week establishes the baseline pipeline, focusing on data ingestion and extracting initial representations.

Milestones Achieved
-Data Processing: Loaded DESI galaxy images and visualized the multi-band (g/r/i/z) optical data.

-Model Deployment: Initialized the pretrained DINOv2 ViT-S/14 and established the image preprocessing workflow.

-Token Extraction: Successfully generated CLS tokens (global context) and patch tokens (local features) via model inference.

-Mapping & Analysis: Validated embedding dimensions and mapped the 16 × 16 spatial patch grid back to the source images.

### Notebook
`Week1_DINOv2_DESI.ipynb`

### Weekly Outcome
Delivered an end-to-end extraction pipeline that transforms raw DESI imagery into structured DINOv2 embeddings, setting the stage for advanced attention analysis next week.