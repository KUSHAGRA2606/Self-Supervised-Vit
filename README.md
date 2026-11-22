# Self-Supervised Vision Transformers for Unsupervised Object and Part Discovery

This project explores how **self-supervised Vision Transformers (ViTs)** learn visual representations from unlabeled images and whether their features and attention patterns can support **unsupervised object and part discovery**.

The primary application is galaxy imaging from the **DESI Legacy Survey**, using pretrained **DINO and DINOv2** models. A compact DINO-style model is also implemented from first principles on CIFAR-10 to understand the underlying self-supervised training process.

The work progresses through weekly stages: dataset exploration, representation extraction, self-supervised training, attention analysis, and downstream evaluation.

## Week 1 — DINOv2 & DESI Exploration

Week 1 establishes the foundation by exploring the DESI dataset and understanding how a pretrained DINOv2 model represents astronomical images.

### Covered

- Explored the **DESI Legacy Survey** dataset and its available fields
- Visualized the **g/r/i/z** optical bands
- Loaded pretrained **DINOv2 ViT-S/14**
- Preprocessed galaxy images and ran inference
- Extracted **CLS tokens** for global image representations
- Extracted **patch tokens** for local spatial representations
- Inspected feature dimensions and embedding statistics
- Visualized the **16 × 16 patch grid** at an input resolution of 224 × 224
- Performed an initial exploration of the extracted embeddings

### Files

- Notebook: `Week1_DINOv2_DESI.ipynb`
- Report: `Week1_Report.pdf`

### Output

The Week 1 pipeline establishes a workflow from **DESI galaxy images → DINOv2 → CLS and patch representations**. These features provide the foundation for subsequent analysis.

## Week 2 — DINO from First Principles

Week 2 moves from using a pretrained model to implementing a compact DINO-style self-supervised training pipeline. It explores how a student–teacher architecture learns from different views of an image without using class labels in the training loss.

### Covered

- Selected **5,000 CIFAR-10 training images** using a fixed random seed
- Implemented **multi-crop augmentation** with two global and two local views
- Built a compact **Tiny Vision Transformer**
- Implemented the **student–teacher architecture**
- Implemented the DINO cross-view loss with teacher centering and sharpening
- Updated the teacher using an **exponential moving average (EMA)** of student parameters
- Configured training for **10 epochs**
- Added teacher CLS embedding extraction for **1,000 test images**
- Implemented **PCA and t-SNE** visualizations
- Added CLS attention visualization from the trained teacher's final Transformer layer
- Included an exploratory comparison of an EMA-updated teacher and a fixed teacher
- Added model checkpoint saving and reloading

### Files

- Notebook: `DINO_Mini_Implementation.ipynb`
- Report: `Week2_Report.pdf`

### Output

The notebook implements **CIFAR-10 images → multiple crops → student and teacher TinyViT → DINO loss and EMA → embeddings and visualizations**.

Running it produces training-loss curves, embedding plots, an attention overlay, checkpoints, and an EMA ablation. The fixed-teacher comparison is exploratory; training loss alone does not establish representation quality.

**Note:** This is an educational implementation, not a reproduction of full-scale DINO or DINOv2 training.

## Week 3 — Attention Maps & Layer-Wise Morphology Probing

Week 3 investigates two questions: whether pretrained attention provides a useful galaxy segmentation signal, and how well frozen DINOv2 features encode **galaxy morphology**.

### Task A — Attention Maps & Naive Segmentation

- Load pretrained **DINO ViT-S/8**, **DINOv2 ViT-S/14**, and **DINOv2 ViT-S/14 with registers**
- Inspect representative DESI galaxy cutouts, including isolated sources, blends, foreground stars, and extended objects
- Extract final-layer **CLS-to-patch attention** for each head
- Compare the **28 × 28** patch grid of DINO ViT-S/8 with the **16 × 16** grid of DINOv2 ViT-S/14
- Construct per-head masks using the smallest set of patches containing at least **60% of patch attention mass**
- Visualize RGB cutouts, individual attention heads, and the union of head masks
- Save attention arrays, masks, figures, and concentration diagnostics
- Compare attention artifacts with and without register tokens

The default input uses GZ10 DESI RGB cutouts. The notebook also accepts raw **g/r/z arrays** or a folder of RGB images.

### Task B — Layer-Wise Features & Galaxy Zoo Probes

- Load the **MultimodalUniverse GZ10** dataset and inspect its ten-class distribution
- Create a reproducible, **stratified 80/20 development/test split**
- Use a separate validation partition within the development set for model selection
- Extract frozen DINOv2 ViT-S/14 representations from **layers 9–12**
- Cache CLS vectors, final-layer mean-patch features, and optionally all patch tokens
- Compare **six feature sets**:
  - CLS tokens from layers 9, 10, 11, and 12 individually: **384 dimensions each**
  - Mean-pooled patch tokens from layer 12: **384 dimensions**
  - Concatenated CLS tokens from layers 9–12: **1,536 dimensions**
- Train a **logistic-regression linear probe** and a **one-hidden-layer MLP** for each feature set
- Train the MLP with **256 hidden units, ReLU, Adam, 30 epochs, and batch size 64**
- Fit feature standardization on training data only
- Report accuracy, balanced accuracy, macro F1, and per-class performance
- Generate confusion matrices and **t-SNE** comparisons of the final layer and the best individual CLS layer

### Files

- Attention notebook: `Week3_Attention_Maps.ipynb`
- Probe notebook: `Week3_Layerwise_Probe.ipynb`
- Report: `Week3_Report.pdf`

### Outputs and Status

The notebooks implement the attention and probing workflows. Running them generates:

- Per-head attention maps and thresholded masks
- Feature caches and reproducible split indices
- Validation and test results for all feature/probe combinations
- Per-class metrics and confusion matrices
- t-SNE plots and a measured-results summary

**The committed Week 3 [probe results](Week3/Results/probe_results/measured_summary.md) select the layer 11 CLS Linear probe on validation. On the 3,548-image test split it measured 69.05% accuracy, 65.25% balanced accuracy, and 0.656 macro F1. The attention visualizations and arrays are in [Week 3 attention results](Week3/Results/attention_results/).

The assignment cites approximately **67% accuracy for ViT-S/14** and **73% for ViT-L/14** with an MLP on CLS embeddings. These are reference values, not results obtained in this project. The current probe notebook evaluates **ViT-S/14**.

**Interpretation:** Attention maps and thresholded masks are qualitative diagnostics. Segmentation accuracy requires ground-truth masks and suitable metrics; attention alone does not establish object or part discovery.

## Week 4 — First Segmentation Models

Week 4 uses a frozen 70/10/20 split of 3,000 DESI cutouts and creates three-class photutils masks. The segmentation notebook compares the Week 3 attention baseline, Linear DINOv2 heads at 224 and 448, Linear DINO ViT-S/8, and +ms with and without test-time augmentation. Checkpoints are selected on validation mIoU.

- [Week 4 run order and outputs](Week4/README.md)
- [Pseudo-label notebook](Week4/Week4_Pseudo_Labels.ipynb)
- [Segmentation notebook](Week4/Week4_Segmentation_Heads.ipynb)
- [Week 4 report](Week4/Week4_Report.pdf)

In the 20-image CPU pilot, target IoU against pseudo-labels was 0.085 for attention, 0.192 for Linear at 224, and 0.418 for +ms with TTA. The full trained-head comparison is pending.

## Week 5 — Ground Truth and Evaluation Benchmark

Week 5 adds BTK DES-like blends with simulated ownership and fluxes, a fixed 100-image real annotation set, and shared IoU and Hungarian instance matching. The benchmark evaluates photutils, attention, the Week 4 heads, and a Linear head trained on simulated data.

- [Week 5 run order and outputs](Week5/README.md)
- [BTK simulation notebook](Week5/Week5_BTK_Simulations.ipynb)
- [Real annotation guide](Week5/labelling_guide.md)
- [Benchmark notebook](Week5/Week5_Benchmark.ipynb)
- [Week 5 report](Week5/Week5_Report.pdf)

Photutils measured 0.461 mIoU and 0.314 target IoU on the 1,000 simulated test images. The real cutouts are ready for primary and repeat annotation. Real-set scores and agreement depend on those exports.

## Running the Notebooks

1. Open the desired `.ipynb` file in **Google Colab**.
2. Select a **GPU runtime**, such as a T4.
3. Run the cells in order, including dependency installation where provided.
4. For Week 3, set `USE_DRIVE = True` to preserve caches and results across Colab sessions.
5. For Weeks 4 and 5, clone the repository in Colab, set the working directory to the clone, and follow each week’s run order.
6. Download generated outputs before ending the runtime if using temporary storage.

Full Week 3 patch-token caching uses approximately **14 GB**, in addition to dataset and model storage. Setting `SAVE_PATCH_TOKENS = False` reduces storage while retaining the six probe feature sets, but omits the assignment's full patch-token cache. Use `MAX_IMAGES` only for a small smoke run; keep it at `None` for the full-catalog experiment.

## Project Direction

The current stages establish pretrained feature extraction, a small self-supervised training implementation, and tools for attention analysis and morphology evaluation.

The next measurement is the full segmentation-head comparison against the fixed Week 4 test split and the Week 5 simulated and annotated test sets. Those results will guide later work on blended sources, fragmented detections, and faint galaxy structure.
