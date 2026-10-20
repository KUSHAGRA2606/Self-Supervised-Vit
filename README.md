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

**Week 3 execution and measured findings are pending.** The current report describes the methods and evaluation protocol; it does not claim a winning layer, measured morphology accuracy, or an observed advantage of registers.

The assignment cites approximately **67% accuracy for ViT-S/14** and **73% for ViT-L/14** with an MLP on CLS embeddings. These are reference values, not results obtained in this project. The current probe notebook evaluates **ViT-S/14**.

**Interpretation:** Attention maps and thresholded masks are qualitative diagnostics. Segmentation accuracy requires ground-truth masks and suitable metrics; attention alone does not establish object or part discovery.

## Running the Notebooks

1. Open the desired `.ipynb` file in **Google Colab**.
2. Select a **GPU runtime**, such as a T4.
3. Run the cells in order, including dependency installation where provided.
4. For Week 3, set `USE_DRIVE = True` to preserve caches and results across Colab sessions.
5. Download generated outputs before ending the runtime if using temporary storage.

Full Week 3 patch-token caching uses approximately **14 GB**, in addition to dataset and model storage. Setting `SAVE_PATCH_TOKENS = False` reduces storage while retaining the six probe feature sets, but omits the assignment's full patch-token cache. Use `MAX_IMAGES` only for a small smoke run; keep it at `None` for the full-catalog experiment.

## Project Direction

The current stages establish pretrained feature extraction, a small self-supervised training implementation, and tools for attention analysis and morphology evaluation.

The next step is to run the Week 3 experiments, inspect their outputs, and update the report with measured findings. Those findings will guide later work on **unsupervised object and part discovery**, particularly the challenges of blended sources, fragmented detections, and faint galaxy structure.
