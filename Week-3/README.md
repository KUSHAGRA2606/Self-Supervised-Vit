# Week 3: DINOv2 Representations and Attention on DESI

## Deliverables

- `Week3_Attention_Maps.ipynb`: DINO ViT-S/8 and DINOv2 ViT-S/14 attention, per-head 60% mass masks, and a register comparison. The executed notebook contains the Task A outputs.
- `Week3_Layerwise_Probe.ipynb`: gz10 layer-wise feature extraction, logistic-regression and MLP probes, metrics, confusion matrix, and t-SNE plots.
- `Week3_Report.md`: methods, observed Task A findings and figures, checklist, and the Task B report section.
- `results/attention_maps/`: raw attention arrays (`.npz`), full-resolution figures, and register peak-mass summary from the executed attention notebook.
- `results/probe_results/` and `results/tsne/`: probe metrics, confusion matrix, and visualizations. The notebook writes reusable `.npy` feature caches to `results/features/` when run.

## Task A results included

The report includes qualitative observations from ten cutouts. The 60% masks are fragmented in many examples and can include bright non-target sources. The measured mean maximum CLS-to-patch attention mass was 0.0710 without registers and 0.0916 with registers; in this sample, the register model did not reduce this concentration statistic. See the figures and caveats in `Week3_Report.md`.

## Running the notebooks

Use Colab with a T4 GPU. The attention notebook needs a Week 1 Legacy Survey cache, an RGB image folder, or access to the public Legacy Survey stream. The probe notebook streams gz10 in batches, saves the required CLS L9-L12 and L12 mean-patch feature arrays, and uses a stratified 80/20 split. Keep the `.npy` feature files and `feature_extraction_complete.json` for probe reruns.

## Submission notes

Probe metrics, the best-setting confusion matrix, per-class report, and both t-SNE plots are stored under `results/probe_results/` and `results/tsne/`. The large `.npy` feature cache was not included with the uploaded results; rerunning feature extraction will create it under `results/features/` for future probe runs.
