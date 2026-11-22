# Week 5 — Ground truth and evaluation benchmark

Week 5 adds DES-like BTK simulations, a held-out DESI annotation set, and one shared metrics module for subsequent experiments.

## Run in order

1. [Simulation notebook](Week5_BTK_Simulations.ipynb) downloads the BTK tutorial CatSim catalog and renders DES g/r/i/z stamps. It saves 3,600 train, 400 validation and 1,000 test blends as individual compressed archives. Each archive contains the float16 bands and their scale factor, RGB, exact instance and semantic masks under the stated ownership rule, target ID, centres and per-band noiseless in-stamp fluxes. The notebook compares domain statistics and plots frozen DINOv2 CLS embeddings.
2. [Real selection](data/real_selection.json) and [labelling guide](labelling_guide.md) define 100 held-out DESI cutouts and the 20-image repeat subset. [Primary tasks](data/primary_tasks.json), [repeat tasks](data/repeat_tasks.json) and the [Label Studio config](data/label_studio.xml) use the same fixed-display PNGs. Export both projects as specified in the guide.
3. [Benchmark notebook](Week5_Benchmark.ipynb) checks [shared metrics](src/metrics.py), imports the human annotations, retrains the Linear head on simulated labels, and writes the simulated and real benchmark table. It requires checkpoints from the Week 4 segmentation notebook. The agreement calculation runs once both human exports are present.
4. [Week 5 report](Week5_Report.pdf) describes the benchmark and records the measurements available so far.

Report per-class IoU and the mean of classes with a nonzero union. Use thresholded Hungarian assignment for instance matching. Binary attention scores are target and all-source IoU. Keep all test IDs out of fitting and calibration.

The PNGs and benchmark manifests are in Git. The [simulation archive checks](results/domain_gap/simulation_validation.json) record all 5,000 generated blends and the number of targets below the ownership threshold. Large generated simulation archives are recreated locally by the notebook, using the pinned [BTK catalog](https://github.com/LSSTDESC/BlendingToolKit/blob/6901e8ce5d2d3f19bf5e1b7463e26332fce816fd/data/input_catalog.fits).

The pilot domain check uses the DESI catalog PSF widths and BTK configuration, and uses photutils apertures to compare r-band S/N and target size. The configured DES PSF and noise are retained. Simulated test scores use the fixed simulated seed and include no real-set claims.
