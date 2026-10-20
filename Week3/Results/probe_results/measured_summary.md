# Week 3 measured summary
Dataset: 17736 images; seed 42; test images 3548.
Selected on validation: cls_L11 + Linear.
Selected-setting test accuracy: 69.05%.
Selected-setting test balanced accuracy: 65.25%.
Selected-setting test macro F1: 0.6560.
Best individual CLS layer on validation: cls_L11.
L12 CLS + MLP test accuracy: 66.07%.
Assignment reference: ViT-S/14 67%; ViT-L/14 73% (not reproduced here).
Reference values are from Week3.pdf; exact comparability depends on preprocessing,
split, training recipe, and checkpoint. This is a single-seed estimate.
Most frequent directed confusions (count, true class, predicted class): [(71, 6, 7), (70, 5, 7), (60, 7, 6), (54, 7, 5), (46, 0, 7)].

Add your own attention observations and inspect examples of confused classes before
claiming a visual reason. Class IDs are authoritative; verify semantic class names
against this dataset release. Do not interpret t-SNE separation as a test metric.
