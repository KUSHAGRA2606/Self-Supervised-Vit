# Week 4 split metadata

`splits.json` contains the fixed seed-42 partition used by the completed run: 2,100 train, 300 validation, and 600 test indices over 3,000 cutouts. The ID-order SHA-256 was recovered from the completed ViT-S/14 feature-cache completion marker; the partition was regenerated with the notebook's `numpy.random.default_rng(42).permutation(3000)` implementation. Large RGB and ID arrays are reproducible caches and are intentionally excluded from Git.
