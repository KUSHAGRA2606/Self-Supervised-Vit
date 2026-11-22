# Week 5 real cutout labelling guide

Use the 160 × 160 PNGs in `data/annotation_images/`. Their orientation and colour stretch are fixed. The g/r/z bands are background-subtracted, divided by their local sky RMS, and mapped to blue/green/red using Lupton asinh with stretch 5 and Q 8. Zoom as needed and leave the display stretch fixed.

1. Create a Label Studio project with the interface in `data/label_studio.xml`. Import `data/primary_tasks.json` to load the primary PNGs listed in `data/real_selection.json` with their original filenames. Enable local file serving and set the document root to the absolute path of `Week5/data` before starting Label Studio.
2. Find the main galaxy nearest the centre. Draw one **Target galaxy** polygon along the boundary where its visible light merges into noise. Follow visible asymmetric structure and include contiguous arms. Keep gaps between separate sources.
3. Draw an **Other galaxy** or **Star** polygon for each distinct visible neighbour. Follow the visible boundary and use the edge of the frame for truncated objects. Skip noise and obvious artefacts.
4. In crowded fields, outline each object using the visible evidence. Resolve ambiguous ownership consistently. The rasterizer gives the target precedence where polygons overlap. Among other objects it gives precedence to the first polygon in export order.
5. Export JSON as `data/primary_annotations.json`. Import `data/repeat_tasks.json` into a separate project. Ask a second annotator to label the 20 repeat IDs without seeing the first annotations. If working alone, repeat after several days and set `AGREEMENT_KIND` to `"self-agreement"` in the benchmark notebook. Export that project as `data/repeat_annotations.json`.

Start Label Studio from the clone with local file serving limited to this data directory.

```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD/Week5/data" label-studio start
```

The benchmark converts the polygons into instance IDs and a three-class semantic mask. Background is 0, the target is 1, and other sources are 2. Save ambiguity notes separately. This is a held-out test set. Keep it out of training, threshold selection, and model selection.
