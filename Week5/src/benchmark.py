import numpy as np
import torch

from cutouts import native_griz, pseudo_label, semantic_mask
from heads import backbone, naive_attention, predict
from metrics import confusion, iou_summary


def score_dataset(name, paths, models, device, annotations=None):
    counts = {key: np.zeros((3, 3), dtype=np.int64) for key in ("photutils", *models)}
    target = np.zeros((2, 2), dtype=np.int64)
    sources = target.copy()
    failure = {"target_split": 0, "target_merged": 0, "missing_target": 0}
    attention = backbone("dino_vits8", device)
    for path in paths:
        with np.load(path) as sample:
            grz = native_griz(sample)[[0, 1, 3]]
            image = torch.from_numpy(sample["rgb"]).permute(2, 0, 1)[None]
            if annotations is None:
                instances, target_id = sample["instance"], int(sample["target"])
                truth = sample["semantic"]
            else:
                instances, target_id = annotations[path.stem]
                truth = semantic_mask(instances, target_id)
        found, classical, detected_target = pseudo_label(grz)
        counts["photutils"] += confusion(truth, classical)
        # Require ten overlapping pixels so boundary speckles do not count as objects.
        _, overlap = np.unique(found[(truth == 1) & (found > 0)], return_counts=True)
        failure["target_split"] += int((overlap >= 10).sum() > 1)
        failure["missing_target"] += int(np.count_nonzero((truth == 1) & (classical == 1)) < 10)
        if detected_target:
            _, overlap = np.unique(instances[(found == detected_target) & (truth == 2)],
                                   return_counts=True)
            failure["target_merged"] += int((overlap >= 10).any())
        union = naive_attention(attention, image, device)[0]
        target += confusion(truth == 1, union, classes=2)
        sources += confusion(truth > 0, union, classes=2)
        for method, (head, model, setup) in models.items():
            pred = predict(head, model, image, setup, device, tta=method == "ms")[0]
            counts[method] += confusion(truth, pred)
    print(name, "photutils diagnostics:", failure, "out of", len(paths))
    rows = [{"method": method, "dataset": name, **iou_summary(cm)}
            for method, cm in counts.items()]
    rows[0].update(failure, n_images=len(paths))
    rows.append({"method": "naive", "dataset": name,
                 "target_iou": iou_summary(target)["target_iou"],
                 "all_sources_iou": iou_summary(sources)["target_iou"]})
    return rows
