import numpy as np
from scipy.optimize import linear_sum_assignment


def confusion(truth, prediction, classes=3):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape:
        raise ValueError("Truth and prediction must have the same shape.")
    for mask in (truth, prediction):
        if mask.dtype.kind not in "biu" or np.any((mask < 0) | (mask >= classes)):
            raise ValueError("Masks must contain integer class IDs in range.")
    codes = classes * truth.astype(np.int64) + prediction
    return np.bincount(codes.ravel(), minlength=classes**2).reshape(classes, classes)


def iou_summary(counts):
    counts = np.asarray(counts)
    intersection = counts.diagonal()
    union = counts.sum(0) + counts.sum(1) - intersection
    scores = np.divide(intersection, union, out=np.full(len(union), np.nan), where=union > 0)
    return {
        "class_iou": scores.tolist(),
        "miou": float(np.mean(scores[union > 0])) if union.any() else np.nan,
        "target_iou": float(scores[1]),
    }


def binary_iou(truth, prediction):
    return iou_summary(confusion(truth, prediction, classes=2))["target_iou"]


def match_instances(truth, prediction, threshold=0.5):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or not 0 < threshold <= 1:
        raise ValueError("Use matching shapes and an IoU threshold in (0, 1].")
    for mask in (truth, prediction):
        if mask.dtype.kind not in "iu" or np.any(mask < 0):
            raise ValueError("Instance IDs must be nonnegative integers.")
    true_ids = np.unique(truth[truth > 0])
    pred_ids = np.unique(prediction[prediction > 0])
    scores = np.empty((len(true_ids), len(pred_ids)))
    for row, true_id in enumerate(true_ids):
        for col, pred_id in enumerate(pred_ids):
            scores[row, col] = binary_iou(truth == true_id, prediction == pred_id)

    # First maximize the number of eligible pairs, then their total IoU.
    eligible = scores >= threshold
    reward = np.where(eligible, min(scores.shape) + 1 + scores, 0)
    rows, cols = linear_sum_assignment(reward, maximize=True)
    matches = [(int(true_ids[r]), int(pred_ids[c]), float(scores[r, c]))
               for r, c in zip(rows, cols) if eligible[r, c]]
    return {"matches": matches, "fp": len(pred_ids) - len(matches),
            "fn": len(true_ids) - len(matches)}
