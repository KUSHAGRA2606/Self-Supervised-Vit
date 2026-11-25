"""Frozen-backbone segmentation experiments for the Week 4 DESI project."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from week4_data import IMAGENET_MEAN, IMAGENET_STD

FEATURE_PREPROCESS_VERSION = "week4_rms_asinh_v3_ivar_authoritative"


class LinearSegmentationHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(in_channels), nn.Conv2d(in_channels, 3, kernel_size=1))

    def forward(self, features):
        return self.net(features)


def load_backbone(kind: str, device: torch.device):
    if kind in ("dinov2_s14", "dinov2_s14_reg"):
        name = "dinov2_vits14_reg" if kind.endswith("_reg") else "dinov2_vits14"
        model = torch.hub.load("facebookresearch/dinov2", name)
    elif kind == "dino_s8":
        model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
    else:
        raise ValueError(f"Unknown backbone: {kind}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def make_input(rgb_batch: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    """uint8 NHWC -> ImageNet-normalized NCHW at the requested model size."""
    x = torch.from_numpy(np.asarray(rgb_batch).copy()).to(device=device, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2).div_(255.0)
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.as_tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x.clamp(0, 1) - mean) / std


def _token_grid(tokens: torch.Tensor, size: int, patch_size: int) -> torch.Tensor:
    if tokens.ndim == 4:
        return tokens
    if tokens.ndim != 3:
        raise ValueError(f"Expected patch tokens [B,N,C] or grid [B,C,H,W], got {tokens.shape}")
    grid = int(round(math.sqrt(tokens.shape[1])))
    if grid * grid != tokens.shape[1] or grid != size // patch_size:
        raise ValueError(f"Patch tokens {tokens.shape} do not match {size}px grid")
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid, grid)


def extract_features(model, x: torch.Tensor, kind: str, mode: str, input_size: int) -> torch.Tensor:
    """Extract frozen backbone features; mode is last or multiscale."""
    # no_grad (rather than inference_mode) returns ordinary tensors that can be
    # consumed by the trainable segmentation head's autograd graph.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=x.is_cuda):
        if kind.startswith("dinov2"):
            n = 4 if mode == "multiscale" else 1
            outputs = model.get_intermediate_layers(x, n=n, reshape=True)
            if not isinstance(outputs, (tuple, list)):
                outputs = [outputs]
            features = torch.cat(list(outputs), dim=1) if mode == "multiscale" else outputs[-1]
        else:
            if mode != "last":
                raise ValueError("Multiscale is only defined for DINOv2 in this assignment")
            outputs = model.get_intermediate_layers(x, n=1)
            patch_tokens = outputs[-1]
            if patch_tokens.ndim == 3:
                expected_patches = (input_size // 8) ** 2
                if patch_tokens.shape[1] == expected_patches + 1:
                    patch_tokens = patch_tokens[:, 1:]  # Official DINO returns CLS + patches.
            features = _token_grid(patch_tokens, input_size, 8)
    return features.float()


def load_arrays(root: str | Path):
    root = Path(root)
    images = np.load(root / "data" / "rgb_images.npy", mmap_mode="r")
    masks = np.load(root / "results" / "pseudo_labels" / "masks.npy", mmap_mode="r")
    ids = np.load(root / "data" / "image_ids.npy", mmap_mode="r").astype(str)
    split = json.loads((root / "data" / "splits.json").read_text(encoding="utf-8"))
    valid_file = root / "data" / "valid_indices.npy"
    valid = np.load(valid_file) if valid_file.exists() else np.arange(len(ids), dtype=np.int32)
    valid_set = set(valid.tolist())
    parts = {}
    for name in ("train", "validation", "test"):
        parts[name] = np.asarray([i for i in split[f"{name}_indices"] if i in valid_set], dtype=np.int32)
    if not len(parts["train"]) or not len(parts["validation"]) or not len(parts["test"]):
        raise ValueError("One or more splits are empty after excluding reviewed masks")
    return images, masks, ids, parts


def estimate_class_weights(masks, train_indices: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.zeros(3, dtype=np.int64)
    for i in train_indices:
        counts += np.bincount(np.asarray(masks[i], dtype=np.uint8).ravel(), minlength=3)[:3]
    # Inverse-square-root weights temper the severe foreground/background imbalance.
    freq = counts / max(1, counts.sum())
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-8))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _batch(images, masks, indices, input_size, device):
    rgb = np.stack([images[int(i)] for i in indices])
    x = make_input(rgb, input_size, device)
    y = None if masks is None else torch.from_numpy(
        np.stack([masks[int(i)] for i in indices]).astype(np.int64, copy=False).copy()
    ).to(device=device, dtype=torch.long)
    return x, y


def _logits_160(head, features):
    return F.interpolate(head(features), size=(160, 160), mode="bilinear", align_corners=False)


def _iou_counts(pred: np.ndarray, truth: np.ndarray) -> dict:
    vals = {}
    for c, name in enumerate(("background", "target_galaxy", "other_sources")):
        p, t = pred == c, truth == c
        union = np.logical_or(p, t).sum()
        vals[name] = float(np.logical_and(p, t).sum() / union) if union else float("nan")
    finite = [v for v in vals.values() if np.isfinite(v)]
    vals["mIoU"] = float(np.mean(finite)) if finite else float("nan")
    vals["target_IoU"] = vals["target_galaxy"]
    return vals


@torch.inference_mode()
def predict_indices(model, head, images, indices, kind, mode, input_size, device, batch_size=16):
    outputs = np.empty((len(indices), 160, 160), dtype=np.uint8)
    head.eval(); model.eval()
    for start in range(0, len(indices), batch_size):
        ix = indices[start:start + batch_size]
        x, _ = _batch(images, None, ix, input_size, device)
        feats = extract_features(model, x, kind, mode, input_size)
        logits = _logits_160(head, feats)
        outputs[start:start + len(ix)] = logits.argmax(1).byte().cpu().numpy()
    return outputs


def _evaluate_predictions(predictions, masks, indices):
    intersections = np.zeros(3, dtype=np.int64)
    unions = np.zeros(3, dtype=np.int64)
    for p, i in zip(predictions, indices):
        truth = np.asarray(masks[int(i)])
        for c in range(3):
            pred_c, truth_c = p == c, truth == c
            intersections[c] += np.logical_and(pred_c, truth_c).sum()
            unions[c] += np.logical_or(pred_c, truth_c).sum()
    ious = np.divide(intersections, unions, out=np.full(3, np.nan), where=unions > 0)
    values = {name: float(ious[i]) for i, name in enumerate(("background", "target_galaxy", "other_sources"))}
    finite = ious[np.isfinite(ious)]
    values["mIoU"] = float(finite.mean()) if finite.size else float("nan")
    values["target_IoU"] = values["target_galaxy"]
    return values


def _validation_score(predictions, masks, indices):
    score = _evaluate_predictions(predictions, masks, indices)
    return float(np.nanmean([score["target_galaxy"], score["other_sources"]]))


def fit_head(images, masks, train_indices, val_indices, model, kind, mode, input_size,
             output_path, device, epochs=20, batch_size=32, learning_rate=1e-3,
             feature_cache: np.ndarray | None = None):
    """Train only BN+1x1-conv head; validation foreground mean IoU selects checkpoint."""
    channels = (1536 if mode == "multiscale" else 384)
    head = LinearSegmentationHead(channels).to(device)
    class_weights = estimate_class_weights(masks, train_indices, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_score, best_epoch = -1.0, 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def features_for(ix):
        if feature_cache is not None:
            batch_np = np.stack([feature_cache[int(i)] for i in ix]).astype(np.float32)
            return torch.from_numpy(batch_np).to(device)
        x, _ = _batch(images, masks, ix, input_size, device)
        return extract_features(model, x, kind, mode, input_size)

    for epoch in range(1, epochs + 1):
        head.train(); model.eval()
        order = np.random.permutation(train_indices)
        loss_sum, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            ix = order[start:start + batch_size]
            y = torch.from_numpy(np.stack([masks[int(i)] for i in ix]).astype(np.int64)).to(device)
            with torch.no_grad():
                feats = features_for(ix)
            logits = _logits_160(head, feats)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(ix); seen += len(ix)
        # Validation is non-TTA for checkpoint selection; the test split stays untouched.
        val_pred = []
        head.eval()
        for start in range(0, len(val_indices), batch_size):
            ix = val_indices[start:start + batch_size]
            with torch.no_grad():
                feats = features_for(ix)
                val_pred.extend(_logits_160(head, feats).argmax(1).byte().cpu().numpy())
        score = _validation_score(val_pred, masks, val_indices)
        print(f"epoch {epoch:02d}/{epochs} | train CE {loss_sum/max(1,seen):.4f} | val foreground mean IoU {score:.4f}")
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"state_dict": head.state_dict(), "input_size": input_size,
                        "kind": kind, "mode": mode, "best_epoch": best_epoch,
                        "best_val_foreground_mean_iou": best_score}, output_path)
    state = torch.load(output_path, map_location=device, weights_only=False)
    head.load_state_dict(state["state_dict"])
    return head, {"best_epoch": best_epoch, "best_val_foreground_mean_iou": best_score,
                  "class_weights": class_weights.cpu().tolist()}


def extract_linear224_cache(images, model, output_path, device, batch_size=32, image_ids=None):
    """Cache DINOv2 ViT-S/14 last-layer 224 features to a disk-backed fp16 .npy."""
    output_path = Path(output_path)
    marker = output_path.with_suffix(".complete.json")
    ids_digest = None
    if image_ids is not None:
        import hashlib
        ids_digest = hashlib.sha256("\n".join(map(str, image_ids)).encode()).hexdigest()
    if output_path.exists():
        arr = np.load(output_path, mmap_mode="r")
        if arr.shape == (len(images), 384, 16, 16) and marker.exists():
            state = json.loads(marker.read_text(encoding="utf-8"))
            if (state.get("n_images") == len(images) and state.get("complete") is True
                    and state.get("ids_sha256") == ids_digest
                    and state.get("normalization") == FEATURE_PREPROCESS_VERSION):
                return arr
        del arr
    if marker.exists():
        marker.unlink()
    arr = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16,
                                    shape=(len(images), 384, 16, 16))
    for start in range(0, len(images), batch_size):
        ix = range(start, min(len(images), start + batch_size))
        x, _ = _batch(images, None, ix, 224, device)
        f = extract_features(model, x, "dinov2_s14", "last", 224)
        arr[start:start + len(ix)] = f.half().cpu().numpy()
        if (start + len(ix)) % 200 == 0 or start + len(ix) == len(images):
            arr.flush(); print(f"Cached 224px patch features: {start+len(ix)}/{len(images)}")
    marker.write_text(json.dumps({"n_images": len(images), "complete": True,
                                  "backbone": "dinov2_vits14", "input_px": 224,
                                  "ids_sha256": ids_digest, "normalization": FEATURE_PREPROCESS_VERSION},
                      indent=2), encoding="utf-8")
    return arr


def train_setting(name, kind, mode, input_size, root, device, epochs=20,
                  batch_size=32, feature_cache=None):
    images, masks, ids, parts = load_arrays(root)
    outdir = Path(root) / "results" / "predictions"
    checkpoint_path = outdir / f"{name}.pt"
    prediction_path = outdir / f"{name}_test.npy"
    metrics_path = outdir / f"{name}_metrics.json"
    # Preserve completed settings when a later ablation is interrupted.
    if checkpoint_path.exists() and prediction_path.exists() and metrics_path.exists():
        cached_result = json.loads(metrics_path.read_text(encoding="utf-8"))
        cached_predictions = np.load(prediction_path, mmap_mode="r")
        if (cached_result.get("n_test") == len(parts["test"])
                and cached_predictions.shape == (len(parts["test"]), 160, 160)):
            print(f"Reusing completed setting: {name}")
            return cached_result
    if kind == "dinov2_s14_reg" and mode == "last":
        pass
    model = load_backbone(kind, device)
    head, training = fit_head(
        images, masks, parts["train"], parts["validation"], model, kind, mode,
        input_size, checkpoint_path,
        device, epochs=epochs, batch_size=batch_size, feature_cache=feature_cache,
    )
    preds = predict_indices(model, head, images, parts["test"], kind, mode, input_size,
                            device, batch_size=max(1, min(batch_size, 16)))
    np.save(prediction_path, preds)
    scores = _evaluate_predictions(preds, masks, parts["test"])
    result = {"setting": name, "backbone": kind, "input_px": input_size,
              "patch_grid": f"{input_size//(8 if kind=='dino_s8' else 14)}x{input_size//(8 if kind=='dino_s8' else 14)}",
              **scores, **training, "n_test": int(len(parts["test"]))}
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    del model, head
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def _attention_mass_mask(values: np.ndarray, fraction: float = 0.60) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values)[::-1]
    cumulative = np.cumsum(values[order])
    k = int(np.searchsorted(cumulative, fraction * cumulative[-1], side="left") + 1)
    result = np.zeros(values.size, np.uint8); result[order[:k]] = 1
    return result


@torch.inference_mode()
def evaluate_naive_attention(root, device, batch_size=16, fraction=0.60):
    """Week 3 DINO ViT-S/8 baseline, evaluated on exactly the Week 4 test split."""
    from PIL import Image
    images, masks, ids, parts = load_arrays(root)
    model = load_backbone("dino_s8", device)
    idx = parts["test"]
    all_source_scores, target_scores = [], []
    gallery_predictions = []
    per_image = []
    for start in range(0, len(idx), batch_size):
        batch_ix = idx[start:start + batch_size]
        x = make_input(np.stack([images[int(i)] for i in batch_ix]), 224, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=x.is_cuda):
            attn = model.get_last_selfattention(x)
        patch = attn[:, :, 0, 1:].float().cpu().numpy()
        for j, i in enumerate(batch_ix):
            truth = np.asarray(masks[int(i)])
            head_masks = []
            for h in range(patch.shape[1]):
                coarse = _attention_mass_mask(patch[j, h], fraction).reshape(28, 28)
                dense = np.asarray(Image.fromarray(coarse).resize((160, 160), Image.Resampling.NEAREST)) > 0
                head_masks.append(dense)
                all_source_scores.append(_binary_iou(dense, truth > 0))
                target_scores.append(_binary_iou(dense, truth == 1))
            mean_map = patch[j].mean(axis=0)
            gallery_predictions.append(np.asarray(Image.fromarray(_attention_mass_mask(mean_map, fraction).reshape(28,28))
                .resize((160,160), Image.Resampling.NEAREST), dtype=np.uint8))
            per_image.append({"index": int(i), "image_id": str(ids[int(i)]),
                              "mean_head_all_source_iou": float(np.mean([_binary_iou(m, truth>0) for m in head_masks])),
                              "mean_head_target_iou": float(np.mean([_binary_iou(m, truth==1) for m in head_masks]))})
        if (start + len(batch_ix)) % 100 == 0 or start + len(batch_ix) == len(idx):
            print(f"Naive attention evaluated {start+len(batch_ix)}/{len(idx)} test cutouts")
    outdir = Path(root) / "results" / "predictions"; outdir.mkdir(parents=True, exist_ok=True)
    metric_dir = Path(root) / "results" / "metrics"; metric_dir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "naive_attention_test.npy", np.stack(gallery_predictions))
    import csv
    with (metric_dir/"naive_attention_per_image.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(per_image[0])); writer.writeheader(); writer.writerows(per_image)
    result = {"setting":"naive_attention_week3_dino_s8", "backbone":"DINO ViT-S/8",
              "input_px":224, "patch_grid":"28x28", "mIoU":None,
              "all_source_iou":float(np.mean(all_source_scores)),
              "target_IoU":float(np.mean(target_scores)), "threshold_attention_mass":fraction,
              "n_test":int(len(idx)), "note":"Per-head IoUs averaged over six heads and test cutouts; no class output."}
    (metric_dir/"naive_attention.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result, np.stack(gallery_predictions)


def _binary_iou(pred, truth):
    union = np.logical_or(pred, truth).sum()
    return float(np.logical_and(pred, truth).sum()/union) if union else float("nan")


@torch.inference_mode()
def evaluate_multiscale_tta(root, checkpoint, device, batch_size=4):
    """Six-view {336,448,560} plus horizontal-flip TTA for the +ms model."""
    from week4_data import IMAGENET_MEAN, IMAGENET_STD
    images, masks, ids, parts = load_arrays(root)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = load_backbone("dinov2_s14", device)
    head = LinearSegmentationHead(1536).to(device); head.load_state_dict(state["state_dict"]); head.eval()
    idx = parts["test"]; preds = np.empty((len(idx),160,160),np.uint8)
    mean=torch.as_tensor(IMAGENET_MEAN,device=device).view(1,3,1,1)
    std=torch.as_tensor(IMAGENET_STD,device=device).view(1,3,1,1)
    for start in range(0,len(idx),batch_size):
        batch_ix=idx[start:start+batch_size]
        base=torch.from_numpy(np.stack([images[int(i)] for i in batch_ix]).copy()).to(device=device,dtype=torch.float32).permute(0,3,1,2)/255.0
        probs=[]
        for size in (336,448,560):
            resized=F.interpolate(base,size=(size,size),mode="bicubic",align_corners=False,antialias=True).clamp(0,1)
            x=(resized-mean)/std
            for do_flip in (False,True):
                xin=torch.flip(x,[-1]) if do_flip else x
                features=extract_features(model,xin,"dinov2_s14","multiscale",size)
                logits=_logits_160(head,features)
                prob=logits.softmax(1)
                if do_flip: prob=torch.flip(prob,[-1])
                probs.append(prob)
        avg=torch.stack(probs).mean(0)
        preds[start:start+len(batch_ix)]=avg.argmax(1).byte().cpu().numpy()
        print(f"+ms TTA evaluated {min(start+len(batch_ix),len(idx))}/{len(idx)}")
    outdir=Path(root)/"results"/"predictions"; np.save(outdir/"plus_ms_tta_test.npy",preds)
    scores=_evaluate_predictions(preds,masks,idx)
    result={"setting":"plus_ms_tta", "backbone":"DINOv2 ViT-S/14", "input_px":448,
            "patch_grid":"32x32", "tta_scales":[336,448,560], "horizontal_flip":True,
            **scores,"n_test":int(len(idx))}
    (Path(root)/"results"/"metrics"/"plus_ms_tta_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    del model,head
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result,preds


def make_hard_gallery(root, naive_predictions, linear_predictions, ms_predictions, max_items=20,
                      selected_ids: list[str] | None = None):
    """Choose blend/contamination-heavy test cutouts and write the requested gallery."""
    import matplotlib.pyplot as plt
    root=Path(root); images,masks,ids,parts=load_arrays(root); test=parts["test"]
    # Automatic candidate ordering prioritizes other-source coverage, then target size.
    rank=[]
    for pos,i in enumerate(test):
        m=np.asarray(masks[int(i)]); other=(m==2).mean(); target=(m==1).mean()
        rank.append((other + 0.15*(target>0.25), int(i), pos))
    chosen=sorted(rank,reverse=True)[:min(max_items,len(rank))]
    if selected_ids:
        id_to_global={str(ids[int(i)]):int(i) for i in test}
        absent=[x for x in selected_ids if x not in id_to_global]
        if absent: raise ValueError(f"Selected hard-set IDs are not in the test split: {absent[:3]}")
        global_ix=[id_to_global[x] for x in selected_ids[:max_items]]
        pos_map={int(index):pos for pos,index in enumerate(test)}
        pred_pos=[pos_map[i] for i in global_ix]
        selection="manually selected test IDs"
    else:
        global_ix=[x[1] for x in chosen]; pred_pos=[x[2] for x in chosen]
        selection="automatic candidates ranked by other-source pixel fraction; inspect and optionally replace"
    hard_ids=[str(ids[i]) for i in global_ix]
    outdir=root/"results"/"predictions"; outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"hard_set_ids.json").write_text(json.dumps({"selection":selection,"indices":global_ix,"ids":hard_ids},indent=2),encoding="utf-8")
    # Four examples per page keeps the 5-column panels readable in notebooks/PDF.
    for page_start in range(0,len(global_ix),4):
        subset=range(page_start,min(page_start+4,len(global_ix)))
        fig,axes=plt.subplots(len(list(subset)),5,figsize=(15,3.3*len(list(subset))),squeeze=False)
        for row,j in enumerate(subset):
            i=global_ix[j]; p=pred_pos[j]
            panels=[images[i],masks[i],naive_predictions[p],linear_predictions[p],ms_predictions[p]]
            titles=["DESI RGB","Photutils pseudo-label","Week 3 attention","Linear DINOv2 224","+ms with TTA"]
            for col,(ax,arr,title) in enumerate(zip(axes[row],panels,titles)):
                ax.imshow(arr, cmap="tab10" if col>0 else None, vmin=0 if col>0 else None, vmax=2 if col>0 else None)
                ax.set_title(title if row==0 else "",fontsize=9); ax.set_xticks([]);ax.set_yticks([])
            axes[row,0].set_ylabel(hard_ids[j][:24],fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir/f"hard_set_gallery_{page_start//4+1:02d}.png",dpi=150,bbox_inches="tight")
        plt.close(fig)


def write_comparison(root, results: list[dict], naive: dict):
    import csv
    root=Path(root); out=root/"results"/"metrics"; out.mkdir(parents=True,exist_ok=True)
    rows=[naive]+results
    keys=["setting","backbone","input_px","patch_grid","mIoU","all_source_iou","target_IoU","n_test"]
    with (out/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    (out/"comparison.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    return rows

