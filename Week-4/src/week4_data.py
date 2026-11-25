"""Data preparation and photutils pseudo-label utilities for Week 4.

This module is designed for a Colab/Kaggle notebook runtime. It streams the
public Legacy Survey dataset, writes disk-backed arrays, and keeps the Week 4
data preparation independent of the earlier notebooks.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

BANDS = ("g", "r", "z")
ALL_BANDS = ("g", "r", "i", "z")
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
DATA_PIPELINE_VERSION = "v3-ivar-authoritative-robust-background"


def unpack_bands(image_field: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return flux and invalid-pixel mask in g,r,i,z order.

    The dataset's ``mask`` field is not assumed to use True=bad semantics.
    Pixel inverse variance is the reliable coverage signal when present.
    """
    if isinstance(image_field, (list, tuple)):
        if not image_field:
            raise ValueError("Empty image field")
        image_field = {k: [row[k] for row in image_field] for k in image_field[0].keys()}
    names = [str(x).strip().split("-")[-1].lower() for x in image_field["band"]]
    order = [names.index(b) for b in ALL_BANDS]
    flux = np.stack([np.asarray(image_field["flux"][i], dtype=np.float32) for i in order])
    raw_mask = np.stack([np.asarray(image_field["mask"][i]) for i in order]) if "mask" in image_field else None
    if "ivar" in image_field:
        ivar = np.stack([np.asarray(image_field["ivar"][i], dtype=np.float32) for i in order])
        invalid = ~np.isfinite(ivar) | (ivar <= 0)
        # IVAR is the explicit coverage field in this dataset. Do not combine
        # the survey mask with it: mask polarity/semantics vary by release, and
        # a wrong union can mark almost an entire cutout as invalid.
        for j in range(len(order)):
            finite_flux = np.isfinite(flux[j])
            if invalid[j].size - int(invalid[j].sum()) < 100 and finite_flux.sum() >= 100:
                # Some records have an empty/misaligned IVAR plane. Keep the
                # finite flux pixels for this band rather than erasing a valid
                # cutout; the QC output will still expose background fallback.
                invalid[j] = ~finite_flux
    elif raw_mask is not None:
        # Fallback for schemas without IVAR: all-True means usable coverage;
        # otherwise infer the sparse True pixels as bad flags.
        invalid = np.zeros(raw_mask.shape, dtype=bool)
        for j in range(raw_mask.shape[0]):
            values = raw_mask[j].astype(bool, copy=False)
            if values.min() == values.max():
                invalid[j] = False
            elif float(values.mean()) > 0.5:
                invalid[j] = ~values
            else:
                invalid[j] = values
    else:
        invalid = ~np.isfinite(flux)
    if flux.shape != invalid.shape or flux.ndim != 3:
        raise ValueError(f"Unexpected band array shapes: flux={flux.shape}, invalid={invalid.shape}")
    invalid |= ~np.isfinite(flux)
    return flux, invalid


def _background2d(data: np.ndarray, mask: np.ndarray):
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground

    valid = np.isfinite(data) & ~mask
    # IVAR can be zero over the full image in some exported records even when
    # the flux cutout itself is usable. In that case estimate on finite flux
    # pixels and record the fallback, instead of silently declaring the image
    # empty.
    if valid.sum() < 100:
        finite = np.isfinite(data)
        if finite.sum() >= 100:
            from types import SimpleNamespace
            values = np.asarray(data, dtype=np.float32)[finite]
            level = float(np.median(values))
            rms = float(1.4826 * np.median(np.abs(values - level)))
            if not np.isfinite(rms) or rms <= 0:
                rms = float(np.std(values))
            rms = max(rms if np.isfinite(rms) else 0.0, 1e-6)
            return SimpleNamespace(
                background=np.full(data.shape, level, dtype=np.float32),
                background_rms=np.full(data.shape, rms, dtype=np.float32),
                fallback=True,
                fallback_reason="IVAR/mask left <100 covered pixels; finite-flux robust estimate",
            )
        else:
            raise ValueError(f"Only {int(valid.sum())} covered / {int(finite.sum())} finite pixels")
    safe_mask = ~valid
    # Keep box dimensions valid for small images while retaining local sky estimates.
    box = min(32, data.shape[0], data.shape[1])
    box = max(8, box)
    try:
        return Background2D(
            data, box_size=(box, box), filter_size=(3, 3),
            sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
            bkg_estimator=MedianBackground(), mask=safe_mask,
            exclude_percentile=80,
        )
    except Exception as exc:
        # Some cutouts do not leave enough pixels in each local box. A robust
        # global sky/RMS fallback keeps these usable while exposing provenance
        # to the QC table.
        from types import SimpleNamespace
        values = np.asarray(data, dtype=np.float32)[valid]
        level = float(np.median(values))
        rms = float(1.4826 * np.median(np.abs(values - level)))
        if not np.isfinite(rms) or rms <= 0:
            rms = float(np.std(values))
        rms = max(rms if np.isfinite(rms) else 0.0, 1e-6)
        return SimpleNamespace(
            background=np.full(data.shape, level, dtype=np.float32),
            background_rms=np.full(data.shape, rms, dtype=np.float32),
            fallback=True,
            fallback_reason=f"{type(exc).__name__}: {str(exc)[:160]}",
        )


def sdss_rgb_from_snr(snr_g_r_z: np.ndarray, m: float = 0.03, q: float = 20.0) -> np.ndarray:
    """Week 1 Legacy-Survey g/r/z asinh composite applied to RMS-normalized bands."""
    if snr_g_r_z.shape[0] != 3:
        raise ValueError("Expected g,r,z planes")
    scales = {"g": (2, 6.0), "r": (1, 3.4), "z": (0, 2.2)}
    planes = {b: snr_g_r_z[i].astype(np.float32, copy=False) for i, b in enumerate(BANDS)}
    intensity = sum(np.maximum(0.0, planes[b] * scales[b][1] + m) for b in BANDS) / 3.0
    f_intensity = np.arcsinh(q * intensity) / math.sqrt(q)
    safe = intensity + (intensity == 0) * 1e-6
    rgb = np.zeros((*intensity.shape, 3), dtype=np.float32)
    for band in BANDS:
        channel, scale = scales[band]
        rgb[..., channel] = (planes[band] * scale + m) * f_intensity / safe
    return np.clip(rgb, 0.0, 1.0)


def normalize_flux_to_rgb(flux4: np.ndarray, input_mask4: np.ndarray) -> tuple[np.ndarray, dict]:
    """Estimate band backgrounds/RMS, convert g/r/z to normalized RGB uint8."""
    snr_planes = []
    stat = {}

    def robust_maps(plane: np.ndarray, badmask: np.ndarray):
        valid = np.isfinite(plane) & ~np.asarray(badmask, dtype=bool)
        vals = plane[valid]
        if vals.size < 100:
            vals = plane[np.isfinite(plane)]
        if vals.size == 0:
            # Missing band: neutral residual and finite epsilon RMS keep RGB valid.
            level, rms = 0.0, 1e-6
            no_data = True
        else:
            level = float(np.median(vals))
            mad_rms = float(1.4826 * np.median(np.abs(vals - level)))
            spread_rms = float(np.std(vals)) * 1e-3
            rms = max(mad_rms, spread_rms, 1e-6)
            no_data = False
        return (np.full(plane.shape, level, np.float32),
                np.full(plane.shape, rms, np.float32), level, rms, no_data)

    for band in BANDS:
        idx = ALL_BANDS.index(band)
        plane = np.asarray(flux4[idx], dtype=np.float32)
        badmask = np.asarray(input_mask4[idx], dtype=bool)
        background = None
        sigma = None
        background_level = float("nan")
        try:
            bkg = _background2d(plane, badmask)
            background = np.asarray(bkg.background, dtype=np.float32)
            sigma = np.asarray(bkg.background_rms, dtype=np.float32)
            background_level = float(np.nanmedian(background))
            if getattr(bkg, "fallback", False):
                stat[f"{band}_background_fallback"] = True
                stat[f"{band}_background_fallback_reason"] = bkg.fallback_reason
        except Exception as exc:
            stat[f"{band}_background_fallback_reason"] = type(exc).__name__

        good_sigma = (np.asarray([], dtype=np.float32) if sigma is None else
                      sigma[np.isfinite(sigma) & (sigma > 0)])
        if good_sigma.size == 0 or not np.isfinite(background_level):
            background, sigma, background_level, replacement, no_data = robust_maps(plane, badmask)
            stat[f"{band}_background_fallback"] = True
            stat.setdefault(f"{band}_background_fallback_reason", "no_positive_finite_rms_map")
            if no_data:
                stat[f"{band}_no_valid_pixels"] = True
        else:
            # Fill isolated edge/background-estimator failures with robust median RMS.
            replacement = float(np.median(good_sigma))
            sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, replacement)
            background = np.where(np.isfinite(background), background, background_level)
        residual = plane - background
        snr = np.nan_to_num(residual / np.maximum(sigma, 1e-6), nan=0.0, posinf=0.0, neginf=0.0)
        snr_planes.append(snr)
        stat[f"{band}_background_median"] = background_level
        stat[f"{band}_rms_median"] = replacement
    rgb = sdss_rgb_from_snr(np.stack(snr_planes))
    return np.rint(rgb * 255).astype(np.uint8), stat


def build_three_class_mask(flux4: np.ndarray, input_mask4: np.ndarray) -> tuple[np.ndarray, dict]:
    """Photutils detection and center-source assignment; returns 160x160 uint8 labels."""
    from astropy.convolution import Gaussian2DKernel
    from photutils.segmentation import SourceFinder
    from scipy.ndimage import gaussian_filter

    planes = []
    masks = []
    for band in BANDS:
        i = ALL_BANDS.index(band)
        plane = np.asarray(flux4[i], dtype=np.float32)
        invalid = np.asarray(input_mask4[i], dtype=bool)
        planes.append(np.where(invalid, 0.0, plane))
        masks.append(invalid)
    mask_stack = np.stack(masks)
    coadd = np.sum(np.stack(planes), axis=0, dtype=np.float32)
    # A pixel is unusable only when every coadd band lacks coverage.
    coadd_mask = np.all(mask_stack, axis=0) | ~np.isfinite(coadd)
    try:
        bkg = _background2d(coadd, coadd_mask)
    except Exception as exc:
        reason = str(exc).replace("\n", " ")[:120]
        return np.zeros(coadd.shape, np.uint8), {"n_sources": 0,
            "status": f"background_failed:{type(exc).__name__}:{reason}"}
    residual = coadd - np.asarray(bkg.background, dtype=np.float32)
    rms_map = np.asarray(bkg.background_rms, dtype=np.float32)
    good = rms_map[np.isfinite(rms_map) & (rms_map > 0)]
    if good.size == 0:
        return np.zeros(coadd.shape, np.uint8), {"n_sources": 0, "status": "invalid_rms"}
    fallback = float(np.median(good))
    rms_map = np.where(np.isfinite(rms_map) & (rms_map > 0), rms_map, fallback)

    sigma = 1.0
    kernel = Gaussian2DKernel(x_stddev=sigma, y_stddev=sigma, x_size=7, y_size=7)
    kernel_arr = np.asarray(kernel.array, dtype=np.float32)
    smooth = gaussian_filter(np.where(coadd_mask, 0.0, residual), sigma=sigma, mode="nearest")
    # Variance of a Gaussian-smoothed image is reduced by sum(kernel**2).
    noise_scale = float(np.sqrt(np.sum(kernel_arr ** 2)))
    smooth_noise = np.maximum(rms_map * noise_scale, 1e-6)
    threshold = 1.5 * smooth_noise
    try:
        # Photutils 3.x accepts the threshold at call time and spells the
        # source-area argument n_pixels/n_levels. Photutils 1.x-2.x used
        # npixels/threshold/nlevels in the constructor. Support both APIs.
        try:
            finder = SourceFinder(n_pixels=10, deblend=True, n_levels=32, contrast=0.001)
            segm = finder(smooth, threshold, mask=coadd_mask)
        except TypeError:
            finder = SourceFinder(npixels=10, threshold=threshold, deblend=True,
                                  nlevels=32, contrast=0.001)
            segm = finder(smooth, mask=coadd_mask)
    except Exception as exc:
        reason = str(exc).replace("\n", " ")[:120]
        return np.zeros(coadd.shape, np.uint8), {"n_sources": 0,
            "status": f"detection_failed:{type(exc).__name__}:{reason}"}
    labels = np.zeros(coadd.shape, dtype=np.uint8)
    if segm is None or segm.nlabels == 0:
        return labels, {"n_sources": 0, "status": "no_sources"}

    source_labels = np.asarray(segm.data, dtype=np.int32)
    source_ids = np.unique(source_labels)
    source_ids = source_ids[source_ids > 0]
    cy, cx = (np.asarray(coadd.shape) - 1) / 2.0
    # Choose the label covering the centre; otherwise choose the source with the
    # nearest detected pixel to the geometric centre.
    iy, ix = int(round(cy)), int(round(cx))
    target = int(source_labels[iy, ix])
    if target == 0:
        best_dist = float("inf")
        for label in source_ids:
            yy, xx = np.nonzero(source_labels == label)
            d2 = (yy - cy) ** 2 + (xx - cx) ** 2
            j = int(np.argmin(d2))
            if float(d2[j]) < best_dist:
                best_dist = float(d2[j])
                target = int(label)
    labels[source_labels > 0] = 2
    labels[source_labels == target] = 1
    n_other = int(np.sum(source_ids != target))
    status = "ok" if int(np.sum(labels == 1)) else "no_target"
    return labels, {"n_sources": int(len(source_ids)), "n_other_sources": n_other,
                    "target_pixels": int(np.sum(labels == 1)), "other_pixels": int(np.sum(labels == 2)),
                    "status": status,
                    "background_method": "robust_global_fallback" if getattr(bkg, "fallback", False) else "Background2D",
                    "background_fallback_reason": getattr(bkg, "fallback_reason", "")}


def stream_cutouts(n_images: int, seed: int = 42, buffer_size: int = 64):
    """Yield deterministic shuffled raw cutouts and stable identifiers."""
    from datasets import load_dataset

    ds = load_dataset("MultimodalUniverse/legacysurvey", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    for i, example in enumerate(ds):
        if i >= n_images:
            break
        flux, badmask = unpack_bands(example["image"])
        raw_id = example.get("object_id", example.get("id", i))
        if isinstance(raw_id, bytes):
            raw_id = raw_id.decode("utf-8", errors="replace")
        yield i, f"{str(raw_id)}__{i:05d}", flux, badmask


def make_splits(ids: list[str], output_path: Path, seed: int = 42) -> dict:
    digest = __import__("hashlib").sha256("\n".join(ids).encode()).hexdigest()
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("n_images") != len(ids) or existing.get("ids_sha256") != digest:
            raise RuntimeError(
                "Existing splits.json belongs to a different image ordering. Restore the matching RGB/ID cache; "
                "do not overwrite a split that later project weeks depend on."
            )
        return existing
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    n_train = int(round(0.70 * len(ids)))
    n_val = int(round(0.10 * len(ids)))
    train = order[:n_train]
    val = order[n_train:n_train + n_val]
    test = order[n_train + n_val:]
    payload = {
        "seed": int(seed), "n_images": len(ids), "fractions": {"train": 0.70, "validation": 0.10, "test": 0.20},
        "train_indices": train.tolist(), "validation_indices": val.tolist(), "test_indices": test.tolist(),
        "ids_sha256": digest,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def create_dataset(output_root: str | Path, n_images: int = 3000, seed: int = 42,
                   resume: bool = True, buffer_size: int = 64) -> dict:
    """Stream and persist Week 4 RGB, pseudo-labels, IDs, stats and split."""
    import csv

    root = Path(output_root)
    data_dir = root / "data"
    pseudo_dir = root / "results" / "pseudo_labels"
    data_dir.mkdir(parents=True, exist_ok=True)
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    images_file = data_dir / "rgb_images.npy"
    masks_file = pseudo_dir / "masks.npy"
    ids_file = data_dir / "image_ids.npy"
    stats_file = data_dir / "normalization_and_qc.csv"
    state_file = data_dir / "preparation_complete.json"
    complete = False
    split_file = data_dir / "splits.json"
    if (resume and state_file.exists() and images_file.exists() and masks_file.exists()
            and ids_file.exists() and stats_file.exists() and split_file.exists()):
        state = json.loads(state_file.read_text(encoding="utf-8"))
        complete = (state.get("n_images") == n_images
                    and state.get("pipeline_version") == DATA_PIPELINE_VERSION and all(
            np.load(p, mmap_mode="r").shape[0] == n_images for p in (images_file, masks_file, ids_file)
        ))
    if complete:
        ids = np.load(ids_file, mmap_mode="r").astype(str).tolist()
        return {"n_images": n_images, "ids": ids, "resumed": True}

    rgb_mm = np.lib.format.open_memmap(images_file, mode="w+", dtype=np.uint8, shape=(n_images, 160, 160, 3))
    mask_mm = np.lib.format.open_memmap(masks_file, mode="w+", dtype=np.uint8, shape=(n_images, 160, 160))
    ids_mm = np.lib.format.open_memmap(ids_file, mode="w+", dtype="<U128", shape=(n_images,))
    rows = []
    ids = []
    for i, image_id, flux, badmask in stream_cutouts(n_images, seed=seed, buffer_size=buffer_size):
        if flux.shape[-2:] != (160, 160):
            raise ValueError(f"Expected 160x160 DESI cutout, got {flux.shape}")
        rgb, norm_stats = normalize_flux_to_rgb(flux, badmask)
        labels, qc = build_three_class_mask(flux, badmask)
        rgb_mm[i] = rgb
        mask_mm[i] = labels
        ids_mm[i] = image_id[:128]
        ids.append(image_id[:128])
        rows.append({"index": i, "image_id": image_id[:128], **norm_stats, **qc})
        if (i + 1) % 50 == 0 or i + 1 == n_images:
            rgb_mm.flush(); mask_mm.flush(); ids_mm.flush()
            print(f"Prepared {i+1}/{n_images} cutouts")
    if len(ids) != n_images:
        raise RuntimeError(f"Dataset stream ended at {len(ids)} images; expected {n_images}")
    with stats_file.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    split = make_splits(ids, data_dir / "splits.json", seed=seed)
    state_file.write_text(json.dumps({"n_images": n_images, "seed": seed,
        "pipeline_version": DATA_PIPELINE_VERSION,
        "normalization": "Background2D per-band RMS; residual/RMS; Week 1 g/r/z asinh RGB",
        "pseudo_label": "Photutils SourceFinder on smoothed background-subtracted g+r+z",
        "complete": True}, indent=2), encoding="utf-8")
    return {"n_images": n_images, "ids": ids, "split": split, "resumed": False}


def update_valid_indices(output_root: str | Path, dropped_ids: list[str]) -> np.ndarray:
    root = Path(output_root)
    ids = np.load(root / "data" / "image_ids.npy", mmap_mode="r").astype(str)
    drop = set(dropped_ids)
    valid = np.asarray([i for i, image_id in enumerate(ids) if image_id not in drop], dtype=np.int32)
    np.save(root / "data" / "valid_indices.npy", valid)
    (root / "data" / "dropped_ids.json").write_text(json.dumps(sorted(drop), indent=2), encoding="utf-8")
    return valid

