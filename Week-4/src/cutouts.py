import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.visualization import make_lupton_rgb
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import SourceFinder


def background(image):
    return Background2D(image, (32, 32), filter_size=(3, 3),
                        bkg_estimator=MedianBackground())


def to_rgb(grz):
    # Use the same fixed stretch on real and simulated images.
    bands = []
    for image in np.asarray(grz, dtype=np.float32):
        sky = background(image)
        bands.append((image - sky.background) / np.maximum(sky.background_rms, 1e-6))
    g, r, z = bands
    return make_lupton_rgb(z, r, g, stretch=5, Q=8)


def semantic_mask(instances, target):
    mask = np.where(instances > 0, 2, 0).astype(np.uint8)
    if target > 0:
        mask[instances == target] = 1
    return mask


def pseudo_label(grz):
    # Detect on the sum of the native flux bands.
    coadd = np.asarray(grz, dtype=np.float32).sum(axis=0)
    sky = background(coadd)
    residual = coadd - sky.background
    smoothed = convolve(residual, Gaussian2DKernel(1))
    smooth_sky = background(smoothed)
    found = SourceFinder(npixels=10, deblend=True, progress_bar=False)(
        smoothed - smooth_sky.background, 1.5 * smooth_sky.background_rms
    )
    if found is None:
        empty = np.zeros(coadd.shape, dtype=np.uint8)
        return empty, empty, 0
    instances = found.data
    cy, cx = (np.array(instances.shape) - 1) / 2
    target = instances[round(cy), round(cx)]
    if target == 0:
        y, x = np.nonzero(instances)
        nearest = ((y - cy)**2 + (x - cx)**2).argmin()
        target = instances[y[nearest], x[nearest]]
    return instances, semantic_mask(instances, target), int(target)


def prepare_cutouts(root, raw_files, revision, count=3000, workers=4):
    import json
    from concurrent.futures import ThreadPoolExecutor
    import pyarrow.parquet as pq

    data = root / "data"
    images = data / "cutouts"
    masks = root / "results" / "pseudo_labels"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    def prepare(row):
        key = object_id(row["object_id"])
        image_file, label_file = images / f"{key}.npz", masks / f"{key}.npy"
        if image_file.exists() and label_file.exists():
            return key if (np.load(label_file) == 1).any() else None
        image = row["image"]
        indices = [image["band"].index(f"des-{band}") for band in "griz"]
        flux = image["flux"]
        if flux.size != 4 * 160 * 160:
            return None
        griz = flux.reshape(4, 160, 160)[indices]
        if griz.shape != (4, 160, 160) or not np.isfinite(griz).all():
            return None
        grz = griz[[0, 1, 3]]
        try:
            rgb = to_rgb(grz)
            _, mask, target = pseudo_label(grz)
        except ValueError:
            return None
        np.savez_compressed(image_file, griz=griz, rgb=rgb,
                            psf=np.asarray(image["psf_fwhm"])[indices],
                            r_flux=row["FLUX_R"], shape_r=row["SHAPE_R"])
        np.save(label_file, mask)
        return key if target else None

    split = freeze_split(root, raw_files, revision, count)
    planned = split["source_rows"]
    rejected = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        seen = set()
        for path in raw_files:
            parquet = pq.ParquetFile(path)
            offset = 0
            for batch in parquet.iter_batches(batch_size=16,
                                              columns=["image", "object_id", "FLUX_R", "SHAPE_R"]):
                image = batch.column("image")
                rows = []
                for i, value in enumerate(batch.column("object_id").to_pylist()):
                    key = object_id(value)
                    if planned.get(key) != [path.name, offset + i]:
                        continue
                    seen.add(key)
                    band_flux = image.field("flux").slice(i, 1).flatten().flatten().flatten()
                    rows.append({"object_id": value,
                                 "image": {"band": image.field("band")[i].as_py(),
                                           "flux": band_flux.to_numpy(),
                                           "psf_fwhm": image.field("psf_fwhm")[i].as_py()},
                                 "FLUX_R": batch.column("FLUX_R")[i].as_py(),
                                 "SHAPE_R": batch.column("SHAPE_R")[i].as_py()})
                for row, key in zip(rows, pool.map(prepare, rows)):
                    if key is None:
                        rejected.append(object_id(row["object_id"]))
                offset += batch.num_rows
            if len(seen) == len(planned):
                break
    if seen != set(planned):
        raise ValueError("The supplied Parquet files do not cover the frozen split.")
    (data / "automatic_exclusions.json").write_text(json.dumps(sorted(set(rejected)), indent=2) + "\n")
    return split


def object_id(value):
    if isinstance(value, bytes):
        return value.decode("ascii")
    return value[2:-1] if value.startswith("b'") and value.endswith("'") else value


def freeze_split(root, raw_files, revision, count=3000):
    import json
    import pyarrow.parquet as pq

    saved = root / "data" / "splits.json"
    selected = {}
    for path in raw_files:
        ids = pq.read_table(path, columns=["object_id"]).column(0).to_pylist()
        for i, raw in enumerate(ids):
            selected.setdefault(object_id(raw), [path.name, i])
            if len(selected) == count:
                break
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Expected {count} unique DESI objects, found {len(selected)}.")
    ids = np.random.default_rng(42).permutation(list(selected)).tolist()
    first, second = int(.7 * count), int(.8 * count)
    split = {"seed": 42, "revision": revision, "requested": count,
             "train": ids[:first], "validation": ids[first:second], "test": ids[second:],
             "source_rows": selected}
    if saved.exists() and json.loads(saved.read_text()) != split:
        raise ValueError("The regenerated split differs from the saved split.")
    saved.write_text(json.dumps(split, indent=2) + "\n")
    return split



def native_griz(sample):
    image = sample["griz"].astype(np.float32)
    if "image_scale" in sample:
        image *= sample["image_scale"]
    return image
