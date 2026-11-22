import numpy as np
from btk.catalog import CatsimCatalog
from btk.draw_blends import CatsimGenerator
from btk.sampling_functions import DefaultSampling
from btk.survey import get_surveys

from cutouts import background, semantic_mask, to_rgb


def generate(catalog_path, output, split, count, seed, jobs=1):
    catalog = CatsimCatalog.from_file(str(catalog_path))
    survey = get_surveys("DES")
    survey.available_filters = list("griz")
    filters = list(survey.available_filters)
    indices = [filters.index(band) for band in "griz"]
    pixel_scale = survey.pixel_scale.to_value("arcsec")
    sampling = DefaultSampling(stamp_size=160.00001 * pixel_scale, min_number=1,
                               max_number=4, seed=seed, max_mag=23.5, mag_name="r_ab")
    generator = CatsimGenerator(catalog, sampling, survey, batch_size=16,
                                seed=seed, njobs=jobs)
    output.mkdir(parents=True, exist_ok=True)
    for start in range(0, count, 16):
        batch = next(generator)
        for offset in range(min(16, count - start)):
            image = batch.blend_images[offset, indices].astype(np.float32)
            table = batch.catalog_list[offset]
            isolated = batch.isolated_images[offset, :len(table)][:, indices]
            assert image.shape == (4, 160, 160)
            distance = (table["x_peak"] - 79.5)**2 + (table["y_peak"] - 79.5)**2
            target = int(np.argmin(distance)) + 1
            # Use r-band ownership at one local sky RMS, including when galaxies overlap.
            r_band = isolated[:, 1]
            rms = background(image[1]).background_rms
            instances = np.where(r_band.max(0) >= rms, 1 + r_band.argmax(0), 0).astype(np.uint8)
            mask = semantic_mask(instances, target)
            rgb = to_rgb(image[[0, 1, 3]])
            # Use one scalar per stamp to keep float16 finite without changing band ratios.
            image_scale = np.float32(max(1, np.max(np.abs(image)) / 60000))
            path = output / f"{split}_{start + offset:05d}.npz"
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary,
                                griz=(image / image_scale).astype(np.float16), image_scale=image_scale,
                                rgb=rgb, instance=instances,
                                semantic=mask, target=target, flux=isolated.sum((2, 3)),
                                x=table["x_peak"], y=table["y_peak"],
                                psf=[survey.get_filter(band).psf_fwhm.to_value("arcsec")
                                     for band in "griz"])
            temporary.replace(path)
