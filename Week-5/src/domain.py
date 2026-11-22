import numpy as np
import matplotlib.pyplot as plt

from cutouts import background, native_griz, pseudo_label


def measure(path, domain):
    with np.load(path) as sample:
        griz = native_griz(sample)
        rms = np.array([np.median(background(band).background_rms) for band in griz])
        # Use the same pseudo-aperture for both domains to avoid mixing mask definitions.
        _, mask, _ = pseudo_label(griz[[0, 1, 3]])
        region = mask == 1
        area = max(int(region.sum()), 1)
        sky = background(griz[1]).background
        values = {"r_aperture_snr": float((griz[1] - sky)[region].sum() / (rms[1] * area**.5)),
                  "target_radius_pixels": (region.sum() / np.pi)**.5}
        for band, noise, width in zip("griz", rms, sample["psf"]):
            values[f"{band}_native_rms"] = float(noise)
            values[f"{band}_psf_arcsec"] = float(width)
        return {"domain": domain, **values}


def compare_domains(real, simulated, output):
    import pandas as pd
    rng = np.random.default_rng(42)
    rows = [measure(path, label) for label, paths in (("real", real), ("sim", simulated))
            for path in rng.choice(paths, min(len(paths), 200), replace=False)]
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "sample.csv", index=False)
    table.groupby("domain").median(numeric_only=True).to_csv(output / "median.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, name in zip(axes.flat, ("r_aperture_snr", "target_radius_pixels",
                                    "g_psf_arcsec", "r_psf_arcsec", "i_psf_arcsec",
                                    "z_psf_arcsec")):
        for label, group in table.groupby("domain"):
            ax.hist(group[name], bins=20, histtype="step", density=True, label=label)
        ax.set_title(name.replace("_", " "))
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "distributions.png", dpi=140)
    plt.close(fig)
