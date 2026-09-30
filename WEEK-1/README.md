# Week 1: Looking at galaxies with DINOv2

This week we learn how a **Vision Transformer (ViT)** reads an image and try a pretrained one, **DINOv2**, on real galaxy images.

We don't train anything. We only use DINOv2 to describe galaxies and look at what it outputs.


## The idea in short

DINOv2 cuts an image into small **14×14-pixel squares (patches)**. A 224×224 image gives **16 × 16 = 256 patches**. It turns each patch into **768 numbers** and adds one extra token called **CLS**, which collects a summary of the whole image. So one galaxy becomes:

- **CLS token (1 × 768):** a description of the whole galaxy
- **Patch tokens (256 × 768):** one description for each small square

## Data

- **DESI Legacy Survey DR10**: galaxy images of 160×160 pixels in 4 colour filters (g, r, i, z).
- **Galaxy Zoo DESI**: volunteers votes on what each galaxy looks like (smooth, spiral, barred, etc).

Both are streamed from Hugging Face, so nothing large is downloaded. They are separate datasets and are not matched up yet.

## What the notebook does

1. Load 32 DESI galaxies and look at their pixel values and catalogue data
2. Show each filter, then combine z, r, g into one colour image (DINOv2 needs RGB)
3. Look at the Galaxy Zoo labels and turn vote counts into vote fractions
4. Load DINOv2, resize images to 224×224, run them through it
5. Pull out the CLS and patch tokens and check their shapes
6. Plot the patch grid to see what the model picks up
7. Use the CLS tokens of 300 Galaxy Zoo galaxies to check whether the model "knows" galaxy shapes

## Main findings

**Our DESI galaxies are small, faint and red.** Most are only about 3 pixels across and look yellow-orange, which suggests they're far away. Some pixels are negative, which is normal: the sky brightness has been subtracted and the leftover noise can dip below zero.

**The CLS summary focuses on the galaxy, not the empty sky.** In the patch grid (right column), the squares most similar to the CLS token sit on the bright sources. The PCA colours (middle column) look mostly random, because each galaxy covers only a few of the 256 squares.

![Patch grid](figures/patch_grid.png)

**DINOv2 already separates smooth and featured galaxies without being taught.** Smooth galaxies (blue) and featured ones (red) land on different sides of the plot.

![CLS tokens](figures/cls_pca.png)

**Similar galaxies get similar CLS tokens.** Searching for a spiral returns spirals, and a round blob returns round blobs.

![Nearest neighbours](figures/nearest_neighbours.png)

**A simple classifier on the CLS tokens gets 84.7% accuracy** at telling smooth from featured galaxies, compared with 74% for always guessing "smooth".

## Limitations

- DINOv2 learned from everyday photos, not telescope images
- We drop the i band to make a 3-colour image
- The DESI galaxies we used are tiny, so the patch plots are mostly empty sky
- The good results in the last step come from the Galaxy Zoo images, which show bigger and clearer galaxies than our DESI cutouts

## How to run

Open `Week1_DINOv2_DESI.ipynb` in Kaggle and run all cells.

