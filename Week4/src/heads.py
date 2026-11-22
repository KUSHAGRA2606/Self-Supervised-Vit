from contextlib import nullcontext
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


SETUPS = {
    "linear_224": ("dinov2_vits14", 224, 1),
    "linear_448": ("dinov2_vits14", 448, 1),
    "linear_s8": ("dino_vits8", 224, 1),
    "ms": ("dinov2_vits14", 448, 4),
}


class Masks(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        key, image_path, mask_path = self.records[index]
        with np.load(image_path) as data:
            image = torch.from_numpy(data["rgb"]).permute(2, 0, 1)
        mask = torch.from_numpy(np.load(mask_path).astype(np.int64))
        return key, image, mask


@lru_cache
def backbone(name, device):
    repos = {
        "dino_vits8": "facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a",
        "dinov2_vits14": "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8",
    }
    repo = repos[name]
    return torch.hub.load(repo, name, trust_repo=True).to(device).eval().requires_grad_(False)


def preprocess(image, size, device):
    image = image.to(device).float() / 255
    image = F.interpolate(image, size=(size, size), mode="bilinear",
                          align_corners=False, antialias=True)
    mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
    std = image.new_tensor([.229, .224, .225])[None, :, None, None]
    return (image - mean) / std


@torch.no_grad()
def features(model, image, size, layers, device):
    x = preprocess(image, size, device)
    amp = torch.autocast("cuda", dtype=torch.float16) if torch.device(device).type == "cuda" else nullcontext()
    with amp:
        if hasattr(model, "patch_size"):
            return torch.cat(model.get_intermediate_layers(x, n=layers, reshape=True), dim=1)
        tokens = model.get_intermediate_layers(x, n=1)[0][:, 1:]
        return tokens.transpose(1, 2).reshape(len(x), 384, size // 8, size // 8)


def make_head(layers):
    return nn.Sequential(nn.BatchNorm2d(384 * layers), nn.Conv2d(384 * layers, 3, 1))


def logits(head, tokens, shape):
    return F.interpolate(head(tokens.float()), size=shape, mode="bilinear", align_corners=False)


@torch.no_grad()
def predict(head, model, image, setup, device, shape=(160, 160), tta=False):
    _, size, layers = setup
    views = [(s, flip) for s in (336, 448, 560) for flip in (False, True)] if tta else [(size, False)]
    scores = []
    for scale, flip in views:
        view = image.flip(-1) if flip else image
        probs = logits(head, features(model, view, scale, layers, device), shape).softmax(1)
        scores.append(probs.flip(-1) if flip else probs)
    return torch.stack(scores).mean(0).argmax(1).cpu().numpy().astype(np.uint8)


@torch.no_grad()
def naive_attention(model, image, device):
    # Reuse the Week 3 union of each head's top 60% attention mass.
    attention = model.get_last_selfattention(preprocess(image, 224, device))[:, :, 0, 1:]
    values, order = attention.sort(dim=-1, descending=True)
    selected = (values.cumsum(-1) - values) < 0.6 * values.sum(-1, keepdim=True)
    masks = torch.zeros_like(selected).scatter_(-1, order, selected).any(dim=1)
    resized = F.interpolate(masks.reshape(-1, 1, 28, 28).float(), (160, 160), mode="nearest")
    return resized[:, 0].bool().cpu().numpy()


def train_head(setup, train_loader, val_loader, device, checkpoint, score, epochs=20,
               feature_cache=None):
    model = backbone(setup[0], device)
    head = make_head(setup[2]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    counts = np.zeros(3, dtype=np.int64)
    for _, _, labels in train_loader:
        counts += torch.bincount(labels.flatten(), minlength=3).numpy()
    weights = 1 / np.maximum(counts, 1)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights / weights.mean(),
                                                       dtype=torch.float32, device=device))
    if feature_cache is not None:
        feature_cache.mkdir(parents=True, exist_ok=True)
    best = -np.inf
    history = []

    def batch_tokens(keys, images):
        if feature_cache is None:
            return features(model, images, *setup[1:], device)
        paths = [feature_cache / f"{key}.npy" for key in keys]
        missing = [i for i, path in enumerate(paths) if not path.exists()]
        if missing:
            tokens = features(model, images[missing], *setup[1:], device).cpu().numpy()
            for i, token in zip(missing, tokens):
                np.save(paths[i], token.astype(np.float16))
        return torch.from_numpy(np.stack([np.load(path) for path in paths])).to(device)

    for epoch in range(epochs):
        head.train()
        mean_loss = 0
        for keys, images, labels in train_loader:
            optimizer.zero_grad()
            output = logits(head, batch_tokens(keys, images), labels.shape[-2:])
            loss = loss_fn(output, labels.to(device))
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
        head.eval()
        counts = np.zeros((3, 3), dtype=np.int64)
        with torch.no_grad():
            for keys, images, labels in val_loader:
                output = logits(head, batch_tokens(keys, images), labels.shape[-2:])
                counts += score(labels.numpy(), output.argmax(1).cpu().numpy())
        from metrics import iou_summary
        value = iou_summary(counts)["miou"]
        history.append({"epoch": epoch + 1, "loss": mean_loss / len(train_loader),
                        "validation_miou": value})
        print(history[-1])
        if value > best:
            best = value
            torch.save({"setup": setup, "state": head.state_dict(), "epoch": epoch + 1}, checkpoint)
    return history


def load_head(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    setup = saved["setup"]
    head = make_head(setup[2]).to(device)
    head.load_state_dict(saved["state"])
    return head.eval(), backbone(setup[0], device), setup
