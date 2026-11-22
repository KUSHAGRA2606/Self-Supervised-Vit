import json
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image, ImageDraw


def read_export(path, expected):
    masks = {}
    for task in json.loads(path.read_text()):
        image = unquote(task["data"]["image"]).split("?d=")[-1]
        key = Path(image).stem
        if key not in expected:
            raise ValueError(f"Unexpected image {key}")
        entries = [a for a in task["annotations"] if not a.get("was_cancelled")]
        if len(entries) != 1 or key in masks:
            raise ValueError(f"Expected one annotation for {key}")
        objects = []
        for region in entries[0]["result"]:
            if region["type"] != "polygonlabels":
                continue
            if (region["original_width"], region["original_height"]) != (160, 160):
                raise ValueError(f"Unexpected dimensions for {key}")
            value = region["value"]
            label, = value["polygonlabels"]
            if label not in {"Target galaxy", "Other galaxy", "Star"}:
                raise ValueError(f"Unexpected label {label}")
            points = [(x * 1.6, y * 1.6) for x, y in value["points"]]
            if len(points) < 3 or any(not 0 <= v <= 160 for point in points for v in point):
                raise ValueError(f"Invalid polygon for {key}")
            canvas = Image.new("1", (160, 160))
            ImageDraw.Draw(canvas).polygon(points, fill=1)
            objects.append((label, np.asarray(canvas)))
        targets = [i for i, (label, _) in enumerate(objects, 1) if label == "Target galaxy"]
        if len(targets) != 1 or len(objects) > 255:
            raise ValueError(f"Expected one target and at most 255 objects for {key}")
        instances = np.zeros((160, 160), dtype=np.uint8)
        for i, (label, region) in enumerate(objects, 1):
            instances[region & (instances == 0)] = i
        target = targets[0]
        instances[objects[target - 1][1]] = target
        masks[key] = (instances, target)
    if set(masks) != set(expected):
        raise ValueError(f"Missing annotations for {len(set(expected) - set(masks))} images")
    return masks
