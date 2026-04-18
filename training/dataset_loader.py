"""COCO-format dataset loader for DETR fine-tuning.

Expects the layout produced by `generate_synthetic.py`:

    root/
      train.json       COCO JSON with image paths relative to root
      val.json
      train/<id>.png
      val/<id>.png

DETR in HuggingFace expects annotations in the COCO style. We use
`DetrImageProcessor` to handle resizing, normalisation, and target
encoding (cxcywh normalised boxes) consistently with the pretrained
model.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ["title_block", "view", "dimension_cluster", "free_text"]


class CocoLayoutDataset(Dataset):
    """Minimal COCO-format dataset returning (PIL image, target dict).

    The target dict follows the HF DETR convention:
        {"image_id": int, "annotations": [{bbox, category_id, area,
                                           iscrowd, id}, ...]}
    which DetrImageProcessor then converts into model-ready targets.
    """

    def __init__(self, root: Path, split: str):
        self.root = Path(root)
        coco = json.loads((self.root / f"{split}.json").read_text())
        self.images = {im["id"]: im for im in coco["images"]}
        # group annotations by image_id
        self.anns_by_image: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            self.anns_by_image.setdefault(a["image_id"], []).append(a)
        self.image_ids = sorted(self.images.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        im = self.images[image_id]
        img = Image.open(self.root / im["file_name"]).convert("RGB")
        target = {
            "image_id": image_id,
            "annotations": self.anns_by_image.get(image_id, []),
        }
        return img, target


def make_collate_fn(processor):
    """Builds a collate_fn that wraps DetrImageProcessor.

    Returns a dict with pixel_values, pixel_mask, and labels — ready
    to splat into the DETR model's forward.
    """

    def _collate(batch):
        images = [b[0] for b in batch]
        targets = [b[1] for b in batch]
        encoded = processor(images=images, annotations=targets,
                            return_tensors="pt")
        return encoded

    return _collate
