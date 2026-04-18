"""COCO-format dataset loader for DETR fine-tuning.

Supports two layouts:

1.  Directory layout (produced by `generate_synthetic.py`):

        root/
          train.json       COCO JSON with image paths relative to root
          val.json
          train/<id>.png
          val/<id>.png

2.  Zip layout (produced by `prepare_triview2cad.py`):

        split_file          COCO JSON
        image_zip           zip archive containing <prefix>/<file_name>

    Passed via CocoLayoutDataset(split_file=..., image_zip=...,
    image_prefix="img_files").

The loader also supports remapping the categories in the source JSON
to our 0-indexed schema (title_block=0, view=1, dimension_cluster=2,
free_text=3) — TriView2CAD uses 1-indexed COCO categories in a
different order.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Mapping

import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ["title_block", "view", "dimension_cluster", "free_text"]
CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


class CocoLayoutDataset(Dataset):
    """COCO-format dataset returning (PIL image, target dict).

    Target dict follows the HF DETR convention:
        {"image_id": int, "annotations": [{bbox, category_id, area,
                                           iscrowd, id}, ...]}
    which DetrImageProcessor then converts into model-ready targets.

    Args:
      root / split:  directory layout mode.
      split_file:    path to a COCO JSON (zip mode).
      image_zip:     path to a zip containing the images (zip mode).
      image_prefix:  subdirectory within the zip (e.g. "img_files").
      category_remap: dict mapping source category_id → target id in
                     CLASS_ID space. Annotations with categories not
                     in the remap are dropped.
    """

    def __init__(self,
                 root: Path | None = None,
                 split: str | None = None,
                 split_file: Path | None = None,
                 image_zip: Path | None = None,
                 image_prefix: str = "",
                 category_remap: Mapping[int, int] | None = None):
        if split_file is not None:
            coco = json.loads(Path(split_file).read_text())
            self._zip_path = Path(image_zip) if image_zip else None
            self._zip = None  # lazy open per-worker
            self._prefix = image_prefix.rstrip("/") + "/" if image_prefix else ""
            self._root = None
        elif root is not None and split is not None:
            coco = json.loads((Path(root) / f"{split}.json").read_text())
            self._zip_path = None
            self._prefix = ""
            self._root = Path(root)
        else:
            raise ValueError("Pass either (root, split) or split_file")

        self.images = {im["id"]: im for im in coco["images"]}
        self.anns_by_image: dict[int, list[dict]] = {}
        remap = category_remap or {}
        for a in coco["annotations"]:
            cid = a["category_id"]
            if remap:
                if cid not in remap:
                    continue  # drop classes not in remap
                a = {**a, "category_id": remap[cid]}
            self.anns_by_image.setdefault(a["image_id"], []).append(a)
        self.image_ids = sorted(self.images.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def _open_image(self, file_name: str) -> Image.Image:
        if self._zip_path is not None:
            # Lazy-open the zip per-worker — ZipFile handles are not
            # safe across forked DataLoader workers.
            if self._zip is None:
                self._zip = zipfile.ZipFile(self._zip_path, "r")
            with self._zip.open(self._prefix + file_name) as f:
                return Image.open(io.BytesIO(f.read())).convert("RGB")
        return Image.open(self._root / file_name).convert("RGB")

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        im = self.images[image_id]
        img = self._open_image(im["file_name"])
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
