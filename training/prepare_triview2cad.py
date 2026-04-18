"""Split a TriView2CAD COCO JSON into train / val splits for DETR.

The upstream labels (`coco_labels_10k.json`) use 1-indexed categories
in a different order than our schema:

    upstream:  1=view, 2=dimension_cluster, 3=title_block, 4=free_text
    ours:      0=title_block, 1=view, 2=dimension_cluster, 3=free_text

The loader handles the remap at runtime via `category_remap=`, so
this script just shuffles and splits.

Usage:
    python training/prepare_triview2cad.py \\
        --in  ~/mlwd/datasets/TriView2CAD-direct/coco_labels_10k.json \\
        --out real_data \\
        --val-frac 0.15 \\
        --limit 3000 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N images (after shuffle)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coco = json.loads(args.inp.read_text())
    rng = random.Random(args.seed)
    image_ids = [im["id"] for im in coco["images"]]
    rng.shuffle(image_ids)
    if args.limit is not None:
        image_ids = image_ids[:args.limit]

    n_val = max(1, int(len(image_ids) * args.val_frac))
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])

    def subset(ids: set[int]) -> dict:
        ids_list = sorted(ids)
        imgs = [im for im in coco["images"] if im["id"] in ids]
        anns = [a for a in coco["annotations"] if a["image_id"] in ids]
        return {
            "info": coco.get("info", {}),
            "categories": coco["categories"],
            "images": imgs,
            "annotations": anns,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.json").write_text(json.dumps(subset(train_ids)))
    (args.out / "val.json").write_text(json.dumps(subset(val_ids)))
    print(f"train: {len(train_ids)} images  "
          f"val: {len(val_ids)} images  "
          f"total: {len(image_ids)}")


if __name__ == "__main__":
    main()
