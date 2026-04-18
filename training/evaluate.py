"""Evaluate a fine-tuned DETR checkpoint.

Computes a simple mean-AP metric over synthetic val images and saves
visualisations of predictions on an arbitrary set of test images
(e.g. the tecs drawings from the session workspace).

Not a replacement for pycocotools COCOeval — this is intentionally
self-contained to avoid a heavy dep during smoke validation. The
computed mAP uses a single IoU threshold (0.5) and averages AP per
class with PR-curve integration.

Usage:
    python training/evaluate.py \
        --ckpt checkpoints/detr_doclayout.pt \
        --data synthetic_data \
        --test-images test_images/sample_layout.png ../../../_shared/test-drawings/tecs_c10.png \
        --out-dir eval_out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).parent))
from dataset_loader import CLASS_NAMES, CocoLayoutDataset, make_collate_fn
from fine_tune import build_model, pick_device

from transformers import DetrImageProcessor

CLASS_COLORS = {
    "title_block": "red",
    "view": "blue",
    "dimension_cluster": "green",
    "free_text": "orange",
}


def load_ckpt(ckpt_path: Path, num_queries: int = 30, num_classes: int = 4):
    model = build_model(num_classes=num_classes, num_queries=num_queries)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def compute_ap(records: list[tuple[float, bool]], n_pos: int) -> float:
    """VOC-style AP: rank-sort predictions, compute precision/recall
    curve, integrate under monotonised PR curve.
    """
    if n_pos == 0:
        return 0.0 if records else 1.0
    records = sorted(records, key=lambda r: -r[0])
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    for _, is_tp in records:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
    # monotonic precision envelope + trapezoidal integration over recall
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * max(0.0, r - prev_r)
        prev_r = r
    return ap


@torch.no_grad()
def run_model(model, processor, image: Image.Image, device) -> dict:
    # DetrImageProcessor handles resize + normalisation.
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model(**inputs)
    target_sizes = torch.tensor([[image.height, image.width]], device=device)
    results = processor.post_process_object_detection(
        out, threshold=0.0, target_sizes=target_sizes)[0]
    return {
        "scores": results["scores"].cpu().tolist(),
        "labels": results["labels"].cpu().tolist(),
        "boxes": results["boxes"].cpu().tolist(),  # xyxy in original image coords
    }


def evaluate_map(model, processor, val_root: Path, device,
                 score_thresh: float = 0.0,
                 iou_thresh: float = 0.5,
                 image_zip: Path | None = None,
                 image_prefix: str = "",
                 category_remap: dict | None = None) -> dict:
    if image_zip is not None:
        ds = CocoLayoutDataset(split_file=val_root / "val.json",
                               image_zip=image_zip,
                               image_prefix=image_prefix,
                               category_remap=category_remap)
    else:
        ds = CocoLayoutDataset(val_root, "val")
    # gt[cls] = [(image_id, xyxy, matched_flag)]
    gt = defaultdict(list)
    preds = defaultdict(list)  # preds[cls] = [(score, is_tp)]
    n_pos = defaultdict(int)

    for idx in range(len(ds)):
        image, target = ds[idx]
        image_id = target["image_id"]
        for ann in target["annotations"]:
            x, y, w, h = ann["bbox"]
            gt[ann["category_id"]].append(
                {"image_id": image_id, "xyxy": [x, y, x + w, y + h],
                 "matched": False})
            n_pos[ann["category_id"]] += 1

        out = run_model(model, processor, image, device)
        # match greedy by score, per class
        per_class_preds = defaultdict(list)
        for s, l, b in zip(out["scores"], out["labels"], out["boxes"]):
            if s < score_thresh:
                continue
            per_class_preds[l].append((s, b))
        for cls, plist in per_class_preds.items():
            plist.sort(key=lambda x: -x[0])
            image_gts = [g for g in gt[cls] if g["image_id"] == image_id]
            for score, box in plist:
                best_i, best_iou = -1, 0.0
                for i, g in enumerate(image_gts):
                    if g["matched"]:
                        continue
                    iou = iou_xyxy(box, g["xyxy"])
                    if iou > best_iou:
                        best_iou, best_i = iou, i
                if best_i >= 0 and best_iou >= iou_thresh:
                    image_gts[best_i]["matched"] = True
                    preds[cls].append((score, True))
                else:
                    preds[cls].append((score, False))

    per_class_ap = {}
    for cls_id in range(len(CLASS_NAMES)):
        per_class_ap[CLASS_NAMES[cls_id]] = compute_ap(
            preds.get(cls_id, []), n_pos.get(cls_id, 0))
    m_ap = sum(per_class_ap.values()) / len(per_class_ap)
    return {"mAP@0.5": m_ap, "per_class_AP@0.5": per_class_ap,
            "n_images": len(ds), "n_gt": dict(n_pos)}


def visualise(model, processor, image_path: Path, out_path: Path,
              device, score_thresh: float = 0.5):
    image = Image.open(image_path).convert("RGB")
    out = run_model(model, processor, image, device)
    kept = [(s, l, b) for s, l, b in zip(out["scores"], out["labels"], out["boxes"])
            if s >= score_thresh]

    im = image.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for s, l, b in kept:
        name = CLASS_NAMES[l] if 0 <= l < len(CLASS_NAMES) else str(l)
        color = CLASS_COLORS.get(name, "purple")
        draw.rectangle(b, outline=color, width=3)
        draw.text((b[0] + 2, b[1] + 2), f"{name} {s:.2f}",
                  fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return [{"label": CLASS_NAMES[l] if 0 <= l < len(CLASS_NAMES) else str(l),
             "score": float(s), "bbox_xyxy": [float(v) for v in b]}
            for s, l, b in kept]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("synthetic_data"))
    ap.add_argument("--image-zip", type=Path, default=None,
                    help="zip archive for zip-mode data (e.g. TriView2CAD)")
    ap.add_argument("--image-prefix", type=str, default="img_files")
    ap.add_argument("--test-images", nargs="*", type=Path, default=[])
    ap.add_argument("--out-dir", type=Path, default=Path("eval_out"))
    ap.add_argument("--num-queries", type=int, default=30)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": args.image_size, "longest_edge": args.image_size},
        do_pad=True,
    )
    model = load_ckpt(args.ckpt, num_queries=args.num_queries).to(device)

    # --- mAP on val
    triview_remap = {1: CLASS_NAMES.index("view"),
                     2: CLASS_NAMES.index("dimension_cluster"),
                     3: CLASS_NAMES.index("title_block"),
                     4: CLASS_NAMES.index("free_text")} if args.image_zip else None
    metrics = evaluate_map(model, processor, args.data, device,
                           image_zip=args.image_zip,
                           image_prefix=args.image_prefix,
                           category_remap=triview_remap)
    print("mAP@0.5:", metrics["mAP@0.5"])
    for k, v in metrics["per_class_AP@0.5"].items():
        print(f"  {k:18s}  AP={v:.3f}  (n_gt={metrics['n_gt'].get(CLASS_NAMES.index(k), 0)})")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- visualisations
    vis_records = {}
    for p in args.test_images:
        if not p.exists():
            print(f"  skipping missing: {p}")
            continue
        out_path = args.out_dir / f"pred_{p.stem}.png"
        dets = visualise(model, processor, p, out_path, device,
                         score_thresh=args.score_thresh)
        vis_records[p.name] = dets
        print(f"  wrote {out_path}  ({len(dets)} detections)")
    (args.out_dir / "visualisations.json").write_text(
        json.dumps(vis_records, indent=2))


if __name__ == "__main__":
    main()
