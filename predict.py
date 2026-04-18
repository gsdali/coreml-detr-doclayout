"""Run the converted CoreML DETR model on an image and print detections.

Python counterpart to `test_ane.swift`. Takes an .mlpackage and one
or more PNG paths, runs inference, applies a score threshold, and
prints / saves the detections.

Usage:
    python predict.py \
        --model artefacts/detr_doclayout_512.mlpackage \
        --image test_images/sample_layout.png \
        --threshold 0.5 \
        --save-to eval_out/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = ["title_block", "view", "dimension_cluster", "free_text"]
CLASS_COLORS = {"title_block": "red", "view": "blue",
                "dimension_cluster": "green", "free_text": "orange"}
INPUT_SIZE = 512


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _cxcywh_norm_to_xyxy(boxes: np.ndarray, W: int, H: int) -> np.ndarray:
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return np.stack([x1, y1, x2, y2], axis=-1)


def parse_detections(logits: np.ndarray, boxes: np.ndarray,
                     W: int, H: int, threshold: float) -> list[dict]:
    """logits: (1, Q, num_classes + 1); boxes: (1, Q, 4) cxcywh normalised.

    Last class index is "no-object". Softmax over classes, argmax over
    the first num_classes (exclude no-object).
    """
    probs = _softmax(logits[0], axis=-1)
    probs = probs[:, :-1]  # drop no-object column
    best_cls = probs.argmax(axis=-1)
    best_score = probs.max(axis=-1)
    xyxy = _cxcywh_norm_to_xyxy(boxes[0], W, H)
    dets = []
    for i in range(logits.shape[1]):
        if best_score[i] < threshold:
            continue
        dets.append({
            "label": CLASS_NAMES[int(best_cls[i])],
            "score": float(best_score[i]),
            "bbox_xyxy": [float(v) for v in xyxy[i]],
        })
    dets.sort(key=lambda d: -d["score"])
    return dets


def _draw(image: Image.Image, dets: list[dict]) -> Image.Image:
    im = image.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for d in dets:
        color = CLASS_COLORS.get(d["label"], "purple")
        draw.rectangle(d["bbox_xyxy"], outline=color, width=3)
        x, y, _, _ = d["bbox_xyxy"]
        draw.text((x + 2, y + 2), f'{d["label"]} {d["score"]:.2f}',
                  fill=color, font=font)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--image", type=Path, nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save-to", type=Path, default=None,
                    help="write annotated PNGs + a JSON next to --image")
    args = ap.parse_args()

    import coremltools as ct
    mlmodel = ct.models.MLModel(str(args.model),
                                compute_units=ct.ComputeUnit.ALL)
    input_name = mlmodel.get_spec().description.input[0].name

    all_records = {}
    for path in args.image:
        pil = Image.open(path).convert("RGB")
        orig_W, orig_H = pil.size
        resized = pil.resize((INPUT_SIZE, INPUT_SIZE))
        out = mlmodel.predict({input_name: resized})
        logits = np.asarray(out["logits"])
        boxes = np.asarray(out["boxes"])
        dets = parse_detections(logits, boxes, orig_W, orig_H, args.threshold)
        print(f"{path}: {len(dets)} detections")
        for d in dets:
            b = d["bbox_xyxy"]
            print(f"  {d['label']:18s} score={d['score']:.2f} "
                  f"bbox=[{b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f}]")
        all_records[path.name] = dets

        if args.save_to is not None:
            args.save_to.mkdir(parents=True, exist_ok=True)
            annotated = _draw(pil, dets)
            annotated.save(args.save_to / f"pred_{path.stem}.png")

    if args.save_to is not None:
        (args.save_to / "predictions.json").write_text(
            json.dumps(all_records, indent=2))


if __name__ == "__main__":
    main()
