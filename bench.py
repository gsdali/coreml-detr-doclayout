"""Measure CoreML inference latency on the fine-tuned DETR model.

Runs the model N times on a random 512×512 image, reports p50/p95/mean
latency, and writes `benchmarks.json`. Also dumps the ANE/GPU/CPU
dispatch counts as reported by coremltools (best-effort — the Xcode
Performance tab is the authoritative source).

Usage:
    python bench.py \
        --model artefacts/detr_doclayout_512.mlpackage \
        --iters 30 \
        --out benchmarks.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--image", type=Path, default=None,
                    help="optional fixture image; random if omitted")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("benchmarks.json"))
    ap.add_argument("--extra", type=str, default=None,
                    help="JSON string merged into the output (e.g. mAP from evaluate.py)")
    args = ap.parse_args()

    import coremltools as ct

    mlmodel = ct.models.MLModel(str(args.model),
                                compute_units=ct.ComputeUnit.ALL)
    desc = mlmodel.get_spec().description
    input_name = desc.input[0].name
    print(f"model: {args.model}")
    print(f"input: {input_name}")

    if args.image and args.image.exists():
        img = Image.open(args.image).convert("RGB").resize((512, 512))
    else:
        img = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
            mode="RGB")

    # Core ML wants a PIL image matching the declared ImageType.
    feed = {input_name: img}

    # warmup
    for _ in range(args.warmup):
        _ = mlmodel.predict(feed)

    timings = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _ = mlmodel.predict(feed)
        timings.append((time.perf_counter() - t0) * 1000.0)

    p50 = statistics.median(timings)
    p95 = sorted(timings)[int(0.95 * len(timings)) - 1]
    mean = statistics.fmean(timings)

    record = {
        "model": str(args.model),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "coremltools": ct.__version__,
        "iters": args.iters,
        "latency_ms": {
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "mean": round(mean, 2),
            "min": round(min(timings), 2),
            "max": round(max(timings), 2),
        },
        "input_size": "1x3x512x512",
        "compute_units": "ALL (ANE preferred, GPU/CPU fallback)",
    }
    if args.extra:
        record.update(json.loads(args.extra))

    args.out.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
