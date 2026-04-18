"""Convert a fine-tuned DETR checkpoint to CoreML (FP16, mlprogram).

Wraps the HuggingFace DETR model in a thin `nn.Module` that:
  - takes a single image tensor (1, 3, 512, 512) normalised in [0, 1]
  - applies ImageNet mean/std normalisation inside the graph
  - returns (logits, pred_boxes) only — no pixel_mask, no aux losses

Fixing the input shape to 512×512 is intentional: it lets coremltools
trace a fully concrete graph and lets Core ML dispatch the ResNet
backbone to ANE.

Usage:
    python convert.py \
        --ckpt checkpoints/detr_doclayout.pt \
        --out artefacts/detr_doclayout_512.mlpackage
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).parent / "training"))
from dataset_loader import CLASS_NAMES  # noqa: E402
from fine_tune import build_model  # noqa: E402


# ImageNet statistics used by DetrImageProcessor
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _disable_mask_machinery():
    """Disable transformers' attention-mask construction.

    The new `create_bidirectional_mask` helper (transformers >=5.5)
    takes a code path during `torch.jit.trace` that crashes on a 0-d
    symbolic tensor. DETR on a full 512×512 image never needs a
    padding mask, so returning None here is equivalent for our
    conversion path. Safe no-op on older transformers.
    """
    import transformers.models.detr.modeling_detr as _detr
    if hasattr(_detr, "create_bidirectional_mask"):
        _detr.create_bidirectional_mask = (
            lambda *_args, **_kwargs: None)


class DetrCoreMLWrapper(nn.Module):
    """Wrap HF DETR so the traced graph has a single image input."""

    def __init__(self, model):
        super().__init__()
        _disable_mask_machinery()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor):
        # image is in [0, 1]; normalise to the ImageNet range DETR was trained on.
        x = (image - self.mean) / self.std
        out = self.model(pixel_values=x, return_dict=True)
        # Return only what downstream Swift code needs.
        return out.logits, out.pred_boxes


def load_model(ckpt_path: Path | None, num_queries: int, num_classes: int):
    model = build_model(num_classes=num_classes, num_queries=num_queries)
    if ckpt_path is not None and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = state["state_dict"] if "state_dict" in state else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  load: {len(missing)} missing keys (first: {missing[:3]})")
        if unexpected:
            print(f"  load: {len(unexpected)} unexpected keys "
                  f"(first: {unexpected[:3]})")
    else:
        print("  warning: no checkpoint provided; converting pretrained-only "
              "weights. This is a pipeline-validation path, not a real model.")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="optional fine-tuned checkpoint from fine_tune.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--num-queries", type=int, default=30)
    ap.add_argument("--num-classes", type=int, default=len(CLASS_NAMES))
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()

    model = load_model(args.ckpt, args.num_queries, args.num_classes)
    wrapper = DetrCoreMLWrapper(model).eval()

    example = torch.rand(1, 3, args.size, args.size)
    print("tracing / exporting...")
    with torch.no_grad():
        # Sanity — make sure the wrapper forward runs cleanly.
        logits, boxes = wrapper(example)
        print(f"  logits {tuple(logits.shape)}  boxes {tuple(boxes.shape)}")

    print("converting with coremltools...")
    import coremltools as ct
    precision = ct.precision.FLOAT16 if args.fp16 else ct.precision.FLOAT32

    ct_inputs = [ct.ImageType(name="image",
                              shape=(1, 3, args.size, args.size),
                              scale=1.0 / 255.0,
                              bias=[0, 0, 0],
                              color_layout=ct.colorlayout.RGB)]
    ct_outputs = [ct.TensorType(name="logits"), ct.TensorType(name="boxes")]

    # Two conversion paths. `torch.export` is preferred — it produces
    # a concrete FX graph that coremltools handles more robustly for
    # transformer models. Fall back to `torch.jit.trace` if export is
    # unavailable or fails.
    mlmodel = None
    try:
        print("  trying torch.export path...")
        exported = torch.export.export(wrapper, (example,))
        # coremltools needs the ATEN/EDGE dialect, not TRAINING. Decompose.
        exported = exported.run_decompositions({})
        mlmodel = ct.convert(
            exported,
            inputs=ct_inputs,
            outputs=ct_outputs,
            convert_to="mlprogram",
            compute_precision=precision,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15,
        )
    except Exception as e:
        print(f"  torch.export path failed: {e}")
        print("  falling back to torch.jit.trace path...")
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, example, strict=False)
        mlmodel = ct.convert(
            traced,
            inputs=ct_inputs,
            outputs=ct_outputs,
            convert_to="mlprogram",
            compute_precision=precision,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15,
        )

    # Metadata — shows up in Xcode and in MLModel.modelDescription.
    mlmodel.short_description = (
        "DETR fine-tuned for engineering-drawing layout (4 classes)")
    mlmodel.author = "gsdali"
    mlmodel.license = "Apache-2.0"
    mlmodel.user_defined_metadata["classes"] = ",".join(CLASS_NAMES)
    mlmodel.user_defined_metadata["num_queries"] = str(args.num_queries)
    mlmodel.user_defined_metadata["input_size"] = str(args.size)
    mlmodel.user_defined_metadata["base_model"] = "facebook/detr-resnet-50"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.out))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
