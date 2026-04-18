"""Fine-tune facebook/detr-resnet-50 for 4-class engineering-drawing layout.

Reduces num_queries from 100 to 30 because engineering drawings very
rarely contain more than ~30 layout regions, and a smaller decoder is
cheaper on ANE/GPU.

Usage:
    python training/fine_tune.py \
        --data synthetic_data \
        --epochs 20 \
        --out checkpoints/detr_doclayout.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor

from dataset_loader import CLASS_NAMES, CocoLayoutDataset, make_collate_fn

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "facebook/detr-resnet-50"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, num_queries: int) -> DetrForObjectDetection:
    """Load pretrained DETR, reinitialise class head + query embeddings."""
    id2label = {i: n for i, n in enumerate(CLASS_NAMES)}
    label2id = {n: i for i, n in enumerate(CLASS_NAMES)}
    model = DetrForObjectDetection.from_pretrained(
        MODEL_ID,
        num_labels=num_classes,
        num_queries=num_queries,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


def train_one_epoch(model, loader, optimizer, device, epoch, log_every=10):
    model.train()
    total, n_steps = 0.0, 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device) if "pixel_mask" in batch else None
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        optimizer.zero_grad()
        out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = out.loss
        if not torch.isfinite(loss):
            print(f"  step {step}: non-finite loss, skipping")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total += float(loss.item())
        n_steps += 1
        if step % log_every == 0:
            print(f"  epoch {epoch} step {step:4d}  loss {loss.item():.4f}")
    dur = time.time() - t0
    return total / max(1, n_steps), dur


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    total, n_steps = 0.0, 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device) if "pixel_mask" in batch else None
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
        out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        if torch.isfinite(out.loss):
            total += float(out.loss.item())
            n_steps += 1
    return total / max(1, n_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None,
                    help="root containing train.json, val.json, train/, val/ "
                         "(directory layout). Mutually exclusive with --train-json.")
    ap.add_argument("--train-json", type=Path, default=None,
                    help="COCO JSON for the train split (zip layout)")
    ap.add_argument("--val-json", type=Path, default=None,
                    help="COCO JSON for the val split (zip layout)")
    ap.add_argument("--image-zip", type=Path, default=None,
                    help="zip archive containing the images")
    ap.add_argument("--image-prefix", type=str, default="img_files",
                    help="subdirectory within --image-zip")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="freeze ResNet-50 backbone weights")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-queries", type=int, default=30)
    ap.add_argument("--image-size", type=int, default=512,
                    help="matches convert.py input shape")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="MPS + multiprocessing is flaky; 0 is safest")
    ap.add_argument("--max-steps-per-epoch", type=int, default=None,
                    help="cap for quick smoke training")
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume from a prior checkpoint file")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # --- processor pinned to the training image size
    processor = DetrImageProcessor.from_pretrained(
        MODEL_ID,
        size={"shortest_edge": args.image_size, "longest_edge": args.image_size},
        do_pad=True,
    )
    collate = make_collate_fn(processor)

    # TriView2CAD uses 1-indexed categories in a different order than
    # ours (see dataset_loader.py). Remap at load time.
    triview_remap = {1: CLASS_NAMES.index("view"),
                     2: CLASS_NAMES.index("dimension_cluster"),
                     3: CLASS_NAMES.index("title_block"),
                     4: CLASS_NAMES.index("free_text")}

    if args.train_json is not None:
        train_ds = CocoLayoutDataset(split_file=args.train_json,
                                     image_zip=args.image_zip,
                                     image_prefix=args.image_prefix,
                                     category_remap=triview_remap)
        val_ds = CocoLayoutDataset(split_file=args.val_json,
                                   image_zip=args.image_zip,
                                   image_prefix=args.image_prefix,
                                   category_remap=triview_remap)
    elif args.data is not None:
        train_ds = CocoLayoutDataset(args.data, "train")
        val_ds = CocoLayoutDataset(args.data, "val")
    else:
        raise SystemExit("Pass either --data or --train-json/--val-json")
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate)

    model = build_model(num_classes=len(CLASS_NAMES),
                        num_queries=args.num_queries)
    if args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"resumed from {args.resume}  "
              f"(epoch {state.get('epoch')}, val_loss {state.get('val_loss'):.4f})"
              if "val_loss" in state else f"resumed from {args.resume}")
        if missing:
            print(f"  load: {len(missing)} missing")
        if unexpected:
            print(f"  load: {len(unexpected)} unexpected")
    if args.freeze_backbone:
        n_frozen = 0
        for n, p in model.named_parameters():
            if "backbone" in n:
                p.requires_grad = False
                n_frozen += p.numel()
        print(f"froze backbone: {n_frozen / 1e6:.1f}M params")

    model.to(device)

    # Separate LR for backbone — standard DETR recipe.
    backbone_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if "backbone" in n else other_params).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr_backbone},
        ],
        weight_decay=args.weight_decay,
    )

    history = []
    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        # optionally cap steps for smoke runs
        loader = train_loader
        if args.max_steps_per_epoch is not None:
            loader = _take(train_loader, args.max_steps_per_epoch)
        train_loss, dur = train_one_epoch(model, loader, optimizer, device, epoch)
        val_loss = eval_loss(model, val_loader, device)
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"({dur:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "seconds": dur})
        payload = {
            "state_dict": model.state_dict(),
            "config": model.config.to_dict(),
            "num_queries": args.num_queries,
            "classes": CLASS_NAMES,
            "image_size": args.image_size,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        # Always keep a "last" checkpoint so we can recover even when
        # val_loss oscillates without improving.
        last_path = args.out.with_name(args.out.stem + "_last" + args.out.suffix)
        torch.save(payload, last_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, args.out)
            print(f"  ↳ saved {args.out} (new best val_loss={val_loss:.4f})")

    (args.out.parent / f"{args.out.stem}_history.json").write_text(
        json.dumps(history, indent=2))


def _take(loader, n):
    """Yield at most n batches from a DataLoader (for smoke runs)."""
    for i, b in enumerate(loader):
        if i >= n:
            break
        yield b


if __name__ == "__main__":
    main()
