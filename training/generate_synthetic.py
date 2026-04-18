"""Procedural synthetic engineering-drawing generator.

Produces PNGs with ground-truth bounding boxes for 4 layout classes:
  0 title_block    1 view    2 dimension_cluster    3 free_text

Rationale: the upstream `teddyz829/Data-Augmentation-Engineering-Drawing`
tool operates on DXF files and targets binary component segmentation
(Zhang et al., ASME IDETC 2022). It does not emit layout-level bounding
boxes. This module draws engineering-drawing-like primitives directly
with Pillow and writes COCO-format annotations, which is what DETR
fine-tuning needs.

The drawings are intentionally schematic: rectangular views with
internal lines, arrow-dimension pairs with numeric labels, a title
block in the bottom-right corner and free-text blocks elsewhere.
This is sufficient to exercise the full fine-tuning + CoreML
conversion pipeline and to sanity-check that the network learns the
class-spatial priors (title_block bottom-right, dimension clusters
adjacent to views, etc.).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = ["title_block", "view", "dimension_cluster", "free_text"]
CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# --- font handling -----------------------------------------------------------

def _find_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


# --- primitive drawers -------------------------------------------------------

@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str

    @property
    def xywh(self) -> list[int]:
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]


@dataclass
class Canvas:
    img: Image.Image
    draw: ImageDraw.ImageDraw
    boxes: list[Box] = field(default_factory=list)

    def rect(self, b: Box, width: int = 2):
        self.draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline="black", width=width)


def draw_view(c: Canvas, x1: int, y1: int, x2: int, y2: int, rng: random.Random):
    b = Box(x1, y1, x2, y2, "view")
    c.rect(b, width=2)
    # random internal geometry — lines, sub-rects, a circle or two
    for _ in range(rng.randint(4, 12)):
        if rng.random() < 0.35:
            cx = rng.randint(x1 + 10, x2 - 10)
            cy = rng.randint(y1 + 10, y2 - 10)
            r = rng.randint(6, min(40, (x2 - x1) // 4, (y2 - y1) // 4))
            c.draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", width=1)
        else:
            a1 = (rng.randint(x1 + 5, x2 - 5), rng.randint(y1 + 5, y2 - 5))
            a2 = (rng.randint(x1 + 5, x2 - 5), rng.randint(y1 + 5, y2 - 5))
            c.draw.line([a1, a2], fill="black", width=1)
    c.boxes.append(b)


def draw_dimension_cluster(c: Canvas, x1: int, y1: int, x2: int, y2: int,
                           rng: random.Random, font: ImageFont.FreeTypeFont):
    b = Box(x1, y1, x2, y2, "dimension_cluster")
    # Horizontal extension + dimension line with arrows
    n_lines = rng.randint(2, 5)
    for i in range(n_lines):
        y = y1 + (i + 1) * (y2 - y1) // (n_lines + 1)
        c.draw.line([(x1 + 4, y), (x2 - 4, y)], fill="black", width=1)
        c.draw.line([(x1 + 4, y - 3), (x1 + 4, y + 3)], fill="black", width=1)
        c.draw.line([(x2 - 4, y - 3), (x2 - 4, y + 3)], fill="black", width=1)
        # arrow heads
        c.draw.polygon([(x1 + 4, y), (x1 + 10, y - 3), (x1 + 10, y + 3)], fill="black")
        c.draw.polygon([(x2 - 4, y), (x2 - 10, y - 3), (x2 - 10, y + 3)], fill="black")
        # dimension text centered above
        val = rng.randint(5, 9999)
        txt = f"{val}"
        tw = c.draw.textlength(txt, font=font)
        tx = (x1 + x2) // 2 - tw // 2
        ty = y - font.size - 2
        if ty > y1:
            c.draw.text((tx, ty), txt, fill="black", font=font)
    c.boxes.append(b)


def draw_title_block(c: Canvas, x1: int, y1: int, x2: int, y2: int,
                     rng: random.Random, font: ImageFont.FreeTypeFont):
    b = Box(x1, y1, x2, y2, "title_block")
    c.rect(b, width=2)
    # 2-3 rows and 2-3 columns of internal grid lines
    n_rows = rng.randint(2, 4)
    n_cols = rng.randint(2, 3)
    for i in range(1, n_rows):
        y = y1 + i * (y2 - y1) // n_rows
        c.draw.line([(x1, y), (x2, y)], fill="black", width=1)
    for j in range(1, n_cols):
        x = x1 + j * (x2 - x1) // n_cols
        c.draw.line([(x, y1), (x, y2)], fill="black", width=1)
    # fill random cells with labels
    words = ["DRAWN", "CHECKED", "APPROVED", "SCALE", "DATE", "SHEET",
             "REV", "DWG NO", "TITLE", "PART"]
    for i in range(n_rows):
        for j in range(n_cols):
            if rng.random() < 0.7:
                cx = x1 + j * (x2 - x1) // n_cols + 4
                cy = y1 + i * (y2 - y1) // n_rows + 3
                c.draw.text((cx, cy), rng.choice(words), fill="black", font=font)
    c.boxes.append(b)


def draw_free_text(c: Canvas, x1: int, y1: int, x2: int, y2: int,
                   rng: random.Random, font: ImageFont.FreeTypeFont):
    b = Box(x1, y1, x2, y2, "free_text")
    lines = rng.randint(2, 5)
    prefixes = ["NOTE:", "REV:", "TOL:", "FINISH:", "MATERIAL:", "SCALE:"]
    for i in range(lines):
        y = y1 + i * font.size + i * 2
        if y + font.size > y2:
            break
        txt = f"{rng.choice(prefixes)} {rng.randint(1, 999)}-{rng.randint(100, 9999)}"
        c.draw.text((x1 + 2, y), txt, fill="black", font=font)
    c.boxes.append(b)


# --- page composer -----------------------------------------------------------

def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int],
                   pad: int = 8) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + pad < bx1 or bx2 + pad < ax1 or
                ay2 + pad < by1 or by2 + pad < ay1)


def _place(used: list[tuple[int, int, int, int]],
           rng: random.Random,
           W: int, H: int,
           wmin: int, wmax: int, hmin: int, hmax: int,
           region: tuple[int, int, int, int] | None = None,
           max_tries: int = 30) -> tuple[int, int, int, int] | None:
    rx1, ry1, rx2, ry2 = region if region else (40, 40, W - 40, H - 40)
    for _ in range(max_tries):
        w = rng.randint(wmin, min(wmax, rx2 - rx1))
        h = rng.randint(hmin, min(hmax, ry2 - ry1))
        x = rng.randint(rx1, rx2 - w)
        y = rng.randint(ry1, ry2 - h)
        r = (x, y, x + w, y + h)
        if any(_rects_overlap(r, u) for u in used):
            continue
        return r
    return None


def render_page(image_size: int, rng: random.Random) -> Canvas:
    W = H = image_size
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, W - 8, H - 8], outline="black", width=2)  # border
    canvas = Canvas(img=img, draw=draw)
    font_small = _find_font(11)
    font_tiny = _find_font(9)
    used: list[tuple[int, int, int, int]] = []

    # --- title block in bottom-right corner (canonical location)
    tb_w = rng.randint(int(W * 0.22), int(W * 0.35))
    tb_h = rng.randint(int(H * 0.10), int(H * 0.18))
    tb = (W - 30 - tb_w, H - 30 - tb_h, W - 30, H - 30)
    used.append(tb)
    draw_title_block(canvas, *tb, rng=rng, font=font_tiny)

    # --- 1 to 3 views, biased to the upper half / left
    n_views = rng.randint(1, 3)
    for _ in range(n_views):
        r = _place(used, rng, W, H,
                   wmin=int(W * 0.18), wmax=int(W * 0.45),
                   hmin=int(H * 0.18), hmax=int(H * 0.40),
                   region=(30, 30, W - 30, int(H * 0.7)))
        if r is None:
            continue
        used.append(r)
        draw_view(canvas, *r, rng=rng)

    # --- 1 to 3 dimension clusters, near views (just place anywhere non-overlapping)
    n_dim = rng.randint(1, 3)
    for _ in range(n_dim):
        r = _place(used, rng, W, H,
                   wmin=int(W * 0.15), wmax=int(W * 0.35),
                   hmin=int(H * 0.05), hmax=int(H * 0.15),
                   region=(30, 30, W - 30, H - 40))
        if r is None:
            continue
        used.append(r)
        draw_dimension_cluster(canvas, *r, rng=rng, font=font_tiny)

    # --- 1 to 3 free-text blocks
    n_txt = rng.randint(1, 3)
    for _ in range(n_txt):
        r = _place(used, rng, W, H,
                   wmin=int(W * 0.12), wmax=int(W * 0.28),
                   hmin=int(H * 0.06), hmax=int(H * 0.14),
                   region=(30, 30, W - 30, H - 40))
        if r is None:
            continue
        used.append(r)
        draw_free_text(canvas, *r, rng=rng, font=font_small)

    return canvas


# --- COCO writer -------------------------------------------------------------

def write_coco(out_dir: Path, split: str, items: list[tuple[str, Canvas]]):
    images = []
    anns = []
    ann_id = 1
    for image_id, (fname, canvas) in enumerate(items, start=1):
        images.append({
            "id": image_id,
            "file_name": fname,
            "width": canvas.img.size[0],
            "height": canvas.img.size[1],
        })
        for b in canvas.boxes:
            anns.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": CLASS_ID[b.label],
                "bbox": b.xywh,
                "area": (b.x2 - b.x1) * (b.y2 - b.y1),
                "iscrowd": 0,
            })
            ann_id += 1
    coco = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)],
    }
    (out_dir / f"{split}.json").write_text(json.dumps(coco))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("synthetic_data"))
    ap.add_argument("--n", type=int, default=200, help="total drawings")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = args.out
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)

    n_val = max(1, int(args.n * args.val_frac))
    n_train = args.n - n_val

    train_items, val_items = [], []
    for i in range(n_train):
        c = render_page(args.size, rng)
        fname = f"train/{i:05d}.png"
        c.img.save(out / fname)
        train_items.append((fname, c))
    for i in range(n_val):
        c = render_page(args.size, rng)
        fname = f"val/{i:05d}.png"
        c.img.save(out / fname)
        val_items.append((fname, c))

    write_coco(out, "train", train_items)
    write_coco(out, "val", val_items)
    print(f"wrote {n_train} train + {n_val} val drawings to {out}")


if __name__ == "__main__":
    main()
