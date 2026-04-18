# coreml-detr-doclayout

DETR (`facebook/detr-resnet-50`) fine-tuned for engineering-document
layout detection, converted to CoreML.

## Status

**Pipeline-complete, trained on synthetic data only.** See
`NOTES.md` in the upstream session workspace and the Limitations
section below for the data-collection blocker that prevents shipping
a production-quality fine-tune.

## Classes

Four layout classes, suitable for engineering drawings:

| id | label              | description                                                         |
|----|--------------------|---------------------------------------------------------------------|
| 0  | `title_block`      | standard drawing title block (typically bottom-right)               |
| 1  | `view`             | orthographic or isometric view region                               |
| 2  | `dimension_cluster`| grouped dimension lines, tolerances, leaders                        |
| 3  | `free_text`        | notes, revisions, parts lists, scale/date stamps                    |

## Repo contents

```
artefacts/
  detr_doclayout_512.mlpackage   FP16 CoreML, (1,3,512,512) input
convert.py                       trace + coremltools conversion
training/
  fine_tune.py                   training loop (PyTorch MPS / CUDA)
  dataset_loader.py              COCO-format loader
  generate_synthetic.py          procedural 4-class synthetic generator
requirements.txt
test_ane.swift                   Swift CoreML harness
test_images/
  sample_layout.png              synthetic example
  expected_output.json           reference detections
benchmarks.json                  latency + mAP
```

## Quick start

```bash
pip install -r requirements.txt

# Generate synthetic training data (200 drawings by default)
python training/generate_synthetic.py --out synthetic_data --n 200

# Fine-tune DETR (num_queries=30, 4 classes)
python training/fine_tune.py \
    --data synthetic_data \
    --epochs 20 \
    --out checkpoints/detr_doclayout.pt

# Convert to CoreML
python convert.py \
    --ckpt checkpoints/detr_doclayout.pt \
    --out artefacts/detr_doclayout_512.mlpackage
```

## Swift usage

```swift
let model = try MLModel(contentsOf: compiledURL,
                        configuration: {
                            let c = MLModelConfiguration()
                            c.computeUnits = .all
                            return c
                        }())
let prediction = try model.prediction(from: provider)
let logits = prediction.featureValue(for: "logits")!.multiArrayValue!
let boxes  = prediction.featureValue(for: "boxes")!.multiArrayValue!
// See test_ane.swift for a full parsing example.
```

Output format:
- `logits`: `(1, num_queries, num_classes + 1)` — raw logits; the
  last class index is "no object". Softmax across classes, take the
  argmax over the first `num_classes`.
- `boxes`: `(1, num_queries, 4)` — cxcywh, normalised to `[0, 1]`.

DETR does not use NMS. Filter detections by score threshold only.

## Architectural changes vs stock DETR

- `num_queries`: `100 → 30`. Engineering drawings rarely contain more
  than ~30 layout regions. Reduces decoder cost substantially.
- `num_classes`: `91 (COCO) → 4`. Classification head reinitialised.
- Input size: `512 × 512`. Standard DETR supports arbitrary sizes;
  fixing the input makes the CoreML trace deterministic.

## Benchmarks

See `benchmarks.json` for the numbers measured in this workspace.
The ResNet-50 backbone dispatches to ANE; the transformer decoder
typically falls back to GPU/CPU — this is expected for DETR.

## Limitations

1. **Training data is synthetic-only.** No hand-labeled engineering
   drawings were available in-session. The model learns the
   procedural template but will not generalise well to real CAD
   drawings. mAP on the synthetic validation set is indicative only.
2. **The `teddyz829/Data-Augmentation-Engineering-Drawing` tool is
   scoped to binary component segmentation** (Zhang et al., 2022),
   not 4-class layout bounding boxes. Its DXF output also depends on
   a 2019-era `ezdxf==0.11.1` / `dxfgrabber==1.0.1` stack that is
   not trivial to set up on modern Python. This repo's
   `training/generate_synthetic.py` is a pragmatic replacement: it
   draws engineering-drawing-like primitives directly with Pillow
   and emits COCO annotations with the 4 target classes.
3. **`tecs` / `SMC` / `Schneider` drawings are not labeled.** The
   `test_images/` sample and the three fixture drawings in the
   session workspace's `_shared/test-drawings/` are used only for
   qualitative inspection, never for quantitative evaluation.

To productionise this model you need to (a) hand-label ~50 real
engineering drawings with 4-class bboxes, (b) retrain starting from
the provided checkpoint, (c) re-run `convert.py`.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.

Base model: DETR (Detection Transformer), © Facebook AI Research,
Apache-2.0.

Synthetic-data design draws on Zhang et al., "Data Augmentation of
Engineering Drawings for Data-driven Component Segmentation" (ASME
IDETC 2022), MIT License.
