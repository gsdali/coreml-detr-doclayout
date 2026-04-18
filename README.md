# coreml-detr-doclayout

DETR (`facebook/detr-resnet-50`) fine-tuned for engineering-document
layout detection, converted to CoreML.

## Status

**Partial fine-tune on real data (view + dimension_cluster).** The
model detects `view` regions on real engineering drawings (AP@0.5 =
**0.112** on held-out TriView2CAD val) and has weak-but-nonzero
signal on `dimension_cluster` (AP = 0.006). The two other classes
in the schema (`title_block`, `free_text`) were **not present** in
the training set, so the model does not learn to detect them here.

Inference latency on M4 ANE: **p50 ≈ 12 ms** for a 512×512 image —
well under the 80 ms success threshold in the session plan.

See `NOTES.md` in the upstream session workspace for the full data
story (synthetic → real unblock via TriView2CAD auto-labels).

## Classes

Four layout classes, suitable for engineering drawings:

| id | label              | description                                                         | trained? |
|----|--------------------|---------------------------------------------------------------------|----------|
| 0  | `title_block`      | standard drawing title block (typically bottom-right)               | **no**   |
| 1  | `view`             | orthographic or isometric view region                               | **yes**  |
| 2  | `dimension_cluster`| grouped dimension lines, tolerances, leaders                        | **yes**  |
| 3  | `free_text`        | notes, revisions, parts lists, scale/date stamps                    | **no**   |

The schema and model head always expose four classes. The current
weights just never saw positive examples of classes 0 or 3, so the
model won't produce them with meaningful confidence until it's
retrained on a dataset that includes them.

## Repo contents

```
artefacts/
  detr_doclayout_512.mlpackage    FP16 CoreML, (1,3,512,512) input
convert.py                        trace + coremltools conversion
training/
  fine_tune.py                    training loop (PyTorch MPS / CUDA)
  dataset_loader.py               COCO-format loader (dir + zip modes)
  generate_synthetic.py           procedural 4-class synthetic generator
  prepare_triview2cad.py          COCO shuffle + train/val split
  evaluate.py                     simple mAP + visualisation
predict.py                        Python CoreML inference + parsing
bench.py                          latency benchmark
requirements.txt
test_ane.swift                    Swift CoreML harness
test_images/
  sample_layout.png               real TriView2CAD val example
  expected_output.json            GT annotations for that image
benchmarks.json                   latency + mAP
```

## Quick start

Two data modes are supported:

### A) Procedural synthetic (Pillow-drawn, fast iteration):

```bash
pip install -r requirements.txt
python training/generate_synthetic.py --out synthetic_data --n 500
python training/fine_tune.py \
    --data synthetic_data --epochs 30 --batch-size 4 \
    --out checkpoints/detr_doclayout.pt
```

### B) TriView2CAD real composites (2-class subset, what's shipped):

```bash
python training/prepare_triview2cad.py \
    --in  ~/mlwd/datasets/TriView2CAD-direct/coco_labels_10k.json \
    --out real_data_10k --val-frac 0.1

python training/fine_tune.py \
    --train-json real_data_10k/train.json \
    --val-json   real_data_10k/val.json \
    --image-zip  ~/mlwd/datasets/TriView2CAD-direct/img_files.zip \
    --image-prefix img_files \
    --freeze-backbone \
    --epochs 4 --batch-size 4 --lr 5e-5 \
    --out checkpoints/detr_doclayout.pt
```

### Convert and use:

```bash
python convert.py \
    --ckpt checkpoints/detr_doclayout.pt \
    --out artefacts/detr_doclayout_512.mlpackage

python predict.py \
    --model artefacts/detr_doclayout_512.mlpackage \
    --image test_images/sample_layout.png \
    --threshold 0.5 --save-to eval_out/
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
- Input size: `512 × 512`. Fixing the input makes the CoreML trace
  deterministic.
- Backbone frozen during fine-tuning — the pretrained ResNet-50 is
  already a strong feature extractor for monochrome line drawings and
  unfreezing it destabilised training on this dataset.

## Benchmarks

From `benchmarks.json` (M4, macOS 26.4, CoreML 8.3):

| metric         | value  |
|----------------|--------|
| p50 latency    | **~12 ms** |
| mean latency   | ~13 ms |
| mAP@0.5 (`view`) | **0.112** |
| mAP@0.5 (`dim_cluster`) | 0.006 |

The ResNet-50 backbone dispatches to ANE; the transformer decoder
typically falls back to GPU/CPU — this is expected for DETR.

## Limitations

1. **Only two classes are trained.** The TriView2CAD dataset has
   ground-truth bounding boxes for `view` and `dimension_cluster`
   only. `title_block` and `free_text` categories exist in the
   schema and the model's output head but have no positive training
   examples yet. To cover all four, you need a dataset containing
   them — ~50 hand- or auto-labelled tecs/SMC drawings would
   typically be enough given transfer learning.

2. **`dimension_cluster` AP is low.** The auto-labeller that
   produced `coco_labels_10k.json` makes the dimension-cluster
   boxes fairly coarse: they can bleed across views, and the
   clusters themselves are defined by a heuristic rather than a
   crisp bounding criterion. DETR's set-prediction loss tolerates
   a bit of box noise but not this much. A crisper label scheme
   or mixing in tighter hand-labels should help.

3. **Not evaluated on hand-labelled real drawings.** The three
   fixture drawings in the session workspace's
   `_shared/test-drawings/` (`tecs_c10`, `tecs_el04`, `telefono`)
   were used only for qualitative inspection; their qualitative
   behaviour — view boxes land on views, dim-cluster boxes land on
   dimension stacks — suggests the model is learning useful priors,
   but there is no quantitative real-drawing mAP.

4. **No evaluation on Xcode Performance tab.** The ~12 ms p50
   latency is consistent with ANE-heavy dispatch; in-Xcode
   profiling would give a per-op breakdown but is a manual UI step.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.

Base model: DETR (Detection Transformer), © Facebook AI Research,
Apache-2.0.

Training images from the TriView2CAD dataset (three-view composite
drawings generated from public DXF corpora).
