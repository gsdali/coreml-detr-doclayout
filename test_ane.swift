// test_ane.swift
//
// Minimal Swift harness that loads the CoreML DETR doc-layout model,
// runs it against a 512×512 image, and prints detections above a
// confidence threshold.
//
// Usage:
//   swift test_ane.swift artefacts/detr_doclayout_512.mlpackage \
//       test_images/sample_layout.png
//
// Build the compiled model once with Xcode ("coreml-compile") or call
// MLModel.compileModel(at:) below. For Apple Neural Engine profiling,
// open the .mlpackage in Xcode > Performance tab and target an M-series
// device.

import Foundation
import CoreML
import CoreImage
#if canImport(AppKit)
import AppKit
#endif

let CLASSES = ["title_block", "view", "dimension_cluster", "free_text"]
let INPUT_SIZE = 512
let SCORE_THRESHOLD: Float = 0.5

struct Detection {
    let label: String
    let confidence: Float
    let bbox: CGRect  // normalised cxcywh converted to xyxy in [0,1]
}

func loadCGImage(path: String) -> CGImage? {
#if canImport(AppKit)
    guard let img = NSImage(contentsOfFile: path) else { return nil }
    var rect = CGRect(x: 0, y: 0, width: img.size.width, height: img.size.height)
    return img.cgImage(forProposedRect: &rect, context: nil, hints: nil)
#else
    return nil
#endif
}

func makePixelBuffer(from cgImage: CGImage, size: Int) -> CVPixelBuffer? {
    var pb: CVPixelBuffer?
    let attrs: [CFString: Any] = [
        kCVPixelBufferCGImageCompatibilityKey: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey: true,
    ]
    CVPixelBufferCreate(kCFAllocatorDefault, size, size,
                        kCVPixelFormatType_32BGRA,
                        attrs as CFDictionary, &pb)
    guard let buf = pb else { return nil }
    CVPixelBufferLockBaseAddress(buf, [])
    defer { CVPixelBufferUnlockBaseAddress(buf, []) }
    let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buf),
                        width: size, height: size,
                        bitsPerComponent: 8,
                        bytesPerRow: CVPixelBufferGetBytesPerRow(buf),
                        space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                                   | CGBitmapInfo.byteOrder32Little.rawValue)
    ctx?.interpolationQuality = .high
    ctx?.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
    return buf
}

func softmax(_ x: [Float]) -> [Float] {
    let m = x.max() ?? 0
    let e = x.map { expf($0 - m) }
    let s = e.reduce(0, +)
    return e.map { $0 / s }
}

func parseDetections(logits: MLMultiArray, boxes: MLMultiArray) -> [Detection] {
    // logits shape: (1, Q, num_classes + 1); last index is "no object"
    // boxes  shape: (1, Q, 4) — cxcywh normalised to [0,1]
    let q = logits.shape[1].intValue
    let c = logits.shape[2].intValue
    var out: [Detection] = []
    for i in 0..<q {
        var row = [Float](repeating: 0, count: c)
        for k in 0..<c {
            row[k] = logits[[0, i as NSNumber, k as NSNumber]].floatValue
        }
        let p = softmax(row)
        var best = 0
        var bestP: Float = 0
        for k in 0..<(c - 1) {  // skip no-object class
            if p[k] > bestP { bestP = p[k]; best = k }
        }
        if bestP < SCORE_THRESHOLD { continue }
        let cx = boxes[[0, i as NSNumber, 0]].floatValue
        let cy = boxes[[0, i as NSNumber, 1]].floatValue
        let w  = boxes[[0, i as NSNumber, 2]].floatValue
        let h  = boxes[[0, i as NSNumber, 3]].floatValue
        let rect = CGRect(x: CGFloat(cx - w/2), y: CGFloat(cy - h/2),
                          width: CGFloat(w), height: CGFloat(h))
        let label = best < CLASSES.count ? CLASSES[best] : "cls\(best)"
        out.append(Detection(label: label, confidence: bestP, bbox: rect))
    }
    return out
}

// --- main ---
let args = CommandLine.arguments
guard args.count >= 3 else {
    print("usage: test_ane.swift <model.mlpackage> <image.png>")
    exit(1)
}
let modelURL = URL(fileURLWithPath: args[1])
let imageURL = URL(fileURLWithPath: args[2])

let compiledURL = try MLModel.compileModel(at: modelURL)
let cfg = MLModelConfiguration()
cfg.computeUnits = .all
let model = try MLModel(contentsOf: compiledURL, configuration: cfg)

guard let cg = loadCGImage(path: imageURL.path),
      let pb = makePixelBuffer(from: cg, size: INPUT_SIZE) else {
    print("failed to load image: \(imageURL.path)"); exit(2)
}
let featureName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
let provider = try MLDictionaryFeatureProvider(
    dictionary: [featureName: MLFeatureValue(pixelBuffer: pb)])

// warmup + timed run
_ = try model.prediction(from: provider)
let start = CFAbsoluteTimeGetCurrent()
let result = try model.prediction(from: provider)
let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0

guard let logits = result.featureValue(for: "logits")?.multiArrayValue,
      let boxes  = result.featureValue(for: "boxes")?.multiArrayValue else {
    print("missing logits/boxes outputs"); exit(3)
}
let dets = parseDetections(logits: logits, boxes: boxes)
print(String(format: "inference: %.2f ms", elapsed))
print("detections: \(dets.count)")
for d in dets {
    print(String(format: "  %-16s %.2f  [%.2f, %.2f, %.2f, %.2f]",
                 d.label, d.confidence,
                 d.bbox.minX, d.bbox.minY, d.bbox.maxX, d.bbox.maxY))
}
