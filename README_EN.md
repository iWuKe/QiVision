<p align="center">
  <h1 align="center">QiVision</h1>
  <p align="center">
    <strong>Industrial machine vision library (C++17) with Halcon‑style API and sub‑pixel accuracy</strong>
  </p>
</p>

<p align="center">
    English | <a href="./README.md">简体中文</a>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/C++-17-blue.svg" alt="C++17">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
    <img src="https://img.shields.io/badge/Platform-Windows%20|%20Linux-lightgrey.svg" alt="Platform">
    <img src="https://img.shields.io/badge/SIMD-AVX2%20|%20SSE4-orange.svg" alt="SIMD">
    <img src="https://img.shields.io/badge/Dependencies-stb__image%20only-brightgreen.svg" alt="Dependencies">
</p>

---

## Positioning

QiVision targets industrial vision workflows with Halcon‑style APIs, Domain/XLD/ROI semantics, sub‑pixel measurement, and high‑performance matching. It is suitable for production alignment, inspection, metrology, barcode/OCR, and geometric analysis.

---

## Core Capabilities

- Template matching: ShapeModel (gradient shape), NCCModel (gray correlation), rotation/scale support
- Component matching: ComponentModel with relative constraints
- Metrology: calipers and metrology models (line/circle/ellipse/rectangle)
- Morphology/segmentation/blob: thresholding, connected components, filtering
- Contours & geometry: XLD, fitting, transforms, Hough
- Calibration & distortion: camera model, undistortion, fisheye model (partial)
- OCR/Barcode: optional modules (ONNXRuntime / ZXing)

---

## Performance & Accuracy (brief)

- Sub‑pixel measurement: < 0.03 px in typical caliper scenarios
- Shape matching: 0–360° with pyramid + SIMD acceleration
- Low dependency: only stb_image for image I/O

---

## Quick Start

### Build (Linux)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Run a sample

```bash
./build/bin/samples/matching_shape_match
```

---

## Build & Runtime Configuration

```bash
# Build tests
cmake -B build -DQIVISION_BUILD_TESTS=ON

# Build samples
cmake -B build -DQIVISION_BUILD_SAMPLES=ON

# GUI display & interactive windows
cmake -B build -DQIVISION_BUILD_GUI=ON

# Optional modules
cmake -B build -DQIVISION_BUILD_OCR=ON -DQIVISION_BUILD_BARCODE=ON
```

Runtime notes:
- `samples/*` binaries are placed under `build/bin/samples/`
- OCR/Barcode require their dependencies to be available (see module docs)

---

## Sample Entry Points

- `samples/matching_shape_match`
- `samples/matching_ncc_match`
- `samples/matching_component_model`
- `samples/measure_circle_metrology`
- `samples/blob_analysis`

---

## Progress & Docs

- Progress: [PROGRESS.md](PROGRESS.md)
- API reference: [docs/API_Reference.md](docs/API_Reference.md)
- Troubleshooting: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- Coding style: [docs/CODING_STYLE.md](docs/CODING_STYLE.md)
- Samples: [samples/](samples/)

---

## License

MIT License
