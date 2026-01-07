# QiVision

Industrial machine vision algorithm library implemented from scratch in C++17, without OpenCV dependency.

**Target**: Match Halcon's core functionality and accuracy for industrial vision applications.

## Status

🚧 **Work in Progress** - Internal algorithm layer under active development.

| Layer | Status | Description |
|-------|--------|-------------|
| Core | ✅ 80% | QImage, QRegion, QContour, QMatrix |
| Platform | ✅ 70% | Memory, SIMD, Thread, Timer |
| Internal | 🟡 40% | Gaussian, Gradient, Edge, Fitting, Contour ops |
| Feature | ⬜ 0% | ShapeModel, Caliper, Blob, OCR (not started) |

## Features (Planned)

- **Shape Matching** - Template matching with rotation/scale invariance
- **Measurement** - Caliper, metrology tools with sub-pixel accuracy
- **Blob Analysis** - Connected component analysis
- **Edge Detection** - Canny, Steger sub-pixel edges
- **Camera Calibration** - Zhang's method, distortion correction
- **OCR/Barcode** - Character and barcode recognition

## Requirements

- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.16+
- No external dependencies (only stb_image for I/O, included)

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Run Tests

```bash
./build/bin/unit_test
./build/bin/accuracy_test
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ API Layer: QImage, QRegion, QContour, QMatrix               │
├─────────────────────────────────────────────────────────────┤
│ Feature Layer: Matching, Measure, Edge, Blob, OCR, Calib    │
├─────────────────────────────────────────────────────────────┤
│ Internal Layer: Gradient, Fitting, Steger, Hessian, etc.    │
├─────────────────────────────────────────────────────────────┤
│ Platform Layer: Memory, Thread, SIMD, Timer                 │
└─────────────────────────────────────────────────────────────┘
```

## Precision Targets

| Module | Metric | Target |
|--------|--------|--------|
| Edge1D | Position | < 0.02 px (1σ) |
| ShapeModel | Position/Angle | < 0.05 px / < 0.05° |
| CircleFit | Center/Radius | < 0.02 px |

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

Contributions welcome! See [PROGRESS.md](PROGRESS.md) for current development status.
