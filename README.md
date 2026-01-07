# QiVision

[中文版](README_CN.md)

**QiVision** is an industrial machine vision library built from scratch in C++17, without OpenCV dependency. Designed to match Halcon's core functionality and precision.

## Features

- **Zero Dependencies** - Only uses stb_image for file I/O
- **Sub-pixel Precision** - < 0.02px accuracy for edge detection
- **Halcon Concepts** - Domain, XLD contours, RLE regions
- **Modern C++17** - Clean API, RAII design
- **SIMD Optimized** - AVX2/SSE acceleration

## Status

| Layer | Progress | Description |
|-------|----------|-------------|
| Core | 80% | QImage, QRegion, QContour, QMatrix |
| Platform | 70% | Memory, SIMD, Thread, Timer |
| Internal | 45% | Gaussian, Gradient, Edge, Fitting, Contour |
| Feature | 0% | ShapeModel, Caliper, Blob, OCR (planned) |

## Quick Start

### Requirements

- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.16+

### Build

```bash
git clone https://github.com/userqz1/QiVision.git
cd QiVision
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Run Tests

```bash
./build/bin/unit_test
```

### Run Samples

```bash
./build/bin/samples/01_basic_image
./build/bin/samples/06_contour_segment
```

## Use in Your Project

**CMake FetchContent:**

```cmake
include(FetchContent)
FetchContent_Declare(QiVision
    GIT_REPOSITORY https://github.com/userqz1/QiVision.git
    GIT_TAG main)
FetchContent_MakeAvailable(QiVision)
target_link_libraries(your_app PRIVATE QiVision)
```

**As Subdirectory:**

```cmake
add_subdirectory(QiVision)
target_link_libraries(your_app PRIVATE QiVision)
```

## Examples

See the [samples/](samples/) folder for complete examples:

- `01_basic_image.cpp` - Image creation, pixel access, save/load
- `06_contour_segment.cpp` - Contour segmentation into lines and arcs

### Basic Image Example

```cpp
#include <QiVision/QiVision.h>
using namespace Qi::Vision;

int main() {
    // Create image
    QImage img(640, 480, PixelType::UInt8, ChannelType::Gray);

    // Fill pixels
    for (int32_t y = 0; y < img.Height(); ++y) {
        uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int32_t x = 0; x < img.Width(); ++x) {
            row[x] = static_cast<uint8_t>((x + y) % 256);
        }
    }

    // Save
    img.SaveToFile("output.png");

    // Load
    QImage loaded = QImage::FromFile("input.png");

    return 0;
}
```

### Contour Segmentation Example

```cpp
#include <QiVision/QiVision.h>
#include <QiVision/Internal/ContourSegment.h>
using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

int main() {
    // Create L-shaped contour
    QContour contour;
    for (int i = 0; i <= 50; ++i) contour.AddPoint(i, 0);
    for (int i = 1; i <= 50; ++i) contour.AddPoint(50, i);

    // Segment into primitives
    SegmentParams params;
    params.mode = SegmentMode::LinesOnly;
    SegmentationResult result = SegmentContour(contour, params);

    printf("Found %zu lines\n", result.LineCount());
    return 0;
}
```

## Architecture

```
┌───────────────────────────────────────────────────┐
│ API: QImage, QRegion, QContour, QMatrix           │
├───────────────────────────────────────────────────┤
│ Feature: ShapeModel, Caliper, Blob, OCR (planned) │
├───────────────────────────────────────────────────┤
│ Internal: Gaussian, Gradient, Canny, Steger,      │
│           Fitting, ContourProcess, ContourSegment │
├───────────────────────────────────────────────────┤
│ Platform: Memory, SIMD, Thread, Timer, FileIO     │
└───────────────────────────────────────────────────┘
```

## Precision Targets

| Module | Target |
|--------|--------|
| Edge1D | < 0.02 px |
| CircleFit | < 0.02 px |
| LineFit | < 0.005° |

## Documentation

- [PROGRESS.md](PROGRESS.md) - Development progress
- [samples/](samples/) - Example programs

## License

MIT License
