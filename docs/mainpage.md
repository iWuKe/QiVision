# QiVision - Industrial Machine Vision Library {#mainpage}

\tableofcontents

## Introduction

**QiVision** is a professional industrial machine vision library providing Halcon-like functionality with sub-pixel accuracy. The library is fully self-implemented with only stb_image for I/O operations.

### Key Features

- **Sub-pixel Accuracy**: Edge detection <0.02px, Shape matching <0.05px
- **Industrial Grade**: Designed for machine vision applications
- **Modern C++17**: Clean API with SIMD/OpenMP optimization
- **Zero Dependencies**: Only stb_image for I/O

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/userqz1/QiVision.git
cd QiVision

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Basic Example

```cpp
#include <QiVision/QiVision.h>

using namespace Qi::Vision;

int main() {
    // Load image
    QImage image;
    ReadImage("test.png", image);

    // Apply Gaussian filter
    QImage filtered;
    Filter::GaussFilter(image, filtered, 1.5);

    // Edge detection
    QContourArray contours;
    Edge::EdgesSubPix(filtered, contours, 30, 60, 1.0);

    // Save result
    WriteImage("result.png", filtered);
    return 0;
}
```

---

## Module Overview

### Core Data Structures

| Class | Description |
|-------|-------------|
| @ref Qi::Vision::QImage "QImage" | Image container with ROI (Domain) support |
| @ref Qi::Vision::QRegion "QRegion" | Run-length encoded region |
| @ref Qi::Vision::QContour "QContour" | XLD contour with sub-pixel coordinates |
| @ref Qi::Vision::QMatrix "QMatrix" | Matrix for geometric transforms |

### Feature Modules

| Module | Description |
|--------|-------------|
| @ref Qi::Vision::Filter "Filter" | Gaussian, Median, Bilateral, Sobel... |
| @ref Qi::Vision::Segment "Segment" | Threshold, Otsu, Adaptive, DynThreshold |
| @ref Qi::Vision::Morphology "Morphology" | Erosion, Dilation, Opening, Closing |
| @ref Qi::Vision::Blob "Blob" | Connected components, SelectShape |
| @ref Qi::Vision::Edge "Edge" | Canny, Steger sub-pixel edge detection |
| @ref Qi::Vision::Matching "Matching" | Shape-based and NCC template matching |
| @ref Qi::Vision::Measure "Measure" | Caliper, Metrology tools |
| @ref Qi::Vision::Contour "Contour" | XLD processing, fitting, segmentation |
| @ref Qi::Vision::Hough "Hough" | Line and circle detection |
| @ref Qi::Vision::Transform "Transform" | Affine, Polar, Homography |
| @ref Qi::Vision::Calib "Calib" | Camera calibration, undistortion |
| @ref Qi::Vision::Defect "Defect" | Variation model for defect detection |
| @ref Qi::Vision::Texture "Texture" | LBP, GLCM, Gabor texture analysis |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ API Layer: QImage, QRegion, QContour, QMatrix               │
├─────────────────────────────────────────────────────────────┤
│ Feature Layer: Matching, Measure, Edge, Blob, Filter...     │
├─────────────────────────────────────────────────────────────┤
│ Internal Layer: Math, Image processing primitives           │
├─────────────────────────────────────────────────────────────┤
│ Platform Layer: Memory, Thread, SIMD, Timer                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Precision Specifications

| Module | Metric | Requirement |
|--------|--------|-------------|
| Edge1D | Position | <0.02px (1σ) |
| Caliper | Position/Width | <0.03px / <0.05px |
| ShapeModel | Position/Angle | <0.05px / <0.05° |
| CircleFit | Center/Radius | <0.02px |
| LineFit | Angle | <0.005° |

*Standard conditions: contrast ≥ 50, noise σ ≤ 5*

---

## License

MIT License - See LICENSE file for details.

---

## Links

- [GitHub Repository](https://github.com/userqz1/QiVision)
- [API Reference](modules.html)
- [Class List](annotated.html)
