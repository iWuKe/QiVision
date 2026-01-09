#include <QiVision/Internal/MorphGray.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace Qi::Vision::Internal {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Get pixel value with border handling (replicate)
inline uint8_t GetPixelReplicate(const uint8_t* data, int32_t width, int32_t height,
                                  int32_t stride, int32_t row, int32_t col) {
    row = std::clamp(row, 0, height - 1);
    col = std::clamp(col, 0, width - 1);
    return data[row * stride + col];
}

// Apply morphological operation with generic SE
template<typename Op>
QImage ApplyMorphOp(const QImage& src, const StructElement& se, Op op, uint8_t initVal) {
    if (src.Empty() || se.Empty()) return src;
    if (src.Type() != PixelType::UInt8 || src.GetChannelType() != ChannelType::Gray) {
        return QImage();
    }

    int32_t width = src.Width();
    int32_t height = src.Height();
    int32_t srcStride = static_cast<int32_t>(src.Stride());

    QImage dst(width, height, PixelType::UInt8, ChannelType::Gray);
    int32_t dstStride = static_cast<int32_t>(dst.Stride());

    const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    auto coords = se.GetCoordinates();

    for (int32_t r = 0; r < height; ++r) {
        for (int32_t c = 0; c < width; ++c) {
            uint8_t result = initVal;

            for (const auto& pt : coords) {
                int32_t sr = r + pt.y;
                int32_t sc = c + pt.x;
                uint8_t val = GetPixelReplicate(srcData, width, height, srcStride, sr, sc);
                result = op(result, val);
            }

            dstData[r * dstStride + c] = result;
        }
    }

    return dst;
}

// Max operation for dilation
struct MaxOp {
    uint8_t operator()(uint8_t a, uint8_t b) const { return std::max(a, b); }
};

// Min operation for erosion
struct MinOp {
    uint8_t operator()(uint8_t a, uint8_t b) const { return std::min(a, b); }
};

// Separable horizontal max/min filter
template<typename Op>
void HorizontalFilter(const uint8_t* src, uint8_t* dst, int32_t width, int32_t height,
                      int32_t srcStride, int32_t dstStride, int32_t radius, Op op, uint8_t initVal) {
    for (int32_t r = 0; r < height; ++r) {
        const uint8_t* srcRow = src + r * srcStride;
        uint8_t* dstRow = dst + r * dstStride;

        for (int32_t c = 0; c < width; ++c) {
            uint8_t result = initVal;
            int32_t left = std::max(0, c - radius);
            int32_t right = std::min(width - 1, c + radius);

            for (int32_t k = left; k <= right; ++k) {
                result = op(result, srcRow[k]);
            }
            dstRow[c] = result;
        }
    }
}

// Separable vertical max/min filter
template<typename Op>
void VerticalFilter(const uint8_t* src, uint8_t* dst, int32_t width, int32_t height,
                    int32_t srcStride, int32_t dstStride, int32_t radius, Op op, uint8_t initVal) {
    for (int32_t c = 0; c < width; ++c) {
        for (int32_t r = 0; r < height; ++r) {
            uint8_t result = initVal;
            int32_t top = std::max(0, r - radius);
            int32_t bottom = std::min(height - 1, r + radius);

            for (int32_t k = top; k <= bottom; ++k) {
                result = op(result, src[k * srcStride + c]);
            }
            dst[r * dstStride + c] = result;
        }
    }
}

// Pixel-wise operation
template<typename Op>
QImage PixelWiseOp(const QImage& a, const QImage& b, Op op) {
    if (a.Empty() || b.Empty()) return QImage();
    if (a.Width() != b.Width() || a.Height() != b.Height()) return QImage();

    int32_t width = a.Width();
    int32_t height = a.Height();

    QImage dst(width, height, PixelType::UInt8, ChannelType::Gray);

    const uint8_t* aData = static_cast<const uint8_t*>(a.Data());
    const uint8_t* bData = static_cast<const uint8_t*>(b.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    int32_t aStride = static_cast<int32_t>(a.Stride());
    int32_t bStride = static_cast<int32_t>(b.Stride());
    int32_t dstStride = static_cast<int32_t>(dst.Stride());

    for (int32_t r = 0; r < height; ++r) {
        const uint8_t* aRow = aData + r * aStride;
        const uint8_t* bRow = bData + r * bStride;
        uint8_t* dstRow = dstData + r * dstStride;

        for (int32_t c = 0; c < width; ++c) {
            dstRow[c] = op(aRow[c], bRow[c]);
        }
    }

    return dst;
}

struct SubOp {
    uint8_t operator()(uint8_t a, uint8_t b) const {
        return static_cast<uint8_t>(std::max(0, static_cast<int>(a) - static_cast<int>(b)));
    }
};

struct PixelMinOp {
    uint8_t operator()(uint8_t a, uint8_t b) const { return std::min(a, b); }
};

struct PixelMaxOp {
    uint8_t operator()(uint8_t a, uint8_t b) const { return std::max(a, b); }
};

} // anonymous namespace

// =============================================================================
// Basic Gray Morphology
// =============================================================================

QImage GrayDilate(const QImage& src, const StructElement& se) {
    return ApplyMorphOp(src, se, MaxOp{}, 0);
}

QImage GrayErode(const QImage& src, const StructElement& se) {
    return ApplyMorphOp(src, se, MinOp{}, 255);
}

QImage GrayDilateRect(const QImage& src, int32_t width, int32_t height) {
    if (src.Empty() || width <= 0 || height <= 0) return src;
    if (src.Type() != PixelType::UInt8 || src.GetChannelType() != ChannelType::Gray) {
        return QImage();
    }

    int32_t imgWidth = src.Width();
    int32_t imgHeight = src.Height();
    int32_t srcStride = static_cast<int32_t>(src.Stride());

    // Use separable filtering
    QImage temp(imgWidth, imgHeight, PixelType::UInt8, ChannelType::Gray);
    QImage dst(imgWidth, imgHeight, PixelType::UInt8, ChannelType::Gray);

    int32_t tempStride = static_cast<int32_t>(temp.Stride());
    int32_t dstStride = static_cast<int32_t>(dst.Stride());

    const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
    uint8_t* tempData = static_cast<uint8_t*>(temp.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    int32_t radiusX = width / 2;
    int32_t radiusY = height / 2;

    // Horizontal pass
    HorizontalFilter(srcData, tempData, imgWidth, imgHeight, srcStride, tempStride, radiusX, MaxOp{}, 0);

    // Vertical pass
    VerticalFilter(tempData, dstData, imgWidth, imgHeight, tempStride, dstStride, radiusY, MaxOp{}, 0);

    return dst;
}

QImage GrayErodeRect(const QImage& src, int32_t width, int32_t height) {
    if (src.Empty() || width <= 0 || height <= 0) return src;
    if (src.Type() != PixelType::UInt8 || src.GetChannelType() != ChannelType::Gray) {
        return QImage();
    }

    int32_t imgWidth = src.Width();
    int32_t imgHeight = src.Height();
    int32_t srcStride = static_cast<int32_t>(src.Stride());

    QImage temp(imgWidth, imgHeight, PixelType::UInt8, ChannelType::Gray);
    QImage dst(imgWidth, imgHeight, PixelType::UInt8, ChannelType::Gray);

    int32_t tempStride = static_cast<int32_t>(temp.Stride());
    int32_t dstStride = static_cast<int32_t>(dst.Stride());

    const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
    uint8_t* tempData = static_cast<uint8_t*>(temp.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    int32_t radiusX = width / 2;
    int32_t radiusY = height / 2;

    HorizontalFilter(srcData, tempData, imgWidth, imgHeight, srcStride, tempStride, radiusX, MinOp{}, 255);
    VerticalFilter(tempData, dstData, imgWidth, imgHeight, tempStride, dstStride, radiusY, MinOp{}, 255);

    return dst;
}

QImage GrayDilateCircle(const QImage& src, int32_t radius) {
    return GrayDilate(src, StructElement::Circle(radius));
}

QImage GrayErodeCircle(const QImage& src, int32_t radius) {
    return GrayErode(src, StructElement::Circle(radius));
}

// =============================================================================
// Compound Operations
// =============================================================================

QImage GrayOpening(const QImage& src, const StructElement& se) {
    QImage eroded = GrayErode(src, se);
    return GrayDilate(eroded, se);
}

QImage GrayClosing(const QImage& src, const StructElement& se) {
    QImage dilated = GrayDilate(src, se);
    return GrayErode(dilated, se);
}

QImage GrayOpeningRect(const QImage& src, int32_t width, int32_t height) {
    QImage eroded = GrayErodeRect(src, width, height);
    return GrayDilateRect(eroded, width, height);
}

QImage GrayClosingRect(const QImage& src, int32_t width, int32_t height) {
    QImage dilated = GrayDilateRect(src, width, height);
    return GrayErodeRect(dilated, width, height);
}

QImage GrayOpeningCircle(const QImage& src, int32_t radius) {
    auto se = StructElement::Circle(radius);
    return GrayOpening(src, se);
}

QImage GrayClosingCircle(const QImage& src, int32_t radius) {
    auto se = StructElement::Circle(radius);
    return GrayClosing(src, se);
}

// =============================================================================
// Derived Operations
// =============================================================================

QImage GrayMorphGradient(const QImage& src, const StructElement& se) {
    QImage dilated = GrayDilate(src, se);
    QImage eroded = GrayErode(src, se);
    return PixelWiseOp(dilated, eroded, SubOp{});
}

QImage GrayInternalGradient(const QImage& src, const StructElement& se) {
    QImage eroded = GrayErode(src, se);
    return PixelWiseOp(src, eroded, SubOp{});
}

QImage GrayExternalGradient(const QImage& src, const StructElement& se) {
    QImage dilated = GrayDilate(src, se);
    return PixelWiseOp(dilated, src, SubOp{});
}

QImage GrayTopHat(const QImage& src, const StructElement& se) {
    QImage opened = GrayOpening(src, se);
    return PixelWiseOp(src, opened, SubOp{});
}

QImage GrayBlackHat(const QImage& src, const StructElement& se) {
    QImage closed = GrayClosing(src, se);
    return PixelWiseOp(closed, src, SubOp{});
}

// =============================================================================
// Range Operations
// =============================================================================

QImage GrayRangeRect(const QImage& src, int32_t width, int32_t height) {
    QImage dilated = GrayDilateRect(src, width, height);
    QImage eroded = GrayErodeRect(src, width, height);
    return PixelWiseOp(dilated, eroded, SubOp{});
}

QImage GrayRangeCircle(const QImage& src, int32_t radius) {
    QImage dilated = GrayDilateCircle(src, radius);
    QImage eroded = GrayErodeCircle(src, radius);
    return PixelWiseOp(dilated, eroded, SubOp{});
}

// =============================================================================
// Iterative Operations
// =============================================================================

QImage GrayDilateN(const QImage& src, const StructElement& se, int iterations) {
    QImage result = src;
    for (int i = 0; i < iterations; ++i) {
        result = GrayDilate(result, se);
    }
    return result;
}

QImage GrayErodeN(const QImage& src, const StructElement& se, int iterations) {
    QImage result = src;
    for (int i = 0; i < iterations; ++i) {
        result = GrayErode(result, se);
    }
    return result;
}

QImage GrayOpeningN(const QImage& src, const StructElement& se, int iterations) {
    QImage result = src;
    for (int i = 0; i < iterations; ++i) {
        result = GrayOpening(result, se);
    }
    return result;
}

QImage GrayClosingN(const QImage& src, const StructElement& se, int iterations) {
    QImage result = src;
    for (int i = 0; i < iterations; ++i) {
        result = GrayClosing(result, se);
    }
    return result;
}

// =============================================================================
// Geodesic Operations
// =============================================================================

QImage GrayGeodesicDilate(const QImage& marker,
                          const QImage& mask,
                          const StructElement& se) {
    QImage dilated = GrayDilate(marker, se);
    return PixelWiseOp(dilated, mask, PixelMinOp{});
}

QImage GrayGeodesicErode(const QImage& marker,
                         const QImage& mask,
                         const StructElement& se) {
    QImage eroded = GrayErode(marker, se);
    return PixelWiseOp(eroded, mask, PixelMaxOp{});
}

QImage GrayReconstructByDilation(const QImage& marker, const QImage& mask) {
    if (marker.Empty() || mask.Empty()) return QImage();

    auto se = StructElement::Square(3);

    // Start with marker clamped to mask
    QImage result = PixelWiseOp(marker, mask, PixelMinOp{});

    const int maxIter = 10000;
    for (int i = 0; i < maxIter; ++i) {
        QImage prev = result;
        result = GrayGeodesicDilate(result, mask, se);

        // Check convergence
        bool converged = true;
        const uint8_t* prevData = static_cast<const uint8_t*>(prev.Data());
        const uint8_t* resultData = static_cast<const uint8_t*>(result.Data());
        int32_t prevStride = static_cast<int32_t>(prev.Stride());
        int32_t resultStride = static_cast<int32_t>(result.Stride());

        for (int32_t r = 0; r < result.Height() && converged; ++r) {
            if (std::memcmp(prevData + r * prevStride,
                           resultData + r * resultStride,
                           result.Width()) != 0) {
                converged = false;
            }
        }

        if (converged) break;
    }

    return result;
}

QImage GrayReconstructByErosion(const QImage& marker, const QImage& mask) {
    if (marker.Empty() || mask.Empty()) return QImage();

    auto se = StructElement::Square(3);

    // Start with marker clamped to mask
    QImage result = PixelWiseOp(marker, mask, PixelMaxOp{});

    const int maxIter = 10000;
    for (int i = 0; i < maxIter; ++i) {
        QImage prev = result;
        result = GrayGeodesicErode(result, mask, se);

        // Check convergence
        bool converged = true;
        const uint8_t* prevData = static_cast<const uint8_t*>(prev.Data());
        const uint8_t* resultData = static_cast<const uint8_t*>(result.Data());
        int32_t prevStride = static_cast<int32_t>(prev.Stride());
        int32_t resultStride = static_cast<int32_t>(result.Stride());

        for (int32_t r = 0; r < result.Height() && converged; ++r) {
            if (std::memcmp(prevData + r * prevStride,
                           resultData + r * resultStride,
                           result.Width()) != 0) {
                converged = false;
            }
        }

        if (converged) break;
    }

    return result;
}

QImage GrayOpeningByReconstruction(const QImage& src, const StructElement& se) {
    QImage eroded = GrayErode(src, se);
    return GrayReconstructByDilation(eroded, src);
}

QImage GrayClosingByReconstruction(const QImage& src, const StructElement& se) {
    QImage dilated = GrayDilate(src, se);
    return GrayReconstructByErosion(dilated, src);
}

QImage GrayFillHoles(const QImage& src) {
    if (src.Empty()) return src;

    int32_t width = src.Width();
    int32_t height = src.Height();

    // Create marker: border pixels from src, interior set to max
    QImage marker(width, height, PixelType::UInt8, ChannelType::Gray);
    uint8_t* markerData = static_cast<uint8_t*>(marker.Data());
    const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
    int32_t markerStride = static_cast<int32_t>(marker.Stride());
    int32_t srcStride = static_cast<int32_t>(src.Stride());

    // Set interior to 255 (max)
    for (int32_t r = 0; r < height; ++r) {
        std::memset(markerData + r * markerStride, 255, width);
    }

    // Set border to source values
    // Top and bottom rows
    std::memcpy(markerData, srcData, width);
    std::memcpy(markerData + (height - 1) * markerStride,
                srcData + (height - 1) * srcStride, width);

    // Left and right columns
    for (int32_t r = 0; r < height; ++r) {
        markerData[r * markerStride] = srcData[r * srcStride];
        markerData[r * markerStride + width - 1] = srcData[r * srcStride + width - 1];
    }

    // Reconstruct by erosion
    return GrayReconstructByErosion(marker, src);
}

// =============================================================================
// Background Correction
// =============================================================================

QImage RollingBallBackground(const QImage& src, int32_t radius) {
    auto se = StructElement::Circle(radius);
    QImage background = GrayOpening(src, se);
    return PixelWiseOp(src, background, SubOp{});
}

QImage EstimateBackground(const QImage& src, const StructElement& se) {
    return GrayOpening(src, se);
}

QImage SubtractBackground(const QImage& src,
                          const QImage& background,
                          int32_t offset) {
    if (src.Empty() || background.Empty()) return QImage();
    if (src.Width() != background.Width() || src.Height() != background.Height()) {
        return QImage();
    }

    int32_t width = src.Width();
    int32_t height = src.Height();

    QImage dst(width, height, PixelType::UInt8, ChannelType::Gray);

    const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
    const uint8_t* bgData = static_cast<const uint8_t*>(background.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    int32_t srcStride = static_cast<int32_t>(src.Stride());
    int32_t bgStride = static_cast<int32_t>(background.Stride());
    int32_t dstStride = static_cast<int32_t>(dst.Stride());

    for (int32_t r = 0; r < height; ++r) {
        const uint8_t* srcRow = srcData + r * srcStride;
        const uint8_t* bgRow = bgData + r * bgStride;
        uint8_t* dstRow = dstData + r * dstStride;

        for (int32_t c = 0; c < width; ++c) {
            int32_t val = static_cast<int32_t>(srcRow[c]) - static_cast<int32_t>(bgRow[c]) + offset;
            dstRow[c] = static_cast<uint8_t>(std::clamp(val, 0, 255));
        }
    }

    return dst;
}

} // namespace Qi::Vision::Internal
