#include <QiVision/Internal/DistanceTransform.h>
#include <QiVision/Internal/RLEOps.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <cstring>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

constexpr float INF_DIST = std::numeric_limits<float>::max();
constexpr int32_t INF_INT = std::numeric_limits<int32_t>::max() / 2;

// Chamfer weights (scaled by denominator for integer arithmetic)
constexpr int32_t CHAMFER_3_4_D = 3;      // Diagonal weight
constexpr int32_t CHAMFER_3_4_O = 4;      // Orthogonal weight (scaled)
constexpr float CHAMFER_3_4_SCALE = 3.0f; // Scale factor to approximate L2

constexpr int32_t CHAMFER_5_7_D = 7;       // Diagonal
constexpr int32_t CHAMFER_5_7_O = 5;       // Orthogonal
constexpr int32_t CHAMFER_5_7_K = 11;      // Knight move
constexpr float CHAMFER_5_7_SCALE = 5.0f;  // Scale factor

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Convert distance image to specified output type
QImage ConvertDistanceOutput(const std::vector<float>& dist,
                              int32_t width, int32_t height,
                              DistanceOutputType outputType) {
    switch (outputType) {
        case DistanceOutputType::Float32: {
            QImage result(width, height, PixelType::Float32, ChannelType::Gray);
            float* data = static_cast<float*>(result.Data());
            int32_t stride = static_cast<int32_t>(result.Stride()) / sizeof(float);
            for (int32_t r = 0; r < height; ++r) {
                for (int32_t c = 0; c < width; ++c) {
                    data[r * stride + c] = dist[r * width + c];
                }
            }
            return result;
        }

        case DistanceOutputType::UInt8: {
            QImage result(width, height, PixelType::UInt8, ChannelType::Gray);
            uint8_t* data = static_cast<uint8_t*>(result.Data());
            int32_t stride = static_cast<int32_t>(result.Stride());
            for (int32_t r = 0; r < height; ++r) {
                for (int32_t c = 0; c < width; ++c) {
                    float v = dist[r * width + c];
                    data[r * stride + c] = static_cast<uint8_t>(std::min(255.0f, v));
                }
            }
            return result;
        }

        case DistanceOutputType::UInt16: {
            QImage result(width, height, PixelType::UInt16, ChannelType::Gray);
            uint16_t* data = static_cast<uint16_t*>(result.Data());
            int32_t stride = static_cast<int32_t>(result.Stride()) / sizeof(uint16_t);
            for (int32_t r = 0; r < height; ++r) {
                for (int32_t c = 0; c < width; ++c) {
                    float v = dist[r * width + c];
                    data[r * stride + c] = static_cast<uint16_t>(std::min(65535.0f, v));
                }
            }
            return result;
        }

        case DistanceOutputType::Int16: {
            QImage result(width, height, PixelType::Int16, ChannelType::Gray);
            int16_t* data = static_cast<int16_t*>(result.Data());
            int32_t stride = static_cast<int32_t>(result.Stride()) / sizeof(int16_t);
            for (int32_t r = 0; r < height; ++r) {
                for (int32_t c = 0; c < width; ++c) {
                    float v = dist[r * width + c];
                    data[r * stride + c] = static_cast<int16_t>(std::min(32767.0f, v));
                }
            }
            return result;
        }
    }

    return QImage();
}

// L1 distance transform (two-pass)
std::vector<float> ComputeL1Distance(const uint8_t* binary, int32_t width, int32_t height,
                                      int32_t stride) {
    std::vector<int32_t> dist(width * height, INF_INT);

    // Initialize: 0 for background, INF for foreground
    for (int32_t r = 0; r < height; ++r) {
        const uint8_t* row = binary + r * stride;
        for (int32_t c = 0; c < width; ++c) {
            if (row[c] == 0) {
                dist[r * width + c] = 0;
            }
        }
    }

    // Forward pass (top-left to bottom-right)
    for (int32_t r = 0; r < height; ++r) {
        for (int32_t c = 0; c < width; ++c) {
            int32_t idx = r * width + c;
            if (r > 0) {
                dist[idx] = std::min(dist[idx], dist[(r - 1) * width + c] + 1);
            }
            if (c > 0) {
                dist[idx] = std::min(dist[idx], dist[r * width + c - 1] + 1);
            }
        }
    }

    // Backward pass (bottom-right to top-left)
    for (int32_t r = height - 1; r >= 0; --r) {
        for (int32_t c = width - 1; c >= 0; --c) {
            int32_t idx = r * width + c;
            if (r < height - 1) {
                dist[idx] = std::min(dist[idx], dist[(r + 1) * width + c] + 1);
            }
            if (c < width - 1) {
                dist[idx] = std::min(dist[idx], dist[r * width + c + 1] + 1);
            }
        }
    }

    // Convert to float
    std::vector<float> result(width * height);
    for (size_t i = 0; i < dist.size(); ++i) {
        result[i] = (dist[i] >= INF_INT) ? INF_DIST : static_cast<float>(dist[i]);
    }

    return result;
}

// LInf (Chessboard) distance transform
std::vector<float> ComputeLInfDistance(const uint8_t* binary, int32_t width, int32_t height,
                                        int32_t stride) {
    std::vector<int32_t> dist(width * height, INF_INT);

    // Initialize
    for (int32_t r = 0; r < height; ++r) {
        const uint8_t* row = binary + r * stride;
        for (int32_t c = 0; c < width; ++c) {
            if (row[c] == 0) {
                dist[r * width + c] = 0;
            }
        }
    }

    // Forward pass
    for (int32_t r = 0; r < height; ++r) {
        for (int32_t c = 0; c < width; ++c) {
            int32_t idx = r * width + c;
            int32_t minD = dist[idx];

            if (r > 0) {
                minD = std::min(minD, dist[(r - 1) * width + c] + 1);
                if (c > 0) minD = std::min(minD, dist[(r - 1) * width + c - 1] + 1);
                if (c < width - 1) minD = std::min(minD, dist[(r - 1) * width + c + 1] + 1);
            }
            if (c > 0) {
                minD = std::min(minD, dist[r * width + c - 1] + 1);
            }

            dist[idx] = minD;
        }
    }

    // Backward pass
    for (int32_t r = height - 1; r >= 0; --r) {
        for (int32_t c = width - 1; c >= 0; --c) {
            int32_t idx = r * width + c;
            int32_t minD = dist[idx];

            if (r < height - 1) {
                minD = std::min(minD, dist[(r + 1) * width + c] + 1);
                if (c > 0) minD = std::min(minD, dist[(r + 1) * width + c - 1] + 1);
                if (c < width - 1) minD = std::min(minD, dist[(r + 1) * width + c + 1] + 1);
            }
            if (c < width - 1) {
                minD = std::min(minD, dist[r * width + c + 1] + 1);
            }

            dist[idx] = minD;
        }
    }

    std::vector<float> result(width * height);
    for (size_t i = 0; i < dist.size(); ++i) {
        result[i] = (dist[i] >= INF_INT) ? INF_DIST : static_cast<float>(dist[i]);
    }

    return result;
}

// Chamfer distance transform (3-4 or 5-7-11 approximation)
std::vector<float> ComputeChamferDistance(const uint8_t* binary, int32_t width, int32_t height,
                                           int32_t stride, bool use5_7) {
    std::vector<int32_t> dist(width * height, INF_INT);

    int32_t dO = use5_7 ? CHAMFER_5_7_O : CHAMFER_3_4_O;
    int32_t dD = use5_7 ? CHAMFER_5_7_D : CHAMFER_3_4_D;
    float scale = use5_7 ? CHAMFER_5_7_SCALE : CHAMFER_3_4_SCALE;

    // Initialize
    for (int32_t r = 0; r < height; ++r) {
        const uint8_t* row = binary + r * stride;
        for (int32_t c = 0; c < width; ++c) {
            if (row[c] == 0) {
                dist[r * width + c] = 0;
            }
        }
    }

    // Forward pass
    for (int32_t r = 0; r < height; ++r) {
        for (int32_t c = 0; c < width; ++c) {
            int32_t idx = r * width + c;
            int32_t minD = dist[idx];

            if (r > 0) {
                minD = std::min(minD, dist[(r - 1) * width + c] + dO);
                if (c > 0) minD = std::min(minD, dist[(r - 1) * width + c - 1] + dD);
                if (c < width - 1) minD = std::min(minD, dist[(r - 1) * width + c + 1] + dD);
            }
            if (c > 0) {
                minD = std::min(minD, dist[r * width + c - 1] + dO);
            }

            // Knight moves for 5-7-11
            if (use5_7) {
                if (r > 1 && c > 0) minD = std::min(minD, dist[(r - 2) * width + c - 1] + CHAMFER_5_7_K);
                if (r > 1 && c < width - 1) minD = std::min(minD, dist[(r - 2) * width + c + 1] + CHAMFER_5_7_K);
                if (r > 0 && c > 1) minD = std::min(minD, dist[(r - 1) * width + c - 2] + CHAMFER_5_7_K);
            }

            dist[idx] = minD;
        }
    }

    // Backward pass
    for (int32_t r = height - 1; r >= 0; --r) {
        for (int32_t c = width - 1; c >= 0; --c) {
            int32_t idx = r * width + c;
            int32_t minD = dist[idx];

            if (r < height - 1) {
                minD = std::min(minD, dist[(r + 1) * width + c] + dO);
                if (c > 0) minD = std::min(minD, dist[(r + 1) * width + c - 1] + dD);
                if (c < width - 1) minD = std::min(minD, dist[(r + 1) * width + c + 1] + dD);
            }
            if (c < width - 1) {
                minD = std::min(minD, dist[r * width + c + 1] + dO);
            }

            // Knight moves for 5-7-11
            if (use5_7) {
                if (r < height - 2 && c > 0) minD = std::min(minD, dist[(r + 2) * width + c - 1] + CHAMFER_5_7_K);
                if (r < height - 2 && c < width - 1) minD = std::min(minD, dist[(r + 2) * width + c + 1] + CHAMFER_5_7_K);
                if (r < height - 1 && c < width - 2) minD = std::min(minD, dist[(r + 1) * width + c + 2] + CHAMFER_5_7_K);
            }

            dist[idx] = minD;
        }
    }

    // Convert to float and scale
    std::vector<float> result(width * height);
    for (size_t i = 0; i < dist.size(); ++i) {
        result[i] = (dist[i] >= INF_INT) ? INF_DIST : static_cast<float>(dist[i]) / scale;
    }

    return result;
}

// Exact Euclidean distance transform (Meijster et al. algorithm)
// Reference: "A General Algorithm for Computing Distance Transforms in Linear Time"
std::vector<float> ComputeL2Distance(const uint8_t* binary, int32_t width, int32_t height,
                                      int32_t stride) {
    // Phase 1: Compute 1D distance in each column
    std::vector<int32_t> g(width * height);

    for (int32_t c = 0; c < width; ++c) {
        // Forward scan - row 0
        // Background (0) has distance 0, Foreground (non-zero) starts at INF
        if (binary[0 * stride + c] == 0) {
            g[0 * width + c] = 0;
        } else {
            g[0 * width + c] = INF_INT;
        }

        for (int32_t r = 1; r < height; ++r) {
            if (binary[r * stride + c] == 0) {
                g[r * width + c] = 0;
            } else {
                g[r * width + c] = (g[(r - 1) * width + c] < INF_INT) ?
                                    g[(r - 1) * width + c] + 1 : INF_INT;
            }
        }

        // Backward scan
        for (int32_t r = height - 2; r >= 0; --r) {
            if (g[(r + 1) * width + c] < g[r * width + c]) {
                g[r * width + c] = std::min(g[r * width + c], g[(r + 1) * width + c] + 1);
            }
        }
    }

    // Phase 2: Compute 2D Euclidean distance using parabola envelope
    std::vector<float> dist(width * height);
    std::vector<int32_t> s(width);  // Parabola positions
    std::vector<int32_t> t(width);  // Parabola right boundaries

    auto f = [&](int32_t x, int32_t i, int32_t row) -> int64_t {
        int64_t gVal = g[row * width + i];
        if (gVal >= INF_INT) return static_cast<int64_t>(INF_INT) * INF_INT;
        int64_t dx = x - i;
        return dx * dx + gVal * gVal;
    };

    auto sep = [&](int32_t i, int32_t u, int32_t row) -> int32_t {
        int64_t gi = g[row * width + i];
        int64_t gu = g[row * width + u];
        if (gi >= INF_INT) return INF_INT;
        if (gu >= INF_INT) return -INF_INT;
        // Separation point between parabolas at i and u
        // (u^2 - i^2 + gu^2 - gi^2) / (2*(u-i))
        return static_cast<int32_t>(((int64_t)u * u - (int64_t)i * i + gu * gu - gi * gi) /
                                    (2 * (u - i)));
    };

    for (int32_t r = 0; r < height; ++r) {
        int32_t q = 0;
        s[0] = 0;
        t[0] = 0;

        // Build lower envelope
        for (int32_t u = 1; u < width; ++u) {
            while (q >= 0 && f(t[q], s[q], r) > f(t[q], u, r)) {
                q--;
            }
            if (q < 0) {
                q = 0;
                s[0] = u;
            } else {
                int32_t sepVal = sep(s[q], u, r);
                // Guard against overflow when sep returns extreme values
                // sepVal = INF_INT means gi is INF, so u dominates everywhere
                // sepVal = -INF_INT means gu is INF, so i dominates everywhere
                if (sepVal >= INF_INT / 2) {
                    // u's parabola dominates everywhere (gi is INF)
                    q = 0;
                    s[0] = u;
                    t[0] = 0;
                } else if (sepVal <= -INF_INT / 2) {
                    // i's parabola dominates everywhere (gu is INF), don't add u
                } else {
                    int32_t w = sepVal + 1;
                    if (w < width && w >= 0) {
                        q++;
                        s[q] = u;
                        t[q] = w;
                    }
                }
            }
        }

        // Scan and compute distances
        for (int32_t c = width - 1; c >= 0; --c) {
            int64_t d2 = f(c, s[q], r);
            dist[r * width + c] = (d2 >= (int64_t)INF_INT * INF_INT) ?
                                   INF_DIST : std::sqrt(static_cast<float>(d2));
            if (c == t[q]) q--;
        }
    }

    return dist;
}

} // anonymous namespace

// =============================================================================
// Distance Transform - Image Based
// =============================================================================

QImage DistanceTransform(const QImage& binary,
                         DistanceType distType,
                         DistanceOutputType outputType) {
    if (binary.Empty()) return QImage();

    int32_t width = binary.Width();
    int32_t height = binary.Height();
    int32_t stride = static_cast<int32_t>(binary.Stride());
    const uint8_t* data = static_cast<const uint8_t*>(binary.Data());

    std::vector<float> dist;

    switch (distType) {
        case DistanceType::L1:
            dist = ComputeL1Distance(data, width, height, stride);
            break;
        case DistanceType::L2:
            dist = ComputeL2Distance(data, width, height, stride);
            break;
        case DistanceType::LInf:
            dist = ComputeLInfDistance(data, width, height, stride);
            break;
        case DistanceType::Chamfer3_4:
            dist = ComputeChamferDistance(data, width, height, stride, false);
            break;
        case DistanceType::Chamfer5_7:
            dist = ComputeChamferDistance(data, width, height, stride, true);
            break;
    }

    return ConvertDistanceOutput(dist, width, height, outputType);
}

QImage DistanceTransformNormalized(const QImage& binary, DistanceType distType) {
    QImage dist = DistanceTransform(binary, distType, DistanceOutputType::Float32);
    if (dist.Empty()) return dist;

    // Find maximum
    float maxDist = 0.0f;
    const float* data = static_cast<const float*>(dist.Data());
    int32_t stride = static_cast<int32_t>(dist.Stride()) / sizeof(float);
    int32_t width = dist.Width();
    int32_t height = dist.Height();

    for (int32_t r = 0; r < height; ++r) {
        for (int32_t c = 0; c < width; ++c) {
            float v = data[r * stride + c];
            if (v < INF_DIST && v > maxDist) {
                maxDist = v;
            }
        }
    }

    // Normalize
    if (maxDist > 0) {
        float* mutableData = static_cast<float*>(dist.Data());
        for (int32_t r = 0; r < height; ++r) {
            for (int32_t c = 0; c < width; ++c) {
                float& v = mutableData[r * stride + c];
                if (v < INF_DIST) {
                    v /= maxDist;
                } else {
                    v = 1.0f;
                }
            }
        }
    }

    return dist;
}

QImage DistanceTransformL1(const QImage& binary) {
    return DistanceTransform(binary, DistanceType::L1, DistanceOutputType::Float32);
}

QImage DistanceTransformL2(const QImage& binary) {
    return DistanceTransform(binary, DistanceType::L2, DistanceOutputType::Float32);
}

QImage DistanceTransformLInf(const QImage& binary) {
    return DistanceTransform(binary, DistanceType::LInf, DistanceOutputType::Float32);
}

QImage DistanceTransformChamfer(const QImage& binary, bool use5_7) {
    DistanceType type = use5_7 ? DistanceType::Chamfer5_7 : DistanceType::Chamfer3_4;
    return DistanceTransform(binary, type, DistanceOutputType::Float32);
}

// =============================================================================
// Distance Transform - Region Based
// =============================================================================

QImage DistanceTransformRegion(const QRegion& region,
                               const Rect2i& bounds,
                               DistanceType distType) {
    if (region.Empty()) return QImage();

    // Convert region to binary image
    QImage binary(bounds.width, bounds.height, PixelType::UInt8, ChannelType::Gray);
    std::memset(binary.Data(), 0, binary.Stride() * binary.Height());

    uint8_t* data = static_cast<uint8_t*>(binary.Data());
    int32_t stride = static_cast<int32_t>(binary.Stride());

    for (const auto& run : region.Runs()) {
        int32_t r = run.row - bounds.y;
        if (r < 0 || r >= bounds.height) continue;

        int32_t cStart = std::max(0, run.colBegin - bounds.x);
        int32_t cEnd = std::min(bounds.width, run.colEnd - bounds.x);

        for (int32_t c = cStart; c < cEnd; ++c) {
            data[r * stride + c] = 255;
        }
    }

    return DistanceTransform(binary, distType, DistanceOutputType::Float32);
}

QImage SignedDistanceTransform(const QRegion& region,
                               const Rect2i& bounds,
                               DistanceType distType) {
    if (region.Empty()) {
        // All negative (outside)
        QImage result(bounds.width, bounds.height, PixelType::Float32, ChannelType::Gray);
        float* data = static_cast<float*>(result.Data());
        int32_t stride = static_cast<int32_t>(result.Stride()) / sizeof(float);
        for (int32_t r = 0; r < bounds.height; ++r) {
            for (int32_t c = 0; c < bounds.width; ++c) {
                data[r * stride + c] = -INF_DIST;
            }
        }
        return result;
    }

    // Convert region to binary image
    QImage binary(bounds.width, bounds.height, PixelType::UInt8, ChannelType::Gray);
    std::memset(binary.Data(), 0, binary.Stride() * binary.Height());

    uint8_t* binData = static_cast<uint8_t*>(binary.Data());
    int32_t binStride = static_cast<int32_t>(binary.Stride());

    for (const auto& run : region.Runs()) {
        int32_t r = run.row - bounds.y;
        if (r < 0 || r >= bounds.height) continue;

        int32_t cStart = std::max(0, run.colBegin - bounds.x);
        int32_t cEnd = std::min(bounds.width, run.colEnd - bounds.x);

        for (int32_t c = cStart; c < cEnd; ++c) {
            binData[r * binStride + c] = 255;
        }
    }

    // Distance inside (foreground to background)
    QImage distInside = DistanceTransform(binary, distType, DistanceOutputType::Float32);

    // Invert binary for distance outside
    for (int32_t r = 0; r < bounds.height; ++r) {
        uint8_t* row = binData + r * binStride;
        for (int32_t c = 0; c < bounds.width; ++c) {
            row[c] = (row[c] == 0) ? 255 : 0;
        }
    }

    // Distance outside (background to foreground in inverted)
    QImage distOutside = DistanceTransform(binary, distType, DistanceOutputType::Float32);

    // Combine: positive inside, negative outside
    QImage result(bounds.width, bounds.height, PixelType::Float32, ChannelType::Gray);
    float* resultData = static_cast<float*>(result.Data());
    const float* insideData = static_cast<const float*>(distInside.Data());
    const float* outsideData = static_cast<const float*>(distOutside.Data());
    int32_t stride = static_cast<int32_t>(result.Stride()) / sizeof(float);

    for (int32_t r = 0; r < bounds.height; ++r) {
        for (int32_t c = 0; c < bounds.width; ++c) {
            float inside = insideData[r * stride + c];
            float outside = outsideData[r * stride + c];

            if (inside > 0 && inside < INF_DIST) {
                resultData[r * stride + c] = inside;
            } else if (outside > 0 && outside < INF_DIST) {
                resultData[r * stride + c] = -outside;
            } else {
                resultData[r * stride + c] = 0;
            }
        }
    }

    return result;
}

// =============================================================================
// Distance to Specific Points/Features
// =============================================================================

QImage DistanceToPoints(int32_t width, int32_t height,
                        const std::vector<Point2i>& seedPoints,
                        DistanceType distType) {
    if (width <= 0 || height <= 0 || seedPoints.empty()) {
        return QImage();
    }

    // Create binary image with seed points as background
    QImage binary(width, height, PixelType::UInt8, ChannelType::Gray);
    std::memset(binary.Data(), 255, binary.Stride() * binary.Height());  // All foreground

    uint8_t* data = static_cast<uint8_t*>(binary.Data());
    int32_t stride = static_cast<int32_t>(binary.Stride());

    for (const auto& pt : seedPoints) {
        if (pt.x >= 0 && pt.x < width && pt.y >= 0 && pt.y < height) {
            data[pt.y * stride + pt.x] = 0;  // Set seed as background
        }
    }

    return DistanceTransform(binary, distType, DistanceOutputType::Float32);
}

QImage DistanceToEdges(const QImage& edges, DistanceType distType) {
    if (edges.Empty()) return QImage();

    // Invert: edges become background, rest becomes foreground
    int32_t width = edges.Width();
    int32_t height = edges.Height();
    int32_t stride = static_cast<int32_t>(edges.Stride());
    const uint8_t* srcData = static_cast<const uint8_t*>(edges.Data());

    QImage inverted(width, height, PixelType::UInt8, ChannelType::Gray);
    uint8_t* dstData = static_cast<uint8_t*>(inverted.Data());
    int32_t dstStride = static_cast<int32_t>(inverted.Stride());

    for (int32_t r = 0; r < height; ++r) {
        const uint8_t* srcRow = srcData + r * stride;
        uint8_t* dstRow = dstData + r * dstStride;
        for (int32_t c = 0; c < width; ++c) {
            dstRow[c] = (srcRow[c] == 0) ? 255 : 0;
        }
    }

    return DistanceTransform(inverted, distType, DistanceOutputType::Float32);
}

// =============================================================================
// Voronoi Diagram
// =============================================================================

QImage VoronoiDiagram(int32_t width, int32_t height,
                      const std::vector<Point2i>& seedPoints,
                      DistanceType distType) {
    if (width <= 0 || height <= 0 || seedPoints.empty()) {
        return QImage();
    }

    int32_t numSeeds = static_cast<int32_t>(seedPoints.size());

    // For small number of seeds, use brute force
    if (numSeeds <= 256) {
        QImage result(width, height, PixelType::UInt8, ChannelType::Gray);
        uint8_t* data = static_cast<uint8_t*>(result.Data());
        int32_t stride = static_cast<int32_t>(result.Stride());

        for (int32_t r = 0; r < height; ++r) {
            for (int32_t c = 0; c < width; ++c) {
                double minDist = std::numeric_limits<double>::max();
                int32_t minIdx = 0;

                for (int32_t i = 0; i < numSeeds; ++i) {
                    double dx = c - seedPoints[i].x;
                    double dy = r - seedPoints[i].y;
                    double dist;

                    switch (distType) {
                        case DistanceType::L1:
                            dist = std::abs(dx) + std::abs(dy);
                            break;
                        case DistanceType::LInf:
                            dist = std::max(std::abs(dx), std::abs(dy));
                            break;
                        default:  // L2 and Chamfer
                            dist = dx * dx + dy * dy;  // Compare squared
                            break;
                    }

                    if (dist < minDist) {
                        minDist = dist;
                        minIdx = i;
                    }
                }

                data[r * stride + c] = static_cast<uint8_t>(minIdx % 256);
            }
        }

        return result;
    }

    // For larger number of seeds, use distance transform approach
    // Create separate distance transforms and find minimum
    std::vector<QImage> distances;
    distances.reserve(numSeeds);

    for (const auto& pt : seedPoints) {
        std::vector<Point2i> single = {pt};
        distances.push_back(DistanceToPoints(width, height, single, distType));
    }

    QImage result(width, height, PixelType::UInt8, ChannelType::Gray);
    uint8_t* data = static_cast<uint8_t*>(result.Data());
    int32_t stride = static_cast<int32_t>(result.Stride());

    for (int32_t r = 0; r < height; ++r) {
        for (int32_t c = 0; c < width; ++c) {
            float minDist = INF_DIST;
            int32_t minIdx = 0;

            for (int32_t i = 0; i < numSeeds; ++i) {
                const float* distData = static_cast<const float*>(distances[i].Data());
                int32_t distStride = static_cast<int32_t>(distances[i].Stride()) / sizeof(float);
                float d = distData[r * distStride + c];

                if (d < minDist) {
                    minDist = d;
                    minIdx = i;
                }
            }

            data[r * stride + c] = static_cast<uint8_t>(minIdx % 256);
        }
    }

    return result;
}

QImage VoronoiFromLabels(const QImage& labels, DistanceType distType) {
    if (labels.Empty()) return QImage();

    int32_t width = labels.Width();
    int32_t height = labels.Height();
    const uint8_t* labelData = static_cast<const uint8_t*>(labels.Data());
    int32_t labelStride = static_cast<int32_t>(labels.Stride());

    // Find unique labels and their seed points
    std::vector<std::vector<Point2i>> labelSeeds(256);

    for (int32_t r = 0; r < height; ++r) {
        const uint8_t* row = labelData + r * labelStride;
        for (int32_t c = 0; c < width; ++c) {
            uint8_t lbl = row[c];
            if (lbl > 0) {
                labelSeeds[lbl].push_back({c, r});
            }
        }
    }

    // Compute distance transform for each label
    std::vector<QImage> distances(256);
    for (int32_t i = 1; i < 256; ++i) {
        if (!labelSeeds[i].empty()) {
            distances[i] = DistanceToPoints(width, height, labelSeeds[i], distType);
        }
    }

    // Assign each pixel to nearest label
    QImage result(width, height, PixelType::UInt8, ChannelType::Gray);
    uint8_t* data = static_cast<uint8_t*>(result.Data());
    int32_t stride = static_cast<int32_t>(result.Stride());

    for (int32_t r = 0; r < height; ++r) {
        for (int32_t c = 0; c < width; ++c) {
            float minDist = INF_DIST;
            uint8_t minLabel = 0;

            for (int32_t i = 1; i < 256; ++i) {
                if (distances[i].Empty()) continue;

                const float* distData = static_cast<const float*>(distances[i].Data());
                int32_t distStride = static_cast<int32_t>(distances[i].Stride()) / sizeof(float);
                float d = distData[r * distStride + c];

                if (d < minDist) {
                    minDist = d;
                    minLabel = static_cast<uint8_t>(i);
                }
            }

            data[r * stride + c] = minLabel;
        }
    }

    return result;
}

// =============================================================================
// Skeleton from Distance Transform
// =============================================================================

QImage SkeletonFromDistance(const QImage& binary, DistanceType distType) {
    if (binary.Empty()) return QImage();

    // Compute distance transform
    QImage dist = DistanceTransform(binary, distType, DistanceOutputType::Float32);

    int32_t width = dist.Width();
    int32_t height = dist.Height();
    const float* distData = static_cast<const float*>(dist.Data());
    int32_t distStride = static_cast<int32_t>(dist.Stride()) / sizeof(float);

    // Find ridge points (local maxima in distance)
    QImage skeleton(width, height, PixelType::UInt8, ChannelType::Gray);
    std::memset(skeleton.Data(), 0, skeleton.Stride() * skeleton.Height());
    uint8_t* skelData = static_cast<uint8_t*>(skeleton.Data());
    int32_t skelStride = static_cast<int32_t>(skeleton.Stride());

    for (int32_t r = 1; r < height - 1; ++r) {
        for (int32_t c = 1; c < width - 1; ++c) {
            float center = distData[r * distStride + c];
            if (center <= 0) continue;

            // Check if local maximum in at least one direction
            bool isRidge = false;

            // Horizontal direction
            float left = distData[r * distStride + c - 1];
            float right = distData[r * distStride + c + 1];
            if (center >= left && center >= right && (center > left || center > right)) {
                isRidge = true;
            }

            // Vertical direction
            float top = distData[(r - 1) * distStride + c];
            float bottom = distData[(r + 1) * distStride + c];
            if (center >= top && center >= bottom && (center > top || center > bottom)) {
                isRidge = true;
            }

            // Diagonal directions
            float tl = distData[(r - 1) * distStride + c - 1];
            float br = distData[(r + 1) * distStride + c + 1];
            if (center >= tl && center >= br && (center > tl || center > br)) {
                isRidge = true;
            }

            float tr = distData[(r - 1) * distStride + c + 1];
            float bl = distData[(r + 1) * distStride + c - 1];
            if (center >= tr && center >= bl && (center > tr || center > bl)) {
                isRidge = true;
            }

            if (isRidge) {
                skelData[r * skelStride + c] = 255;
            }
        }
    }

    return skeleton;
}

QImage MedialAxisTransform(const QImage& binary, QImage& skeleton) {
    if (binary.Empty()) return QImage();

    // Compute distance transform
    QImage dist = DistanceTransformL2(binary);
    skeleton = SkeletonFromDistance(binary, DistanceType::L2);

    // Mask distance transform with skeleton
    int32_t width = dist.Width();
    int32_t height = dist.Height();

    QImage mat(width, height, PixelType::Float32, ChannelType::Gray);
    std::memset(mat.Data(), 0, mat.Stride() * mat.Height());

    float* matData = static_cast<float*>(mat.Data());
    const float* distData = static_cast<const float*>(dist.Data());
    const uint8_t* skelData = static_cast<const uint8_t*>(skeleton.Data());
    int32_t matStride = static_cast<int32_t>(mat.Stride()) / sizeof(float);
    int32_t distStride = static_cast<int32_t>(dist.Stride()) / sizeof(float);
    int32_t skelStride = static_cast<int32_t>(skeleton.Stride());

    for (int32_t r = 0; r < height; ++r) {
        for (int32_t c = 0; c < width; ++c) {
            if (skelData[r * skelStride + c] != 0) {
                matData[r * matStride + c] = distData[r * distStride + c];
            }
        }
    }

    return mat;
}

// =============================================================================
// Utility Functions
// =============================================================================

double GetMaxDistance(const QImage& distanceImage) {
    if (distanceImage.Empty()) return 0.0;

    double maxDist = 0.0;
    int32_t width = distanceImage.Width();
    int32_t height = distanceImage.Height();

    if (distanceImage.Type() == PixelType::Float32) {
        const float* data = static_cast<const float*>(distanceImage.Data());
        int32_t stride = static_cast<int32_t>(distanceImage.Stride()) / sizeof(float);

        for (int32_t r = 0; r < height; ++r) {
            for (int32_t c = 0; c < width; ++c) {
                float v = data[r * stride + c];
                if (v < INF_DIST && v > maxDist) {
                    maxDist = v;
                }
            }
        }
    }

    return maxDist;
}

QImage ThresholdDistance(const QImage& distanceImage, double threshold, bool invert) {
    if (distanceImage.Empty()) return QImage();

    int32_t width = distanceImage.Width();
    int32_t height = distanceImage.Height();

    QImage result(width, height, PixelType::UInt8, ChannelType::Gray);
    uint8_t* dstData = static_cast<uint8_t*>(result.Data());
    int32_t dstStride = static_cast<int32_t>(result.Stride());

    if (distanceImage.Type() == PixelType::Float32) {
        const float* srcData = static_cast<const float*>(distanceImage.Data());
        int32_t srcStride = static_cast<int32_t>(distanceImage.Stride()) / sizeof(float);
        float thresh = static_cast<float>(threshold);

        for (int32_t r = 0; r < height; ++r) {
            for (int32_t c = 0; c < width; ++c) {
                bool above = srcData[r * srcStride + c] >= thresh;
                dstData[r * dstStride + c] = (above != invert) ? 255 : 0;
            }
        }
    }

    return result;
}

std::vector<Point2i> FindPixelsAtDistance(const QImage& distanceImage,
                                          double distance,
                                          double tolerance) {
    std::vector<Point2i> points;
    if (distanceImage.Empty()) return points;

    int32_t width = distanceImage.Width();
    int32_t height = distanceImage.Height();
    double minDist = distance - tolerance;
    double maxDist = distance + tolerance;

    if (distanceImage.Type() == PixelType::Float32) {
        const float* data = static_cast<const float*>(distanceImage.Data());
        int32_t stride = static_cast<int32_t>(distanceImage.Stride()) / sizeof(float);

        for (int32_t r = 0; r < height; ++r) {
            for (int32_t c = 0; c < width; ++c) {
                float d = data[r * stride + c];
                if (d >= minDist && d <= maxDist) {
                    points.push_back({c, r});
                }
            }
        }
    }

    return points;
}

std::vector<Point2i> FindDistanceMaxima(const QImage& distanceImage, double minDistance) {
    std::vector<Point2i> maxima;
    if (distanceImage.Empty()) return maxima;

    int32_t width = distanceImage.Width();
    int32_t height = distanceImage.Height();

    if (distanceImage.Type() != PixelType::Float32) return maxima;

    const float* data = static_cast<const float*>(distanceImage.Data());
    int32_t stride = static_cast<int32_t>(distanceImage.Stride()) / sizeof(float);
    float minDist = static_cast<float>(minDistance);

    for (int32_t r = 1; r < height - 1; ++r) {
        for (int32_t c = 1; c < width - 1; ++c) {
            float center = data[r * stride + c];
            if (center < minDist) continue;

            // Check if strict local maximum
            bool isMax = true;
            for (int32_t dr = -1; dr <= 1 && isMax; ++dr) {
                for (int32_t dc = -1; dc <= 1 && isMax; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    if (data[(r + dr) * stride + c + dc] >= center) {
                        isMax = false;
                    }
                }
            }

            if (isMax) {
                maxima.push_back({c, r});
            }
        }
    }

    return maxima;
}

} // namespace Qi::Vision::Internal
