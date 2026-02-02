/**
 * @file Canny.cpp
 * @brief Canny edge detection implementation
 */

#include <QiVision/Internal/Canny.h>
#include <QiVision/Internal/Gaussian.h>
#include <QiVision/Internal/NonMaxSuppression.h>
#include <QiVision/Internal/EdgeLinking.h>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <queue>

using Qi::Vision::PixelType;
using Qi::Vision::ChannelType;

namespace Qi::Vision::Internal {

// ============================================================================
// Gaussian Smoothing
// ============================================================================

void CannySmooth(const uint8_t* src, float* dst,
                 int32_t width, int32_t height,
                 double sigma) {
    if (sigma <= 0.0) {
        // No smoothing, just convert to float
        const int32_t size = width * height;
        for (int32_t i = 0; i < size; ++i) {
            dst[i] = static_cast<float>(src[i]);
        }
        return;
    }

    // Generate Gaussian kernel
    auto kernel = Gaussian::Kernel1D(sigma);
    const int32_t kernelSize = static_cast<int32_t>(kernel.size());
    const int32_t radius = kernelSize / 2;

    // Allocate temporary buffer for separable convolution
    std::vector<float> temp(width * height);

    // Horizontal pass
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int32_t k = -radius; k <= radius; ++k) {
                int32_t sx = x + k;
                // Reflect101 border handling
                if (sx < 0) sx = -sx;
                if (sx >= width) sx = 2 * width - 2 - sx;
                sx = std::clamp(sx, 0, width - 1);
                sum += src[y * width + sx] * kernel[k + radius];
            }
            temp[y * width + x] = static_cast<float>(sum);
        }
    }

    // Vertical pass
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int32_t k = -radius; k <= radius; ++k) {
                int32_t sy = y + k;
                // Reflect101 border handling
                if (sy < 0) sy = -sy;
                if (sy >= height) sy = 2 * height - 2 - sy;
                sy = std::clamp(sy, 0, height - 1);
                sum += temp[sy * width + x] * kernel[k + radius];
            }
            dst[y * width + x] = static_cast<float>(sum);
        }
    }
}

// ============================================================================
// Gradient Computation
// ============================================================================

void CannyGradient(const float* src, float* magnitude, float* direction,
                   int32_t width, int32_t height,
                   CannyGradientOp op) {
    // Allocate gradient buffers
    std::vector<float> gx(width * height);
    std::vector<float> gy(width * height);

    // Convert operator type
    GradientOperator gradOp = ToGradientOperator(op);

    // Compute gradients using template function
    Gradient<float, float>(src, gx.data(), gy.data(), width, height, gradOp);

    // Compute magnitude and direction
    for (int32_t i = 0; i < width * height; ++i) {
        float dx = gx[i];
        float dy = gy[i];
        magnitude[i] = std::sqrt(dx * dx + dy * dy);
        direction[i] = std::atan2(dy, dx);
    }
}

// ============================================================================
// Non-Maximum Suppression
// ============================================================================

void CannyNMS(const float* magnitude, const float* direction,
              float* output,
              int32_t width, int32_t height) {
    // Use existing NMS implementation
    NMS2DGradient(magnitude, direction, output, width, height);
}

// ============================================================================
// Hysteresis Thresholding
// ============================================================================

void CannyHysteresis(const float* nmsOutput, uint8_t* output,
                     int32_t width, int32_t height,
                     double lowThreshold, double highThreshold) {
    // Use existing hysteresis implementation
    HysteresisThreshold(nmsOutput, output, width, height,
                        static_cast<float>(lowThreshold),
                        static_cast<float>(highThreshold));
}

// ============================================================================
// Automatic Threshold Computation
// ============================================================================

void ComputeAutoThresholds(const float* magnitude,
                           int32_t width, int32_t height,
                           double& lowThreshold, double& highThreshold) {
    const int32_t size = width * height;

    // Compute histogram of gradient magnitudes
    // Use 256 bins covering the range [0, maxMag]
    float maxMag = 0.0f;
    for (int32_t i = 0; i < size; ++i) {
        maxMag = std::max(maxMag, magnitude[i]);
    }

    if (maxMag < 1e-6f) {
        lowThreshold = 0.0;
        highThreshold = 0.0;
        return;
    }

    // Build histogram
    constexpr int32_t numBins = 256;
    std::vector<int32_t> histogram(numBins, 0);
    const float binScale = (numBins - 1) / maxMag;

    for (int32_t i = 0; i < size; ++i) {
        int32_t bin = static_cast<int32_t>(magnitude[i] * binScale);
        bin = std::clamp(bin, 0, numBins - 1);
        histogram[bin]++;
    }

    // Use median-based method
    // Find the 70th and 90th percentile of non-zero gradients
    int32_t nonZeroCount = 0;
    for (int32_t i = 1; i < numBins; ++i) {
        nonZeroCount += histogram[i];
    }

    if (nonZeroCount == 0) {
        lowThreshold = 0.0;
        highThreshold = 0.0;
        return;
    }

    // Find percentile values
    int32_t lowTarget = static_cast<int32_t>(nonZeroCount * 0.7);
    int32_t highTarget = static_cast<int32_t>(nonZeroCount * 0.9);

    int32_t cumSum = 0;
    int32_t lowBin = 1, highBin = 1;
    bool foundLow = false, foundHigh = false;

    for (int32_t i = 1; i < numBins; ++i) {
        cumSum += histogram[i];
        if (!foundLow && cumSum >= lowTarget) {
            lowBin = i;
            foundLow = true;
        }
        if (!foundHigh && cumSum >= highTarget) {
            highBin = i;
            foundHigh = true;
            break;
        }
    }

    // Convert bins back to magnitude values
    lowThreshold = (lowBin / binScale);
    highThreshold = (highBin / binScale);

    // Ensure minimum separation
    if (highThreshold < lowThreshold * 1.5) {
        highThreshold = lowThreshold * 2.0;
    }
}

// ============================================================================
// Subpixel Refinement
// ============================================================================

double RefineEdgeSubpixel(const float* magnitude, const float* direction,
                          int32_t width, int32_t height,
                          int32_t x, int32_t y,
                          double& subX, double& subY) {
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        subX = x;
        subY = y;
        return magnitude[y * width + x];
    }

    // Get gradient direction (perpendicular to edge)
    double dir = direction[y * width + x];

    // Get the gradient direction components
    double dx = std::cos(dir);
    double dy = std::sin(dir);

    // Sample along the gradient direction
    // p0 = point at -1 step, p1 = center, p2 = point at +1 step
    auto sample = [&](double ox, double oy) -> double {
        int32_t ix = x + static_cast<int32_t>(std::round(ox));
        int32_t iy = y + static_cast<int32_t>(std::round(oy));
        ix = std::clamp(ix, 0, width - 1);
        iy = std::clamp(iy, 0, height - 1);
        return magnitude[iy * width + ix];
    };

    double p0 = sample(-dx, -dy);
    double p1 = magnitude[y * width + x];
    double p2 = sample(dx, dy);

    // Parabolic interpolation to find subpixel offset
    double denom = 2.0 * (p0 - 2.0 * p1 + p2);
    double offset = 0.0;

    if (std::abs(denom) > 1e-10) {
        offset = (p0 - p2) / denom;
        offset = std::clamp(offset, -0.5, 0.5);
    }

    // Apply offset along gradient direction
    subX = x + offset * dx;
    subY = y + offset * dy;

    // Interpolate magnitude at subpixel position
    double interpMag = p1 - 0.25 * (p0 - p2) * offset;
    return std::max(0.0, interpMag);
}

// ============================================================================
// Edge Point Extraction
// ============================================================================

std::vector<CannyEdgePoint> ExtractEdgePoints(const uint8_t* edgeImage,
                                               const float* magnitude,
                                               const float* direction,
                                               int32_t width, int32_t height,
                                               bool refineSubpixel) {
    std::vector<CannyEdgePoint> points;
    points.reserve(width * height / 10);  // Rough estimate

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            if (edgeImage[y * width + x] > 0) {
                CannyEdgePoint pt;

                if (refineSubpixel) {
                    pt.magnitude = RefineEdgeSubpixel(magnitude, direction,
                                                       width, height,
                                                       x, y,
                                                       pt.x, pt.y);
                } else {
                    pt.x = x;
                    pt.y = y;
                    pt.magnitude = magnitude[y * width + x];
                }

                // Edge direction is perpendicular to gradient
                pt.direction = EdgeDirection(direction[y * width + x]);

                points.push_back(pt);
            }
        }
    }

    return points;
}

// ============================================================================
// Edge Linking
// ============================================================================

std::vector<QContour> LinkCannyEdges(const std::vector<CannyEdgePoint>& edgePoints,
                                      int32_t width, int32_t height,
                                      double minLength,
                                      int32_t minPoints) {
    (void)width;
    (void)height;
    if (edgePoints.empty()) {
        return {};
    }

    // Convert CannyEdgePoint to EdgePoint for linking
    std::vector<EdgePoint> linkPoints;
    linkPoints.reserve(edgePoints.size());

    for (size_t i = 0; i < edgePoints.size(); ++i) {
        EdgePoint ep;
        ep.x = edgePoints[i].x;
        ep.y = edgePoints[i].y;
        ep.direction = edgePoints[i].direction;
        ep.magnitude = edgePoints[i].magnitude;
        ep.id = static_cast<int32_t>(i);
        linkPoints.push_back(ep);
    }

    // Set up linking parameters
    EdgeLinkingParams params;
    params.maxGap = 2.0;           // Max gap between consecutive points
    params.maxAngleDiff = 0.5;     // ~30 degrees
    params.minChainLength = minLength;
    params.minChainPoints = minPoints;
    params.closedContours = true;
    params.closureMaxGap = 3.0;
    params.closureMaxAngle = 0.6;

    // Link and convert to contours
    return LinkToContours(linkPoints, params);
}

std::vector<QContour> LinkEdgePixels(const uint8_t* edgeImage,
                                      const float* magnitude,
                                      const float* direction,
                                      int32_t width, int32_t height,
                                      double minLength) {
    // Extract edge points with subpixel refinement
    auto edgePoints = ExtractEdgePoints(edgeImage, magnitude, direction,
                                        width, height, true);

    // Link into contours
    return LinkCannyEdges(edgePoints, width, height, minLength, 3);
}

// ============================================================================
// Main Detection Functions
// ============================================================================

std::vector<QContour> DetectEdgesCanny(const QImage& image,
                                        const CannyParams& params) {
    auto result = DetectEdgesCannyFull(image, params);
    return std::move(result.contours);
}

CannyResult DetectEdgesCannyFull(const QImage& image,
                                  const CannyParams& params) {
    CannyResult result;

    // Validate input
    if (image.Empty()) {
        return result;
    }

    const int32_t width = image.Width();
    const int32_t height = image.Height();
    const int32_t size = width * height;

    // Get image data (assume grayscale)
    const uint8_t* srcData = static_cast<const uint8_t*>(image.Data());
    if (!srcData) {
        return result;
    }

    // Step 1: Gaussian smoothing
    std::vector<float> smoothed(size);
    CannySmooth(srcData, smoothed.data(), width, height, params.sigma);

    // Step 2: Compute gradients
    std::vector<float> magnitude(size);
    std::vector<float> direction(size);
    CannyGradient(smoothed.data(), magnitude.data(), direction.data(),
                  width, height, params.gradientOp);

    // Step 3: Non-maximum suppression
    std::vector<float> nmsOutput(size);
    CannyNMS(magnitude.data(), direction.data(), nmsOutput.data(),
             width, height);

    // Step 4: Determine thresholds
    double lowThresh = params.lowThreshold;
    double highThresh = params.highThreshold;

    if (params.autoThreshold) {
        ComputeAutoThresholds(magnitude.data(), width, height,
                              lowThresh, highThresh);
    }

    // Step 5: Hysteresis thresholding
    std::vector<uint8_t> edgeBinary(size);
    CannyHysteresis(nmsOutput.data(), edgeBinary.data(),
                    width, height, lowThresh, highThresh);

    // Count edge pixels and compute average magnitude
    int32_t edgeCount = 0;
    double magSum = 0.0;

    for (int32_t i = 0; i < size; ++i) {
        if (edgeBinary[i] > 0) {
            edgeCount++;
            magSum += magnitude[i];
        }
    }

    result.numEdgePixels = edgeCount;
    result.avgMagnitude = edgeCount > 0 ? magSum / edgeCount : 0.0;

    // Step 6: Extract edge points with optional subpixel refinement
    result.edgePoints = ExtractEdgePoints(edgeBinary.data(),
                                          magnitude.data(),
                                          direction.data(),
                                          width, height,
                                          params.subpixelRefinement);

    // Step 7: Link edges into contours if requested
    if (params.linkEdges) {
        result.contours = LinkCannyEdges(result.edgePoints,
                                         width, height,
                                         params.minContourLength,
                                         params.minContourPoints);
    }

    // Step 8: Create binary edge image
    result.edgeImage = QImage(width, height, PixelType::UInt8, ChannelType::Gray);
    std::memcpy(static_cast<uint8_t*>(result.edgeImage.Data()), edgeBinary.data(), size);

    return result;
}

QImage DetectEdgesCannyImage(const QImage& image,
                              const CannyParams& params) {
    // Use simplified version without linking
    CannyParams simpleParams = params;
    simpleParams.linkEdges = false;
    simpleParams.subpixelRefinement = false;

    auto result = DetectEdgesCannyFull(image, simpleParams);
    return std::move(result.edgeImage);
}

} // namespace Qi::Vision::Internal
