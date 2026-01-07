/**
 * @file NonMaxSuppression.cpp
 * @brief Implementation of Non-Maximum Suppression algorithms
 */

#include <QiVision/Internal/NonMaxSuppression.h>

#include <queue>
#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <cstring>

namespace Qi::Vision::Internal {

// ============================================================================
// 1D Non-Maximum Suppression
// ============================================================================

std::vector<int32_t> FindLocalMaxima1D(const double* signal, size_t size,
                                        double minValue) {
    std::vector<int32_t> maxima;
    if (size < 3) return maxima;

    for (size_t i = 1; i < size - 1; ++i) {
        double v = signal[i];
        if (v >= minValue &&
            v > signal[i - 1] &&
            v > signal[i + 1]) {
            maxima.push_back(static_cast<int32_t>(i));
        }
    }

    return maxima;
}

std::vector<Peak1D> FindLocalMaxima1DRadius(const double* signal, size_t size,
                                             int32_t radius,
                                             double minValue) {
    std::vector<Peak1D> peaks;
    if (size < static_cast<size_t>(2 * radius + 1)) return peaks;

    for (size_t i = static_cast<size_t>(radius);
         i < size - static_cast<size_t>(radius); ++i) {
        double v = signal[i];
        if (v < minValue) continue;

        bool isMax = true;
        for (int32_t k = -radius; k <= radius; ++k) {
            if (k == 0) continue;
            if (signal[i + k] >= v) {
                isMax = false;
                break;
            }
        }

        if (isMax) {
            Peak1D peak;
            peak.index = static_cast<int32_t>(i);
            peak.value = v;
            peak.subpixelIndex = static_cast<double>(i);
            peaks.push_back(peak);
        }
    }

    return peaks;
}

std::vector<Peak1D> FindPeaks1D(const double* signal, size_t size,
                                 int32_t radius,
                                 double minValue,
                                 bool refineSubpixel) {
    std::vector<Peak1D> peaks = FindLocalMaxima1DRadius(signal, size,
                                                         radius, minValue);

    if (refineSubpixel && !peaks.empty()) {
        for (auto& peak : peaks) {
            int32_t i = peak.index;
            if (i > 0 && i < static_cast<int32_t>(size) - 1) {
                double offset = RefineSubpixelParabolic(
                    signal[i - 1], signal[i], signal[i + 1]);
                peak.subpixelIndex = static_cast<double>(i) + offset;
                peak.value = InterpolatedPeakValue(
                    signal[i - 1], signal[i], signal[i + 1], offset);
            }
        }
    }

    return peaks;
}

std::vector<Peak1D> FindValleys1D(const double* signal, size_t size,
                                   int32_t radius,
                                   double maxValue,
                                   bool refineSubpixel) {
    // Create negated signal
    std::vector<double> negSignal(size);
    for (size_t i = 0; i < size; ++i) {
        negSignal[i] = -signal[i];
    }

    // Find peaks in negated signal
    std::vector<Peak1D> valleys = FindPeaks1D(negSignal.data(), size,
                                               radius, -maxValue, refineSubpixel);

    // Restore original values
    for (auto& valley : valleys) {
        valley.value = -valley.value;
    }

    return valleys;
}

std::vector<Peak1D> SuppressPeaks1D(std::vector<Peak1D> peaks,
                                     int32_t maxCount,
                                     double minDistance) {
    if (peaks.empty()) return peaks;

    // Sort by value (descending)
    std::sort(peaks.begin(), peaks.end());

    std::vector<Peak1D> result;
    result.reserve(std::min(static_cast<size_t>(maxCount), peaks.size()));

    for (const auto& peak : peaks) {
        if (static_cast<int32_t>(result.size()) >= maxCount) break;

        // Check distance to all kept peaks
        bool suppress = false;
        for (const auto& kept : result) {
            if (std::abs(peak.subpixelIndex - kept.subpixelIndex) < minDistance) {
                suppress = true;
                break;
            }
        }

        if (!suppress) {
            result.push_back(peak);
        }
    }

    return result;
}

// ============================================================================
// 2D Gradient-based NMS (Canny)
// ============================================================================

void NMS2DGradient(const float* magnitude, const float* direction,
                   float* output,
                   int32_t width, int32_t height,
                   float lowThreshold) {
    std::memset(output, 0, sizeof(float) * width * height);

    for (int32_t y = 1; y < height - 1; ++y) {
        for (int32_t x = 1; x < width - 1; ++x) {
            int32_t idx = y * width + x;
            float mag = magnitude[idx];

            if (mag < lowThreshold) continue;

            float angle = direction[idx];
            if (angle < 0) angle += static_cast<float>(M_PI);

            // Compute neighbor positions along gradient direction
            float dx = std::cos(angle);
            float dy = std::sin(angle);

            // Bilinear interpolation for neighbors along gradient
            auto interpolate = [&](float fx, float fy) -> float {
                int32_t x0 = static_cast<int32_t>(std::floor(fx));
                int32_t y0 = static_cast<int32_t>(std::floor(fy));
                int32_t x1 = x0 + 1;
                int32_t y1 = y0 + 1;

                // Clamp to image bounds
                x0 = std::max(0, std::min(x0, width - 1));
                x1 = std::max(0, std::min(x1, width - 1));
                y0 = std::max(0, std::min(y0, height - 1));
                y1 = std::max(0, std::min(y1, height - 1));

                float tx = fx - std::floor(fx);
                float ty = fy - std::floor(fy);

                float v00 = magnitude[y0 * width + x0];
                float v01 = magnitude[y0 * width + x1];
                float v10 = magnitude[y1 * width + x0];
                float v11 = magnitude[y1 * width + x1];

                return (1 - ty) * ((1 - tx) * v00 + tx * v01) +
                       ty * ((1 - tx) * v10 + tx * v11);
            };

            float mag1 = interpolate(x + dx, y + dy);
            float mag2 = interpolate(x - dx, y - dy);

            // Keep only if local maximum along gradient direction
            if (mag >= mag1 && mag >= mag2) {
                output[idx] = mag;
            }
        }
    }
}

void NMS2DGradientQuantized(const float* magnitude, const float* direction,
                            float* output,
                            int32_t width, int32_t height) {
    std::memset(output, 0, sizeof(float) * width * height);

    // Neighbor offsets for 4 quantized directions
    // Direction 0 (horizontal): check left/right
    // Direction 1 (45°): check top-right/bottom-left
    // Direction 2 (vertical): check top/bottom
    // Direction 3 (135°): check top-left/bottom-right
    const int32_t dx1[4] = {1, 1, 0, -1};
    const int32_t dy1[4] = {0, -1, -1, -1};
    const int32_t dx2[4] = {-1, -1, 0, 1};
    const int32_t dy2[4] = {0, 1, 1, 1};

    for (int32_t y = 1; y < height - 1; ++y) {
        for (int32_t x = 1; x < width - 1; ++x) {
            int32_t idx = y * width + x;
            float mag = magnitude[idx];

            if (mag < 1e-6f) continue;

            int32_t dir = QuantizeDirection(direction[idx]);

            // Get neighbor magnitudes
            float mag1 = magnitude[(y + dy1[dir]) * width + (x + dx1[dir])];
            float mag2 = magnitude[(y + dy2[dir]) * width + (x + dx2[dir])];

            // Keep only if local maximum
            if (mag >= mag1 && mag >= mag2) {
                output[idx] = mag;
            }
        }
    }
}

// ============================================================================
// 2D Feature Point NMS
// ============================================================================

std::vector<Peak2D> FindLocalMaxima2D(const float* response,
                                       int32_t width, int32_t height,
                                       int32_t radius,
                                       double minValue) {
    std::vector<Peak2D> peaks;

    for (int32_t y = radius; y < height - radius; ++y) {
        for (int32_t x = radius; x < width - radius; ++x) {
            float v = response[y * width + x];
            if (v < static_cast<float>(minValue)) continue;

            bool isMax = true;
            for (int32_t dy = -radius; dy <= radius && isMax; ++dy) {
                for (int32_t dx = -radius; dx <= radius && isMax; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (response[(y + dy) * width + (x + dx)] >= v) {
                        isMax = false;
                    }
                }
            }

            if (isMax) {
                Peak2D peak;
                peak.x = x;
                peak.y = y;
                peak.value = v;
                peak.subpixelX = x;
                peak.subpixelY = y;
                peaks.push_back(peak);
            }
        }
    }

    return peaks;
}

std::vector<Peak2D> FindPeaks2D(const float* response,
                                 int32_t width, int32_t height,
                                 int32_t radius,
                                 double minValue,
                                 bool refineSubpixel) {
    std::vector<Peak2D> peaks = FindLocalMaxima2D(response, width, height,
                                                   radius, minValue);

    if (refineSubpixel && !peaks.empty()) {
        for (auto& peak : peaks) {
            double subX, subY;
            double val = RefineSubpixel2D(response, width, height,
                                          peak.x, peak.y, subX, subY);
            peak.subpixelX = subX;
            peak.subpixelY = subY;
            peak.value = val;
        }
    }

    return peaks;
}

std::vector<Peak2D> SuppressPeaks2D(std::vector<Peak2D> peaks,
                                     int32_t maxCount,
                                     double minDistance) {
    if (peaks.empty()) return peaks;

    // Sort by value (descending)
    std::sort(peaks.begin(), peaks.end());

    std::vector<Peak2D> result;
    result.reserve(std::min(static_cast<size_t>(maxCount), peaks.size()));

    for (const auto& peak : peaks) {
        if (static_cast<int32_t>(result.size()) >= maxCount) break;

        bool suppress = false;
        for (const auto& kept : result) {
            if (PeakDistance(peak, kept) < minDistance) {
                suppress = true;
                break;
            }
        }

        if (!suppress) {
            result.push_back(peak);
        }
    }

    return result;
}

std::vector<Peak2D> SuppressPeaks2DGrid(const std::vector<Peak2D>& peaks,
                                         int32_t width, int32_t height,
                                         int32_t cellSize) {
    if (peaks.empty() || cellSize <= 0) return peaks;

    int32_t gridW = (width + cellSize - 1) / cellSize;
    int32_t gridH = (height + cellSize - 1) / cellSize;

    // Grid stores best peak index for each cell (-1 = empty)
    std::vector<int32_t> grid(gridW * gridH, -1);
    std::vector<double> gridValues(gridW * gridH, -std::numeric_limits<double>::max());

    for (size_t i = 0; i < peaks.size(); ++i) {
        int32_t gx = peaks[i].x / cellSize;
        int32_t gy = peaks[i].y / cellSize;
        int32_t gIdx = gy * gridW + gx;

        if (peaks[i].value > gridValues[gIdx]) {
            gridValues[gIdx] = peaks[i].value;
            grid[gIdx] = static_cast<int32_t>(i);
        }
    }

    std::vector<Peak2D> result;
    for (int32_t idx : grid) {
        if (idx >= 0) {
            result.push_back(peaks[idx]);
        }
    }

    return result;
}

// ============================================================================
// 2D Subpixel Refinement
// ============================================================================

double RefineSubpixel2D(const float* response,
                        int32_t width, int32_t height,
                        int32_t x, int32_t y,
                        double& subX, double& subY) {
    // Get 3x3 neighborhood
    auto get = [&](int32_t dx, int32_t dy) -> double {
        int32_t px = std::max(0, std::min(width - 1, x + dx));
        int32_t py = std::max(0, std::min(height - 1, y + dy));
        return static_cast<double>(response[py * width + px]);
    };

    double v00 = get(-1, -1), v01 = get(0, -1), v02 = get(1, -1);
    double v10 = get(-1,  0), v11 = get(0,  0), v12 = get(1,  0);
    double v20 = get(-1,  1), v21 = get(0,  1), v22 = get(1,  1);

    // Separate 1D refinement along x and y
    double offsetX = RefineSubpixelParabolic(v10, v11, v12);
    double offsetY = RefineSubpixelParabolic(v01, v11, v21);

    subX = x + offsetX;
    subY = y + offsetY;

    // Interpolate value at refined position (bilinear approximation)
    double fx = offsetX;
    double fy = offsetY;
    double value = v11 +
        0.5 * fx * (v12 - v10) +
        0.5 * fy * (v21 - v01) +
        0.25 * fx * fy * (v00 - v02 - v20 + v22);

    return value;
}

double RefineSubpixel2DTaylor(const float* response,
                              int32_t width, int32_t height,
                              int32_t x, int32_t y,
                              double& subX, double& subY) {
    // Get 3x3 neighborhood
    auto get = [&](int32_t dx, int32_t dy) -> double {
        int32_t px = std::max(0, std::min(width - 1, x + dx));
        int32_t py = std::max(0, std::min(height - 1, y + dy));
        return static_cast<double>(response[py * width + px]);
    };

    double v00 = get(-1, -1), v01 = get(0, -1), v02 = get(1, -1);
    double v10 = get(-1,  0), v11 = get(0,  0), v12 = get(1,  0);
    double v20 = get(-1,  1), v21 = get(0,  1), v22 = get(1,  1);

    // Compute gradient
    double dx = (v12 - v10) * 0.5;
    double dy = (v21 - v01) * 0.5;

    // Compute Hessian
    double dxx = v12 - 2.0 * v11 + v10;
    double dyy = v21 - 2.0 * v11 + v01;
    double dxy = (v22 - v20 - v02 + v00) * 0.25;

    // Solve 2x2 system: H * offset = -gradient
    double det = dxx * dyy - dxy * dxy;
    if (std::abs(det) < 1e-10) {
        // Degenerate case, no refinement
        subX = x;
        subY = y;
        return v11;
    }

    double offsetX = -(dyy * dx - dxy * dy) / det;
    double offsetY = -(dxx * dy - dxy * dx) / det;

    // Clamp offset to prevent large jumps
    offsetX = std::max(-1.0, std::min(1.0, offsetX));
    offsetY = std::max(-1.0, std::min(1.0, offsetY));

    subX = x + offsetX;
    subY = y + offsetY;

    // Compute interpolated value using Taylor expansion
    double value = v11 + 0.5 * (dx * offsetX + dy * offsetY);

    return value;
}

// ============================================================================
// Box NMS
// ============================================================================

std::vector<int32_t> NMSBoxes(const std::vector<BoundingBox>& boxes,
                               double iouThreshold) {
    if (boxes.empty()) return {};

    // Create sorted indices
    std::vector<int32_t> indices(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        indices[i] = static_cast<int32_t>(i);
    }
    std::sort(indices.begin(), indices.end(),
              [&boxes](int32_t a, int32_t b) {
                  return boxes[a].score > boxes[b].score;
              });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<int32_t> result;

    for (int32_t idx : indices) {
        if (suppressed[idx]) continue;

        result.push_back(idx);

        // Suppress overlapping boxes
        for (size_t j = 0; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            if (ComputeIoU(boxes[idx], boxes[j]) >= iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

std::vector<int32_t> NMSBoxesMultiClass(const std::vector<BoundingBox>& boxes,
                                         double iouThreshold) {
    if (boxes.empty()) return {};

    // Group by class
    std::unordered_map<int32_t, std::vector<int32_t>> classBoxes;
    for (size_t i = 0; i < boxes.size(); ++i) {
        classBoxes[boxes[i].classId].push_back(static_cast<int32_t>(i));
    }

    std::vector<int32_t> result;

    // Apply NMS per class
    for (auto& [classId, boxIndices] : classBoxes) {
        // Sort by score
        std::sort(boxIndices.begin(), boxIndices.end(),
                  [&boxes](int32_t a, int32_t b) {
                      return boxes[a].score > boxes[b].score;
                  });

        std::vector<bool> suppressed(boxIndices.size(), false);

        for (size_t i = 0; i < boxIndices.size(); ++i) {
            if (suppressed[i]) continue;

            int32_t idx = boxIndices[i];
            result.push_back(idx);

            for (size_t j = i + 1; j < boxIndices.size(); ++j) {
                if (suppressed[j]) continue;
                if (ComputeIoU(boxes[idx], boxes[boxIndices[j]]) >= iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    // Sort result by original order for consistency
    std::sort(result.begin(), result.end());

    return result;
}

std::vector<int32_t> SoftNMSBoxes(std::vector<BoundingBox>& boxes,
                                   double sigma,
                                   double scoreThreshold) {
    if (boxes.empty()) return {};

    std::vector<int32_t> indices(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        indices[i] = static_cast<int32_t>(i);
    }

    std::vector<int32_t> result;

    while (!indices.empty()) {
        // Find box with highest score
        auto maxIt = std::max_element(indices.begin(), indices.end(),
            [&boxes](int32_t a, int32_t b) {
                return boxes[a].score < boxes[b].score;
            });

        int32_t maxIdx = *maxIt;
        indices.erase(maxIt);

        if (boxes[maxIdx].score >= scoreThreshold) {
            result.push_back(maxIdx);
        }

        // Decay overlapping scores
        for (int32_t idx : indices) {
            double iou = ComputeIoU(boxes[maxIdx], boxes[idx]);
            double decay = std::exp(-(iou * iou) / sigma);
            boxes[idx].score *= decay;
        }
    }

    return result;
}

// ============================================================================
// Hysteresis Thresholding
// ============================================================================

void HysteresisThreshold(const float* edges, uint8_t* output,
                         int32_t width, int32_t height,
                         float lowThreshold, float highThreshold) {
    // Initialize output
    std::memset(output, 0, width * height);

    // Mark strong and weak edges
    constexpr uint8_t STRONG = 255;
    constexpr uint8_t WEAK = 128;

    std::queue<int32_t> strongQueue;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            int32_t idx = y * width + x;
            float v = edges[idx];

            if (v >= highThreshold) {
                output[idx] = STRONG;
                strongQueue.push(idx);
            } else if (v >= lowThreshold) {
                output[idx] = WEAK;
            }
        }
    }

    // BFS from strong edges to connect weak edges
    const int32_t dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int32_t dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    while (!strongQueue.empty()) {
        int32_t idx = strongQueue.front();
        strongQueue.pop();

        int32_t x = idx % width;
        int32_t y = idx / width;

        for (int32_t i = 0; i < 8; ++i) {
            int32_t nx = x + dx[i];
            int32_t ny = y + dy[i];

            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            int32_t nidx = ny * width + nx;
            if (output[nidx] == WEAK) {
                output[nidx] = STRONG;
                strongQueue.push(nidx);
            }
        }
    }

    // Clear remaining weak edges
    for (int32_t i = 0; i < width * height; ++i) {
        if (output[i] == WEAK) {
            output[i] = 0;
        }
    }
}

void HysteresisThresholdInPlace(float* edges,
                                 int32_t width, int32_t height,
                                 float lowThreshold, float highThreshold) {
    // Create temporary output
    std::vector<uint8_t> output(width * height);
    HysteresisThreshold(edges, output.data(), width, height,
                        lowThreshold, highThreshold);

    // Copy back to edges
    for (int32_t i = 0; i < width * height; ++i) {
        edges[i] = (output[i] == 255) ? 255.0f : 0.0f;
    }
}

} // namespace Qi::Vision::Internal
