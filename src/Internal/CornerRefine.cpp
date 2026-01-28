/**
 * @file CornerRefine.cpp
 * @brief Subpixel corner refinement implementation
 */

#include <QiVision/Internal/CornerRefine.h>
#include <QiVision/Internal/Gradient.h>
#include <QiVision/Internal/Interpolate.h>

#include <algorithm>
#include <cmath>

namespace Qi::Vision::Internal {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

/**
 * @brief Get pixel value with border handling
 */
inline double GetPixelSafe(const uint8_t* data, int32_t width, int32_t height,
                           int32_t x, int32_t y) {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    return static_cast<double>(data[y * width + x]);
}

/**
 * @brief Compute gradient at integer pixel location
 */
inline void ComputeGradient(const uint8_t* data, int32_t width, int32_t height,
                            int32_t x, int32_t y, double& gx, double& gy) {
    // Central difference
    double left = GetPixelSafe(data, width, height, x - 1, y);
    double right = GetPixelSafe(data, width, height, x + 1, y);
    double top = GetPixelSafe(data, width, height, x, y - 1);
    double bottom = GetPixelSafe(data, width, height, x, y + 1);

    gx = (right - left) * 0.5;
    gy = (bottom - top) * 0.5;
}

} // anonymous namespace

// =============================================================================
// Corner Refinement
// =============================================================================

bool RefineCornerGradient(
    const QImage& image,
    Point2d& corner,
    int32_t winSize,
    int32_t maxIterations,
    double epsilon)
{
    if (image.Empty() || image.Type() != PixelType::UInt8) {
        return false;
    }

    const int32_t width = image.Width();
    const int32_t height = image.Height();
    const uint8_t* data = static_cast<const uint8_t*>(image.Data());

    // Check if corner is within valid range
    if (corner.x < winSize || corner.x >= width - winSize ||
        corner.y < winSize || corner.y >= height - winSize) {
        return false;
    }

    Point2d current = corner;

    for (int32_t iter = 0; iter < maxIterations; ++iter) {
        // Build the 2x2 linear system:
        // [sum(Ix*Ix)  sum(Ix*Iy)] [dx]   [sum(Ix*(xi-x) + Iy*(yi-y))]
        // [sum(Ix*Iy)  sum(Iy*Iy)] [dy] = [...]
        //
        // The goal is to find the point where sum(gradient . (p - corner)) = 0
        // This is the point where gradients point away radially (corner)

        double a = 0.0, b = 0.0, c = 0.0;  // Structure tensor elements
        double bx = 0.0, by = 0.0;          // Right-hand side

        int32_t cx = static_cast<int32_t>(std::round(current.x));
        int32_t cy = static_cast<int32_t>(std::round(current.y));

        // Clamp center to valid range for iteration
        cx = std::clamp(cx, winSize, width - winSize - 1);
        cy = std::clamp(cy, winSize, height - winSize - 1);

        for (int32_t dy = -winSize; dy <= winSize; ++dy) {
            for (int32_t dx = -winSize; dx <= winSize; ++dx) {
                int32_t px = cx + dx;
                int32_t py = cy + dy;

                double gx, gy;
                ComputeGradient(data, width, height, px, py, gx, gy);

                // Structure tensor
                a += gx * gx;
                b += gx * gy;
                c += gy * gy;

                // Right-hand side
                // We want to find corner where dot(gradient, p - corner) = 0
                // Rearranging: sum(Ix*Ix)*cx + sum(Ix*Iy)*cy = sum(Ix*px + Iy*py)
                bx += gx * gx * px + gx * gy * py;
                by += gx * gy * px + gy * gy * py;
            }
        }

        // Solve 2x2 system: [a b][newX] = [bx]
        //                   [b c][newY]   [by]
        double det = a * c - b * b;
        if (std::abs(det) < 1e-10) {
            // Singular matrix, cannot refine
            break;
        }

        double newX = (c * bx - b * by) / det;
        double newY = (a * by - b * bx) / det;

        // Check convergence
        double moveX = newX - current.x;
        double moveY = newY - current.y;
        double movement = std::sqrt(moveX * moveX + moveY * moveY);

        current.x = newX;
        current.y = newY;

        if (movement < epsilon) {
            corner = current;
            return true;
        }

        // Safety check: don't move too far from original estimate
        if (std::abs(current.x - corner.x) > winSize * 2 ||
            std::abs(current.y - corner.y) > winSize * 2) {
            // Diverging, keep original
            return false;
        }
    }

    corner = current;
    return true;
}

void RefineCorners(
    const QImage& image,
    std::vector<Point2d>& corners,
    int32_t winSize,
    int32_t maxIterations,
    double epsilon)
{
    for (auto& corner : corners) {
        RefineCornerGradient(image, corner, winSize, maxIterations, epsilon);
    }
}

// =============================================================================
// Structure Tensor
// =============================================================================

void ComputeStructureTensor(
    const QImage& image,
    int32_t x,
    int32_t y,
    int32_t blockSize,
    double& Ixx,
    double& Ixy,
    double& Iyy)
{
    Ixx = Ixy = Iyy = 0.0;

    if (image.Empty() || image.Type() != PixelType::UInt8) {
        return;
    }

    const int32_t width = image.Width();
    const int32_t height = image.Height();
    const uint8_t* data = static_cast<const uint8_t*>(image.Data());

    int32_t halfSize = blockSize / 2;

    for (int32_t dy = -halfSize; dy <= halfSize; ++dy) {
        for (int32_t dx = -halfSize; dx <= halfSize; ++dx) {
            int32_t px = x + dx;
            int32_t py = y + dy;

            if (px < 1 || px >= width - 1 || py < 1 || py >= height - 1) {
                continue;
            }

            double gx, gy;
            ComputeGradient(data, width, height, px, py, gx, gy);

            Ixx += gx * gx;
            Ixy += gx * gy;
            Iyy += gy * gy;
        }
    }
}

void Eigenvalues2x2(double a, double b, double c, double& lambda1, double& lambda2) {
    // For symmetric 2x2 matrix:
    // | a  b |
    // | b  c |
    // Eigenvalues: lambda = (a+c)/2 +/- sqrt((a-c)^2/4 + b^2)

    double trace = a + c;
    double det = a * c - b * b;
    double disc = std::sqrt(std::max(0.0, trace * trace - 4.0 * det));

    lambda1 = (trace + disc) * 0.5;
    lambda2 = (trace - disc) * 0.5;
}

// =============================================================================
// Harris Corner Detection
// =============================================================================

double HarrisResponse(
    const QImage& image,
    int32_t x,
    int32_t y,
    int32_t blockSize,
    double k)
{
    double Ixx, Ixy, Iyy;
    ComputeStructureTensor(image, x, y, blockSize, Ixx, Ixy, Iyy);

    // Harris response: det(M) - k * trace(M)^2
    double det = Ixx * Iyy - Ixy * Ixy;
    double trace = Ixx + Iyy;

    return det - k * trace * trace;
}

void HarrisResponseImage(
    const QImage& image,
    QImage& response,
    int32_t blockSize,
    double k)
{
    if (image.Empty()) {
        return;
    }

    const int32_t width = image.Width();
    const int32_t height = image.Height();

    response = QImage(width, height, PixelType::Float32, ChannelType::Gray);
    float* respData = static_cast<float*>(response.Data());

    const int32_t border = blockSize / 2 + 1;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            if (x < border || x >= width - border ||
                y < border || y >= height - border) {
                respData[y * width + x] = 0.0f;
            } else {
                respData[y * width + x] = static_cast<float>(
                    HarrisResponse(image, x, y, blockSize, k));
            }
        }
    }
}

std::vector<Point2d> DetectHarrisCorners(
    const QImage& image,
    int32_t maxCorners,
    double qualityLevel,
    double minDistance,
    int32_t blockSize,
    double k)
{
    std::vector<Point2d> corners;

    if (image.Empty()) {
        return corners;
    }

    // Compute Harris response image
    QImage response;
    HarrisResponseImage(image, response, blockSize, k);

    const int32_t width = image.Width();
    const int32_t height = image.Height();
    const float* respData = static_cast<const float*>(response.Data());

    // Find maximum response
    float maxResp = 0.0f;
    for (int32_t i = 0; i < width * height; ++i) {
        if (respData[i] > maxResp) {
            maxResp = respData[i];
        }
    }

    if (maxResp <= 0) {
        return corners;
    }

    float threshold = static_cast<float>(maxResp * qualityLevel);

    // Non-maximum suppression and collect corners
    std::vector<std::pair<double, Point2d>> candidates;
    const int32_t nmsSize = 3;  // NMS window size
    const int32_t halfNms = nmsSize / 2;

    for (int32_t y = halfNms; y < height - halfNms; ++y) {
        for (int32_t x = halfNms; x < width - halfNms; ++x) {
            float val = respData[y * width + x];
            if (val < threshold) continue;

            // Check if local maximum
            bool isMax = true;
            for (int32_t dy = -halfNms; dy <= halfNms && isMax; ++dy) {
                for (int32_t dx = -halfNms; dx <= halfNms && isMax; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (respData[(y + dy) * width + (x + dx)] >= val) {
                        isMax = false;
                    }
                }
            }

            if (isMax) {
                candidates.emplace_back(val, Point2d(x, y));
            }
        }
    }

    // Sort by response (strongest first)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Filter by minimum distance
    std::vector<double> strengths;
    for (const auto& c : candidates) {
        strengths.push_back(c.first);
        corners.push_back(c.second);
    }

    corners = FilterByDistance(corners, strengths, minDistance);

    // Limit number of corners
    if (maxCorners > 0 && corners.size() > static_cast<size_t>(maxCorners)) {
        corners.resize(maxCorners);
    }

    return corners;
}

// =============================================================================
// Shi-Tomasi Corner Detection
// =============================================================================

double ShiTomasiResponse(
    const QImage& image,
    int32_t x,
    int32_t y,
    int32_t blockSize)
{
    double Ixx, Ixy, Iyy;
    ComputeStructureTensor(image, x, y, blockSize, Ixx, Ixy, Iyy);

    // Shi-Tomasi response: min(lambda1, lambda2)
    double lambda1, lambda2;
    Eigenvalues2x2(Ixx, Ixy, Iyy, lambda1, lambda2);

    return std::min(lambda1, lambda2);
}

std::vector<Point2d> DetectShiTomasiCorners(
    const QImage& image,
    int32_t maxCorners,
    double qualityLevel,
    double minDistance,
    int32_t blockSize)
{
    std::vector<Point2d> corners;

    if (image.Empty()) {
        return corners;
    }

    const int32_t width = image.Width();
    const int32_t height = image.Height();

    // Compute Shi-Tomasi response
    std::vector<float> response(width * height, 0.0f);
    const int32_t border = blockSize / 2 + 1;

    float maxResp = 0.0f;
    for (int32_t y = border; y < height - border; ++y) {
        for (int32_t x = border; x < width - border; ++x) {
            float val = static_cast<float>(ShiTomasiResponse(image, x, y, blockSize));
            response[y * width + x] = val;
            if (val > maxResp) {
                maxResp = val;
            }
        }
    }

    if (maxResp <= 0) {
        return corners;
    }

    float threshold = static_cast<float>(maxResp * qualityLevel);

    // Non-maximum suppression and collect corners
    std::vector<std::pair<double, Point2d>> candidates;
    const int32_t nmsSize = 3;
    const int32_t halfNms = nmsSize / 2;

    for (int32_t y = std::max(border, halfNms); y < height - std::max(border, halfNms); ++y) {
        for (int32_t x = std::max(border, halfNms); x < width - std::max(border, halfNms); ++x) {
            float val = response[y * width + x];
            if (val < threshold) continue;

            bool isMax = true;
            for (int32_t dy = -halfNms; dy <= halfNms && isMax; ++dy) {
                for (int32_t dx = -halfNms; dx <= halfNms && isMax; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (response[(y + dy) * width + (x + dx)] >= val) {
                        isMax = false;
                    }
                }
            }

            if (isMax) {
                candidates.emplace_back(val, Point2d(x, y));
            }
        }
    }

    // Sort by response (strongest first)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Filter by minimum distance
    std::vector<double> strengths;
    for (const auto& c : candidates) {
        strengths.push_back(c.first);
        corners.push_back(c.second);
    }

    corners = FilterByDistance(corners, strengths, minDistance);

    // Limit number of corners
    if (maxCorners > 0 && corners.size() > static_cast<size_t>(maxCorners)) {
        corners.resize(maxCorners);
    }

    return corners;
}

// =============================================================================
// Non-Maximum Suppression
// =============================================================================

std::vector<Point2d> NonMaximumSuppressionCorners(
    const QImage& response,
    int32_t winSize,
    double threshold)
{
    std::vector<Point2d> corners;

    if (response.Empty() || response.Type() != PixelType::Float32) {
        return corners;
    }

    const int32_t width = response.Width();
    const int32_t height = response.Height();
    const float* data = static_cast<const float*>(response.Data());

    const int32_t halfSize = winSize / 2;

    for (int32_t y = halfSize; y < height - halfSize; ++y) {
        for (int32_t x = halfSize; x < width - halfSize; ++x) {
            float val = data[y * width + x];
            if (val <= threshold) continue;

            bool isMax = true;
            for (int32_t dy = -halfSize; dy <= halfSize && isMax; ++dy) {
                for (int32_t dx = -halfSize; dx <= halfSize && isMax; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (data[(y + dy) * width + (x + dx)] >= val) {
                        isMax = false;
                    }
                }
            }

            if (isMax) {
                corners.emplace_back(x, y);
            }
        }
    }

    return corners;
}

std::vector<Point2d> FilterByDistance(
    const std::vector<Point2d>& corners,
    const std::vector<double>& strengths,
    double minDistance)
{
    std::vector<Point2d> filtered;

    if (corners.empty()) {
        return filtered;
    }

    // Corners should already be sorted by strength (strongest first)
    double minDistSq = minDistance * minDistance;

    for (size_t i = 0; i < corners.size(); ++i) {
        bool tooClose = false;
        for (const auto& kept : filtered) {
            double dx = corners[i].x - kept.x;
            double dy = corners[i].y - kept.y;
            if (dx * dx + dy * dy < minDistSq) {
                tooClose = true;
                break;
            }
        }

        if (!tooClose) {
            filtered.push_back(corners[i]);
        }
    }

    return filtered;
}

} // namespace Qi::Vision::Internal
