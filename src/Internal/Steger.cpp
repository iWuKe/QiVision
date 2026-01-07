/**
 * @file Steger.cpp
 * @brief Steger subpixel edge detection implementation
 */

#include <QiVision/Internal/Steger.h>
#include <QiVision/Internal/Interpolate.h>

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

namespace Qi::Vision::Internal {

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

// Grid cell for spatial indexing
struct GridCell {
    int32_t x, y;
    bool operator==(const GridCell& other) const {
        return x == other.x && y == other.y;
    }
};

struct GridCellHash {
    size_t operator()(const GridCell& cell) const {
        return std::hash<int64_t>()(
            (static_cast<int64_t>(cell.x) << 32) | static_cast<uint32_t>(cell.y)
        );
    }
};

} // anonymous namespace

// ============================================================================
// Subpixel Refinement
// ============================================================================

Point2d RefineSubpixelSteger(const float* dxx, const float* dxy, const float* dyy,
                              int32_t width, int32_t height,
                              int32_t x, int32_t y,
                              double nx, double ny) {
    // The subpixel position is found by solving the Taylor expansion
    // of the gradient projected onto the principal direction.
    //
    // At the edge, the gradient magnitude is maximum along the principal
    // direction, which means the second derivative in that direction is zero.
    //
    // Using Taylor expansion:
    // g(t) = g(0) + g'(0)*t + 0.5*g''(0)*t^2
    //
    // For the gradient along n = (nx, ny):
    // d/dn = nx * d/dx + ny * d/dy
    //
    // The zero-crossing of the gradient is at:
    // t = -g'(0) / g''(0)
    //
    // Where:
    // g'(0) = nx * Gx + ny * Gy (gradient projected onto n)
    // g''(0) = nx^2 * Gxx + 2*nx*ny*Gxy + ny^2*Gyy (second derivative along n)

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        return Point2d(0, 0);
    }

    size_t idx = static_cast<size_t>(y * width + x);

    // Get Hessian components at current position
    double gxx = static_cast<double>(dxx[idx]);
    double gxy = static_cast<double>(dxy[idx]);
    double gyy = static_cast<double>(dyy[idx]);

    // Compute first derivatives using central differences
    // Gx = (I(x+1) - I(x-1)) / 2
    // We approximate this from the Hessian by computing the gradient
    // of the second derivative image at the edge point
    //
    // Actually, for Steger's method, we need the first derivative of
    // the image to find where it's zero along the principal direction.
    //
    // However, since we're working with Hessian images, we can use:
    // The gradient of the response along n is approximately:
    // r(t) = λ1(x + t*nx, y + t*ny)
    // r'(0) ≈ (λ1(x+nx, y+ny) - λ1(x-nx, y-ny)) / 2

    // For subpixel refinement, use the gradient of the Hessian trace
    // or use the original image gradient.
    //
    // Simplified approach: solve for zero-crossing of directional derivative
    // Using the Hessian components directly:
    // The offset t along the normal direction satisfies:
    // nx * dxx * t + ny * dxy * t = -gradient_x
    // nx * dxy * t + ny * dyy * t = -gradient_y
    //
    // This reduces to:
    // (nx^2 * gxx + 2*nx*ny*gxy + ny^2*gyy) * t = -(nx*gx + ny*gy)

    // Compute gradient from Hessian derivative differences
    double gx = 0.0, gy = 0.0;

    // Use central differences on the Hessian components to estimate gradient
    if (x > 0 && x < width - 1) {
        gx = (dxx[y * width + x + 1] - dxx[y * width + x - 1]) * 0.5 * nx +
             (dxy[y * width + x + 1] - dxy[y * width + x - 1]) * 0.5 * ny;
    }
    if (y > 0 && y < height - 1) {
        gy = (dxx[(y + 1) * width + x] - dxx[(y - 1) * width + x]) * 0.5 * nx +
             (dxy[(y + 1) * width + x] - dxy[(y - 1) * width + x]) * 0.5 * ny;
    }

    // Second derivative along the normal direction
    double d2n = nx * nx * gxx + 2.0 * nx * ny * gxy + ny * ny * gyy;

    // First derivative along the normal direction (approximate)
    double dn = nx * gx + ny * gy;

    // Solve for offset t
    double t = 0.0;
    if (std::abs(d2n) > 1e-10) {
        t = -dn / d2n;
    }

    // Clamp offset to reasonable range (within 1 pixel)
    t = std::max(-0.5, std::min(0.5, t));

    return Point2d(t * nx, t * ny);
}

void RefineAllSubpixel(std::vector<StegerPoint>& points,
                       const float* dxx, const float* dxy, const float* dyy,
                       int32_t width, int32_t height) {
    for (auto& pt : points) {
        Point2d offset = RefineSubpixelSteger(dxx, dxy, dyy, width, height,
                                               pt.pixelX, pt.pixelY,
                                               pt.nx, pt.ny);
        pt.x = pt.pixelX + offset.x;
        pt.y = pt.pixelY + offset.y;
    }
}

// ============================================================================
// Candidate Detection
// ============================================================================

std::vector<StegerPoint> DetectCandidatePoints(
    const float* lambda1, const float* lambda2,
    const float* nx, const float* ny,
    int32_t width, int32_t height,
    const StegerParams& params) {

    std::vector<StegerPoint> points;
    points.reserve(static_cast<size_t>(width * height / 10));  // Estimate

    for (int32_t y = 1; y < height - 1; ++y) {
        for (int32_t x = 1; x < width - 1; ++x) {
            size_t idx = static_cast<size_t>(y * width + x);

            double l1 = static_cast<double>(lambda1[idx]);
            double l2 = static_cast<double>(lambda2[idx]);

            // Check if this is a valid edge candidate
            if (!IsEdgeCandidate(l1, l2, params.lowThreshold, params.lineType)) {
                continue;
            }

            // Check local maximum along principal direction (NMS)
            double nxVal = static_cast<double>(nx[idx]);
            double nyVal = static_cast<double>(ny[idx]);

            // Sample response at neighboring positions along normal
            double response = std::abs(l1);

            // Forward neighbor
            int32_t fx = static_cast<int32_t>(std::round(x + nxVal));
            int32_t fy = static_cast<int32_t>(std::round(y + nyVal));
            if (fx >= 0 && fx < width && fy >= 0 && fy < height) {
                double fResponse = std::abs(lambda1[fy * width + fx]);
                if (fResponse > response) continue;  // Not a local maximum
            }

            // Backward neighbor
            int32_t bx = static_cast<int32_t>(std::round(x - nxVal));
            int32_t by = static_cast<int32_t>(std::round(y - nyVal));
            if (bx >= 0 && bx < width && by >= 0 && by < height) {
                double bResponse = std::abs(lambda1[by * width + bx]);
                if (bResponse > response) continue;  // Not a local maximum
            }

            // Create edge point
            StegerPoint pt;
            pt.x = static_cast<double>(x);
            pt.y = static_cast<double>(y);
            pt.pixelX = x;
            pt.pixelY = y;
            pt.nx = nxVal;
            pt.ny = nyVal;
            pt.tx = -nyVal;  // Tangent is perpendicular to normal
            pt.ty = nxVal;
            pt.response = response;
            pt.isRidge = (l1 < 0);

            points.push_back(pt);
        }
    }

    return points;
}

// ============================================================================
// Spatial Index
// ============================================================================

std::vector<std::vector<int32_t>> BuildSpatialIndex(
    const std::vector<StegerPoint>& points,
    int32_t width, int32_t height,
    int32_t cellSize) {

    int32_t gridWidth = (width + cellSize - 1) / cellSize;
    int32_t gridHeight = (height + cellSize - 1) / cellSize;
    size_t numCells = static_cast<size_t>(gridWidth * gridHeight);

    std::vector<std::vector<int32_t>> grid(numCells);

    for (size_t i = 0; i < points.size(); ++i) {
        int32_t cellX = static_cast<int32_t>(points[i].x) / cellSize;
        int32_t cellY = static_cast<int32_t>(points[i].y) / cellSize;

        cellX = std::max(0, std::min(cellX, gridWidth - 1));
        cellY = std::max(0, std::min(cellY, gridHeight - 1));

        size_t cellIdx = static_cast<size_t>(cellY * gridWidth + cellX);
        grid[cellIdx].push_back(static_cast<int32_t>(i));
    }

    return grid;
}

// ============================================================================
// Edge Linking
// ============================================================================

std::vector<QContour> LinkEdgePoints(const std::vector<StegerPoint>& points,
                                      double maxGap,
                                      double maxAngleDiff) {
    if (points.empty()) {
        return {};
    }

    // Build spatial index for efficient neighbor lookup
    int32_t maxX = 0, maxY = 0;
    for (const auto& pt : points) {
        maxX = std::max(maxX, static_cast<int32_t>(pt.x) + 1);
        maxY = std::max(maxY, static_cast<int32_t>(pt.y) + 1);
    }

    int32_t cellSize = static_cast<int32_t>(std::ceil(maxGap)) + 1;
    int32_t gridWidth = (maxX + cellSize - 1) / cellSize;
    int32_t gridHeight = (maxY + cellSize - 1) / cellSize;

    auto grid = BuildSpatialIndex(points, maxX, maxY, cellSize);

    // Track which points have been used
    std::vector<bool> used(points.size(), false);
    std::vector<QContour> contours;

    // Find neighbors for a given point
    auto findNeighbors = [&](int32_t pointIdx) -> std::vector<int32_t> {
        std::vector<int32_t> neighbors;
        const auto& pt = points[pointIdx];

        int32_t cellX = static_cast<int32_t>(pt.x) / cellSize;
        int32_t cellY = static_cast<int32_t>(pt.y) / cellSize;

        // Check neighboring cells
        for (int32_t dy = -1; dy <= 1; ++dy) {
            for (int32_t dx = -1; dx <= 1; ++dx) {
                int32_t nx = cellX + dx;
                int32_t ny = cellY + dy;

                if (nx < 0 || nx >= gridWidth || ny < 0 || ny >= gridHeight) {
                    continue;
                }

                size_t cellIdx = static_cast<size_t>(ny * gridWidth + nx);
                for (int32_t otherIdx : grid[cellIdx]) {
                    if (otherIdx == pointIdx || used[otherIdx]) {
                        continue;
                    }

                    // Check distance
                    double dist = PointDistance(pt, points[otherIdx]);
                    if (dist > maxGap) continue;

                    // Check tangent angle compatibility
                    double angleDiff = TangentAngleDiff(
                        pt.tx, pt.ty,
                        points[otherIdx].tx, points[otherIdx].ty
                    );
                    if (angleDiff > maxAngleDiff) continue;

                    // Check that they're the same type (ridge/valley)
                    if (pt.isRidge != points[otherIdx].isRidge) continue;

                    neighbors.push_back(otherIdx);
                }
            }
        }

        return neighbors;
    };

    // Link points into chains
    for (size_t startIdx = 0; startIdx < points.size(); ++startIdx) {
        if (used[startIdx]) continue;

        // Start a new contour from this point
        std::vector<int32_t> chain;
        chain.push_back(static_cast<int32_t>(startIdx));
        used[startIdx] = true;

        // Extend forward from the tangent direction
        int32_t currentIdx = static_cast<int32_t>(startIdx);
        while (true) {
            auto neighbors = findNeighbors(currentIdx);
            if (neighbors.empty()) break;

            // Find the neighbor that best continues in the tangent direction
            int32_t bestIdx = -1;
            double bestScore = -1.0;

            const auto& current = points[currentIdx];
            for (int32_t nIdx : neighbors) {
                const auto& neighbor = points[nIdx];

                // Compute direction from current to neighbor
                double dx = neighbor.x - current.x;
                double dy = neighbor.y - current.y;
                double dist = std::sqrt(dx * dx + dy * dy);
                if (dist < 1e-10) continue;

                dx /= dist;
                dy /= dist;

                // Score is dot product with tangent (prefer points in tangent direction)
                double score = std::abs(dx * current.tx + dy * current.ty);
                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = nIdx;
                }
            }

            if (bestIdx < 0 || bestScore < 0.5) break;  // No good continuation

            chain.push_back(bestIdx);
            used[bestIdx] = true;
            currentIdx = bestIdx;
        }

        // Extend backward (reverse tangent direction)
        currentIdx = static_cast<int32_t>(startIdx);
        std::vector<int32_t> backChain;
        while (true) {
            auto neighbors = findNeighbors(currentIdx);
            if (neighbors.empty()) break;

            int32_t bestIdx = -1;
            double bestScore = -1.0;

            const auto& current = points[currentIdx];
            for (int32_t nIdx : neighbors) {
                const auto& neighbor = points[nIdx];

                double dx = neighbor.x - current.x;
                double dy = neighbor.y - current.y;
                double dist = std::sqrt(dx * dx + dy * dy);
                if (dist < 1e-10) continue;

                dx /= dist;
                dy /= dist;

                // For backward, we want points in negative tangent direction
                double score = std::abs(-dx * current.tx - dy * current.ty);
                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = nIdx;
                }
            }

            if (bestIdx < 0 || bestScore < 0.5) break;

            backChain.push_back(bestIdx);
            used[bestIdx] = true;
            currentIdx = bestIdx;
        }

        // Combine backward and forward chains
        std::reverse(backChain.begin(), backChain.end());
        backChain.insert(backChain.end(), chain.begin(), chain.end());

        // Only keep if long enough
        if (backChain.size() >= 2) {
            std::vector<StegerPoint> contourPoints;
            contourPoints.reserve(backChain.size());
            for (int32_t idx : backChain) {
                contourPoints.push_back(points[idx]);
            }
            contours.push_back(CreateContour(contourPoints));
        }
    }

    return contours;
}

// ============================================================================
// Filtering
// ============================================================================

std::vector<StegerPoint> FilterByHysteresis(
    const std::vector<StegerPoint>& points,
    double lowThreshold,
    double highThreshold) {

    // Simple filtering: keep all points above low threshold
    // Full hysteresis would require connected component analysis
    // TODO: Implement true hysteresis with connected component tracking
    (void)highThreshold;  // Reserved for future hysteresis implementation

    std::vector<StegerPoint> filtered;
    filtered.reserve(points.size());

    for (const auto& pt : points) {
        if (pt.response >= lowThreshold) {
            filtered.push_back(pt);
        }
    }

    return filtered;
}

std::vector<QContour> FilterByLength(const std::vector<QContour>& contours,
                                      double minLength) {
    std::vector<QContour> filtered;
    filtered.reserve(contours.size());

    for (const auto& contour : contours) {
        if (contour.Length() >= minLength) {
            filtered.push_back(contour);
        }
    }

    return filtered;
}

// ============================================================================
// Non-Maximum Suppression
// ============================================================================

std::vector<StegerPoint> NonMaxSuppressionSteger(
    const std::vector<StegerPoint>& points,
    const float* lambda1,
    int32_t width, int32_t height) {

    std::vector<StegerPoint> filtered;
    filtered.reserve(points.size());

    for (const auto& pt : points) {
        if (IsRidgeMaximum(lambda1, nullptr, nullptr, width, height,
                           pt.pixelX, pt.pixelY)) {
            filtered.push_back(pt);
        }
    }

    return filtered;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::vector<ContourPoint> ToContourPoints(const std::vector<StegerPoint>& points) {
    std::vector<ContourPoint> result;
    result.reserve(points.size());

    for (const auto& pt : points) {
        // Direction is the tangent angle
        double direction = std::atan2(pt.ty, pt.tx);

        result.emplace_back(pt.x, pt.y, pt.response, direction, 0.0);
    }

    return result;
}

QContour CreateContour(const std::vector<StegerPoint>& points) {
    auto contourPoints = ToContourPoints(points);
    QContour contour;

    for (const auto& pt : contourPoints) {
        contour.AddPoint(pt);
    }

    return contour;
}

// ============================================================================
// Main Detection Functions
// ============================================================================

std::vector<QContour> DetectStegerEdges(const QImage& image,
                                         const StegerParams& params) {
    auto result = DetectStegerEdgesFull(image, params);
    return std::move(result.contours);
}

StegerResult DetectStegerEdgesFull(const QImage& image,
                                    const StegerParams& params) {
    StegerResult result;

    if (!image.IsValid()) {
        return result;
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    // Step 1: Compute Hessian components
    std::vector<float> dxx, dxy, dyy;

    switch (image.Type()) {
        case PixelType::UInt8: {
            const uint8_t* data = static_cast<const uint8_t*>(image.Data());
            ComputeHessianImage(data, width, height, params.sigma, dxx, dxy, dyy);
            break;
        }
        case PixelType::UInt16: {
            const uint16_t* data = static_cast<const uint16_t*>(image.Data());
            ComputeHessianImage(data, width, height, params.sigma, dxx, dxy, dyy);
            break;
        }
        case PixelType::Float32: {
            const float* data = static_cast<const float*>(image.Data());
            ComputeHessianImage(data, width, height, params.sigma, dxx, dxy, dyy);
            break;
        }
        default:
            return result;
    }

    // Step 2: Compute eigenvalues and eigenvectors
    std::vector<float> lambda1, lambda2, nx, ny;
    ComputeEigenvalueImages(dxx.data(), dxy.data(), dyy.data(),
                             width, height, lambda1, lambda2);
    ComputeEigenvectorImages(dxx.data(), dxy.data(), dyy.data(),
                              width, height, nx, ny);

    // Step 3: Detect candidate points with NMS
    result.points = DetectCandidatePoints(
        lambda1.data(), lambda2.data(),
        nx.data(), ny.data(),
        width, height, params);

    // Step 4: Subpixel refinement
    if (params.subPixelRefinement) {
        RefineAllSubpixel(result.points, dxx.data(), dxy.data(), dyy.data(),
                          width, height);
    }

    // Step 5: Filter by threshold
    result.points = FilterByHysteresis(result.points,
                                        params.lowThreshold,
                                        params.highThreshold);

    // Count ridge/valley points
    for (const auto& pt : result.points) {
        if (pt.isRidge) {
            ++result.numRidgePoints;
        } else {
            ++result.numValleyPoints;
        }
    }

    // Step 6: Link edge points
    result.contours = LinkEdgePoints(result.points,
                                      params.maxGap,
                                      params.maxAngleDiff);

    // Step 7: Filter by length
    result.contours = FilterByLength(result.contours, params.minLength);

    return result;
}

} // namespace Qi::Vision::Internal
