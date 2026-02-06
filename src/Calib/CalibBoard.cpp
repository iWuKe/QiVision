/**
 * @file CalibBoard.cpp
 * @brief Calibration board detection implementation
 *
 * Algorithm overview:
 * 1. Preprocess image (normalize, blur)
 * 2. Apply adaptive threshold to binarize
 * 3. Find contours and filter for quadrilaterals
 * 4. Extract corner candidates from quad vertices
 * 5. Cluster nearby corners
 * 6. Organize corners into a regular grid
 * 7. Refine corners to subpixel accuracy
 *
 * Reference: OpenCV chessboard / circle grid detection
 */

#include <QiVision/Calib/CalibBoard.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>
#include <QiVision/Internal/CornerRefine.h>
#include <QiVision/Internal/Threshold.h>
#include <QiVision/Internal/Histogram.h>
#include <QiVision/Internal/ConnectedComponent.h>
#include <QiVision/Display/Display.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <queue>

namespace Qi::Vision::Calib {

namespace {

// =============================================================================
// Constants
// =============================================================================

/// Minimum contour area ratio to image size
constexpr double MIN_CONTOUR_AREA_RATIO = 0.0001;

/// Maximum contour area ratio to image size
constexpr double MAX_CONTOUR_AREA_RATIO = 0.25;

/// Corner clustering distance
constexpr double CORNER_CLUSTER_DISTANCE = 5.0;

/// Minimum number of quads for detection
constexpr int MIN_QUADS = 1;

/// Angle tolerance for quad corners (degrees)
constexpr double QUAD_ANGLE_TOLERANCE = 30.0;

/// Circle grid detection thresholds
constexpr double MIN_CIRCLE_FILL_RATIO = 0.55;
constexpr double MAX_CIRCLE_FILL_RATIO = 0.95;
constexpr double MAX_ASPECT_RATIO = 1.6;

// =============================================================================
// Helper Structures
// =============================================================================

struct Quad {
    Point2d corners[4];     // Vertices in order
    Point2d center;         // Centroid
    double area;            // Area
    double perimeter;       // Perimeter

    Quad() : area(0), perimeter(0) {
        center = Point2d(0, 0);
        for (int i = 0; i < 4; ++i) {
            corners[i] = Point2d(0, 0);
        }
    }
};

struct CornerCandidate {
    Point2d position;
    int count;              // Number of quads sharing this corner
    double response;        // Corner response strength

    CornerCandidate() : count(0), response(0) {}
    CornerCandidate(const Point2d& pos) : position(pos), count(1), response(0) {}
};

struct CircleCandidate {
    Point2d center;
    double area = 0.0;
    Rect2i box;
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Normalize image to use full dynamic range
 */
void NormalizeImage(const QImage& src, QImage& dst) {
    if (!Validate::RequireImageU8(src, "NormalizeImage")) {
        dst = QImage();
        return;
    }

    dst = QImage(src.Width(), src.Height(), src.Type(), src.GetChannelType());

    const int32_t width = src.Width();
    const int32_t height = src.Height();
    const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    // Find min/max
    uint8_t minVal = 255, maxVal = 0;
    for (int32_t i = 0; i < width * height; ++i) {
        if (srcData[i] < minVal) minVal = srcData[i];
        if (srcData[i] > maxVal) maxVal = srcData[i];
    }

    if (maxVal <= minVal) {
        dst = src.Clone();
        return;
    }

    double scale = 255.0 / (maxVal - minVal);
    for (int32_t i = 0; i < width * height; ++i) {
        dstData[i] = static_cast<uint8_t>((srcData[i] - minVal) * scale);
    }
}

/**
 * @brief Simple box blur
 */
void BoxBlur(const QImage& src, QImage& dst, int32_t kernelSize) {
    if (!Validate::RequireImageU8(src, "BoxBlur") || kernelSize < 3) {
        dst = src.Clone();
        return;
    }

    const int32_t width = src.Width();
    const int32_t height = src.Height();
    dst = QImage(width, height, src.Type(), src.GetChannelType());

    const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    int32_t half = kernelSize / 2;
    double invArea = 1.0 / (kernelSize * kernelSize);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            double sum = 0;
            for (int32_t ky = -half; ky <= half; ++ky) {
                for (int32_t kx = -half; kx <= half; ++kx) {
                    int32_t px = std::clamp(x + kx, 0, width - 1);
                    int32_t py = std::clamp(y + ky, 0, height - 1);
                    sum += srcData[py * width + px];
                }
            }
            dstData[y * width + x] = static_cast<uint8_t>(sum * invArea);
        }
    }
}

/**
 * @brief Compute polygon area (shoelace formula)
 */
double PolygonArea(const std::vector<Point2d>& pts) {
    if (pts.size() < 3) return 0;
    double area = 0;
    size_t n = pts.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        area += pts[i].x * pts[j].y;
        area -= pts[j].x * pts[i].y;
    }
    return std::abs(area) * 0.5;
}

/**
 * @brief Compute polygon perimeter
 */
double PolygonPerimeter(const std::vector<Point2d>& pts) {
    if (pts.size() < 2) return 0;
    double peri = 0;
    size_t n = pts.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        peri += pts[i].DistanceTo(pts[j]);
    }
    return peri;
}

/**
 * @brief Check if 4 points form a valid quadrilateral
 */
bool IsValidQuad(const Point2d corners[4]) {
    // Check that corners are in roughly convex order
    // and form a reasonable quadrilateral

    // Compute cross products to check convexity
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        int k = (i + 2) % 4;

        Point2d v1 = corners[j] - corners[i];
        Point2d v2 = corners[k] - corners[j];

        double cross = v1.Cross(v2);
        if (cross < 0) return false;  // Concave or self-intersecting
    }

    // Check angles are within tolerance
    for (int i = 0; i < 4; ++i) {
        int prev = (i + 3) % 4;
        int next = (i + 1) % 4;

        Point2d v1 = corners[prev] - corners[i];
        Point2d v2 = corners[next] - corners[i];

        double dot = v1.Dot(v2);
        double len1 = v1.Norm();
        double len2 = v2.Norm();

        if (len1 < 1e-6 || len2 < 1e-6) return false;

        double cosAngle = dot / (len1 * len2);
        double angleDeg = std::acos(std::clamp(cosAngle, -1.0, 1.0)) * 180.0 / M_PI;

        // Corners should be roughly 90 degrees
        if (angleDeg < 90 - QUAD_ANGLE_TOLERANCE || angleDeg > 90 + QUAD_ANGLE_TOLERANCE) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Douglas-Peucker line simplification
 */
void SimplifyContour(const std::vector<Point2d>& input, std::vector<Point2d>& output,
                     double epsilon) {
    if (input.size() < 3) {
        output = input;
        return;
    }

    std::vector<bool> keep(input.size(), false);
    keep[0] = keep[input.size() - 1] = true;

    std::function<void(size_t, size_t)> rdp = [&](size_t start, size_t end) {
        if (end <= start + 1) return;

        double maxDist = 0;
        size_t maxIdx = start;

        // Line from input[start] to input[end]
        Point2d a = input[start];
        Point2d b = input[end];
        Point2d ab = b - a;
        double abLen = ab.Norm();

        if (abLen < 1e-10) return;

        for (size_t i = start + 1; i < end; ++i) {
            Point2d ap = input[i] - a;
            double dist = std::abs(ab.Cross(ap)) / abLen;
            if (dist > maxDist) {
                maxDist = dist;
                maxIdx = i;
            }
        }

        if (maxDist > epsilon) {
            keep[maxIdx] = true;
            rdp(start, maxIdx);
            rdp(maxIdx, end);
        }
    };

    rdp(0, input.size() - 1);

    output.clear();
    for (size_t i = 0; i < input.size(); ++i) {
        if (keep[i]) output.push_back(input[i]);
    }
}

/**
 * @brief Extract contour points from region boundary
 */
std::vector<Point2d> ExtractContourPoints(const QRegion& region) {
    std::vector<Point2d> points;

    if (region.Empty()) return points;

    // Get all runs and extract edge points
    const auto& runs = region.Runs();

    for (const auto& run : runs) {
        // Add edge points (start and end of each run)
        points.emplace_back(run.colBegin, run.row);
        if (run.colEnd > run.colBegin + 1) {
            points.emplace_back(run.colEnd - 1, run.row);
        }
    }

    return points;
}

/**
 * @brief Try to fit region to quad
 */
bool FitQuad(const QRegion& region, Quad& quad, double imageArea) {
    Rect2i bbox = region.BoundingBox();
    double area = static_cast<double>(region.Area());

    // Size checks
    if (area < MIN_CONTOUR_AREA_RATIO * imageArea ||
        area > MAX_CONTOUR_AREA_RATIO * imageArea) {
        return false;
    }

    // Aspect ratio check - shouldn't be too elongated
    double aspectRatio = static_cast<double>(bbox.width) / std::max(1, bbox.height);
    if (aspectRatio > 5.0 || aspectRatio < 0.2) {
        return false;
    }

    // Extract contour and simplify to quad
    std::vector<Point2d> contour = ExtractContourPoints(region);
    if (contour.size() < 4) return false;

    // Approximate to polygon with 4 vertices
    double epsilon = PolygonPerimeter(contour) * 0.02;  // 2% of perimeter
    std::vector<Point2d> approx;
    SimplifyContour(contour, approx, epsilon);

    // Keep simplifying until we get close to 4 points
    while (approx.size() > 6 && epsilon < PolygonPerimeter(contour) * 0.1) {
        epsilon *= 1.5;
        SimplifyContour(contour, approx, epsilon);
    }

    if (approx.size() != 4) {
        // Try to find 4 most prominent corners
        if (approx.size() > 4) {
            // Take corners with largest turning angles
            std::vector<std::pair<double, size_t>> angles;
            for (size_t i = 0; i < approx.size(); ++i) {
                size_t prev = (i + approx.size() - 1) % approx.size();
                size_t next = (i + 1) % approx.size();

                Point2d v1 = approx[prev] - approx[i];
                Point2d v2 = approx[next] - approx[i];

                double angle = std::abs(std::atan2(v1.Cross(v2), v1.Dot(v2)));
                angles.emplace_back(angle, i);
            }

            std::sort(angles.begin(), angles.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

            std::vector<size_t> selected;
            for (size_t i = 0; i < 4 && i < angles.size(); ++i) {
                selected.push_back(angles[i].second);
            }
            std::sort(selected.begin(), selected.end());

            if (selected.size() == 4) {
                for (size_t i = 0; i < 4; ++i) {
                    quad.corners[i] = approx[selected[i]];
                }
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else {
        for (size_t i = 0; i < 4; ++i) {
            quad.corners[i] = approx[i];
        }
    }

    // Validate quad shape
    if (!IsValidQuad(quad.corners)) {
        return false;
    }

    // Compute properties
    quad.center = Point2d(0, 0);
    for (int i = 0; i < 4; ++i) {
        quad.center.x += quad.corners[i].x;
        quad.center.y += quad.corners[i].y;
    }
    quad.center.x *= 0.25;
    quad.center.y *= 0.25;

    std::vector<Point2d> qpts(quad.corners, quad.corners + 4);
    quad.area = PolygonArea(qpts);
    quad.perimeter = PolygonPerimeter(qpts);

    return true;
}

/**
 * @brief Cluster nearby corner candidates
 */
std::vector<Point2d> ClusterCorners(const std::vector<Point2d>& candidates,
                                     double clusterDist) {
    if (candidates.empty()) return {};

    std::vector<bool> processed(candidates.size(), false);
    std::vector<Point2d> clusters;

    double distSq = clusterDist * clusterDist;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (processed[i]) continue;

        // Find all points in cluster
        std::vector<size_t> clusterIndices;
        std::queue<size_t> queue;
        queue.push(i);
        processed[i] = true;

        while (!queue.empty()) {
            size_t curr = queue.front();
            queue.pop();
            clusterIndices.push_back(curr);

            for (size_t j = 0; j < candidates.size(); ++j) {
                if (processed[j]) continue;

                double dx = candidates[curr].x - candidates[j].x;
                double dy = candidates[curr].y - candidates[j].y;
                if (dx * dx + dy * dy < distSq) {
                    processed[j] = true;
                    queue.push(j);
                }
            }
        }

        // Compute cluster centroid
        double cx = 0, cy = 0;
        for (size_t idx : clusterIndices) {
            cx += candidates[idx].x;
            cy += candidates[idx].y;
        }
        cx /= clusterIndices.size();
        cy /= clusterIndices.size();

        clusters.emplace_back(cx, cy);
    }

    return clusters;
}

/**
 * @brief Try to organize corners into a regular grid
 */
bool OrganizeGrid(std::vector<Point2d>& corners, int32_t patternCols, int32_t patternRows) {
    int32_t expectedCount = patternCols * patternRows;
    if (static_cast<int32_t>(corners.size()) != expectedCount) {
        return false;
    }

    // Find bounding box
    double minX = corners[0].x, maxX = corners[0].x;
    double minY = corners[0].y, maxY = corners[0].y;
    for (const auto& c : corners) {
        minX = std::min(minX, c.x);
        maxX = std::max(maxX, c.x);
        minY = std::min(minY, c.y);
        maxY = std::max(maxY, c.y);
    }

    // Estimate grid spacing
    double spacingX = (maxX - minX) / (patternCols - 1);
    double spacingY = (maxY - minY) / (patternRows - 1);

    if (spacingX < 5 || spacingY < 5) {
        return false;  // Spacing too small
    }

    // Unused variable removed - grid organization doesn't need explicit top-left detection
    // since we use greedy assignment based on expected positions
    (void)0;

    // Organize into grid starting from top-left
    std::vector<Point2d> organized(expectedCount);
    std::vector<bool> used(corners.size(), false);

    // Greedy assignment: for each grid position, find nearest unassigned corner
    for (int32_t row = 0; row < patternRows; ++row) {
        for (int32_t col = 0; col < patternCols; ++col) {
            // Expected position
            double expectedX = minX + col * spacingX;
            double expectedY = minY + row * spacingY;

            // Find nearest unassigned corner
            size_t bestIdx = 0;
            double bestDist = std::numeric_limits<double>::max();

            for (size_t i = 0; i < corners.size(); ++i) {
                if (used[i]) continue;

                double dx = corners[i].x - expectedX;
                double dy = corners[i].y - expectedY;
                double dist = dx * dx + dy * dy;

                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = i;
                }
            }

            // Check if close enough
            double tolerance = std::max(spacingX, spacingY) * 0.5;
            if (bestDist > tolerance * tolerance) {
                return false;  // No corner close enough
            }

            used[bestIdx] = true;
            organized[row * patternCols + col] = corners[bestIdx];
        }
    }

    corners = organized;
    return true;
}

std::vector<CircleCandidate> ExtractCircleCandidates(const QImage& binary) {
    std::vector<CircleCandidate> candidates;
    int32_t numLabels = 0;
    QImage labels = Internal::LabelConnectedComponents(binary, Connectivity::Eight, numLabels);
    if (numLabels <= 0) {
        return candidates;
    }

    auto stats = Internal::GetComponentStats(labels, numLabels);
    if (stats.empty()) {
        return candidates;
    }

    std::vector<double> areas;
    areas.reserve(stats.size());
    for (const auto& s : stats) {
        if (s.area <= 0) continue;
        areas.push_back(static_cast<double>(s.area));
    }
    if (areas.empty()) {
        return candidates;
    }

    std::nth_element(areas.begin(), areas.begin() + areas.size() / 2, areas.end());
    double medianArea = areas[areas.size() / 2];
    double minArea = medianArea * 0.3;
    double maxArea = medianArea * 3.0;

    for (const auto& s : stats) {
        double area = static_cast<double>(s.area);
        if (area < minArea || area > maxArea) continue;

        int32_t w = s.boundingBox.width;
        int32_t h = s.boundingBox.height;
        if (w <= 1 || h <= 1) continue;

        double aspect = (w > h) ? static_cast<double>(w) / h : static_cast<double>(h) / w;
        if (aspect > MAX_ASPECT_RATIO) continue;

        double fill = area / (static_cast<double>(w) * h);
        if (fill < MIN_CIRCLE_FILL_RATIO || fill > MAX_CIRCLE_FILL_RATIO) continue;

        CircleCandidate c;
        c.center = Point2d(s.centroidX, s.centroidY);
        c.area = area;
        c.box = s.boundingBox;
        candidates.push_back(c);
    }

    return candidates;
}

bool ComputePCAAxes(const std::vector<Point2d>& points, Point2d& axisU, Point2d& axisV) {
    if (points.size() < 2) {
        return false;
    }
    double meanX = 0.0, meanY = 0.0;
    for (const auto& p : points) {
        meanX += p.x;
        meanY += p.y;
    }
    meanX /= points.size();
    meanY /= points.size();

    double xx = 0.0, xy = 0.0, yy = 0.0;
    for (const auto& p : points) {
        double dx = p.x - meanX;
        double dy = p.y - meanY;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }

    double trace = xx + yy;
    double det = xx * yy - xy * xy;
    double disc = std::sqrt(std::max(0.0, trace * trace * 0.25 - det));
    double lambda1 = trace * 0.5 + disc;

    Point2d u;
    if (std::abs(xy) > 1e-12) {
        u.x = lambda1 - yy;
        u.y = xy;
    } else {
        u.x = (xx >= yy) ? 1.0 : 0.0;
        u.y = (xx >= yy) ? 0.0 : 1.0;
    }
    double norm = std::sqrt(u.x * u.x + u.y * u.y);
    if (norm < 1e-12) {
        return false;
    }
    u.x /= norm;
    u.y /= norm;

    Point2d v{-u.y, u.x};
    axisU = u;
    axisV = v;
    return true;
}

} // anonymous namespace

// =============================================================================
// Public API Implementation
// =============================================================================

CornerGrid FindChessboardCorners(
    const QImage& image,
    int32_t patternCols,
    int32_t patternRows,
    ChessboardFlags flags)
{
    CornerGrid result;
    result.cols = patternCols;
    result.rows = patternRows;
    result.found = false;

    if (!Validate::RequireImageValid(image, "FindChessboardCorners")) {
        return result;
    }

    if (patternCols < 2 || patternRows < 2) {
        throw InvalidArgumentException("FindChessboardCorners: pattern size must be >= 2x2");
    }

    // Ensure grayscale
    QImage gray;
    if (image.Channels() == 1) {
        gray = image;
    } else {
        gray = image.ToGray();
    }

    if (gray.Type() != PixelType::UInt8) {
        gray = gray.ConvertTo(PixelType::UInt8);
    }

    const int32_t width = gray.Width();
    const int32_t height = gray.Height();
    const double imageArea = static_cast<double>(width * height);

    // Preprocess
    QImage processed = gray;

    if (flags & ChessboardFlags::NormalizeImage) {
        NormalizeImage(gray, processed);
    }

    // Light blur to reduce noise
    QImage blurred;
    BoxBlur(processed, blurred, 5);

    // Apply adaptive threshold to get binary image
    Internal::AdaptiveThresholdParams params;
    params.method = (flags & ChessboardFlags::AdaptiveThresh)
                    ? Internal::AdaptiveMethod::Mean
                    : Internal::AdaptiveMethod::Mean;
    params.blockSize = std::max(11, std::min(width, height) / 20 | 1);  // Odd number
    params.C = 10.0;
    params.maxValue = 255.0;

    QImage binary;
    Internal::ThresholdAdaptive(blurred, binary, params);

    // Find connected components in both binary and its inverse
    // (we need both black and white squares)
    std::vector<QRegion> components;

    int32_t numLabels;
    QImage labels = Internal::LabelConnectedComponents(binary, Connectivity::Eight, numLabels);

    // Get components from binary image by extracting each labeled region
    for (int32_t i = 1; i <= numLabels; ++i) {
        // Create region from label i
        QRegion region;
        const int32_t w = labels.Width();
        const int32_t h = labels.Height();
        const uint8_t* labelData = static_cast<const uint8_t*>(labels.Data());

        std::vector<QRegion::Run> runs;
        for (int32_t y = 0; y < h; ++y) {
            int32_t start = -1;
            for (int32_t x = 0; x < w; ++x) {
                bool inLabel = (labelData[y * w + x] == i);
                if (inLabel && start < 0) {
                    start = x;
                } else if (!inLabel && start >= 0) {
                    runs.emplace_back(y, start, x);
                    start = -1;
                }
            }
            if (start >= 0) {
                runs.emplace_back(y, start, w);
            }
        }

        if (!runs.empty()) {
            region = QRegion(runs);
            components.push_back(region);
        }
    }

    // Also try inverted binary
    QImage invBinary;
    Internal::BinaryInvert(binary, invBinary);
    QImage invLabels = Internal::LabelConnectedComponents(invBinary, Connectivity::Eight, numLabels);

    // Get components from inverted binary
    for (int32_t i = 1; i <= numLabels; ++i) {
        QRegion region;
        const int32_t w = invLabels.Width();
        const int32_t h = invLabels.Height();
        const uint8_t* labelData = static_cast<const uint8_t*>(invLabels.Data());

        std::vector<QRegion::Run> runs;
        for (int32_t y = 0; y < h; ++y) {
            int32_t start = -1;
            for (int32_t x = 0; x < w; ++x) {
                bool inLabel = (labelData[y * w + x] == i);
                if (inLabel && start < 0) {
                    start = x;
                } else if (!inLabel && start >= 0) {
                    runs.emplace_back(y, start, x);
                    start = -1;
                }
            }
            if (start >= 0) {
                runs.emplace_back(y, start, w);
            }
        }

        if (!runs.empty()) {
            region = QRegion(runs);
            components.push_back(region);
        }
    }

    // Find quads from components
    std::vector<Quad> quads;
    for (const auto& comp : components) {
        Quad quad;
        if (FitQuad(comp, quad, imageArea)) {
            quads.push_back(quad);
        }
    }

    if (quads.size() < static_cast<size_t>(MIN_QUADS)) {
        return result;
    }

    // Collect all corner candidates from quads
    std::vector<Point2d> candidates;
    for (const auto& quad : quads) {
        for (int i = 0; i < 4; ++i) {
            candidates.push_back(quad.corners[i]);
        }
    }

    // Cluster nearby corners
    std::vector<Point2d> corners = ClusterCorners(candidates, CORNER_CLUSTER_DISTANCE);

    // Check if we have enough corners
    int32_t expectedCount = patternCols * patternRows;
    if (static_cast<int32_t>(corners.size()) < expectedCount) {
        // Try to detect corners directly if not enough from quads
        std::vector<Point2d> detected = Internal::DetectHarrisCorners(
            gray, expectedCount * 2, 0.01, 10.0, 3, 0.04);

        // Merge with existing corners
        for (const auto& d : detected) {
            bool exists = false;
            for (const auto& c : corners) {
                if (c.DistanceTo(d) < CORNER_CLUSTER_DISTANCE) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                corners.push_back(d);
            }
        }

        corners = ClusterCorners(corners, CORNER_CLUSTER_DISTANCE);
    }

    // Try to organize into grid
    if (!OrganizeGrid(corners, patternCols, patternRows)) {
        return result;
    }

    // Refine corners to subpixel
    Internal::RefineCorners(gray, corners, 5, 30, 0.001);

    result.corners = corners;
    result.found = true;
    return result;
}

CircleGrid FindCircleGrid(
    const QImage& image,
    int32_t patternCols,
    int32_t patternRows,
    CircleGridType type)
{
    CircleGrid grid;
    grid.rows = patternRows;
    grid.cols = patternCols;
    grid.found = false;

    if (!Validate::RequireImageU8Gray(image, "FindCircleGrid")) {
        return grid;
    }
    if (patternCols <= 0 || patternRows <= 0) {
        throw InvalidArgumentException("FindCircleGrid: invalid pattern size");
    }

    QImage bin;
    Internal::ThresholdOtsu(image, bin, 255.0, nullptr);

    auto candidates = ExtractCircleCandidates(bin);
    if (candidates.size() < static_cast<size_t>(patternCols * patternRows)) {
        // Try inverted binary if dark circles
        QImage binInv = bin.Clone();
        uint8_t* data = static_cast<uint8_t*>(binInv.Data());
        const int32_t total = binInv.Width() * binInv.Height();
        for (int32_t i = 0; i < total; ++i) {
            data[i] = static_cast<uint8_t>(255 - data[i]);
        }
        candidates = ExtractCircleCandidates(binInv);
    }

    if (candidates.size() < static_cast<size_t>(patternCols * patternRows)) {
        return grid;
    }

    std::vector<Point2d> centers;
    centers.reserve(candidates.size());
    for (const auto& c : candidates) {
        centers.push_back(c.center);
    }

    Point2d axisU, axisV;
    if (!ComputePCAAxes(centers, axisU, axisV)) {
        return grid;
    }

    // Project points to PCA axes
    struct ProjPoint {
        Point2d p;
        double u;
        double v;
    };
    std::vector<ProjPoint> proj;
    proj.reserve(centers.size());

    double meanX = 0.0, meanY = 0.0;
    for (const auto& p : centers) {
        meanX += p.x;
        meanY += p.y;
    }
    meanX /= centers.size();
    meanY /= centers.size();

    for (const auto& p : centers) {
        double dx = p.x - meanX;
        double dy = p.y - meanY;
        ProjPoint pp;
        pp.p = p;
        pp.u = dx * axisU.x + dy * axisU.y;
        pp.v = dx * axisV.x + dy * axisV.y;
        proj.push_back(pp);
    }

    // Estimate row spacing from v diffs
    std::sort(proj.begin(), proj.end(), [](const ProjPoint& a, const ProjPoint& b) { return a.v < b.v; });
    std::vector<double> vDiffs;
    for (size_t i = 1; i < proj.size(); ++i) {
        double d = proj[i].v - proj[i - 1].v;
        if (d > 1e-3) vDiffs.push_back(d);
    }
    if (vDiffs.empty()) {
        return grid;
    }
    std::nth_element(vDiffs.begin(), vDiffs.begin() + vDiffs.size() / 2, vDiffs.end());
    double rowSpacing = vDiffs[vDiffs.size() / 2];

    struct RowGroup {
        double meanV = 0.0;
        std::vector<ProjPoint> points;
    };
    std::vector<RowGroup> rows;
    for (const auto& p : proj) {
        bool assigned = false;
        for (auto& row : rows) {
            if (std::abs(p.v - row.meanV) <= rowSpacing * 0.5) {
                row.points.push_back(p);
                row.meanV = 0.0;
                for (const auto& rp : row.points) row.meanV += rp.v;
                row.meanV /= row.points.size();
                assigned = true;
                break;
            }
        }
        if (!assigned) {
            RowGroup row;
            row.meanV = p.v;
            row.points.push_back(p);
            rows.push_back(row);
        }
    }

    if (static_cast<int32_t>(rows.size()) != patternRows) {
        return grid;
    }

    // Sort rows by meanV
    std::sort(rows.begin(), rows.end(), [](const RowGroup& a, const RowGroup& b) { return a.meanV < b.meanV; });

    std::vector<Point2d> ordered;
    ordered.reserve(patternCols * patternRows);
    for (auto& row : rows) {
        if (static_cast<int32_t>(row.points.size()) != patternCols) {
            return grid;
        }
        std::sort(row.points.begin(), row.points.end(),
                  [](const ProjPoint& a, const ProjPoint& b) { return a.u < b.u; });
        for (const auto& p : row.points) {
            ordered.push_back(p.p);
        }
    }

    // For asymmetric grid, ordering is still row-major; geometry differs in object points
    (void)type;

    grid.centers = ordered;
    grid.found = true;
    return grid;
}

void CornerSubPix(
    const QImage& image,
    std::vector<Point2d>& corners,
    int32_t winSize,
    int32_t maxIterations,
    double epsilon)
{
    if (image.Empty() || corners.empty()) {
        return;
    }
    if (!image.IsValid()) {
        throw InvalidArgumentException("CornerSubPix: invalid image");
    }
    if (winSize <= 0) {
        throw InvalidArgumentException("CornerSubPix: winSize must be > 0");
    }
    if (maxIterations <= 0) {
        throw InvalidArgumentException("CornerSubPix: maxIterations must be > 0");
    }
    if (epsilon <= 0.0) {
        throw InvalidArgumentException("CornerSubPix: epsilon must be > 0");
    }
    for (const auto& p : corners) {
        if (!p.IsValid()) {
            throw InvalidArgumentException("CornerSubPix: invalid corner");
        }
    }
    Internal::RefineCorners(image, corners, winSize, maxIterations, epsilon);
}

std::vector<Point3d> GenerateChessboardPoints(
    int32_t patternCols,
    int32_t patternRows,
    double squareSize)
{
    if (patternCols <= 0 || patternRows <= 0) {
        throw InvalidArgumentException("GenerateChessboardPoints: pattern size must be > 0");
    }
    if (squareSize <= 0.0) {
        throw InvalidArgumentException("GenerateChessboardPoints: squareSize must be > 0");
    }
    std::vector<Point3d> points;
    points.reserve(patternCols * patternRows);

    for (int32_t row = 0; row < patternRows; ++row) {
        for (int32_t col = 0; col < patternCols; ++col) {
            points.emplace_back(col * squareSize, row * squareSize, 0.0);
        }
    }

    return points;
}

std::vector<Point3d> GenerateCircleGridPoints(
    int32_t patternCols,
    int32_t patternRows,
    double spacing,
    CircleGridType type)
{
    std::vector<Point3d> points;
    points.reserve(static_cast<size_t>(patternCols * patternRows));

    for (int32_t r = 0; r < patternRows; ++r) {
        for (int32_t c = 0; c < patternCols; ++c) {
            double x = static_cast<double>(c) * spacing;
            double y = static_cast<double>(r) * spacing;
            if (type == CircleGridType::Asymmetric) {
                x = static_cast<double>(2 * c + (r % 2)) * spacing;
            }
            points.emplace_back(x, y, 0.0);
        }
    }
    return points;
}

void DrawChessboardCorners(
    QImage& image,
    const CornerGrid& grid,
    bool drawOrder)
{
    if (!grid.IsValid() || image.Empty()) {
        return;
    }
    if (!image.IsValid()) {
        throw InvalidArgumentException("DrawChessboardCorners: invalid image");
    }
    for (const auto& p : grid.corners) {
        if (!p.IsValid()) {
            throw InvalidArgumentException("DrawChessboardCorners: invalid corner");
        }
    }

    // Ensure image is RGB for color drawing
    QImage colorImg;
    if (image.Channels() == 1) {
        // Convert to RGB
        const int32_t w = image.Width();
        const int32_t h = image.Height();
        colorImg = QImage(w, h, PixelType::UInt8, ChannelType::RGB);

        const uint8_t* src = static_cast<const uint8_t*>(image.Data());
        uint8_t* dst = static_cast<uint8_t*>(colorImg.Data());

        for (int32_t i = 0; i < w * h; ++i) {
            dst[i * 3 + 0] = src[i];
            dst[i * 3 + 1] = src[i];
            dst[i * 3 + 2] = src[i];
        }
    } else {
        colorImg = image.Clone();
    }

    // Draw connecting lines if requested
    if (drawOrder && grid.corners.size() > 1) {
        for (size_t row = 0; row < static_cast<size_t>(grid.rows); ++row) {
            for (size_t col = 0; col < static_cast<size_t>(grid.cols); ++col) {
                size_t idx = row * grid.cols + col;
                const Point2d& p = grid.corners[idx];

                // Draw line to right neighbor
                if (col + 1 < static_cast<size_t>(grid.cols)) {
                    const Point2d& q = grid.corners[idx + 1];
                    DispLine(colorImg, p.x, p.y, q.x, q.y, Scalar::Green(), 1);
                }

                // Draw line to bottom neighbor
                if (row + 1 < static_cast<size_t>(grid.rows)) {
                    const Point2d& q = grid.corners[idx + grid.cols];
                    DispLine(colorImg, p.x, p.y, q.x, q.y, Scalar::Green(), 1);
                }
            }
        }
    }

    // Draw corners with color gradient based on position
    for (size_t i = 0; i < grid.corners.size(); ++i) {
        const Point2d& p = grid.corners[i];

        // Color gradient from red (start) to blue (end)
        double t = static_cast<double>(i) / std::max(1.0, static_cast<double>(grid.corners.size() - 1));
        uint8_t r = static_cast<uint8_t>(255 * (1 - t));
        uint8_t g = 0;
        uint8_t b = static_cast<uint8_t>(255 * t);

        Scalar color(r, g, b);

        // Draw cross marker
        DispCross(colorImg, p.x, p.y, 5, 0.0, color, 1);

        // Draw small circle
        DispCircle(colorImg, p.x, p.y, 3, color, 1);
    }

    image = colorImg;
}

void DrawCircleGrid(
    QImage& image,
    const CircleGrid& grid,
    bool drawOrder)
{
    if (!grid.IsValid() || image.Empty()) {
        return;
    }

    QImage colorImg = image;
    if (image.Channels() == 1) {
        colorImg = QImage(image.Width(), image.Height(), PixelType::UInt8, ChannelType::RGB);
        for (int y = 0; y < image.Height(); ++y) {
            const uint8_t* src = static_cast<const uint8_t*>(image.RowPtr(y));
            uint8_t* dst = static_cast<uint8_t*>(colorImg.RowPtr(y));
            for (int x = 0; x < image.Width(); ++x) {
                uint8_t v = src[x];
                dst[x * 3 + 0] = v;
                dst[x * 3 + 1] = v;
                dst[x * 3 + 2] = v;
            }
        }
    }

    if (drawOrder) {
        for (int32_t row = 0; row < grid.rows; ++row) {
            for (int32_t col = 0; col < grid.cols; ++col) {
                int idx = row * grid.cols + col;
                const Point2d& p = grid.centers[idx];

                if (col + 1 < grid.cols) {
                    const Point2d& q = grid.centers[idx + 1];
                    DispLine(colorImg, p.x, p.y, q.x, q.y, Scalar::Green(), 1);
                }
                if (row + 1 < grid.rows) {
                    const Point2d& q = grid.centers[idx + grid.cols];
                    DispLine(colorImg, p.x, p.y, q.x, q.y, Scalar::Green(), 1);
                }
            }
        }
    }

    for (size_t i = 0; i < grid.centers.size(); ++i) {
        const Point2d& p = grid.centers[i];
        double t = static_cast<double>(i) / std::max(1.0, static_cast<double>(grid.centers.size() - 1));
        uint8_t r = static_cast<uint8_t>(255 * (1 - t));
        uint8_t g = 0;
        uint8_t b = static_cast<uint8_t>(255 * t);
        Scalar color(r, g, b);
        DispCross(colorImg, p.x, p.y, 5, 0.0, color, 1);
        DispCircle(colorImg, p.x, p.y, 3, color, 1);
    }

    image = colorImg;
}

} // namespace Qi::Vision::Calib
