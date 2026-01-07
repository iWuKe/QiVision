#include <QiVision/Internal/ContourConvert.h>
#include <QiVision/Core/Exception.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <unordered_set>

namespace Qi::Vision::Internal {

// =============================================================================
// Helper: Point Inside Contour (Ray Casting)
// =============================================================================

bool IsPointInsideContour(const QContour& contour, const Point2d& point) {
    if (contour.Size() < 3) {
        return false;
    }

    const auto& pts = contour.GetContourPoints();
    size_t n = pts.size();
    int crossings = 0;

    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        double yi = pts[i].y;
        double yj = pts[j].y;
        double xi = pts[i].x;
        double xj = pts[j].x;

        // Check if ray from point to right crosses this edge
        if ((yi <= point.y && yj > point.y) || (yj <= point.y && yi > point.y)) {
            // Compute x coordinate of intersection
            double t = (point.y - yi) / (yj - yi);
            double xIntersect = xi + t * (xj - xi);

            if (point.x < xIntersect) {
                ++crossings;
            }
        }
    }

    return (crossings % 2) == 1;
}

// =============================================================================
// Helper: Winding Number
// =============================================================================

int ContourWindingNumber(const QContour& contour, const Point2d& point) {
    if (contour.Size() < 3) {
        return 0;
    }

    const auto& pts = contour.GetContourPoints();
    size_t n = pts.size();
    int winding = 0;

    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        double yi = pts[i].y;
        double yj = pts[j].y;
        double xi = pts[i].x;
        double xj = pts[j].x;

        if (yi <= point.y) {
            if (yj > point.y) {
                // Upward crossing
                double cross = (xj - xi) * (point.y - yi) - (point.x - xi) * (yj - yi);
                if (cross > 0) {
                    ++winding;
                }
            }
        } else {
            if (yj <= point.y) {
                // Downward crossing
                double cross = (xj - xi) * (point.y - yi) - (point.x - xi) * (yj - yi);
                if (cross < 0) {
                    --winding;
                }
            }
        }
    }

    return winding;
}

// =============================================================================
// Helper: Contour Direction (CCW check)
// =============================================================================

bool IsContourCCW(const QContour& contour) {
    return contour.SignedArea() > 0;
}

// =============================================================================
// Helper: Reverse Contour
// Note: ReverseContour is already implemented in ContourProcess.h
// =============================================================================

// =============================================================================
// Scanline Fill Helper - Simple and Robust Implementation
// =============================================================================

namespace {

// Find all x intersections of scanline y with contour edges
std::vector<double> FindScanlineIntersections(const QContour& contour, double y) {
    std::vector<double> intersections;
    const auto& pts = contour.GetContourPoints();
    size_t n = pts.size();

    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;

        double y1 = pts[i].y;
        double y2 = pts[j].y;
        double x1 = pts[i].x;
        double x2 = pts[j].x;

        // Check if scanline crosses this edge
        // Use half-open interval: include bottom, exclude top
        if ((y1 <= y && y < y2) || (y2 <= y && y < y1)) {
            // Compute intersection x
            double t = (y - y1) / (y2 - y1);
            double x = x1 + t * (x2 - x1);
            intersections.push_back(x);
        }
    }

    std::sort(intersections.begin(), intersections.end());
    return intersections;
}

} // namespace

// =============================================================================
// Contour to Region Conversion
// =============================================================================

QRegion ContourToRegion(const QContour& contour, ContourFillMode mode) {
    if (contour.Size() < 3) {
        return QRegion();
    }

    if (mode == ContourFillMode::Margin) {
        return ContourLineToRegion(contour);
    }

    // Get bounding box
    Rect2d bbox = contour.BoundingBox();
    int32_t yMin = static_cast<int32_t>(std::floor(bbox.y));
    int32_t yMax = static_cast<int32_t>(std::ceil(bbox.y + bbox.height));

    std::vector<QRegion::Run> runs;
    runs.reserve((yMax - yMin) * 2);

    // Scanline fill
    for (int32_t y = yMin; y < yMax; ++y) {
        // Sample at pixel center
        double scanY = y + 0.5;

        std::vector<double> intersections = FindScanlineIntersections(contour, scanY);

        // Fill between pairs (even-odd rule)
        for (size_t i = 0; i + 1 < intersections.size(); i += 2) {
            int32_t xStart = static_cast<int32_t>(std::ceil(intersections[i]));
            int32_t xEnd = static_cast<int32_t>(std::ceil(intersections[i + 1]));

            if (xStart < xEnd) {
                runs.emplace_back(y, xStart, xEnd);
            }
        }
    }

    return QRegion(runs);
}

QRegion ContoursToRegion(const QContourArray& contours, ContourFillMode mode) {
    if (contours.Empty()) {
        return QRegion();
    }

    QRegion result;
    for (size_t i = 0; i < contours.Size(); ++i) {
        QRegion contourRegion = ContourToRegion(contours[i], mode);
        result = result.Union(contourRegion);
    }

    return result;
}

QRegion ContourWithHolesToRegion(const QContour& contour,
                                  const QContourArray& holes,
                                  ContourFillMode mode) {
    QRegion outerRegion = ContourToRegion(contour, mode);

    if (holes.Empty()) {
        return outerRegion;
    }

    // Subtract holes
    for (size_t i = 0; i < holes.Size(); ++i) {
        QRegion holeRegion = ContourToRegion(holes[i], ContourFillMode::Filled);
        outerRegion = outerRegion.Difference(holeRegion);
    }

    return outerRegion;
}

// =============================================================================
// Region to Contour Conversion (Boundary Tracing)
// =============================================================================

namespace {

// Moore-Neighbor tracing directions (8-connected)
// Order: E, NE, N, NW, W, SW, S, SE
const int32_t dx8[8] = { 1,  1,  0, -1, -1, -1,  0,  1};
const int32_t dy8[8] = { 0, -1, -1, -1,  0,  1,  1,  1};

// 4-connected directions: E, N, W, S
const int32_t dx4[4] = { 1,  0, -1,  0};
const int32_t dy4[4] = { 0, -1,  0,  1};

// Find starting point for boundary tracing
bool FindBoundaryStart(const QRegion& region, const std::set<std::pair<int32_t, int32_t>>& visited,
                       int32_t& startX, int32_t& startY) {
    for (const auto& run : region.Runs()) {
        // Left edge of run is always boundary
        if (visited.find({run.colBegin, run.row}) == visited.end()) {
            // Check if there's no pixel to the left
            if (!region.Contains(run.colBegin - 1, run.row)) {
                startX = run.colBegin;
                startY = run.row;
                return true;
            }
        }
    }
    return false;
}

// Trace a single boundary contour using Moore-Neighbor tracing
QContour TraceBoundary8(const QRegion& region, int32_t startX, int32_t startY,
                        std::set<std::pair<int32_t, int32_t>>& visited) {
    QContour contour;
    contour.SetClosed(true);

    int32_t x = startX;
    int32_t y = startY;
    int dir = 0; // Start looking East

    // First point
    contour.AddPoint(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5);
    visited.insert({x, y});

    bool first = true;
    int32_t firstX = x, firstY = y;

    do {
        // Search for next boundary pixel in CCW order
        bool found = false;
        int searchStart = (dir + 5) % 8; // Start from (dir - 3 + 8) % 8 = backtrack + 1

        for (int i = 0; i < 8; ++i) {
            int checkDir = (searchStart + i) % 8;
            int32_t nx = x + dx8[checkDir];
            int32_t ny = y + dy8[checkDir];

            if (region.Contains(nx, ny)) {
                x = nx;
                y = ny;
                dir = checkDir;
                found = true;
                break;
            }
        }

        if (!found) {
            break; // Isolated pixel
        }

        if (first) {
            first = false;
        } else if (x == firstX && y == firstY) {
            break; // Back to start
        }

        contour.AddPoint(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5);
        visited.insert({x, y});

    } while (true);

    return contour;
}

// 4-connected boundary tracing
QContour TraceBoundary4(const QRegion& region, int32_t startX, int32_t startY,
                        std::set<std::pair<int32_t, int32_t>>& visited) {
    QContour contour;
    contour.SetClosed(true);

    int32_t x = startX;
    int32_t y = startY;
    int dir = 0;

    contour.AddPoint(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5);
    visited.insert({x, y});

    int32_t firstX = x, firstY = y;
    bool first = true;

    do {
        bool found = false;
        int searchStart = (dir + 3) % 4;

        for (int i = 0; i < 4; ++i) {
            int checkDir = (searchStart + i) % 4;
            int32_t nx = x + dx4[checkDir];
            int32_t ny = y + dy4[checkDir];

            if (region.Contains(nx, ny)) {
                x = nx;
                y = ny;
                dir = checkDir;
                found = true;
                break;
            }
        }

        if (!found) {
            break;
        }

        if (first) {
            first = false;
        } else if (x == firstX && y == firstY) {
            break;
        }

        contour.AddPoint(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5);
        visited.insert({x, y});

    } while (true);

    return contour;
}


} // namespace

QContourArray RegionToContours(const QRegion& region, BoundaryMode mode,
                                BoundaryConnectivity connectivity) {
    QContourArray result;

    if (region.Empty()) {
        return result;
    }

    std::set<std::pair<int32_t, int32_t>> visited;
    Rect2i bbox = region.BoundingBox();

    if (mode == BoundaryMode::Outer || mode == BoundaryMode::Both) {
        // Trace outer boundaries
        int32_t startX, startY;
        while (FindBoundaryStart(region, visited, startX, startY)) {
            QContour contour;
            if (connectivity == BoundaryConnectivity::EightConnected) {
                contour = TraceBoundary8(region, startX, startY, visited);
            } else {
                contour = TraceBoundary4(region, startX, startY, visited);
            }

            if (contour.Size() >= 3) {
                result.Add(std::move(contour));
            }
        }
    }

    if (mode == BoundaryMode::Inner || mode == BoundaryMode::Both) {
        // For inner boundaries (holes), we need to find pixels that are
        // inside the bounding box but not in the region
        QRegion complement = region.Complement(bbox);

        std::set<std::pair<int32_t, int32_t>> holeVisited;
        int32_t startX, startY;

        while (FindBoundaryStart(complement, holeVisited, startX, startY)) {
            // Skip if this is the outer complement (touches bbox boundary)
            if (startX == bbox.x || startX == bbox.x + bbox.width - 1 ||
                startY == bbox.y || startY == bbox.y + bbox.height - 1) {
                // Mark all connected pixels as visited
                QContour dummy;
                if (connectivity == BoundaryConnectivity::EightConnected) {
                    dummy = TraceBoundary8(complement, startX, startY, holeVisited);
                } else {
                    dummy = TraceBoundary4(complement, startX, startY, holeVisited);
                }
                continue;
            }

            QContour contour;
            if (connectivity == BoundaryConnectivity::EightConnected) {
                contour = TraceBoundary8(complement, startX, startY, holeVisited);
            } else {
                contour = TraceBoundary4(complement, startX, startY, holeVisited);
            }

            if (contour.Size() >= 3) {
                result.Add(std::move(contour));
            }
        }
    }

    return result;
}

QContour RegionToContour(const QRegion& region, BoundaryConnectivity connectivity) {
    QContourArray contours = RegionToContours(region, BoundaryMode::Outer, connectivity);

    if (contours.Empty()) {
        return QContour();
    }

    // Return the largest contour by area
    size_t maxIdx = 0;
    double maxArea = std::abs(contours[0].Area());

    for (size_t i = 1; i < contours.Size(); ++i) {
        double area = std::abs(contours[i].Area());
        if (area > maxArea) {
            maxArea = area;
            maxIdx = i;
        }
    }

    return contours[maxIdx];
}

QContourArray RegionToSubpixelContours(const QRegion& region, BoundaryMode mode) {
    // Get pixel-level contours first
    QContourArray contours = RegionToContours(region, mode, BoundaryConnectivity::EightConnected);

    // For subpixel precision, we interpolate boundary positions
    // Simple approach: smooth the contours slightly
    QContourArray result;

    for (size_t i = 0; i < contours.Size(); ++i) {
        if (contours[i].Size() < 5) {
            result.Add(contours[i]);
            continue;
        }

        // Apply mild smoothing for subpixel effect
        QContour smoothed = contours[i].Smooth(0.5);
        result.Add(std::move(smoothed));
    }

    return result;
}

// =============================================================================
// Contour Line to Region (Bresenham)
// =============================================================================

namespace {

// Bresenham's line algorithm
void RasterizeLine(int32_t x0, int32_t y0, int32_t x1, int32_t y1,
                   std::vector<QRegion::Run>& runs) {
    int32_t dx = std::abs(x1 - x0);
    int32_t dy = std::abs(y1 - y0);
    int32_t sx = (x0 < x1) ? 1 : -1;
    int32_t sy = (y0 < y1) ? 1 : -1;
    int32_t err = dx - dy;

    while (true) {
        runs.emplace_back(y0, x0, x0 + 1);

        if (x0 == x1 && y0 == y1) break;

        int32_t e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

} // namespace

QRegion ContourLineToRegion(const QContour& contour) {
    if (contour.Size() < 2) {
        if (contour.Size() == 1) {
            int32_t x = static_cast<int32_t>(std::round(contour[0].x));
            int32_t y = static_cast<int32_t>(std::round(contour[0].y));
            return QRegion({QRegion::Run(y, x, x + 1)});
        }
        return QRegion();
    }

    std::vector<QRegion::Run> runs;
    runs.reserve(contour.Size() * 2);

    for (size_t i = 0; i < contour.Size(); ++i) {
        size_t j = (i + 1) % contour.Size();
        if (!contour.IsClosed() && j == 0) break;

        int32_t x0 = static_cast<int32_t>(std::round(contour[i].x));
        int32_t y0 = static_cast<int32_t>(std::round(contour[i].y));
        int32_t x1 = static_cast<int32_t>(std::round(contour[j].x));
        int32_t y1 = static_cast<int32_t>(std::round(contour[j].y));

        RasterizeLine(x0, y0, x1, y1, runs);
    }

    return QRegion(runs);
}

// =============================================================================
// Thick Line Region
// =============================================================================

QRegion ContourToThickLineRegion(const QContour& contour, double thickness) {
    if (contour.Size() < 2 || thickness < 1.0) {
        return ContourLineToRegion(contour);
    }

    // Dilate the line region
    QRegion lineRegion = ContourLineToRegion(contour);
    int32_t dilateSize = static_cast<int32_t>(std::ceil(thickness / 2.0));

    return lineRegion.Dilate(dilateSize * 2 + 1, dilateSize * 2 + 1);
}

// =============================================================================
// Polygon Conversion
// =============================================================================

QContour ContourToPolygon(const QContour& contour, double tolerance) {
    return contour.Simplify(tolerance);
}

QContour RegionToPolygon(const QRegion& region, double tolerance) {
    QContour boundary = RegionToContour(region);
    return boundary.Simplify(tolerance);
}

// =============================================================================
// Point Set Conversion
// =============================================================================

QRegion ContourPointsToRegion(const QContour& contour) {
    std::vector<QRegion::Run> runs;
    runs.reserve(contour.Size());

    for (size_t i = 0; i < contour.Size(); ++i) {
        int32_t x = static_cast<int32_t>(std::round(contour[i].x));
        int32_t y = static_cast<int32_t>(std::round(contour[i].y));
        runs.emplace_back(y, x, x + 1);
    }

    return QRegion(runs);
}

QContour RegionPixelsToContour(const QRegion& region) {
    QContour contour;
    contour.SetClosed(false);

    // Collect all pixels, ordered by row then column
    for (const auto& run : region.Runs()) {
        for (int32_t x = run.colBegin; x < run.colEnd; ++x) {
            contour.AddPoint(static_cast<double>(x) + 0.5,
                           static_cast<double>(run.row) + 0.5);
        }
    }

    return contour;
}

} // namespace Qi::Vision::Internal
