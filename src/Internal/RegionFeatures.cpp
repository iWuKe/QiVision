#include <QiVision/Internal/RegionFeatures.h>
#include <QiVision/Internal/RLEOps.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stack>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

constexpr double PI = 3.14159265358979323846;
constexpr double EPSILON = 1e-10;

// =============================================================================
// Basic Features
// =============================================================================

int64_t ComputeArea(const QRegion& region) {
    return region.Area();
}

// Note: ComputePerimeter is defined in RLEOps.h and reused here

Point2d ComputeRegionCentroid(const QRegion& region) {
    auto centroid = ComputeCentroid(region.Runs());
    return Point2d(centroid.x, centroid.y);
}

Rect2i ComputeBoundingBox(const QRegion& region) {
    return region.BoundingBox();
}

RegionBasicFeatures ComputeBasicFeatures(const QRegion& region) {
    RegionBasicFeatures features;

    if (region.Empty()) return features;

    features.area = ComputeArea(region);
    features.perimeter = ComputePerimeter(region);

    Point2d centroid = ComputeRegionCentroid(region);
    features.centroidX = centroid.x;
    features.centroidY = centroid.y;

    features.boundingBox = ComputeBoundingBox(region);

    return features;
}

// =============================================================================
// Moment Computation
// =============================================================================

double ComputeRawMoment(const QRegion& region, int32_t p, int32_t q) {
    if (region.Empty()) return 0.0;

    double moment = 0.0;
    const auto& runs = region.Runs();

    for (const auto& run : runs) {
        double y = static_cast<double>(run.row);
        double yPow = std::pow(y, q);

        for (int32_t x = run.colBegin; x < run.colEnd; ++x) {
            double xPow = std::pow(static_cast<double>(x), p);
            moment += xPow * yPow;
        }
    }

    return moment;
}

double ComputeCentralMoment(const QRegion& region, int32_t p, int32_t q) {
    if (region.Empty()) return 0.0;

    Point2d centroid = ComputeRegionCentroid(region);
    double cx = centroid.x;
    double cy = centroid.y;

    double moment = 0.0;
    const auto& runs = region.Runs();

    for (const auto& run : runs) {
        double dy = static_cast<double>(run.row) - cy;
        double dyPow = std::pow(dy, q);

        for (int32_t x = run.colBegin; x < run.colEnd; ++x) {
            double dx = static_cast<double>(x) - cx;
            double dxPow = std::pow(dx, p);
            moment += dxPow * dyPow;
        }
    }

    return moment;
}

RegionMoments ComputeMoments(const QRegion& region) {
    RegionMoments moments;

    if (region.Empty()) return moments;

    // Compute centroid first for central moments
    Point2d centroid = ComputeRegionCentroid(region);
    double cx = centroid.x;
    double cy = centroid.y;

    const auto& runs = region.Runs();

    // Accumulate all moments in single pass for efficiency
    for (const auto& run : runs) {
        double y = static_cast<double>(run.row);
        double dy = y - cy;

        for (int32_t col = run.colBegin; col < run.colEnd; ++col) {
            double x = static_cast<double>(col);
            double dx = x - cx;

            // Raw moments
            moments.m00 += 1.0;
            moments.m10 += x;
            moments.m01 += y;
            moments.m20 += x * x;
            moments.m11 += x * y;
            moments.m02 += y * y;
            moments.m30 += x * x * x;
            moments.m21 += x * x * y;
            moments.m12 += x * y * y;
            moments.m03 += y * y * y;

            // Central moments
            moments.mu20 += dx * dx;
            moments.mu11 += dx * dy;
            moments.mu02 += dy * dy;
            moments.mu30 += dx * dx * dx;
            moments.mu21 += dx * dx * dy;
            moments.mu12 += dx * dy * dy;
            moments.mu03 += dy * dy * dy;
        }
    }

    // Normalized central moments
    if (moments.m00 > EPSILON) {
        double m00_2 = moments.m00 * moments.m00;
        double m00_25 = std::pow(moments.m00, 2.5);

        moments.nu20 = moments.mu20 / m00_2;
        moments.nu11 = moments.mu11 / m00_2;
        moments.nu02 = moments.mu02 / m00_2;
        moments.nu30 = moments.mu30 / m00_25;
        moments.nu21 = moments.mu21 / m00_25;
        moments.nu12 = moments.mu12 / m00_25;
        moments.nu03 = moments.mu03 / m00_25;
    }

    // Hu moments
    moments.hu = ComputeHuMoments(region);

    return moments;
}

std::array<double, 7> ComputeHuMoments(const QRegion& region) {
    std::array<double, 7> hu = {0};

    if (region.Empty()) return hu;

    // Get normalized central moments
    Point2d centroid = ComputeRegionCentroid(region);
    double cx = centroid.x;
    double cy = centroid.y;

    double mu00 = 0, mu20 = 0, mu11 = 0, mu02 = 0;
    double mu30 = 0, mu21 = 0, mu12 = 0, mu03 = 0;

    const auto& runs = region.Runs();
    for (const auto& run : runs) {
        double dy = static_cast<double>(run.row) - cy;
        for (int32_t col = run.colBegin; col < run.colEnd; ++col) {
            double dx = static_cast<double>(col) - cx;

            mu00 += 1.0;
            mu20 += dx * dx;
            mu11 += dx * dy;
            mu02 += dy * dy;
            mu30 += dx * dx * dx;
            mu21 += dx * dx * dy;
            mu12 += dx * dy * dy;
            mu03 += dy * dy * dy;
        }
    }

    if (mu00 < EPSILON) return hu;

    // Normalize
    double m2 = mu00 * mu00;
    double m25 = std::pow(mu00, 2.5);

    double n20 = mu20 / m2;
    double n11 = mu11 / m2;
    double n02 = mu02 / m2;
    double n30 = mu30 / m25;
    double n21 = mu21 / m25;
    double n12 = mu12 / m25;
    double n03 = mu03 / m25;

    // Hu moments
    hu[0] = n20 + n02;
    hu[1] = (n20 - n02) * (n20 - n02) + 4 * n11 * n11;
    hu[2] = (n30 - 3 * n12) * (n30 - 3 * n12) + (3 * n21 - n03) * (3 * n21 - n03);
    hu[3] = (n30 + n12) * (n30 + n12) + (n21 + n03) * (n21 + n03);

    double a = n30 + n12;
    double b = n21 + n03;
    hu[4] = (n30 - 3 * n12) * a * (a * a - 3 * b * b) +
            (3 * n21 - n03) * b * (3 * a * a - b * b);

    hu[5] = (n20 - n02) * (a * a - b * b) + 4 * n11 * a * b;

    hu[6] = (3 * n21 - n03) * a * (a * a - 3 * b * b) -
            (n30 - 3 * n12) * b * (3 * a * a - b * b);

    return hu;
}

// =============================================================================
// Shape Features
// =============================================================================

// Note: ComputeCircularity, ComputeCompactness, ComputeRectangularity
// are defined in RLEOps.cpp and reused here

double ComputeElongation(const QRegion& region) {
    double majorAxis, minorAxis;
    ComputePrincipalAxes(region, majorAxis, minorAxis);

    if (minorAxis < EPSILON) return std::numeric_limits<double>::max();

    return majorAxis / minorAxis;
}

double ComputeConvexity(const QRegion& region) {
    if (region.Empty()) return 0.0;

    double perimeter = ComputePerimeter(region);
    double convexPerimeter = ComputeConvexHullPerimeter(region);

    if (perimeter < EPSILON) return 0.0;

    return convexPerimeter / perimeter;
}

double ComputeSolidity(const QRegion& region) {
    if (region.Empty()) return 0.0;

    double area = static_cast<double>(ComputeArea(region));
    double convexArea = ComputeConvexHullArea(region);

    if (convexArea < EPSILON) return 0.0;

    return area / convexArea;
}

RegionShapeFeatures ComputeShapeFeatures(const QRegion& region) {
    RegionShapeFeatures features;

    if (region.Empty()) return features;

    features.circularity = ComputeCircularity(region);
    features.compactness = ComputeCompactness(region);
    features.elongation = ComputeElongation(region);
    features.rectangularity = ComputeRectangularity(region);
    features.convexity = ComputeConvexity(region);
    features.solidity = ComputeSolidity(region);

    // Compute roundness and aspect ratio
    RegionEllipseFeatures ellipse = ComputeEllipseFeatures(region);
    double area = static_cast<double>(ComputeArea(region));

    if (ellipse.majorAxis > EPSILON) {
        features.roundness = 4.0 * area / (PI * ellipse.majorAxis * ellipse.majorAxis);
    }

    Rect2i bbox = ComputeBoundingBox(region);
    if (bbox.height > 0) {
        features.aspectRatio = static_cast<double>(bbox.width) / bbox.height;
    }

    return features;
}

// =============================================================================
// Ellipse and Orientation Features
// =============================================================================

double ComputeOrientation(const QRegion& region) {
    if (region.Empty()) return 0.0;

    double mu20 = ComputeCentralMoment(region, 2, 0);
    double mu11 = ComputeCentralMoment(region, 1, 1);
    double mu02 = ComputeCentralMoment(region, 0, 2);

    // Principal axis orientation
    double angle = 0.5 * std::atan2(2.0 * mu11, mu20 - mu02);

    return angle;
}

void ComputePrincipalAxes(const QRegion& region, double& majorAxis, double& minorAxis) {
    majorAxis = 0.0;
    minorAxis = 0.0;

    if (region.Empty()) return;

    double mu00 = static_cast<double>(ComputeArea(region));
    if (mu00 < EPSILON) return;

    double mu20 = ComputeCentralMoment(region, 2, 0);
    double mu11 = ComputeCentralMoment(region, 1, 1);
    double mu02 = ComputeCentralMoment(region, 0, 2);

    // Eigenvalues of covariance matrix
    double a = mu20 / mu00;
    double b = mu11 / mu00;
    double c = mu02 / mu00;

    double discriminant = std::sqrt((a - c) * (a - c) + 4 * b * b);
    double lambda1 = (a + c + discriminant) / 2.0;
    double lambda2 = (a + c - discriminant) / 2.0;

    // Principal axes lengths (2 * sqrt of eigenvalues, scaled)
    majorAxis = 2.0 * std::sqrt(std::max(0.0, lambda1));
    minorAxis = 2.0 * std::sqrt(std::max(0.0, lambda2));
}

RegionEllipseFeatures ComputeEllipseFeatures(const QRegion& region) {
    RegionEllipseFeatures features;

    if (region.Empty()) return features;

    Point2d centroid = ComputeRegionCentroid(region);
    features.centerX = centroid.x;
    features.centerY = centroid.y;

    ComputePrincipalAxes(region, features.majorAxis, features.minorAxis);
    features.angle = ComputeOrientation(region);

    if (features.majorAxis > EPSILON) {
        double ratio = features.minorAxis / features.majorAxis;
        features.eccentricity = std::sqrt(1.0 - ratio * ratio);
    }

    return features;
}

// =============================================================================
// Convex Hull
// =============================================================================

namespace {

// Cross product of vectors OA and OB
int64_t Cross(const Point2i& O, const Point2i& A, const Point2i& B) {
    return static_cast<int64_t>(A.x - O.x) * (B.y - O.y) -
           static_cast<int64_t>(A.y - O.y) * (B.x - O.x);
}

} // anonymous namespace

std::vector<Point2i> ComputeConvexHull(const QRegion& region) {
    if (region.Empty()) return {};

    // Extract boundary points
    std::vector<Point2i> points;
    const auto& runs = region.Runs();

    for (const auto& run : runs) {
        // Add left and right endpoints
        points.push_back({run.colBegin, run.row});
        if (run.colEnd - 1 != run.colBegin) {
            points.push_back({run.colEnd - 1, run.row});
        }
    }

    if (points.size() < 3) {
        return points;
    }

    // Sort points lexicographically
    std::sort(points.begin(), points.end(), [](const Point2i& a, const Point2i& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });

    // Remove duplicates
    points.erase(std::unique(points.begin(), points.end(),
        [](const Point2i& a, const Point2i& b) {
            return a.x == b.x && a.y == b.y;
        }), points.end());

    if (points.size() < 3) {
        return points;
    }

    // Andrew's monotone chain algorithm
    std::vector<Point2i> hull;
    int n = static_cast<int>(points.size());

    // Build lower hull
    for (int i = 0; i < n; ++i) {
        while (hull.size() >= 2 && Cross(hull[hull.size()-2], hull[hull.size()-1], points[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }

    // Build upper hull
    int lowerSize = static_cast<int>(hull.size());
    for (int i = n - 2; i >= 0; --i) {
        while (static_cast<int>(hull.size()) > lowerSize &&
               Cross(hull[hull.size()-2], hull[hull.size()-1], points[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }

    // Remove last point (same as first)
    hull.pop_back();

    return hull;
}

double ComputeConvexHullArea(const QRegion& region) {
    auto hull = ComputeConvexHull(region);

    if (hull.size() < 3) return 0.0;

    // Shoelace formula
    double area = 0.0;
    int n = static_cast<int>(hull.size());

    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += static_cast<double>(hull[i].x) * hull[j].y;
        area -= static_cast<double>(hull[j].x) * hull[i].y;
    }

    return std::abs(area) / 2.0;
}

double ComputeConvexHullPerimeter(const QRegion& region) {
    auto hull = ComputeConvexHull(region);

    if (hull.size() < 2) return 0.0;

    double perimeter = 0.0;
    int n = static_cast<int>(hull.size());

    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        double dx = hull[j].x - hull[i].x;
        double dy = hull[j].y - hull[i].y;
        perimeter += std::sqrt(dx * dx + dy * dy);
    }

    return perimeter;
}

// =============================================================================
// Enclosing Shapes
// =============================================================================

MinAreaRect ComputeMinAreaRect(const QRegion& region) {
    MinAreaRect rect;

    auto hull = ComputeConvexHull(region);
    if (hull.size() < 3) {
        if (!hull.empty()) {
            rect.center = Point2d(hull[0].x, hull[0].y);
        }
        return rect;
    }

    // Rotating calipers algorithm
    int n = static_cast<int>(hull.size());
    double minArea = std::numeric_limits<double>::max();
    MinAreaRect bestRect;

    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;

        // Edge vector
        double edgeX = hull[j].x - hull[i].x;
        double edgeY = hull[j].y - hull[i].y;
        double edgeLen = std::sqrt(edgeX * edgeX + edgeY * edgeY);

        if (edgeLen < EPSILON) continue;

        // Normalize
        edgeX /= edgeLen;
        edgeY /= edgeLen;

        // Perpendicular
        double perpX = -edgeY;
        double perpY = edgeX;

        // Project all points
        double minProj = std::numeric_limits<double>::max();
        double maxProj = -std::numeric_limits<double>::max();
        double minPerp = std::numeric_limits<double>::max();
        double maxPerp = -std::numeric_limits<double>::max();

        for (const auto& pt : hull) {
            double proj = pt.x * edgeX + pt.y * edgeY;
            double perp = pt.x * perpX + pt.y * perpY;

            minProj = std::min(minProj, proj);
            maxProj = std::max(maxProj, proj);
            minPerp = std::min(minPerp, perp);
            maxPerp = std::max(maxPerp, perp);
        }

        double width = maxProj - minProj;
        double height = maxPerp - minPerp;
        double area = width * height;

        if (area < minArea) {
            minArea = area;
            bestRect.width = width;
            bestRect.height = height;
            bestRect.angle = std::atan2(edgeY, edgeX);

            // Center
            double centerProj = (minProj + maxProj) / 2.0;
            double centerPerp = (minPerp + maxPerp) / 2.0;
            bestRect.center.x = centerProj * edgeX + centerPerp * perpX;
            bestRect.center.y = centerProj * edgeY + centerPerp * perpY;

            // Corners
            double hw = width / 2.0;
            double hh = height / 2.0;
            bestRect.corners[0] = Point2d(bestRect.center.x - hw * edgeX - hh * perpX,
                                          bestRect.center.y - hw * edgeY - hh * perpY);
            bestRect.corners[1] = Point2d(bestRect.center.x + hw * edgeX - hh * perpX,
                                          bestRect.center.y + hw * edgeY - hh * perpY);
            bestRect.corners[2] = Point2d(bestRect.center.x + hw * edgeX + hh * perpX,
                                          bestRect.center.y + hw * edgeY + hh * perpY);
            bestRect.corners[3] = Point2d(bestRect.center.x - hw * edgeX + hh * perpX,
                                          bestRect.center.y - hw * edgeY + hh * perpY);
        }
    }

    return bestRect;
}

MinEnclosingCircle ComputeMinEnclosingCircle(const QRegion& region) {
    MinEnclosingCircle circle;

    if (region.Empty()) return circle;

    // Get convex hull points
    auto hull = ComputeConvexHull(region);
    if (hull.empty()) return circle;

    if (hull.size() == 1) {
        circle.center = Point2d(hull[0].x, hull[0].y);
        circle.radius = 0.0;
        return circle;
    }

    if (hull.size() == 2) {
        circle.center.x = (hull[0].x + hull[1].x) / 2.0;
        circle.center.y = (hull[0].y + hull[1].y) / 2.0;
        double dx = hull[1].x - hull[0].x;
        double dy = hull[1].y - hull[0].y;
        circle.radius = std::sqrt(dx * dx + dy * dy) / 2.0;
        return circle;
    }

    // Welzl's algorithm (simplified for small point sets)
    std::vector<Point2d> points;
    for (const auto& pt : hull) {
        points.push_back(Point2d(pt.x, pt.y));
    }

    // Simple O(n^3) algorithm for minimum enclosing circle
    double minRadius = std::numeric_limits<double>::max();

    // Try all pairs (circle through 2 points)
    int n = static_cast<int>(points.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            Point2d center((points[i].x + points[j].x) / 2.0,
                          (points[i].y + points[j].y) / 2.0);
            double dx = points[j].x - points[i].x;
            double dy = points[j].y - points[i].y;
            double radius = std::sqrt(dx * dx + dy * dy) / 2.0;

            // Check if all points are inside
            bool valid = true;
            for (int k = 0; k < n && valid; ++k) {
                double d = std::sqrt((points[k].x - center.x) * (points[k].x - center.x) +
                                    (points[k].y - center.y) * (points[k].y - center.y));
                if (d > radius + EPSILON) valid = false;
            }

            if (valid && radius < minRadius) {
                minRadius = radius;
                circle.center = center;
                circle.radius = radius;
            }
        }
    }

    // Try all triples (circumscribed circle)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            for (int k = j + 1; k < n; ++k) {
                // Circumcircle of three points
                double ax = points[i].x, ay = points[i].y;
                double bx = points[j].x, by = points[j].y;
                double cx = points[k].x, cy = points[k].y;

                double d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
                if (std::abs(d) < EPSILON) continue;

                double ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) +
                            (cx*cx + cy*cy) * (ay - by)) / d;
                double uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) +
                            (cx*cx + cy*cy) * (bx - ax)) / d;

                double radius = std::sqrt((ax - ux) * (ax - ux) + (ay - uy) * (ay - uy));

                // Check if all points are inside
                bool valid = true;
                for (int m = 0; m < n && valid; ++m) {
                    double dist = std::sqrt((points[m].x - ux) * (points[m].x - ux) +
                                           (points[m].y - uy) * (points[m].y - uy));
                    if (dist > radius + EPSILON) valid = false;
                }

                if (valid && radius < minRadius) {
                    minRadius = radius;
                    circle.center = Point2d(ux, uy);
                    circle.radius = radius;
                }
            }
        }
    }

    return circle;
}

// =============================================================================
// Comprehensive Feature Extraction
// =============================================================================

RegionFeatures ComputeAllFeatures(const QRegion& region) {
    RegionFeatures features;

    if (region.Empty()) return features;

    features.basic = ComputeBasicFeatures(region);
    features.shape = ComputeShapeFeatures(region);
    features.moments = ComputeMoments(region);
    features.ellipse = ComputeEllipseFeatures(region);

    return features;
}

std::vector<RegionFeatures> ComputeAllFeatures(const std::vector<QRegion>& regions) {
    std::vector<RegionFeatures> features;
    features.reserve(regions.size());

    for (const auto& region : regions) {
        features.push_back(ComputeAllFeatures(region));
    }

    return features;
}

// =============================================================================
// Feature-based Selection
// =============================================================================

std::vector<QRegion> SelectByCircularity(const std::vector<QRegion>& regions,
                                          double minCirc,
                                          double maxCirc) {
    std::vector<QRegion> result;

    for (const auto& region : regions) {
        double circ = ComputeCircularity(region);
        if (circ >= minCirc && circ <= maxCirc) {
            result.push_back(region);
        }
    }

    return result;
}

std::vector<QRegion> SelectByCompactness(const std::vector<QRegion>& regions,
                                          double minComp,
                                          double maxComp) {
    std::vector<QRegion> result;

    for (const auto& region : regions) {
        double comp = ComputeCompactness(region);
        if (comp >= minComp && comp <= maxComp) {
            result.push_back(region);
        }
    }

    return result;
}

std::vector<QRegion> SelectByElongation(const std::vector<QRegion>& regions,
                                         double minElong,
                                         double maxElong) {
    std::vector<QRegion> result;

    for (const auto& region : regions) {
        double elong = ComputeElongation(region);
        if (elong >= minElong && elong <= maxElong) {
            result.push_back(region);
        }
    }

    return result;
}

std::vector<QRegion> SelectByOrientation(const std::vector<QRegion>& regions,
                                          double minAngle,
                                          double maxAngle) {
    std::vector<QRegion> result;

    for (const auto& region : regions) {
        double angle = ComputeOrientation(region);
        if (angle >= minAngle && angle <= maxAngle) {
            result.push_back(region);
        }
    }

    return result;
}

} // namespace Qi::Vision::Internal
