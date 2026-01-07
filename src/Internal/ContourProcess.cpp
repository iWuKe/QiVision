/**
 * @file ContourProcess.cpp
 * @brief Implementation of contour processing operations
 */

#include <QiVision/Internal/ContourProcess.h>
#include <QiVision/Internal/Gaussian.h>

#include <algorithm>
#include <cmath>
#include <queue>
#include <functional>
#include <limits>

namespace Qi::Vision::Internal {

// =============================================================================
// Helper Functions (Internal)
// =============================================================================

namespace {

/**
 * @brief Compute perpendicular distance from point to line segment
 */
double PerpendicularDistance(const Point2d& p, const Point2d& lineStart, const Point2d& lineEnd) {
    double dx = lineEnd.x - lineStart.x;
    double dy = lineEnd.y - lineStart.y;
    double lenSq = dx * dx + dy * dy;

    if (lenSq < EPSILON * EPSILON) {
        // Degenerate line (points are the same)
        double ddx = p.x - lineStart.x;
        double ddy = p.y - lineStart.y;
        return std::sqrt(ddx * ddx + ddy * ddy);
    }

    // Perpendicular distance = |cross product| / |line length|
    double cross = (p.x - lineStart.x) * dy - (p.y - lineStart.y) * dx;
    return std::abs(cross) / std::sqrt(lenSq);
}

/**
 * @brief Compute triangle area formed by three points
 */
double TriangleArea(const Point2d& p0, const Point2d& p1, const Point2d& p2) {
    return 0.5 * std::abs((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y));
}

/**
 * @brief Compute distance between two points
 */
double PointDistance(const Point2d& p1, const Point2d& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return std::sqrt(dx * dx + dy * dy);
}

/**
 * @brief Get index with wrapping for closed contours
 */
size_t WrapIndex(int64_t index, size_t size, bool closed) {
    if (size == 0) return 0;

    if (closed) {
        // Wrap around for closed contours
        index = index % static_cast<int64_t>(size);
        if (index < 0) index += static_cast<int64_t>(size);
        return static_cast<size_t>(index);
    } else {
        // Reflect for open contours
        if (index < 0) {
            index = -index;
        }
        if (index >= static_cast<int64_t>(size)) {
            index = 2 * static_cast<int64_t>(size) - index - 2;
        }
        return static_cast<size_t>(Clamp(index, static_cast<int64_t>(0), static_cast<int64_t>(size) - 1));
    }
}

} // anonymous namespace

// =============================================================================
// Utility Functions Implementation
// =============================================================================

double InterpolateAngle(double a1, double a2, double t) {
    // Ensure shortest path interpolation
    double diff = a2 - a1;
    while (diff > PI) diff -= TWO_PI;
    while (diff < -PI) diff += TWO_PI;
    return NormalizeAngle(a1 + t * diff);
}

ContourPoint InterpolateContourPoint(const ContourPoint& p1, const ContourPoint& p2,
                                      double t, AttributeMode attrMode) {
    ContourPoint result;

    // Position always linearly interpolated
    result.x = p1.x + t * (p2.x - p1.x);
    result.y = p1.y + t * (p2.y - p1.y);

    switch (attrMode) {
        case AttributeMode::None:
            result.amplitude = 0.0;
            result.direction = 0.0;
            result.curvature = 0.0;
            break;

        case AttributeMode::Interpolate:
            result.amplitude = p1.amplitude + t * (p2.amplitude - p1.amplitude);
            result.direction = InterpolateAngle(p1.direction, p2.direction, t);
            result.curvature = p1.curvature + t * (p2.curvature - p1.curvature);
            break;

        case AttributeMode::NearestNeighbor:
            if (t < 0.5) {
                result.amplitude = p1.amplitude;
                result.direction = p1.direction;
                result.curvature = p1.curvature;
            } else {
                result.amplitude = p2.amplitude;
                result.direction = p2.direction;
                result.curvature = p2.curvature;
            }
            break;
    }

    return result;
}

double ComputeContourLength(const QContour& contour) {
    if (contour.Size() < 2) return 0.0;

    double length = 0.0;
    for (size_t i = 1; i < contour.Size(); ++i) {
        length += contour[i - 1].DistanceTo(contour[i]);
    }

    // Add closing segment for closed contours
    if (contour.IsClosed() && contour.Size() >= 2) {
        length += contour[contour.Size() - 1].DistanceTo(contour[0]);
    }

    return length;
}

std::vector<double> ComputeCumulativeLength(const QContour& contour) {
    std::vector<double> cumLen(contour.Size(), 0.0);
    if (contour.Size() < 2) return cumLen;

    for (size_t i = 1; i < contour.Size(); ++i) {
        cumLen[i] = cumLen[i - 1] + contour[i - 1].DistanceTo(contour[i]);
    }

    return cumLen;
}

size_t FindSegmentByArcLength(const QContour& contour, double arcLength, double& localT) {
    if (contour.Size() < 2) {
        localT = 0.0;
        return 0;
    }

    std::vector<double> cumLen = ComputeCumulativeLength(contour);
    double totalLen = cumLen.back();

    // Handle closed contour total length
    if (contour.IsClosed()) {
        totalLen += contour[contour.Size() - 1].DistanceTo(contour[0]);
    }

    // Clamp arc length
    if (arcLength <= 0.0) {
        localT = 0.0;
        return 0;
    }
    if (arcLength >= totalLen) {
        localT = 1.0;
        return contour.Size() - 2;
    }

    // Binary search for segment
    size_t segIdx = 0;
    for (size_t i = 1; i < contour.Size(); ++i) {
        if (cumLen[i] >= arcLength) {
            segIdx = i - 1;
            break;
        }
    }

    // Check closing segment for closed contours
    if (contour.IsClosed() && arcLength > cumLen.back()) {
        segIdx = contour.Size() - 1;
        double segStart = cumLen.back();
        double segLen = contour[contour.Size() - 1].DistanceTo(contour[0]);
        localT = (segLen > EPSILON) ? (arcLength - segStart) / segLen : 0.0;
        return segIdx;
    }

    // Compute local parameter
    double segStart = cumLen[segIdx];
    double segEnd = (segIdx + 1 < contour.Size()) ? cumLen[segIdx + 1] : totalLen;
    double segLen = segEnd - segStart;
    localT = (segLen > EPSILON) ? (arcLength - segStart) / segLen : 0.0;

    return segIdx;
}

ContourPoint FindPointByArcLength(const QContour& contour, double arcLength, AttributeMode attrMode) {
    if (contour.Empty()) {
        return ContourPoint();
    }
    if (contour.Size() == 1) {
        return contour[0];
    }

    double localT;
    size_t segIdx = FindSegmentByArcLength(contour, arcLength, localT);

    // Handle closing segment for closed contours
    size_t nextIdx = (segIdx + 1) % contour.Size();
    if (!contour.IsClosed() && segIdx + 1 >= contour.Size()) {
        return contour[contour.Size() - 1];
    }

    return InterpolateContourPoint(contour[segIdx], contour[nextIdx], localT, attrMode);
}

ContourPoint FindPointByParameter(const QContour& contour, double t, AttributeMode attrMode) {
    double totalLen = ComputeContourLength(contour);
    return FindPointByArcLength(contour, t * totalLen, attrMode);
}

// =============================================================================
// Smoothing Functions Implementation
// =============================================================================

QContour SmoothContourGaussian(const QContour& contour, const GaussianSmoothParams& params) {
    if (contour.Size() < MIN_CONTOUR_POINTS_FOR_SMOOTH) {
        return contour;
    }

    double sigma = std::max(params.sigma, 0.1);
    int32_t halfWindow = static_cast<int32_t>(std::ceil(3.0 * sigma));
    int32_t windowSize = params.windowSize > 0 ? params.windowSize : 2 * halfWindow + 1;

    // Ensure odd window size
    if (windowSize % 2 == 0) windowSize++;
    windowSize = Clamp(windowSize, MIN_SMOOTH_WINDOW, MAX_SMOOTH_WINDOW);
    halfWindow = windowSize / 2;

    // Generate Gaussian kernel
    std::vector<double> kernel = Gaussian::Kernel1D(sigma, windowSize, true);

    QContour result;
    result.Reserve(contour.Size());
    size_t n = contour.Size();
    bool closed = contour.IsClosed();

    for (size_t i = 0; i < n; ++i) {
        double sumX = 0.0, sumY = 0.0;
        double sumAmp = 0.0, sumCurv = 0.0;
        double sumDirCos = 0.0, sumDirSin = 0.0;

        for (int32_t j = -halfWindow; j <= halfWindow; ++j) {
            size_t idx = WrapIndex(static_cast<int64_t>(i) + j, n, closed);
            const auto& pt = contour[idx];
            double w = kernel[static_cast<size_t>(j + halfWindow)];

            sumX += w * pt.x;
            sumY += w * pt.y;

            if (params.attrMode == AttributeMode::Interpolate) {
                sumAmp += w * pt.amplitude;
                sumDirCos += w * std::cos(pt.direction);
                sumDirSin += w * std::sin(pt.direction);
                sumCurv += w * pt.curvature;
            }
        }

        ContourPoint p;
        p.x = sumX;
        p.y = sumY;

        if (params.attrMode == AttributeMode::Interpolate) {
            p.amplitude = sumAmp;
            p.direction = std::atan2(sumDirSin, sumDirCos);
            p.curvature = sumCurv;
        } else if (params.attrMode == AttributeMode::NearestNeighbor) {
            p.amplitude = contour[i].amplitude;
            p.direction = contour[i].direction;
            p.curvature = contour[i].curvature;
        }

        result.AddPoint(p);
    }

    result.SetClosed(closed);
    return result;
}

QContour SmoothContourMovingAverage(const QContour& contour, const MovingAverageSmoothParams& params) {
    if (contour.Size() < MIN_CONTOUR_POINTS_FOR_SMOOTH) {
        return contour;
    }

    int32_t windowSize = params.windowSize;
    // Ensure odd window size
    if (windowSize % 2 == 0) windowSize++;
    windowSize = Clamp(windowSize, MIN_SMOOTH_WINDOW, MAX_SMOOTH_WINDOW);
    int32_t halfWindow = windowSize / 2;

    double weight = 1.0 / static_cast<double>(windowSize);

    QContour result;
    result.Reserve(contour.Size());
    size_t n = contour.Size();
    bool closed = contour.IsClosed();

    for (size_t i = 0; i < n; ++i) {
        double sumX = 0.0, sumY = 0.0;
        double sumAmp = 0.0, sumCurv = 0.0;
        double sumDirCos = 0.0, sumDirSin = 0.0;

        for (int32_t j = -halfWindow; j <= halfWindow; ++j) {
            size_t idx = WrapIndex(static_cast<int64_t>(i) + j, n, closed);
            const auto& pt = contour[idx];

            sumX += pt.x;
            sumY += pt.y;

            if (params.attrMode == AttributeMode::Interpolate) {
                sumAmp += pt.amplitude;
                sumDirCos += std::cos(pt.direction);
                sumDirSin += std::sin(pt.direction);
                sumCurv += pt.curvature;
            }
        }

        ContourPoint p;
        p.x = sumX * weight;
        p.y = sumY * weight;

        if (params.attrMode == AttributeMode::Interpolate) {
            p.amplitude = sumAmp * weight;
            p.direction = std::atan2(sumDirSin, sumDirCos);
            p.curvature = sumCurv * weight;
        } else if (params.attrMode == AttributeMode::NearestNeighbor) {
            p.amplitude = contour[i].amplitude;
            p.direction = contour[i].direction;
            p.curvature = contour[i].curvature;
        }

        result.AddPoint(p);
    }

    result.SetClosed(closed);
    return result;
}

QContour SmoothContourBilateral(const QContour& contour, const BilateralSmoothParams& params) {
    if (contour.Size() < MIN_CONTOUR_POINTS_FOR_SMOOTH) {
        return contour;
    }

    double sigmaSpace = std::max(params.sigmaSpace, 0.1);
    double sigmaRange = std::max(params.sigmaRange, 0.1);
    int32_t halfWindow = static_cast<int32_t>(std::ceil(3.0 * sigmaSpace));
    int32_t windowSize = params.windowSize > 0 ? params.windowSize : 2 * halfWindow + 1;

    if (windowSize % 2 == 0) windowSize++;
    windowSize = Clamp(windowSize, MIN_SMOOTH_WINDOW, MAX_SMOOTH_WINDOW);
    halfWindow = windowSize / 2;

    double spaceCoeff = -0.5 / (sigmaSpace * sigmaSpace);
    double rangeCoeff = -0.5 / (sigmaRange * sigmaRange);

    QContour result;
    result.Reserve(contour.Size());
    size_t n = contour.Size();
    bool closed = contour.IsClosed();

    // Pre-compute curvature if not available (use local angle difference)
    std::vector<double> curvatures(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        size_t prev = WrapIndex(static_cast<int64_t>(i) - 1, n, closed);
        size_t next = WrapIndex(static_cast<int64_t>(i) + 1, n, closed);

        double dx1 = contour[i].x - contour[prev].x;
        double dy1 = contour[i].y - contour[prev].y;
        double dx2 = contour[next].x - contour[i].x;
        double dy2 = contour[next].y - contour[i].y;

        double angle1 = std::atan2(dy1, dx1);
        double angle2 = std::atan2(dy2, dx2);
        double angleDiff = angle2 - angle1;
        while (angleDiff > PI) angleDiff -= TWO_PI;
        while (angleDiff < -PI) angleDiff += TWO_PI;

        curvatures[i] = std::abs(angleDiff);
    }

    for (size_t i = 0; i < n; ++i) {
        double sumX = 0.0, sumY = 0.0;
        double sumWeight = 0.0;
        double sumAmp = 0.0, sumCurv = 0.0;
        double sumDirCos = 0.0, sumDirSin = 0.0;

        double centerCurv = curvatures[i];

        for (int32_t j = -halfWindow; j <= halfWindow; ++j) {
            size_t idx = WrapIndex(static_cast<int64_t>(i) + j, n, closed);
            const auto& pt = contour[idx];

            // Spatial weight
            double spatialDist = static_cast<double>(j * j);
            double spatialWeight = std::exp(spatialDist * spaceCoeff);

            // Range weight (based on curvature difference)
            double curvDiff = curvatures[idx] - centerCurv;
            double rangeWeight = std::exp(curvDiff * curvDiff * rangeCoeff);

            double w = spatialWeight * rangeWeight;

            sumX += w * pt.x;
            sumY += w * pt.y;
            sumWeight += w;

            if (params.attrMode == AttributeMode::Interpolate) {
                sumAmp += w * pt.amplitude;
                sumDirCos += w * std::cos(pt.direction);
                sumDirSin += w * std::sin(pt.direction);
                sumCurv += w * pt.curvature;
            }
        }

        if (sumWeight < EPSILON) {
            result.AddPoint(contour[i]);
            continue;
        }

        ContourPoint p;
        p.x = sumX / sumWeight;
        p.y = sumY / sumWeight;

        if (params.attrMode == AttributeMode::Interpolate) {
            p.amplitude = sumAmp / sumWeight;
            p.direction = std::atan2(sumDirSin, sumDirCos);
            p.curvature = sumCurv / sumWeight;
        } else if (params.attrMode == AttributeMode::NearestNeighbor) {
            p.amplitude = contour[i].amplitude;
            p.direction = contour[i].direction;
            p.curvature = contour[i].curvature;
        }

        result.AddPoint(p);
    }

    result.SetClosed(closed);
    return result;
}

QContour SmoothContour(const QContour& contour, SmoothMethod method,
                       double sigma, int32_t windowSize) {
    switch (method) {
        case SmoothMethod::Gaussian:
            return SmoothContourGaussian(contour, {sigma, windowSize, AttributeMode::Interpolate});
        case SmoothMethod::MovingAverage:
            return SmoothContourMovingAverage(contour, {windowSize > 0 ? windowSize : DEFAULT_SMOOTH_WINDOW,
                                                         AttributeMode::Interpolate});
        case SmoothMethod::Bilateral:
            return SmoothContourBilateral(contour, {sigma, 30.0, windowSize, AttributeMode::Interpolate});
        default:
            return contour;
    }
}

// =============================================================================
// Simplification Functions Implementation
// =============================================================================

QContour SimplifyContourDouglasPeucker(const QContour& contour, const DouglasPeuckerParams& params) {
    if (contour.Size() < MIN_CONTOUR_POINTS_FOR_SIMPLIFY) {
        return contour;
    }

    double tolerance = std::max(params.tolerance, MIN_SIMPLIFY_TOLERANCE);
    size_t n = contour.Size();
    std::vector<bool> keep(n, false);

    // Always keep first and last points
    keep[0] = true;
    keep[n - 1] = true;

    // Recursive Douglas-Peucker
    std::function<void(size_t, size_t)> simplify = [&](size_t start, size_t end) {
        if (end <= start + 1) return;

        Point2d p1 = contour.GetPoint(start);
        Point2d p2 = contour.GetPoint(end);

        double maxDist = 0.0;
        size_t maxIdx = start;

        for (size_t i = start + 1; i < end; ++i) {
            Point2d p = contour.GetPoint(i);
            double dist = PerpendicularDistance(p, p1, p2);

            if (dist > maxDist) {
                maxDist = dist;
                maxIdx = i;
            }
        }

        if (maxDist > tolerance) {
            keep[maxIdx] = true;
            simplify(start, maxIdx);
            simplify(maxIdx, end);
        }
    };

    // For closed contours, we need special handling
    if (contour.IsClosed()) {
        // Find the point furthest from the line connecting first and last
        // (which are conceptually the same point for closed contours)
        Point2d p0 = contour.GetPoint(0);
        double maxDist = 0.0;
        size_t splitIdx = n / 2;

        for (size_t i = 1; i < n - 1; ++i) {
            Point2d p = contour.GetPoint(i);
            double dist = PointDistance(p, p0);
            if (dist > maxDist) {
                maxDist = dist;
                splitIdx = i;
            }
        }

        keep[splitIdx] = true;
        simplify(0, splitIdx);
        simplify(splitIdx, n - 1);
    } else {
        simplify(0, n - 1);
    }

    // Build result contour
    QContour result;
    result.SetClosed(contour.IsClosed());
    for (size_t i = 0; i < n; ++i) {
        if (keep[i]) {
            result.AddPoint(contour[i]);
        }
    }

    return result;
}

QContour SimplifyContourVisvalingam(const QContour& contour, const VisvalingamParams& params) {
    if (contour.Size() < MIN_CONTOUR_POINTS_FOR_SIMPLIFY) {
        return contour;
    }

    size_t n = contour.Size();
    double minArea = params.minArea;
    size_t minPoints = params.minPoints > 0 ? params.minPoints : 2;

    // Point info structure for linked list and priority queue
    struct PointInfo {
        size_t index;
        double area;
        bool removed = false;
        size_t prevValid;
        size_t nextValid;
    };

    std::vector<PointInfo> points(n);

    // Compute effective area for a point
    auto computeArea = [&](size_t idx) -> double {
        size_t prev = points[idx].prevValid;
        size_t next = points[idx].nextValid;

        if (prev == idx || next == idx) return std::numeric_limits<double>::max();

        Point2d p0 = contour.GetPoint(prev);
        Point2d p1 = contour.GetPoint(idx);
        Point2d p2 = contour.GetPoint(next);

        return TriangleArea(p0, p1, p2);
    };

    // Initialize linked list
    for (size_t i = 0; i < n; ++i) {
        points[i].index = i;
        points[i].prevValid = (i == 0) ? (contour.IsClosed() ? n - 1 : 0) : i - 1;
        points[i].nextValid = (i == n - 1) ? (contour.IsClosed() ? 0 : n - 1) : i + 1;
    }

    // Compute initial areas
    for (size_t i = 0; i < n; ++i) {
        // Don't remove endpoints for open contours
        if (!contour.IsClosed() && (i == 0 || i == n - 1)) {
            points[i].area = std::numeric_limits<double>::max();
        } else {
            points[i].area = computeArea(i);
        }
    }

    // Priority queue (min-heap by area)
    auto cmp = [&](size_t a, size_t b) {
        return points[a].area > points[b].area;
    };
    std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> pq(cmp);

    for (size_t i = 0; i < n; ++i) {
        pq.push(i);
    }

    size_t remaining = n;

    // Remove points iteratively
    while (!pq.empty() && remaining > minPoints) {
        size_t idx = pq.top();
        pq.pop();

        if (points[idx].removed) continue;
        if (points[idx].area >= minArea) break;
        if (points[idx].area == std::numeric_limits<double>::max()) continue;

        // Remove this point
        points[idx].removed = true;
        remaining--;

        // Update neighbors
        size_t prev = points[idx].prevValid;
        size_t next = points[idx].nextValid;

        if (prev != idx) {
            points[prev].nextValid = next;
        }
        if (next != idx) {
            points[next].prevValid = prev;
        }

        // Recompute areas for neighbors and add to queue
        if (prev != idx && !points[prev].removed) {
            points[prev].area = std::max(points[prev].area, computeArea(prev));
            pq.push(prev);
        }
        if (next != idx && !points[next].removed) {
            points[next].area = std::max(points[next].area, computeArea(next));
            pq.push(next);
        }
    }

    // Build result contour
    QContour result;
    result.SetClosed(contour.IsClosed());
    for (size_t i = 0; i < n; ++i) {
        if (!points[i].removed) {
            result.AddPoint(contour[i]);
        }
    }

    return result;
}

QContour SimplifyContourRadialDistance(const QContour& contour, const RadialDistanceParams& params) {
    if (contour.Size() < 2) {
        return contour;
    }

    double tolerance = std::max(params.tolerance, MIN_SIMPLIFY_TOLERANCE);
    double toleranceSq = tolerance * tolerance;

    QContour result;
    result.SetClosed(contour.IsClosed());
    result.AddPoint(contour[0]);

    Point2d lastKept = contour.GetPoint(0);

    for (size_t i = 1; i < contour.Size(); ++i) {
        Point2d current = contour.GetPoint(i);
        double dx = current.x - lastKept.x;
        double dy = current.y - lastKept.y;
        double distSq = dx * dx + dy * dy;

        if (distSq >= toleranceSq || i == contour.Size() - 1) {
            result.AddPoint(contour[i]);
            lastKept = current;
        }
    }

    return result;
}

QContour SimplifyContourNthPoint(const QContour& contour, size_t n) {
    if (n < 2 || contour.Size() < 3) {
        return contour;
    }

    QContour result;
    result.SetClosed(contour.IsClosed());

    // Always include first point
    result.AddPoint(contour[0]);

    // Keep every Nth point
    for (size_t i = n; i < contour.Size() - 1; i += n) {
        result.AddPoint(contour[i]);
    }

    // Always include last point for open contours
    if (!contour.IsClosed() && contour.Size() > 1) {
        size_t lastIdx = contour.Size() - 1;
        // Avoid duplicate if last kept point is the last point
        if (result.Size() == 0 || result.GetPoint(result.Size() - 1).x != contour[lastIdx].x ||
            result.GetPoint(result.Size() - 1).y != contour[lastIdx].y) {
            result.AddPoint(contour[lastIdx]);
        }
    }

    return result;
}

QContour SimplifyContour(const QContour& contour, SimplifyMethod method, double tolerance) {
    switch (method) {
        case SimplifyMethod::DouglasPeucker:
            return SimplifyContourDouglasPeucker(contour, {tolerance});
        case SimplifyMethod::Visvalingam:
            return SimplifyContourVisvalingam(contour, {tolerance, 0});
        case SimplifyMethod::RadialDistance:
            return SimplifyContourRadialDistance(contour, {tolerance});
        case SimplifyMethod::NthPoint:
            return SimplifyContourNthPoint(contour, static_cast<size_t>(std::max(2.0, tolerance)));
        default:
            return contour;
    }
}

// =============================================================================
// Resampling Functions Implementation
// =============================================================================

QContour ResampleContourByDistance(const QContour& contour, const ResampleByDistanceParams& params) {
    if (contour.Size() < 2) {
        return contour;
    }

    double distance = std::max(params.distance, MIN_RESAMPLE_DISTANCE);
    double totalLength = ComputeContourLength(contour);

    if (totalLength < distance) {
        // Contour is too short, return endpoints
        QContour result;
        result.AddPoint(contour[0]);
        if (contour.Size() > 1) {
            result.AddPoint(contour[contour.Size() - 1]);
        }
        result.SetClosed(contour.IsClosed());
        return result;
    }

    // Compute cumulative arc length
    std::vector<double> cumLen = ComputeCumulativeLength(contour);

    QContour result;
    result.AddPoint(contour[0]);  // First point

    double targetLen = distance;
    size_t segIdx = 0;

    while (targetLen < totalLength - EPSILON) {
        // Find segment containing targetLen
        while (segIdx < contour.Size() - 1 && cumLen[segIdx + 1] < targetLen) {
            ++segIdx;
        }

        if (segIdx >= contour.Size() - 1) break;

        // Interpolate within segment
        double segStart = cumLen[segIdx];
        double segEnd = cumLen[segIdx + 1];
        double segLen = segEnd - segStart;
        double localT = (segLen > EPSILON) ? (targetLen - segStart) / segLen : 0.0;

        ContourPoint pt = InterpolateContourPoint(contour[segIdx], contour[segIdx + 1],
                                                   localT, params.attrMode);
        result.AddPoint(pt);

        targetLen += distance;
    }

    // Add last point for open contours
    if (params.preserveEndpoints && !contour.IsClosed()) {
        result.AddPoint(contour[contour.Size() - 1]);
    }

    result.SetClosed(contour.IsClosed());
    return result;
}

QContour ResampleContourByCount(const QContour& contour, const ResampleByCountParams& params) {
    if (contour.Size() < 2 || params.count < 2) {
        return contour;
    }

    double totalLength = ComputeContourLength(contour);

    // For closed contours, we want 'count' points evenly distributed
    // For open contours, we want 'count' points including both endpoints
    size_t numSegments = contour.IsClosed() ? params.count : params.count - 1;
    double segmentLength = totalLength / static_cast<double>(numSegments);

    QContour result;

    // Add first point
    result.AddPoint(contour[0]);

    // Add intermediate points
    for (size_t i = 1; i < (contour.IsClosed() ? params.count : params.count - 1); ++i) {
        double targetLen = static_cast<double>(i) * segmentLength;
        ContourPoint pt = FindPointByArcLength(contour, targetLen, params.attrMode);
        result.AddPoint(pt);
    }

    // Add last point for open contours
    if (!contour.IsClosed() && params.preserveEndpoints) {
        result.AddPoint(contour[contour.Size() - 1]);
    }

    result.SetClosed(contour.IsClosed());
    return result;
}

QContour ResampleContourByArcLength(const QContour& contour, size_t numSegments, AttributeMode attrMode) {
    if (numSegments < 1 || contour.Size() < 2) {
        return contour;
    }

    double totalLength = ComputeContourLength(contour);
    double segmentLength = totalLength / static_cast<double>(numSegments);

    QContour result;

    // For closed contours
    size_t numPoints = contour.IsClosed() ? numSegments : numSegments + 1;

    for (size_t i = 0; i < numPoints; ++i) {
        double targetLen = static_cast<double>(i) * segmentLength;
        ContourPoint pt = FindPointByArcLength(contour, targetLen, attrMode);
        result.AddPoint(pt);
    }

    result.SetClosed(contour.IsClosed());
    return result;
}

QContour ResampleContour(const QContour& contour, ResampleMethod method, double param) {
    switch (method) {
        case ResampleMethod::ByDistance:
            return ResampleContourByDistance(contour, {param, true, AttributeMode::Interpolate});
        case ResampleMethod::ByCount:
            return ResampleContourByCount(contour, {static_cast<size_t>(std::max(2.0, param)),
                                                     true, AttributeMode::Interpolate});
        case ResampleMethod::ByArcLength:
            return ResampleContourByArcLength(contour, static_cast<size_t>(std::max(1.0, param)),
                                               AttributeMode::Interpolate);
        default:
            return contour;
    }
}

// =============================================================================
// Other Processing Functions Implementation
// =============================================================================

QContour ReverseContour(const QContour& contour) {
    if (contour.Size() < 2) {
        return contour;
    }

    QContour result;
    result.Reserve(contour.Size());

    for (size_t i = contour.Size(); i > 0; --i) {
        ContourPoint pt = contour[i - 1];
        // Rotate direction by PI
        pt.direction = NormalizeAngle(pt.direction + PI);
        // Negate curvature (reversed direction means opposite curvature sign)
        pt.curvature = -pt.curvature;
        result.AddPoint(pt);
    }

    result.SetClosed(contour.IsClosed());
    return result;
}

QContour CloseContour(const QContour& contour) {
    QContour result = contour;
    result.SetClosed(true);
    return result;
}

QContour OpenContour(const QContour& contour) {
    QContour result = contour;
    result.SetClosed(false);
    return result;
}

QContour RemoveDuplicatePoints(const QContour& contour, double tolerance) {
    if (contour.Size() < 2) {
        return contour;
    }

    double toleranceSq = tolerance * tolerance;

    QContour result;
    result.AddPoint(contour[0]);

    for (size_t i = 1; i < contour.Size(); ++i) {
        const auto& prev = result[result.Size() - 1];
        const auto& curr = contour[i];

        double dx = curr.x - prev.x;
        double dy = curr.y - prev.y;
        double distSq = dx * dx + dy * dy;

        if (distSq > toleranceSq) {
            result.AddPoint(curr);
        }
    }

    // Check if last point duplicates first point for closed contours
    if (contour.IsClosed() && result.Size() >= 2) {
        const auto& first = result[0];
        const auto& last = result[result.Size() - 1];
        double dx = last.x - first.x;
        double dy = last.y - first.y;
        if (dx * dx + dy * dy <= toleranceSq) {
            result.RemovePoint(result.Size() - 1);
        }
    }

    result.SetClosed(contour.IsClosed());
    return result;
}

QContour RemoveCollinearPoints(const QContour& contour, double tolerance) {
    if (contour.Size() < 3) {
        return contour;
    }

    size_t n = contour.Size();
    std::vector<bool> keep(n, true);

    for (size_t i = 1; i < n - 1; ++i) {
        Point2d prev = contour.GetPoint(i - 1);
        Point2d curr = contour.GetPoint(i);
        Point2d next = contour.GetPoint(i + 1);

        double dist = PerpendicularDistance(curr, prev, next);
        if (dist <= tolerance) {
            keep[i] = false;
        }
    }

    // For closed contours, also check first and last points
    if (contour.IsClosed() && n >= 3) {
        // Check first point
        Point2d prev = contour.GetPoint(n - 1);
        Point2d curr = contour.GetPoint(0);
        Point2d next = contour.GetPoint(1);
        if (PerpendicularDistance(curr, prev, next) <= tolerance) {
            keep[0] = false;
        }

        // Check last point
        prev = contour.GetPoint(n - 2);
        curr = contour.GetPoint(n - 1);
        next = contour.GetPoint(0);
        if (PerpendicularDistance(curr, prev, next) <= tolerance) {
            keep[n - 1] = false;
        }
    }

    QContour result;
    result.SetClosed(contour.IsClosed());
    for (size_t i = 0; i < n; ++i) {
        if (keep[i]) {
            result.AddPoint(contour[i]);
        }
    }

    return result;
}

QContour ShiftContourStart(const QContour& contour, const Point2d& newStart) {
    if (!contour.IsClosed() || contour.Size() < 2) {
        return contour;
    }

    // Find nearest point to newStart
    double minDist = std::numeric_limits<double>::max();
    size_t nearestIdx = 0;

    for (size_t i = 0; i < contour.Size(); ++i) {
        Point2d p = contour.GetPoint(i);
        double dist = PointDistance(p, newStart);
        if (dist < minDist) {
            minDist = dist;
            nearestIdx = i;
        }
    }

    return ShiftContourStartByIndex(contour, nearestIdx);
}

QContour ShiftContourStartByIndex(const QContour& contour, size_t startIndex) {
    if (!contour.IsClosed() || contour.Size() < 2) {
        return contour;
    }

    if (startIndex >= contour.Size()) {
        startIndex = startIndex % contour.Size();
    }

    if (startIndex == 0) {
        return contour;
    }

    QContour result;
    result.Reserve(contour.Size());

    for (size_t i = 0; i < contour.Size(); ++i) {
        size_t idx = (startIndex + i) % contour.Size();
        result.AddPoint(contour[idx]);
    }

    result.SetClosed(true);
    return result;
}

QContour ExtractSubContour(const QContour& contour, double t1, double t2) {
    if (contour.Size() < 2) {
        return contour;
    }

    // Clamp parameters
    t1 = Clamp(t1, 0.0, 1.0);
    t2 = Clamp(t2, 0.0, 1.0);

    // Handle wrap-around for closed contours
    if (t2 < t1 && !contour.IsClosed()) {
        std::swap(t1, t2);
    }

    double totalLen = ComputeContourLength(contour);
    (void)totalLen;  // Used implicitly in FindPointByParameter

    QContour result;

    if (t2 >= t1) {
        // Simple case: extract from t1 to t2
        double step = 0.1;  // Use small steps to preserve shape
        size_t numSteps = static_cast<size_t>((t2 - t1) / step) + 2;
        step = (t2 - t1) / static_cast<double>(numSteps - 1);

        for (size_t i = 0; i < numSteps; ++i) {
            double t = t1 + static_cast<double>(i) * step;
            result.AddPoint(FindPointByParameter(contour, t, AttributeMode::Interpolate));
        }
    } else {
        // Wrap-around case for closed contours
        // Extract from t1 to 1.0, then from 0.0 to t2
        double step = 0.1;

        // t1 to 1.0
        size_t numSteps1 = static_cast<size_t>((1.0 - t1) / step) + 1;
        for (size_t i = 0; i < numSteps1; ++i) {
            double t = t1 + static_cast<double>(i) * step;
            if (t > 1.0) t = 1.0;
            result.AddPoint(FindPointByParameter(contour, t, AttributeMode::Interpolate));
        }

        // 0.0 to t2
        size_t numSteps2 = static_cast<size_t>(t2 / step) + 1;
        for (size_t i = 1; i < numSteps2; ++i) {  // Start from 1 to avoid duplicate at wrap point
            double t = static_cast<double>(i) * step;
            if (t > t2) t = t2;
            result.AddPoint(FindPointByParameter(contour, t, AttributeMode::Interpolate));
        }
    }

    result.SetClosed(false);  // Extracted sub-contour is always open
    return result;
}

QContour ExtractSubContourByIndex(const QContour& contour, size_t startIdx, size_t endIdx) {
    if (contour.Empty()) {
        return QContour();
    }

    if (startIdx >= contour.Size()) startIdx = contour.Size() - 1;
    if (endIdx > contour.Size()) endIdx = contour.Size();
    if (endIdx <= startIdx) {
        QContour result;
        result.AddPoint(contour[startIdx]);
        return result;
    }

    QContour result;
    for (size_t i = startIdx; i < endIdx; ++i) {
        result.AddPoint(contour[i]);
    }

    result.SetClosed(false);
    return result;
}

} // namespace Qi::Vision::Internal
