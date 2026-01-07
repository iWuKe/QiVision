/**
 * @file ContourAnalysis.cpp
 * @brief Implementation of contour geometric property analysis
 */

#include <QiVision/Internal/ContourAnalysis.h>
#include <QiVision/Internal/ContourProcess.h>
#include <QiVision/Internal/Fitting.h>
#include <QiVision/Internal/GeomConstruct.h>
#include <QiVision/Internal/Geometry2d.h>
#include <QiVision/Internal/Distance.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace Qi::Vision::Internal {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

/**
 * @brief Extract points from contour as vector of Point2d
 */
std::vector<Point2d> ExtractPoints(const QContour& contour) {
    std::vector<Point2d> points;
    points.reserve(contour.Size());
    for (size_t i = 0; i < contour.Size(); ++i) {
        const auto& cp = contour[i];
        points.push_back({cp.x, cp.y});
    }
    return points;
}

/**
 * @brief Compute curvature at a point using 3-point method
 */
double ComputeCurvatureThreePoint(const Point2d& p0, const Point2d& p1, const Point2d& p2) {
    // Curvature = 2 * cross / (|p0-p1| * |p1-p2| * |p2-p0|)
    double dx1 = p1.x - p0.x, dy1 = p1.y - p0.y;
    double dx2 = p2.x - p1.x, dy2 = p2.y - p1.y;
    double dx3 = p0.x - p2.x, dy3 = p0.y - p2.y;

    double cross = dx1 * dy2 - dy1 * dx2;  // 2 * signed area

    double a = std::sqrt(dx1 * dx1 + dy1 * dy1);
    double b = std::sqrt(dx2 * dx2 + dy2 * dy2);
    double c = std::sqrt(dx3 * dx3 + dy3 * dy3);

    double denom = a * b * c;
    if (denom < CURVATURE_TOLERANCE) return 0.0;

    return 2.0 * cross / denom;
}

/**
 * @brief Simple average of points
 */
Point2d SimpleAverageOfPoints(const QContour& contour) {
    double sumX = 0, sumY = 0;
    for (size_t i = 0; i < contour.Size(); ++i) {
        sumX += contour[i].x;
        sumY += contour[i].y;
    }
    return {sumX / contour.Size(), sumY / contour.Size()};
}

/**
 * @brief Compute point to segment distance
 */
double PointToSegmentDist(const Point2d& p, const Point2d& a, const Point2d& b) {
    double dx = b.x - a.x;
    double dy = b.y - a.y;
    double lenSq = dx * dx + dy * dy;

    if (lenSq < CURVATURE_TOLERANCE) {
        // Segment is a point
        double px = p.x - a.x;
        double py = p.y - a.y;
        return std::sqrt(px * px + py * py);
    }

    // Project point onto line
    double t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq;
    t = std::max(0.0, std::min(1.0, t));

    double projX = a.x + t * dx;
    double projY = a.y + t * dy;

    double px = p.x - projX;
    double py = p.y - projY;
    return std::sqrt(px * px + py * py);
}

} // anonymous namespace

// =============================================================================
// Basic Property Functions
// =============================================================================

double ContourLength(const QContour& contour) {
    // Delegate to ContourProcess
    return ComputeContourLength(contour);
}

double ContourSignedArea(const QContour& contour) {
    if (contour.Size() < MIN_POINTS_FOR_AREA) {
        return 0.0;
    }

    double area = 0.0;
    size_t n = contour.Size();

    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        const auto& pi = contour[i];
        const auto& pj = contour[j];
        area += pi.x * pj.y - pj.x * pi.y;
    }

    return area * 0.5;
}

double ContourArea(const QContour& contour) {
    return std::abs(ContourSignedArea(contour));
}

double ContourPerimeter(const QContour& contour) {
    return ContourLength(contour);
}

Point2d ContourCentroid(const QContour& contour) {
    if (contour.Empty()) return {0, 0};
    if (contour.Size() < 3) {
        return SimpleAverageOfPoints(contour);
    }

    double area = 0.0;
    double cx = 0.0, cy = 0.0;
    size_t n = contour.Size();

    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        const auto& pi = contour[i];
        const auto& pj = contour[j];
        double cross = pi.x * pj.y - pj.x * pi.y;
        area += cross;
        cx += (pi.x + pj.x) * cross;
        cy += (pi.y + pj.y) * cross;
    }

    if (std::abs(area) < MOMENT_TOLERANCE) {
        return SimpleAverageOfPoints(contour);
    }

    area *= 0.5;
    cx /= (6.0 * area);
    cy /= (6.0 * area);

    return {cx, cy};
}

AreaCenterResult ContourAreaCenter(const QContour& contour) {
    AreaCenterResult result;

    if (contour.Size() < MIN_POINTS_FOR_AREA) {
        if (!contour.Empty()) {
            result.centroid = SimpleAverageOfPoints(contour);
        }
        return result;
    }

    double area = 0.0;
    double cx = 0.0, cy = 0.0;
    size_t n = contour.Size();

    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        const auto& pi = contour[i];
        const auto& pj = contour[j];
        double cross = pi.x * pj.y - pj.x * pi.y;
        area += cross;
        cx += (pi.x + pj.x) * cross;
        cy += (pi.y + pj.y) * cross;
    }

    result.area = area * 0.5;

    if (std::abs(area) < MOMENT_TOLERANCE) {
        result.centroid = SimpleAverageOfPoints(contour);
    } else {
        cx /= (6.0 * result.area);
        cy /= (6.0 * result.area);
        result.centroid = {cx, cy};
    }

    result.valid = true;
    return result;
}

// =============================================================================
// Curvature Analysis Functions
// =============================================================================

std::vector<double> ComputeContourCurvature(const QContour& contour,
                                             CurvatureMethod method,
                                             int32_t windowSize) {
    std::vector<double> curvatures;

    if (contour.Size() < MIN_POINTS_FOR_CURVATURE) {
        curvatures.resize(contour.Size(), 0.0);
        return curvatures;
    }

    size_t n = contour.Size();
    curvatures.resize(n, 0.0);
    bool closed = contour.IsClosed();

    switch (method) {
        case CurvatureMethod::ThreePoint: {
            for (size_t i = 0; i < n; ++i) {
                size_t prev, next;
                if (closed) {
                    prev = (i + n - 1) % n;
                    next = (i + 1) % n;
                } else {
                    if (i == 0 || i == n - 1) {
                        curvatures[i] = 0.0;
                        continue;
                    }
                    prev = i - 1;
                    next = i + 1;
                }

                Point2d p0 = {contour[prev].x, contour[prev].y};
                Point2d p1 = {contour[i].x, contour[i].y};
                Point2d p2 = {contour[next].x, contour[next].y};

                curvatures[i] = ComputeCurvatureThreePoint(p0, p1, p2);
            }
            break;
        }

        case CurvatureMethod::FivePoint: {
            int halfWin = windowSize / 2;
            for (size_t i = 0; i < n; ++i) {
                size_t p0Idx, p2Idx;
                if (closed) {
                    p0Idx = (i + n - halfWin) % n;
                    p2Idx = (i + halfWin) % n;
                } else {
                    if (i < static_cast<size_t>(halfWin) || i >= n - static_cast<size_t>(halfWin)) {
                        curvatures[i] = 0.0;
                        continue;
                    }
                    p0Idx = i - halfWin;
                    p2Idx = i + halfWin;
                }

                Point2d p0 = {contour[p0Idx].x, contour[p0Idx].y};
                Point2d p1 = {contour[i].x, contour[i].y};
                Point2d p2 = {contour[p2Idx].x, contour[p2Idx].y};

                curvatures[i] = ComputeCurvatureThreePoint(p0, p1, p2);
            }
            break;
        }

        case CurvatureMethod::Derivative: {
            // Compute derivatives using finite differences
            for (size_t i = 0; i < n; ++i) {
                size_t prev, next;
                if (closed) {
                    prev = (i + n - 1) % n;
                    next = (i + 1) % n;
                } else {
                    if (i == 0 || i == n - 1) {
                        curvatures[i] = 0.0;
                        continue;
                    }
                    prev = i - 1;
                    next = i + 1;
                }

                // First derivatives
                double dx = (contour[next].x - contour[prev].x) * 0.5;
                double dy = (contour[next].y - contour[prev].y) * 0.5;

                // Second derivatives
                double ddx = contour[next].x - 2.0 * contour[i].x + contour[prev].x;
                double ddy = contour[next].y - 2.0 * contour[i].y + contour[prev].y;

                // Curvature = (x'y'' - x''y') / (x'^2 + y'^2)^1.5
                double denom = std::pow(dx * dx + dy * dy, 1.5);
                if (denom < CURVATURE_TOLERANCE) {
                    curvatures[i] = 0.0;
                } else {
                    curvatures[i] = (dx * ddy - ddx * dy) / denom;
                }
            }
            break;
        }

        case CurvatureMethod::Regression: {
            // Use windowed polynomial regression
            int halfWin = windowSize / 2;
            for (size_t i = 0; i < n; ++i) {
                // Simple approximation: average of neighboring 3-point curvatures
                double sumCurv = 0.0;
                int count = 0;

                for (int k = -halfWin; k <= halfWin; ++k) {
                    size_t idx;
                    if (closed) {
                        idx = (i + n + k) % n;
                    } else {
                        int idx_int = static_cast<int>(i) + k;
                        if (idx_int < 1 || idx_int >= static_cast<int>(n) - 1) continue;
                        idx = static_cast<size_t>(idx_int);
                    }

                    size_t prev = closed ? (idx + n - 1) % n : idx - 1;
                    size_t next = closed ? (idx + 1) % n : idx + 1;

                    if (!closed && (prev >= n || next >= n)) continue;

                    Point2d p0 = {contour[prev].x, contour[prev].y};
                    Point2d p1 = {contour[idx].x, contour[idx].y};
                    Point2d p2 = {contour[next].x, contour[next].y};

                    sumCurv += ComputeCurvatureThreePoint(p0, p1, p2);
                    count++;
                }

                curvatures[i] = (count > 0) ? sumCurv / count : 0.0;
            }
            break;
        }
    }

    return curvatures;
}

double ContourMeanCurvature(const QContour& contour, CurvatureMethod method) {
    std::vector<double> curvatures = ComputeContourCurvature(contour, method);
    if (curvatures.empty()) return 0.0;

    double sum = 0.0;
    for (double k : curvatures) {
        sum += std::abs(k);
    }
    return sum / curvatures.size();
}

double ContourMaxCurvature(const QContour& contour, CurvatureMethod method) {
    std::vector<double> curvatures = ComputeContourCurvature(contour, method);
    if (curvatures.empty()) return 0.0;

    double maxVal = 0.0;
    for (double k : curvatures) {
        maxVal = std::max(maxVal, std::abs(k));
    }
    return maxVal;
}

double ContourMinCurvature(const QContour& contour, CurvatureMethod method) {
    std::vector<double> curvatures = ComputeContourCurvature(contour, method);
    if (curvatures.empty()) return 0.0;

    double minVal = std::numeric_limits<double>::max();
    for (double k : curvatures) {
        minVal = std::min(minVal, std::abs(k));
    }
    return minVal;
}

CurvatureStats ContourCurvatureStats(const QContour& contour, CurvatureMethod method) {
    CurvatureStats stats;

    std::vector<double> curvatures = ComputeContourCurvature(contour, method);
    if (curvatures.empty()) return stats;

    size_t n = curvatures.size();

    // Min, max, mean
    double sum = 0.0;
    stats.min = curvatures[0];
    stats.max = curvatures[0];
    stats.minIndex = 0;
    stats.maxIndex = 0;

    for (size_t i = 0; i < n; ++i) {
        sum += curvatures[i];
        if (curvatures[i] < stats.min) {
            stats.min = curvatures[i];
            stats.minIndex = i;
        }
        if (curvatures[i] > stats.max) {
            stats.max = curvatures[i];
            stats.maxIndex = i;
        }
    }
    stats.mean = sum / n;

    // Standard deviation
    double sumSq = 0.0;
    for (double k : curvatures) {
        double diff = k - stats.mean;
        sumSq += diff * diff;
    }
    stats.stddev = std::sqrt(sumSq / n);

    // Median
    std::vector<double> sorted = curvatures;
    std::sort(sorted.begin(), sorted.end());
    if (n % 2 == 0) {
        stats.median = (sorted[n/2 - 1] + sorted[n/2]) * 0.5;
    } else {
        stats.median = sorted[n/2];
    }

    return stats;
}

std::vector<int32_t> ContourCurvatureHistogram(const QContour& contour,
                                                int32_t numBins,
                                                double minCurvature,
                                                double maxCurvature,
                                                CurvatureMethod method) {
    std::vector<int32_t> histogram(numBins, 0);

    std::vector<double> curvatures = ComputeContourCurvature(contour, method);
    if (curvatures.empty()) return histogram;

    // Auto range if needed
    if (minCurvature >= maxCurvature) {
        auto [minIt, maxIt] = std::minmax_element(curvatures.begin(), curvatures.end());
        minCurvature = *minIt;
        maxCurvature = *maxIt;

        // Add small margin
        double range = maxCurvature - minCurvature;
        if (range < CURVATURE_TOLERANCE) range = 1.0;
        minCurvature -= range * 0.01;
        maxCurvature += range * 0.01;
    }

    double binWidth = (maxCurvature - minCurvature) / numBins;
    if (binWidth < CURVATURE_TOLERANCE) return histogram;

    for (double k : curvatures) {
        int bin = static_cast<int>((k - minCurvature) / binWidth);
        bin = std::max(0, std::min(numBins - 1, bin));
        histogram[bin]++;
    }

    return histogram;
}

// =============================================================================
// Orientation Functions
// =============================================================================

double ContourOrientation(const QContour& contour) {
    PrincipalAxesResult axes = ContourPrincipalAxes(contour);
    return axes.valid ? axes.angle : 0.0;
}

double ContourOrientationEllipse(const QContour& contour) {
    if (contour.Size() < 5) return 0.0;

    std::vector<Point2d> points = ExtractPoints(contour);
    EllipseFitResult result = FitEllipseFitzgibbon(points);

    if (!result.success) return 0.0;
    return result.ellipse.angle;
}

PrincipalAxesResult ContourPrincipalAxes(const QContour& contour) {
    PrincipalAxesResult result;

    if (contour.Size() < 2) return result;

    // Compute centroid
    result.centroid = ContourCentroid(contour);

    // Compute central moments (covariance matrix)
    double mu20 = 0.0, mu11 = 0.0, mu02 = 0.0;

    for (size_t i = 0; i < contour.Size(); ++i) {
        double dx = contour[i].x - result.centroid.x;
        double dy = contour[i].y - result.centroid.y;
        mu20 += dx * dx;
        mu11 += dx * dy;
        mu02 += dy * dy;
    }

    mu20 /= contour.Size();
    mu11 /= contour.Size();
    mu02 /= contour.Size();

    // Eigenanalysis of covariance matrix [[mu20, mu11], [mu11, mu02]]
    // Eigenvalues: (mu20 + mu02) / 2 +/- sqrt(((mu20 - mu02)/2)^2 + mu11^2)
    double trace = mu20 + mu02;
    double diff = mu20 - mu02;
    double disc = std::sqrt(diff * diff * 0.25 + mu11 * mu11);

    double lambda1 = trace * 0.5 + disc;  // Major eigenvalue
    double lambda2 = trace * 0.5 - disc;  // Minor eigenvalue

    if (lambda1 < MOMENT_TOLERANCE) {
        // Degenerate case
        result.angle = 0.0;
        result.majorLength = 0.0;
        result.minorLength = 0.0;
        result.majorAxis = {1.0, 0.0};
        result.minorAxis = {0.0, 1.0};
        result.valid = true;
        return result;
    }

    // Principal angle
    if (std::abs(mu11) < MOMENT_TOLERANCE && std::abs(diff) < MOMENT_TOLERANCE) {
        result.angle = 0.0;
    } else {
        result.angle = 0.5 * std::atan2(2.0 * mu11, diff);
    }

    // Axis lengths (proportional to sqrt of eigenvalues)
    result.majorLength = 2.0 * std::sqrt(std::max(0.0, lambda1));
    result.minorLength = 2.0 * std::sqrt(std::max(0.0, lambda2));

    // Unit vectors
    result.majorAxis = {std::cos(result.angle), std::sin(result.angle)};
    result.minorAxis = {-std::sin(result.angle), std::cos(result.angle)};

    result.valid = true;
    return result;
}

// =============================================================================
// Moment Functions
// =============================================================================

MomentsResult ContourMoments(const QContour& contour) {
    MomentsResult result;

    if (contour.Empty()) return result;

    // Use polygon formula for closed contours
    size_t n = contour.Size();

    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        double xi = contour[i].x, yi = contour[i].y;
        double xj = contour[j].x, yj = contour[j].y;

        double cross = xi * yj - xj * yi;

        // m00 (area)
        result.m00 += cross;

        // m10, m01
        result.m10 += (xi + xj) * cross;
        result.m01 += (yi + yj) * cross;

        // m20, m11, m02
        result.m20 += (xi * xi + xi * xj + xj * xj) * cross;
        result.m11 += (2.0 * xi * yi + xi * yj + xj * yi + 2.0 * xj * yj) * cross;
        result.m02 += (yi * yi + yi * yj + yj * yj) * cross;

        // m30, m21, m12, m03
        double xi2 = xi * xi, xj2 = xj * xj;
        double yi2 = yi * yi, yj2 = yj * yj;

        result.m30 += (xi2 * xi + xi2 * xj + xi * xj2 + xj2 * xj) * cross;
        result.m21 += (xi2 * yi + 2.0 * xi * xj * yi + xj2 * yi +
                       xi2 * yj + 2.0 * xi * xj * yj + xj2 * yj) * cross;
        result.m12 += (xi * yi2 + 2.0 * xi * yi * yj + xi * yj2 +
                       xj * yi2 + 2.0 * xj * yi * yj + xj * yj2) * cross;
        result.m03 += (yi2 * yi + yi2 * yj + yi * yj2 + yj2 * yj) * cross;
    }

    // Apply normalization factors
    result.m00 /= 2.0;
    result.m10 /= 6.0;
    result.m01 /= 6.0;
    result.m20 /= 12.0;
    result.m11 /= 24.0;
    result.m02 /= 12.0;
    result.m30 /= 20.0;
    result.m21 /= 60.0;
    result.m12 /= 60.0;
    result.m03 /= 20.0;

    // Handle negative area (clockwise orientation)
    if (result.m00 < 0) {
        result.m00 = -result.m00;
        result.m10 = -result.m10;
        result.m01 = -result.m01;
        result.m20 = -result.m20;
        result.m11 = -result.m11;
        result.m02 = -result.m02;
        result.m30 = -result.m30;
        result.m21 = -result.m21;
        result.m12 = -result.m12;
        result.m03 = -result.m03;
    }

    return result;
}

CentralMomentsResult ContourCentralMoments(const QContour& contour) {
    CentralMomentsResult result;

    MomentsResult m = ContourMoments(contour);
    result.mu00 = m.m00;

    if (m.m00 < MOMENT_TOLERANCE) return result;

    result.centroid = m.Centroid();
    double cx = result.centroid.x;
    double cy = result.centroid.y;

    // Central moments from raw moments (using standard formulas)
    // Reference: OpenCV imgproc/moments.cpp
    result.mu20 = m.m20 - cx * m.m10;
    result.mu11 = m.m11 - cx * m.m01;  // or equivalently: m11 - cy * m10
    result.mu02 = m.m02 - cy * m.m01;

    result.mu30 = m.m30 - 3.0 * cx * m.m20 + 2.0 * cx * cx * m.m10;
    result.mu21 = m.m21 - 2.0 * cx * m.m11 - cy * m.m20 + 2.0 * cx * cy * m.m10;
    result.mu12 = m.m12 - 2.0 * cy * m.m11 - cx * m.m02 + 2.0 * cx * cy * m.m01;
    result.mu03 = m.m03 - 3.0 * cy * m.m02 + 2.0 * cy * cy * m.m01;

    return result;
}

NormalizedMomentsResult ContourNormalizedMoments(const QContour& contour) {
    NormalizedMomentsResult result;

    CentralMomentsResult cm = ContourCentralMoments(contour);

    if (cm.mu00 < MOMENT_TOLERANCE) return result;

    // eta_pq = mu_pq / mu00^((p+q)/2 + 1)
    double factor2 = std::pow(cm.mu00, 2.0);    // for p+q=2
    double factor3 = std::pow(cm.mu00, 2.5);    // for p+q=3

    result.eta20 = cm.mu20 / factor2;
    result.eta11 = cm.mu11 / factor2;
    result.eta02 = cm.mu02 / factor2;

    result.eta30 = cm.mu30 / factor3;
    result.eta21 = cm.mu21 / factor3;
    result.eta12 = cm.mu12 / factor3;
    result.eta03 = cm.mu03 / factor3;

    return result;
}

HuMomentsResult ContourHuMoments(const QContour& contour) {
    HuMomentsResult result;

    NormalizedMomentsResult nm = ContourNormalizedMoments(contour);

    double eta20 = nm.eta20, eta02 = nm.eta02, eta11 = nm.eta11;
    double eta30 = nm.eta30, eta21 = nm.eta21, eta12 = nm.eta12, eta03 = nm.eta03;

    // h1 = eta20 + eta02
    result.hu[0] = eta20 + eta02;

    // h2 = (eta20 - eta02)^2 + 4*eta11^2
    double diff20_02 = eta20 - eta02;
    result.hu[1] = diff20_02 * diff20_02 + 4.0 * eta11 * eta11;

    // h3 = (eta30 - 3*eta12)^2 + (3*eta21 - eta03)^2
    double t1 = eta30 - 3.0 * eta12;
    double t2 = 3.0 * eta21 - eta03;
    result.hu[2] = t1 * t1 + t2 * t2;

    // h4 = (eta30 + eta12)^2 + (eta21 + eta03)^2
    double s1 = eta30 + eta12;
    double s2 = eta21 + eta03;
    result.hu[3] = s1 * s1 + s2 * s2;

    // Intermediate values for h5, h6, h7
    double s1_sq = s1 * s1;
    double s2_sq = s2 * s2;

    // h5 = (eta30 - 3*eta12)(eta30 + eta12)[(eta30 + eta12)^2 - 3(eta21 + eta03)^2]
    //    + (3*eta21 - eta03)(eta21 + eta03)[3(eta30 + eta12)^2 - (eta21 + eta03)^2]
    result.hu[4] = t1 * s1 * (s1_sq - 3.0 * s2_sq) + t2 * s2 * (3.0 * s1_sq - s2_sq);

    // h6 = (eta20 - eta02)[(eta30 + eta12)^2 - (eta21 + eta03)^2]
    //    + 4*eta11*(eta30 + eta12)(eta21 + eta03)
    result.hu[5] = diff20_02 * (s1_sq - s2_sq) + 4.0 * eta11 * s1 * s2;

    // h7 = (3*eta21 - eta03)(eta30 + eta12)[(eta30 + eta12)^2 - 3(eta21 + eta03)^2]
    //    - (eta30 - 3*eta12)(eta21 + eta03)[3(eta30 + eta12)^2 - (eta21 + eta03)^2]
    result.hu[6] = t2 * s1 * (s1_sq - 3.0 * s2_sq) - t1 * s2 * (3.0 * s1_sq - s2_sq);

    return result;
}

// =============================================================================
// Shape Descriptor Functions
// =============================================================================

double ContourCircularity(const QContour& contour) {
    if (contour.Size() < MIN_POINTS_FOR_AREA) return 0.0;

    double area = ContourArea(contour);
    double perimeter = ContourPerimeter(contour);

    if (perimeter < MOMENT_TOLERANCE) return 0.0;

    return 4.0 * PI * area / (perimeter * perimeter);
}

double ContourCompactness(const QContour& contour) {
    if (contour.Size() < MIN_POINTS_FOR_AREA) return 0.0;

    double area = ContourArea(contour);
    double perimeter = ContourPerimeter(contour);

    if (area < MOMENT_TOLERANCE) return 0.0;

    return perimeter * perimeter / area;
}

double ContourConvexity(const QContour& contour) {
    if (contour.Size() < MIN_POINTS_FOR_CONVEX_HULL) return 0.0;

    double contourPerimeter = ContourPerimeter(contour);
    if (contourPerimeter < MOMENT_TOLERANCE) return 0.0;

    QContour hull = ContourConvexHull(contour);
    double hullPerimeter = ContourPerimeter(hull);

    return hullPerimeter / contourPerimeter;
}

double ContourSolidity(const QContour& contour) {
    if (contour.Size() < MIN_POINTS_FOR_AREA) return 0.0;

    double contourArea = ContourArea(contour);

    QContour hull = ContourConvexHull(contour);
    double hullArea = ContourArea(hull);

    if (hullArea < MOMENT_TOLERANCE) return 0.0;

    return contourArea / hullArea;
}

double ContourEccentricity(const QContour& contour) {
    PrincipalAxesResult axes = ContourPrincipalAxes(contour);

    if (!axes.valid || axes.majorLength < MOMENT_TOLERANCE) return 0.0;

    double ratio = axes.minorLength / axes.majorLength;
    return std::sqrt(1.0 - ratio * ratio);
}

double ContourElongation(const QContour& contour) {
    PrincipalAxesResult axes = ContourPrincipalAxes(contour);

    if (!axes.valid || axes.majorLength < MOMENT_TOLERANCE) return 0.0;

    return 1.0 - axes.minorLength / axes.majorLength;
}

double ContourRectangularity(const QContour& contour) {
    if (contour.Size() < MIN_POINTS_FOR_AREA) return 0.0;

    double contourArea = ContourArea(contour);
    auto minRect = ContourMinAreaRect(contour);

    if (!minRect.has_value()) return 0.0;

    double rectArea = minRect->Area();
    if (rectArea < MOMENT_TOLERANCE) return 0.0;

    return contourArea / rectArea;
}

double ContourExtent(const QContour& contour) {
    if (contour.Size() < MIN_POINTS_FOR_AREA) return 0.0;

    double contourArea = ContourArea(contour);
    Rect2d bbox = ContourBoundingBox(contour);

    double bboxArea = bbox.Area();
    if (bboxArea < MOMENT_TOLERANCE) return 0.0;

    return contourArea / bboxArea;
}

double ContourAspectRatio(const QContour& contour) {
    PrincipalAxesResult axes = ContourPrincipalAxes(contour);

    if (!axes.valid || axes.minorLength < MOMENT_TOLERANCE) return 1.0;

    return axes.majorLength / axes.minorLength;
}

ShapeDescriptors ContourAllDescriptors(const QContour& contour) {
    ShapeDescriptors desc;

    if (contour.Size() < MIN_POINTS_FOR_AREA) {
        return desc;
    }

    // Basic properties
    double area = ContourArea(contour);
    double perimeter = ContourPerimeter(contour);

    if (area < MOMENT_TOLERANCE || perimeter < MOMENT_TOLERANCE) {
        return desc;
    }

    // Circularity
    desc.circularity = 4.0 * PI * area / (perimeter * perimeter);

    // Compactness
    desc.compactness = perimeter * perimeter / area;

    // Convex hull related
    QContour hull = ContourConvexHull(contour);
    double hullPerimeter = ContourPerimeter(hull);
    double hullArea = ContourArea(hull);

    desc.convexity = (hullPerimeter > MOMENT_TOLERANCE) ? (hullPerimeter / perimeter) : 0.0;
    desc.solidity = (hullArea > MOMENT_TOLERANCE) ? (area / hullArea) : 0.0;

    // Principal axes related
    PrincipalAxesResult axes = ContourPrincipalAxes(contour);
    if (axes.valid && axes.majorLength > MOMENT_TOLERANCE && axes.minorLength > MOMENT_TOLERANCE) {
        double ratio = axes.minorLength / axes.majorLength;
        desc.eccentricity = std::sqrt(1.0 - ratio * ratio);
        desc.elongation = 1.0 - ratio;
        desc.aspectRatio = axes.majorLength / axes.minorLength;
    } else {
        desc.aspectRatio = 1.0;
    }

    // Rectangularity
    auto minRect = ContourMinAreaRect(contour);
    if (minRect.has_value()) {
        double rectArea = minRect->Area();
        desc.rectangularity = (rectArea > MOMENT_TOLERANCE) ? (area / rectArea) : 0.0;
    }

    // Extent
    Rect2d bbox = ContourBoundingBox(contour);
    double bboxArea = bbox.Area();
    desc.extent = (bboxArea > MOMENT_TOLERANCE) ? (area / bboxArea) : 0.0;

    desc.valid = true;
    return desc;
}

// =============================================================================
// Bounding Geometry Functions
// =============================================================================

Rect2d ContourBoundingBox(const QContour& contour) {
    if (contour.Empty()) {
        return Rect2d(0, 0, 0, 0);
    }

    double minX = contour[0].x, maxX = contour[0].x;
    double minY = contour[0].y, maxY = contour[0].y;

    for (size_t i = 1; i < contour.Size(); ++i) {
        minX = std::min(minX, contour[i].x);
        maxX = std::max(maxX, contour[i].x);
        minY = std::min(minY, contour[i].y);
        maxY = std::max(maxY, contour[i].y);
    }

    return Rect2d(minX, minY, maxX - minX, maxY - minY);
}

std::optional<RotatedRect2d> ContourMinAreaRect(const QContour& contour) {
    if (contour.Size() < 3) return std::nullopt;

    std::vector<Point2d> points = ExtractPoints(contour);
    return MinAreaRect(points);
}

std::optional<Circle2d> ContourMinEnclosingCircle(const QContour& contour) {
    if (contour.Empty()) return std::nullopt;

    std::vector<Point2d> points = ExtractPoints(contour);
    return MinEnclosingCircle(points);
}

std::optional<Ellipse2d> ContourMinEnclosingEllipse(const QContour& contour) {
    if (contour.Size() < 5) return std::nullopt;

    std::vector<Point2d> points = ExtractPoints(contour);
    EllipseFitResult result = FitEllipseFitzgibbon(points);

    if (!result.success) return std::nullopt;
    return result.ellipse;
}

// =============================================================================
// Convexity Analysis Functions
// =============================================================================

QContour ContourConvexHull(const QContour& contour) {
    if (contour.Size() < MIN_POINTS_FOR_CONVEX_HULL) {
        return contour;
    }

    std::vector<Point2d> points = ExtractPoints(contour);
    std::vector<Point2d> hull = ConvexHull(points);

    QContour result;
    for (const auto& pt : hull) {
        result.AddPoint({pt.x, pt.y, 0.0, 0.0, 0.0});
    }
    result.SetClosed(true);

    return result;
}

double ContourConvexHullArea(const QContour& contour) {
    QContour hull = ContourConvexHull(contour);
    return ContourArea(hull);
}

bool IsContourConvex(const QContour& contour) {
    if (contour.Size() < 3) return true;

    std::vector<Point2d> points = ExtractPoints(contour);
    return IsConvex(points);
}

std::vector<ConvexityDefect> ContourConvexityDefects(const QContour& contour, double minDepth) {
    std::vector<ConvexityDefect> defects;

    if (contour.Size() < MIN_POINTS_FOR_CONVEX_HULL) {
        return defects;
    }

    // Get convex hull indices
    std::vector<Point2d> points = ExtractPoints(contour);
    std::vector<size_t> hullIndices = ConvexHullIndices(points);

    if (hullIndices.size() < 3) {
        return defects;
    }

    size_t n = contour.Size();

    // For each pair of adjacent hull points
    for (size_t i = 0; i < hullIndices.size(); ++i) {
        size_t startIdx = hullIndices[i];
        size_t endIdx = hullIndices[(i + 1) % hullIndices.size()];

        Point2d startPt = {contour[startIdx].x, contour[startIdx].y};
        Point2d endPt = {contour[endIdx].x, contour[endIdx].y};

        // Find deepest point between start and end on contour
        double maxDist = 0.0;
        size_t deepestIdx = startIdx;

        // Traverse from startIdx to endIdx along the contour
        size_t idx = (startIdx + 1) % n;
        while (idx != endIdx) {
            Point2d pt = {contour[idx].x, contour[idx].y};
            double dist = PointToSegmentDist(pt, startPt, endPt);

            if (dist > maxDist) {
                maxDist = dist;
                deepestIdx = idx;
            }

            idx = (idx + 1) % n;
        }

        if (maxDist >= minDepth) {
            ConvexityDefect defect;
            defect.startIndex = startIdx;
            defect.endIndex = endIdx;
            defect.deepestIndex = deepestIdx;
            defect.startPoint = startPt;
            defect.endPoint = endPt;
            defect.deepestPoint = {contour[deepestIdx].x, contour[deepestIdx].y};
            defect.depth = maxDist;
            defects.push_back(defect);
        }
    }

    return defects;
}

// =============================================================================
// Shape Comparison Functions
// =============================================================================

double MatchShapesHu(const QContour& contour1, const QContour& contour2, int method) {
    HuMomentsResult hu1 = ContourHuMoments(contour1);
    HuMomentsResult hu2 = ContourHuMoments(contour2);

    double result = 0.0;

    // Apply sign and log transform for better comparison
    std::array<double, 7> m1, m2;
    for (int i = 0; i < 7; ++i) {
        double val1 = hu1[i];
        double val2 = hu2[i];

        // Sign-preserving log transform
        if (std::abs(val1) > MOMENT_TOLERANCE) {
            m1[i] = (val1 > 0 ? 1.0 : -1.0) * std::log10(std::abs(val1));
        } else {
            m1[i] = 0.0;
        }

        if (std::abs(val2) > MOMENT_TOLERANCE) {
            m2[i] = (val2 > 0 ? 1.0 : -1.0) * std::log10(std::abs(val2));
        } else {
            m2[i] = 0.0;
        }
    }

    switch (method) {
        case 1: {
            // sum(|1/m1_i - 1/m2_i|)
            for (int i = 0; i < 7; ++i) {
                double inv1 = (std::abs(m1[i]) > MOMENT_TOLERANCE) ? 1.0 / m1[i] : 0.0;
                double inv2 = (std::abs(m2[i]) > MOMENT_TOLERANCE) ? 1.0 / m2[i] : 0.0;
                result += std::abs(inv1 - inv2);
            }
            break;
        }
        case 2: {
            // sum(|m1_i - m2_i|)
            for (int i = 0; i < 7; ++i) {
                result += std::abs(m1[i] - m2[i]);
            }
            break;
        }
        case 3: {
            // max(|m1_i - m2_i| / |m1_i|)
            for (int i = 0; i < 7; ++i) {
                if (std::abs(m1[i]) > MOMENT_TOLERANCE) {
                    double r = std::abs(m1[i] - m2[i]) / std::abs(m1[i]);
                    result = std::max(result, r);
                }
            }
            break;
        }
        default:
            // Default to method 2
            for (int i = 0; i < 7; ++i) {
                result += std::abs(m1[i] - m2[i]);
            }
            break;
    }

    return result;
}

double MatchShapesContour(const QContour& contour1, const QContour& contour2) {
    // Simple implementation using Hu moment similarity
    // Could be extended with shape context or Fourier descriptors

    double huDist = MatchShapesHu(contour1, contour2, 2);

    // Convert distance to similarity [0, 1]
    return 1.0 / (1.0 + huDist);
}

} // namespace Qi::Vision::Internal
