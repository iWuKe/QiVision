/**
 * @file Fitting.cpp
 * @brief Implementation of geometric fitting algorithms
 */

#include <QiVision/Internal/Fitting.h>
#include <QiVision/Platform/Random.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace Qi::Vision::Internal {

// =============================================================================
// Helper Functions (anonymous namespace)
// =============================================================================

namespace {

/// Compute median of a vector (modifies input)
double Median(std::vector<double>& values) {
    if (values.empty()) return 0.0;

    size_t n = values.size();
    size_t mid = n / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());

    if (n % 2 == 0) {
        double upper = values[mid];
        std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
        return (values[mid - 1] + upper) / 2.0;
    }
    return values[mid];
}

/// Compute percentile of a vector
double Percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;

    std::sort(values.begin(), values.end());
    double idx = p * (values.size() - 1);
    size_t lower = static_cast<size_t>(idx);
    size_t upper = std::min(lower + 1, values.size() - 1);
    double frac = idx - lower;
    return values[lower] * (1.0 - frac) + values[upper] * frac;
}

/// Fill result statistics from residuals
void FillResidualStats(FitResultBase& result, const std::vector<double>& residuals,
                       const FitParams& params) {
    result.numPoints = static_cast<int>(residuals.size());

    if (residuals.empty()) {
        return;
    }

    // Compute statistics
    double sumAbs = 0.0;
    double sumSq = 0.0;
    double maxAbs = 0.0;

    for (double r : residuals) {
        double absR = std::abs(r);
        sumAbs += absR;
        sumSq += r * r;
        maxAbs = std::max(maxAbs, absR);
    }

    int n = static_cast<int>(residuals.size());
    result.residualMean = sumAbs / n;
    result.residualRMS = std::sqrt(sumSq / n);
    result.residualMax = maxAbs;

    // Standard deviation
    double sumSqDev = 0.0;
    for (double r : residuals) {
        double absR = std::abs(r);
        double dev = absR - result.residualMean;
        sumSqDev += dev * dev;
    }
    result.residualStd = std::sqrt(sumSqDev / n);

    if (params.computeResiduals) {
        result.residuals = residuals;
    }
}

} // anonymous namespace

// =============================================================================
// Robust Scale Estimators
// =============================================================================

double RobustScaleMAD(const std::vector<double>& residuals) {
    if (residuals.empty()) return 0.0;

    // Compute median
    std::vector<double> values = residuals;
    for (auto& v : values) v = std::abs(v);
    double med = Median(values);

    // Compute MAD (Median Absolute Deviation)
    for (auto& v : values) v = std::abs(v - med);
    double mad = Median(values);

    // Scale factor for consistency with Gaussian (MAD / 0.6745)
    return mad / 0.6745;
}

double RobustScaleIQR(const std::vector<double>& residuals) {
    if (residuals.size() < 4) {
        return RobustScaleMAD(residuals);
    }

    std::vector<double> absRes;
    absRes.reserve(residuals.size());
    for (double r : residuals) {
        absRes.push_back(std::abs(r));
    }

    double q1 = Percentile(absRes, 0.25);
    double q3 = Percentile(absRes, 0.75);
    double iqr = q3 - q1;

    // Scale factor for consistency with Gaussian
    return iqr / 1.349;
}

// =============================================================================
// Utility Functions
// =============================================================================

Point2d ComputeCentroid(const std::vector<Point2d>& points) {
    if (points.empty()) {
        return Point2d(0.0, 0.0);
    }

    double sx = 0.0, sy = 0.0;
    for (const auto& p : points) {
        sx += p.x;
        sy += p.y;
    }

    int n = static_cast<int>(points.size());
    return Point2d(sx / n, sy / n);
}

Point2d ComputeWeightedCentroid(const std::vector<Point2d>& points,
                                 const std::vector<double>& weights) {
    if (points.empty() || weights.size() != points.size()) {
        return ComputeCentroid(points);
    }

    double sx = 0.0, sy = 0.0, sw = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        double w = weights[i];
        sx += w * points[i].x;
        sy += w * points[i].y;
        sw += w;
    }

    if (sw < 1e-15) {
        return ComputeCentroid(points);
    }

    return Point2d(sx / sw, sy / sw);
}

bool ArePointsCollinear(const std::vector<Point2d>& points, double tolerance) {
    if (points.size() < 3) {
        return true;  // 2 or fewer points are always collinear
    }

    // Fit line and check maximum residual
    auto result = FitLine(points);
    if (!result.success) {
        return true;  // Degenerate case
    }

    return result.residualMax < tolerance;
}

std::pair<std::vector<Point2d>, Mat33> NormalizePoints(const std::vector<Point2d>& points) {
    int n = static_cast<int>(points.size());

    if (n == 0) {
        return {{}, Mat33::Identity()};
    }

    // Compute centroid
    Point2d centroid = ComputeCentroid(points);

    // Compute RMS distance from centroid
    double sumSqDist = 0.0;
    for (const auto& p : points) {
        double dx = p.x - centroid.x;
        double dy = p.y - centroid.y;
        sumSqDist += dx * dx + dy * dy;
    }

    double rmsDist = std::sqrt(sumSqDist / n);
    double scale = (rmsDist > 1e-15) ? std::sqrt(2.0) / rmsDist : 1.0;

    // Build normalization matrix: T = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]]
    Mat33 T;
    T(0, 0) = scale;  T(0, 1) = 0.0;    T(0, 2) = -scale * centroid.x;
    T(1, 0) = 0.0;    T(1, 1) = scale;  T(1, 2) = -scale * centroid.y;
    T(2, 0) = 0.0;    T(2, 1) = 0.0;    T(2, 2) = 1.0;

    // Normalize points
    std::vector<Point2d> normalized;
    normalized.reserve(n);
    for (const auto& p : points) {
        normalized.emplace_back(
            scale * (p.x - centroid.x),
            scale * (p.y - centroid.y)
        );
    }

    return {normalized, T};
}

Line2d DenormalizeLine(const Line2d& normalizedLine, const Mat33& T) {
    // Line l' = [a', b', c'] in normalized coords
    // Original line l = T^T * l'
    // l = [a*s, b*s, c - a*s*cx - b*s*cy]

    double scale = T(0, 0);  // s
    double tx = -T(0, 2) / scale;  // cx
    double ty = -T(1, 2) / scale;  // cy

    double a = normalizedLine.a * scale;
    double b = normalizedLine.b * scale;
    double c = normalizedLine.c - normalizedLine.a * scale * tx - normalizedLine.b * scale * ty;

    // Normalize
    double norm = std::sqrt(a * a + b * b);
    if (norm > 1e-15) {
        a /= norm;
        b /= norm;
        c /= norm;
    }

    return Line2d(a, b, c);
}

Circle2d DenormalizeCircle(const Circle2d& normalizedCircle, const Mat33& T) {
    double scale = T(0, 0);
    double tx = -T(0, 2) / scale;
    double ty = -T(1, 2) / scale;

    double cx = normalizedCircle.center.x / scale + tx;
    double cy = normalizedCircle.center.y / scale + ty;
    double r = normalizedCircle.radius / scale;

    return Circle2d(Point2d(cx, cy), r);
}

Ellipse2d DenormalizeEllipse(const Ellipse2d& normalizedEllipse, const Mat33& T) {
    double scale = T(0, 0);
    double tx = -T(0, 2) / scale;
    double ty = -T(1, 2) / scale;

    double cx = normalizedEllipse.center.x / scale + tx;
    double cy = normalizedEllipse.center.y / scale + ty;
    double a = normalizedEllipse.a / scale;
    double b = normalizedEllipse.b / scale;

    return Ellipse2d(Point2d(cx, cy), a, b, normalizedEllipse.angle);
}

// =============================================================================
// Residual Computation
// =============================================================================

std::vector<double> ComputeLineResiduals(const std::vector<Point2d>& points,
                                          const Line2d& line) {
    std::vector<double> residuals;
    residuals.reserve(points.size());

    double norm = std::sqrt(line.a * line.a + line.b * line.b);
    if (norm < 1e-15) norm = 1.0;

    for (const auto& p : points) {
        double dist = (line.a * p.x + line.b * p.y + line.c) / norm;
        residuals.push_back(dist);
    }

    return residuals;
}

std::vector<double> ComputeCircleResiduals(const std::vector<Point2d>& points,
                                            const Circle2d& circle) {
    std::vector<double> residuals;
    residuals.reserve(points.size());

    for (const auto& p : points) {
        double dx = p.x - circle.center.x;
        double dy = p.y - circle.center.y;
        double dist = std::sqrt(dx * dx + dy * dy) - circle.radius;
        residuals.push_back(dist);
    }

    return residuals;
}

std::vector<double> ComputeEllipseResiduals(const std::vector<Point2d>& points,
                                             const Ellipse2d& ellipse) {
    std::vector<double> residuals;
    residuals.reserve(points.size());

    double cosA = std::cos(-ellipse.angle);
    double sinA = std::sin(-ellipse.angle);

    for (const auto& p : points) {
        // Transform point to ellipse-centered coordinates
        double dx = p.x - ellipse.center.x;
        double dy = p.y - ellipse.center.y;

        // Rotate to align with ellipse axes
        double x = dx * cosA - dy * sinA;
        double y = dx * sinA + dy * cosA;

        // Algebraic distance (approximate)
        double a2 = ellipse.a * ellipse.a;
        double b2 = ellipse.b * ellipse.b;
        double val = (x * x) / a2 + (y * y) / b2 - 1.0;

        // Approximate geometric distance
        double dist = val * std::sqrt(a2 * b2) / std::sqrt(b2 * x * x + a2 * y * y + 1e-15);
        residuals.push_back(dist);
    }

    return residuals;
}

void ComputeResidualStats(const std::vector<double>& residuals,
                           double& mean, double& stdDev, double& maxAbs, double& rms) {
    mean = stdDev = maxAbs = rms = 0.0;

    if (residuals.empty()) return;

    int n = static_cast<int>(residuals.size());
    double sumAbs = 0.0, sumSq = 0.0;

    for (double r : residuals) {
        double absR = std::abs(r);
        sumAbs += absR;
        sumSq += r * r;
        maxAbs = std::max(maxAbs, absR);
    }

    mean = sumAbs / n;
    rms = std::sqrt(sumSq / n);

    double sumSqDev = 0.0;
    for (double r : residuals) {
        double dev = std::abs(r) - mean;
        sumSqDev += dev * dev;
    }
    stdDev = std::sqrt(sumSqDev / n);
}

// =============================================================================
// Line Fitting Implementation
// =============================================================================

LineFitResult FitLine(const std::vector<Point2d>& points, const FitParams& params) {
    LineFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < LINE_FIT_MIN_POINTS) {
        return result;
    }

    // Compute centroid
    Point2d centroid = ComputeCentroid(points);

    // Build covariance matrix
    double sxx = 0.0, sxy = 0.0, syy = 0.0;
    for (const auto& p : points) {
        double dx = p.x - centroid.x;
        double dy = p.y - centroid.y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }

    // Eigendecomposition of 2x2 covariance matrix
    // The eigenvector for smaller eigenvalue gives line normal
    // For M = [[sxx, sxy], [sxy, syy]]
    // eigenvalues: lambda = (trace +/- sqrt(trace^2 - 4*det)) / 2

    double trace = sxx + syy;
    double det = sxx * syy - sxy * sxy;
    double disc = trace * trace - 4.0 * det;

    if (disc < 0) disc = 0;
    double sqrtDisc = std::sqrt(disc);

    double lambda1 = (trace + sqrtDisc) / 2.0;
    double lambda2 = (trace - sqrtDisc) / 2.0;

    // Eigenvector for smaller eigenvalue (lambda2)
    double a, b;
    if (std::abs(sxy) > 1e-15) {
        // Normal direction from smaller eigenvalue
        a = lambda2 - syy;
        b = sxy;
    } else if (sxx < syy) {
        a = 1.0;
        b = 0.0;
    } else {
        a = 0.0;
        b = 1.0;
    }

    // Normalize
    double norm = std::sqrt(a * a + b * b);
    if (norm < 1e-15) {
        return result;
    }
    a /= norm;
    b /= norm;

    // Compute c: ax + by + c = 0 passes through centroid
    double c = -(a * centroid.x + b * centroid.y);

    result.line = Line2d(a, b, c);
    result.success = true;

    // Compute residuals and statistics
    auto residuals = ComputeLineResiduals(points, result.line);
    FillResidualStats(result, residuals, params);
    result.numInliers = result.numPoints;

    return result;
}

LineFitResult FitLine(const std::vector<Point2d>& points, FitMethod method,
                      const FitParams& params) {
    switch (method) {
        case FitMethod::LeastSquares:
            return FitLine(points, params);
        case FitMethod::Huber:
            return FitLineHuber(points, 0.0, params);
        case FitMethod::Tukey:
            return FitLineTukey(points, 0.0, params);
        case FitMethod::RANSAC:
            return FitLineRANSAC(points, RansacParams(), params);
        default:
            return FitLine(points, params);
    }
}

LineFitResult FitLineWeighted(const std::vector<Point2d>& points,
                               const std::vector<double>& weights,
                               const FitParams& params) {
    LineFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < LINE_FIT_MIN_POINTS || weights.size() != points.size()) {
        return result;
    }

    // Compute weighted centroid
    Point2d centroid = ComputeWeightedCentroid(points, weights);

    // Build weighted covariance matrix
    double sxx = 0.0, sxy = 0.0, syy = 0.0, sw = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        double w = weights[i];
        double dx = points[i].x - centroid.x;
        double dy = points[i].y - centroid.y;
        sxx += w * dx * dx;
        sxy += w * dx * dy;
        syy += w * dy * dy;
        sw += w;
    }

    if (sw < 1e-15) {
        return FitLine(points, params);
    }

    // Normalize
    sxx /= sw;
    sxy /= sw;
    syy /= sw;

    // Eigendecomposition
    double trace = sxx + syy;
    double det = sxx * syy - sxy * sxy;
    double disc = trace * trace - 4.0 * det;
    if (disc < 0) disc = 0;

    double lambda2 = (trace - std::sqrt(disc)) / 2.0;

    double a, b;
    if (std::abs(sxy) > 1e-15) {
        a = lambda2 - syy;
        b = sxy;
    } else if (sxx < syy) {
        a = 1.0;
        b = 0.0;
    } else {
        a = 0.0;
        b = 1.0;
    }

    double norm = std::sqrt(a * a + b * b);
    if (norm < 1e-15) return result;
    a /= norm;
    b /= norm;

    double c = -(a * centroid.x + b * centroid.y);

    result.line = Line2d(a, b, c);
    result.success = true;

    auto residuals = ComputeLineResiduals(points, result.line);
    FillResidualStats(result, residuals, params);
    result.numInliers = result.numPoints;

    return result;
}

LineFitResult FitLineHuber(const std::vector<Point2d>& points, double sigma,
                            const FitParams& params) {
    LineFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < LINE_FIT_MIN_POINTS) {
        return result;
    }

    // Initial fit
    result = FitLine(points, params);
    if (!result.success) return result;

    // IRLS iterations
    const int maxIter = 20;
    const double tol = 1e-6;

    std::vector<double> weights(points.size(), 1.0);

    for (int iter = 0; iter < maxIter; ++iter) {
        // Compute residuals
        auto residuals = ComputeLineResiduals(points, result.line);

        // Estimate scale if not provided
        double scale = sigma;
        if (scale <= 0) {
            scale = RobustScaleMAD(residuals);
            if (scale < 1e-10) scale = 1.0;
        }

        // Compute weights
        double maxWeightChange = 0.0;
        for (size_t i = 0; i < points.size(); ++i) {
            double r = residuals[i] / scale;
            double newWeight = HuberWeight(r);
            maxWeightChange = std::max(maxWeightChange, std::abs(newWeight - weights[i]));
            weights[i] = newWeight;
        }

        // Refit with weights
        result = FitLineWeighted(points, weights, params);
        if (!result.success) return result;

        // Check convergence
        if (maxWeightChange < tol) break;
    }

    // Final residuals
    auto residuals = ComputeLineResiduals(points, result.line);
    FillResidualStats(result, residuals, params);
    result.numInliers = result.numPoints;

    return result;
}

LineFitResult FitLineTukey(const std::vector<Point2d>& points, double sigma,
                            const FitParams& params) {
    LineFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < LINE_FIT_MIN_POINTS) {
        return result;
    }

    // Initial fit
    result = FitLine(points, params);
    if (!result.success) return result;

    // IRLS iterations
    const int maxIter = 20;
    const double tol = 1e-6;

    std::vector<double> weights(points.size(), 1.0);

    for (int iter = 0; iter < maxIter; ++iter) {
        auto residuals = ComputeLineResiduals(points, result.line);

        double scale = sigma;
        if (scale <= 0) {
            scale = RobustScaleMAD(residuals);
            if (scale < 1e-10) scale = 1.0;
        }

        double maxWeightChange = 0.0;
        int numNonZero = 0;
        for (size_t i = 0; i < points.size(); ++i) {
            double r = residuals[i] / scale;
            double newWeight = TukeyWeight(r);
            maxWeightChange = std::max(maxWeightChange, std::abs(newWeight - weights[i]));
            weights[i] = newWeight;
            if (newWeight > 0) ++numNonZero;
        }

        // Need at least MIN_POINTS non-zero weights
        if (numNonZero < LINE_FIT_MIN_POINTS) {
            break;
        }

        result = FitLineWeighted(points, weights, params);
        if (!result.success) return result;

        if (maxWeightChange < tol) break;
    }

    auto residuals = ComputeLineResiduals(points, result.line);
    FillResidualStats(result, residuals, params);

    // Count inliers (non-zero weights)
    result.numInliers = 0;
    double scale = RobustScaleMAD(residuals);
    if (scale < 1e-10) scale = 1.0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (TukeyWeight(residuals[i] / scale) > 0) {
            ++result.numInliers;
        }
    }

    return result;
}

LineFitResult FitLineRANSAC(const std::vector<Point2d>& points,
                             const RansacParams& ransacParams,
                             const FitParams& params) {
    LineFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < LINE_FIT_MIN_POINTS) {
        return result;
    }

    // Define RANSAC model for line
    RansacModel<Line2d> lineModel;
    lineModel.minSampleSize = 2;

    lineModel.fitMinimal = [](const std::vector<Point2d>& pts) -> std::optional<Line2d> {
        if (pts.size() < 2) return std::nullopt;

        const Point2d& p1 = pts[0];
        const Point2d& p2 = pts[1];

        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        double len = std::sqrt(dx * dx + dy * dy);

        if (len < 1e-15) return std::nullopt;

        // Normal direction: (-dy, dx)
        double a = -dy / len;
        double b = dx / len;
        double c = -(a * p1.x + b * p1.y);

        return Line2d(a, b, c);
    };

    lineModel.fitAll = [](const std::vector<Point2d>& pts) -> std::optional<Line2d> {
        if (pts.size() < 2) return std::nullopt;
        auto res = FitLine(pts);
        if (!res.success) return std::nullopt;
        return res.line;
    };

    lineModel.distance = [](const Line2d& line, const Point2d& p) -> double {
        double norm = std::sqrt(line.a * line.a + line.b * line.b);
        if (norm < 1e-15) return 0.0;
        return (line.a * p.x + line.b * p.y + line.c) / norm;
    };

    // Run RANSAC
    auto ransacResult = RANSAC<Line2d>(points, lineModel, ransacParams);

    if (!ransacResult.success) {
        return result;
    }

    result.line = ransacResult.model;
    result.success = true;
    result.numInliers = ransacResult.numInliers;

    if (params.computeInlierMask) {
        result.inlierMask = ransacResult.inlierMask;
    }

    auto residuals = ComputeLineResiduals(points, result.line);
    FillResidualStats(result, residuals, params);

    return result;
}

// =============================================================================
// Circle Fitting Implementation
// =============================================================================

CircleFitResult FitCircleAlgebraic(const std::vector<Point2d>& points,
                                    const FitParams& params) {
    CircleFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < CIRCLE_FIT_MIN_POINTS) {
        return result;
    }

    // Kasa method: (x - a)^2 + (y - b)^2 = r^2
    // Linearize: x^2 + y^2 + Ax + By + C = 0
    // where A = -2a, B = -2b, C = a^2 + b^2 - r^2

    int n = static_cast<int>(points.size());

    // Build normal equations: M^T M x = M^T b
    // where M = [x, y, 1], b = -(x^2 + y^2)

    double sxx = 0.0, sxy = 0.0, sx = 0.0;
    double syy = 0.0, sy = 0.0;
    double s1 = static_cast<double>(n);
    double bx = 0.0, by = 0.0, b1 = 0.0;

    for (const auto& p : points) {
        double x2 = p.x * p.x;
        double y2 = p.y * p.y;
        double rhs = -(x2 + y2);

        sxx += x2;
        sxy += p.x * p.y;
        sx += p.x;
        syy += y2;
        sy += p.y;

        bx += p.x * rhs;
        by += p.y * rhs;
        b1 += rhs;
    }

    // Solve 3x3 system using Cramer's rule or LU
    // [sxx sxy sx ] [A]   [bx]
    // [sxy syy sy ] [B] = [by]
    // [sx  sy  s1 ] [C]   [b1]

    Mat33 M;
    M(0, 0) = sxx; M(0, 1) = sxy; M(0, 2) = sx;
    M(1, 0) = sxy; M(1, 1) = syy; M(1, 2) = sy;
    M(2, 0) = sx;  M(2, 1) = sy;  M(2, 2) = s1;

    Vec3 rhs{bx, by, b1};
    Vec3 sol = Solve3x3(M, rhs);

    double A = sol[0];
    double B = sol[1];
    double C = sol[2];

    // Extract circle parameters
    double cx = -A / 2.0;
    double cy = -B / 2.0;
    double r2 = cx * cx + cy * cy - C;

    if (r2 <= 0) {
        return result;  // Invalid circle
    }

    result.circle = Circle2d(Point2d(cx, cy), std::sqrt(r2));
    result.success = true;

    auto residuals = ComputeCircleResiduals(points, result.circle);
    FillResidualStats(result, residuals, params);
    result.numInliers = result.numPoints;

    return result;
}

CircleFitResult FitCircleGeometric(const std::vector<Point2d>& points,
                                    const GeometricFitParams& geoParams,
                                    const FitParams& params) {
    CircleFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < CIRCLE_FIT_MIN_POINTS) {
        return result;
    }

    // Initial estimate from algebraic fitting
    auto algebraicResult = FitCircleAlgebraic(points, FitParams());
    if (!algebraicResult.success) {
        return result;
    }

    double cx = algebraicResult.circle.center.x;
    double cy = algebraicResult.circle.center.y;
    double r = algebraicResult.circle.radius;

    int n = static_cast<int>(points.size());

    // Gauss-Newton iterations
    for (int iter = 0; iter < geoParams.maxIterations; ++iter) {
        // Build Jacobian J and residual vector
        MatX J(n, 3);
        VecX residual(n);

        for (int i = 0; i < n; ++i) {
            double dx = points[i].x - cx;
            double dy = points[i].y - cy;
            double d = std::sqrt(dx * dx + dy * dy);

            if (d < 1e-15) d = 1e-15;

            // Residual: d_i - r
            residual[i] = d - r;

            // Jacobian: dr/d(cx) = -dx/d, dr/d(cy) = -dy/d, dr/d(r) = -1
            J(i, 0) = -dx / d;
            J(i, 1) = -dy / d;
            J(i, 2) = -1.0;
        }

        // Solve J^T J delta = -J^T r
        VecX delta = SolveLeastSquares(J, residual);

        // Update parameters
        cx -= delta[0];
        cy -= delta[1];
        r -= delta[2];

        // Check convergence
        if (delta.Norm() < geoParams.tolerance) {
            break;
        }
    }

    if (r <= 0) {
        return result;
    }

    result.circle = Circle2d(Point2d(cx, cy), r);
    result.success = true;

    auto residuals = ComputeCircleResiduals(points, result.circle);
    FillResidualStats(result, residuals, params);
    result.numInliers = result.numPoints;

    return result;
}

CircleFitResult FitCircle(const std::vector<Point2d>& points,
                           CircleFitMethod method,
                           const FitParams& params) {
    switch (method) {
        case CircleFitMethod::Algebraic:
            return FitCircleAlgebraic(points, params);
        case CircleFitMethod::Geometric:
            return FitCircleGeometric(points, GeometricFitParams(), params);
        case CircleFitMethod::AlgebraicHuber:
        case CircleFitMethod::GeoHuber:
            return FitCircleHuber(points, method == CircleFitMethod::GeoHuber, 0.0, params);
        case CircleFitMethod::AlgebraicTukey:
        case CircleFitMethod::GeoTukey:
            return FitCircleTukey(points, method == CircleFitMethod::GeoTukey, 0.0, params);
        case CircleFitMethod::RANSAC:
            return FitCircleRANSAC(points, RansacParams(), params);
        default:
            return FitCircleAlgebraic(points, params);
    }
}

CircleFitResult FitCircleWeighted(const std::vector<Point2d>& points,
                                   const std::vector<double>& weights,
                                   const FitParams& params) {
    CircleFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < CIRCLE_FIT_MIN_POINTS || weights.size() != points.size()) {
        return result;
    }

    // Weighted Kasa method
    double sxx = 0.0, sxy = 0.0, sx = 0.0;
    double syy = 0.0, sy = 0.0, sw = 0.0;
    double bx = 0.0, by = 0.0, b1 = 0.0;

    for (size_t i = 0; i < points.size(); ++i) {
        double w = weights[i];
        double x = points[i].x;
        double y = points[i].y;
        double x2 = x * x;
        double y2 = y * y;
        double rhs = -(x2 + y2);

        sxx += w * x2;
        sxy += w * x * y;
        sx += w * x;
        syy += w * y2;
        sy += w * y;
        sw += w;

        bx += w * x * rhs;
        by += w * y * rhs;
        b1 += w * rhs;
    }

    if (sw < 1e-15) {
        return FitCircleAlgebraic(points, params);
    }

    Mat33 M;
    M(0, 0) = sxx; M(0, 1) = sxy; M(0, 2) = sx;
    M(1, 0) = sxy; M(1, 1) = syy; M(1, 2) = sy;
    M(2, 0) = sx;  M(2, 1) = sy;  M(2, 2) = sw;

    Vec3 rhs{bx, by, b1};
    Vec3 sol = Solve3x3(M, rhs);

    double cx = -sol[0] / 2.0;
    double cy = -sol[1] / 2.0;
    double r2 = cx * cx + cy * cy - sol[2];

    if (r2 <= 0) return result;

    result.circle = Circle2d(Point2d(cx, cy), std::sqrt(r2));
    result.success = true;

    auto residuals = ComputeCircleResiduals(points, result.circle);
    FillResidualStats(result, residuals, params);
    result.numInliers = result.numPoints;

    return result;
}

CircleFitResult FitCircleHuber(const std::vector<Point2d>& points,
                                bool geometric, double sigma,
                                const FitParams& params) {
    CircleFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < CIRCLE_FIT_MIN_POINTS) {
        return result;
    }

    // Initial fit
    result = geometric ? FitCircleGeometric(points, GeometricFitParams(), params)
                       : FitCircleAlgebraic(points, params);
    if (!result.success) return result;

    const int maxIter = 20;
    const double tol = 1e-6;
    std::vector<double> weights(points.size(), 1.0);

    for (int iter = 0; iter < maxIter; ++iter) {
        auto residuals = ComputeCircleResiduals(points, result.circle);

        double scale = sigma;
        if (scale <= 0) {
            scale = RobustScaleMAD(residuals);
            if (scale < 1e-10) scale = 1.0;
        }

        double maxWeightChange = 0.0;
        for (size_t i = 0; i < points.size(); ++i) {
            double r = residuals[i] / scale;
            double newWeight = HuberWeight(r);
            maxWeightChange = std::max(maxWeightChange, std::abs(newWeight - weights[i]));
            weights[i] = newWeight;
        }

        result = FitCircleWeighted(points, weights, params);
        if (!result.success) return result;

        if (geometric) {
            // Refine with geometric fitting
            auto geoResult = FitCircleGeometric(points, GeometricFitParams(), params);
            if (geoResult.success) {
                result = geoResult;
            }
        }

        if (maxWeightChange < tol) break;
    }

    auto residuals = ComputeCircleResiduals(points, result.circle);
    FillResidualStats(result, residuals, params);
    result.numInliers = result.numPoints;

    return result;
}

CircleFitResult FitCircleTukey(const std::vector<Point2d>& points,
                                bool geometric, double sigma,
                                const FitParams& params) {
    CircleFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < CIRCLE_FIT_MIN_POINTS) {
        return result;
    }

    result = geometric ? FitCircleGeometric(points, GeometricFitParams(), params)
                       : FitCircleAlgebraic(points, params);
    if (!result.success) return result;

    const int maxIter = 20;
    const double tol = 1e-6;
    std::vector<double> weights(points.size(), 1.0);

    for (int iter = 0; iter < maxIter; ++iter) {
        auto residuals = ComputeCircleResiduals(points, result.circle);

        double scale = sigma;
        if (scale <= 0) {
            scale = RobustScaleMAD(residuals);
            if (scale < 1e-10) scale = 1.0;
        }

        double maxWeightChange = 0.0;
        int numNonZero = 0;
        for (size_t i = 0; i < points.size(); ++i) {
            double r = residuals[i] / scale;
            double newWeight = TukeyWeight(r);
            maxWeightChange = std::max(maxWeightChange, std::abs(newWeight - weights[i]));
            weights[i] = newWeight;
            if (newWeight > 0) ++numNonZero;
        }

        if (numNonZero < CIRCLE_FIT_MIN_POINTS) break;

        result = FitCircleWeighted(points, weights, params);
        if (!result.success) return result;

        if (maxWeightChange < tol) break;
    }

    auto residuals = ComputeCircleResiduals(points, result.circle);
    FillResidualStats(result, residuals, params);

    double scale = RobustScaleMAD(residuals);
    if (scale < 1e-10) scale = 1.0;
    result.numInliers = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (TukeyWeight(residuals[i] / scale) > 0) ++result.numInliers;
    }

    return result;
}

CircleFitResult FitCircleRANSAC(const std::vector<Point2d>& points,
                                 const RansacParams& ransacParams,
                                 const FitParams& params) {
    CircleFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < CIRCLE_FIT_MIN_POINTS) {
        return result;
    }

    RansacModel<Circle2d> circleModel;
    circleModel.minSampleSize = 3;

    circleModel.fitMinimal = [](const std::vector<Point2d>& pts) -> std::optional<Circle2d> {
        if (pts.size() < 3) return std::nullopt;
        return FitCircleExact3Points(pts[0], pts[1], pts[2]);
    };

    circleModel.fitAll = [](const std::vector<Point2d>& pts) -> std::optional<Circle2d> {
        if (pts.size() < 3) return std::nullopt;
        auto res = FitCircleAlgebraic(pts);
        if (!res.success) return std::nullopt;
        return res.circle;
    };

    circleModel.distance = [](const Circle2d& c, const Point2d& p) -> double {
        double dx = p.x - c.center.x;
        double dy = p.y - c.center.y;
        return std::sqrt(dx * dx + dy * dy) - c.radius;
    };

    auto ransacResult = RANSAC<Circle2d>(points, circleModel, ransacParams);

    if (!ransacResult.success) return result;

    result.circle = ransacResult.model;
    result.success = true;
    result.numInliers = ransacResult.numInliers;

    if (params.computeInlierMask) {
        result.inlierMask = ransacResult.inlierMask;
    }

    auto residuals = ComputeCircleResiduals(points, result.circle);
    FillResidualStats(result, residuals, params);

    return result;
}

std::optional<Circle2d> FitCircleExact3Points(const Point2d& p1,
                                               const Point2d& p2,
                                               const Point2d& p3) {
    // Check collinearity
    if (AreCollinear(p1, p2, p3)) {
        return std::nullopt;
    }

    // Perpendicular bisector method
    // Midpoints
    double mx1 = (p1.x + p2.x) / 2.0;
    double my1 = (p1.y + p2.y) / 2.0;
    double mx2 = (p2.x + p3.x) / 2.0;
    double my2 = (p2.y + p3.y) / 2.0;

    // Direction vectors of edges
    double dx1 = p2.x - p1.x;
    double dy1 = p2.y - p1.y;
    double dx2 = p3.x - p2.x;
    double dy2 = p3.y - p2.y;

    // Perpendicular directions
    double px1 = -dy1, py1 = dx1;
    double px2 = -dy2, py2 = dx2;

    // Intersection of perpendicular bisectors
    // (mx1, my1) + t * (px1, py1) = (mx2, my2) + s * (px2, py2)
    double denom = px1 * py2 - py1 * px2;
    if (std::abs(denom) < 1e-15) {
        return std::nullopt;
    }

    double t = ((mx2 - mx1) * py2 - (my2 - my1) * px2) / denom;

    double cx = mx1 + t * px1;
    double cy = my1 + t * py1;
    double r = std::sqrt((cx - p1.x) * (cx - p1.x) + (cy - p1.y) * (cy - p1.y));

    return Circle2d(Point2d(cx, cy), r);
}

// =============================================================================
// Ellipse Fitting Implementation
// =============================================================================

EllipseFitResult FitEllipseFitzgibbon(const std::vector<Point2d>& points,
                                       const FitParams& params) {
    EllipseFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < ELLIPSE_FIT_MIN_POINTS) {
        return result;
    }

    int n = static_cast<int>(points.size());

    // Normalize points for numerical stability
    auto [normPoints, T] = NormalizePoints(points);

    // Build design matrix D = [x^2, xy, y^2, x, y, 1]
    MatX D(n, 6);
    for (int i = 0; i < n; ++i) {
        double x = normPoints[i].x;
        double y = normPoints[i].y;
        D(i, 0) = x * x;
        D(i, 1) = x * y;
        D(i, 2) = y * y;
        D(i, 3) = x;
        D(i, 4) = y;
        D(i, 5) = 1.0;
    }

    // Scatter matrix S = D^T D
    MatX S = D.Transpose() * D;

    // Constraint matrix C for ellipse: 4ac - b^2 > 0
    // Using constraint b^2 - 4ac = -1 (normalized)
    // C = [0 0 -2; 0 1 0; -2 0 0; 0 0 0; 0 0 0; 0 0 0] extended to 6x6
    Mat33 C1, C2;
    C1(0, 0) = 0;  C1(0, 1) = 0;  C1(0, 2) = -2;
    C1(1, 0) = 0;  C1(1, 1) = 1;  C1(1, 2) = 0;
    C1(2, 0) = -2; C1(2, 1) = 0;  C1(2, 2) = 0;

    // Partition S into blocks
    // S = [S1  S2]  where S1 is 3x3 (quadratic terms)
    //     [S2' S3]        S3 is 3x3 (linear terms)
    Mat33 S1, S3;
    MatX S2(3, 3);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            S1(i, j) = S(i, j);
            S3(i, j) = S(i + 3, j + 3);
            S2(i, j) = S(i, j + 3);
        }
    }

    // Reduced problem: M * a1 = lambda * C1 * a1
    // where M = S1 - S2 * S3^{-1} * S2^T

    // Solve S3 * X = S2^T for X
    Mat33 S3inv;
    if (!S3.IsInvertible()) {
        return result;
    }
    S3inv = S3.Inverse();

    MatX S2T = S2.Transpose();
    MatX temp(3, 3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            temp(i, j) = 0;
            for (int k = 0; k < 3; ++k) {
                temp(i, j) += S3inv(i, k) * S2T(k, j);
            }
        }
    }

    Mat33 M;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            M(i, j) = S1(i, j);
            for (int k = 0; k < 3; ++k) {
                M(i, j) -= S2(i, k) * temp(k, j);
            }
        }
    }

    // Solve C1^{-1} * M * a1 = lambda * a1
    // C1^{-1} = [0 0 -0.5; 0 1 0; -0.5 0 0]
    Mat33 C1inv;
    C1inv(0, 0) = 0;    C1inv(0, 1) = 0;  C1inv(0, 2) = -0.5;
    C1inv(1, 0) = 0;    C1inv(1, 1) = 1;  C1inv(1, 2) = 0;
    C1inv(2, 0) = -0.5; C1inv(2, 1) = 0;  C1inv(2, 2) = 0;

    Mat33 MC = C1inv * M;

    // Find eigenvalues using characteristic polynomial
    // For 3x3 matrix, use analytic formula
    double a11 = MC(0, 0), a12 = MC(0, 1), a13 = MC(0, 2);
    double a21 = MC(1, 0), a22 = MC(1, 1), a23 = MC(1, 2);
    double a31 = MC(2, 0), a32 = MC(2, 1), a33 = MC(2, 2);

    double tr = a11 + a22 + a33;
    double q = tr / 3.0;

    double p1 = (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) + (a33 - q) * (a33 - q);
    p1 += 2.0 * (a12 * a12 + a13 * a13 + a23 * a23);
    double p = std::sqrt(p1 / 6.0);

    if (p < 1e-15) {
        return result;
    }

    Mat33 B;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            B(i, j) = (MC(i, j) - (i == j ? q : 0)) / p;
        }
    }

    double detB = B.Determinant();
    double r = detB / 2.0;
    r = std::max(-1.0, std::min(1.0, r));
    double phi = std::acos(r) / 3.0;

    // Eigenvalues
    double eig1 = q + 2.0 * p * std::cos(phi);
    double eig2 = q + 2.0 * p * std::cos(phi + 2.0 * M_PI / 3.0);
    double eig3 = q + 2.0 * p * std::cos(phi + 4.0 * M_PI / 3.0);

    // Find the smallest positive eigenvalue (for ellipse constraint)
    double targetEig = std::numeric_limits<double>::max();
    if (eig1 > 0 && eig1 < targetEig) targetEig = eig1;
    if (eig2 > 0 && eig2 < targetEig) targetEig = eig2;
    if (eig3 > 0 && eig3 < targetEig) targetEig = eig3;

    if (targetEig == std::numeric_limits<double>::max()) {
        return result;
    }

    // Find eigenvector for targetEig
    // (MC - targetEig * I) * v = 0
    Mat33 A = MC;
    A(0, 0) -= targetEig;
    A(1, 1) -= targetEig;
    A(2, 2) -= targetEig;

    // Use SVD to find null space
    // Simple approach: find row with smallest norm, compute cross product of other two
    Vec3 r0{A(0, 0), A(0, 1), A(0, 2)};
    Vec3 r1{A(1, 0), A(1, 1), A(1, 2)};
    Vec3 r2{A(2, 0), A(2, 1), A(2, 2)};

    double n0 = r0.NormSquared();
    double n1 = r1.NormSquared();
    double n2 = r2.NormSquared();

    Vec3 a1;
    if (n0 <= n1 && n0 <= n2) {
        a1 = Cross(r1, r2);
    } else if (n1 <= n0 && n1 <= n2) {
        a1 = Cross(r0, r2);
    } else {
        a1 = Cross(r0, r1);
    }

    double norm = a1.Norm();
    if (norm < 1e-15) {
        return result;
    }
    a1 = a1 / norm;

    // Compute a2 = -S3^{-1} * S2^T * a1
    Vec3 a2;
    for (int i = 0; i < 3; ++i) {
        a2[i] = 0;
        for (int j = 0; j < 3; ++j) {
            a2[i] -= temp(i, j) * a1[j];
        }
    }

    // Full conic coefficients [A, B, C, D, E, F]
    double A_coef = a1[0];
    double B_coef = a1[1];
    double C_coef = a1[2];
    double D_coef = a2[0];
    double E_coef = a2[1];
    double F_coef = a2[2];

    // Extract ellipse parameters from conic Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    // Check discriminant: B^2 - 4AC < 0 for ellipse
    double disc = B_coef * B_coef - 4.0 * A_coef * C_coef;
    if (disc >= 0) {
        return result;  // Not an ellipse
    }

    // Center: solve dF/dx = 0, dF/dy = 0
    // 2Ax + By + D = 0
    // Bx + 2Cy + E = 0
    double det = 4.0 * A_coef * C_coef - B_coef * B_coef;
    double cx = (B_coef * E_coef - 2.0 * C_coef * D_coef) / det;
    double cy = (B_coef * D_coef - 2.0 * A_coef * E_coef) / det;

    // Value at center
    double F_center = A_coef * cx * cx + B_coef * cx * cy + C_coef * cy * cy
                     + D_coef * cx + E_coef * cy + F_coef;

    // Semi-axes and angle
    // Principal axis angle: tan(2*theta) = B / (A - C)
    double theta = 0.5 * std::atan2(B_coef, A_coef - C_coef);

    double cosT = std::cos(theta);
    double sinT = std::sin(theta);

    // Rotated coefficients
    double Ap = A_coef * cosT * cosT + B_coef * cosT * sinT + C_coef * sinT * sinT;
    double Cp = A_coef * sinT * sinT - B_coef * cosT * sinT + C_coef * cosT * cosT;

    if (Ap * F_center >= 0 || Cp * F_center >= 0) {
        return result;  // Invalid ellipse
    }

    double a = std::sqrt(-F_center / Ap);  // semi-axis along rotated x
    double b = std::sqrt(-F_center / Cp);  // semi-axis along rotated y

    // Ensure a >= b (a is semi-major)
    if (a < b) {
        std::swap(a, b);
        theta += M_PI / 2.0;
    }

    // Normalize angle to [-pi/2, pi/2]
    while (theta > M_PI / 2.0) theta -= M_PI;
    while (theta < -M_PI / 2.0) theta += M_PI;

    // Denormalize
    Ellipse2d normEllipse(Point2d(cx, cy), a, b, theta);
    result.ellipse = DenormalizeEllipse(normEllipse, T);
    result.success = true;

    auto residuals = ComputeEllipseResiduals(points, result.ellipse);
    FillResidualStats(result, residuals, params);
    result.numInliers = result.numPoints;

    return result;
}

EllipseFitResult FitEllipseGeometric(const std::vector<Point2d>& points,
                                      const GeometricFitParams& geoParams,
                                      const FitParams& params) {
    EllipseFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < ELLIPSE_FIT_MIN_POINTS) {
        return result;
    }

    // Initial estimate from Fitzgibbon
    auto fitzResult = FitEllipseFitzgibbon(points, FitParams());
    if (!fitzResult.success) {
        return result;
    }

    // For now, just return Fitzgibbon result
    // Full geometric fitting would require Levenberg-Marquardt iteration
    result = fitzResult;

    auto residuals = ComputeEllipseResiduals(points, result.ellipse);
    FillResidualStats(result, residuals, params);

    return result;
}

EllipseFitResult FitEllipse(const std::vector<Point2d>& points,
                             EllipseFitMethod method,
                             const FitParams& params) {
    switch (method) {
        case EllipseFitMethod::Fitzgibbon:
            return FitEllipseFitzgibbon(points, params);
        case EllipseFitMethod::Geometric:
            return FitEllipseGeometric(points, GeometricFitParams(), params);
        case EllipseFitMethod::RANSAC:
            return FitEllipseRANSAC(points, RansacParams(), params);
        default:
            return FitEllipseFitzgibbon(points, params);
    }
}

EllipseFitResult FitEllipseRANSAC(const std::vector<Point2d>& points,
                                   const RansacParams& ransacParams,
                                   const FitParams& params) {
    EllipseFitResult result;
    result.success = false;
    result.numPoints = static_cast<int>(points.size());

    if (points.size() < ELLIPSE_FIT_MIN_POINTS) {
        return result;
    }

    RansacModel<Ellipse2d> ellipseModel;
    ellipseModel.minSampleSize = 5;

    ellipseModel.fitMinimal = [](const std::vector<Point2d>& pts) -> std::optional<Ellipse2d> {
        if (pts.size() < 5) return std::nullopt;
        auto res = FitEllipseFitzgibbon(pts);
        if (!res.success) return std::nullopt;
        return res.ellipse;
    };

    ellipseModel.fitAll = [](const std::vector<Point2d>& pts) -> std::optional<Ellipse2d> {
        if (pts.size() < 5) return std::nullopt;
        auto res = FitEllipseFitzgibbon(pts);
        if (!res.success) return std::nullopt;
        return res.ellipse;
    };

    ellipseModel.distance = [](const Ellipse2d& e, const Point2d& p) -> double {
        double cosA = std::cos(-e.angle);
        double sinA = std::sin(-e.angle);
        double dx = p.x - e.center.x;
        double dy = p.y - e.center.y;
        double x = dx * cosA - dy * sinA;
        double y = dx * sinA + dy * cosA;

        double a2 = e.a * e.a;
        double b2 = e.b * e.b;
        double val = (x * x) / a2 + (y * y) / b2 - 1.0;

        return val * std::sqrt(a2 * b2) / std::sqrt(b2 * x * x + a2 * y * y + 1e-15);
    };

    auto ransacResult = RANSAC<Ellipse2d>(points, ellipseModel, ransacParams);

    if (!ransacResult.success) return result;

    result.ellipse = ransacResult.model;
    result.success = true;
    result.numInliers = ransacResult.numInliers;

    if (params.computeInlierMask) {
        result.inlierMask = ransacResult.inlierMask;
    }

    auto residuals = ComputeEllipseResiduals(points, result.ellipse);
    FillResidualStats(result, residuals, params);

    return result;
}

} // namespace Qi::Vision::Internal
