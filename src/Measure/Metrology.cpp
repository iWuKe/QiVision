/**
 * @file Metrology.cpp
 * @brief Implementation of Metrology module
 */

#include <QiVision/Measure/Metrology.h>
#include <QiVision/Measure/Caliper.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>
#include <QiVision/Internal/Fitting.h>
#include <QiVision/Internal/Profiler.h>
#include <QiVision/Internal/Edge1D.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace Qi::Vision::Measure {

namespace {
    constexpr double PI = 3.14159265358979323846;
    constexpr double TWO_PI = 2.0 * PI;

    void ValidateMetrologyParams(const MetrologyMeasureParams& params, const char* funcName) {
        if (!std::isfinite(params.measureLength1) || !std::isfinite(params.measureLength2) ||
            params.measureLength1 <= 0.0 || params.measureLength2 <= 0.0) {
            throw InvalidArgumentException(std::string(funcName) + ": measure lengths must be > 0");
        }
        if (!std::isfinite(params.measureSigma) || params.measureSigma <= 0.0) {
            throw InvalidArgumentException(std::string(funcName) + ": measureSigma must be > 0");
        }
        if (!std::isfinite(params.measureThreshold) || params.measureThreshold < 0.0) {
            throw InvalidArgumentException(std::string(funcName) + ": measureThreshold must be >= 0");
        }
        if (params.numMeasures <= 0) {
            throw InvalidArgumentException(std::string(funcName) + ": numMeasures must be > 0");
        }
        if (params.numInstances <= 0) {
            throw InvalidArgumentException(std::string(funcName) + ": numInstances must be > 0");
        }
        if (!std::isfinite(params.minScore) || params.minScore < 0.0 || params.minScore > 1.0) {
            throw InvalidArgumentException(std::string(funcName) + ": minScore must be in [0,1]");
        }
        if (!std::isfinite(params.minCoverage) || params.minCoverage < 0.0 || params.minCoverage > 1.0) {
            throw InvalidArgumentException(std::string(funcName) + ": minCoverage must be in [0,1]");
        }
        if (!std::isfinite(params.maxRmsError)) {
            throw InvalidArgumentException(std::string(funcName) + ": maxRmsError must be finite");
        }
        if (params.maxRmsError > 0.0 && params.maxRmsError < 1e-9) {
            throw InvalidArgumentException(std::string(funcName) + ": maxRmsError must be >= 1e-9 when enabled");
        }
        if (!std::isfinite(params.distanceThreshold) || params.distanceThreshold <= 0.0) {
            throw InvalidArgumentException(std::string(funcName) + ": distanceThreshold must be > 0");
        }
        if (params.maxIterations < -1) {
            throw InvalidArgumentException(std::string(funcName) + ": maxIterations must be >= -1");
        }
        if (params.ignorePointCount < 0) {
            throw InvalidArgumentException(std::string(funcName) + ": ignorePointCount must be >= 0");
        }
    }

    // Helper: Merge params with measureLength and transition/select strings
    MetrologyMeasureParams MergeParams(
        double measureLength1, double measureLength2,
        const std::string& transition, const std::string& select,
        const MetrologyMeasureParams& params,
        const char* funcName)
    {
        MetrologyMeasureParams result = params;
        result.measureLength1 = measureLength1;
        result.measureLength2 = measureLength2;

        // Parse transition string (override if specified)
        result.measureTransition = ParseTransition(transition);

        // Parse select string (override if specified)
        std::string lowerSelect = select;
        std::transform(lowerSelect.begin(), lowerSelect.end(), lowerSelect.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lowerSelect == "best") {
            lowerSelect = "strongest";
        }
        result.measureSelect = ParseEdgeSelect(lowerSelect);

        ValidateMetrologyParams(result, funcName);

        return result;
    }

    // Normalize angle to [0, 2*PI) - may be used for arc handling
    [[maybe_unused]] double NormalizeAngle(double angle) {
        while (angle < 0.0) angle += TWO_PI;
        while (angle >= TWO_PI) angle -= TWO_PI;
        return angle;
    }

    // Rotate point around origin
    Point2d RotatePoint(const Point2d& p, double phi, const Point2d& center) {
        double cosPhi = std::cos(phi);
        double sinPhi = std::sin(phi);
        double dx = p.x - center.x;
        double dy = p.y - center.y;
        return {
            center.x + dx * cosPhi - dy * sinPhi,
            center.y + dx * sinPhi + dy * cosPhi
        };
    }

    struct IgnoreSelection {
        std::vector<Point2d> points;
        std::vector<double> scores;
        std::vector<size_t> keptIndices;
        bool applied = false;
    };

    IgnoreSelection SelectPointsForIgnore(const std::vector<Point2d>& points,
                                          const std::vector<double>& scores,
                                          int32_t ignorePointCount,
                                          IgnorePointPolicy policy,
                                          const std::vector<double>& residuals,
                                          size_t minRequired) {
        IgnoreSelection selection;
        selection.points = points;
        selection.scores = scores;
        selection.keptIndices.resize(points.size());
        std::iota(selection.keptIndices.begin(), selection.keptIndices.end(), 0);

        const size_t n = points.size();
        if (ignorePointCount <= 0 || n <= minRequired) {
            return selection;
        }

        const size_t removable = n - minRequired;
        const size_t removeCount = std::min(removable, static_cast<size_t>(ignorePointCount));
        if (removeCount == 0) {
            return selection;
        }

        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);

        if (policy == IgnorePointPolicy::ByResidual) {
            if (residuals.size() != n) {
                return selection;
            }
            std::sort(order.begin(), order.end(),
                      [&residuals](size_t a, size_t b) { return std::abs(residuals[a]) > std::abs(residuals[b]); });
        } else {
            if (scores.size() != n) {
                return selection;
            }
            std::sort(order.begin(), order.end(),
                      [&scores](size_t a, size_t b) { return scores[a] < scores[b]; });
        }

        std::vector<bool> removed(n, false);
        for (size_t i = 0; i < removeCount; ++i) {
            removed[order[i]] = true;
        }

        selection.points.clear();
        selection.scores.clear();
        selection.keptIndices.clear();
        selection.points.reserve(n - removeCount);
        selection.scores.reserve(n - removeCount);
        selection.keptIndices.reserve(n - removeCount);

        for (size_t i = 0; i < n; ++i) {
            if (removed[i]) continue;
            selection.points.push_back(points[i]);
            if (!scores.empty()) {
                selection.scores.push_back(scores[i]);
            }
            selection.keptIndices.push_back(i);
        }

        selection.applied = (selection.points.size() < points.size());
        return selection;
    }

    std::vector<double> ExpandMappedValues(const std::vector<double>& compact,
                                           const std::vector<size_t>& keptIndices,
                                           size_t originalSize) {
        if (compact.empty()) return {};
        if (keptIndices.empty()) return compact;
        if (compact.size() != keptIndices.size()) return {};

        std::vector<double> expanded(originalSize, 0.0);
        for (size_t i = 0; i < keptIndices.size(); ++i) {
            expanded[keptIndices[i]] = compact[i];
        }
        return expanded;
    }

    template<typename FitResultT>
    std::vector<double> BuildWeightsForDisplay(const FitResultT& fitResult,
                                               const std::vector<size_t>& keptIndices,
                                               size_t originalSize) {
        if (!fitResult.weights.empty()) {
            return ExpandMappedValues(fitResult.weights, keptIndices, originalSize);
        }
        if (!fitResult.inlierMask.empty()) {
            std::vector<double> compact(fitResult.inlierMask.size(), 0.0);
            for (size_t i = 0; i < fitResult.inlierMask.size(); ++i) {
                compact[i] = fitResult.inlierMask[i] ? 1.0 : 0.0;
            }
            return ExpandMappedValues(compact, keptIndices, originalSize);
        }
        return {};
    }

    template<typename FitResultT>
    std::vector<bool> BuildInstanceCompactMask(const FitResultT& fitResult,
                                               size_t compactSize,
                                               double distanceThreshold,
                                               size_t minRequired) {
        if (compactSize < minRequired) {
            return {};
        }

        std::vector<bool> mask(compactSize, false);

        if (fitResult.inlierMask.size() == compactSize) {
            mask = fitResult.inlierMask;
        } else if (fitResult.weights.size() == compactSize) {
            for (size_t i = 0; i < compactSize; ++i) {
                mask[i] = fitResult.weights[i] > 0.5;
            }
        } else if (fitResult.residuals.size() == compactSize &&
                   std::isfinite(distanceThreshold) && distanceThreshold > 0.0) {
            for (size_t i = 0; i < compactSize; ++i) {
                mask[i] = std::abs(fitResult.residuals[i]) <= distanceThreshold;
            }
        } else {
            return {};
        }

        size_t inlierCount = 0;
        for (bool m : mask) {
            if (m) ++inlierCount;
        }
        if (inlierCount < minRequired) {
            return {};
        }
        return mask;
    }

    double ComputeMedian(std::vector<double> values) {
        if (values.empty()) return 0.0;
        size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        double m = values[mid];
        if ((values.size() % 2) == 0) {
            auto maxIt = std::max_element(values.begin(), values.begin() + mid);
            if (maxIt != values.begin() + mid) {
                m = 0.5 * (m + *maxIt);
            }
        }
        return m;
    }

    // Adaptive residual threshold from robust statistics (MAD).
    double ComputeAdaptiveResidualThreshold(const std::vector<double>& residuals,
                                            double baseThreshold) {
        std::vector<double> absResiduals;
        absResiduals.reserve(residuals.size());
        for (double r : residuals) {
            if (std::isfinite(r)) {
                absResiduals.push_back(std::abs(r));
            }
        }
        if (absResiduals.size() < 3) {
            return baseThreshold;
        }

        double median = ComputeMedian(absResiduals);
        for (double& v : absResiduals) {
            v = std::abs(v - median);
        }
        double mad = ComputeMedian(absResiduals);
        double robustSigma = 1.4826 * mad;
        double adaptive = median + 2.5 * robustSigma;
        double floorValue = std::max(baseThreshold, 1e-6);
        return std::max(floorValue, adaptive);
    }

    bool FilterByResidualThreshold(const std::vector<double>& residuals,
                                   double threshold,
                                   size_t minRequired,
                                   std::vector<Point2d>& points,
                                   std::vector<double>& scores,
                                   std::vector<size_t>& keptIndices) {
        if (residuals.size() != points.size() || keptIndices.size() != points.size()) {
            return false;
        }

        std::vector<bool> keep(points.size(), false);
        size_t keptCount = 0;
        for (size_t i = 0; i < residuals.size(); ++i) {
            keep[i] = std::isfinite(residuals[i]) && std::abs(residuals[i]) <= threshold;
            if (keep[i]) ++keptCount;
        }
        if (keptCount < minRequired || keptCount == points.size()) {
            return false;
        }

        std::vector<Point2d> filteredPoints;
        std::vector<double> filteredScores;
        std::vector<size_t> filteredIndices;
        filteredPoints.reserve(keptCount);
        filteredScores.reserve(scores.size());
        filteredIndices.reserve(keptCount);

        for (size_t i = 0; i < points.size(); ++i) {
            if (!keep[i]) continue;
            filteredPoints.push_back(points[i]);
            if (!scores.empty() && i < scores.size()) {
                filteredScores.push_back(scores[i]);
            }
            filteredIndices.push_back(keptIndices[i]);
        }

        points = std::move(filteredPoints);
        scores = std::move(filteredScores);
        keptIndices = std::move(filteredIndices);
        return true;
    }

    void RemoveMaskedPoints(std::vector<Point2d>& points,
                            std::vector<double>& scores,
                            std::vector<size_t>& indexMap,
                            const std::vector<bool>& removeMask) {
        if (points.size() != removeMask.size()) {
            return;
        }

        std::vector<Point2d> newPoints;
        std::vector<double> newScores;
        std::vector<size_t> newMap;
        newPoints.reserve(points.size());
        newScores.reserve(scores.size());
        newMap.reserve(indexMap.size());

        for (size_t i = 0; i < points.size(); ++i) {
            if (removeMask[i]) continue;
            newPoints.push_back(points[i]);
            if (!scores.empty() && i < scores.size()) {
                newScores.push_back(scores[i]);
            }
            if (i < indexMap.size()) {
                newMap.push_back(indexMap[i]);
            }
        }

        points = std::move(newPoints);
        scores = std::move(newScores);
        indexMap = std::move(newMap);
    }
}

// =============================================================================
// MetrologyObjectLine Implementation
// =============================================================================

MetrologyObjectLine::MetrologyObjectLine(double row1, double col1,
                                          double row2, double col2,
                                          double measureLength1, double measureLength2,
                                          int32_t numMeasures)
    : row1_(row1), col1_(col1), row2_(row2), col2_(col2) {
    params_.measureLength1 = measureLength1;
    params_.measureLength2 = measureLength2;
    params_.numMeasures = numMeasures;
}

double MetrologyObjectLine::Length() const {
    double dx = col2_ - col1_;
    double dy = row2_ - row1_;
    return std::sqrt(dx * dx + dy * dy);
}

double MetrologyObjectLine::Angle() const {
    return std::atan2(row2_ - row1_, col2_ - col1_);
}

std::vector<MeasureRectangle2> MetrologyObjectLine::GetCalipers() const {
    std::vector<MeasureRectangle2> calipers;
    int32_t numMeasures = params_.numMeasures;
    if (numMeasures < 1) numMeasures = 1;

    double length = Length();
    if (length < 1e-6) return calipers;

    // Line angle (perpendicular to profile direction)
    double lineAngle = Angle();
    // Profile is perpendicular to the line direction
    double profilePhi = lineAngle;  // Phi for rectangle2 (perpendicular to profile)

    // Distribute calipers evenly along the line
    for (int32_t i = 0; i < numMeasures; ++i) {
        double t = (numMeasures == 1) ? 0.5 : static_cast<double>(i) / (numMeasures - 1);
        double row = row1_ + t * (row2_ - row1_);
        double col = col1_ + t * (col2_ - col1_);

        MeasureRectangle2 caliper(row, col, profilePhi,
                                   params_.measureLength1, params_.measureLength2);
        calipers.push_back(caliper);
    }

    return calipers;
}

QContour MetrologyObjectLine::GetContour() const {
    QContour contour;
    contour.AddPoint(Point2d{col1_, row1_});
    contour.AddPoint(Point2d{col2_, row2_});
    return contour;
}

void MetrologyObjectLine::Transform(double rowOffset, double colOffset, double phi) {
    if (std::abs(phi) > 1e-9) {
        // Rotate around the line center
        double centerRow = (row1_ + row2_) * 0.5;
        double centerCol = (col1_ + col2_) * 0.5;

        Point2d p1 = RotatePoint({col1_, row1_}, phi, {centerCol, centerRow});
        Point2d p2 = RotatePoint({col2_, row2_}, phi, {centerCol, centerRow});

        row1_ = p1.y;
        col1_ = p1.x;
        row2_ = p2.y;
        col2_ = p2.x;
    }

    row1_ += rowOffset;
    col1_ += colOffset;
    row2_ += rowOffset;
    col2_ += colOffset;
}

// =============================================================================
// MetrologyObjectCircle Implementation
// =============================================================================

MetrologyObjectCircle::MetrologyObjectCircle(double row, double column, double radius,
                                              double measureLength1, double measureLength2,
                                              int32_t numMeasures)
    : row_(row), column_(column), radius_(radius),
      angleStart_(0.0), angleEnd_(TWO_PI) {
    params_.measureLength1 = measureLength1;
    params_.measureLength2 = measureLength2;
    params_.numMeasures = numMeasures;
}

MetrologyObjectCircle::MetrologyObjectCircle(double row, double column, double radius,
                                              double angleStart, double angleEnd,
                                              double measureLength1, double measureLength2,
                                              int32_t numMeasures)
    : row_(row), column_(column), radius_(radius),
      angleStart_(angleStart), angleEnd_(angleEnd) {
    params_.measureLength1 = measureLength1;
    params_.measureLength2 = measureLength2;
    params_.numMeasures = numMeasures;
}

bool MetrologyObjectCircle::IsFullCircle() const {
    return std::abs(angleEnd_ - angleStart_ - TWO_PI) < 1e-6 ||
           std::abs(angleEnd_ - angleStart_ + TWO_PI) < 1e-6;
}

std::vector<MeasureRectangle2> MetrologyObjectCircle::GetCalipers() const {
    std::vector<MeasureRectangle2> calipers;
    int32_t numMeasures = params_.numMeasures;
    if (numMeasures < 1) numMeasures = 1;

    double angleExtent = angleEnd_ - angleStart_;
    bool fullCircle = IsFullCircle();

    for (int32_t i = 0; i < numMeasures; ++i) {
        double t;
        if (fullCircle) {
            t = static_cast<double>(i) / numMeasures;  // Don't overlap at start/end
        } else {
            t = (numMeasures == 1) ? 0.5 : static_cast<double>(i) / (numMeasures - 1);
        }

        double angle = angleStart_ + t * angleExtent;

        // Position on circle
        double row = row_ + radius_ * std::sin(angle);
        double col = column_ + radius_ * std::cos(angle);

        // MeasureRectangle2::ProfileAngle() returns phi + PI/2
        // To get radial search direction, we need phi = angle - PI/2
        double profilePhi = angle - PI / 2.0;  // Radial direction after ProfileAngle transform

        MeasureRectangle2 caliper(row, col, profilePhi,
                                   params_.measureLength1, params_.measureLength2);
        calipers.push_back(caliper);
    }

    return calipers;
}

QContour MetrologyObjectCircle::GetContour() const {
    QContour contour;
    int32_t numPoints = std::max(32, static_cast<int32_t>(radius_ * 0.5));
    double angleExtent = angleEnd_ - angleStart_;

    for (int32_t i = 0; i <= numPoints; ++i) {
        double t = static_cast<double>(i) / numPoints;
        double angle = angleStart_ + t * angleExtent;
        double x = column_ + radius_ * std::cos(angle);
        double y = row_ + radius_ * std::sin(angle);
        contour.AddPoint(Point2d{x, y});
    }

    return contour;
}

void MetrologyObjectCircle::Transform(double rowOffset, double colOffset, double phi) {
    if (std::abs(phi) > 1e-9) {
        // Rotate angles
        angleStart_ += phi;
        angleEnd_ += phi;
    }

    row_ += rowOffset;
    column_ += colOffset;
}

// =============================================================================
// MetrologyObjectEllipse Implementation
// =============================================================================

MetrologyObjectEllipse::MetrologyObjectEllipse(double row, double column, double phi,
                                                double ra, double rb,
                                                double measureLength1, double measureLength2,
                                                int32_t numMeasures)
    : row_(row), column_(column), phi_(phi), ra_(ra), rb_(rb) {
    params_.measureLength1 = measureLength1;
    params_.measureLength2 = measureLength2;
    params_.numMeasures = numMeasures;
}

std::vector<MeasureRectangle2> MetrologyObjectEllipse::GetCalipers() const {
    std::vector<MeasureRectangle2> calipers;
    int32_t numMeasures = params_.numMeasures;
    if (numMeasures < 1) numMeasures = 1;

    double cosPhi = std::cos(phi_);
    double sinPhi = std::sin(phi_);

    for (int32_t i = 0; i < numMeasures; ++i) {
        double t = static_cast<double>(i) / numMeasures;  // Full ellipse
        double angle = t * TWO_PI;

        // Point on ellipse (local coordinates)
        double localX = ra_ * std::cos(angle);
        double localY = rb_ * std::sin(angle);

        // Rotate by phi and translate
        double col = column_ + localX * cosPhi - localY * sinPhi;
        double row = row_ + localX * sinPhi + localY * cosPhi;

        // Normal direction at this point (gradient of ellipse equation)
        // For ellipse: (x/ra)^2 + (y/rb)^2 = 1
        // Gradient: (2x/ra^2, 2y/rb^2) -> normal angle
        double gradX = localX / (ra_ * ra_);
        double gradY = localY / (rb_ * rb_);

        // Rotate gradient by phi
        double normalX = gradX * cosPhi - gradY * sinPhi;
        double normalY = gradX * sinPhi + gradY * cosPhi;
        double profilePhi = std::atan2(normalY, normalX);

        MeasureRectangle2 caliper(row, col, profilePhi,
                                   params_.measureLength1, params_.measureLength2);
        calipers.push_back(caliper);
    }

    return calipers;
}

QContour MetrologyObjectEllipse::GetContour() const {
    QContour contour;
    int32_t numPoints = std::max(32, static_cast<int32_t>((ra_ + rb_) * 0.5));

    double cosPhi = std::cos(phi_);
    double sinPhi = std::sin(phi_);

    for (int32_t i = 0; i <= numPoints; ++i) {
        double t = static_cast<double>(i) / numPoints;
        double angle = t * TWO_PI;

        double localX = ra_ * std::cos(angle);
        double localY = rb_ * std::sin(angle);

        double x = column_ + localX * cosPhi - localY * sinPhi;
        double y = row_ + localX * sinPhi + localY * cosPhi;
        contour.AddPoint(Point2d{x, y});
    }

    return contour;
}

void MetrologyObjectEllipse::Transform(double rowOffset, double colOffset, double phi) {
    if (std::abs(phi) > 1e-9) {
        phi_ += phi;
    }

    row_ += rowOffset;
    column_ += colOffset;
}

// =============================================================================
// MetrologyObjectRectangle2 Implementation
// =============================================================================

MetrologyObjectRectangle2::MetrologyObjectRectangle2(double row, double column, double phi,
                                                       double length1, double length2,
                                                       double measureLength1, double measureLength2,
                                                       int32_t numMeasuresPerSide)
    : row_(row), column_(column), phi_(phi), length1_(length1), length2_(length2) {
    params_.measureLength1 = measureLength1;
    params_.measureLength2 = measureLength2;
    // Total numMeasures = 4 * numMeasuresPerSide (one per side)
    params_.numMeasures = 4 * numMeasuresPerSide;
}

std::vector<MeasureRectangle2> MetrologyObjectRectangle2::GetCalipers() const {
    std::vector<MeasureRectangle2> calipers;
    int32_t numMeasures = params_.numMeasures;
    if (numMeasures < 1) numMeasures = 4;  // At least one per side

    double cosPhi = std::cos(phi_);
    double sinPhi = std::sin(phi_);

    // Distribute calipers along 4 sides
    int32_t perSide = std::max(1, numMeasures / 4);

    // Corners (local coordinates)
    std::vector<std::pair<Point2d, Point2d>> sides = {
        // Side 1: top (from corner to corner)
        {{-length1_, -length2_}, {length1_, -length2_}},
        // Side 2: right
        {{length1_, -length2_}, {length1_, length2_}},
        // Side 3: bottom
        {{length1_, length2_}, {-length1_, length2_}},
        // Side 4: left
        {{-length1_, length2_}, {-length1_, -length2_}}
    };

    // Edge directions for each side (caliper phi should be edge direction,
    // so profile direction = phi + 90° is perpendicular to edge)
    std::vector<double> edgeAngles = {
        phi_,              // Top: horizontal edge, profile perpendicular (vertical)
        phi_ + PI * 0.5,   // Right: vertical edge, profile perpendicular (horizontal)
        phi_,              // Bottom: horizontal edge, profile perpendicular (vertical)
        phi_ + PI * 0.5    // Left: vertical edge, profile perpendicular (horizontal)
    };

    for (size_t side = 0; side < 4; ++side) {
        auto& s = sides[side];

        for (int32_t i = 0; i < perSide; ++i) {
            double t = (perSide == 1) ? 0.5 : static_cast<double>(i) / (perSide - 1);

            // Interpolate along side
            double localX = s.first.x + t * (s.second.x - s.first.x);
            double localY = s.first.y + t * (s.second.y - s.first.y);

            // Transform to image coordinates
            double col = column_ + localX * cosPhi - localY * sinPhi;
            double row = row_ + localX * sinPhi + localY * cosPhi;

            MeasureRectangle2 caliper(row, col, edgeAngles[side],
                                       params_.measureLength1, params_.measureLength2);
            calipers.push_back(caliper);
        }
    }

    return calipers;
}

QContour MetrologyObjectRectangle2::GetContour() const {
    QContour contour;

    double cosPhi = std::cos(phi_);
    double sinPhi = std::sin(phi_);

    // 4 corners
    std::vector<std::pair<double, double>> corners = {
        {-length1_, -length2_},
        {length1_, -length2_},
        {length1_, length2_},
        {-length1_, length2_},
        {-length1_, -length2_}  // Close the contour
    };

    for (auto& c : corners) {
        double x = column_ + c.first * cosPhi - c.second * sinPhi;
        double y = row_ + c.first * sinPhi + c.second * cosPhi;
        contour.AddPoint(Point2d{x, y});
    }

    return contour;
}

void MetrologyObjectRectangle2::Transform(double rowOffset, double colOffset, double phi) {
    if (std::abs(phi) > 1e-9) {
        phi_ += phi;
    }

    row_ += rowOffset;
    column_ += colOffset;
}

// =============================================================================
// MetrologyModel Implementation
// =============================================================================

struct MetrologyModel::Impl {
    std::vector<std::unique_ptr<MetrologyObject>> objects;
    MetrologyMeasureParams defaultParams;

    // Per-object results
    std::unordered_map<int32_t, std::vector<MetrologyLineResult>> lineResults;
    std::unordered_map<int32_t, std::vector<MetrologyCircleResult>> circleResults;
    std::unordered_map<int32_t, std::vector<MetrologyEllipseResult>> ellipseResults;
    std::unordered_map<int32_t, std::vector<MetrologyRectangle2Result>> rectangleResults;

    // Per-object measured points
    std::unordered_map<int32_t, std::vector<Point2d>> measuredPoints;

    // Per-object point weights (from robust fitting, e.g., Huber/Tukey)
    std::unordered_map<int32_t, std::vector<double>> pointWeights;

    // Alignment state
    double alignRow = 0.0;
    double alignCol = 0.0;
    double alignPhi = 0.0;

    void ClearResults() {
        lineResults.clear();
        circleResults.clear();
        ellipseResults.clear();
        rectangleResults.clear();
        measuredPoints.clear();
        pointWeights.clear();
    }
};

MetrologyModel::MetrologyModel() : impl_(std::make_unique<Impl>()) {}

MetrologyModel::~MetrologyModel() = default;

MetrologyModel::MetrologyModel(MetrologyModel&&) noexcept = default;
MetrologyModel& MetrologyModel::operator=(MetrologyModel&&) noexcept = default;

int32_t MetrologyModel::AddLineMeasure(double row1, double col1,
                                         double row2, double col2,
                                         double measureLength1, double measureLength2,
                                         const std::string& transition,
                                         const std::string& select,
                                         const MetrologyMeasureParams& params) {
    constexpr const char* funcName = "AddLineMeasure";
    if (!std::isfinite(row1) || !std::isfinite(col1) ||
        !std::isfinite(row2) || !std::isfinite(col2)) {
        throw InvalidArgumentException(std::string(funcName) + ": invalid points");
    }
    if (std::abs(row1 - row2) < 1e-12 && std::abs(col1 - col2) < 1e-12) {
        throw InvalidArgumentException(std::string(funcName) + ": points must be distinct");
    }
    // measureLength and numMeasures validated in MergeParams
    auto merged = MergeParams(measureLength1, measureLength2, transition, select, params, funcName);
    auto obj = std::make_unique<MetrologyObjectLine>(row1, col1, row2, col2,
                                                      measureLength1, measureLength2,
                                                      merged.numMeasures);
    // Copy merged params
    obj->params_ = merged;
    int32_t idx = static_cast<int32_t>(impl_->objects.size());
    obj->index_ = idx;
    impl_->objects.push_back(std::move(obj));
    return idx;
}

int32_t MetrologyModel::AddCircleMeasure(double row, double column, double radius,
                                          double measureLength1, double measureLength2,
                                          const std::string& transition,
                                          const std::string& select,
                                          const MetrologyMeasureParams& params) {
    constexpr const char* funcName = "AddCircleMeasure";
    if (!std::isfinite(row) || !std::isfinite(column) || !std::isfinite(radius)) {
        throw InvalidArgumentException(std::string(funcName) + ": invalid parameters");
    }
    if (radius <= 0.0) {
        throw InvalidArgumentException(std::string(funcName) + ": radius must be > 0");
    }
    // measureLength and numMeasures validated in MergeParams
    auto merged = MergeParams(measureLength1, measureLength2, transition, select, params, funcName);
    auto obj = std::make_unique<MetrologyObjectCircle>(row, column, radius,
                                                        measureLength1, measureLength2,
                                                        merged.numMeasures);
    // Copy merged params
    obj->params_ = merged;
    int32_t idx = static_cast<int32_t>(impl_->objects.size());
    obj->index_ = idx;
    impl_->objects.push_back(std::move(obj));
    return idx;
}

int32_t MetrologyModel::AddArcMeasure(double row, double column, double radius,
                                        double angleStart, double angleEnd,
                                        double measureLength1, double measureLength2,
                                        const std::string& transition,
                                        const std::string& select,
                                        const MetrologyMeasureParams& params) {
    constexpr const char* funcName = "AddArcMeasure";
    if (!std::isfinite(row) || !std::isfinite(column) || !std::isfinite(radius) ||
        !std::isfinite(angleStart) || !std::isfinite(angleEnd)) {
        throw InvalidArgumentException(std::string(funcName) + ": invalid parameters");
    }
    if (radius <= 0.0) {
        throw InvalidArgumentException(std::string(funcName) + ": radius must be > 0");
    }
    if (std::abs(angleEnd - angleStart) < 1e-9) {
        throw InvalidArgumentException(std::string(funcName) + ": angle range must be non-zero");
    }
    // measureLength and numMeasures validated in MergeParams
    auto merged = MergeParams(measureLength1, measureLength2, transition, select, params, funcName);
    auto obj = std::make_unique<MetrologyObjectCircle>(row, column, radius,
                                                        angleStart, angleEnd,
                                                        measureLength1, measureLength2,
                                                        merged.numMeasures);
    // Copy merged params
    obj->params_ = merged;
    int32_t idx = static_cast<int32_t>(impl_->objects.size());
    obj->index_ = idx;
    impl_->objects.push_back(std::move(obj));
    return idx;
}

int32_t MetrologyModel::AddEllipseMeasure(double row, double column, double phi,
                                           double ra, double rb,
                                           double measureLength1, double measureLength2,
                                           const std::string& transition,
                                           const std::string& select,
                                           const MetrologyMeasureParams& params) {
    constexpr const char* funcName = "AddEllipseMeasure";
    if (!std::isfinite(row) || !std::isfinite(column) || !std::isfinite(phi) ||
        !std::isfinite(ra) || !std::isfinite(rb)) {
        throw InvalidArgumentException(std::string(funcName) + ": invalid parameters");
    }
    if (ra <= 0.0 || rb <= 0.0) {
        throw InvalidArgumentException(std::string(funcName) + ": radii must be > 0");
    }
    // measureLength and numMeasures validated in MergeParams
    auto merged = MergeParams(measureLength1, measureLength2, transition, select, params, funcName);
    auto obj = std::make_unique<MetrologyObjectEllipse>(row, column, phi, ra, rb,
                                                         measureLength1, measureLength2,
                                                         merged.numMeasures);
    // Copy merged params
    obj->params_ = merged;
    int32_t idx = static_cast<int32_t>(impl_->objects.size());
    obj->index_ = idx;
    impl_->objects.push_back(std::move(obj));
    return idx;
}

int32_t MetrologyModel::AddRectangle2Measure(double row, double column, double phi,
                                               double length1, double length2,
                                               double measureLength1, double measureLength2,
                                               const std::string& transition,
                                               const std::string& select,
                                               const MetrologyMeasureParams& params) {
    constexpr const char* funcName = "AddRectangle2Measure";
    if (!std::isfinite(row) || !std::isfinite(column) || !std::isfinite(phi) ||
        !std::isfinite(length1) || !std::isfinite(length2)) {
        throw InvalidArgumentException(std::string(funcName) + ": invalid parameters");
    }
    if (length1 <= 0.0 || length2 <= 0.0) {
        throw InvalidArgumentException(std::string(funcName) + ": lengths must be > 0");
    }
    // measureLength and numMeasures validated in MergeParams
    auto merged = MergeParams(measureLength1, measureLength2, transition, select, params, funcName);
    // For rectangle, distribute numMeasures across 4 sides fairly
    // Base per-side count + distribute remainder to longer sides first
    int32_t total = merged.numMeasures;
    int32_t base = total / 4;
    int32_t remainder = total % 4;
    // Ensure at least 1 per side; distribute remainder: 2 to length1 sides, then length2
    int32_t numMeasuresPerSide = std::max(1, base + (remainder >= 2 ? 1 : 0));
    auto obj = std::make_unique<MetrologyObjectRectangle2>(row, column, phi, length1, length2,
                                                            measureLength1, measureLength2,
                                                            numMeasuresPerSide);
    // Copy merged params (actual total = numMeasuresPerSide * 4)
    obj->params_ = merged;
    int32_t idx = static_cast<int32_t>(impl_->objects.size());
    obj->index_ = idx;
    impl_->objects.push_back(std::move(obj));
    return idx;
}

void MetrologyModel::ClearObject(int32_t index) {
    if (index < 0 || index >= static_cast<int32_t>(impl_->objects.size())) {
        throw InvalidArgumentException("ClearObject: index out of range");
    }
    impl_->objects[index].reset();
}

void MetrologyModel::ClearAll() {
    impl_->objects.clear();
    impl_->ClearResults();
}

int32_t MetrologyModel::NumObjects() const {
    return static_cast<int32_t>(impl_->objects.size());
}

const MetrologyObject* MetrologyModel::GetObject(int32_t index) const {
    if (index < 0 || index >= static_cast<int32_t>(impl_->objects.size())) {
        throw InvalidArgumentException("GetObject: index out of range");
    }
    auto* obj = impl_->objects[index].get();
    if (!obj) {
        throw InvalidArgumentException("GetObject: object cleared");
    }
    return obj;
}

bool MetrologyModel::Apply(const QImage& image) {
    if (!Validate::RequireImageValid(image, "Apply")) {
        impl_->ClearResults();
        return false;
    }

    impl_->ClearResults();

    // Convert to grayscale UInt8 for consistent measurement
    QImage grayImage = image;
    if (image.Channels() > 1) {
        grayImage = image.ToGray();
    }
    if (grayImage.Type() != PixelType::UInt8) {
        grayImage = grayImage.ConvertTo(PixelType::UInt8);
    }

    for (auto& objPtr : impl_->objects) {
        if (!objPtr) continue;

        auto& obj = *objPtr;
        int32_t idx = obj.Index();

        // Get calipers
        auto calipers = obj.GetCalipers();

        // Measure edge positions
        std::vector<Point2d> edgePoints;
        std::vector<double> edgeScores;
        double sigma = obj.Params().measureSigma;
        double userThreshold = obj.Params().measureThreshold;
        ThresholdMode thresholdMode = obj.Params().thresholdMode;

        EdgeTransition measureTransition = obj.Params().measureTransition;
        EdgeSelectMode measureSelect = obj.Params().measureSelect;

        int caliperIdx = 0;
        for (auto& caliper : calipers) {
            // Extract profile directly for adaptive threshold calculation
            Internal::RectProfileParams profParams;
            profParams.centerX = caliper.Column();
            profParams.centerY = caliper.Row();
            profParams.length = caliper.ProfileLength();
            profParams.width = 2.0 * caliper.Length2();
            profParams.angle = caliper.ProfileAngle();
            profParams.numLines = caliper.NumLines();
            profParams.samplesPerPixel = caliper.SamplesPerPixel();
            profParams.interp = Internal::InterpolationMethod::Bilinear;
            profParams.method = Internal::ProfileMethod::Average;

            auto profile = Internal::ExtractRectProfile(grayImage, profParams);
            if (profile.data.size() < 3) continue;

            // Determine threshold for edge detection
            // Each profile region computes its own threshold independently
            double threshold;
            if (thresholdMode == ThresholdMode::Manual) {
                // User specified threshold - use directly
                threshold = userThreshold;
            } else {
                // Auto threshold mode: compute based on this profile's statistics
                const auto& data = profile.data;
                size_t n = data.size();

                // 1. Compute contrast (max - min)
                double minVal = *std::min_element(data.begin(), data.end());
                double maxVal = *std::max_element(data.begin(), data.end());
                double contrast = maxVal - minVal;

                // 2. Compute gradient (central difference)
                std::vector<double> gradient(n);
                gradient[0] = data[1] - data[0];
                for (size_t i = 1; i < n - 1; ++i) {
                    gradient[i] = (data[i + 1] - data[i - 1]) * 0.5;
                }
                gradient[n - 1] = data[n - 1] - data[n - 2];

                // 3. Estimate noise using MAD (Median Absolute Deviation)
                //    More robust than standard deviation
                std::vector<double> absGrad(n);
                for (size_t i = 0; i < n; ++i) {
                    absGrad[i] = std::abs(gradient[i]);
                }
                std::nth_element(absGrad.begin(), absGrad.begin() + n / 2, absGrad.end());
                double medianAbsGrad = absGrad[n / 2];
                double noiseSigma = medianAbsGrad / 0.6745;  // MAD to sigma conversion

                // 4. Compute threshold = max(base, contrast*ratio, k*noise)
                constexpr double BASE_THRESHOLD = 5.0;      // Absolute minimum
                constexpr double CONTRAST_RATIO = 0.2;      // 20% of contrast
                constexpr double NOISE_MULTIPLIER = 4.0;    // 4 sigma confidence

                double contrastThreshold = contrast * CONTRAST_RATIO;
                double noiseThreshold = noiseSigma * NOISE_MULTIPLIER;
                threshold = std::max({BASE_THRESHOLD, contrastThreshold, noiseThreshold});
            }

            // Detect edges with threshold
            auto edges1D = Internal::DetectEdges1D(
                profile.data.data(),
                profile.data.size(),
                threshold,
                ToEdgePolarity(measureTransition),
                sigma
            );

            if (edges1D.empty()) {
                caliperIdx++;
                continue;
            }

            std::vector<size_t> selectedIndices;
            selectedIndices.reserve(edges1D.size());

            switch (measureSelect) {
                case EdgeSelectMode::All:
                    for (size_t i = 0; i < edges1D.size(); ++i) {
                        selectedIndices.push_back(i);
                    }
                    break;
                case EdgeSelectMode::First: {
                    auto it = std::min_element(edges1D.begin(), edges1D.end(),
                        [](const auto& a, const auto& b) { return a.position < b.position; });
                    if (it != edges1D.end()) selectedIndices.push_back(static_cast<size_t>(std::distance(edges1D.begin(), it)));
                    break;
                }
                case EdgeSelectMode::Last: {
                    auto it = std::max_element(edges1D.begin(), edges1D.end(),
                        [](const auto& a, const auto& b) { return a.position < b.position; });
                    if (it != edges1D.end()) selectedIndices.push_back(static_cast<size_t>(std::distance(edges1D.begin(), it)));
                    break;
                }
                case EdgeSelectMode::Weakest: {
                    auto it = std::min_element(edges1D.begin(), edges1D.end(),
                        [](const auto& a, const auto& b) {
                            return std::abs(a.amplitude) < std::abs(b.amplitude);
                        });
                    if (it != edges1D.end()) selectedIndices.push_back(static_cast<size_t>(std::distance(edges1D.begin(), it)));
                    break;
                }
                case EdgeSelectMode::Strongest:
                default: {
                    auto it = std::max_element(edges1D.begin(), edges1D.end(),
                        [](const auto& a, const auto& b) {
                            return std::abs(a.amplitude) < std::abs(b.amplitude);
                        });
                    if (it != edges1D.end()) selectedIndices.push_back(static_cast<size_t>(std::distance(edges1D.begin(), it)));
                    break;
                }
            }

            // Convert selected profile positions to image coordinates
            double profileLength = caliper.ProfileLength();
            double stepSize = profileLength / (profile.data.size() - 1);
            double profileAngle = caliper.ProfileAngle();
            double halfLen = caliper.Length1();
            double startX = caliper.Column() - halfLen * std::cos(profileAngle);
            double startY = caliper.Row() - halfLen * std::sin(profileAngle);

            for (size_t idxSel : selectedIndices) {
                if (idxSel >= edges1D.size()) continue;
                const auto& sel = edges1D[idxSel];
                double profilePos = sel.position * stepSize;
                double edgeX = startX + profilePos * std::cos(profileAngle);
                double edgeY = startY + profilePos * std::sin(profileAngle);
                edgePoints.push_back({edgeX, edgeY});
                edgeScores.push_back(std::abs(sel.amplitude));
            }
            caliperIdx++;
        }

        impl_->measuredPoints[idx] = edgePoints;

        // Get fitting parameters
        auto fitMethod = obj.Params().fitMethod;
        double distThreshold = obj.Params().distanceThreshold;
        int32_t maxIter = obj.Params().maxIterations;
        int32_t numInstances = std::max(1, obj.Params().numInstances);
        double minScore = obj.Params().minScore;
        double minCoverage = obj.Params().minCoverage;
        double maxRmsError = obj.Params().maxRmsError;
        double totalMeasures = static_cast<double>(std::max(1, obj.Params().numMeasures));
        auto isAccepted = [&](int32_t numUsed, double rms, double score) -> bool {
            double coverage = static_cast<double>(std::max(0, numUsed)) / totalMeasures;
            if (score < minScore) return false;
            if (coverage < minCoverage) return false;
            if (maxRmsError > 0.0 && rms > maxRmsError) return false;
            return true;
        };

        // Setup RANSAC parameters
        Internal::RansacParams ransacParams;
        ransacParams.threshold = distThreshold;
        ransacParams.maxIterations = (maxIter < 0) ? 10000 : maxIter;  // -1 means high limit
        ransacParams.confidence = 0.99;

        // Setup fit params
        Internal::FitParams fitParams;
        fitParams.computeResiduals = true;
        fitParams.computeInlierMask = true;  // For outlier visualization

        // Fit geometric primitive based on object type
        switch (obj.Type()) {
            case MetrologyObjectType::Line: {
                auto fitLine = [&](const std::vector<Point2d>& pts) -> Internal::LineFitResult {
                    switch (fitMethod) {
                        case MetrologyFitMethod::RANSAC:
                            return Internal::FitLineRANSAC(pts, ransacParams, fitParams);
                        case MetrologyFitMethod::Huber:
                            return Internal::FitLineHuber(pts, 0.0, fitParams);
                        case MetrologyFitMethod::Tukey:
                            return Internal::FitLineTukey(pts, 0.0, fitParams);
                    }
                    return {};
                };

                std::vector<Point2d> remainPoints = edgePoints;
                std::vector<double> remainScores = edgeScores;
                std::vector<size_t> remainMap(remainPoints.size());
                std::iota(remainMap.begin(), remainMap.end(), 0);

                for (int32_t inst = 0; inst < numInstances; ++inst) {
                    if (remainPoints.size() < 2) break;

                    std::vector<Point2d> fitPoints = remainPoints;
                    std::vector<double> fitScores = remainScores;
                    std::vector<size_t> keptIndices(fitPoints.size());
                    std::iota(keptIndices.begin(), keptIndices.end(), 0);
                    Internal::LineFitResult fitResult = fitLine(fitPoints);

                    if (fitResult.success && fitResult.residuals.size() == fitPoints.size()) {
                        double adaptiveThreshold =
                            ComputeAdaptiveResidualThreshold(fitResult.residuals, distThreshold);
                        if (FilterByResidualThreshold(fitResult.residuals, adaptiveThreshold, 2,
                                                      fitPoints, fitScores, keptIndices)) {
                            auto refit = fitLine(fitPoints);
                            if (refit.success) {
                                fitResult = std::move(refit);
                            }
                        }
                    }

                    if (fitResult.success && obj.Params().ignorePointCount > 0 && fitPoints.size() >= 2) {
                        IgnoreSelection sel = SelectPointsForIgnore(
                            fitPoints,
                            fitScores,
                            obj.Params().ignorePointCount,
                            obj.Params().ignorePointPolicy,
                            fitResult.residuals,
                            2
                        );
                        if (sel.applied) {
                            auto refit = fitLine(sel.points);
                            if (refit.success) {
                                fitResult = std::move(refit);
                                std::vector<size_t> remapped(sel.keptIndices.size(), 0);
                                for (size_t i = 0; i < sel.keptIndices.size(); ++i) {
                                    if (sel.keptIndices[i] < keptIndices.size()) {
                                        remapped[i] = keptIndices[sel.keptIndices[i]];
                                    }
                                }
                                keptIndices = std::move(remapped);
                                fitPoints = std::move(sel.points);
                                fitScores = std::move(sel.scores);
                            }
                        }
                    }

                    if (!fitResult.success) break;

                    MetrologyLineResult result;
                    auto& line = fitResult.line;
                    auto* lineObj = static_cast<MetrologyObjectLine*>(&obj);
                    Point2d p1 = {lineObj->Col1(), lineObj->Row1()};
                    Point2d p2 = {lineObj->Col2(), lineObj->Row2()};
                    auto project = [&line](const Point2d& p) -> Point2d {
                        double d = line.a * p.x + line.b * p.y + line.c;
                        return {p.x - d * line.a, p.y - d * line.b};
                    };

                    Point2d proj1 = project(p1);
                    Point2d proj2 = project(p2);
                    result.row1 = proj1.y;
                    result.col1 = proj1.x;
                    result.row2 = proj2.y;
                    result.col2 = proj2.x;
                    result.nr = line.b;
                    result.nc = line.a;
                    result.dist = -line.c;
                    result.numUsed = fitResult.numInliers > 0 ? fitResult.numInliers : static_cast<int>(keptIndices.size());
                    result.rmsError = fitResult.residualRMS;
                    result.score = result.numUsed > 0 ? 1.0 / (1.0 + result.rmsError) : 0.0;
                    bool accepted = isAccepted(result.numUsed, result.rmsError, result.score);
                    if (accepted) {
                        impl_->lineResults[idx].push_back(result);
                    }

                    std::vector<size_t> mappedKept;
                    mappedKept.reserve(keptIndices.size());
                    for (size_t k : keptIndices) {
                        if (k < remainMap.size()) mappedKept.push_back(remainMap[k]);
                    }
                    auto displayWeights = BuildWeightsForDisplay(fitResult, mappedKept, edgePoints.size());
                    if (accepted && !displayWeights.empty() && impl_->pointWeights.find(idx) == impl_->pointWeights.end()) {
                        impl_->pointWeights[idx] = std::move(displayWeights);
                    }

                    double instanceThreshold =
                        ComputeAdaptiveResidualThreshold(fitResult.residuals, distThreshold);
                    auto compactMask = BuildInstanceCompactMask(fitResult, keptIndices.size(), instanceThreshold, 2);
                    if (compactMask.empty()) break;
                    std::vector<bool> removeMask(remainPoints.size(), false);
                    for (size_t i = 0; i < keptIndices.size() && i < compactMask.size(); ++i) {
                        if (compactMask[i] && keptIndices[i] < removeMask.size()) {
                            removeMask[keptIndices[i]] = true;
                        }
                    }
                    RemoveMaskedPoints(remainPoints, remainScores, remainMap, removeMask);
                }
                break;
            }

            case MetrologyObjectType::Circle: {
                auto fitCircle = [&](const std::vector<Point2d>& pts) -> Internal::CircleFitResult {
                    switch (fitMethod) {
                        case MetrologyFitMethod::RANSAC:
                            return Internal::FitCircleRANSAC(pts, ransacParams, fitParams);
                        case MetrologyFitMethod::Huber:
                            return Internal::FitCircleHuber(pts, true, 0.0, fitParams);
                        case MetrologyFitMethod::Tukey:
                            return Internal::FitCircleTukey(pts, true, 0.0, fitParams);
                    }
                    return {};
                };

                std::vector<Point2d> remainPoints = edgePoints;
                std::vector<double> remainScores = edgeScores;
                std::vector<size_t> remainMap(remainPoints.size());
                std::iota(remainMap.begin(), remainMap.end(), 0);

                for (int32_t inst = 0; inst < numInstances; ++inst) {
                    if (remainPoints.size() < 3) break;

                    std::vector<Point2d> fitPoints = remainPoints;
                    std::vector<double> fitScores = remainScores;
                    std::vector<size_t> keptIndices(fitPoints.size());
                    std::iota(keptIndices.begin(), keptIndices.end(), 0);
                    Internal::CircleFitResult fitResult = fitCircle(fitPoints);

                    if (fitResult.success && fitResult.residuals.size() == fitPoints.size()) {
                        double adaptiveThreshold =
                            ComputeAdaptiveResidualThreshold(fitResult.residuals, distThreshold);
                        if (FilterByResidualThreshold(fitResult.residuals, adaptiveThreshold, 3,
                                                      fitPoints, fitScores, keptIndices)) {
                            auto refit = fitCircle(fitPoints);
                            if (refit.success) {
                                fitResult = std::move(refit);
                            }
                        }
                    }

                    if (fitResult.success && obj.Params().ignorePointCount > 0 && fitPoints.size() >= 3) {
                        IgnoreSelection sel = SelectPointsForIgnore(
                            fitPoints,
                            fitScores,
                            obj.Params().ignorePointCount,
                            obj.Params().ignorePointPolicy,
                            fitResult.residuals,
                            3
                        );
                        if (sel.applied) {
                            auto refit = fitCircle(sel.points);
                            if (refit.success) {
                                fitResult = std::move(refit);
                                std::vector<size_t> remapped(sel.keptIndices.size(), 0);
                                for (size_t i = 0; i < sel.keptIndices.size(); ++i) {
                                    if (sel.keptIndices[i] < keptIndices.size()) {
                                        remapped[i] = keptIndices[sel.keptIndices[i]];
                                    }
                                }
                                keptIndices = std::move(remapped);
                                fitPoints = std::move(sel.points);
                                fitScores = std::move(sel.scores);
                            }
                        }
                    }

                    if (!fitResult.success) break;

                    MetrologyCircleResult result;
                    result.row = fitResult.circle.center.y;
                    result.column = fitResult.circle.center.x;
                    result.radius = fitResult.circle.radius;
                    result.numUsed = fitResult.numInliers > 0 ? fitResult.numInliers : static_cast<int32_t>(keptIndices.size());
                    result.rmsError = fitResult.residualRMS;
                    result.score = result.numUsed > 0 ? 1.0 / (1.0 + result.rmsError) : 0.0;

                    auto* circleObj = static_cast<MetrologyObjectCircle*>(&obj);
                    result.startAngle = circleObj->AngleStart();
                    result.endAngle = circleObj->AngleEnd();
                    bool accepted = isAccepted(result.numUsed, result.rmsError, result.score);
                    if (accepted) {
                        impl_->circleResults[idx].push_back(result);
                    }

                    std::vector<size_t> mappedKept;
                    mappedKept.reserve(keptIndices.size());
                    for (size_t k : keptIndices) {
                        if (k < remainMap.size()) mappedKept.push_back(remainMap[k]);
                    }
                    auto displayWeights = BuildWeightsForDisplay(fitResult, mappedKept, edgePoints.size());
                    if (accepted && !displayWeights.empty() && impl_->pointWeights.find(idx) == impl_->pointWeights.end()) {
                        impl_->pointWeights[idx] = std::move(displayWeights);
                    }

                    double instanceThreshold =
                        ComputeAdaptiveResidualThreshold(fitResult.residuals, distThreshold);
                    auto compactMask = BuildInstanceCompactMask(fitResult, keptIndices.size(), instanceThreshold, 3);
                    if (compactMask.empty()) break;
                    std::vector<bool> removeMask(remainPoints.size(), false);
                    for (size_t i = 0; i < keptIndices.size() && i < compactMask.size(); ++i) {
                        if (compactMask[i] && keptIndices[i] < removeMask.size()) {
                            removeMask[keptIndices[i]] = true;
                        }
                    }
                    RemoveMaskedPoints(remainPoints, remainScores, remainMap, removeMask);
                }
                break;
            }

            case MetrologyObjectType::Ellipse: {
                auto fitEllipse = [&](const std::vector<Point2d>& pts) -> Internal::EllipseFitResult {
                    switch (fitMethod) {
                        case MetrologyFitMethod::RANSAC:
                            return Internal::FitEllipseRANSAC(pts, ransacParams, fitParams);
                        case MetrologyFitMethod::Huber:
                            return Internal::FitEllipseHuber(pts, 0.0, fitParams);
                        case MetrologyFitMethod::Tukey:
                            return Internal::FitEllipseTukey(pts, 0.0, fitParams);
                    }
                    return {};
                };

                std::vector<Point2d> remainPoints = edgePoints;
                std::vector<double> remainScores = edgeScores;
                std::vector<size_t> remainMap(remainPoints.size());
                std::iota(remainMap.begin(), remainMap.end(), 0);

                for (int32_t inst = 0; inst < numInstances; ++inst) {
                    if (remainPoints.size() < 5) break;

                    std::vector<Point2d> fitPoints = remainPoints;
                    std::vector<double> fitScores = remainScores;
                    std::vector<size_t> keptIndices(fitPoints.size());
                    std::iota(keptIndices.begin(), keptIndices.end(), 0);
                    Internal::EllipseFitResult fitResult = fitEllipse(fitPoints);

                    if (fitResult.success && fitResult.residuals.size() == fitPoints.size()) {
                        double adaptiveThreshold =
                            ComputeAdaptiveResidualThreshold(fitResult.residuals, distThreshold);
                        if (FilterByResidualThreshold(fitResult.residuals, adaptiveThreshold, 5,
                                                      fitPoints, fitScores, keptIndices)) {
                            auto refit = fitEllipse(fitPoints);
                            if (refit.success) {
                                fitResult = std::move(refit);
                            }
                        }
                    }

                    if (fitResult.success && obj.Params().ignorePointCount > 0 && fitPoints.size() >= 5) {
                        IgnoreSelection sel = SelectPointsForIgnore(
                            fitPoints,
                            fitScores,
                            obj.Params().ignorePointCount,
                            obj.Params().ignorePointPolicy,
                            fitResult.residuals,
                            5
                        );
                        if (sel.applied) {
                            auto refit = fitEllipse(sel.points);
                            if (refit.success) {
                                fitResult = std::move(refit);
                                std::vector<size_t> remapped(sel.keptIndices.size(), 0);
                                for (size_t i = 0; i < sel.keptIndices.size(); ++i) {
                                    if (sel.keptIndices[i] < keptIndices.size()) {
                                        remapped[i] = keptIndices[sel.keptIndices[i]];
                                    }
                                }
                                keptIndices = std::move(remapped);
                                fitPoints = std::move(sel.points);
                                fitScores = std::move(sel.scores);
                            }
                        }
                    }

                    if (!fitResult.success) break;

                    MetrologyEllipseResult result;
                    result.row = fitResult.ellipse.center.y;
                    result.column = fitResult.ellipse.center.x;
                    result.phi = fitResult.ellipse.angle;
                    result.ra = fitResult.ellipse.a;
                    result.rb = fitResult.ellipse.b;
                    result.numUsed = fitResult.numInliers > 0 ? fitResult.numInliers : static_cast<int32_t>(keptIndices.size());
                    result.rmsError = fitResult.residualRMS;
                    result.score = result.numUsed > 0 ? 1.0 / (1.0 + result.rmsError) : 0.0;
                    bool accepted = isAccepted(result.numUsed, result.rmsError, result.score);
                    if (accepted) {
                        impl_->ellipseResults[idx].push_back(result);
                    }

                    std::vector<size_t> mappedKept;
                    mappedKept.reserve(keptIndices.size());
                    for (size_t k : keptIndices) {
                        if (k < remainMap.size()) mappedKept.push_back(remainMap[k]);
                    }
                    auto displayWeights = BuildWeightsForDisplay(fitResult, mappedKept, edgePoints.size());
                    if (accepted && !displayWeights.empty() && impl_->pointWeights.find(idx) == impl_->pointWeights.end()) {
                        impl_->pointWeights[idx] = std::move(displayWeights);
                    }

                    double instanceThreshold =
                        ComputeAdaptiveResidualThreshold(fitResult.residuals, distThreshold);
                    auto compactMask = BuildInstanceCompactMask(fitResult, keptIndices.size(), instanceThreshold, 5);
                    if (compactMask.empty()) break;
                    std::vector<bool> removeMask(remainPoints.size(), false);
                    for (size_t i = 0; i < keptIndices.size() && i < compactMask.size(); ++i) {
                        if (compactMask[i] && keptIndices[i] < removeMask.size()) {
                            removeMask[keptIndices[i]] = true;
                        }
                    }
                    RemoveMaskedPoints(remainPoints, remainScores, remainMap, removeMask);
                }
                break;
            }

            case MetrologyObjectType::Rectangle2: {
                auto* rectObj = static_cast<MetrologyObjectRectangle2*>(&obj);

                RotatedRect2d initialRect;
                initialRect.center = {rectObj->Column(), rectObj->Row()};
                initialRect.width = rectObj->Length1() * 2.0;
                initialRect.height = rectObj->Length2() * 2.0;
                initialRect.angle = rectObj->Phi();

                auto fitRectangle = [&](const std::vector<Point2d>& points) -> Internal::RectangleFitResult {
                    Internal::RectangleFitResult fitResult;
                    if (points.size() < 8) {
                        return fitResult;
                    }

                    if (fitMethod == MetrologyFitMethod::RANSAC) {
                        auto sidedPoints = Internal::SegmentPointsByRectangleSide(points, initialRect);

                        std::array<Internal::LineFitResult, 4> sideResults;
                        std::array<Line2d, 4> fittedLines;
                        bool allSidesOK = true;
                        size_t totalInliers = 0;
                        double totalResidual = 0.0;
                        int validSides = 0;

                        Internal::RansacParams sideRansacParams;
                        sideRansacParams.threshold = distThreshold;
                        sideRansacParams.maxIterations = (maxIter < 0) ? 1000 : maxIter;

                        for (int side = 0; side < 4; ++side) {
                            if (sidedPoints[side].size() >= 2) {
                                sideResults[side] = Internal::FitLineRANSAC(
                                    sidedPoints[side], sideRansacParams, fitParams);

                                if (sideResults[side].success) {
                                    fittedLines[side] = sideResults[side].line;
                                    totalInliers += sideResults[side].numInliers;
                                    totalResidual += sideResults[side].residualRMS * sideResults[side].numInliers;
                                    validSides++;
                                } else {
                                    allSidesOK = false;
                                }
                            } else {
                                allSidesOK = false;
                            }
                        }

                        if (allSidesOK && validSides == 4) {
                            auto rectOpt = Internal::RectangleFromLines(fittedLines);
                            if (rectOpt.has_value()) {
                                fitResult.success = true;
                                fitResult.rect = rectOpt.value();
                                fitResult.numInliers = totalInliers;
                                fitResult.residualRMS = totalInliers > 0 ? totalResidual / totalInliers : 0.0;
                                fitResult.sideResults = sideResults;

                                std::vector<double> weights(points.size(), 0.0);
                                for (int side = 0; side < 4; ++side) {
                                    const auto& sidePts = sidedPoints[side];
                                    const auto& mask = sideResults[side].inlierMask;
                                    for (size_t j = 0; j < sidePts.size(); ++j) {
                                        const auto& pt = sidePts[j];
                                        for (size_t i = 0; i < points.size(); ++i) {
                                            if (std::abs(points[i].x - pt.x) < 1e-6 &&
                                                std::abs(points[i].y - pt.y) < 1e-6) {
                                                weights[i] = (j < mask.size() && mask[j]) ? 1.0 : 0.0;
                                                break;
                                            }
                                        }
                                    }
                                }
                                fitResult.weights = std::move(weights);
                            }
                        }
                    } else {
                        fitResult = Internal::FitRectangleIterative(points, initialRect, 10, 0.01, fitParams);
                        if (fitResult.success && distThreshold > 0) {
                            size_t inlierCount = 0;
                            for (const auto& pt : points) {
                                double dist = std::numeric_limits<double>::max();
                                for (int side = 0; side < 4; ++side) {
                                    if (fitResult.sideResults[side].success) {
                                        double d = fitResult.sideResults[side].line.Distance(pt);
                                        dist = std::min(dist, d);
                                    }
                                }
                                if (dist <= distThreshold) {
                                    inlierCount++;
                                }
                            }
                            fitResult.numInliers = inlierCount;
                        }
                    }

                    if (fitResult.success && fitResult.residuals.size() != points.size()) {
                        fitResult.residuals.assign(points.size(), 0.0);
                        for (size_t i = 0; i < points.size(); ++i) {
                            double dist = std::numeric_limits<double>::max();
                            for (int side = 0; side < 4; ++side) {
                                if (fitResult.sideResults[side].success) {
                                    double d = fitResult.sideResults[side].line.Distance(points[i]);
                                    dist = std::min(dist, d);
                                }
                            }
                            fitResult.residuals[i] = std::isfinite(dist) ? dist : 0.0;
                        }
                    }

                    return fitResult;
                };

                bool pushedAny = false;
                std::vector<Point2d> remainPoints = edgePoints;
                std::vector<double> remainScores = edgeScores;
                std::vector<size_t> remainMap(remainPoints.size());
                std::iota(remainMap.begin(), remainMap.end(), 0);

                for (int32_t inst = 0; inst < numInstances; ++inst) {
                    if (remainPoints.size() < 8) break;

                    std::vector<Point2d> fitPoints = remainPoints;
                    std::vector<double> fitScores = remainScores;
                    std::vector<size_t> keptIndices(fitPoints.size());
                    std::iota(keptIndices.begin(), keptIndices.end(), 0);
                    Internal::RectangleFitResult fitResult = fitRectangle(fitPoints);

                    if (fitResult.success && fitResult.residuals.size() == fitPoints.size()) {
                        double adaptiveThreshold =
                            ComputeAdaptiveResidualThreshold(fitResult.residuals, distThreshold);
                        if (FilterByResidualThreshold(fitResult.residuals, adaptiveThreshold, 8,
                                                      fitPoints, fitScores, keptIndices)) {
                            auto refit = fitRectangle(fitPoints);
                            if (refit.success) {
                                fitResult = std::move(refit);
                            }
                        }
                    }

                    if (fitResult.success && obj.Params().ignorePointCount > 0 && fitPoints.size() >= 8) {
                        IgnoreSelection sel = SelectPointsForIgnore(
                            fitPoints,
                            fitScores,
                            obj.Params().ignorePointCount,
                            obj.Params().ignorePointPolicy,
                            fitResult.residuals,
                            8
                        );
                        if (sel.applied) {
                            auto refit = fitRectangle(sel.points);
                            if (refit.success) {
                                fitResult = std::move(refit);
                                std::vector<size_t> remapped(sel.keptIndices.size(), 0);
                                for (size_t i = 0; i < sel.keptIndices.size(); ++i) {
                                    if (sel.keptIndices[i] < keptIndices.size()) {
                                        remapped[i] = keptIndices[sel.keptIndices[i]];
                                    }
                                }
                                keptIndices = std::move(remapped);
                                fitPoints = std::move(sel.points);
                                fitScores = std::move(sel.scores);
                            }
                        }
                    }

                    if (!fitResult.success) break;

                    MetrologyRectangle2Result result;
                    result.row = fitResult.rect.center.y;
                    result.column = fitResult.rect.center.x;
                    result.phi = fitResult.rect.angle;
                    result.length1 = fitResult.rect.width / 2.0;
                    result.length2 = fitResult.rect.height / 2.0;
                    result.numUsed = static_cast<int32_t>(fitResult.numInliers > 0 ? fitResult.numInliers : keptIndices.size());
                    result.rmsError = fitResult.residualRMS;
                    result.score = result.numUsed > 0 ? 1.0 / (1.0 + result.rmsError) : 0.0;
                    bool accepted = isAccepted(result.numUsed, result.rmsError, result.score);
                    if (accepted) {
                        impl_->rectangleResults[idx].push_back(result);
                        pushedAny = true;
                    }

                    std::vector<size_t> mappedKept;
                    mappedKept.reserve(keptIndices.size());
                    for (size_t k : keptIndices) {
                        if (k < remainMap.size()) mappedKept.push_back(remainMap[k]);
                    }
                    auto displayWeights = BuildWeightsForDisplay(fitResult, mappedKept, edgePoints.size());
                    if (accepted && !displayWeights.empty() && impl_->pointWeights.find(idx) == impl_->pointWeights.end()) {
                        impl_->pointWeights[idx] = std::move(displayWeights);
                    }

                    double instanceThreshold =
                        ComputeAdaptiveResidualThreshold(fitResult.residuals, distThreshold);
                    auto compactMask = BuildInstanceCompactMask(fitResult, keptIndices.size(), instanceThreshold, 8);
                    if (compactMask.empty()) break;
                    std::vector<bool> removeMask(remainPoints.size(), false);
                    for (size_t i = 0; i < keptIndices.size() && i < compactMask.size(); ++i) {
                        if (compactMask[i] && keptIndices[i] < removeMask.size()) {
                            removeMask[keptIndices[i]] = true;
                        }
                    }
                    RemoveMaskedPoints(remainPoints, remainScores, remainMap, removeMask);
                }

                if (!pushedAny) {
                    MetrologyRectangle2Result result;
                    if (edgePoints.size() >= 8) {
                        result.row = rectObj->Row();
                        result.column = rectObj->Column();
                        result.phi = rectObj->Phi();
                        result.length1 = rectObj->Length1();
                        result.length2 = rectObj->Length2();
                        result.numUsed = static_cast<int32_t>(edgePoints.size());
                        result.score = 0.5;
                        result.rmsError = 0.0;
                        if (isAccepted(result.numUsed, result.rmsError, result.score)) {
                            impl_->rectangleResults[idx].push_back(result);
                        }
                    } else if (edgePoints.size() >= 4) {
                        result.row = rectObj->Row();
                        result.column = rectObj->Column();
                        result.phi = rectObj->Phi();
                        result.length1 = rectObj->Length1();
                        result.length2 = rectObj->Length2();
                        result.numUsed = static_cast<int32_t>(edgePoints.size());
                        result.score = 0.3;
                        result.rmsError = 0.0;
                        if (isAccepted(result.numUsed, result.rmsError, result.score)) {
                            impl_->rectangleResults[idx].push_back(result);
                        }
                    }
                }
                break;
            }
        }
    }

    return true;
}

MetrologyLineResult MetrologyModel::GetLineResult(int32_t index, int32_t instanceIndex) const {
    if (instanceIndex < 0) {
        throw InvalidArgumentException("GetLineResult: instanceIndex must be >= 0");
    }
    (void)GetObject(index);
    auto it = impl_->lineResults.find(index);
    if (it != impl_->lineResults.end() && instanceIndex < static_cast<int32_t>(it->second.size())) {
        return it->second[instanceIndex];
    }
    return MetrologyLineResult();
}

MetrologyCircleResult MetrologyModel::GetCircleResult(int32_t index, int32_t instanceIndex) const {
    if (instanceIndex < 0) {
        throw InvalidArgumentException("GetCircleResult: instanceIndex must be >= 0");
    }
    (void)GetObject(index);
    auto it = impl_->circleResults.find(index);
    if (it != impl_->circleResults.end() && instanceIndex < static_cast<int32_t>(it->second.size())) {
        return it->second[instanceIndex];
    }
    return MetrologyCircleResult();
}

MetrologyEllipseResult MetrologyModel::GetEllipseResult(int32_t index, int32_t instanceIndex) const {
    if (instanceIndex < 0) {
        throw InvalidArgumentException("GetEllipseResult: instanceIndex must be >= 0");
    }
    (void)GetObject(index);
    auto it = impl_->ellipseResults.find(index);
    if (it != impl_->ellipseResults.end() && instanceIndex < static_cast<int32_t>(it->second.size())) {
        return it->second[instanceIndex];
    }
    return MetrologyEllipseResult();
}

MetrologyRectangle2Result MetrologyModel::GetRectangle2Result(int32_t index, int32_t instanceIndex) const {
    if (instanceIndex < 0) {
        throw InvalidArgumentException("GetRectangle2Result: instanceIndex must be >= 0");
    }
    (void)GetObject(index);
    auto it = impl_->rectangleResults.find(index);
    if (it != impl_->rectangleResults.end() && instanceIndex < static_cast<int32_t>(it->second.size())) {
        return it->second[instanceIndex];
    }
    return MetrologyRectangle2Result();
}

QContour MetrologyModel::GetResultContour(int32_t index, int32_t instanceIndex) const {
    auto* obj = GetObject(index);

    switch (obj->Type()) {
        case MetrologyObjectType::Line: {
            auto result = GetLineResult(index, instanceIndex);
            if (result.IsValid()) {
                QContour contour;
                contour.AddPoint(Point2d{result.col1, result.row1});
                contour.AddPoint(Point2d{result.col2, result.row2});
                return contour;
            }
            break;
        }

        case MetrologyObjectType::Circle: {
            auto result = GetCircleResult(index, instanceIndex);
            if (result.IsValid()) {
                QContour contour;
                int32_t numPoints = std::max(32, static_cast<int32_t>(result.radius * 0.5));
                double angleExtent = result.endAngle - result.startAngle;
                for (int32_t i = 0; i <= numPoints; ++i) {
                    double t = static_cast<double>(i) / numPoints;
                    double angle = result.startAngle + t * angleExtent;
                    double x = result.column + result.radius * std::cos(angle);
                    double y = result.row + result.radius * std::sin(angle);
                    contour.AddPoint(Point2d{x, y});
                }
                return contour;
            }
            break;
        }

        case MetrologyObjectType::Ellipse: {
            auto result = GetEllipseResult(index, instanceIndex);
            if (result.IsValid()) {
                QContour contour;
                int32_t numPoints = std::max(32, static_cast<int32_t>((result.ra + result.rb) * 0.5));
                double cosPhi = std::cos(result.phi);
                double sinPhi = std::sin(result.phi);
                for (int32_t i = 0; i <= numPoints; ++i) {
                    double t = static_cast<double>(i) / numPoints;
                    double angle = t * TWO_PI;
                    double localX = result.ra * std::cos(angle);
                    double localY = result.rb * std::sin(angle);
                    double x = result.column + localX * cosPhi - localY * sinPhi;
                    double y = result.row + localX * sinPhi + localY * cosPhi;
                    contour.AddPoint(Point2d{x, y});
                }
                return contour;
            }
            break;
        }

        case MetrologyObjectType::Rectangle2: {
            auto result = GetRectangle2Result(index, instanceIndex);
            if (result.IsValid()) {
                QContour contour;
                double cosPhi = std::cos(result.phi);
                double sinPhi = std::sin(result.phi);
                std::vector<std::pair<double, double>> corners = {
                    {-result.length1, -result.length2},
                    {result.length1, -result.length2},
                    {result.length1, result.length2},
                    {-result.length1, result.length2},
                    {-result.length1, -result.length2}
                };
                for (auto& c : corners) {
                    double x = result.column + c.first * cosPhi - c.second * sinPhi;
                    double y = result.row + c.first * sinPhi + c.second * cosPhi;
                    contour.AddPoint(Point2d{x, y});
                }
                return contour;
            }
            break;
        }
    }

    return QContour();
}

std::vector<Point2d> MetrologyModel::GetMeasuredPoints(int32_t index) const {
    (void)GetObject(index);
    auto it = impl_->measuredPoints.find(index);
    if (it != impl_->measuredPoints.end()) {
        return it->second;
    }
    return {};
}

std::vector<double> MetrologyModel::GetPointWeights(int32_t index) const {
    (void)GetObject(index);
    auto it = impl_->pointWeights.find(index);
    if (it != impl_->pointWeights.end()) {
        return it->second;
    }
    return {};
}

void MetrologyModel::Align(double row, double column, double phi) {
    if (!std::isfinite(row) || !std::isfinite(column) || !std::isfinite(phi)) {
        throw InvalidArgumentException("Align: invalid parameters");
    }
    double deltaRow = row - impl_->alignRow;
    double deltaCol = column - impl_->alignCol;
    double deltaPhi = phi - impl_->alignPhi;

    for (auto& obj : impl_->objects) {
        if (obj) {
            obj->Transform(deltaRow, deltaCol, deltaPhi);
        }
    }

    impl_->alignRow = row;
    impl_->alignCol = column;
    impl_->alignPhi = phi;
}

void MetrologyModel::ResetAlignment() {
    Align(0.0, 0.0, 0.0);
}

} // namespace Qi::Vision::Measure
