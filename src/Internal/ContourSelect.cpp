/**
 * @file ContourSelect.cpp
 * @brief Implementation of contour selection/filtering functions
 */

#include <QiVision/Internal/ContourSelect.h>
#include <QiVision/Internal/ContourAnalysis.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace Qi::Vision::Internal {

// =============================================================================
// Feature Name Conversion
// =============================================================================

std::string ContourFeatureToString(ContourFeature feature) {
    static const std::unordered_map<ContourFeature, std::string> names = {
        {ContourFeature::Length, "length"},
        {ContourFeature::Area, "area"},
        {ContourFeature::Perimeter, "perimeter"},
        {ContourFeature::NumPoints, "num_points"},
        {ContourFeature::MeanCurvature, "mean_curvature"},
        {ContourFeature::MaxCurvature, "max_curvature"},
        {ContourFeature::Circularity, "circularity"},
        {ContourFeature::Compactness, "compactness"},
        {ContourFeature::Convexity, "convexity"},
        {ContourFeature::Solidity, "solidity"},
        {ContourFeature::Eccentricity, "eccentricity"},
        {ContourFeature::Elongation, "elongation"},
        {ContourFeature::Rectangularity, "rectangularity"},
        {ContourFeature::Extent, "extent"},
        {ContourFeature::AspectRatio, "aspect_ratio"},
        {ContourFeature::CentroidRow, "centroid_row"},
        {ContourFeature::CentroidCol, "centroid_col"},
        {ContourFeature::BoundingBoxWidth, "bbox_width"},
        {ContourFeature::BoundingBoxHeight, "bbox_height"},
        {ContourFeature::Orientation, "orientation"}
    };

    auto it = names.find(feature);
    if (it != names.end()) {
        return it->second;
    }
    return "unknown";
}

ContourFeature StringToContourFeature(const std::string& name) {
    // Convert to lowercase for case-insensitive comparison
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    static const std::unordered_map<std::string, ContourFeature> features = {
        {"length", ContourFeature::Length},
        {"area", ContourFeature::Area},
        {"perimeter", ContourFeature::Perimeter},
        {"num_points", ContourFeature::NumPoints},
        {"numpoints", ContourFeature::NumPoints},
        {"mean_curvature", ContourFeature::MeanCurvature},
        {"meancurvature", ContourFeature::MeanCurvature},
        {"max_curvature", ContourFeature::MaxCurvature},
        {"maxcurvature", ContourFeature::MaxCurvature},
        {"circularity", ContourFeature::Circularity},
        {"compactness", ContourFeature::Compactness},
        {"convexity", ContourFeature::Convexity},
        {"solidity", ContourFeature::Solidity},
        {"eccentricity", ContourFeature::Eccentricity},
        {"elongation", ContourFeature::Elongation},
        {"rectangularity", ContourFeature::Rectangularity},
        {"extent", ContourFeature::Extent},
        {"aspect_ratio", ContourFeature::AspectRatio},
        {"aspectratio", ContourFeature::AspectRatio},
        {"centroid_row", ContourFeature::CentroidRow},
        {"centroidrow", ContourFeature::CentroidRow},
        {"row", ContourFeature::CentroidRow},
        {"centroid_col", ContourFeature::CentroidCol},
        {"centroidcol", ContourFeature::CentroidCol},
        {"col", ContourFeature::CentroidCol},
        {"column", ContourFeature::CentroidCol},
        {"bbox_width", ContourFeature::BoundingBoxWidth},
        {"bboxwidth", ContourFeature::BoundingBoxWidth},
        {"width", ContourFeature::BoundingBoxWidth},
        {"bbox_height", ContourFeature::BoundingBoxHeight},
        {"bboxheight", ContourFeature::BoundingBoxHeight},
        {"height", ContourFeature::BoundingBoxHeight},
        {"orientation", ContourFeature::Orientation},
        {"angle", ContourFeature::Orientation}
    };

    auto it = features.find(lower);
    if (it != features.end()) {
        return it->second;
    }
    return ContourFeature::Length; // Default
}

// =============================================================================
// Feature Computation
// =============================================================================

double ComputeContourFeature(const QContour& contour, ContourFeature feature) {
    if (contour.Empty()) {
        return 0.0;
    }

    switch (feature) {
        case ContourFeature::Length:
            return ContourLength(contour);

        case ContourFeature::Area:
            return ContourArea(contour);

        case ContourFeature::Perimeter:
            return ContourPerimeter(contour);

        case ContourFeature::NumPoints:
            return static_cast<double>(contour.Size());

        case ContourFeature::MeanCurvature:
            return ContourMeanCurvature(contour);

        case ContourFeature::MaxCurvature:
            return ContourMaxCurvature(contour);

        case ContourFeature::Circularity:
            return ContourCircularity(contour);

        case ContourFeature::Compactness:
            return ContourCompactness(contour);

        case ContourFeature::Convexity:
            return ContourConvexity(contour);

        case ContourFeature::Solidity:
            return ContourSolidity(contour);

        case ContourFeature::Eccentricity:
            return ContourEccentricity(contour);

        case ContourFeature::Elongation:
            return ContourElongation(contour);

        case ContourFeature::Rectangularity:
            return ContourRectangularity(contour);

        case ContourFeature::Extent:
            return ContourExtent(contour);

        case ContourFeature::AspectRatio: {
            auto axes = ContourPrincipalAxes(contour);
            if (!axes.valid || axes.minorLength < 1e-10) {
                return 1.0;
            }
            return axes.majorLength / axes.minorLength;
        }

        case ContourFeature::CentroidRow: {
            Point2d centroid = ContourCentroid(contour);
            return centroid.y;
        }

        case ContourFeature::CentroidCol: {
            Point2d centroid = ContourCentroid(contour);
            return centroid.x;
        }

        case ContourFeature::BoundingBoxWidth: {
            Rect2d bbox = ContourBoundingBox(contour);
            return bbox.width;
        }

        case ContourFeature::BoundingBoxHeight: {
            Rect2d bbox = ContourBoundingBox(contour);
            return bbox.height;
        }

        case ContourFeature::Orientation:
            return ContourOrientation(contour);

        default:
            return 0.0;
    }
}

std::vector<double> ComputeContourFeatures(const QContour& contour,
                                            const std::vector<ContourFeature>& features) {
    std::vector<double> values;
    values.reserve(features.size());

    if (contour.Empty()) {
        values.resize(features.size(), 0.0);
        return values;
    }

    // Cache commonly used values to avoid redundant computation
    double length = -1.0;
    double area = -1.0;
    Point2d centroid = {-1e9, -1e9};
    Rect2d bbox = {-1, -1, -1, -1};
    PrincipalAxesResult axes;
    axes.valid = false;
    bool centroidComputed = false;
    bool bboxComputed = false;

    for (ContourFeature feature : features) {
        double value = 0.0;

        switch (feature) {
            case ContourFeature::Length:
            case ContourFeature::Perimeter:
                if (length < 0) {
                    length = ContourLength(contour);
                }
                value = length;
                break;

            case ContourFeature::Area:
                if (area < 0) {
                    area = ContourArea(contour);
                }
                value = area;
                break;

            case ContourFeature::NumPoints:
                value = static_cast<double>(contour.Size());
                break;

            case ContourFeature::CentroidRow:
                if (!centroidComputed) {
                    centroid = ContourCentroid(contour);
                    centroidComputed = true;
                }
                value = centroid.y;
                break;

            case ContourFeature::CentroidCol:
                if (!centroidComputed) {
                    centroid = ContourCentroid(contour);
                    centroidComputed = true;
                }
                value = centroid.x;
                break;

            case ContourFeature::BoundingBoxWidth:
                if (!bboxComputed) {
                    bbox = ContourBoundingBox(contour);
                    bboxComputed = true;
                }
                value = bbox.width;
                break;

            case ContourFeature::BoundingBoxHeight:
                if (!bboxComputed) {
                    bbox = ContourBoundingBox(contour);
                    bboxComputed = true;
                }
                value = bbox.height;
                break;

            case ContourFeature::AspectRatio:
                if (!axes.valid) {
                    axes = ContourPrincipalAxes(contour);
                }
                if (axes.valid && axes.minorLength > 1e-10) {
                    value = axes.majorLength / axes.minorLength;
                } else {
                    value = 1.0;
                }
                break;

            case ContourFeature::Elongation:
                if (!axes.valid) {
                    axes = ContourPrincipalAxes(contour);
                }
                if (axes.valid && axes.majorLength > 1e-10) {
                    value = 1.0 - axes.minorLength / axes.majorLength;
                } else {
                    value = 0.0;
                }
                break;

            case ContourFeature::Orientation:
                if (!axes.valid) {
                    axes = ContourPrincipalAxes(contour);
                }
                value = axes.valid ? axes.angle : 0.0;
                break;

            default:
                // For other features, compute directly
                value = ComputeContourFeature(contour, feature);
                break;
        }

        values.push_back(value);
    }

    return values;
}

// =============================================================================
// Single-Feature Selection Functions
// =============================================================================

QContourArray SelectContoursByLength(const QContourArray& contours,
                                      double minLength,
                                      double maxLength) {
    return SelectContoursByFeature(contours, ContourFeature::Length, minLength, maxLength);
}

QContourArray SelectContoursByArea(const QContourArray& contours,
                                    double minArea,
                                    double maxArea) {
    return SelectContoursByFeature(contours, ContourFeature::Area, minArea, maxArea);
}

QContourArray SelectContoursByNumPoints(const QContourArray& contours,
                                         size_t minPoints,
                                         size_t maxPoints) {
    return SelectContoursByFeature(contours, ContourFeature::NumPoints,
                                   static_cast<double>(minPoints),
                                   static_cast<double>(maxPoints));
}

QContourArray SelectContoursByCircularity(const QContourArray& contours,
                                           double minCircularity,
                                           double maxCircularity) {
    return SelectContoursByFeature(contours, ContourFeature::Circularity,
                                   minCircularity, maxCircularity);
}

QContourArray SelectContoursByConvexity(const QContourArray& contours,
                                         double minConvexity,
                                         double maxConvexity) {
    return SelectContoursByFeature(contours, ContourFeature::Convexity,
                                   minConvexity, maxConvexity);
}

QContourArray SelectContoursBySolidity(const QContourArray& contours,
                                        double minSolidity,
                                        double maxSolidity) {
    return SelectContoursByFeature(contours, ContourFeature::Solidity,
                                   minSolidity, maxSolidity);
}

QContourArray SelectContoursByCompactness(const QContourArray& contours,
                                           double minCompactness,
                                           double maxCompactness) {
    return SelectContoursByFeature(contours, ContourFeature::Compactness,
                                   minCompactness, maxCompactness);
}

QContourArray SelectContoursByElongation(const QContourArray& contours,
                                          double minElongation,
                                          double maxElongation) {
    return SelectContoursByFeature(contours, ContourFeature::Elongation,
                                   minElongation, maxElongation);
}

QContourArray SelectContoursByAspectRatio(const QContourArray& contours,
                                           double minRatio,
                                           double maxRatio) {
    return SelectContoursByFeature(contours, ContourFeature::AspectRatio,
                                   minRatio, maxRatio);
}

QContourArray SelectContoursByMeanCurvature(const QContourArray& contours,
                                             double minCurvature,
                                             double maxCurvature) {
    return SelectContoursByFeature(contours, ContourFeature::MeanCurvature,
                                   minCurvature, maxCurvature);
}

QContourArray SelectContoursByMaxCurvature(const QContourArray& contours,
                                            double minCurvature,
                                            double maxCurvature) {
    return SelectContoursByFeature(contours, ContourFeature::MaxCurvature,
                                   minCurvature, maxCurvature);
}

// =============================================================================
// Generic Selection Functions
// =============================================================================

QContourArray SelectContoursByFeature(const QContourArray& contours,
                                       ContourFeature feature,
                                       double minValue,
                                       double maxValue) {
    QContourArray result;

    for (size_t i = 0; i < contours.Size(); ++i) {
        const QContour& contour = contours[i];
        double value = ComputeContourFeature(contour, feature);

        if (value >= minValue && value <= maxValue) {
            result.Add(contour);
        }
    }

    return result;
}

QContourArray SelectContoursByCriteria(const QContourArray& contours,
                                        const std::vector<SelectionCriterion>& criteria,
                                        SelectionLogic logic) {
    if (criteria.empty()) {
        return contours; // No criteria = select all
    }

    QContourArray result;

    for (size_t i = 0; i < contours.Size(); ++i) {
        const QContour& contour = contours[i];

        // Compute all needed features
        std::vector<ContourFeature> features;
        features.reserve(criteria.size());
        for (const auto& c : criteria) {
            features.push_back(c.feature);
        }
        std::vector<double> values = ComputeContourFeatures(contour, features);

        // Check criteria
        bool selected = false;

        if (logic == SelectionLogic::And) {
            selected = true;
            for (size_t j = 0; j < criteria.size(); ++j) {
                if (!criteria[j].Passes(values[j])) {
                    selected = false;
                    break;
                }
            }
        } else { // Or
            selected = false;
            for (size_t j = 0; j < criteria.size(); ++j) {
                if (criteria[j].Passes(values[j])) {
                    selected = true;
                    break;
                }
            }
        }

        if (selected) {
            result.Add(contour);
        }
    }

    return result;
}

QContourArray SelectContoursIf(const QContourArray& contours,
                                const std::function<bool(const QContour&)>& predicate) {
    QContourArray result;

    for (size_t i = 0; i < contours.Size(); ++i) {
        if (predicate(contours[i])) {
            result.Add(contours[i]);
        }
    }

    return result;
}

// =============================================================================
// Index-Based Selection
// =============================================================================

QContourArray SelectContoursByIndex(const QContourArray& contours,
                                     const std::vector<size_t>& indices) {
    QContourArray result;

    for (size_t idx : indices) {
        if (idx < contours.Size()) {
            result.Add(contours[idx]);
        }
    }

    return result;
}

QContourArray SelectContourRange(const QContourArray& contours,
                                  size_t startIndex,
                                  size_t endIndex) {
    QContourArray result;

    size_t start = std::min(startIndex, contours.Size());
    size_t end = std::min(endIndex, contours.Size());

    for (size_t i = start; i < end; ++i) {
        result.Add(contours[i]);
    }

    return result;
}

QContourArray SelectFirstContours(const QContourArray& contours, size_t count) {
    return SelectContourRange(contours, 0, count);
}

QContourArray SelectLastContours(const QContourArray& contours, size_t count) {
    if (count >= contours.Size()) {
        return contours;
    }
    return SelectContourRange(contours, contours.Size() - count, contours.Size());
}

// =============================================================================
// Sorting and Ranking
// =============================================================================

QContourArray SortContoursByFeature(const QContourArray& contours,
                                     ContourFeature feature,
                                     bool ascending) {
    if (contours.Size() <= 1) {
        return contours;
    }

    // Create index-value pairs
    std::vector<std::pair<size_t, double>> indexValues;
    indexValues.reserve(contours.Size());

    for (size_t i = 0; i < contours.Size(); ++i) {
        double value = ComputeContourFeature(contours[i], feature);
        indexValues.emplace_back(i, value);
    }

    // Sort by value
    if (ascending) {
        std::sort(indexValues.begin(), indexValues.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
    } else {
        std::sort(indexValues.begin(), indexValues.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
    }

    // Build sorted result
    QContourArray result;
    for (const auto& [idx, val] : indexValues) {
        result.Add(contours[idx]);
    }

    return result;
}

QContourArray SelectTopContoursByFeature(const QContourArray& contours,
                                          ContourFeature feature,
                                          size_t count,
                                          bool largest) {
    if (count >= contours.Size()) {
        return SortContoursByFeature(contours, feature, !largest);
    }

    // Create index-value pairs
    std::vector<std::pair<size_t, double>> indexValues;
    indexValues.reserve(contours.Size());

    for (size_t i = 0; i < contours.Size(); ++i) {
        double value = ComputeContourFeature(contours[i], feature);
        indexValues.emplace_back(i, value);
    }

    // Partial sort to get top N
    if (largest) {
        std::partial_sort(indexValues.begin(), indexValues.begin() + count,
                          indexValues.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
    } else {
        std::partial_sort(indexValues.begin(), indexValues.begin() + count,
                          indexValues.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });
    }

    // Build result
    QContourArray result;
    for (size_t i = 0; i < count; ++i) {
        result.Add(contours[indexValues[i].first]);
    }

    return result;
}

// =============================================================================
// Spatial Selection
// =============================================================================

QContourArray SelectContoursInRect(const QContourArray& contours,
                                    double minRow, double maxRow,
                                    double minCol, double maxCol) {
    QContourArray result;

    for (size_t i = 0; i < contours.Size(); ++i) {
        Point2d centroid = ContourCentroid(contours[i]);

        if (centroid.y >= minRow && centroid.y <= maxRow &&
            centroid.x >= minCol && centroid.x <= maxCol) {
            result.Add(contours[i]);
        }
    }

    return result;
}

QContourArray SelectContoursInCircle(const QContourArray& contours,
                                      double centerRow, double centerCol,
                                      double radius) {
    QContourArray result;
    double radiusSq = radius * radius;

    for (size_t i = 0; i < contours.Size(); ++i) {
        Point2d centroid = ContourCentroid(contours[i]);

        double dx = centroid.x - centerCol;
        double dy = centroid.y - centerRow;
        double distSq = dx * dx + dy * dy;

        if (distSq <= radiusSq) {
            result.Add(contours[i]);
        }
    }

    return result;
}

// =============================================================================
// Closed/Open Selection
// =============================================================================

QContourArray SelectClosedContours(const QContourArray& contours) {
    return SelectContoursIf(contours,
                            [](const QContour& c) { return c.IsClosed(); });
}

QContourArray SelectOpenContours(const QContourArray& contours) {
    return SelectContoursIf(contours,
                            [](const QContour& c) { return !c.IsClosed(); });
}

// =============================================================================
// Utility Functions
// =============================================================================

std::vector<size_t> GetContourIndicesByFeature(const QContourArray& contours,
                                                ContourFeature feature,
                                                double minValue,
                                                double maxValue) {
    std::vector<size_t> indices;

    for (size_t i = 0; i < contours.Size(); ++i) {
        double value = ComputeContourFeature(contours[i], feature);
        if (value >= minValue && value <= maxValue) {
            indices.push_back(i);
        }
    }

    return indices;
}

void PartitionContoursByFeature(const QContourArray& contours,
                                 ContourFeature feature,
                                 double threshold,
                                 QContourArray& below,
                                 QContourArray& aboveOrEqual) {
    below = QContourArray();
    aboveOrEqual = QContourArray();

    for (size_t i = 0; i < contours.Size(); ++i) {
        double value = ComputeContourFeature(contours[i], feature);
        if (value < threshold) {
            below.Add(contours[i]);
        } else {
            aboveOrEqual.Add(contours[i]);
        }
    }
}

} // namespace Qi::Vision::Internal
