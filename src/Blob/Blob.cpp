/**
 * @file Blob.cpp
 * @brief Blob analysis implementation
 */

#include <QiVision/Blob/Blob.h>
#include <QiVision/Internal/ConnectedComponent.h>
#include <QiVision/Internal/RegionFeatures.h>
#include <QiVision/Internal/RLEOps.h>
#include <QiVision/Core/Exception.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace Qi::Vision::Blob {

// =============================================================================
// Connection
// =============================================================================

std::vector<QRegion> Connection(const QRegion& region) {
    if (region.Empty()) return {};

    // Get runs and find connected components
    const auto& runs = region.Runs();
    if (runs.empty()) return {};

    // Build adjacency using RLE
    // Two runs are connected if they are in adjacent rows and overlap in columns
    std::vector<int32_t> labels(runs.size(), -1);
    std::vector<int32_t> parent(runs.size());
    for (size_t i = 0; i < runs.size(); ++i) {
        parent[i] = static_cast<int32_t>(i);
    }

    // Union-Find helpers
    auto find = [&parent](int32_t x) -> int32_t {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    auto unite = [&parent, &find](int32_t x, int32_t y) {
        int32_t px = find(x);
        int32_t py = find(y);
        if (px != py) {
            parent[px] = py;
        }
    };

    // Build index for fast lookup by row
    std::unordered_map<int32_t, std::vector<size_t>> rowIndex;
    for (size_t i = 0; i < runs.size(); ++i) {
        rowIndex[runs[i].row].push_back(i);
    }

    // Connect adjacent runs (8-connectivity)
    for (size_t i = 0; i < runs.size(); ++i) {
        const auto& run = runs[i];

        // Check previous row
        auto it = rowIndex.find(run.row - 1);
        if (it != rowIndex.end()) {
            for (size_t j : it->second) {
                const auto& other = runs[j];
                // Check overlap (8-connectivity: overlap or adjacent by 1)
                if (other.colEnd >= run.colBegin - 1 && other.colBegin <= run.colEnd + 1) {
                    unite(static_cast<int32_t>(i), static_cast<int32_t>(j));
                }
            }
        }
    }

    // Collect components
    std::unordered_map<int32_t, std::vector<QRegion::Run>> componentRuns;
    for (size_t i = 0; i < runs.size(); ++i) {
        int32_t root = find(static_cast<int32_t>(i));
        componentRuns[root].push_back(runs[i]);
    }

    // Create result regions
    std::vector<QRegion> result;
    result.reserve(componentRuns.size());
    for (auto& [label, compRuns] : componentRuns) {
        result.emplace_back(std::move(compRuns));
    }

    return result;
}

std::vector<QRegion> Connection(const QImage& binaryImage, Connectivity connectivity) {
    if (binaryImage.Empty()) return {};

    int32_t numLabels = 0;
    QImage labels = Internal::LabelConnectedComponents(binaryImage, connectivity, numLabels);

    if (numLabels == 0) return {};

    std::vector<QRegion> result;
    result.reserve(numLabels);

    for (int32_t label = 1; label <= numLabels; ++label) {
        QImage component = Internal::ExtractComponent(labels, label);
        QRegion region = Internal::NonZeroToRegion(component);
        if (!region.Empty()) {
            result.push_back(std::move(region));
        }
    }

    return result;
}

QRegion SelectObj(const std::vector<QRegion>& regions, int32_t index) {
    if (index < 1 || index > static_cast<int32_t>(regions.size())) {
        return QRegion();
    }
    return regions[index - 1];
}

// =============================================================================
// Region Features
// =============================================================================

void AreaCenter(const QRegion& region, int64_t& area, double& row, double& column) {
    auto features = Internal::ComputeBasicFeatures(region);
    area = features.area;
    row = features.centroidY;
    column = features.centroidX;
}

void AreaCenter(const std::vector<QRegion>& regions,
                std::vector<int64_t>& areas,
                std::vector<double>& rows,
                std::vector<double>& columns) {
    areas.resize(regions.size());
    rows.resize(regions.size());
    columns.resize(regions.size());

    for (size_t i = 0; i < regions.size(); ++i) {
        AreaCenter(regions[i], areas[i], rows[i], columns[i]);
    }
}

void SmallestRectangle1(const QRegion& region,
                         int32_t& row1, int32_t& column1,
                         int32_t& row2, int32_t& column2) {
    Rect2i bbox = Internal::ComputeBoundingBox(region);
    row1 = bbox.y;
    column1 = bbox.x;
    row2 = bbox.y + bbox.height - 1;
    column2 = bbox.x + bbox.width - 1;
}

void SmallestRectangle2(const QRegion& region,
                         double& row, double& column, double& phi,
                         double& length1, double& length2) {
    auto rect = Internal::ComputeMinAreaRect(region);
    row = rect.center.y;
    column = rect.center.x;
    phi = rect.angle;
    length1 = rect.width / 2.0;
    length2 = rect.height / 2.0;
}

void SmallestCircle(const QRegion& region,
                     double& row, double& column, double& radius) {
    auto circle = Internal::ComputeMinEnclosingCircle(region);
    row = circle.center.y;
    column = circle.center.x;
    radius = circle.radius;
}

double Circularity(const QRegion& region) {
    auto features = Internal::ComputeShapeFeatures(region);
    return features.circularity;
}

double Compactness(const QRegion& region) {
    auto features = Internal::ComputeShapeFeatures(region);
    return features.compactness;
}

double Convexity(const QRegion& region) {
    return Internal::ComputeConvexity(region);
}

double Rectangularity(const QRegion& region) {
    auto features = Internal::ComputeShapeFeatures(region);
    return features.rectangularity;
}

void EllipticAxis(const QRegion& region, double& ra, double& rb, double& phi) {
    auto ellipse = Internal::ComputeEllipseFeatures(region);
    ra = ellipse.majorAxis;
    rb = ellipse.minorAxis;
    phi = ellipse.angle;
}

double OrientationRegion(const QRegion& region) {
    return Internal::ComputeOrientation(region);
}

void MomentsRegion2nd(const QRegion& region,
                       double& m11, double& m20, double& m02,
                       double& ia, double& ib) {
    auto moments = Internal::ComputeMoments(region);
    m11 = moments.mu11;
    m20 = moments.mu20;
    m02 = moments.mu02;

    // Compute principal moments of inertia
    double trace = m20 + m02;
    double det = m20 * m02 - m11 * m11;
    double disc = std::sqrt(std::max(0.0, trace * trace / 4.0 - det));
    ia = trace / 2.0 + disc;
    ib = trace / 2.0 - disc;
}

void Eccentricity(const QRegion& region,
                   double& anisometry, double& bulkiness, double& structureFactor) {
    auto ellipse = Internal::ComputeEllipseFeatures(region);
    auto basic = Internal::ComputeBasicFeatures(region);

    double ra = ellipse.majorAxis;
    double rb = ellipse.minorAxis;

    anisometry = (rb > 0) ? ra / rb : 0.0;
    bulkiness = (basic.area > 0) ? (M_PI * ra * rb / basic.area) : 0.0;
    structureFactor = anisometry * bulkiness - 1.0;
}

// =============================================================================
// Region Selection
// =============================================================================

double GetRegionFeature(const QRegion& region, ShapeFeature feature) {
    switch (feature) {
        case ShapeFeature::Area: {
            return static_cast<double>(Internal::ComputeArea(region));
        }
        case ShapeFeature::Row:
        case ShapeFeature::Column: {
            auto centroid = Internal::ComputeRegionCentroid(region);
            return (feature == ShapeFeature::Row) ? centroid.y : centroid.x;
        }
        case ShapeFeature::Width:
        case ShapeFeature::Height: {
            auto bbox = Internal::ComputeBoundingBox(region);
            return (feature == ShapeFeature::Width) ? bbox.width : bbox.height;
        }
        case ShapeFeature::Circularity:
            return Circularity(region);
        case ShapeFeature::Compactness:
            return Compactness(region);
        case ShapeFeature::Convexity:
            return Convexity(region);
        case ShapeFeature::Rectangularity:
            return Rectangularity(region);
        case ShapeFeature::Elongation: {
            auto ellipse = Internal::ComputeEllipseFeatures(region);
            return (ellipse.minorAxis > 0) ? ellipse.majorAxis / ellipse.minorAxis : 1.0;
        }
        case ShapeFeature::Orientation:
            return OrientationRegion(region);
        case ShapeFeature::Ra:
        case ShapeFeature::Rb:
        case ShapeFeature::Phi: {
            auto ellipse = Internal::ComputeEllipseFeatures(region);
            if (feature == ShapeFeature::Ra) return ellipse.majorAxis;
            if (feature == ShapeFeature::Rb) return ellipse.minorAxis;
            return ellipse.angle;
        }
        case ShapeFeature::Anisometry:
        case ShapeFeature::Bulkiness:
        case ShapeFeature::StructureFactor: {
            double ani, bulk, sf;
            Eccentricity(region, ani, bulk, sf);
            if (feature == ShapeFeature::Anisometry) return ani;
            if (feature == ShapeFeature::Bulkiness) return bulk;
            return sf;
        }
        case ShapeFeature::OuterRadius: {
            auto circle = Internal::ComputeMinEnclosingCircle(region);
            return circle.radius;
        }
        case ShapeFeature::InnerRadius:
            return 0.0; // TODO: implement inscribed circle
        case ShapeFeature::Holes:
            return 0.0; // TODO: implement hole counting
        default:
            return 0.0;
    }
}

std::vector<double> GetRegionFeatures(const std::vector<QRegion>& regions,
                                       ShapeFeature feature) {
    std::vector<double> result;
    result.reserve(regions.size());
    for (const auto& region : regions) {
        result.push_back(GetRegionFeature(region, feature));
    }
    return result;
}

std::vector<QRegion> SelectShape(const std::vector<QRegion>& regions,
                                  ShapeFeature feature,
                                  SelectOperation /*operation*/,
                                  double minValue,
                                  double maxValue) {
    std::vector<QRegion> result;
    result.reserve(regions.size());

    for (const auto& region : regions) {
        double value = GetRegionFeature(region, feature);
        if (value >= minValue && value <= maxValue) {
            result.push_back(region);
        }
    }

    return result;
}

ShapeFeature ParseShapeFeature(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "area") return ShapeFeature::Area;
    if (lower == "row") return ShapeFeature::Row;
    if (lower == "column" || lower == "col") return ShapeFeature::Column;
    if (lower == "width") return ShapeFeature::Width;
    if (lower == "height") return ShapeFeature::Height;
    if (lower == "circularity") return ShapeFeature::Circularity;
    if (lower == "compactness") return ShapeFeature::Compactness;
    if (lower == "convexity") return ShapeFeature::Convexity;
    if (lower == "rectangularity") return ShapeFeature::Rectangularity;
    if (lower == "elongation") return ShapeFeature::Elongation;
    if (lower == "orientation" || lower == "phi") return ShapeFeature::Orientation;
    if (lower == "ra") return ShapeFeature::Ra;
    if (lower == "rb") return ShapeFeature::Rb;
    if (lower == "anisometry") return ShapeFeature::Anisometry;
    if (lower == "bulkiness") return ShapeFeature::Bulkiness;
    if (lower == "structure_factor") return ShapeFeature::StructureFactor;
    if (lower == "outer_radius") return ShapeFeature::OuterRadius;
    if (lower == "inner_radius") return ShapeFeature::InnerRadius;
    if (lower == "holes") return ShapeFeature::Holes;

    return ShapeFeature::Area; // default
}

std::string GetShapeFeatureName(ShapeFeature feature) {
    switch (feature) {
        case ShapeFeature::Area: return "area";
        case ShapeFeature::Row: return "row";
        case ShapeFeature::Column: return "column";
        case ShapeFeature::Width: return "width";
        case ShapeFeature::Height: return "height";
        case ShapeFeature::Circularity: return "circularity";
        case ShapeFeature::Compactness: return "compactness";
        case ShapeFeature::Convexity: return "convexity";
        case ShapeFeature::Rectangularity: return "rectangularity";
        case ShapeFeature::Elongation: return "elongation";
        case ShapeFeature::Orientation: return "orientation";
        case ShapeFeature::Ra: return "ra";
        case ShapeFeature::Rb: return "rb";
        case ShapeFeature::Phi: return "phi";
        case ShapeFeature::Anisometry: return "anisometry";
        case ShapeFeature::Bulkiness: return "bulkiness";
        case ShapeFeature::StructureFactor: return "structure_factor";
        case ShapeFeature::OuterRadius: return "outer_radius";
        case ShapeFeature::InnerRadius: return "inner_radius";
        case ShapeFeature::Holes: return "holes";
        default: return "unknown";
    }
}

std::vector<QRegion> SelectShape(const std::vector<QRegion>& regions,
                                  const std::string& features,
                                  const std::string& operation,
                                  double minValue,
                                  double maxValue) {
    ShapeFeature feature = ParseShapeFeature(features);
    SelectOperation op = SelectOperation::And;
    std::string lowerOp = operation;
    std::transform(lowerOp.begin(), lowerOp.end(), lowerOp.begin(), ::tolower);
    if (lowerOp == "or") {
        op = SelectOperation::Or;
    }
    return SelectShape(regions, feature, op, minValue, maxValue);
}

std::vector<QRegion> SelectShapeArea(const std::vector<QRegion>& regions,
                                      int64_t minArea,
                                      int64_t maxArea) {
    return SelectShape(regions, ShapeFeature::Area, SelectOperation::And,
                       static_cast<double>(minArea), static_cast<double>(maxArea));
}

std::vector<QRegion> SelectShapeCircularity(const std::vector<QRegion>& regions,
                                             double minCirc,
                                             double maxCirc) {
    return SelectShape(regions, ShapeFeature::Circularity, SelectOperation::And, minCirc, maxCirc);
}

std::vector<QRegion> SelectShapeRectangularity(const std::vector<QRegion>& regions,
                                                double minRect,
                                                double maxRect) {
    return SelectShape(regions, ShapeFeature::Rectangularity, SelectOperation::And, minRect, maxRect);
}

// =============================================================================
// Region Sorting
// =============================================================================

std::vector<QRegion> SortRegion(const std::vector<QRegion>& regions,
                                 SortMode mode,
                                 bool ascending) {
    if (regions.empty() || mode == SortMode::None) {
        return regions;
    }

    std::vector<std::pair<double, size_t>> sortKeys;
    sortKeys.reserve(regions.size());

    for (size_t i = 0; i < regions.size(); ++i) {
        double key = 0.0;
        switch (mode) {
            case SortMode::Area:
                key = static_cast<double>(Internal::ComputeArea(regions[i]));
                break;
            case SortMode::Row:
            case SortMode::Column: {
                auto centroid = Internal::ComputeRegionCentroid(regions[i]);
                key = (mode == SortMode::Row) ? centroid.y : centroid.x;
                break;
            }
            case SortMode::FirstPoint: {
                const auto& runs = regions[i].Runs();
                if (!runs.empty()) {
                    key = runs.front().row * 100000 + runs.front().colBegin;
                }
                break;
            }
            case SortMode::LastPoint: {
                const auto& runs = regions[i].Runs();
                if (!runs.empty()) {
                    key = runs.back().row * 100000 + runs.back().colEnd;
                }
                break;
            }
            default:
                break;
        }
        sortKeys.emplace_back(key, i);
    }

    if (ascending) {
        std::sort(sortKeys.begin(), sortKeys.end());
    } else {
        std::sort(sortKeys.begin(), sortKeys.end(), std::greater<>());
    }

    std::vector<QRegion> result;
    result.reserve(regions.size());
    for (const auto& [key, idx] : sortKeys) {
        result.push_back(regions[idx]);
    }

    return result;
}

std::vector<QRegion> SortRegion(const std::vector<QRegion>& regions,
                                 const std::string& sortMode,
                                 const std::string& order,
                                 const std::string& /*rowOrCol*/) {
    std::string lowerMode = sortMode;
    std::transform(lowerMode.begin(), lowerMode.end(), lowerMode.begin(), ::tolower);

    SortMode mode = SortMode::None;
    if (lowerMode == "area") mode = SortMode::Area;
    else if (lowerMode == "first_point") mode = SortMode::FirstPoint;
    else if (lowerMode == "last_point") mode = SortMode::LastPoint;
    else if (lowerMode == "character" || lowerMode == "row") mode = SortMode::Row;
    else if (lowerMode == "column") mode = SortMode::Column;

    std::string lowerOrder = order;
    std::transform(lowerOrder.begin(), lowerOrder.end(), lowerOrder.begin(), ::tolower);
    bool ascending = (lowerOrder != "false" && lowerOrder != "descending");

    return SortRegion(regions, mode, ascending);
}

} // namespace Qi::Vision::Blob
