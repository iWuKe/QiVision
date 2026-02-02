/**
 * @file Contour.cpp
 * @brief XLD Contour operations implementation
 *
 * Wraps Internal layer contour operations into Halcon-style public API.
 */

#include <QiVision/Contour/Contour.h>

#include <QiVision/Internal/ContourProcess.h>
#include <QiVision/Internal/ContourAnalysis.h>
#include <QiVision/Internal/ContourSelect.h>
#include <QiVision/Internal/ContourSegment.h>
#include <QiVision/Internal/ContourConvert.h>
#include <QiVision/Internal/Fitting.h>
#include <QiVision/Internal/Distance.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Core/Exception.h>

#include <cmath>
#include <algorithm>
#include <string>

namespace Qi::Vision::Contour {

namespace {

std::string ToLower(const std::string& value) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower;
}

bool IsEmpty(const QContour& contour) {
    return contour.Size() == 0;
}

bool IsEmpty(const QContourArray& contours) {
    return contours.Size() == 0;
}

} // anonymous namespace

// =============================================================================
// Contour Processing
// =============================================================================

void SmoothContoursXld(
    const QContourArray& contours,
    QContourArray& smoothed,
    int32_t numPoints)
{
    smoothed.Clear();
    if (IsEmpty(contours)) {
        return;
    }
    if (numPoints <= 0) {
        throw InvalidArgumentException("SmoothContoursXld: numPoints must be > 0");
    }
    smoothed.Reserve(contours.Size());

    Internal::MovingAverageSmoothParams params;
    params.windowSize = numPoints;

    for (size_t i = 0; i < contours.Size(); ++i) {
        QContour result = Internal::SmoothContourMovingAverage(contours[i], params);
        smoothed.Add(std::move(result));
    }
}

void SmoothContoursGaussXld(
    const QContourArray& contours,
    QContourArray& smoothed,
    double sigma)
{
    smoothed.Clear();
    if (IsEmpty(contours)) {
        return;
    }
    if (sigma <= 0.0) {
        throw InvalidArgumentException("SmoothContoursGaussXld: sigma must be > 0");
    }
    smoothed.Reserve(contours.Size());

    Internal::GaussianSmoothParams params;
    params.sigma = sigma;

    for (size_t i = 0; i < contours.Size(); ++i) {
        QContour result = Internal::SmoothContourGaussian(contours[i], params);
        smoothed.Add(std::move(result));
    }
}

void SimplifyContoursXld(
    const QContourArray& contours,
    QContourArray& simplified,
    double epsilon)
{
    simplified.Clear();
    if (IsEmpty(contours)) {
        return;
    }
    if (epsilon <= 0.0) {
        throw InvalidArgumentException("SimplifyContoursXld: epsilon must be > 0");
    }
    simplified.Reserve(contours.Size());

    Internal::DouglasPeuckerParams params;
    params.tolerance = epsilon;

    for (size_t i = 0; i < contours.Size(); ++i) {
        QContour result = Internal::SimplifyContourDouglasPeucker(contours[i], params);
        simplified.Add(std::move(result));
    }
}

void ResampleContoursXld(
    const QContourArray& contours,
    QContourArray& resampled,
    double distance)
{
    resampled.Clear();
    if (IsEmpty(contours)) {
        return;
    }
    if (distance <= 0.0) {
        throw InvalidArgumentException("ResampleContoursXld: distance must be > 0");
    }
    resampled.Reserve(contours.Size());

    Internal::ResampleByDistanceParams params;
    params.distance = distance;

    for (size_t i = 0; i < contours.Size(); ++i) {
        QContour result = Internal::ResampleContourByDistance(contours[i], params);
        resampled.Add(std::move(result));
    }
}

void ResampleContoursNumXld(
    const QContourArray& contours,
    QContourArray& resampled,
    int32_t numPoints)
{
    resampled.Clear();
    if (IsEmpty(contours)) {
        return;
    }
    if (numPoints <= 0) {
        throw InvalidArgumentException("ResampleContoursNumXld: numPoints must be > 0");
    }
    resampled.Reserve(contours.Size());

    Internal::ResampleByCountParams params;
    params.count = static_cast<size_t>(numPoints);

    for (size_t i = 0; i < contours.Size(); ++i) {
        QContour result = Internal::ResampleContourByCount(contours[i], params);
        resampled.Add(std::move(result));
    }
}

void CloseContoursXld(
    const QContourArray& contours,
    QContourArray& closed)
{
    closed.Clear();
    if (IsEmpty(contours)) {
        return;
    }
    closed.Reserve(contours.Size());

    for (size_t i = 0; i < contours.Size(); ++i) {
        QContour result = Internal::CloseContour(contours[i]);
        closed.Add(std::move(result));
    }
}

void ReverseContoursXld(
    const QContourArray& contours,
    QContourArray& reversed)
{
    reversed.Clear();
    if (IsEmpty(contours)) {
        return;
    }
    reversed.Reserve(contours.Size());

    for (size_t i = 0; i < contours.Size(); ++i) {
        QContour result = Internal::ReverseContour(contours[i]);
        reversed.Add(std::move(result));
    }
}

// =============================================================================
// Contour Analysis - Basic Properties
// =============================================================================

double LengthXld(const QContour& contour)
{
    if (IsEmpty(contour)) {
        return 0.0;
    }
    return Internal::ContourLength(contour);
}

std::vector<double> LengthXld(const QContourArray& contours)
{
    std::vector<double> lengths;
    if (IsEmpty(contours)) {
        return lengths;
    }
    lengths.reserve(contours.Size());

    for (size_t i = 0; i < contours.Size(); ++i) {
        lengths.push_back(Internal::ContourLength(contours[i]));
    }

    return lengths;
}

double AreaCenterXld(const QContour& contour, double& row, double& column)
{
    if (IsEmpty(contour)) {
        row = 0.0;
        column = 0.0;
        return 0.0;
    }
    auto result = Internal::ContourAreaCenter(contour);
    row = result.centroid.y;
    column = result.centroid.x;
    return result.area;
}

double PerimeterXld(const QContour& contour)
{
    if (IsEmpty(contour)) {
        return 0.0;
    }
    return Internal::ContourPerimeter(contour);
}

// =============================================================================
// Contour Analysis - Bounding Geometry
// =============================================================================

void SmallestRectangle1Xld(
    const QContour& contour,
    double& row1, double& col1,
    double& row2, double& col2)
{
    if (IsEmpty(contour)) {
        row1 = col1 = row2 = col2 = 0.0;
        return;
    }
    Rect2d bbox = Internal::ContourBoundingBox(contour);
    row1 = bbox.y;
    col1 = bbox.x;
    row2 = bbox.y + bbox.height;
    col2 = bbox.x + bbox.width;
}

void SmallestRectangle2Xld(
    const QContour& contour,
    double& row, double& column,
    double& phi,
    double& length1, double& length2)
{
    if (IsEmpty(contour)) {
        row = column = phi = length1 = length2 = 0.0;
        return;
    }
    auto result = Internal::ContourMinAreaRect(contour);
    if (result.has_value()) {
        row = result->center.y;
        column = result->center.x;
        phi = result->angle;
        length1 = result->width / 2.0;
        length2 = result->height / 2.0;
    } else {
        // Fallback to AABB
        Rect2d bbox = Internal::ContourBoundingBox(contour);
        row = bbox.y + bbox.height / 2.0;
        column = bbox.x + bbox.width / 2.0;
        phi = 0.0;
        length1 = bbox.width / 2.0;
        length2 = bbox.height / 2.0;
    }
}

void SmallestCircleXld(
    const QContour& contour,
    double& row, double& column,
    double& radius)
{
    if (IsEmpty(contour)) {
        row = column = radius = 0.0;
        return;
    }
    auto result = Internal::ContourMinEnclosingCircle(contour);
    if (result.has_value()) {
        row = result->center.y;
        column = result->center.x;
        radius = result->radius;
    } else {
        // Fallback to centroid
        Point2d center = Internal::ContourCentroid(contour);
        row = center.y;
        column = center.x;
        radius = 0.0;
    }
}

// =============================================================================
// Contour Analysis - Curvature
// =============================================================================

std::vector<double> CurvatureXld(
    const QContour& contour,
    int32_t windowSize)
{
    if (IsEmpty(contour)) {
        return {};
    }
    if (windowSize <= 0) {
        throw InvalidArgumentException("CurvatureXld: windowSize must be > 0");
    }
    return Internal::ComputeContourCurvature(
        contour,
        Internal::CurvatureMethod::ThreePoint,
        windowSize
    );
}

double MeanCurvatureXld(const QContour& contour)
{
    if (IsEmpty(contour)) {
        return 0.0;
    }
    return Internal::ContourMeanCurvature(contour);
}

double MaxCurvatureXld(const QContour& contour)
{
    if (IsEmpty(contour)) {
        return 0.0;
    }
    return Internal::ContourMaxCurvature(contour);
}

// =============================================================================
// Contour Analysis - Moments
// =============================================================================

void MomentsXld(
    const QContour& contour,
    double& m00, double& m10, double& m01,
    double& m20, double& m11, double& m02)
{
    if (IsEmpty(contour)) {
        m00 = m10 = m01 = m20 = m11 = m02 = 0.0;
        return;
    }
    auto moments = Internal::ContourMoments(contour);
    m00 = moments.m00;
    m10 = moments.m10;
    m01 = moments.m01;
    m20 = moments.m20;
    m11 = moments.m11;
    m02 = moments.m02;
}

void CentralMomentsXld(
    const QContour& contour,
    double& mu00, double& mu20,
    double& mu11, double& mu02)
{
    if (IsEmpty(contour)) {
        mu00 = mu20 = mu11 = mu02 = 0.0;
        return;
    }
    auto moments = Internal::ContourCentralMoments(contour);
    mu00 = moments.mu00;
    mu20 = moments.mu20;
    mu11 = moments.mu11;
    mu02 = moments.mu02;
}

double OrientationXld(const QContour& contour)
{
    if (IsEmpty(contour)) {
        return 0.0;
    }
    return Internal::ContourOrientation(contour);
}

// =============================================================================
// Contour Analysis - Shape Descriptors
// =============================================================================

double CircularityXld(const QContour& contour)
{
    if (IsEmpty(contour)) {
        return 0.0;
    }
    return Internal::ContourCircularity(contour);
}

double ConvexityXld(const QContour& contour)
{
    if (IsEmpty(contour)) {
        return 0.0;
    }
    return Internal::ContourConvexity(contour);
}

double SolidityXld(const QContour& contour)
{
    if (IsEmpty(contour)) {
        return 0.0;
    }
    return Internal::ContourSolidity(contour);
}

double EccentricityXld(const QContour& contour)
{
    if (IsEmpty(contour)) {
        return 0.0;
    }
    return Internal::ContourEccentricity(contour);
}

double CompactnessXld(const QContour& contour)
{
    if (IsEmpty(contour)) {
        return 0.0;
    }
    return Internal::ContourCompactness(contour);
}

// =============================================================================
// Contour Fitting
// =============================================================================

bool FitEllipseContourXld(
    const QContour& contour,
    double& row, double& column,
    double& phi, double& ra, double& rb)
{
    if (contour.Size() < 5) {
        return false;
    }

    // Extract points
    std::vector<Point2d> points;
    points.reserve(contour.Size());
    for (size_t i = 0; i < contour.Size(); ++i) {
        const auto& pt = contour.GetPoint(i);
        points.push_back({pt.x, pt.y});
    }

    // Fit ellipse
    auto result = Internal::FitEllipseFitzgibbon(points);
    if (!result.success) {
        return false;
    }

    row = result.ellipse.center.y;
    column = result.ellipse.center.x;
    phi = result.ellipse.angle;
    ra = result.ellipse.a;
    rb = result.ellipse.b;

    return true;
}

bool FitLineContourXld(
    const QContour& contour,
    double& rowBegin, double& colBegin,
    double& rowEnd, double& colEnd,
    double& row, double& column,
    double& phi)
{
    if (contour.Size() < 2) {
        return false;
    }

    // Extract points
    std::vector<Point2d> points;
    points.reserve(contour.Size());
    for (size_t i = 0; i < contour.Size(); ++i) {
        const auto& pt = contour.GetPoint(i);
        points.push_back({pt.x, pt.y});
    }

    // Fit line
    auto result = Internal::FitLine(points);
    if (!result.success) {
        return false;
    }

    // Get line endpoints from contour extent
    Point2d first = contour.GetPoint(0);
    Point2d last = contour.GetPoint(contour.Size() - 1);

    // Get line angle and direction
    double lineAngle = result.line.Angle();
    Point2d dir = result.line.Direction();

    // Compute a reference point on the line (closest to origin)
    // For ax + by + c = 0, the closest point to origin is (-a*c, -b*c)
    Point2d refPoint(-result.line.a * result.line.c, -result.line.b * result.line.c);

    // Project endpoints onto the fitted line
    auto projectPoint = [&](const Point2d& pt) -> Point2d {
        double t = (pt.x - refPoint.x) * dir.x + (pt.y - refPoint.y) * dir.y;
        return {refPoint.x + t * dir.x, refPoint.y + t * dir.y};
    };

    Point2d projFirst = projectPoint(first);
    Point2d projLast = projectPoint(last);

    // Compute center point
    Point2d center = {(projFirst.x + projLast.x) / 2.0, (projFirst.y + projLast.y) / 2.0};

    rowBegin = projFirst.y;
    colBegin = projFirst.x;
    rowEnd = projLast.y;
    colEnd = projLast.x;
    row = center.y;
    column = center.x;
    phi = lineAngle;

    return true;
}

bool FitCircleContourXld(
    const QContour& contour,
    double& row, double& column,
    double& radius,
    double& startPhi, double& endPhi,
    const std::string& algorithm)
{
    if (contour.Size() < 3) {
        return false;
    }

    // Extract points
    std::vector<Point2d> points;
    points.reserve(contour.Size());
    for (size_t i = 0; i < contour.Size(); ++i) {
        const auto& pt = contour.GetPoint(i);
        points.push_back({pt.x, pt.y});
    }

    // Fit circle
    std::string lowerAlg = ToLower(algorithm);
    if (lowerAlg.empty()) {
        lowerAlg = "algebraic";
    }
    Internal::CircleFitResult result;
    if (lowerAlg == "geometric") {
        result = Internal::FitCircleGeometric(points);
    } else if (lowerAlg == "algebraic") {
        result = Internal::FitCircleAlgebraic(points);
    } else {
        throw InvalidArgumentException("FitCircleContourXld: unknown algorithm: " + algorithm);
    }

    if (!result.success) {
        return false;
    }

    row = result.circle.center.y;
    column = result.circle.center.x;
    radius = result.circle.radius;

    // Compute arc angles from first and last points
    Point2d first = contour.GetPoint(0);
    Point2d last = contour.GetPoint(contour.Size() - 1);

    startPhi = std::atan2(first.y - result.circle.center.y,
                          first.x - result.circle.center.x);
    endPhi = std::atan2(last.y - result.circle.center.y,
                        last.x - result.circle.center.x);

    return true;
}

void ConvexHullXld(const QContour& contour, QContour& hull)
{
    hull = Internal::ContourConvexHull(contour);
}

// =============================================================================
// Contour Selection
// =============================================================================

void SelectContoursXld(
    const QContourArray& contours,
    QContourArray& selected,
    const std::string& feature,
    double minValue,
    double maxValue)
{
    if (IsEmpty(contours)) {
        selected.Clear();
        return;
    }
    // Map feature string to enum
    Internal::ContourFeature featureEnum = Internal::StringToContourFeature(feature);

    selected = Internal::SelectContoursByFeature(contours, featureEnum, minValue, maxValue);
}

void SelectClosedXld(
    const QContourArray& contours,
    QContourArray& closed)
{
    if (IsEmpty(contours)) {
        closed.Clear();
        return;
    }
    closed = Internal::SelectClosedContours(contours);
}

void SelectOpenXld(
    const QContourArray& contours,
    QContourArray& open)
{
    if (IsEmpty(contours)) {
        open.Clear();
        return;
    }
    open = Internal::SelectOpenContours(contours);
}

void SortContoursXld(
    const QContourArray& contours,
    QContourArray& sorted,
    const std::string& feature,
    bool ascending)
{
    if (IsEmpty(contours)) {
        sorted.Clear();
        return;
    }
    Internal::ContourFeature featureEnum = Internal::StringToContourFeature(feature);
    sorted = Internal::SortContoursByFeature(contours, featureEnum, ascending);
}

void SelectTopContoursXld(
    const QContourArray& contours,
    QContourArray& selected,
    const std::string& feature,
    int32_t count,
    bool largest)
{
    if (IsEmpty(contours)) {
        selected.Clear();
        return;
    }
    if (count <= 0) {
        throw InvalidArgumentException("SelectTopContoursXld: count must be > 0");
    }
    Internal::ContourFeature featureEnum = Internal::StringToContourFeature(feature);
    selected = Internal::SelectTopContoursByFeature(contours, featureEnum, static_cast<size_t>(count), largest);
}

// =============================================================================
// Contour Segmentation
// =============================================================================

void SegmentContoursXld(
    const QContour& contour,
    QContourArray& segments,
    const std::string& mode,
    double maxLineDev,
    double maxArcDev)
{
    if (IsEmpty(contour)) {
        segments.Clear();
        return;
    }
    if (maxLineDev <= 0.0 || maxArcDev <= 0.0) {
        throw InvalidArgumentException("SegmentContoursXld: maxLineDev/maxArcDev must be > 0");
    }
    Internal::SegmentParams params;

    std::string lowerMode = ToLower(mode);
    if (lowerMode.empty() || lowerMode == "lines_and_arcs" ||
        lowerMode == "lines+arcs" || lowerMode == "all") {
        params.mode = Internal::SegmentMode::LinesAndArcs;
    } else if (lowerMode == "lines") {
        params.mode = Internal::SegmentMode::LinesOnly;
    } else if (lowerMode == "circles" || lowerMode == "arcs") {
        params.mode = Internal::SegmentMode::ArcsOnly;
    } else {
        throw InvalidArgumentException("SegmentContoursXld: unknown mode: " + mode);
    }

    params.maxLineError = maxLineDev;
    params.maxArcError = maxArcDev;

    auto result = Internal::SegmentContour(contour, params);
    segments = Internal::PrimitivesToContours(result, 1.0);
}

void SplitContoursXld(
    const QContourArray& contours,
    QContourArray& split,
    double maxAngle)
{
    split.Clear();
    if (IsEmpty(contours)) {
        return;
    }
    if (maxAngle <= 0.0) {
        throw InvalidArgumentException("SplitContoursXld: maxAngle must be > 0");
    }

    for (size_t i = 0; i < contours.Size(); ++i) {
        // Detect corner points
        double curvatureThreshold = 1.0 / (10.0 * maxAngle);  // Approximate conversion
        auto corners = Internal::DetectCorners(contours[i], curvatureThreshold);

        if (corners.empty()) {
            split.Add(contours[i]);
        } else {
            auto subContours = Internal::SplitContourAtIndices(contours[i], corners);
            for (size_t j = 0; j < subContours.Size(); ++j) {
                split.Add(subContours[j]);
            }
        }
    }
}

std::vector<int32_t> DetectCornersXld(
    const QContour& contour,
    double curvatureThreshold)
{
    if (IsEmpty(contour)) {
        return {};
    }
    if (curvatureThreshold <= 0.0) {
        throw InvalidArgumentException("DetectCornersXld: curvatureThreshold must be > 0");
    }
    auto corners = Internal::DetectCorners(contour, curvatureThreshold);

    std::vector<int32_t> result;
    result.reserve(corners.size());
    for (size_t idx : corners) {
        result.push_back(static_cast<int32_t>(idx));
    }
    return result;
}

// =============================================================================
// Contour-Region Conversion
// =============================================================================

void GenContourRegionXld(
    const QRegion& region,
    QContourArray& contours,
    const std::string& mode)
{
    if (region.Empty()) {
        contours.Clear();
        return;
    }
    std::string lowerMode = ToLower(mode);
    Internal::BoundaryMode bmode = Internal::BoundaryMode::Outer;
    if (lowerMode.empty() || lowerMode == "border" || lowerMode == "outer") {
        bmode = Internal::BoundaryMode::Outer;
    } else if (lowerMode == "border_holes" || lowerMode == "both") {
        bmode = Internal::BoundaryMode::Both;
    } else {
        throw InvalidArgumentException("GenContourRegionXld: unknown mode: " + mode);
    }

    contours = Internal::RegionToContours(region, bmode);
}

void GenRegionContourXld(
    const QContour& contour,
    QRegion& region,
    const std::string& mode)
{
    if (IsEmpty(contour)) {
        region = QRegion();
        return;
    }
    std::string lowerMode = ToLower(mode);
    Internal::ContourFillMode fmode = Internal::ContourFillMode::Filled;
    if (lowerMode.empty() || lowerMode == "filled") {
        fmode = Internal::ContourFillMode::Filled;
    } else if (lowerMode == "margin") {
        fmode = Internal::ContourFillMode::Margin;
    } else {
        throw InvalidArgumentException("GenRegionContourXld: unknown mode: " + mode);
    }

    region = Internal::ContourToRegion(contour, fmode);
}

void GenRegionContoursXld(
    const QContourArray& contours,
    QRegion& region,
    const std::string& mode)
{
    if (IsEmpty(contours)) {
        region = QRegion();
        return;
    }
    std::string lowerMode = ToLower(mode);
    Internal::ContourFillMode fmode = Internal::ContourFillMode::Filled;
    if (lowerMode.empty() || lowerMode == "filled") {
        fmode = Internal::ContourFillMode::Filled;
    } else if (lowerMode == "margin") {
        fmode = Internal::ContourFillMode::Margin;
    } else {
        throw InvalidArgumentException("GenRegionContoursXld: unknown mode: " + mode);
    }

    region = Internal::ContoursToRegion(contours, fmode);
}

// =============================================================================
// Contour Generation
// =============================================================================

QContour GenContourPolygonXld(const std::vector<Point2d>& points)
{
    QContour contour;
    if (points.empty()) {
        return contour;
    }
    for (const auto& pt : points) {
        if (!pt.IsValid()) {
            throw InvalidArgumentException("GenContourPolygonXld: invalid point");
        }
    }
    contour.Reserve(points.size());

    for (const auto& pt : points) {
        contour.AddPoint(pt.x, pt.y);
    }

    return contour;
}

QContour GenContourPolygonXld(
    const std::vector<double>& rows,
    const std::vector<double>& cols)
{
    if (rows.size() != cols.size()) {
        throw InvalidArgumentException("GenContourPolygonXld: rows and columns must have same size");
    }

    QContour contour;
    if (rows.empty()) {
        return contour;
    }
    contour.Reserve(rows.size());

    for (size_t i = 0; i < rows.size(); ++i) {
        if (!std::isfinite(rows[i]) || !std::isfinite(cols[i])) {
            throw InvalidArgumentException("GenContourPolygonXld: invalid point");
        }
        contour.AddPoint(cols[i], rows[i]);
    }

    return contour;
}

QContour GenCircleContourXld(
    double row, double column,
    double radius,
    double startAngle,
    double endAngle,
    const std::string& resolution,
    double stepAngle)
{
    if (!std::isfinite(row) || !std::isfinite(column) ||
        !std::isfinite(radius) || !std::isfinite(startAngle) ||
        !std::isfinite(endAngle) || !std::isfinite(stepAngle)) {
        throw InvalidArgumentException("GenCircleContourXld: invalid parameters");
    }
    if (radius <= 0.0 || stepAngle <= 0.0) {
        throw InvalidArgumentException("GenCircleContourXld: radius and stepAngle must be > 0");
    }
    QContour contour;

    std::string lowerRes = ToLower(resolution);
    if (lowerRes.empty()) {
        lowerRes = "positive";
    }
    bool ccw = false;
    if (lowerRes == "positive") {
        ccw = true;
    } else if (lowerRes == "negative") {
        ccw = false;
    } else {
        throw InvalidArgumentException("GenCircleContourXld: unknown resolution: " + resolution);
    }

    // Normalize angles
    double angleExtent = endAngle - startAngle;
    if (std::abs(angleExtent) > 2.0 * PI) {
        angleExtent = 2.0 * PI;
    }

    int32_t numPoints = static_cast<int32_t>(std::abs(angleExtent) / stepAngle) + 1;
    numPoints = std::max(numPoints, 3);

    contour.Reserve(numPoints);

    double step = angleExtent / (numPoints - 1);
    if (!ccw) {
        step = -step;
    }

    for (int32_t i = 0; i < numPoints; ++i) {
        double angle = startAngle + i * step;
        double x = column + radius * std::cos(angle);
        double y = row + radius * std::sin(angle);
        contour.AddPoint(x, y);
    }

    // Close if full circle
    if (std::abs(angleExtent - 2.0 * PI) < 1e-6) {
        contour.SetClosed(true);
    }

    return contour;
}

QContour GenEllipseContourXld(
    double row, double column,
    double phi,
    double ra, double rb,
    double startAngle,
    double endAngle,
    double stepAngle)
{
    if (!std::isfinite(row) || !std::isfinite(column) || !std::isfinite(phi) ||
        !std::isfinite(ra) || !std::isfinite(rb) ||
        !std::isfinite(startAngle) || !std::isfinite(endAngle) || !std::isfinite(stepAngle)) {
        throw InvalidArgumentException("GenEllipseContourXld: invalid parameters");
    }
    if (ra <= 0.0 || rb <= 0.0 || stepAngle <= 0.0) {
        throw InvalidArgumentException("GenEllipseContourXld: ra/rb/stepAngle must be > 0");
    }
    QContour contour;

    // Normalize angles
    double angleExtent = endAngle - startAngle;
    if (std::abs(angleExtent) > 2.0 * PI) {
        angleExtent = 2.0 * PI;
    }

    int32_t numPoints = static_cast<int32_t>(std::abs(angleExtent) / stepAngle) + 1;
    numPoints = std::max(numPoints, 3);

    contour.Reserve(numPoints);

    double cosPhi = std::cos(phi);
    double sinPhi = std::sin(phi);
    double step = angleExtent / (numPoints - 1);

    for (int32_t i = 0; i < numPoints; ++i) {
        double t = startAngle + i * step;

        // Ellipse in local coordinates
        double localX = ra * std::cos(t);
        double localY = rb * std::sin(t);

        // Rotate and translate
        double x = column + localX * cosPhi - localY * sinPhi;
        double y = row + localX * sinPhi + localY * cosPhi;

        contour.AddPoint(x, y);
    }

    // Close if full ellipse
    if (std::abs(angleExtent - 2.0 * PI) < 1e-6) {
        contour.SetClosed(true);
    }

    return contour;
}

QContour GenRectangle2ContourXld(
    double row, double column,
    double phi,
    double length1, double length2)
{
    if (!std::isfinite(row) || !std::isfinite(column) || !std::isfinite(phi) ||
        !std::isfinite(length1) || !std::isfinite(length2)) {
        throw InvalidArgumentException("GenRectangle2ContourXld: invalid parameters");
    }
    if (length1 <= 0.0 || length2 <= 0.0) {
        throw InvalidArgumentException("GenRectangle2ContourXld: length1/length2 must be > 0");
    }
    QContour contour;
    contour.Reserve(4);

    double cosPhi = std::cos(phi);
    double sinPhi = std::sin(phi);

    // Four corners in local coordinates
    double dx1 = length1 * cosPhi;
    double dy1 = length1 * sinPhi;
    double dx2 = length2 * (-sinPhi);
    double dy2 = length2 * cosPhi;

    // Add corners (CCW order)
    contour.AddPoint(column - dx1 - dx2, row - dy1 - dy2);
    contour.AddPoint(column + dx1 - dx2, row + dy1 - dy2);
    contour.AddPoint(column + dx1 + dx2, row + dy1 + dy2);
    contour.AddPoint(column - dx1 + dx2, row - dy1 + dy2);

    contour.SetClosed(true);

    return contour;
}

QContour GenRectangle1ContourXld(
    double row1, double col1,
    double row2, double col2)
{
    if (!std::isfinite(row1) || !std::isfinite(col1) ||
        !std::isfinite(row2) || !std::isfinite(col2)) {
        throw InvalidArgumentException("GenRectangle1ContourXld: invalid parameters");
    }
    if (row2 <= row1 || col2 <= col1) {
        throw InvalidArgumentException("GenRectangle1ContourXld: invalid rectangle bounds");
    }
    QContour contour;
    contour.Reserve(4);

    contour.AddPoint(col1, row1);
    contour.AddPoint(col2, row1);
    contour.AddPoint(col2, row2);
    contour.AddPoint(col1, row2);

    contour.SetClosed(true);

    return contour;
}

QContour GenLineContourXld(
    double row1, double col1,
    double row2, double col2)
{
    if (!std::isfinite(row1) || !std::isfinite(col1) ||
        !std::isfinite(row2) || !std::isfinite(col2)) {
        throw InvalidArgumentException("GenLineContourXld: invalid parameters");
    }
    if (row1 == row2 && col1 == col2) {
        throw InvalidArgumentException("GenLineContourXld: line endpoints must differ");
    }
    QContour contour;
    contour.Reserve(2);

    contour.AddPoint(col1, row1);
    contour.AddPoint(col2, row2);

    contour.SetClosed(false);

    return contour;
}

// =============================================================================
// Utility Functions
// =============================================================================

int32_t CountPointsXld(const QContour& contour)
{
    if (IsEmpty(contour)) {
        return 0;
    }
    return static_cast<int32_t>(contour.Size());
}

std::vector<int32_t> CountPointsXld(const QContourArray& contours)
{
    std::vector<int32_t> counts;
    if (IsEmpty(contours)) {
        return counts;
    }
    counts.reserve(contours.Size());

    for (size_t i = 0; i < contours.Size(); ++i) {
        counts.push_back(static_cast<int32_t>(contours[i].Size()));
    }

    return counts;
}

void GetContourXld(
    const QContour& contour,
    std::vector<double>& rows,
    std::vector<double>& cols)
{
    if (IsEmpty(contour)) {
        rows.clear();
        cols.clear();
        return;
    }
    size_t n = contour.Size();
    rows.resize(n);
    cols.resize(n);

    for (size_t i = 0; i < n; ++i) {
        const auto& pt = contour.GetPoint(i);
        rows[i] = pt.y;
        cols[i] = pt.x;
    }
}

bool TestPointXld(
    const QContour& contour,
    double row, double column)
{
    if (IsEmpty(contour)) {
        return false;
    }
    return Internal::IsPointInsideContour(contour, {column, row});
}

double DistancePointXld(
    const QContour& contour,
    double row, double column)
{
    if (IsEmpty(contour)) {
        return 0.0;
    }
    // Extract contour points
    std::vector<Point2d> points = contour.GetPoints();
    auto result = Internal::DistancePointToContour({column, row}, points, contour.IsClosed());
    return result.distance;
}

void UnionContoursXld(
    const QContourArray& contours1,
    const QContourArray& contours2,
    QContourArray& result)
{
    result.Clear();
    if (IsEmpty(contours1) && IsEmpty(contours2)) {
        return;
    }
    result.Reserve(contours1.Size() + contours2.Size());

    for (size_t i = 0; i < contours1.Size(); ++i) {
        result.Add(contours1[i]);
    }
    for (size_t i = 0; i < contours2.Size(); ++i) {
        result.Add(contours2[i]);
    }
}

void SelectObjXld(
    const QContourArray& contours,
    QContourArray& selected,
    const std::vector<int32_t>& indices)
{
    selected.Clear();
    selected.Reserve(indices.size());

    for (int32_t idx : indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= contours.Size()) {
            throw InvalidArgumentException("SelectObjXld: index out of range");
        }
        selected.Add(contours[static_cast<size_t>(idx)]);
    }
}

QContour SelectObjXld(
    const QContourArray& contours,
    int32_t index)
{
    if (IsEmpty(contours)) {
        return QContour();
    }
    if (index < 0 || static_cast<size_t>(index) >= contours.Size()) {
        throw InvalidArgumentException("SelectObjXld: index out of range");
    }
    return contours[static_cast<size_t>(index)];
}

int32_t CountObjXld(const QContourArray& contours)
{
    return static_cast<int32_t>(contours.Size());
}

} // namespace Qi::Vision::Contour
