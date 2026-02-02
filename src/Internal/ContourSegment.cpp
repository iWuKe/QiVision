#include <QiVision/Internal/ContourSegment.h>
#include <QiVision/Internal/ContourAnalysis.h>
#include <QiVision/Internal/ContourProcess.h>
#include <QiVision/Internal/Fitting.h>
#include <QiVision/Internal/Geometry2d.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace Qi::Vision::Internal {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

const double PI = 3.14159265358979323846;

// Helper to normalize angle to [-PI, PI]
inline double NormalizeAngleLocal(double angle) {
    while (angle > PI) angle -= 2 * PI;
    while (angle < -PI) angle += 2 * PI;
    return angle;
}

// Extract points from contour between indices
std::vector<Point2d> ExtractPoints(const QContour& contour, size_t startIdx, size_t endIdx) {
    std::vector<Point2d> points;
    size_t n = contour.Size();

    if (n == 0 || startIdx >= n) return points;

    endIdx = std::min(endIdx, n - 1);
    points.reserve(endIdx - startIdx + 1);

    for (size_t i = startIdx; i <= endIdx; ++i) {
        points.push_back(contour.GetPoint(i));
    }

    return points;
}

// Compute RMS error between points and line
[[maybe_unused]] double ComputeLineError(const std::vector<Point2d>& points, const Line2d& line) {
    if (points.empty()) return 0.0;

    double sumSqError = 0.0;
    for (const auto& p : points) {
        double d = line.Distance(p);
        sumSqError += d * d;
    }

    return std::sqrt(sumSqError / points.size());
}

// Compute max error between points and line
[[maybe_unused]] double ComputeLineMaxError(const std::vector<Point2d>& points, const Line2d& line) {
    double maxError = 0.0;
    for (const auto& p : points) {
        double d = line.Distance(p);
        maxError = std::max(maxError, d);
    }
    return maxError;
}

// Compute RMS error between points and circle
double ComputeCircleError(const std::vector<Point2d>& points, const Circle2d& circle) {
    if (points.empty()) return 0.0;

    double sumSqError = 0.0;
    for (const auto& p : points) {
        double d = std::abs(p.DistanceTo(circle.center) - circle.radius);
        sumSqError += d * d;
    }

    return std::sqrt(sumSqError / points.size());
}

// Compute max error between points and circle
double ComputeCircleMaxError(const std::vector<Point2d>& points, const Circle2d& circle) {
    double maxError = 0.0;
    for (const auto& p : points) {
        double d = std::abs(p.DistanceTo(circle.center) - circle.radius);
        maxError = std::max(maxError, d);
    }
    return maxError;
}

// Compute angle of point relative to circle center
double ComputeAngle(const Point2d& point, const Point2d& center) {
    return std::atan2(point.y - center.y, point.x - center.x);
}


// Find index of point with maximum distance to line segment
size_t FindMaxDistancePoint(const std::vector<Point2d>& points,
                            size_t startIdx, size_t endIdx) {
    if (endIdx <= startIdx + 1) return startIdx;

    Line2d line = Line2d::FromPoints(points[startIdx], points[endIdx]);

    double maxDist = 0.0;
    size_t maxIdx = startIdx;

    for (size_t i = startIdx + 1; i < endIdx; ++i) {
        double d = line.Distance(points[i]);
        if (d > maxDist) {
            maxDist = d;
            maxIdx = i;
        }
    }

    return maxIdx;
}

// Check if two angles are within threshold (handling wraparound)
[[maybe_unused]] bool AnglesClose(double a1, double a2, double threshold) {
    double diff = std::abs(NormalizeAngleLocal(a1 - a2));
    return diff < threshold || diff > (2 * PI - threshold);
}

} // namespace

// =============================================================================
// Corner Detection
// =============================================================================

std::vector<size_t> DetectCorners(const QContour& contour,
                                   double curvatureThreshold,
                                   int windowSize) {
    std::vector<size_t> corners;

    if (contour.Size() < 3) return corners;

    // Compute curvature at each point
    std::vector<double> curvature = ComputeContourCurvature(contour,
                                                             CurvatureMethod::ThreePoint,
                                                             windowSize);

    // Find local maxima of absolute curvature above threshold
    for (size_t i = 1; i + 1 < curvature.size(); ++i) {
        double absCurv = std::abs(curvature[i]);

        if (absCurv >= curvatureThreshold) {
            // Check if local maximum
            if (absCurv >= std::abs(curvature[i - 1]) &&
                absCurv >= std::abs(curvature[i + 1])) {
                corners.push_back(i);
            }
        }
    }

    // Handle closed contour - check first and last points
    if (contour.IsClosed() && curvature.size() >= 3) {
        double firstCurv = std::abs(curvature[0]);
        if (firstCurv >= curvatureThreshold &&
            firstCurv >= std::abs(curvature[curvature.size() - 1]) &&
            firstCurv >= std::abs(curvature[1])) {
            corners.insert(corners.begin(), 0);
        }
    }

    return corners;
}

std::vector<size_t> DetectDominantPoints(const QContour& contour,
                                          double curvatureThreshold,
                                          int windowSize) {
    std::vector<size_t> dominant;

    if (contour.Size() < 3) return dominant;

    std::vector<double> curvature = ComputeContourCurvature(contour,
                                                             CurvatureMethod::ThreePoint,
                                                             windowSize);

    // Find corners (local maxima above threshold)
    for (size_t i = 1; i + 1 < curvature.size(); ++i) {
        double absCurv = std::abs(curvature[i]);

        // Check for corner
        if (absCurv >= curvatureThreshold &&
            absCurv >= std::abs(curvature[i - 1]) &&
            absCurv >= std::abs(curvature[i + 1])) {
            dominant.push_back(i);
        }
        // Check for inflection point (sign change)
        else if (curvature[i - 1] * curvature[i + 1] < 0) {
            // Curvature changes sign
            dominant.push_back(i);
        }
    }

    return dominant;
}

// =============================================================================
// Split Point Finding
// =============================================================================

std::vector<size_t> FindLineSplitPoints(const QContour& contour,
                                         double maxError) {
    std::vector<size_t> splits;

    if (contour.Size() < 3) return splits;

    std::vector<Point2d> points = contour.GetPoints();

    // Douglas-Peucker style recursive splitting
    std::vector<std::pair<size_t, size_t>> stack;
    stack.push_back({0, points.size() - 1});

    std::vector<bool> isSplit(points.size(), false);
    isSplit[0] = true;
    isSplit[points.size() - 1] = true;

    while (!stack.empty()) {
        auto [start, end] = stack.back();
        stack.pop_back();

        if (end <= start + 1) continue;

        size_t maxIdx = FindMaxDistancePoint(points, start, end);
        Line2d line = Line2d::FromPoints(points[start], points[end]);
        double dist = line.Distance(points[maxIdx]);

        if (dist > maxError) {
            isSplit[maxIdx] = true;
            stack.push_back({start, maxIdx});
            stack.push_back({maxIdx, end});
        }
    }

    for (size_t i = 0; i < points.size(); ++i) {
        if (isSplit[i]) splits.push_back(i);
    }

    return splits;
}

std::vector<size_t> FindArcSplitPoints(const QContour& contour,
                                        double maxError) {
    std::vector<size_t> splits;

    if (contour.Size() < 5) return splits;

    // Use curvature change to find splits
    std::vector<double> curvature = ComputeContourCurvature(contour,
                                                             CurvatureMethod::ThreePoint, 5);

    splits.push_back(0);

    // Find points where curvature changes significantly
    double prevCurv = curvature[0];
    for (size_t i = 1; i + 1 < curvature.size(); ++i) {
        double currCurv = curvature[i];

        // Check for significant curvature change
        double diff = std::abs(currCurv - prevCurv);
        if (diff > maxError * 0.1) {  // Scaled threshold
            splits.push_back(i);
            prevCurv = currCurv;
        }
    }

    splits.push_back(contour.Size() - 1);

    return splits;
}

// =============================================================================
// Primitive Fitting
// =============================================================================

PrimitiveFitResult FitLineToContour(const QContour& contour,
                                     size_t startIdx,
                                     size_t endIdx) {
    PrimitiveFitResult result;
    result.type = PrimitiveType::Unknown;
    result.startIndex = startIdx;
    result.endIndex = endIdx;

    std::vector<Point2d> points = ExtractPoints(contour, startIdx, endIdx);

    if (points.size() < MIN_POINTS_FOR_LINE) {
        return result;
    }

    result.numPoints = points.size();

    // Fit line
    LineFitResult fitResult = FitLine(points);

    if (!fitResult.success) {
        return result;
    }

    result.type = PrimitiveType::Line;

    // Create segment from fitted line
    // Project first and last points onto line
    Point2d dir = fitResult.line.Direction();
    Point2d p0 = points.front();
    Point2d pn = points.back();

    // Use actual endpoints projected onto line
    Point2d start = p0 - dir * fitResult.line.SignedDistance(p0);
    Point2d end = pn - dir * fitResult.line.SignedDistance(pn);

    result.segment = Segment2d(start, end);
    result.error = fitResult.residualRMS;
    result.maxError = fitResult.residualMax;

    return result;
}

PrimitiveFitResult FitArcToContour(const QContour& contour,
                                    size_t startIdx,
                                    size_t endIdx) {
    PrimitiveFitResult result;
    result.type = PrimitiveType::Unknown;
    result.startIndex = startIdx;
    result.endIndex = endIdx;

    std::vector<Point2d> points = ExtractPoints(contour, startIdx, endIdx);

    if (points.size() < MIN_POINTS_FOR_ARC) {
        return result;
    }

    result.numPoints = points.size();

    // Fit circle
    CircleFitResult fitResult = FitCircle(points, CircleFitMethod::Algebraic);

    if (!fitResult.success || fitResult.circle.radius <= 0) {
        return result;
    }

    // Compute arc angles
    double startAngle = ComputeAngle(points.front(), fitResult.circle.center);
    double endAngle = ComputeAngle(points.back(), fitResult.circle.center);

    // Determine sweep direction by checking middle points
    double midAngle = ComputeAngle(points[points.size() / 2], fitResult.circle.center);

    double sweep1 = NormalizeAngleLocal(endAngle - startAngle);
    double sweep2 = (sweep1 > 0) ? (sweep1 - 2 * PI) : (sweep1 + 2 * PI);

    // Choose sweep that includes the middle point
    double midDiff1 = NormalizeAngleLocal(midAngle - startAngle);
    double sweepAngle = (std::abs(midDiff1) < std::abs(sweep1)) ? sweep1 : sweep2;

    // Check sweep angle is reasonable
    if (std::abs(sweepAngle) < MIN_ARC_SWEEP_ANGLE) {
        return result;
    }

    result.type = PrimitiveType::Arc;
    result.arc = Arc2d(fitResult.circle.center, fitResult.circle.radius, startAngle, sweepAngle);
    result.error = ComputeCircleError(points, fitResult.circle);
    result.maxError = ComputeCircleMaxError(points, fitResult.circle);

    return result;
}

PrimitiveFitResult FitBestPrimitive(const QContour& contour,
                                     size_t startIdx,
                                     size_t endIdx,
                                     bool preferLine) {
    PrimitiveFitResult lineResult = FitLineToContour(contour, startIdx, endIdx);
    PrimitiveFitResult arcResult = FitArcToContour(contour, startIdx, endIdx);

    // If only one is valid, return it
    if (!lineResult.IsValid() && !arcResult.IsValid()) {
        return PrimitiveFitResult();
    }
    if (!lineResult.IsValid()) return arcResult;
    if (!arcResult.IsValid()) return lineResult;

    // Both valid - choose based on error
    double lineBias = preferLine ? 0.9 : 1.1;  // Slight preference adjustment

    if (lineResult.error * lineBias <= arcResult.error) {
        return lineResult;
    } else {
        return arcResult;
    }
}

// =============================================================================
// Primitive Classification
// =============================================================================

PrimitiveType ClassifyContourSegment(const QContour& contour,
                                      size_t startIdx,
                                      size_t endIdx) {
    double linearity = ComputeLinearity(contour, startIdx, endIdx);
    double circularity = ComputeCircularity(contour, startIdx, endIdx);

    // Threshold for classification
    const double threshold = 0.8;

    if (linearity >= threshold && linearity >= circularity) {
        return PrimitiveType::Line;
    } else if (circularity >= threshold && circularity > linearity) {
        return PrimitiveType::Arc;
    }

    return PrimitiveType::Unknown;
}

double ComputeLinearity(const QContour& contour,
                         size_t startIdx,
                         size_t endIdx) {
    std::vector<Point2d> points = ExtractPoints(contour, startIdx, endIdx);

    if (points.size() < 2) return 1.0;

    // Compute chord length
    double chordLength = points.front().DistanceTo(points.back());
    if (chordLength < 1e-10) return 0.0;

    // Compute actual path length
    double pathLength = 0.0;
    for (size_t i = 1; i < points.size(); ++i) {
        pathLength += points[i - 1].DistanceTo(points[i]);
    }

    if (pathLength < 1e-10) return 0.0;

    // Linearity = chord/path (1.0 for straight line)
    return std::min(1.0, chordLength / pathLength);
}

double ComputeCircularity(const QContour& contour,
                           size_t startIdx,
                           size_t endIdx) {
    std::vector<Point2d> points = ExtractPoints(contour, startIdx, endIdx);

    if (points.size() < 3) return 0.0;

    // Fit circle and compute how well points match
    CircleFitResult fitResult = FitCircle(points, CircleFitMethod::Algebraic);

    if (!fitResult.success || fitResult.circle.radius <= 0) {
        return 0.0;
    }

    // Compute mean distance to circle
    double sumDist = 0.0;
    for (const auto& p : points) {
        double d = std::abs(p.DistanceTo(fitResult.circle.center) - fitResult.circle.radius);
        sumDist += d;
    }
    double meanDist = sumDist / points.size();

    // Circularity based on how close points are to fitted circle
    // Scale by radius to be resolution-independent
    double normalized = meanDist / std::max(1.0, fitResult.circle.radius);
    return std::max(0.0, 1.0 - normalized * 10.0);
}

// =============================================================================
// Sub-Contour Extraction
// =============================================================================

QContourArray SplitContourAtIndices(const QContour& contour,
                                     const std::vector<size_t>& splitIndices) {
    QContourArray result;

    if (contour.Empty() || splitIndices.empty()) {
        if (!contour.Empty()) {
            result.Add(contour);
        }
        return result;
    }

    // Sort and remove duplicates
    std::vector<size_t> indices = splitIndices;
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

    // Add first index if not present
    if (indices.front() != 0) {
        indices.insert(indices.begin(), 0);
    }

    // Add last index if not present
    if (indices.back() != contour.Size() - 1) {
        indices.push_back(contour.Size() - 1);
    }

    // Extract sub-contours
    for (size_t i = 0; i + 1 < indices.size(); ++i) {
        QContour sub = ExtractSubContour(contour, indices[i], indices[i + 1]);
        if (sub.Size() >= 2) {
            result.Add(std::move(sub));
        }
    }

    return result;
}

QContour ExtractSubContour(const QContour& contour,
                            size_t startIdx,
                            size_t endIdx) {
    QContour result;

    if (contour.Empty() || startIdx >= contour.Size()) {
        return result;
    }

    endIdx = std::min(endIdx, contour.Size() - 1);
    result.Reserve(endIdx - startIdx + 1);

    for (size_t i = startIdx; i <= endIdx; ++i) {
        result.AddPoint(contour[i]);
    }

    result.SetClosed(false);  // Sub-contours are open
    return result;
}

// =============================================================================
// Main Segmentation Functions
// =============================================================================

SegmentationResult SegmentContour(const QContour& contour,
                                   const SegmentParams& params) {
    SegmentationResult result;

    if (contour.Size() < 3) {
        return result;
    }

    // Optionally smooth first
    QContour workContour = contour;
    if (params.smoothBeforeSegment) {
        GaussianSmoothParams smoothParams;
        smoothParams.sigma = params.smoothSigma;
        workContour = SmoothContourGaussian(contour, smoothParams);
    }

    // Find split points based on algorithm
    std::vector<size_t> splits;

    switch (params.algorithm) {
        case SegmentAlgorithm::Curvature:
            splits = DetectCorners(workContour, params.curvatureThreshold,
                                   params.curvatureWindowSize);
            break;

        case SegmentAlgorithm::ErrorBased:
            splits = FindLineSplitPoints(workContour, params.maxLineError);
            break;

        case SegmentAlgorithm::Hybrid:
        default: {
            // Combine corner detection and error-based splitting
            std::vector<size_t> corners = DetectCorners(workContour,
                                                         params.curvatureThreshold,
                                                         params.curvatureWindowSize);
            std::vector<size_t> errorSplits = FindLineSplitPoints(workContour,
                                                                   params.maxLineError);

            // Merge both sets
            splits.reserve(corners.size() + errorSplits.size());
            splits.insert(splits.end(), corners.begin(), corners.end());
            splits.insert(splits.end(), errorSplits.begin(), errorSplits.end());
            std::sort(splits.begin(), splits.end());
            splits.erase(std::unique(splits.begin(), splits.end()), splits.end());
            break;
        }
    }

    // Ensure first and last points are included
    if (splits.empty() || splits.front() != 0) {
        splits.insert(splits.begin(), 0);
    }
    if (splits.back() != workContour.Size() - 1) {
        splits.push_back(workContour.Size() - 1);
    }

    // Fit primitives to each segment
    double totalError = 0.0;
    size_t totalPoints = 0;

    for (size_t i = 0; i + 1 < splits.size(); ++i) {
        size_t startIdx = splits[i];
        size_t endIdx = splits[i + 1];

        if (endIdx <= startIdx) continue;

        PrimitiveFitResult fit;

        switch (params.mode) {
            case SegmentMode::LinesOnly:
                fit = FitLineToContour(workContour, startIdx, endIdx);
                break;

            case SegmentMode::ArcsOnly:
                fit = FitArcToContour(workContour, startIdx, endIdx);
                break;

            case SegmentMode::LinesAndArcs:
            default:
                fit = FitBestPrimitive(workContour, startIdx, endIdx, true);
                break;
        }

        // Check if fit meets constraints
        if (fit.IsValid()) {
            // Check minimum length
            if (fit.Length() < params.minSegmentLength) {
                continue;
            }

            // Check arc constraints
            if (fit.type == PrimitiveType::Arc) {
                if (std::abs(fit.arc.sweepAngle) < params.minArcSweep ||
                    fit.arc.radius > params.maxArcRadius) {
                    // Try fitting as line instead
                    fit = FitLineToContour(workContour, startIdx, endIdx);
                }
            }

            if (fit.IsValid()) {
                result.primitives.push_back(fit);
                totalError += fit.error * fit.numPoints;
                totalPoints += fit.numPoints;
            }
        }
    }

    // Merge collinear segments if requested
    if (params.mergeCollinear && params.mode != SegmentMode::ArcsOnly) {
        // Simple merge of adjacent collinear lines
        std::vector<PrimitiveFitResult> merged;

        for (size_t i = 0; i < result.primitives.size(); ++i) {
            if (merged.empty()) {
                merged.push_back(result.primitives[i]);
                continue;
            }

            PrimitiveFitResult& prev = merged.back();
            const PrimitiveFitResult& curr = result.primitives[i];

            // Only merge lines
            if (prev.type == PrimitiveType::Line && curr.type == PrimitiveType::Line) {
                // Check if collinear
                double angleDiff = std::abs(prev.segment.Angle() - curr.segment.Angle());
                if (angleDiff < 0.05 || std::abs(angleDiff - PI) < 0.05) {
                    // Merge
                    prev.segment.p2 = curr.segment.p2;
                    prev.endIndex = curr.endIndex;
                    prev.numPoints += curr.numPoints;
                    prev.error = (prev.error + curr.error) / 2;  // Approximate
                    prev.maxError = std::max(prev.maxError, curr.maxError);
                    continue;
                }
            }

            merged.push_back(curr);
        }

        result.primitives = std::move(merged);
    }

    // Compute overall statistics
    if (totalPoints > 0) {
        result.totalError = totalError / totalPoints;
    }
    // Cap at 1.0 since overlapping points at segment boundaries are counted multiple times
    result.coverageRatio = std::min(1.0, static_cast<double>(totalPoints) / workContour.Size());

    return result;
}

std::vector<Segment2d> SegmentContourToLines(const QContour& contour,
                                              double maxError) {
    SegmentParams params;
    params.mode = SegmentMode::LinesOnly;
    params.maxLineError = maxError;

    SegmentationResult result = SegmentContour(contour, params);
    return result.GetLines();
}

std::vector<Arc2d> SegmentContourToArcs(const QContour& contour,
                                         double maxError) {
    SegmentParams params;
    params.mode = SegmentMode::ArcsOnly;
    params.maxArcError = maxError;

    SegmentationResult result = SegmentContour(contour, params);
    return result.GetArcs();
}

// =============================================================================
// Utility Functions
// =============================================================================

std::vector<Segment2d> MergeCollinearSegments(const std::vector<Segment2d>& segments,
                                               double angleThreshold,
                                               double gapThreshold) {
    if (segments.empty()) return {};

    std::vector<Segment2d> result;
    result.push_back(segments[0]);

    for (size_t i = 1; i < segments.size(); ++i) {
        Segment2d& prev = result.back();
        const Segment2d& curr = segments[i];

        // Check angle difference
        double angleDiff = std::abs(NormalizeAngleLocal(prev.Angle() - curr.Angle()));
        bool angleMatch = angleDiff < angleThreshold ||
                          std::abs(angleDiff - PI) < angleThreshold;

        // Check gap
        double gap = prev.p2.DistanceTo(curr.p1);
        bool gapOk = gap < gapThreshold;

        if (angleMatch && gapOk) {
            // Merge
            prev.p2 = curr.p2;
        } else {
            result.push_back(curr);
        }
    }

    return result;
}

std::vector<Arc2d> MergeSimilarArcs(const std::vector<Arc2d>& arcs,
                                     double radiusThreshold,
                                     double gapThreshold) {
    if (arcs.empty()) return {};

    std::vector<Arc2d> result;
    result.push_back(arcs[0]);

    for (size_t i = 1; i < arcs.size(); ++i) {
        Arc2d& prev = result.back();
        const Arc2d& curr = arcs[i];

        // Check radius similarity
        double radiusDiff = std::abs(prev.radius - curr.radius) /
                           std::max(prev.radius, curr.radius);
        bool radiusMatch = radiusDiff < radiusThreshold;

        // Check center proximity
        double centerDist = prev.center.DistanceTo(curr.center);
        bool centerMatch = centerDist < prev.radius * radiusThreshold;

        // Check gap
        Point2d prevEnd = prev.EndPoint();
        Point2d currStart = curr.StartPoint();
        double gap = prevEnd.DistanceTo(currStart);
        bool gapOk = gap < gapThreshold;

        if (radiusMatch && centerMatch && gapOk) {
            // Merge - extend sweep angle
            prev.sweepAngle += curr.sweepAngle;
        } else {
            result.push_back(curr);
        }
    }

    return result;
}

QContour SegmentToContour(const Segment2d& segment, double pointSpacing) {
    QContour contour;

    double len = segment.Length();
    if (len < 1e-10) {
        contour.AddPoint(segment.p1.x, segment.p1.y);
        return contour;
    }

    int numPoints = std::max(2, static_cast<int>(std::ceil(len / pointSpacing)) + 1);
    contour.Reserve(numPoints);

    for (int i = 0; i < numPoints; ++i) {
        double t = static_cast<double>(i) / (numPoints - 1);
        double x = segment.p1.x + t * (segment.p2.x - segment.p1.x);
        double y = segment.p1.y + t * (segment.p2.y - segment.p1.y);
        contour.AddPoint(x, y);
    }

    contour.SetClosed(false);
    return contour;
}

QContour ArcToContour(const Arc2d& arc, double pointSpacing) {
    QContour contour;

    double arcLen = arc.Length();
    if (arcLen < 1e-10 || arc.radius < 1e-10) {
        contour.AddPoint(arc.StartPoint().x, arc.StartPoint().y);
        return contour;
    }

    int numPoints = std::max(3, static_cast<int>(std::ceil(arcLen / pointSpacing)) + 1);
    contour.Reserve(numPoints);

    for (int i = 0; i < numPoints; ++i) {
        double t = static_cast<double>(i) / (numPoints - 1);
        double angle = arc.startAngle + t * arc.sweepAngle;
        double x = arc.center.x + arc.radius * std::cos(angle);
        double y = arc.center.y + arc.radius * std::sin(angle);
        contour.AddPoint(x, y);
    }

    contour.SetClosed(false);
    return contour;
}

QContourArray PrimitivesToContours(const SegmentationResult& result,
                                    double pointSpacing) {
    QContourArray contours;

    for (const auto& prim : result.primitives) {
        if (prim.type == PrimitiveType::Line) {
            contours.Add(SegmentToContour(prim.segment, pointSpacing));
        } else if (prim.type == PrimitiveType::Arc) {
            contours.Add(ArcToContour(prim.arc, pointSpacing));
        }
    }

    return contours;
}

} // namespace Qi::Vision::Internal
