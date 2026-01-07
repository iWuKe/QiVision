#pragma once

/**
 * @file ContourSegment.h
 * @brief Contour segmentation into geometric primitives for QiVision
 *
 * This module provides:
 * - Contour segmentation into line segments and circular arcs
 * - Curvature-based segmentation
 * - Error-based segmentation (Douglas-Peucker style)
 * - Corner detection and split point finding
 *
 * Reference Halcon operators:
 * - segment_contours_xld: Segment XLD contours into lines and arcs
 * - fit_line_contour_xld: Fit lines to contour segments
 * - fit_circle_contour_xld: Fit circles/arcs to contour segments
 *
 * Design principles:
 * - All functions are pure (input not modified)
 * - All coordinates use double for subpixel precision
 * - Support both open and closed contours
 */

#include <QiVision/Core/QContour.h>
#include <QiVision/Core/QContourArray.h>
#include <QiVision/Core/Types.h>

#include <vector>
#include <string>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Default maximum error for line fitting (pixels)
constexpr double DEFAULT_LINE_FIT_MAX_ERROR = 1.0;

/// Default maximum error for arc fitting (pixels)
constexpr double DEFAULT_ARC_FIT_MAX_ERROR = 1.0;

/// Minimum number of points for line fitting
constexpr size_t MIN_POINTS_FOR_LINE = 2;

/// Minimum number of points for arc fitting
constexpr size_t MIN_POINTS_FOR_ARC = 3;

/// Default curvature threshold for corner detection (1/radius)
constexpr double DEFAULT_CURVATURE_THRESHOLD = 0.1;

/// Minimum segment length for primitives (pixels)
constexpr double MIN_PRIMITIVE_LENGTH = 3.0;

/// Minimum arc sweep angle (radians)
constexpr double MIN_ARC_SWEEP_ANGLE = 0.1;

// =============================================================================
// Segmentation Mode
// =============================================================================

/**
 * @brief Segmentation mode for contour splitting
 */
enum class SegmentMode {
    LinesOnly,      ///< Segment into lines only
    ArcsOnly,       ///< Segment into arcs only
    LinesAndArcs    ///< Segment into both lines and arcs (default)
};

/**
 * @brief Segmentation algorithm
 */
enum class SegmentAlgorithm {
    Curvature,      ///< Split at high curvature points (corners)
    ErrorBased,     ///< Split when fitting error exceeds threshold
    Hybrid          ///< Combination of curvature and error (default)
};

// =============================================================================
// Segment Result Types
// =============================================================================

/**
 * @brief Type of geometric primitive
 */
enum class PrimitiveType {
    Line,           ///< Straight line segment
    Arc,            ///< Circular arc
    Unknown         ///< Unclassified segment
};

/**
 * @brief Result of fitting a geometric primitive to contour points
 */
struct PrimitiveFitResult {
    PrimitiveType type = PrimitiveType::Unknown;

    // For lines
    Segment2d segment;

    // For arcs
    Arc2d arc;

    // Fitting quality
    double error = 0.0;         ///< RMS fitting error
    double maxError = 0.0;      ///< Maximum point-to-primitive distance
    size_t startIndex = 0;      ///< Start index in original contour
    size_t endIndex = 0;        ///< End index in original contour
    size_t numPoints = 0;       ///< Number of points fitted

    /// Check if fit is valid
    bool IsValid() const { return type != PrimitiveType::Unknown && numPoints > 0; }

    /// Get primitive length
    double Length() const {
        if (type == PrimitiveType::Line) return segment.Length();
        if (type == PrimitiveType::Arc) return arc.Length();
        return 0.0;
    }
};

/**
 * @brief Contour segmentation result
 */
struct SegmentationResult {
    std::vector<PrimitiveFitResult> primitives;  ///< Fitted primitives in order
    double totalError = 0.0;                      ///< Total fitting error
    double coverageRatio = 0.0;                   ///< Ratio of contour covered

    /// Number of line segments
    size_t LineCount() const {
        size_t count = 0;
        for (const auto& p : primitives) {
            if (p.type == PrimitiveType::Line) ++count;
        }
        return count;
    }

    /// Number of arcs
    size_t ArcCount() const {
        size_t count = 0;
        for (const auto& p : primitives) {
            if (p.type == PrimitiveType::Arc) ++count;
        }
        return count;
    }

    /// Get all line segments
    std::vector<Segment2d> GetLines() const {
        std::vector<Segment2d> lines;
        for (const auto& p : primitives) {
            if (p.type == PrimitiveType::Line) {
                lines.push_back(p.segment);
            }
        }
        return lines;
    }

    /// Get all arcs
    std::vector<Arc2d> GetArcs() const {
        std::vector<Arc2d> arcs;
        for (const auto& p : primitives) {
            if (p.type == PrimitiveType::Arc) {
                arcs.push_back(p.arc);
            }
        }
        return arcs;
    }
};

// =============================================================================
// Segmentation Parameters
// =============================================================================

/**
 * @brief Parameters for contour segmentation
 */
struct SegmentParams {
    SegmentMode mode = SegmentMode::LinesAndArcs;
    SegmentAlgorithm algorithm = SegmentAlgorithm::Hybrid;

    // Error thresholds
    double maxLineError = DEFAULT_LINE_FIT_MAX_ERROR;   ///< Max error for lines
    double maxArcError = DEFAULT_ARC_FIT_MAX_ERROR;     ///< Max error for arcs

    // Curvature thresholds
    double curvatureThreshold = DEFAULT_CURVATURE_THRESHOLD;  ///< Corner detection
    int curvatureWindowSize = 5;                               ///< Window for curvature

    // Constraints
    double minSegmentLength = MIN_PRIMITIVE_LENGTH;   ///< Min primitive length
    double minArcSweep = MIN_ARC_SWEEP_ANGLE;       ///< Min arc sweep angle
    double maxArcRadius = 1e6;                      ///< Max arc radius (reject near-lines)

    // Options
    bool mergeCollinear = true;     ///< Merge adjacent collinear segments
    bool smoothBeforeSegment = false;  ///< Apply smoothing first
    double smoothSigma = 0.5;       ///< Smoothing sigma if enabled
};

// =============================================================================
// Main Segmentation Functions
// =============================================================================

/**
 * @brief Segment a contour into geometric primitives (lines and/or arcs)
 *
 * @param contour Input contour
 * @param params Segmentation parameters
 * @return Segmentation result with fitted primitives
 */
SegmentationResult SegmentContour(const QContour& contour,
                                   const SegmentParams& params = SegmentParams());

/**
 * @brief Segment a contour into line segments only
 *
 * Convenience function for line-only segmentation.
 *
 * @param contour Input contour
 * @param maxError Maximum fitting error (pixels)
 * @return Vector of line segments
 */
std::vector<Segment2d> SegmentContourToLines(const QContour& contour,
                                              double maxError = DEFAULT_LINE_FIT_MAX_ERROR);

/**
 * @brief Segment a contour into circular arcs only
 *
 * @param contour Input contour
 * @param maxError Maximum fitting error (pixels)
 * @return Vector of arcs
 */
std::vector<Arc2d> SegmentContourToArcs(const QContour& contour,
                                         double maxError = DEFAULT_ARC_FIT_MAX_ERROR);

// =============================================================================
// Corner Detection
// =============================================================================

/**
 * @brief Detect corner points in a contour
 *
 * Uses curvature analysis to find high curvature points (corners).
 *
 * @param contour Input contour
 * @param curvatureThreshold Minimum curvature for corner detection
 * @param windowSize Window size for curvature computation
 * @return Indices of corner points
 */
std::vector<size_t> DetectCorners(const QContour& contour,
                                   double curvatureThreshold = DEFAULT_CURVATURE_THRESHOLD,
                                   int windowSize = 5);

/**
 * @brief Detect dominant points (corners and inflection points)
 *
 * More comprehensive than corner detection - also finds inflection points
 * where curvature changes sign.
 *
 * @param contour Input contour
 * @param curvatureThreshold Threshold for corner detection
 * @param windowSize Window size for curvature
 * @return Indices of dominant points
 */
std::vector<size_t> DetectDominantPoints(const QContour& contour,
                                          double curvatureThreshold = DEFAULT_CURVATURE_THRESHOLD,
                                          int windowSize = 5);

// =============================================================================
// Split Point Finding
// =============================================================================

/**
 * @brief Find optimal split points for line fitting
 *
 * Uses Douglas-Peucker style algorithm to find points where
 * line fitting error exceeds threshold.
 *
 * @param contour Input contour
 * @param maxError Maximum allowed fitting error
 * @return Indices where contour should be split
 */
std::vector<size_t> FindLineSplitPoints(const QContour& contour,
                                         double maxError = DEFAULT_LINE_FIT_MAX_ERROR);

/**
 * @brief Find optimal split points for arc fitting
 *
 * @param contour Input contour
 * @param maxError Maximum allowed fitting error
 * @return Indices where contour should be split
 */
std::vector<size_t> FindArcSplitPoints(const QContour& contour,
                                        double maxError = DEFAULT_ARC_FIT_MAX_ERROR);

// =============================================================================
// Primitive Fitting Functions
// =============================================================================

/**
 * @brief Fit a line segment to contour points
 *
 * @param contour Input contour
 * @param startIdx Start index
 * @param endIdx End index (inclusive)
 * @return Fitted line segment with error statistics
 */
PrimitiveFitResult FitLineToContour(const QContour& contour,
                                     size_t startIdx,
                                     size_t endIdx);

/**
 * @brief Fit a circular arc to contour points
 *
 * @param contour Input contour
 * @param startIdx Start index
 * @param endIdx End index (inclusive)
 * @return Fitted arc with error statistics
 */
PrimitiveFitResult FitArcToContour(const QContour& contour,
                                    size_t startIdx,
                                    size_t endIdx);

/**
 * @brief Fit best primitive (line or arc) to contour points
 *
 * Automatically chooses between line and arc based on fitting error.
 *
 * @param contour Input contour
 * @param startIdx Start index
 * @param endIdx End index (inclusive)
 * @param preferLine If true, prefer line when both fit equally well
 * @return Best fitting primitive
 */
PrimitiveFitResult FitBestPrimitive(const QContour& contour,
                                     size_t startIdx,
                                     size_t endIdx,
                                     bool preferLine = true);

// =============================================================================
// Sub-Contour Extraction
// =============================================================================

/**
 * @brief Extract sub-contours at split points
 *
 * Splits contour at given indices and returns array of sub-contours.
 *
 * @param contour Input contour
 * @param splitIndices Indices where to split
 * @return Array of sub-contours
 */
QContourArray SplitContourAtIndices(const QContour& contour,
                                     const std::vector<size_t>& splitIndices);

/**
 * @brief Extract sub-contour between two indices
 *
 * @param contour Input contour
 * @param startIdx Start index
 * @param endIdx End index (inclusive)
 * @return Sub-contour
 */
QContour ExtractSubContour(const QContour& contour,
                            size_t startIdx,
                            size_t endIdx);

// =============================================================================
// Primitive Classification
// =============================================================================

/**
 * @brief Determine if contour segment is more like a line or arc
 *
 * @param contour Input contour
 * @param startIdx Start index
 * @param endIdx End index (inclusive)
 * @return PrimitiveType (Line, Arc, or Unknown if ambiguous)
 */
PrimitiveType ClassifyContourSegment(const QContour& contour,
                                      size_t startIdx,
                                      size_t endIdx);

/**
 * @brief Compute linearity measure for contour segment
 *
 * Returns a value from 0 (curved) to 1 (perfectly straight).
 *
 * @param contour Input contour
 * @param startIdx Start index
 * @param endIdx End index (inclusive)
 * @return Linearity measure [0, 1]
 */
double ComputeLinearity(const QContour& contour,
                         size_t startIdx,
                         size_t endIdx);

/**
 * @brief Compute circularity measure for contour segment
 *
 * Returns a value from 0 (not circular) to 1 (perfect arc).
 *
 * @param contour Input contour
 * @param startIdx Start index
 * @param endIdx End index (inclusive)
 * @return Circularity measure [0, 1]
 */
double ComputeCircularity(const QContour& contour,
                           size_t startIdx,
                           size_t endIdx);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Merge adjacent collinear line segments
 *
 * @param segments Input segments
 * @param angleThreshold Maximum angle difference (radians) to merge
 * @param gapThreshold Maximum gap between segments to merge
 * @return Merged segments
 */
std::vector<Segment2d> MergeCollinearSegments(const std::vector<Segment2d>& segments,
                                               double angleThreshold = 0.05,
                                               double gapThreshold = 2.0);

/**
 * @brief Merge adjacent arcs with similar curvature
 *
 * @param arcs Input arcs
 * @param radiusThreshold Maximum radius difference ratio to merge
 * @param gapThreshold Maximum gap between arcs to merge
 * @return Merged arcs
 */
std::vector<Arc2d> MergeSimilarArcs(const std::vector<Arc2d>& arcs,
                                     double radiusThreshold = 0.1,
                                     double gapThreshold = 2.0);

/**
 * @brief Convert line segment to contour
 *
 * @param segment Line segment
 * @param pointSpacing Distance between points
 * @return Contour representing the segment
 */
QContour SegmentToContour(const Segment2d& segment, double pointSpacing = 1.0);

/**
 * @brief Convert arc to contour
 *
 * @param arc Circular arc
 * @param pointSpacing Arc length between points
 * @return Contour representing the arc
 */
QContour ArcToContour(const Arc2d& arc, double pointSpacing = 1.0);

/**
 * @brief Convert all primitives in segmentation result to a contour array
 *
 * @param result Segmentation result
 * @param pointSpacing Distance between points
 * @return Array of contours for each primitive
 */
QContourArray PrimitivesToContours(const SegmentationResult& result,
                                    double pointSpacing = 1.0);

} // namespace Qi::Vision::Internal
