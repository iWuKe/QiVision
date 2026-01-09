#pragma once

/**
 * @file Hough.h
 * @brief Hough Transform for line and circle detection
 *
 * This module provides:
 * - Standard Hough Transform for line detection
 * - Probabilistic Hough Transform (faster line segments)
 * - Hough Circle Transform
 * - Accumulator space operations
 *
 * Used by:
 * - Edge module (line/circle detection from edge images)
 * - Metrology (geometric feature extraction)
 * - Blob analysis (shape detection)
 *
 * Design principles:
 * - Works with binary edge images or edge point lists
 * - Configurable resolution and threshold parameters
 * - Multiple detection methods for different use cases
 * - Sub-pixel refinement for high precision
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Internal/Matrix.h>

#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Default angular resolution for Hough lines (1 degree)
constexpr double HOUGH_DEFAULT_THETA_RESOLUTION = 0.017453292519943295;  // PI/180

/// Default distance resolution for Hough lines (1 pixel)
constexpr double HOUGH_DEFAULT_RHO_RESOLUTION = 1.0;

/// Default threshold ratio for line detection
constexpr double HOUGH_DEFAULT_THRESHOLD_RATIO = 0.3;

/// Minimum line length for probabilistic Hough (pixels)
constexpr double HOUGH_DEFAULT_MIN_LINE_LENGTH = 30.0;

/// Maximum line gap for probabilistic Hough (pixels)
constexpr double HOUGH_DEFAULT_MAX_LINE_GAP = 10.0;

/// Default radius step for circle detection
constexpr double HOUGH_DEFAULT_RADIUS_STEP = 1.0;

/// Default accumulator threshold for circles
constexpr double HOUGH_DEFAULT_CIRCLE_THRESHOLD = 0.5;

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Detected line in Hough space (polar form)
 *
 * Line equation: x*cos(theta) + y*sin(theta) = rho
 * where theta is angle from x-axis, rho is perpendicular distance from origin
 */
struct HoughLine {
    double rho;         ///< Distance from origin to line (can be negative)
    double theta;       ///< Angle in radians [0, PI)
    double score;       ///< Accumulator vote count / strength

    HoughLine() : rho(0), theta(0), score(0) {}
    HoughLine(double r, double t, double s = 0) : rho(r), theta(t), score(s) {}

    /// Convert to Line2d (requires image size for bounds)
    Line2d ToLine2d() const;

    /// Get two points on the line (for visualization)
    std::pair<Point2d, Point2d> GetTwoPoints(double length = 1000.0) const;
};

/**
 * @brief Detected line segment from probabilistic Hough
 */
struct HoughLineSegment {
    Point2d p1;         ///< Start point
    Point2d p2;         ///< End point
    double score;       ///< Detection confidence

    HoughLineSegment() : score(0) {}
    HoughLineSegment(const Point2d& a, const Point2d& b, double s = 0)
        : p1(a), p2(b), score(s) {}

    /// Get length of segment
    double Length() const;

    /// Convert to Segment2d
    Segment2d ToSegment2d() const { return Segment2d(p1, p2); }

    /// Get angle of segment
    double Angle() const;
};

/**
 * @brief Detected circle from Hough Circle Transform
 */
struct HoughCircle {
    Point2d center;     ///< Circle center
    double radius;      ///< Circle radius
    double score;       ///< Accumulator vote count / strength

    HoughCircle() : radius(0), score(0) {}
    HoughCircle(const Point2d& c, double r, double s = 0)
        : center(c), radius(r), score(s) {}

    /// Convert to Circle2d
    Circle2d ToCircle2d() const { return Circle2d(center, radius); }
};

/**
 * @brief Hough accumulator (for visualization/debugging)
 */
struct HoughAccumulator {
    MatX data;          ///< Accumulator matrix
    double rhoMin;      ///< Minimum rho value
    double rhoMax;      ///< Maximum rho value
    double rhoStep;     ///< Rho resolution
    double thetaMin;    ///< Minimum theta (usually 0)
    double thetaMax;    ///< Maximum theta (usually PI)
    double thetaStep;   ///< Theta resolution

    /// Get rho value for row index
    double GetRho(int row) const { return rhoMin + row * rhoStep; }

    /// Get theta value for column index
    double GetTheta(int col) const { return thetaMin + col * thetaStep; }

    /// Get row index for rho value
    int GetRhoIndex(double rho) const;

    /// Get column index for theta value
    int GetThetaIndex(double theta) const;
};

// =============================================================================
// Configuration Structures
// =============================================================================

/**
 * @brief Parameters for standard Hough line transform
 */
struct HoughLineParams {
    double rhoResolution;       ///< Distance resolution in pixels
    double thetaResolution;     ///< Angle resolution in radians
    double threshold;           ///< Minimum votes to consider a line (absolute or ratio)
    bool thresholdIsRatio;      ///< If true, threshold is ratio of max votes
    int maxLines;               ///< Maximum number of lines to return (0 = unlimited)
    double minDistance;         ///< Minimum distance between detected lines (in rho-theta space)
    bool suppressOverlapping;   ///< Remove near-duplicate lines

    HoughLineParams()
        : rhoResolution(HOUGH_DEFAULT_RHO_RESOLUTION)
        , thetaResolution(HOUGH_DEFAULT_THETA_RESOLUTION)
        , threshold(HOUGH_DEFAULT_THRESHOLD_RATIO)
        , thresholdIsRatio(true)
        , maxLines(0)
        , minDistance(10.0)
        , suppressOverlapping(true) {}
};

/**
 * @brief Parameters for probabilistic Hough line transform
 */
struct HoughLineProbParams {
    double rhoResolution;       ///< Distance resolution in pixels
    double thetaResolution;     ///< Angle resolution in radians
    double threshold;           ///< Minimum votes
    double minLineLength;       ///< Minimum length of line segment
    double maxLineGap;          ///< Maximum gap between points on same line
    int maxLines;               ///< Maximum number of segments to return

    HoughLineProbParams()
        : rhoResolution(HOUGH_DEFAULT_RHO_RESOLUTION)
        , thetaResolution(HOUGH_DEFAULT_THETA_RESOLUTION)
        , threshold(50)
        , minLineLength(HOUGH_DEFAULT_MIN_LINE_LENGTH)
        , maxLineGap(HOUGH_DEFAULT_MAX_LINE_GAP)
        , maxLines(0) {}
};

/**
 * @brief Parameters for Hough circle transform
 */
struct HoughCircleParams {
    double dp;                  ///< Inverse ratio of accumulator resolution to image
    double minDist;             ///< Minimum distance between circle centers
    double param1;              ///< Higher Canny edge threshold (for gradient method)
    double param2;              ///< Accumulator threshold for circle centers
    int minRadius;              ///< Minimum circle radius
    int maxRadius;              ///< Maximum circle radius (0 = use image diagonal)
    int maxCircles;             ///< Maximum number of circles to return

    HoughCircleParams()
        : dp(1.0)
        , minDist(20.0)
        , param1(100.0)
        , param2(50.0)
        , minRadius(5)
        , maxRadius(0)
        , maxCircles(0) {}
};

// =============================================================================
// Line Detection Functions
// =============================================================================

/**
 * @brief Standard Hough Transform for line detection
 *
 * Detects lines in a binary edge image using accumulator voting.
 * Lines are returned in polar form (rho, theta).
 *
 * @param edgeImage Binary edge image (non-zero pixels are edge points)
 * @param params Detection parameters
 * @return Vector of detected lines sorted by score (descending)
 *
 * Complexity: O(n * m * numAngles) where n*m is image size
 */
std::vector<HoughLine> HoughLines(const QImage& edgeImage,
                                   const HoughLineParams& params = HoughLineParams());

/**
 * @brief Standard Hough Transform from edge point list
 *
 * More efficient when edge points are already extracted.
 *
 * @param points List of edge points
 * @param imageWidth Image width (for rho range calculation)
 * @param imageHeight Image height
 * @param params Detection parameters
 * @return Vector of detected lines
 */
std::vector<HoughLine> HoughLines(const std::vector<Point2d>& points,
                                   int imageWidth, int imageHeight,
                                   const HoughLineParams& params = HoughLineParams());

/**
 * @brief Probabilistic Hough Transform for line segment detection
 *
 * Faster than standard Hough, returns actual line segments instead of
 * infinite lines. Good for detecting finite line segments in images.
 *
 * @param edgeImage Binary edge image
 * @param params Detection parameters
 * @return Vector of detected line segments
 *
 * Based on: Matas, Galambos, Kittler (2000) "Robust Detection of Lines Using
 * the Progressive Probabilistic Hough Transform"
 */
std::vector<HoughLineSegment> HoughLinesP(const QImage& edgeImage,
                                           const HoughLineProbParams& params = HoughLineProbParams());

/**
 * @brief Probabilistic Hough from edge point list
 */
std::vector<HoughLineSegment> HoughLinesP(const std::vector<Point2d>& points,
                                           int imageWidth, int imageHeight,
                                           const HoughLineProbParams& params = HoughLineProbParams());

/**
 * @brief Get Hough accumulator (for visualization/debugging)
 *
 * @param edgeImage Binary edge image
 * @param params Detection parameters
 * @return Accumulator structure with vote counts
 */
HoughAccumulator GetHoughAccumulator(const QImage& edgeImage,
                                      const HoughLineParams& params = HoughLineParams());

/**
 * @brief Get Hough accumulator from points
 */
HoughAccumulator GetHoughAccumulator(const std::vector<Point2d>& points,
                                      int imageWidth, int imageHeight,
                                      const HoughLineParams& params = HoughLineParams());

// =============================================================================
// Circle Detection Functions
// =============================================================================

/**
 * @brief Hough Circle Transform (gradient-based)
 *
 * Detects circles using the gradient direction at edge points.
 * More efficient than standard 3D Hough for circles.
 *
 * @param image Grayscale image (edge detection done internally)
 * @param params Detection parameters
 * @return Vector of detected circles sorted by score (descending)
 *
 * Note: Uses two-stage detection:
 * 1. Find circle centers using gradient voting
 * 2. Find radii for each center
 */
std::vector<HoughCircle> HoughCircles(const QImage& image,
                                       const HoughCircleParams& params = HoughCircleParams());

/**
 * @brief Hough Circle Transform from edge image with gradient
 *
 * @param edgeImage Binary edge image
 * @param gradientX X-component of gradient
 * @param gradientY Y-component of gradient
 * @param params Detection parameters
 * @return Vector of detected circles
 */
std::vector<HoughCircle> HoughCircles(const QImage& edgeImage,
                                       const QImage& gradientX,
                                       const QImage& gradientY,
                                       const HoughCircleParams& params = HoughCircleParams());

/**
 * @brief Standard Hough Circle Transform (3D accumulator)
 *
 * Full 3D Hough accumulator (x, y, r). More accurate but slower
 * than gradient-based method.
 *
 * @param edgeImage Binary edge image
 * @param minRadius Minimum circle radius
 * @param maxRadius Maximum circle radius
 * @param threshold Minimum accumulator votes
 * @param maxCircles Maximum circles to return (0 = unlimited)
 * @return Vector of detected circles
 */
std::vector<HoughCircle> HoughCirclesStandard(const QImage& edgeImage,
                                               int minRadius, int maxRadius,
                                               double threshold = 0.5,
                                               int maxCircles = 0);

/**
 * @brief Hough Circle from edge point list
 *
 * @param points List of edge points
 * @param minRadius Minimum circle radius
 * @param maxRadius Maximum circle radius
 * @param threshold Accumulator threshold
 * @return Vector of detected circles
 */
std::vector<HoughCircle> HoughCircles(const std::vector<Point2d>& points,
                                       int minRadius, int maxRadius,
                                       double threshold = 0.5,
                                       int maxCircles = 0);

// =============================================================================
// Refinement Functions
// =============================================================================

/**
 * @brief Refine line parameters using edge points
 *
 * Performs least-squares fitting on edge points near the detected line.
 *
 * @param line Initial line estimate
 * @param edgeImage Binary edge image
 * @param searchWidth Width of search region perpendicular to line
 * @return Refined line parameters
 */
HoughLine RefineHoughLine(const HoughLine& line,
                          const QImage& edgeImage,
                          double searchWidth = 5.0);

/**
 * @brief Refine line using point list
 */
HoughLine RefineHoughLine(const HoughLine& line,
                          const std::vector<Point2d>& points,
                          double searchWidth = 5.0);

/**
 * @brief Refine circle parameters using edge points
 *
 * Performs circle fitting on edge points near the detected circle.
 *
 * @param circle Initial circle estimate
 * @param edgeImage Binary edge image
 * @param searchWidth Width of search region
 * @return Refined circle parameters
 */
HoughCircle RefineHoughCircle(const HoughCircle& circle,
                               const QImage& edgeImage,
                               double searchWidth = 5.0);

/**
 * @brief Refine circle using point list
 */
HoughCircle RefineHoughCircle(const HoughCircle& circle,
                               const std::vector<Point2d>& points,
                               double searchWidth = 5.0);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Convert HoughLine to Cartesian form (ax + by + c = 0)
 *
 * @param line Hough line in polar form
 * @return Line2d in Cartesian form
 */
Line2d HoughLineToCartesian(const HoughLine& line);

/**
 * @brief Convert Cartesian line to Hough form
 *
 * @param line Line in Cartesian form
 * @return HoughLine in polar form
 */
HoughLine CartesianToHoughLine(const Line2d& line);

/**
 * @brief Get points on a Hough line within image bounds
 *
 * @param line Hough line
 * @param width Image width
 * @param height Image height
 * @return Segment clipped to image bounds, or empty segment if no intersection
 */
Segment2d ClipHoughLineToImage(const HoughLine& line, int width, int height);

/**
 * @brief Merge similar lines (non-maximum suppression)
 *
 * @param lines Input lines
 * @param rhoThreshold Maximum rho difference for merging
 * @param thetaThreshold Maximum theta difference for merging
 * @return Merged lines (weighted average by score)
 */
std::vector<HoughLine> MergeHoughLines(const std::vector<HoughLine>& lines,
                                        double rhoThreshold = 10.0,
                                        double thetaThreshold = 0.1);

/**
 * @brief Merge similar circles (non-maximum suppression)
 *
 * @param circles Input circles
 * @param centerThreshold Maximum center distance for merging
 * @param radiusThreshold Maximum radius difference for merging
 * @return Merged circles
 */
std::vector<HoughCircle> MergeHoughCircles(const std::vector<HoughCircle>& circles,
                                            double centerThreshold = 10.0,
                                            double radiusThreshold = 5.0);

/**
 * @brief Find peaks in Hough accumulator
 *
 * @param accumulator Hough accumulator
 * @param threshold Minimum vote threshold
 * @param numPeaks Maximum number of peaks to return
 * @param minDistance Minimum distance between peaks (in accumulator space)
 * @return Vector of (rho, theta, score) tuples
 */
std::vector<HoughLine> FindAccumulatorPeaks(const HoughAccumulator& accumulator,
                                             double threshold,
                                             int numPeaks = 0,
                                             int minDistance = 10);

/**
 * @brief Calculate distance from point to Hough line
 *
 * @param point Query point
 * @param line Hough line
 * @return Signed perpendicular distance
 */
double PointToHoughLineDistance(const Point2d& point, const HoughLine& line);

/**
 * @brief Check if two Hough lines are approximately parallel
 *
 * @param line1 First line
 * @param line2 Second line
 * @param angleTolerance Maximum angle difference (radians)
 * @return True if parallel within tolerance
 */
bool AreHoughLinesParallel(const HoughLine& line1, const HoughLine& line2,
                           double angleTolerance = 0.05);

/**
 * @brief Check if two Hough lines are approximately perpendicular
 */
bool AreHoughLinesPerpendicular(const HoughLine& line1, const HoughLine& line2,
                                double angleTolerance = 0.05);

/**
 * @brief Get intersection point of two Hough lines
 *
 * @param line1 First line
 * @param line2 Second line
 * @param intersection Output intersection point
 * @return True if lines intersect (not parallel)
 */
bool HoughLinesIntersection(const HoughLine& line1, const HoughLine& line2,
                            Point2d& intersection);

} // namespace Qi::Vision::Internal
