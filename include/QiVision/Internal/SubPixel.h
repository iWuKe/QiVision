#pragma once

/**
 * @file SubPixel.h
 * @brief Subpixel refinement algorithms for QiVision
 *
 * This module provides:
 * - 1D subpixel peak/extremum localization
 * - 2D subpixel peak localization (response surfaces)
 * - Edge subpixel localization
 * - Corner subpixel refinement
 * - Template matching refinement
 *
 * Used by:
 * - Matching/ShapeModel: Match position refinement
 * - Measure/Caliper: Edge position refinement
 * - Edge/SubPixelEdge: Edge detection
 * - Calib: Calibration point localization
 *
 * Precision targets (standard conditions: contrast>=50, noise sigma<=5):
 * - 1D extremum: < 0.02 px (1 sigma)
 * - 2D peak: < 0.05 px (1 sigma)
 * - Edge position: < 0.02 px (1 sigma)
 *
 * Design principles:
 * - Pure functions, no global state
 * - Multiple methods for different scenarios
 * - Graceful degradation for edge cases
 * - Confidence estimation for quality assessment
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Internal/Matrix.h>
#include <QiVision/Internal/Interpolate.h>

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Maximum allowed subpixel offset from integer position (prevents runaway)
constexpr double SUBPIXEL_MAX_OFFSET = 0.5;

/// Minimum curvature for valid parabolic fit (prevents flat region false positives)
constexpr double SUBPIXEL_MIN_CURVATURE = 1e-6;

/// Default window half-size for centroid calculation
constexpr int32_t SUBPIXEL_CENTROID_HALF_WINDOW = 2;

/// Minimum contrast for subpixel edge refinement
constexpr double SUBPIXEL_EDGE_MIN_CONTRAST = 5.0;

/// Maximum iterations for iterative refinement methods
constexpr int32_t SUBPIXEL_MAX_ITERATIONS = 10;

/// Convergence tolerance for iterative methods
constexpr double SUBPIXEL_CONVERGENCE_TOLERANCE = 1e-6;

/// Gaussian fitting sigma lower bound
constexpr double GAUSSIAN_FIT_MIN_SIGMA = 0.5;

/// Gaussian fitting sigma upper bound
constexpr double GAUSSIAN_FIT_MAX_SIGMA = 10.0;

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief 1D subpixel refinement methods
 */
enum class SubPixelMethod1D {
    Parabolic,      ///< Quadratic (parabolic) fitting [default, most common]
    Gaussian,       ///< Gaussian peak fitting (better for Gaussian-shaped peaks)
    Centroid,       ///< Center of gravity / centroid method (fast, symmetric)
    Quartic,        ///< 4th order polynomial fitting (5 points, higher accuracy)
    Linear          ///< Linear interpolation (for monotonic signals)
};

/**
 * @brief 2D subpixel refinement methods
 */
enum class SubPixelMethod2D {
    Quadratic,      ///< Quadratic surface (paraboloid) fitting [default]
    Taylor,         ///< Taylor expansion with gradient descent iteration
    Centroid,       ///< 2D center of gravity (fast)
    BiQuadratic,    ///< Bi-quadratic (4th order) surface fitting
    Gaussian2D      ///< 2D Gaussian fitting (expensive but accurate)
};

/**
 * @brief Edge subpixel refinement methods
 */
enum class EdgeSubPixelMethod {
    GradientInterp,     ///< Gradient interpolation (fast, robust)
    ZeroCrossing,       ///< Second derivative zero crossing (for step edges)
    ParabolicGradient,  ///< Parabolic fit on gradient profile [default]
    Moment              ///< First moment (centroid) of gradient
};

/**
 * @brief Corner subpixel refinement methods
 */
enum class CornerSubPixelMethod {
    GradientLeastSquares,   ///< Gradient-based least squares [default]
    TemplateMatching,       ///< Local template matching refinement
    QuadraticSurface        ///< Quadratic surface on corner response
};

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief 1D subpixel refinement result
 */
struct SubPixelResult1D {
    bool success = false;           ///< Whether refinement succeeded

    int32_t integerPosition = 0;    ///< Original integer position
    double subpixelPosition = 0.0;  ///< Refined subpixel position
    double offset = 0.0;            ///< Offset from integer position [-0.5, 0.5]

    double peakValue = 0.0;         ///< Interpolated value at subpixel position
    double curvature = 0.0;         ///< Local curvature (second derivative)
    double confidence = 0.0;        ///< Confidence score [0, 1]

    /// Get the subpixel position
    double Position() const { return subpixelPosition; }

    /// Check if result is valid and trustworthy
    bool IsValid(double minConfidence = 0.5) const {
        return success && confidence >= minConfidence &&
               std::abs(offset) <= SUBPIXEL_MAX_OFFSET;
    }
};

/**
 * @brief 2D subpixel refinement result
 */
struct SubPixelResult2D {
    bool success = false;           ///< Whether refinement succeeded

    int32_t integerX = 0;           ///< Original integer X
    int32_t integerY = 0;           ///< Original integer Y
    double subpixelX = 0.0;         ///< Refined subpixel X
    double subpixelY = 0.0;         ///< Refined subpixel Y
    double offsetX = 0.0;           ///< X offset from integer [-0.5, 0.5]
    double offsetY = 0.0;           ///< Y offset from integer [-0.5, 0.5]

    double peakValue = 0.0;         ///< Interpolated value at subpixel position
    double curvatureX = 0.0;        ///< Curvature in X direction
    double curvatureY = 0.0;        ///< Curvature in Y direction
    double curvatureMixed = 0.0;    ///< Mixed partial derivative (dxdy)
    double confidence = 0.0;        ///< Confidence score [0, 1]

    bool isSaddlePoint = false;     ///< True if detected as saddle point

    /// Get subpixel position as Point2d
    Point2d Position() const { return {subpixelX, subpixelY}; }

    /// Get offset as Point2d
    Point2d Offset() const { return {offsetX, offsetY}; }

    /// Check if result is valid and trustworthy
    bool IsValid(double minConfidence = 0.5) const {
        return success && !isSaddlePoint && confidence >= minConfidence &&
               std::abs(offsetX) <= SUBPIXEL_MAX_OFFSET &&
               std::abs(offsetY) <= SUBPIXEL_MAX_OFFSET;
    }
};

/**
 * @brief Edge subpixel refinement result
 */
struct SubPixelEdgeResult {
    bool success = false;           ///< Whether refinement succeeded

    double position = 0.0;          ///< Subpixel edge position (along profile)
    double gradient = 0.0;          ///< Gradient magnitude at edge
    double direction = 0.0;         ///< Gradient direction (radians)
    double amplitude = 0.0;         ///< Edge amplitude (intensity difference)
    double confidence = 0.0;        ///< Confidence score [0, 1]

    /// Check if result is valid
    bool IsValid(double minConfidence = 0.5) const {
        return success && confidence >= minConfidence;
    }
};

// =============================================================================
// 1D Subpixel Refinement Functions
// =============================================================================

/**
 * @brief Refine 1D extremum position using specified method
 *
 * @param signal Signal data array
 * @param size Signal length
 * @param index Integer position of extremum (local max/min)
 * @param method Refinement method
 * @param windowHalfSize Half window size for centroid method (default 2)
 * @return Subpixel refinement result
 *
 * @note For maximum detection, signal values should be positive peaks
 * @note For minimum detection, negate the signal first
 */
SubPixelResult1D RefineSubPixel1D(const double* signal, size_t size,
                                   int32_t index,
                                   SubPixelMethod1D method = SubPixelMethod1D::Parabolic,
                                   int32_t windowHalfSize = SUBPIXEL_CENTROID_HALF_WINDOW);

/**
 * @brief Refine 1D extremum using parabolic (quadratic) fit
 *
 * Fits a parabola to 3 points: (i-1, v0), (i, v1), (i+1, v2)
 * and finds the vertex.
 *
 * @param v0 Value at position i-1
 * @param v1 Value at position i (the peak)
 * @param v2 Value at position i+1
 * @return Subpixel offset from i (in range [-0.5, 0.5])
 *
 * Accuracy: < 0.02 px for symmetric peaks with reasonable SNR
 */
inline double RefineParabolic1D(double v0, double v1, double v2) {
    // Parabola: y = a*x^2 + b*x + c
    // At x=-1: v0 = a - b + c
    // At x=0:  v1 = c
    // At x=1:  v2 = a + b + c
    // => a = (v0 + v2)/2 - v1
    // => b = (v2 - v0)/2
    // Vertex at x = -b/(2a)

    double denom = 2.0 * (v0 - 2.0 * v1 + v2);
    if (std::abs(denom) < SUBPIXEL_MIN_CURVATURE) {
        return 0.0;  // Flat region, no refinement
    }

    double offset = (v0 - v2) / denom;

    // Clamp to prevent runaway extrapolation
    return std::clamp(offset, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
}

/**
 * @brief Compute interpolated value at parabolic peak
 *
 * @param v0 Value at position i-1
 * @param v1 Value at position i
 * @param v2 Value at position i+1
 * @param offset Offset from RefineParabolic1D
 * @return Interpolated peak value
 */
inline double ParabolicPeakValue(double v0, double v1, double v2, double offset) {
    double a = (v0 + v2) * 0.5 - v1;
    double b = (v2 - v0) * 0.5;
    return a * offset * offset + b * offset + v1;
}

/**
 * @brief Compute local curvature (second derivative)
 *
 * @param v0 Value at position i-1
 * @param v1 Value at position i
 * @param v2 Value at position i+1
 * @return Curvature (negative for maximum, positive for minimum)
 */
inline double ComputeCurvature1D(double v0, double v1, double v2) {
    return v0 - 2.0 * v1 + v2;
}

/**
 * @brief Refine 1D extremum using Gaussian peak fitting
 *
 * Assumes peak has Gaussian shape: y = A * exp(-x^2 / (2*sigma^2))
 * Fits log(y) as a parabola.
 *
 * @param signal Signal data
 * @param size Signal length
 * @param index Peak position
 * @return SubPixelResult1D with sigma stored in curvature field
 */
SubPixelResult1D RefineGaussian1D(const double* signal, size_t size, int32_t index);

/**
 * @brief Refine 1D extremum using centroid (center of gravity)
 *
 * Computes weighted centroid of values in window around peak.
 * Fast and robust for symmetric, well-separated peaks.
 *
 * @param signal Signal data
 * @param size Signal length
 * @param index Peak position
 * @param halfWindow Half window size (full window = 2*halfWindow + 1)
 * @param useAbsValues If true, use absolute values as weights
 * @return SubPixelResult1D
 */
SubPixelResult1D RefineCentroid1D(const double* signal, size_t size,
                                   int32_t index, int32_t halfWindow = 2,
                                   bool useAbsValues = false);

/**
 * @brief Refine 1D extremum using quartic (4th order) polynomial
 *
 * Fits 5 points to 4th order polynomial for higher accuracy.
 * Requires index to be at least 2 away from boundaries.
 *
 * @param signal Signal data
 * @param size Signal length
 * @param index Peak position
 * @return SubPixelResult1D
 */
SubPixelResult1D RefineQuartic1D(const double* signal, size_t size, int32_t index);

// =============================================================================
// 2D Subpixel Refinement Functions
// =============================================================================

/**
 * @brief Refine 2D extremum position using specified method
 *
 * @param data 2D data array (row-major)
 * @param width Image width
 * @param height Image height
 * @param x Integer X position
 * @param y Integer Y position
 * @param method Refinement method
 * @return SubPixelResult2D
 */
SubPixelResult2D RefineSubPixel2D(const float* data, int32_t width, int32_t height,
                                   int32_t x, int32_t y,
                                   SubPixelMethod2D method = SubPixelMethod2D::Quadratic);

/**
 * @brief Refine 2D extremum using double data type
 */
SubPixelResult2D RefineSubPixel2D(const double* data, int32_t width, int32_t height,
                                   int32_t x, int32_t y,
                                   SubPixelMethod2D method = SubPixelMethod2D::Quadratic);

/**
 * @brief Refine 2D extremum using quadratic (paraboloid) surface fit
 *
 * Fits z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f to 3x3 neighborhood.
 * Finds the extremum by solving the gradient equation.
 *
 * @tparam T Pixel data type (float or double)
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param x Center X
 * @param y Center Y
 * @return SubPixelResult2D
 *
 * Accuracy: < 0.05 px for smooth response surfaces
 */
template<typename T>
SubPixelResult2D RefineQuadratic2D(const T* data, int32_t width, int32_t height,
                                    int32_t x, int32_t y);

/**
 * @brief Refine 2D extremum using Taylor expansion iteration
 *
 * Iteratively refines position using gradient and Hessian.
 * More accurate for non-symmetric response surfaces.
 *
 * @tparam T Pixel data type
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param x Initial X
 * @param y Initial Y
 * @param maxIterations Maximum iterations
 * @param tolerance Convergence tolerance
 * @return SubPixelResult2D
 */
template<typename T>
SubPixelResult2D RefineTaylor2D(const T* data, int32_t width, int32_t height,
                                 int32_t x, int32_t y,
                                 int32_t maxIterations = SUBPIXEL_MAX_ITERATIONS,
                                 double tolerance = SUBPIXEL_CONVERGENCE_TOLERANCE);

/**
 * @brief Refine 2D extremum using centroid
 *
 * @tparam T Pixel data type
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param x Center X
 * @param y Center Y
 * @param halfWindow Half window size
 * @return SubPixelResult2D
 */
template<typename T>
SubPixelResult2D RefineCentroid2D(const T* data, int32_t width, int32_t height,
                                   int32_t x, int32_t y,
                                   int32_t halfWindow = SUBPIXEL_CENTROID_HALF_WINDOW);

/**
 * @brief Refine 2D corner position using gradient-based method
 *
 * Uses the structure tensor and gradient constraints to refine corner position.
 * Based on: sum_window (grad dot (p - corner)) = 0
 *
 * @tparam T Pixel data type
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param x Corner X
 * @param y Corner Y
 * @param windowSize Window size for gradient accumulation
 * @param maxIterations Maximum iterations
 * @return SubPixelResult2D
 */
template<typename T>
SubPixelResult2D RefineCorner2D(const T* data, int32_t width, int32_t height,
                                 int32_t x, int32_t y,
                                 int32_t windowSize = 5,
                                 int32_t maxIterations = SUBPIXEL_MAX_ITERATIONS);

// =============================================================================
// Edge Subpixel Refinement Functions
// =============================================================================

/**
 * @brief Refine edge position in 1D profile
 *
 * @param profile 1D intensity profile perpendicular to edge
 * @param size Profile length
 * @param edgeIndex Approximate integer edge position
 * @param method Refinement method
 * @return SubPixelEdgeResult
 */
SubPixelEdgeResult RefineEdgeSubPixel(const double* profile, size_t size,
                                       int32_t edgeIndex,
                                       EdgeSubPixelMethod method = EdgeSubPixelMethod::ParabolicGradient);

/**
 * @brief Refine edge using gradient interpolation
 *
 * Finds subpixel position where gradient equals a specific value
 * between two integer positions.
 *
 * @param g0 Gradient at position i
 * @param g1 Gradient at position i+1
 * @param targetGradient Target gradient value (typically peak gradient)
 * @return Offset from position i [0, 1]
 */
inline double RefineEdgeGradient(double g0, double g1, double targetGradient) {
    double denom = g1 - g0;
    if (std::abs(denom) < 1e-10) {
        return 0.5;  // Linear interpolation midpoint
    }
    double offset = (targetGradient - g0) / denom;
    return std::clamp(offset, 0.0, 1.0);
}

/**
 * @brief Refine edge using second derivative zero crossing
 *
 * Finds subpixel position where second derivative crosses zero.
 * Optimal for ideal step edges.
 *
 * @param profile Intensity profile
 * @param size Profile length
 * @param edgeIndex Approximate edge position
 * @return SubPixelEdgeResult
 */
SubPixelEdgeResult RefineEdgeZeroCrossing(const double* profile, size_t size,
                                           int32_t edgeIndex);

/**
 * @brief Refine edge using parabolic fit on gradient peak
 *
 * Fits parabola to gradient profile around peak and finds maximum.
 *
 * @param gradient Gradient profile
 * @param size Profile length
 * @param peakIndex Gradient peak position
 * @return SubPixelEdgeResult
 */
SubPixelEdgeResult RefineEdgeParabolic(const double* gradient, size_t size,
                                        int32_t peakIndex);

// =============================================================================
// Template Matching Subpixel Refinement
// =============================================================================

/**
 * @brief Refine template match position
 *
 * Refines the position of a template match result using the response surface.
 *
 * @param response Match response/score image
 * @param width Response image width
 * @param height Response image height
 * @param x Match X position
 * @param y Match Y position
 * @param method 2D refinement method
 * @return SubPixelResult2D
 */
SubPixelResult2D RefineMatchSubPixel(const float* response, int32_t width, int32_t height,
                                      int32_t x, int32_t y,
                                      SubPixelMethod2D method = SubPixelMethod2D::Quadratic);

/**
 * @brief Refine NCC (Normalized Cross-Correlation) match position
 *
 * Specialized refinement for NCC response surfaces which have known properties.
 *
 * @param nccResponse NCC response image (values in [-1, 1])
 * @param width Image width
 * @param height Image height
 * @param x Match X
 * @param y Match Y
 * @return SubPixelResult2D
 */
SubPixelResult2D RefineNCCSubPixel(const float* nccResponse, int32_t width, int32_t height,
                                    int32_t x, int32_t y);

// =============================================================================
// Angle Subpixel Refinement
// =============================================================================

/**
 * @brief Refine angle in angle-response lookup
 *
 * For shape matching where response is stored for discrete angles.
 *
 * @param responses Response values for consecutive angles
 * @param numAngles Number of angles
 * @param angleStep Angle step in radians
 * @param bestIndex Index of best response
 * @return Refined angle in radians
 */
double RefineAngleSubPixel(const double* responses, size_t numAngles,
                           double angleStep, int32_t bestIndex);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Compute confidence score for 1D subpixel result
 *
 * Based on curvature, SNR, and offset magnitude.
 *
 * @param curvature Second derivative at peak
 * @param peakValue Peak value
 * @param backgroundValue Estimated background level
 * @param offset Subpixel offset
 * @return Confidence score [0, 1]
 */
double ComputeSubPixelConfidence1D(double curvature, double peakValue,
                                    double backgroundValue, double offset);

/**
 * @brief Compute confidence score for 2D subpixel result
 *
 * @param curvatureX X curvature
 * @param curvatureY Y curvature
 * @param curvatureMixed Mixed curvature
 * @param peakValue Peak value
 * @param offsetX X offset
 * @param offsetY Y offset
 * @return Confidence score [0, 1]
 */
double ComputeSubPixelConfidence2D(double curvatureX, double curvatureY,
                                    double curvatureMixed, double peakValue,
                                    double offsetX, double offsetY);

/**
 * @brief Check if 2D Hessian indicates a true maximum
 *
 * @param hxx Second derivative in X
 * @param hyy Second derivative in Y
 * @param hxy Mixed second derivative
 * @return true if local maximum, false if minimum, saddle, or degenerate
 */
inline bool IsLocalMaximum2D(double hxx, double hyy, double hxy) {
    double det = hxx * hyy - hxy * hxy;
    return det > 0 && hxx < 0;  // Negative definite Hessian
}

/**
 * @brief Check if 2D Hessian indicates a saddle point
 *
 * @param hxx Second derivative in X
 * @param hyy Second derivative in Y
 * @param hxy Mixed second derivative
 * @return true if saddle point
 */
inline bool IsSaddlePoint2D(double hxx, double hyy, double hxy) {
    double det = hxx * hyy - hxy * hxy;
    return det < 0;  // Indefinite Hessian
}

/**
 * @brief Sample 3x3 neighborhood from 2D array
 *
 * @tparam T Data type
 * @param data 2D data array
 * @param width Image width
 * @param height Image height
 * @param x Center X
 * @param y Center Y
 * @param values Output 9 values [NW, N, NE, W, C, E, SW, S, SE]
 * @return true if all values are valid (not at boundary)
 */
template<typename T>
bool Sample3x3(const T* data, int32_t width, int32_t height,
               int32_t x, int32_t y, double values[9]);

/**
 * @brief Compute 2D gradient at a point using central differences
 *
 * @tparam T Data type
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param x Point X
 * @param y Point Y
 * @param dx Output gradient X
 * @param dy Output gradient Y
 */
template<typename T>
void ComputeGradient2D(const T* data, int32_t width, int32_t height,
                       int32_t x, int32_t y, double& dx, double& dy);

/**
 * @brief Compute 2D Hessian at a point
 *
 * @tparam T Data type
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param x Point X
 * @param y Point Y
 * @param hxx Output second derivative XX
 * @param hyy Output second derivative YY
 * @param hxy Output mixed derivative XY
 */
template<typename T>
void ComputeHessian2D(const T* data, int32_t width, int32_t height,
                      int32_t x, int32_t y, double& hxx, double& hyy, double& hxy);

// =============================================================================
// Template Implementations
// =============================================================================

template<typename T>
bool Sample3x3(const T* data, int32_t width, int32_t height,
               int32_t x, int32_t y, double values[9]) {
    // Check bounds
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        return false;
    }

    int32_t idx = y * width + x;
    values[0] = static_cast<double>(data[idx - width - 1]);  // NW
    values[1] = static_cast<double>(data[idx - width]);      // N
    values[2] = static_cast<double>(data[idx - width + 1]);  // NE
    values[3] = static_cast<double>(data[idx - 1]);          // W
    values[4] = static_cast<double>(data[idx]);              // C
    values[5] = static_cast<double>(data[idx + 1]);          // E
    values[6] = static_cast<double>(data[idx + width - 1]);  // SW
    values[7] = static_cast<double>(data[idx + width]);      // S
    values[8] = static_cast<double>(data[idx + width + 1]);  // SE

    return true;
}

template<typename T>
void ComputeGradient2D(const T* data, int32_t width, int32_t height,
                       int32_t x, int32_t y, double& dx, double& dy) {
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        dx = dy = 0.0;
        return;
    }

    int32_t idx = y * width + x;
    dx = 0.5 * (static_cast<double>(data[idx + 1]) - static_cast<double>(data[idx - 1]));
    dy = 0.5 * (static_cast<double>(data[idx + width]) - static_cast<double>(data[idx - width]));
}

template<typename T>
void ComputeHessian2D(const T* data, int32_t width, int32_t height,
                      int32_t x, int32_t y, double& hxx, double& hyy, double& hxy) {
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        hxx = hyy = hxy = 0.0;
        return;
    }

    int32_t idx = y * width + x;
    double c = static_cast<double>(data[idx]);
    double w = static_cast<double>(data[idx - 1]);
    double e = static_cast<double>(data[idx + 1]);
    double n = static_cast<double>(data[idx - width]);
    double s = static_cast<double>(data[idx + width]);
    double nw = static_cast<double>(data[idx - width - 1]);
    double ne = static_cast<double>(data[idx - width + 1]);
    double sw = static_cast<double>(data[idx + width - 1]);
    double se = static_cast<double>(data[idx + width + 1]);

    hxx = w - 2.0 * c + e;
    hyy = n - 2.0 * c + s;
    hxy = 0.25 * (se - sw - ne + nw);
}

template<typename T>
SubPixelResult2D RefineQuadratic2D(const T* data, int32_t width, int32_t height,
                                    int32_t x, int32_t y) {
    SubPixelResult2D result;
    result.integerX = x;
    result.integerY = y;
    result.subpixelX = static_cast<double>(x);
    result.subpixelY = static_cast<double>(y);

    // Sample 3x3 neighborhood
    double v[9];
    if (!Sample3x3(data, width, height, x, y, v)) {
        result.success = false;
        result.confidence = 0.0;
        return result;
    }

    // Compute gradient and Hessian using 3x3 values
    // v[0]=NW v[1]=N  v[2]=NE
    // v[3]=W  v[4]=C  v[5]=E
    // v[6]=SW v[7]=S  v[8]=SE

    double dx = 0.5 * (v[5] - v[3]);  // (E - W) / 2
    double dy = 0.5 * (v[7] - v[1]);  // (S - N) / 2
    double hxx = v[3] - 2.0 * v[4] + v[5];  // W - 2C + E
    double hyy = v[1] - 2.0 * v[4] + v[7];  // N - 2C + S
    double hxy = 0.25 * (v[8] - v[6] - v[2] + v[0]);  // (SE - SW - NE + NW) / 4

    result.curvatureX = hxx;
    result.curvatureY = hyy;
    result.curvatureMixed = hxy;

    // Check for saddle point
    double det = hxx * hyy - hxy * hxy;
    if (det <= 0) {
        result.success = true;
        result.isSaddlePoint = (det < 0);
        result.confidence = 0.2;  // Low confidence for non-maximum
        result.peakValue = v[4];
        return result;
    }

    // Solve 2x2 system: H * offset = -gradient
    // [hxx hxy] [dx'] = [-dx]
    // [hxy hyy] [dy']   [-dy]
    double invDet = 1.0 / det;
    double offsetX = (hxy * dy - hyy * dx) * invDet;
    double offsetY = (hxy * dx - hxx * dy) * invDet;

    // Clamp offsets
    offsetX = std::clamp(offsetX, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
    offsetY = std::clamp(offsetY, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);

    result.success = true;
    result.offsetX = offsetX;
    result.offsetY = offsetY;
    result.subpixelX = x + offsetX;
    result.subpixelY = y + offsetY;

    // Interpolate peak value
    result.peakValue = v[4] + 0.5 * (dx * offsetX + dy * offsetY);

    // Compute confidence based on curvature and offset
    result.confidence = ComputeSubPixelConfidence2D(hxx, hyy, hxy, v[4], offsetX, offsetY);

    return result;
}

template<typename T>
SubPixelResult2D RefineTaylor2D(const T* data, int32_t width, int32_t height,
                                 int32_t x, int32_t y,
                                 int32_t maxIterations, double tolerance) {
    SubPixelResult2D result;
    result.integerX = x;
    result.integerY = y;

    double currentX = static_cast<double>(x);
    double currentY = static_cast<double>(y);

    for (int32_t iter = 0; iter < maxIterations; ++iter) {
        // Get integer position for current estimate
        int32_t ix = static_cast<int32_t>(std::round(currentX));
        int32_t iy = static_cast<int32_t>(std::round(currentY));

        // Check bounds
        if (ix < 1 || ix >= width - 1 || iy < 1 || iy >= height - 1) {
            result.success = (iter > 0);
            result.subpixelX = currentX;
            result.subpixelY = currentY;
            return result;
        }

        // Compute gradient and Hessian at current position
        double dx, dy, hxx, hyy, hxy;
        ComputeGradient2D(data, width, height, ix, iy, dx, dy);
        ComputeHessian2D(data, width, height, ix, iy, hxx, hyy, hxy);

        // Newton step: delta = -H^(-1) * gradient
        double det = hxx * hyy - hxy * hxy;
        if (std::abs(det) < SUBPIXEL_MIN_CURVATURE) {
            break;  // Degenerate Hessian
        }

        double invDet = 1.0 / det;
        double deltaX = (hxy * dy - hyy * dx) * invDet;
        double deltaY = (hxy * dx - hxx * dy) * invDet;

        // Update position
        double newX = currentX + deltaX;
        double newY = currentY + deltaY;

        // Check convergence
        double change = std::sqrt(deltaX * deltaX + deltaY * deltaY);
        if (change < tolerance) {
            currentX = newX;
            currentY = newY;
            break;
        }

        currentX = newX;
        currentY = newY;

        // Check if we've moved too far from original position
        if (std::abs(currentX - x) > 1.0 || std::abs(currentY - y) > 1.0) {
            // Reset to quadratic result
            return RefineQuadratic2D(data, width, height, x, y);
        }
    }

    result.success = true;
    result.subpixelX = currentX;
    result.subpixelY = currentY;
    result.offsetX = currentX - x;
    result.offsetY = currentY - y;

    // Clamp final offsets
    if (std::abs(result.offsetX) > SUBPIXEL_MAX_OFFSET ||
        std::abs(result.offsetY) > SUBPIXEL_MAX_OFFSET) {
        return RefineQuadratic2D(data, width, height, x, y);
    }

    // Interpolate value at final position
    result.peakValue = InterpolateBilinear(data, width, height, currentX, currentY);

    // Get curvature at final integer position
    int32_t fx = static_cast<int32_t>(std::round(currentX));
    int32_t fy = static_cast<int32_t>(std::round(currentY));
    ComputeHessian2D(data, width, height, fx, fy,
                     result.curvatureX, result.curvatureY, result.curvatureMixed);

    double det = result.curvatureX * result.curvatureY -
                 result.curvatureMixed * result.curvatureMixed;
    result.isSaddlePoint = (det < 0);

    result.confidence = ComputeSubPixelConfidence2D(
        result.curvatureX, result.curvatureY, result.curvatureMixed,
        result.peakValue, result.offsetX, result.offsetY);

    return result;
}

template<typename T>
SubPixelResult2D RefineCentroid2D(const T* data, int32_t width, int32_t height,
                                   int32_t x, int32_t y, int32_t halfWindow) {
    SubPixelResult2D result;
    result.integerX = x;
    result.integerY = y;

    // Check bounds for window
    if (x - halfWindow < 0 || x + halfWindow >= width ||
        y - halfWindow < 0 || y + halfWindow >= height) {
        result.success = false;
        result.subpixelX = static_cast<double>(x);
        result.subpixelY = static_cast<double>(y);
        result.confidence = 0.0;
        return result;
    }

    // Compute weighted centroid
    double sumX = 0.0, sumY = 0.0, sumW = 0.0;
    double centerValue = static_cast<double>(data[y * width + x]);

    for (int32_t dy = -halfWindow; dy <= halfWindow; ++dy) {
        for (int32_t dx = -halfWindow; dx <= halfWindow; ++dx) {
            double value = static_cast<double>(data[(y + dy) * width + (x + dx)]);
            // Use value relative to center as weight
            double weight = std::max(0.0, value);
            sumX += weight * (x + dx);
            sumY += weight * (y + dy);
            sumW += weight;
        }
    }

    if (sumW < 1e-10) {
        result.success = false;
        result.subpixelX = static_cast<double>(x);
        result.subpixelY = static_cast<double>(y);
        result.confidence = 0.0;
        return result;
    }

    result.success = true;
    result.subpixelX = sumX / sumW;
    result.subpixelY = sumY / sumW;
    result.offsetX = result.subpixelX - x;
    result.offsetY = result.subpixelY - y;
    result.peakValue = centerValue;

    // Clamp offsets
    if (std::abs(result.offsetX) > SUBPIXEL_MAX_OFFSET ||
        std::abs(result.offsetY) > SUBPIXEL_MAX_OFFSET) {
        result.offsetX = std::clamp(result.offsetX, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
        result.offsetY = std::clamp(result.offsetY, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
        result.subpixelX = x + result.offsetX;
        result.subpixelY = y + result.offsetY;
    }

    result.confidence = 0.7;  // Centroid typically has moderate confidence

    return result;
}

template<typename T>
SubPixelResult2D RefineCorner2D(const T* data, int32_t width, int32_t height,
                                 int32_t x, int32_t y,
                                 int32_t windowSize, int32_t maxIterations) {
    SubPixelResult2D result;
    result.integerX = x;
    result.integerY = y;

    int32_t halfWindow = windowSize / 2;

    // Check bounds
    if (x - halfWindow - 1 < 0 || x + halfWindow + 1 >= width ||
        y - halfWindow - 1 < 0 || y + halfWindow + 1 >= height) {
        result.success = false;
        result.subpixelX = static_cast<double>(x);
        result.subpixelY = static_cast<double>(y);
        return result;
    }

    double currentX = static_cast<double>(x);
    double currentY = static_cast<double>(y);

    for (int32_t iter = 0; iter < maxIterations; ++iter) {
        // Build structure tensor and gradient sum
        // Solve: sum_window (grad * grad^T) * corner = sum_window (grad * grad^T * p)
        double a11 = 0, a12 = 0, a22 = 0;
        double b1 = 0, b2 = 0;

        int32_t cx = static_cast<int32_t>(std::round(currentX));
        int32_t cy = static_cast<int32_t>(std::round(currentY));

        for (int32_t dy = -halfWindow; dy <= halfWindow; ++dy) {
            for (int32_t dx = -halfWindow; dx <= halfWindow; ++dx) {
                int32_t px = cx + dx;
                int32_t py = cy + dy;

                // Compute gradient at this point
                double gx, gy;
                ComputeGradient2D(data, width, height, px, py, gx, gy);

                // Accumulate structure tensor
                a11 += gx * gx;
                a12 += gx * gy;
                a22 += gy * gy;

                // Accumulate gradient-weighted position
                b1 += gx * gx * px + gx * gy * py;
                b2 += gx * gy * px + gy * gy * py;
            }
        }

        // Solve 2x2 system
        double det = a11 * a22 - a12 * a12;
        if (std::abs(det) < SUBPIXEL_MIN_CURVATURE) {
            break;  // Singular matrix, stop iteration
        }

        double invDet = 1.0 / det;
        double newX = (a22 * b1 - a12 * b2) * invDet;
        double newY = (a11 * b2 - a12 * b1) * invDet;

        // Check convergence
        double change = std::sqrt((newX - currentX) * (newX - currentX) +
                                  (newY - currentY) * (newY - currentY));
        currentX = newX;
        currentY = newY;

        if (change < SUBPIXEL_CONVERGENCE_TOLERANCE) {
            break;
        }
    }

    result.success = true;
    result.subpixelX = currentX;
    result.subpixelY = currentY;
    result.offsetX = currentX - x;
    result.offsetY = currentY - y;

    // Clamp if moved too far
    if (std::abs(result.offsetX) > 1.0 || std::abs(result.offsetY) > 1.0) {
        result.offsetX = std::clamp(result.offsetX, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
        result.offsetY = std::clamp(result.offsetY, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
        result.subpixelX = x + result.offsetX;
        result.subpixelY = y + result.offsetY;
        result.confidence = 0.5;
    } else {
        result.confidence = 0.9;
    }

    return result;
}

} // namespace Qi::Vision::Internal
