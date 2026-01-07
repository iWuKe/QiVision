#pragma once

/**
 * @file Profiler.h
 * @brief 1D profile extraction for Caliper measurement
 *
 * Provides profile extraction along various geometric paths:
 * - Line profiles (single and parallel)
 * - Rectangle profiles (averaged perpendicular lines)
 * - Arc profiles (along circular arcs)
 * - Annular profiles (along concentric circles)
 *
 * Used by:
 * - Measure/Caliper: Rectangle and arc calipers
 * - Measure/Metrology: Combined measurements
 * - Edge detection: Profile-based edge finding
 *
 * Features:
 * - Subpixel interpolation (bilinear/bicubic)
 * - Multi-line averaging for noise reduction
 * - Profile statistics and projections
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Internal/Interpolate.h>

#include <algorithm>
#include <cstdint>
#include <vector>
#include <cmath>

namespace Qi::Vision::Internal {

// ============================================================================
// Constants
// ============================================================================

/// Default number of samples per pixel for profile extraction
constexpr double DEFAULT_SAMPLES_PER_PIXEL = 1.0;

/// Minimum profile length
constexpr int32_t MIN_PROFILE_LENGTH = 2;

/// Maximum number of averaging lines
constexpr int32_t MAX_AVERAGING_LINES = 256;

// ============================================================================
// Enumerations
// ============================================================================

/**
 * @brief Profile extraction method
 */
enum class ProfileMethod {
    Single,         ///< Single line profile
    Average,        ///< Average of multiple parallel lines
    Maximum,        ///< Maximum along perpendicular
    Minimum,        ///< Minimum along perpendicular
    Median          ///< Median along perpendicular
};

/**
 * @brief Profile normalization mode
 */
enum class ProfileNormalize {
    None,           ///< No normalization
    MinMax,         ///< Normalize to [0, 1]
    ZScore,         ///< Normalize to mean=0, std=1
    Sum             ///< Normalize sum to 1
};

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief 1D profile data with metadata
 */
struct Profile1D {
    std::vector<double> data;       ///< Profile values
    double startX = 0;              ///< Start X coordinate in image
    double startY = 0;              ///< Start Y coordinate in image
    double endX = 0;                ///< End X coordinate in image
    double endY = 0;                ///< End Y coordinate in image
    double stepSize = 1.0;          ///< Step size between samples
    double angle = 0;               ///< Profile direction angle (radians)

    /// Number of samples
    size_t Size() const { return data.size(); }

    /// Check if empty
    bool Empty() const { return data.empty(); }

    /// Profile length in pixels
    double Length() const {
        double dx = endX - startX;
        double dy = endY - startY;
        return std::sqrt(dx * dx + dy * dy);
    }

    /// Get value at index
    double At(size_t idx) const {
        if (idx >= data.size()) return 0.0;
        return data[idx];
    }

    /// Get value with bounds checking
    double AtSafe(int32_t idx) const {
        if (idx < 0) idx = 0;
        if (idx >= static_cast<int32_t>(data.size())) {
            idx = static_cast<int32_t>(data.size()) - 1;
        }
        return data[idx];
    }

    /// Convert profile index to image coordinates
    void IndexToCoord(double index, double& x, double& y) const {
        if (data.empty()) {
            x = startX;
            y = startY;
            return;
        }
        double t = index / (data.size() - 1);
        x = startX + t * (endX - startX);
        y = startY + t * (endY - startY);
    }

    /// Convert image coordinates to profile index
    double CoordToIndex(double x, double y) const {
        double dx = endX - startX;
        double dy = endY - startY;
        double len2 = dx * dx + dy * dy;
        if (len2 < 1e-10) return 0.0;

        double t = ((x - startX) * dx + (y - startY) * dy) / len2;
        return t * (data.size() - 1);
    }
};

/**
 * @brief Profile statistics
 */
struct ProfileStats {
    double min = 0;         ///< Minimum value
    double max = 0;         ///< Maximum value
    double mean = 0;        ///< Mean value
    double stddev = 0;      ///< Standard deviation
    double sum = 0;         ///< Sum of values
    size_t minIdx = 0;      ///< Index of minimum
    size_t maxIdx = 0;      ///< Index of maximum
    size_t count = 0;       ///< Number of samples
};

/**
 * @brief Rectangle profile parameters
 */
struct RectProfileParams {
    double centerX = 0;         ///< Rectangle center X
    double centerY = 0;         ///< Rectangle center Y
    double length = 100;        ///< Profile length (along profile direction)
    double width = 10;          ///< Rectangle width (perpendicular extent)
    double angle = 0;           ///< Profile direction angle (radians)
    int32_t numLines = 1;       ///< Number of averaging lines
    ProfileMethod method = ProfileMethod::Average;  ///< Aggregation method
    InterpolationMethod interp = InterpolationMethod::Bilinear;
    double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL;

    /// Create params for a line from (x0,y0) to (x1,y1)
    static RectProfileParams FromLine(double x0, double y0, double x1, double y1,
                                       double width = 1.0, int32_t numLines = 1) {
        RectProfileParams p;
        p.centerX = (x0 + x1) / 2.0;
        p.centerY = (y0 + y1) / 2.0;
        double dx = x1 - x0;
        double dy = y1 - y0;
        p.length = std::sqrt(dx * dx + dy * dy);
        p.angle = std::atan2(dy, dx);
        p.width = width;
        p.numLines = numLines;
        return p;
    }

    /// Create params from center, length, and angle
    static RectProfileParams FromCenter(double cx, double cy, double length,
                                         double angle, double width = 1.0,
                                         int32_t numLines = 1) {
        RectProfileParams p;
        p.centerX = cx;
        p.centerY = cy;
        p.length = length;
        p.angle = angle;
        p.width = width;
        p.numLines = numLines;
        return p;
    }
};

/**
 * @brief Arc profile parameters
 */
struct ArcProfileParams {
    double centerX = 0;         ///< Arc center X
    double centerY = 0;         ///< Arc center Y
    double radius = 50;         ///< Arc radius
    double startAngle = 0;      ///< Start angle (radians)
    double endAngle = M_PI;     ///< End angle (radians)
    double width = 10;          ///< Radial width (for averaging)
    int32_t numLines = 1;       ///< Number of radial averaging lines
    ProfileMethod method = ProfileMethod::Average;
    InterpolationMethod interp = InterpolationMethod::Bilinear;
    double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL;

    /// Arc sweep angle
    double SweepAngle() const { return endAngle - startAngle; }

    /// Arc length at center radius
    double ArcLength() const { return std::abs(SweepAngle()) * radius; }

    /// Create full circle params
    static ArcProfileParams FullCircle(double cx, double cy, double radius,
                                        double width = 1.0, int32_t numLines = 1) {
        ArcProfileParams p;
        p.centerX = cx;
        p.centerY = cy;
        p.radius = radius;
        p.startAngle = 0;
        p.endAngle = 2.0 * M_PI;
        p.width = width;
        p.numLines = numLines;
        return p;
    }

    /// Create arc params
    static ArcProfileParams FromArc(double cx, double cy, double radius,
                                     double startAngle, double endAngle,
                                     double width = 1.0, int32_t numLines = 1) {
        ArcProfileParams p;
        p.centerX = cx;
        p.centerY = cy;
        p.radius = radius;
        p.startAngle = startAngle;
        p.endAngle = endAngle;
        p.width = width;
        p.numLines = numLines;
        return p;
    }
};

/**
 * @brief Annular (concentric circles) profile parameters
 */
struct AnnularProfileParams {
    double centerX = 0;         ///< Center X
    double centerY = 0;         ///< Center Y
    double innerRadius = 20;    ///< Inner radius
    double outerRadius = 50;    ///< Outer radius
    double angle = 0;           ///< Direction angle for profile (radians)
    double angularWidth = 0.1;  ///< Angular width for averaging (radians)
    int32_t numLines = 1;       ///< Number of angular averaging lines
    ProfileMethod method = ProfileMethod::Average;
    InterpolationMethod interp = InterpolationMethod::Bilinear;
    double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL;

    /// Radial extent
    double RadialExtent() const { return outerRadius - innerRadius; }

    /// Create annular params
    static AnnularProfileParams FromRadii(double cx, double cy,
                                           double innerR, double outerR,
                                           double angle = 0,
                                           double angularWidth = 0.1,
                                           int32_t numLines = 1) {
        AnnularProfileParams p;
        p.centerX = cx;
        p.centerY = cy;
        p.innerRadius = innerR;
        p.outerRadius = outerR;
        p.angle = angle;
        p.angularWidth = angularWidth;
        p.numLines = numLines;
        return p;
    }
};

// ============================================================================
// Line Profile Extraction
// ============================================================================

/**
 * @brief Extract profile along a line
 *
 * @param image Input image
 * @param x0, y0 Start point
 * @param x1, y1 End point
 * @param numSamples Number of samples (0 = auto)
 * @param method Interpolation method
 * @return Profile data
 */
Profile1D ExtractLineProfile(const QImage& image,
                              double x0, double y0,
                              double x1, double y1,
                              size_t numSamples = 0,
                              InterpolationMethod method = InterpolationMethod::Bilinear);

/**
 * @brief Extract profile from raw data
 */
template<typename T>
Profile1D ExtractLineProfile(const T* data, int32_t width, int32_t height,
                              double x0, double y0,
                              double x1, double y1,
                              size_t numSamples = 0,
                              InterpolationMethod method = InterpolationMethod::Bilinear);

/**
 * @brief Extract multiple parallel line profiles
 *
 * @param image Input image
 * @param x0, y0 Start point (center line)
 * @param x1, y1 End point (center line)
 * @param spacing Spacing between parallel lines
 * @param numLines Number of parallel lines
 * @param numSamples Samples per line
 * @param method Interpolation method
 * @return Vector of profiles
 */
std::vector<Profile1D> ExtractParallelProfiles(
    const QImage& image,
    double x0, double y0,
    double x1, double y1,
    double spacing,
    int32_t numLines,
    size_t numSamples = 0,
    InterpolationMethod method = InterpolationMethod::Bilinear);

// ============================================================================
// Rectangle Profile Extraction
// ============================================================================

/**
 * @brief Extract profile from rectangle region
 *
 * Extracts multiple perpendicular lines and combines them.
 *
 * @param image Input image
 * @param params Rectangle profile parameters
 * @return Combined profile
 */
Profile1D ExtractRectProfile(const QImage& image, const RectProfileParams& params);

/**
 * @brief Extract profile from raw data
 */
template<typename T>
Profile1D ExtractRectProfile(const T* data, int32_t width, int32_t height,
                              const RectProfileParams& params);

// ============================================================================
// Arc Profile Extraction
// ============================================================================

/**
 * @brief Extract profile along an arc
 *
 * @param image Input image
 * @param params Arc profile parameters
 * @return Arc profile (angular samples)
 */
Profile1D ExtractArcProfile(const QImage& image, const ArcProfileParams& params);

/**
 * @brief Extract profile from raw data
 */
template<typename T>
Profile1D ExtractArcProfile(const T* data, int32_t width, int32_t height,
                             const ArcProfileParams& params);

// ============================================================================
// Annular (Radial) Profile Extraction
// ============================================================================

/**
 * @brief Extract radial profile from annular region
 *
 * Samples from inner to outer radius at specified angle.
 *
 * @param image Input image
 * @param params Annular profile parameters
 * @return Radial profile
 */
Profile1D ExtractAnnularProfile(const QImage& image, const AnnularProfileParams& params);

/**
 * @brief Extract profile from raw data
 */
template<typename T>
Profile1D ExtractAnnularProfile(const T* data, int32_t width, int32_t height,
                                 const AnnularProfileParams& params);

// ============================================================================
// Profile Operations
// ============================================================================

/**
 * @brief Compute profile statistics
 */
ProfileStats ComputeProfileStats(const Profile1D& profile);

/**
 * @brief Compute profile statistics from raw data
 */
ProfileStats ComputeProfileStats(const double* data, size_t length);

/**
 * @brief Normalize profile values
 */
void NormalizeProfile(Profile1D& profile, ProfileNormalize mode);

/**
 * @brief Smooth profile with Gaussian kernel
 */
void SmoothProfile(Profile1D& profile, double sigma);

/**
 * @brief Compute profile gradient (first derivative)
 */
Profile1D ComputeProfileGradient(const Profile1D& profile, double sigma = 0.0);

/**
 * @brief Compute profile second derivative
 */
Profile1D ComputeProfileSecondDerivative(const Profile1D& profile, double sigma = 0.0);

/**
 * @brief Resample profile to new number of samples
 */
Profile1D ResampleProfile(const Profile1D& profile, size_t newSize);

/**
 * @brief Combine multiple profiles into one
 *
 * @param profiles Vector of profiles
 * @param method Combination method
 * @return Combined profile
 */
Profile1D CombineProfiles(const std::vector<Profile1D>& profiles, ProfileMethod method);

// ============================================================================
// Profile Projection (for 2D regions)
// ============================================================================

/**
 * @brief Project image region along one axis
 *
 * @param image Input image
 * @param rect Region of interest
 * @param horizontal If true, project horizontally; otherwise vertically
 * @param method Projection method
 * @return 1D projection profile
 */
Profile1D ProjectRegion(const QImage& image, const Rect2i& rect,
                         bool horizontal, ProfileMethod method = ProfileMethod::Average);

/**
 * @brief Project rotated rectangle region
 *
 * @param image Input image
 * @param center Rectangle center
 * @param width Rectangle width
 * @param height Rectangle height
 * @param angle Rotation angle
 * @param alongWidth If true, project along width; otherwise along height
 * @param method Projection method
 * @return 1D projection profile
 */
Profile1D ProjectRotatedRect(const QImage& image,
                              Point2d center, double width, double height,
                              double angle, bool alongWidth,
                              ProfileMethod method = ProfileMethod::Average);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Compute sample positions along a line
 *
 * @param x0, y0 Start point
 * @param x1, y1 End point
 * @param numSamples Number of samples
 * @param[out] xCoords X coordinates of samples
 * @param[out] yCoords Y coordinates of samples
 */
void ComputeLineSamples(double x0, double y0, double x1, double y1,
                        size_t numSamples,
                        std::vector<double>& xCoords,
                        std::vector<double>& yCoords);

/**
 * @brief Compute sample positions along an arc
 *
 * @param cx, cy Arc center
 * @param radius Arc radius
 * @param startAngle Start angle
 * @param endAngle End angle
 * @param numSamples Number of samples
 * @param[out] xCoords X coordinates
 * @param[out] yCoords Y coordinates
 */
void ComputeArcSamples(double cx, double cy, double radius,
                       double startAngle, double endAngle,
                       size_t numSamples,
                       std::vector<double>& xCoords,
                       std::vector<double>& yCoords);

/**
 * @brief Check if point is inside image bounds
 */
inline bool IsInsideImage(double x, double y, int32_t width, int32_t height) {
    return x >= 0 && x < width && y >= 0 && y < height;
}

/**
 * @brief Compute perpendicular offset points
 *
 * @param x, y Center point
 * @param angle Direction angle
 * @param offset Perpendicular offset distance
 * @param[out] px, py Offset point coordinates
 */
inline void ComputePerpendicularPoint(double x, double y, double angle,
                                       double offset, double& px, double& py) {
    double perpAngle = angle - M_PI / 2.0;
    px = x + offset * std::cos(perpAngle);
    py = y + offset * std::sin(perpAngle);
}

// ============================================================================
// Template Implementations
// ============================================================================

template<typename T>
Profile1D ExtractLineProfile(const T* data, int32_t width, int32_t height,
                              double x0, double y0,
                              double x1, double y1,
                              size_t numSamples,
                              InterpolationMethod method) {
    Profile1D profile;
    profile.startX = x0;
    profile.startY = y0;
    profile.endX = x1;
    profile.endY = y1;

    // Compute line length and direction
    double dx = x1 - x0;
    double dy = y1 - y0;
    double length = std::sqrt(dx * dx + dy * dy);
    profile.angle = std::atan2(dy, dx);

    // Auto-determine sample count
    if (numSamples == 0) {
        numSamples = static_cast<size_t>(std::ceil(length)) + 1;
    }
    if (numSamples < MIN_PROFILE_LENGTH) {
        numSamples = MIN_PROFILE_LENGTH;
    }

    profile.data.resize(numSamples);
    profile.stepSize = (numSamples > 1) ? length / (numSamples - 1) : 0.0;

    if (numSamples == 1) {
        profile.data[0] = Interpolate(data, width, height, x0, y0, method);
        return profile;
    }

    // Sample along line
    double stepX = dx / (numSamples - 1);
    double stepY = dy / (numSamples - 1);

    for (size_t i = 0; i < numSamples; ++i) {
        double x = x0 + i * stepX;
        double y = y0 + i * stepY;
        profile.data[i] = Interpolate(data, width, height, x, y, method);
    }

    return profile;
}

template<typename T>
Profile1D ExtractRectProfile(const T* data, int32_t width, int32_t height,
                              const RectProfileParams& params) {
    // Compute start and end points of center line
    double halfLen = params.length / 2.0;
    double cosA = std::cos(params.angle);
    double sinA = std::sin(params.angle);

    double x0 = params.centerX - halfLen * cosA;
    double y0 = params.centerY - halfLen * sinA;
    double x1 = params.centerX + halfLen * cosA;
    double y1 = params.centerY + halfLen * sinA;

    // Determine number of samples
    size_t numSamples = static_cast<size_t>(
        std::ceil(params.length * params.samplesPerPixel)) + 1;
    if (numSamples < MIN_PROFILE_LENGTH) numSamples = MIN_PROFILE_LENGTH;

    // Single line case
    if (params.numLines <= 1 || params.width <= 0) {
        return ExtractLineProfile(data, width, height, x0, y0, x1, y1,
                                   numSamples, params.interp);
    }

    // Multiple lines - compute perpendicular offsets
    int32_t nLines = std::min(params.numLines, MAX_AVERAGING_LINES);
    double halfWidth = params.width / 2.0;
    double lineSpacing = (nLines > 1) ? params.width / (nLines - 1) : 0.0;

    std::vector<Profile1D> profiles;
    profiles.reserve(nLines);

    for (int32_t i = 0; i < nLines; ++i) {
        double offset = (nLines > 1) ? (-halfWidth + i * lineSpacing) : 0.0;

        double ox0, oy0, ox1, oy1;
        ComputePerpendicularPoint(x0, y0, params.angle, offset, ox0, oy0);
        ComputePerpendicularPoint(x1, y1, params.angle, offset, ox1, oy1);

        profiles.push_back(ExtractLineProfile(data, width, height,
                                               ox0, oy0, ox1, oy1,
                                               numSamples, params.interp));
    }

    // Combine profiles
    Profile1D result = CombineProfiles(profiles, params.method);
    result.startX = x0;
    result.startY = y0;
    result.endX = x1;
    result.endY = y1;
    result.angle = params.angle;

    return result;
}

template<typename T>
Profile1D ExtractArcProfile(const T* data, int32_t width, int32_t height,
                             const ArcProfileParams& params) {
    Profile1D profile;

    // Compute arc length and number of samples
    double arcLength = params.ArcLength();
    size_t numSamples = static_cast<size_t>(
        std::ceil(arcLength * params.samplesPerPixel)) + 1;
    if (numSamples < MIN_PROFILE_LENGTH) numSamples = MIN_PROFILE_LENGTH;

    profile.data.resize(numSamples);
    profile.stepSize = arcLength / (numSamples - 1);

    // Store arc endpoints
    double startX = params.centerX + params.radius * std::cos(params.startAngle);
    double startY = params.centerY + params.radius * std::sin(params.startAngle);
    double endX = params.centerX + params.radius * std::cos(params.endAngle);
    double endY = params.centerY + params.radius * std::sin(params.endAngle);

    profile.startX = startX;
    profile.startY = startY;
    profile.endX = endX;
    profile.endY = endY;
    profile.angle = params.startAngle;

    // Single radius case
    if (params.numLines <= 1 || params.width <= 0) {
        double angleStep = params.SweepAngle() / (numSamples - 1);

        for (size_t i = 0; i < numSamples; ++i) {
            double angle = params.startAngle + i * angleStep;
            double x = params.centerX + params.radius * std::cos(angle);
            double y = params.centerY + params.radius * std::sin(angle);
            profile.data[i] = Interpolate(data, width, height, x, y, params.interp);
        }
        return profile;
    }

    // Multiple radii - average along radial direction
    int32_t nLines = std::min(params.numLines, MAX_AVERAGING_LINES);
    double halfWidth = params.width / 2.0;
    double radiusStep = (nLines > 1) ? params.width / (nLines - 1) : 0.0;

    double angleStep = params.SweepAngle() / (numSamples - 1);

    for (size_t i = 0; i < numSamples; ++i) {
        double angle = params.startAngle + i * angleStep;
        double cosAngle = std::cos(angle);
        double sinAngle = std::sin(angle);

        std::vector<double> values;
        values.reserve(nLines);

        for (int32_t j = 0; j < nLines; ++j) {
            double r = params.radius + (-halfWidth + j * radiusStep);
            double x = params.centerX + r * cosAngle;
            double y = params.centerY + r * sinAngle;
            values.push_back(Interpolate(data, width, height, x, y, params.interp));
        }

        // Combine values based on method
        switch (params.method) {
            case ProfileMethod::Average: {
                double sum = 0;
                for (double v : values) sum += v;
                profile.data[i] = sum / values.size();
                break;
            }
            case ProfileMethod::Maximum: {
                double maxVal = values[0];
                for (double v : values) if (v > maxVal) maxVal = v;
                profile.data[i] = maxVal;
                break;
            }
            case ProfileMethod::Minimum: {
                double minVal = values[0];
                for (double v : values) if (v < minVal) minVal = v;
                profile.data[i] = minVal;
                break;
            }
            case ProfileMethod::Median: {
                std::sort(values.begin(), values.end());
                size_t mid = values.size() / 2;
                profile.data[i] = (values.size() % 2 == 0)
                    ? (values[mid - 1] + values[mid]) / 2.0
                    : values[mid];
                break;
            }
            default:
                profile.data[i] = values[0];
                break;
        }
    }

    return profile;
}

template<typename T>
Profile1D ExtractAnnularProfile(const T* data, int32_t width, int32_t height,
                                 const AnnularProfileParams& params) {
    Profile1D profile;

    // Radial extent
    double radialExtent = params.RadialExtent();
    if (radialExtent <= 0) return profile;

    size_t numSamples = static_cast<size_t>(
        std::ceil(radialExtent * params.samplesPerPixel)) + 1;
    if (numSamples < MIN_PROFILE_LENGTH) numSamples = MIN_PROFILE_LENGTH;

    profile.data.resize(numSamples);
    profile.stepSize = radialExtent / (numSamples - 1);

    // Store endpoints (along the radial direction)
    double cosA = std::cos(params.angle);
    double sinA = std::sin(params.angle);

    profile.startX = params.centerX + params.innerRadius * cosA;
    profile.startY = params.centerY + params.innerRadius * sinA;
    profile.endX = params.centerX + params.outerRadius * cosA;
    profile.endY = params.centerY + params.outerRadius * sinA;
    profile.angle = params.angle;

    // Single angle case
    if (params.numLines <= 1 || params.angularWidth <= 0) {
        double radiusStep = radialExtent / (numSamples - 1);

        for (size_t i = 0; i < numSamples; ++i) {
            double r = params.innerRadius + i * radiusStep;
            double x = params.centerX + r * cosA;
            double y = params.centerY + r * sinA;
            profile.data[i] = Interpolate(data, width, height, x, y, params.interp);
        }
        return profile;
    }

    // Multiple angles - average along angular direction
    int32_t nLines = std::min(params.numLines, MAX_AVERAGING_LINES);
    double halfAngWidth = params.angularWidth / 2.0;
    double angleStep = (nLines > 1) ? params.angularWidth / (nLines - 1) : 0.0;

    double radiusStep = radialExtent / (numSamples - 1);

    for (size_t i = 0; i < numSamples; ++i) {
        double r = params.innerRadius + i * radiusStep;

        std::vector<double> values;
        values.reserve(nLines);

        for (int32_t j = 0; j < nLines; ++j) {
            double angle = params.angle + (-halfAngWidth + j * angleStep);
            double x = params.centerX + r * std::cos(angle);
            double y = params.centerY + r * std::sin(angle);
            values.push_back(Interpolate(data, width, height, x, y, params.interp));
        }

        // Combine values
        switch (params.method) {
            case ProfileMethod::Average: {
                double sum = 0;
                for (double v : values) sum += v;
                profile.data[i] = sum / values.size();
                break;
            }
            case ProfileMethod::Maximum: {
                double maxVal = values[0];
                for (double v : values) if (v > maxVal) maxVal = v;
                profile.data[i] = maxVal;
                break;
            }
            case ProfileMethod::Minimum: {
                double minVal = values[0];
                for (double v : values) if (v < minVal) minVal = v;
                profile.data[i] = minVal;
                break;
            }
            case ProfileMethod::Median: {
                std::sort(values.begin(), values.end());
                size_t mid = values.size() / 2;
                profile.data[i] = (values.size() % 2 == 0)
                    ? (values[mid - 1] + values[mid]) / 2.0
                    : values[mid];
                break;
            }
            default:
                profile.data[i] = values[0];
                break;
        }
    }

    return profile;
}

} // namespace Qi::Vision::Internal
