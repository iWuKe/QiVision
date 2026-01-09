#pragma once

/**
 * @file ResponseMap.h
 * @brief Response Map for ultra-fast shape-based matching
 *
 * Based on LINE-MOD algorithm (Hinterstoisser et al., 2012) and
 * Halcon's shape-based matching implementation.
 *
 * Key features:
 * - 8-bin gradient orientation quantization (45° resolution)
 * - Spatial spreading for robustness to small translations
 * - O(1) lookup per model point instead of bilinear interpolation
 * - Memory layout optimized for SIMD access
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Internal/AnglePyramid.h>

#include <array>
#include <cstdint>
#include <vector>
#include <cmath>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Number of orientation bins (8 = 45 degree resolution)
constexpr int32_t RESPONSE_NUM_BINS = 8;

/// Angle covered by each bin in radians (45 degrees)
constexpr double RESPONSE_BIN_ANGLE = 2.0 * 3.14159265358979323846 / RESPONSE_NUM_BINS;

/// Default spreading radius in pixels
constexpr int32_t DEFAULT_SPREAD_RADIUS = 2;

/// Minimum gradient magnitude for response
constexpr float MIN_RESPONSE_MAGNITUDE = 5.0f;

/// Spreading grid size (T×T region for OR operation)
constexpr int32_t SPREAD_T = 4;

// =============================================================================
// Response Model Point (optimized for fast matching)
// =============================================================================

/**
 * @brief Model point optimized for Response Map matching
 *
 * Uses integer offsets and quantized angles for cache-friendly access.
 * Memory: 6 bytes per point
 */
struct ResponseModelPoint {
    int16_t offsetX;      ///< X offset from search position
    int16_t offsetY;      ///< Y offset from search position
    uint8_t angleBin;     ///< Quantized gradient direction [0, 7]
    uint8_t weight;       ///< Point weight [0, 255], 255 = 1.0

    ResponseModelPoint() = default;
    ResponseModelPoint(int16_t ox, int16_t oy, uint8_t bin, uint8_t w)
        : offsetX(ox), offsetY(oy), angleBin(bin), weight(w) {}
};

// =============================================================================
// Rotated Model (precomputed for each discrete angle)
// =============================================================================

/**
 * @brief Precomputed rotated model for a specific angle
 *
 * Contains all model points transformed to a specific rotation angle,
 * with precomputed bounding box for fast bounds checking.
 */
struct RotatedResponseModel {
    double angle = 0.0;                          ///< Rotation angle (radians)
    std::vector<ResponseModelPoint> points;      ///< Rotated points
    int16_t minX = 0, maxX = 0;                  ///< Bounding box X
    int16_t minY = 0, maxY = 0;                  ///< Bounding box Y

    /// Check if position is valid (model fits in image)
    bool IsValidPosition(int32_t posX, int32_t posY,
                         int32_t imageWidth, int32_t imageHeight) const {
        return posX + minX >= 0 && posX + maxX < imageWidth &&
               posY + minY >= 0 && posY + maxY < imageHeight;
    }
};

// =============================================================================
// Response Map Class
// =============================================================================

/**
 * @brief Response Map for ultra-fast gradient orientation matching
 *
 * Based on: Hinterstoisser et al., "Gradient Response Maps for Real-Time
 * Detection of Texture-Less Objects", IEEE TPAMI 2012.
 *
 * Algorithm:
 * 1. Quantize gradient orientations to 8 bins (45° each)
 * 2. For each bin, create a binary mask where gradient matches
 * 3. Apply spreading (morphological dilation) for robustness
 * 4. Store as 8 response maps for O(1) lookup
 *
 * Memory: 8 × width × height bytes per pyramid level
 *
 * @code
 * // Build from angle pyramid
 * ResponseMap responseMap;
 * responseMap.Build(anglePyramid);
 *
 * // Prepare rotated model
 * auto rotatedModel = ResponseMap::PrepareRotatedModel(modelPoints, angle);
 *
 * // Compute score at position
 * double score = responseMap.ComputeScore(rotatedModel, level, x, y);
 * @endcode
 */
class ResponseMap {
public:
    ResponseMap() = default;
    ~ResponseMap() = default;

    // Non-copyable, movable
    ResponseMap(const ResponseMap&) = delete;
    ResponseMap& operator=(const ResponseMap&) = delete;
    ResponseMap(ResponseMap&&) noexcept = default;
    ResponseMap& operator=(ResponseMap&&) noexcept = default;

    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Build response map from angle pyramid
     *
     * @param pyramid Source angle pyramid
     * @param spreadRadius Spreading radius (default 2 pixels)
     * @return true if successful
     *
     * Time: O(levels × 8 × width × height)
     */
    bool Build(const AnglePyramid& pyramid, int32_t spreadRadius = DEFAULT_SPREAD_RADIUS);

    /**
     * @brief Clear all data
     */
    void Clear();

    /**
     * @brief Check if valid
     */
    bool IsValid() const { return valid_; }

    // =========================================================================
    // Properties
    // =========================================================================

    int32_t NumLevels() const { return numLevels_; }
    int32_t GetWidth(int32_t level) const;
    int32_t GetHeight(int32_t level) const;
    int32_t GetStride(int32_t level) const;
    int32_t SpreadRadius() const { return spreadRadius_; }

    // =========================================================================
    // Score Computation
    // =========================================================================

    /**
     * @brief Compute match score at position using rotated model
     *
     * @param model Precomputed rotated model
     * @param level Pyramid level
     * @param posX Search X position at this level
     * @param posY Search Y position at this level
     * @param outCoverage Optional output: fraction of points with response
     * @return Match score in [0, 1]
     *
     * Time: O(N) where N = number of model points
     * Each point: 1 addition + 1 array lookup = O(1)
     */
    double ComputeScore(const RotatedResponseModel& model,
                        int32_t level,
                        int32_t posX, int32_t posY,
                        double* outCoverage = nullptr) const;

    /**
     * @brief Get response value at position for specific bin
     *
     * @param level Pyramid level
     * @param bin Orientation bin [0, 7]
     * @param x X coordinate
     * @param y Y coordinate
     * @return Response value [0, 255]
     */
    uint8_t GetResponse(int32_t level, int32_t bin, int32_t x, int32_t y) const;

    /**
     * @brief Get raw response data for a bin at a level
     */
    const uint8_t* GetResponseData(int32_t level, int32_t bin) const;

    // =========================================================================
    // Model Preparation (Static Methods)
    // =========================================================================

    /**
     * @brief Prepare rotated model for matching
     *
     * Converts continuous model points to quantized Response Map format.
     *
     * @param modelPoints Original model points with (x, y, angle, weight)
     * @param angle Rotation angle (radians)
     * @return Optimized rotated model
     */
    template<typename ModelPointType>
    static RotatedResponseModel PrepareRotatedModel(
        const std::vector<ModelPointType>& modelPoints,
        double angle);

    /**
     * @brief Prepare all rotated models for angle range
     *
     * @param modelPoints Original model points
     * @param angleStart Start angle (radians)
     * @param angleExtent Angle extent (radians)
     * @param angleStep Angle step (radians)
     * @return Vector of precomputed rotated models
     */
    template<typename ModelPointType>
    static std::vector<RotatedResponseModel> PrepareAllRotations(
        const std::vector<ModelPointType>& modelPoints,
        double angleStart, double angleExtent, double angleStep);

    // =========================================================================
    // Utility
    // =========================================================================

    /**
     * @brief Convert angle to 8-bin index
     * @param angle Angle in radians
     * @return Bin index [0, 7]
     */
    static int32_t AngleToBin(double angle);

    /**
     * @brief Get center angle of bin
     * @param bin Bin index [0, 7]
     * @return Center angle in radians
     */
    static double BinToAngle(int32_t bin);

    /**
     * @brief Get memory usage in bytes
     */
    size_t MemoryBytes() const;

private:
    // Response data storage
    struct LevelData {
        std::array<std::vector<uint8_t>, RESPONSE_NUM_BINS> bins;
        int32_t width = 0;
        int32_t height = 0;
        int32_t stride = 0;  // Row stride (64-byte aligned for SIMD)
    };

    std::vector<LevelData> levels_;
    int32_t numLevels_ = 0;
    int32_t spreadRadius_ = DEFAULT_SPREAD_RADIUS;
    bool valid_ = false;

    // Internal methods
    bool BuildLevel(const AnglePyramid& pyramid, int32_t levelIdx);
    void SpreadBinMask(const uint8_t* input, uint8_t* output,
                       int32_t width, int32_t height, int32_t stride);
    void SpreadBinMaskOR(uint8_t* data, int32_t width, int32_t height, int32_t stride);
};

// =============================================================================
// Inline Implementations
// =============================================================================

inline int32_t ResponseMap::AngleToBin(double angle)
{
    // Normalize to [0, 2*PI)
    constexpr double PI2 = 2.0 * 3.14159265358979323846;
    constexpr double INV_2PI = 1.0 / PI2;
    angle = angle - std::floor(angle * INV_2PI) * PI2;

    // Quantize to [0, 7]
    int32_t bin = static_cast<int32_t>(angle * RESPONSE_NUM_BINS * INV_2PI);
    if (bin < 0) bin = 0;
    if (bin >= RESPONSE_NUM_BINS) bin = RESPONSE_NUM_BINS - 1;
    return bin;
}

inline double ResponseMap::BinToAngle(int32_t bin)
{
    return (bin + 0.5) * RESPONSE_BIN_ANGLE;
}

inline int32_t ResponseMap::GetWidth(int32_t level) const
{
    if (level < 0 || level >= numLevels_) return 0;
    return levels_[level].width;
}

inline int32_t ResponseMap::GetHeight(int32_t level) const
{
    if (level < 0 || level >= numLevels_) return 0;
    return levels_[level].height;
}

inline int32_t ResponseMap::GetStride(int32_t level) const
{
    if (level < 0 || level >= numLevels_) return 0;
    return levels_[level].stride;
}

inline uint8_t ResponseMap::GetResponse(int32_t level, int32_t bin, int32_t x, int32_t y) const
{
    if (!valid_ || level < 0 || level >= numLevels_) return 0;
    if (bin < 0 || bin >= RESPONSE_NUM_BINS) return 0;

    const auto& levelData = levels_[level];
    if (x < 0 || x >= levelData.width || y < 0 || y >= levelData.height) return 0;

    return levelData.bins[bin][y * levelData.stride + x];
}

inline const uint8_t* ResponseMap::GetResponseData(int32_t level, int32_t bin) const
{
    if (!valid_ || level < 0 || level >= numLevels_) return nullptr;
    if (bin < 0 || bin >= RESPONSE_NUM_BINS) return nullptr;

    return levels_[level].bins[bin].data();
}

// =============================================================================
// Template Implementations
// =============================================================================

template<typename ModelPointType>
RotatedResponseModel ResponseMap::PrepareRotatedModel(
    const std::vector<ModelPointType>& modelPoints,
    double angle)
{
    RotatedResponseModel result;
    result.angle = angle;
    result.points.reserve(modelPoints.size());

    const double cosA = std::cos(angle);
    const double sinA = std::sin(angle);

    result.minX = result.minY = INT16_MAX;
    result.maxX = result.maxY = INT16_MIN;

    // Find max weight for proper scaling
    // Weights may be normalized (sum=1) or magnitude-based
    double maxWeight = 0.0;
    for (const auto& pt : modelPoints) {
        if (pt.weight > maxWeight) maxWeight = pt.weight;
    }
    // If weights are normalized (all very small), use equal weighting
    const bool useEqualWeight = (maxWeight < 0.01);

    for (const auto& pt : modelPoints) {
        // Rotate position
        double rx = cosA * pt.x - sinA * pt.y;
        double ry = sinA * pt.x + cosA * pt.y;

        ResponseModelPoint rpt;
        rpt.offsetX = static_cast<int16_t>(std::round(rx));
        rpt.offsetY = static_cast<int16_t>(std::round(ry));

        // Rotate and quantize angle
        double rotatedAngle = pt.angle + angle;
        rpt.angleBin = static_cast<uint8_t>(AngleToBin(rotatedAngle));

        // Quantize weight to [0, 255]
        // LINE-MOD style: equal weight when normalized, otherwise scale by max
        if (useEqualWeight) {
            rpt.weight = 255;
        } else {
            rpt.weight = static_cast<uint8_t>(std::min(255.0, (pt.weight / maxWeight) * 255.0));
            if (rpt.weight < 1) rpt.weight = 1;
        }

        result.points.push_back(rpt);

        // Update bounding box
        if (rpt.offsetX < result.minX) result.minX = rpt.offsetX;
        if (rpt.offsetX > result.maxX) result.maxX = rpt.offsetX;
        if (rpt.offsetY < result.minY) result.minY = rpt.offsetY;
        if (rpt.offsetY > result.maxY) result.maxY = rpt.offsetY;
    }

    return result;
}

template<typename ModelPointType>
std::vector<RotatedResponseModel> ResponseMap::PrepareAllRotations(
    const std::vector<ModelPointType>& modelPoints,
    double angleStart, double angleExtent, double angleStep)
{
    if (angleStep <= 0) {
        angleStep = 0.05;  // Default ~3 degrees
    }

    const int32_t numAngles = static_cast<int32_t>(
        std::ceil(angleExtent / angleStep)) + 1;

    std::vector<RotatedResponseModel> result;
    result.reserve(numAngles);

    for (int32_t i = 0; i < numAngles; ++i) {
        double angle = angleStart + i * angleStep;
        result.push_back(PrepareRotatedModel(modelPoints, angle));
    }

    return result;
}

} // namespace Qi::Vision::Internal
