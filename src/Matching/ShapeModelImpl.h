/**
 * @file ShapeModelImpl.h
 * @brief Internal implementation structures for ShapeModel
 *
 * This file contains:
 * - LevelModel: Model data for a single pyramid level
 * - SearchAngleData: Precomputed search angle data
 * - FastCosTable: Fast cosine lookup table
 * - ShapeModelImpl: Implementation class
 */

#pragma once

#include <QiVision/Matching/ShapeModel.h>
#include <QiVision/Matching/MatchTypes.h>
#include <QiVision/Internal/AnglePyramid.h>
#include <QiVision/Internal/EdgesSubPix.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>

#include <vector>
#include <set>
#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD intrinsics
#if defined(__AVX2__)
#include <immintrin.h>
#define HAVE_AVX2 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define HAVE_SSE4 1
#endif

namespace Qi::Vision::Matching {
namespace Internal {

// Import Internal types
using Qi::Vision::Internal::AnglePyramid;
using Qi::Vision::Internal::AnglePyramidParams;
using Qi::Vision::Internal::PyramidLevelData;
using Qi::Vision::Internal::EdgePoint;

// =============================================================================
// LevelCreateData: Per-level creation metadata (MinAreaRect alignment)
// =============================================================================

struct LevelCreateData {
    double alignmentAngle = 0.0;   ///< Alignment angle from MinAreaRect (radians)
    double bboxMinX = 0, bboxMaxX = 0;  ///< Rotated AABB bounds
    double bboxMinY = 0, bboxMaxY = 0;
    double maxRadius = 0.0;        ///< Max distance from center to edge point
};

// =============================================================================
// LevelModel: Model data for a single pyramid level
// =============================================================================

/**
 * @brief Model data for a single pyramid level
 *
 * Halcon-style dual block storage:
 * - Block 1 (points/soaX,Y...): Subpixel edge points for high-precision matching
 * - Block 2 (gridPoints/gridSoaX,Y...): Integer grid samples for fast coarse search
 *
 * Usage strategy:
 * - Coarse search (top pyramid levels): Use gridPoints (faster, integer positions)
 * - Fine refinement (level 0): Use points (subpixel accuracy)
 */
struct LevelModel {
    // Block 1: Subpixel edge points (from edge detection, irregular positions)
    std::vector<ModelPoint> points;

    // Block 2: Integer grid sample points (regular positions, faster for coarse search)
    std::vector<ModelPoint> gridPoints;

    // Contour topology for XLD visualization
    // contourStarts[i] is the first point of contour i
    // contourStarts[i+1] - contourStarts[i] is the number of points in contour i
    // Last element is sentinel (total point count)
    std::vector<int32_t> contourStarts;
    // Whether each contour is closed (forms a loop)
    std::vector<bool> contourClosed;

    int32_t width = 0;
    int32_t height = 0;
    double scale = 1.0;
    int32_t numAngleBins = 16;  ///< Angle bins at this level (halves per level, min=2)

    // SoA for Block 1 (subpixel points)
    std::vector<float> soaX;
    std::vector<float> soaY;
    std::vector<float> soaCosAngle;
    std::vector<float> soaSinAngle;
    std::vector<float> soaWeight;
    std::vector<int16_t> soaAngleBin;

    // SoA for Block 2 (grid points)
    std::vector<float> gridSoaX;
    std::vector<float> gridSoaY;
    std::vector<float> gridSoaCosAngle;
    std::vector<float> gridSoaSinAngle;
    std::vector<float> gridSoaWeight;
    std::vector<int16_t> gridSoaAngleBin;

    void BuildSoA();
    void RegenerateGridPoints();

private:
    static void BuildSoAForPoints(const std::vector<ModelPoint>& pts,
                                   std::vector<float>& x, std::vector<float>& y,
                                   std::vector<float>& cosA, std::vector<float>& sinA,
                                   std::vector<float>& w, std::vector<int16_t>& bins);
};

// =============================================================================
// Supporting Structures
// =============================================================================

/**
 * @brief Precomputed search angle data with rotated bounds per level
 *
 * Halcon pregeneration strategy: Pre-compute all rotation-dependent data
 * at model creation time to avoid expensive computation during search.
 *
 * Key optimizations:
 * - cos/sin computed once per angle (not per search position)
 * - Rotated bounds computed once per angle per level (not per search position)
 * - Search region calculation becomes O(1) lookup
 */
struct SearchAngleData {
    double angle = 0.0;           ///< Rotation angle (radians)
    float cosA = 1.0f;            ///< cos(angle)
    float sinA = 0.0f;            ///< sin(angle)

    /// Precomputed rotated model bounds for each pyramid level
    struct LevelBounds {
        int32_t minX = 0, maxX = 0;  ///< Integer bounds for fast region computation
        int32_t minY = 0, maxY = 0;
    };
    std::vector<LevelBounds> levelBounds;  ///< Bounds[level]
};

/**
 * @brief Fast cosine lookup table for angle difference computation
 * Optimized with O(1) angle normalization instead of while loops
 */
class FastCosTable {
public:
    static constexpr int TABLE_SIZE = 2048;
    static constexpr int TABLE_MASK = TABLE_SIZE - 1;
    static constexpr double TABLE_SCALE = TABLE_SIZE / (2.0 * PI);
    static constexpr double INV_2PI = 1.0 / (2.0 * PI);
    static constexpr double INV_PI = 1.0 / PI;

    FastCosTable();

    // Fast cosine with angle in radians - O(1) normalization
    float FastCos(double angle) const;

    // Fast abs(cos) for symmetric similarity - O(1) normalization
    float FastAbsCos(double angleDiff) const;

private:
    float table_[TABLE_SIZE];
};

// Global cosine lookup table (defined in ShapeModelScore.cpp)
extern const FastCosTable g_cosTable;

// =============================================================================
// ShapeModelImpl: Implementation Class
// =============================================================================

class ShapeModelImpl {
public:
    struct ScaledModelData {
        double scale = 1.0;
        std::vector<LevelModel> levels;
        Size2i templateSize;
        double modelMinX = 0, modelMaxX = 0;
        double modelMinY = 0, modelMaxY = 0;
        double minCoverage = 0.7;
        std::vector<SearchAngleData> searchAngleCache;
        double searchAngleStart = 0.0;
        double searchAngleExtent = 2.0 * PI;
        double searchAngleStep = 0.0;
    };

    // ==========================================================================
    // Create-side fields (populated by CreateModel / FinalizeModel)
    // ==========================================================================

    std::vector<LevelModel> levels_;               ///< Model data per pyramid level
    std::vector<LevelCreateData> levelCreateData_;  ///< Per-level creation metadata
    ModelParams params_;                            ///< User-supplied model parameters
    Point2d origin_;                                ///< Model origin (template coords)
    Size2i templateSize_;                           ///< Template image size
    bool valid_ = false;                            ///< True after successful CreateModel

    QImage templateImage_;                          ///< Saved template for scale model creation

    ShapeModelTimingParams timingParams_;            ///< Timing configuration
    ShapeModelCreateTiming createTiming_;            ///< Create timing results
    mutable ShapeModelFindTiming findTiming_;        ///< Find timing results (mutable for const Find)

    double modelMinX_ = 0, modelMaxX_ = 0;          ///< Model bounding box (cached)
    double modelMinY_ = 0, modelMaxY_ = 0;

    int32_t numAngleBins_ = 0;                       ///< Angle quantization bins (top level)

    double minCoverage_ = 0.7;                       ///< Dynamic coverage threshold

    std::vector<ScaledModelData> scaledModels_;      ///< Cached scaled models for scale search

    // ==========================================================================
    // Search-side fields (used by SearchPyramid / ComputeScore)
    // ==========================================================================

    std::vector<float> cosLUT_;                      ///< Direction quantization lookup table

    std::vector<SearchAngleData> searchAngleCache_;  ///< Precomputed angle data for search
    double searchAngleStart_ = 0.0;                  ///< Search angle range start
    double searchAngleExtent_ = 2.0 * PI;            ///< Search angle range extent
    double searchAngleStep_ = 0.0;                   ///< Search angle step (0 = auto)

    // ==========================================================================
    // Model Creation (ShapeModelCreate.cpp)
    // ==========================================================================

    bool CreateModel(const QImage& image, const QRegion& region, const Point2d& origin);
    void OptimizeModel(std::vector<LevelModel>& levels);
    bool FinalizeModel();
    void ComputeMinCoverage();
    void BuildCosLUT(int32_t numBins);
    void BuildSearchAngleCache(double angleStart, double angleExtent, double angleStep);
    void ComputeModelBounds();
    static void ComputeRotatedBounds(const std::vector<ModelPoint>& points, double angle,
                                     double& minX, double& maxX, double& minY, double& maxY);
    void BuildScaledModels();
    const ScaledModelData* GetScaledModelData(double scale) const;

    // ==========================================================================
    // Search Functions (ShapeModelSearch.cpp)
    // ==========================================================================

    std::vector<MatchResult> SearchPyramid(const AnglePyramid& targetPyramid,
                                            const SearchParams& params) const;

    std::vector<MatchResult> SearchPyramidScaled(const AnglePyramid& targetPyramid,
                                                  const SearchParams& params,
                                                  double scale) const;

    std::vector<MatchResult> SearchLevel(const AnglePyramid& targetPyramid,
                                          int32_t level,
                                          const std::vector<MatchResult>& candidates,
                                          const SearchParams& params,
                                          int32_t positionRadius = -1,
                                          double angleRadiusDeg = -1) const;

    // ==========================================================================
    // Score Computation (ShapeModelScore.cpp)
    // ==========================================================================

    double ComputeScoreAtPosition(const AnglePyramid& pyramid, int32_t level,
                                   double x, double y, double angle, double scale,
                                   double greediness, double* outCoverage = nullptr,
                                   bool useGridPoints = false) const;

    double ComputeScoreAtPositionFast(const AnglePyramid& pyramid, int32_t level,
                                       int32_t x, int32_t y, double angle, double scale,
                                       double* outCoverage = nullptr) const;

    double ComputeScoreBilinearScalar(const AnglePyramid& pyramid, int32_t level,
                                       double x, double y, double angle, double scale,
                                       double greediness, double* outCoverage = nullptr,
                                       bool useGridPoints = false) const;

    double ComputeScoreBilinearSSE(const AnglePyramid& pyramid, int32_t level,
                                    double x, double y, double angle, double scale,
                                    double greediness, double* outCoverage = nullptr,
                                    bool useGridPoints = false) const;

    double ComputeScoreWithSinCos(const AnglePyramid& pyramid, int32_t level,
                                   double x, double y, float cosR, float sinR, double scale,
                                   double greediness, double* outCoverage = nullptr,
                                   bool useGridPoints = false,
                                   float minMagSq = 25.0f) const;

    /// Fast score using nearest-neighbor interpolation (4x less memory access)
    double ComputeScoreNearestNeighbor(const AnglePyramid& pyramid, int32_t level,
                                        int32_t x, int32_t y, float cosR, float sinR,
                                        double greediness, double* outCoverage = nullptr) const;

#ifdef __AVX2__
    /// True AVX2 vectorized score computation (8 points parallel)
    /// Uses nearest-neighbor interpolation with gather instructions
    /// Designed for coarse search where speed is critical
    double ComputeScoreNearestNeighborAVX2(
        const AnglePyramid& pyramid, int32_t level,
        double x, double y, float cosR, float sinR, double scale,
        float greediness, double* outCoverage, bool useGridPoints) const;
#endif

    double ComputeScoreQuantized(const AnglePyramid& pyramid, int32_t level,
                                  double x, double y, float cosR, float sinR, int32_t rotationBin,
                                  double greediness, double* outCoverage = nullptr,
                                  bool useGridPoints = false) const;

    void RefinePosition(const AnglePyramid& pyramid, MatchResult& match,
                        SubpixelMethod method, double scale) const;
};

} // namespace Internal
} // namespace Qi::Vision::Matching
