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
#include "ShapeModelResponseMap.h"

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

    // Fixed 16-bin angle quantization for response map LUT (independent of per-level bins)
    std::vector<int16_t> gridSoaAngleBin16;

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

/// Decompiled dword_1800D4CF0[21] — Search radius per refinement level
/// Used during model creation to populate searchRadiusPerLevel_ via table lookup.
/// Assigned from coarsest level downward: level[N-1] gets [0], level[N-2] gets [1], etc.
static constexpr int32_t kAngleBinSizeTable[21] = {
    2, 3, 3, 4, 4, 4, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7
};

class ShapeModelImpl {
public:
    // ==========================================================================
    // Create-side fields (populated by CreateModel / FinalizeModel)
    // ==========================================================================

    std::vector<LevelModel> levels_;               ///< Model data per pyramid level
    std::vector<LevelCreateData> levelCreateData_;  ///< Per-level creation metadata
    ModelParams params_;                            ///< User-supplied model parameters
    Point2d origin_;                                ///< Model origin (template coords)
    Size2i templateSize_;                           ///< Template image size
    bool valid_ = false;                            ///< True after successful CreateModel

    ShapeModelTimingParams timingParams_;            ///< Timing configuration
    ShapeModelCreateTiming createTiming_;            ///< Create timing results
    mutable ShapeModelFindTiming findTiming_;        ///< Find timing results (mutable for const Find)

    double modelMinX_ = 0, modelMaxX_ = 0;          ///< Model bounding box (cached)
    double modelMinY_ = 0, modelMaxY_ = 0;

    int32_t numAngleBins_ = 0;                       ///< Angle quantization bins (top level)

    double minCoverage_ = 0.7;                       ///< Dynamic coverage threshold

    std::vector<int32_t> searchRadiusPerLevel_;      ///< Decompiled dword_1800D4CF0 lookup results per level

    // ==========================================================================
    // Search-side fields (used by SearchPyramid / ComputeScore)
    // ==========================================================================

    std::vector<float> cosLUT_;                      ///< Direction quantization lookup table
    ResponseMapLUT responseMapLUT_;                   ///< 16x16 cos-based LUT for response map

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
    // ==========================================================================
    // Search Pipeline (ShapeModelSearch.cpp) — 4-stage architecture
    // ==========================================================================

    /// Main entry point: 4-stage pipeline (CoarseSearch → PyramidRefine → SubPixelRefine → FinalizeResults)
    /// @param applyNMS If true (default), FinalizeResults applies NMS.
    ///                 If false, NMS is skipped (caller handles it).
    std::vector<MatchResult> SearchPyramid(const AnglePyramid& targetPyramid,
                                            const SearchParams& params,
                                            bool applyNMS = true) const;

    /// Stage 1: Coarse search via response map + LUT (primary path)
    std::vector<MatchResult> CoarseSearch(const AnglePyramid& targetPyramid,
                                           int32_t startLevel,
                                           const SearchParams& params) const;

    /// Stage 1 fallback: float dot-product coarse search (original implementation)
    std::vector<MatchResult> CoarseSearchFloat(const AnglePyramid& targetPyramid,
                                                int32_t startLevel,
                                                const SearchParams& params) const;

    /// Stage 2: Per-level pyramid refinement (coarse → fine)
    std::vector<MatchResult> PyramidRefine(const AnglePyramid& targetPyramid,
                                            int32_t startLevel,
                                            std::vector<MatchResult> candidates,
                                            const SearchParams& params) const;

    /// Per-level refinement: position grid + angle iteration convergence + parabolic interpolation
    /// Decompiled sub_18003C7B0: all levels do both position and angle refinement (Path A: no scale)
    std::vector<MatchResult> RefineAtLevel(
        const AnglePyramid& pyramid, int32_t level, int32_t startLevel,
        std::vector<MatchResult> candidates, const SearchParams& params) const;

    /// Per-level scaled refinement: position grid + angle iteration + scale iteration
    /// Decompiled sub_180040150: Path B with scale refinement
    std::vector<MatchResult> RefineAtLevelScaled(
        const AnglePyramid& pyramid, int32_t level, int32_t startLevel,
        std::vector<MatchResult> candidates, const SearchParams& params) const;

    /// Stage 3: Subpixel position/angle refinement at level 0
    std::vector<MatchResult> SubPixelRefine(const AnglePyramid& targetPyramid,
                                             std::vector<MatchResult> candidates,
                                             const SearchParams& params) const;

    /// Stage 4: Final precise scoring + NMS + maxMatches
    /// @param applyNMS If true, apply NonMaxSuppressionOverlap (FindShapeModel path).
    ///                 If false, skip NMS (FindScaledShapeModel path — NMS done at caller).
    std::vector<MatchResult> FinalizeResults(const AnglePyramid& targetPyramid,
                                              std::vector<MatchResult> candidates,
                                              const SearchParams& params,
                                              bool applyNMS = true) const;

    /// Halcon-style angle step: min(11.25°, acos(1 - safety²/(2*R²)))
    static double ComputeHalconAngleStep(double maxRadius, double safety);

    /// Decompiled sub_1800B82C0: geometry-based scale step = CONST / ceil(maxRadius)
    static double ComputeScaleStep(double maxRadius);

    /// Candidate collection with spatial hash NMS + angle distance suppression
    /// Replaces naive sort+truncate. Aligned with decompiled sub_18004A5A0.
    static std::vector<MatchResult> CollectCandidatesNMS(
        std::vector<MatchResult> candidates,
        int32_t imageWidth,
        double angleStep);

    // ==========================================================================
    // Score Computation (ShapeModelScore.cpp) — Unified template-based
    // ==========================================================================

    /// Single scoring entry point (dispatches to template instantiations in ScoreCore.h)
    /// Replaces: ComputeScoreAtPosition, ComputeScoreBilinearSSE/Scalar,
    ///           ComputeScoreWithSinCos, ComputeScoreNearestNeighbor/AVX2, ComputeScoreQuantized
    double ComputeScore(const AnglePyramid& pyramid, int32_t level,
                        double x, double y, float cosR, float sinR, double scale,
                        double greediness, double minScore,
                        double* outCoverage = nullptr,
                        bool useGridPoints = false) const;

    /// Subpixel refinement (Parabolic / Gauss-Newton gradient profile)
    void RefinePosition(const AnglePyramid& pyramid, MatchResult& match,
                        SubpixelMethod method, double scale) const;

    /// Gauss-Newton gradient profile refinement (used by LeastSquares/LeastSquaresHigh)
    void RefineGaussNewton(const AnglePyramid& pyramid, MatchResult& match,
                           double scale, int32_t numIterations) const;
};

} // namespace Internal
} // namespace Qi::Vision::Matching
