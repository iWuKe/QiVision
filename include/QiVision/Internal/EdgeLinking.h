#pragma once

/**
 * @file EdgeLinking.h
 * @brief Edge point linking algorithms for contour extraction
 *
 * Provides:
 * - Spatial indexing for efficient neighbor search
 * - Direction-based edge linking
 * - Gap bridging for broken edges
 * - Chain following algorithms
 * - Contour output and filtering
 *
 * Used by:
 * - Internal/Steger: Subpixel edge contour extraction
 * - Internal/Canny: Edge contour extraction
 * - Edge/SubPixelEdge: High-precision edge detection
 *
 * Linking strategies:
 * 1. Greedy: Always link to nearest compatible neighbor
 * 2. Best-First: Prioritize strongest connections
 * 3. Graph-based: Global optimization of edge chains
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QContour.h>

#include <cstdint>
#include <vector>
#include <cmath>
#include <limits>

namespace Qi::Vision::Internal {

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief An edge point for linking
 */
struct EdgePoint {
    double x = 0.0;             ///< X position (subpixel)
    double y = 0.0;             ///< Y position (subpixel)
    double direction = 0.0;     ///< Edge direction in radians
    double magnitude = 0.0;     ///< Edge strength/response
    int32_t id = -1;            ///< Unique ID for linking

    EdgePoint() = default;
    EdgePoint(double px, double py, double dir = 0.0, double mag = 0.0, int32_t pid = -1)
        : x(px), y(py), direction(dir), magnitude(mag), id(pid) {}
};

/**
 * @brief Parameters for edge linking
 */
struct EdgeLinkingParams {
    double maxGap = 3.0;            ///< Maximum distance to link across
    double maxAngleDiff = 0.5;      ///< Maximum angle difference (radians, ~30°)
    double minChainLength = 5.0;    ///< Minimum chain length to keep
    int32_t minChainPoints = 3;     ///< Minimum number of points in chain
    bool allowBranching = false;    ///< Allow Y-junctions (multiple links)
    bool closedContours = true;     ///< Try to close contours
    double closureMaxGap = 5.0;     ///< Maximum gap for closing contours
    double closureMaxAngle = 0.7;   ///< Maximum angle for closure (~40°)
};

/**
 * @brief An edge chain (sequence of linked points)
 */
struct EdgeChain {
    std::vector<int32_t> pointIds;  ///< Indices into the point array
    double length = 0.0;            ///< Total chain length
    bool isClosed = false;          ///< Whether the chain forms a loop

    int32_t Size() const { return static_cast<int32_t>(pointIds.size()); }
    bool Empty() const { return pointIds.empty(); }
};

/**
 * @brief Grid-based spatial index for fast neighbor queries
 */
class SpatialGrid {
public:
    /**
     * @brief Build spatial index from edge points
     * @param points Edge points
     * @param cellSize Grid cell size
     */
    void Build(const std::vector<EdgePoint>& points, double cellSize = 4.0);

    /**
     * @brief Find points within radius of a query point
     * @param x Query x
     * @param y Query y
     * @param radius Search radius
     * @return Vector of point indices
     */
    std::vector<int32_t> FindNeighbors(double x, double y, double radius) const;

    /**
     * @brief Find the nearest point to a query location
     * @param x Query x
     * @param y Query y
     * @param maxDist Maximum search distance
     * @return Point index, or -1 if not found
     */
    int32_t FindNearest(double x, double y, double maxDist) const;

    /**
     * @brief Get all points in a grid cell
     * @param cellX Cell x index
     * @param cellY Cell y index
     * @return Vector of point indices in the cell
     */
    const std::vector<int32_t>& GetCell(int32_t cellX, int32_t cellY) const;

    /**
     * @brief Check if the grid is empty
     */
    bool Empty() const { return points_ == nullptr || points_->empty(); }

    /**
     * @brief Get grid dimensions
     */
    void GetGridSize(int32_t& gridW, int32_t& gridH) const {
        gridW = gridWidth_;
        gridH = gridHeight_;
    }

private:
    const std::vector<EdgePoint>* points_ = nullptr;
    std::vector<std::vector<int32_t>> grid_;
    double cellSize_ = 4.0;
    double minX_ = 0.0, minY_ = 0.0;
    int32_t gridWidth_ = 0, gridHeight_ = 0;
    std::vector<int32_t> emptyCell_;  // For returning when cell is out of bounds
};

// ============================================================================
// Main Linking Functions
// ============================================================================

/**
 * @brief Link edge points into chains
 *
 * Uses direction and proximity to connect edge points into ordered chains.
 *
 * @param points Edge points to link
 * @param params Linking parameters
 * @return Vector of edge chains
 */
std::vector<EdgeChain> LinkEdgePoints(const std::vector<EdgePoint>& points,
                                       const EdgeLinkingParams& params);

/**
 * @brief Link edge points using a pre-built spatial index
 *
 * @param points Edge points
 * @param grid Pre-built spatial grid
 * @param params Linking parameters
 * @return Vector of edge chains
 */
std::vector<EdgeChain> LinkEdgePointsWithGrid(const std::vector<EdgePoint>& points,
                                               const SpatialGrid& grid,
                                               const EdgeLinkingParams& params);

/**
 * @brief Convert edge chains to QContour objects
 *
 * @param points Original edge points
 * @param chains Edge chains
 * @return Vector of contours
 */
std::vector<QContour> ChainsToContours(const std::vector<EdgePoint>& points,
                                        const std::vector<EdgeChain>& chains);

/**
 * @brief Link edge points directly to contours (convenience function)
 *
 * @param points Edge points
 * @param params Linking parameters
 * @return Vector of contours
 */
std::vector<QContour> LinkToContours(const std::vector<EdgePoint>& points,
                                      const EdgeLinkingParams& params);

// ============================================================================
// Chain Operations
// ============================================================================

/**
 * @brief Filter chains by minimum length
 *
 * @param points Edge points
 * @param chains Input chains
 * @param minLength Minimum length in pixels
 * @return Filtered chains
 */
std::vector<EdgeChain> FilterChainsByLength(const std::vector<EdgePoint>& points,
                                             const std::vector<EdgeChain>& chains,
                                             double minLength);

/**
 * @brief Filter chains by minimum point count
 *
 * @param chains Input chains
 * @param minPoints Minimum number of points
 * @return Filtered chains
 */
std::vector<EdgeChain> FilterChainsByPointCount(const std::vector<EdgeChain>& chains,
                                                 int32_t minPoints);

/**
 * @brief Merge compatible chains at their endpoints
 *
 * Attempts to merge chains that have close, compatible endpoints.
 *
 * @param points Edge points
 * @param chains Input chains
 * @param maxGap Maximum gap between endpoints
 * @param maxAngleDiff Maximum angle difference at junction
 * @return Merged chains
 */
std::vector<EdgeChain> MergeChains(const std::vector<EdgePoint>& points,
                                    const std::vector<EdgeChain>& chains,
                                    double maxGap,
                                    double maxAngleDiff);

/**
 * @brief Try to close open chains
 *
 * Checks if chain endpoints are close enough to form a loop.
 *
 * @param points Edge points
 * @param chains Input chains
 * @param maxGap Maximum gap for closure
 * @param maxAngleDiff Maximum angle difference for closure
 */
void TryCloseChains(const std::vector<EdgePoint>& points,
                    std::vector<EdgeChain>& chains,
                    double maxGap,
                    double maxAngleDiff);

/**
 * @brief Compute the length of a chain
 *
 * @param points Edge points
 * @param chain The chain
 * @return Total length in pixels
 */
double ComputeChainLength(const std::vector<EdgePoint>& points,
                          const EdgeChain& chain);

/**
 * @brief Reverse the direction of a chain
 *
 * @param chain Chain to reverse (modified in place)
 */
void ReverseChain(EdgeChain& chain);

// ============================================================================
// Linking Utilities
// ============================================================================

/**
 * @brief Check if two directions are compatible for linking
 *
 * Edge directions can point in either direction along the edge,
 * so we compare the absolute angle difference.
 *
 * @param dir1 First direction (radians)
 * @param dir2 Second direction (radians)
 * @param maxDiff Maximum allowed difference
 * @return true if directions are compatible
 */
inline bool DirectionsCompatible(double dir1, double dir2, double maxDiff) {
    // Normalize both to [0, 2π)
    while (dir1 < 0) dir1 += 2.0 * M_PI;
    while (dir1 >= 2.0 * M_PI) dir1 -= 2.0 * M_PI;
    while (dir2 < 0) dir2 += 2.0 * M_PI;
    while (dir2 >= 2.0 * M_PI) dir2 -= 2.0 * M_PI;

    // Compute angle difference
    double diff = std::abs(dir1 - dir2);
    if (diff > M_PI) diff = 2.0 * M_PI - diff;

    // Edge directions can be opposite (pointing along the edge either way)
    // So also check if they're π apart
    double diffOpposite = std::abs(diff - M_PI);

    return diff <= maxDiff || diffOpposite <= maxDiff;
}

/**
 * @brief Compute distance between two edge points
 */
inline double PointDistance(const EdgePoint& p1, const EdgePoint& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

/**
 * @brief Compute the tangent direction from p1 to p2
 */
inline double TangentDirection(const EdgePoint& p1, const EdgePoint& p2) {
    return std::atan2(p2.y - p1.y, p2.x - p1.x);
}

/**
 * @brief Check if linking from p1 to p2 is geometrically consistent
 *
 * The direction from p1 to p2 should be roughly perpendicular to the
 * edge direction (edges are perpendicular to the intensity gradient).
 *
 * @param p1 First point
 * @param p2 Second point
 * @param maxDeviation Maximum angular deviation from perpendicular
 * @return true if linking is geometrically consistent
 */
bool IsLinkGeometricallyConsistent(const EdgePoint& p1, const EdgePoint& p2,
                                    double maxDeviation = 0.7);

/**
 * @brief Score a potential link between two points
 *
 * Higher scores indicate better links. Considers distance, angle, and magnitude.
 *
 * @param p1 First point
 * @param p2 Second point
 * @param maxGap Maximum allowed gap
 * @param maxAngleDiff Maximum allowed angle difference
 * @return Link score (0 = invalid, higher = better)
 */
double ScoreLink(const EdgePoint& p1, const EdgePoint& p2,
                 double maxGap, double maxAngleDiff);

// ============================================================================
// Advanced Linking
// ============================================================================

/**
 * @brief Find the best next point to link from a given point
 *
 * Considers all nearby points and returns the one with the best link score.
 *
 * @param points All edge points
 * @param grid Spatial index
 * @param currentIdx Current point index
 * @param usedPoints Set of already-used point indices
 * @param params Linking parameters
 * @param searchBackward If true, search in backward direction
 * @return Index of best next point, or -1 if none found
 */
int32_t FindBestNextPoint(const std::vector<EdgePoint>& points,
                          const SpatialGrid& grid,
                          int32_t currentIdx,
                          const std::vector<bool>& usedPoints,
                          const EdgeLinkingParams& params,
                          bool searchBackward = false);

/**
 * @brief Build a chain starting from a seed point
 *
 * Follows the edge in both forward and backward directions.
 *
 * @param points All edge points
 * @param grid Spatial index
 * @param seedIdx Starting point index
 * @param usedPoints Set of already-used points (updated)
 * @param params Linking parameters
 * @return The built chain
 */
EdgeChain BuildChainFromSeed(const std::vector<EdgePoint>& points,
                             const SpatialGrid& grid,
                             int32_t seedIdx,
                             std::vector<bool>& usedPoints,
                             const EdgeLinkingParams& params);

// ============================================================================
// Gap Bridging
// ============================================================================

/**
 * @brief Bridge gaps between chain endpoints
 *
 * Attempts to connect chain endpoints that are within a certain distance,
 * even if there are no intermediate edge points.
 *
 * @param points Edge points
 * @param chains Input chains
 * @param maxGap Maximum gap to bridge
 * @param maxAngleDiff Maximum angle difference at junction
 * @return Bridged chains (may have fewer chains if some were merged)
 */
std::vector<EdgeChain> BridgeGaps(const std::vector<EdgePoint>& points,
                                   std::vector<EdgeChain>& chains,
                                   double maxGap,
                                   double maxAngleDiff);

// ============================================================================
// Contour Conversion
// ============================================================================

/**
 * @brief Convert an edge chain to a QContour
 *
 * @param points Edge points
 * @param chain The chain to convert
 * @return QContour with the chain points
 */
QContour ChainToContour(const std::vector<EdgePoint>& points,
                        const EdgeChain& chain);

/**
 * @brief Extract contour points from edge points
 *
 * @param points Edge points
 * @param chain The chain
 * @return Vector of ContourPoint
 */
std::vector<ContourPoint> ExtractContourPoints(const std::vector<EdgePoint>& points,
                                                const EdgeChain& chain);

} // namespace Qi::Vision::Internal
