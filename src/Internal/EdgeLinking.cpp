/**
 * @file EdgeLinking.cpp
 * @brief Implementation of edge point linking algorithms
 */

#include <QiVision/Internal/EdgeLinking.h>

#include <algorithm>
#include <cstring>

namespace Qi::Vision::Internal {

// ============================================================================
// SpatialGrid Implementation
// ============================================================================

void SpatialGrid::Build(const std::vector<EdgePoint>& points, double cellSize) {
    points_ = &points;
    cellSize_ = cellSize;
    grid_.clear();

    if (points.empty()) {
        gridWidth_ = 0;
        gridHeight_ = 0;
        return;
    }

    // Find bounding box
    minX_ = points[0].x;
    minY_ = points[0].y;
    double maxX = points[0].x;
    double maxY = points[0].y;

    for (const auto& p : points) {
        minX_ = std::min(minX_, p.x);
        minY_ = std::min(minY_, p.y);
        maxX = std::max(maxX, p.x);
        maxY = std::max(maxY, p.y);
    }

    // Add small margin
    minX_ -= cellSize_;
    minY_ -= cellSize_;

    // Compute grid dimensions
    gridWidth_ = static_cast<int32_t>((maxX - minX_) / cellSize_) + 2;
    gridHeight_ = static_cast<int32_t>((maxY - minY_) / cellSize_) + 2;

    // Initialize grid
    grid_.resize(gridWidth_ * gridHeight_);

    // Add points to grid cells
    for (size_t i = 0; i < points.size(); ++i) {
        int32_t cellX = static_cast<int32_t>((points[i].x - minX_) / cellSize_);
        int32_t cellY = static_cast<int32_t>((points[i].y - minY_) / cellSize_);

        // Clamp to grid bounds
        cellX = std::max(0, std::min(gridWidth_ - 1, cellX));
        cellY = std::max(0, std::min(gridHeight_ - 1, cellY));

        grid_[cellY * gridWidth_ + cellX].push_back(static_cast<int32_t>(i));
    }
}

std::vector<int32_t> SpatialGrid::FindNeighbors(double x, double y, double radius) const {
    std::vector<int32_t> result;
    if (Empty()) return result;

    // Compute cell range to search
    int32_t cellX = static_cast<int32_t>((x - minX_) / cellSize_);
    int32_t cellY = static_cast<int32_t>((y - minY_) / cellSize_);
    int32_t cellRadius = static_cast<int32_t>(std::ceil(radius / cellSize_));

    int32_t minCellX = std::max(0, cellX - cellRadius);
    int32_t maxCellX = std::min(gridWidth_ - 1, cellX + cellRadius);
    int32_t minCellY = std::max(0, cellY - cellRadius);
    int32_t maxCellY = std::min(gridHeight_ - 1, cellY + cellRadius);

    double radiusSq = radius * radius;

    // Search all cells in range
    for (int32_t cy = minCellY; cy <= maxCellY; ++cy) {
        for (int32_t cx = minCellX; cx <= maxCellX; ++cx) {
            const auto& cell = grid_[cy * gridWidth_ + cx];
            for (int32_t idx : cell) {
                double dx = (*points_)[idx].x - x;
                double dy = (*points_)[idx].y - y;
                if (dx * dx + dy * dy <= radiusSq) {
                    result.push_back(idx);
                }
            }
        }
    }

    return result;
}

int32_t SpatialGrid::FindNearest(double x, double y, double maxDist) const {
    if (Empty()) return -1;

    auto neighbors = FindNeighbors(x, y, maxDist);
    if (neighbors.empty()) return -1;

    int32_t nearest = -1;
    double minDistSq = maxDist * maxDist;

    for (int32_t idx : neighbors) {
        double dx = (*points_)[idx].x - x;
        double dy = (*points_)[idx].y - y;
        double distSq = dx * dx + dy * dy;
        if (distSq < minDistSq) {
            minDistSq = distSq;
            nearest = idx;
        }
    }

    return nearest;
}

const std::vector<int32_t>& SpatialGrid::GetCell(int32_t cellX, int32_t cellY) const {
    if (cellX < 0 || cellX >= gridWidth_ || cellY < 0 || cellY >= gridHeight_) {
        return emptyCell_;
    }
    return grid_[cellY * gridWidth_ + cellX];
}

// ============================================================================
// Main Linking Functions
// ============================================================================

std::vector<EdgeChain> LinkEdgePoints(const std::vector<EdgePoint>& points,
                                       const EdgeLinkingParams& params) {
    if (points.empty()) return {};

    // Build spatial index
    SpatialGrid grid;
    grid.Build(points, params.maxGap);

    return LinkEdgePointsWithGrid(points, grid, params);
}

std::vector<EdgeChain> LinkEdgePointsWithGrid(const std::vector<EdgePoint>& points,
                                               const SpatialGrid& grid,
                                               const EdgeLinkingParams& params) {
    if (points.empty()) return {};

    std::vector<EdgeChain> chains;
    std::vector<bool> usedPoints(points.size(), false);

    // Sort points by magnitude (start from strongest edges)
    std::vector<int32_t> sortedIndices(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        sortedIndices[i] = static_cast<int32_t>(i);
    }
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [&points](int32_t a, int32_t b) {
                  return points[a].magnitude > points[b].magnitude;
              });

    // Build chains starting from unused strong points
    for (int32_t seedIdx : sortedIndices) {
        if (usedPoints[seedIdx]) continue;

        EdgeChain chain = BuildChainFromSeed(points, grid, seedIdx,
                                              usedPoints, params);

        if (chain.Size() >= params.minChainPoints) {
            chain.length = ComputeChainLength(points, chain);
            chains.push_back(std::move(chain));
        }
    }

    // Try to close chains
    if (params.closedContours) {
        TryCloseChains(points, chains, params.closureMaxGap, params.closureMaxAngle);
    }

    // Filter by length
    if (params.minChainLength > 0) {
        chains = FilterChainsByLength(points, chains, params.minChainLength);
    }

    return chains;
}

std::vector<QContour> ChainsToContours(const std::vector<EdgePoint>& points,
                                        const std::vector<EdgeChain>& chains) {
    std::vector<QContour> contours;
    contours.reserve(chains.size());

    for (const auto& chain : chains) {
        contours.push_back(ChainToContour(points, chain));
    }

    return contours;
}

std::vector<QContour> LinkToContours(const std::vector<EdgePoint>& points,
                                      const EdgeLinkingParams& params) {
    auto chains = LinkEdgePoints(points, params);
    return ChainsToContours(points, chains);
}

// ============================================================================
// Chain Operations
// ============================================================================

std::vector<EdgeChain> FilterChainsByLength(const std::vector<EdgePoint>& points,
                                             const std::vector<EdgeChain>& chains,
                                             double minLength) {
    std::vector<EdgeChain> result;
    result.reserve(chains.size());

    for (const auto& chain : chains) {
        double length = chain.length;
        if (length <= 0) {
            length = ComputeChainLength(points, chain);
        }
        if (length >= minLength) {
            EdgeChain filtered = chain;
            filtered.length = length;
            result.push_back(std::move(filtered));
        }
    }

    return result;
}

std::vector<EdgeChain> FilterChainsByPointCount(const std::vector<EdgeChain>& chains,
                                                 int32_t minPoints) {
    std::vector<EdgeChain> result;
    result.reserve(chains.size());

    for (const auto& chain : chains) {
        if (chain.Size() >= minPoints) {
            result.push_back(chain);
        }
    }

    return result;
}

std::vector<EdgeChain> MergeChains(const std::vector<EdgePoint>& points,
                                    const std::vector<EdgeChain>& chains,
                                    double maxGap,
                                    double maxAngleDiff) {
    if (chains.size() <= 1) return chains;

    std::vector<EdgeChain> result;
    std::vector<bool> merged(chains.size(), false);

    for (size_t i = 0; i < chains.size(); ++i) {
        if (merged[i]) continue;

        EdgeChain current = chains[i];

        // Try to extend the chain by merging with others
        bool extended = true;
        while (extended) {
            extended = false;

            for (size_t j = 0; j < chains.size(); ++j) {
                if (i == j || merged[j]) continue;

                const auto& other = chains[j];

                // Get endpoints of current chain
                const auto& currStart = points[current.pointIds.front()];
                const auto& currEnd = points[current.pointIds.back()];

                // Get endpoints of other chain
                const auto& otherStart = points[other.pointIds.front()];
                const auto& otherEnd = points[other.pointIds.back()];

                // Try all 4 endpoint combinations
                double d1 = PointDistance(currEnd, otherStart);
                double d2 = PointDistance(currEnd, otherEnd);
                double d3 = PointDistance(currStart, otherStart);
                double d4 = PointDistance(currStart, otherEnd);

                if (d1 <= maxGap && DirectionsCompatible(currEnd.direction,
                                                          otherStart.direction, maxAngleDiff)) {
                    // Append other to end of current
                    for (int32_t idx : other.pointIds) {
                        current.pointIds.push_back(idx);
                    }
                    merged[j] = true;
                    extended = true;
                } else if (d2 <= maxGap && DirectionsCompatible(currEnd.direction,
                                                                  otherEnd.direction, maxAngleDiff)) {
                    // Append reversed other to end of current
                    for (auto it = other.pointIds.rbegin(); it != other.pointIds.rend(); ++it) {
                        current.pointIds.push_back(*it);
                    }
                    merged[j] = true;
                    extended = true;
                } else if (d3 <= maxGap && DirectionsCompatible(currStart.direction,
                                                                  otherStart.direction, maxAngleDiff)) {
                    // Prepend reversed other to start of current
                    std::vector<int32_t> newIds;
                    for (auto it = other.pointIds.rbegin(); it != other.pointIds.rend(); ++it) {
                        newIds.push_back(*it);
                    }
                    for (int32_t idx : current.pointIds) {
                        newIds.push_back(idx);
                    }
                    current.pointIds = std::move(newIds);
                    merged[j] = true;
                    extended = true;
                } else if (d4 <= maxGap && DirectionsCompatible(currStart.direction,
                                                                  otherEnd.direction, maxAngleDiff)) {
                    // Prepend other to start of current
                    std::vector<int32_t> newIds;
                    for (int32_t idx : other.pointIds) {
                        newIds.push_back(idx);
                    }
                    for (int32_t idx : current.pointIds) {
                        newIds.push_back(idx);
                    }
                    current.pointIds = std::move(newIds);
                    merged[j] = true;
                    extended = true;
                }
            }
        }

        current.length = ComputeChainLength(points, current);
        result.push_back(std::move(current));
    }

    return result;
}

void TryCloseChains(const std::vector<EdgePoint>& points,
                    std::vector<EdgeChain>& chains,
                    double maxGap,
                    double maxAngleDiff) {
    for (auto& chain : chains) {
        if (chain.isClosed || chain.Size() < 3) continue;

        const auto& first = points[chain.pointIds.front()];
        const auto& last = points[chain.pointIds.back()];

        double dist = PointDistance(first, last);
        if (dist <= maxGap && DirectionsCompatible(first.direction,
                                                     last.direction, maxAngleDiff)) {
            chain.isClosed = true;
        }
    }
}

double ComputeChainLength(const std::vector<EdgePoint>& points,
                          const EdgeChain& chain) {
    if (chain.Size() < 2) return 0.0;

    double length = 0.0;
    for (size_t i = 1; i < chain.pointIds.size(); ++i) {
        length += PointDistance(points[chain.pointIds[i - 1]],
                                points[chain.pointIds[i]]);
    }

    return length;
}

void ReverseChain(EdgeChain& chain) {
    std::reverse(chain.pointIds.begin(), chain.pointIds.end());
}

// ============================================================================
// Linking Utilities
// ============================================================================

bool IsLinkGeometricallyConsistent(const EdgePoint& p1, const EdgePoint& p2,
                                    double maxDeviation) {
    double tangent = TangentDirection(p1, p2);

    // Edge direction should be roughly perpendicular to tangent
    // (edge direction is the gradient direction, which is perpendicular to the edge)
    double perpendicular = tangent + M_PI / 2.0;

    bool dir1Ok = DirectionsCompatible(p1.direction, perpendicular, maxDeviation);
    bool dir2Ok = DirectionsCompatible(p2.direction, perpendicular, maxDeviation);

    return dir1Ok && dir2Ok;
}

double ScoreLink(const EdgePoint& p1, const EdgePoint& p2,
                 double maxGap, double maxAngleDiff) {
    double dist = PointDistance(p1, p2);
    if (dist > maxGap || dist < 1e-6) return 0.0;

    // Check direction compatibility
    if (!DirectionsCompatible(p1.direction, p2.direction, maxAngleDiff)) {
        return 0.0;
    }

    // Score based on distance (closer is better)
    double distScore = 1.0 - dist / maxGap;

    // Score based on magnitude (prefer stronger edges)
    double magScore = (p1.magnitude + p2.magnitude) * 0.5;

    // Score based on direction consistency
    double tangent = TangentDirection(p1, p2);
    double perpendicular = tangent + M_PI / 2.0;

    double angleDiff1 = std::abs(p1.direction - perpendicular);
    while (angleDiff1 > M_PI) angleDiff1 = 2.0 * M_PI - angleDiff1;
    if (angleDiff1 > M_PI / 2.0) angleDiff1 = M_PI - angleDiff1;

    double angleDiff2 = std::abs(p2.direction - perpendicular);
    while (angleDiff2 > M_PI) angleDiff2 = 2.0 * M_PI - angleDiff2;
    if (angleDiff2 > M_PI / 2.0) angleDiff2 = M_PI - angleDiff2;

    double angleScore = 1.0 - (angleDiff1 + angleDiff2) / M_PI;
    angleScore = std::max(0.0, angleScore);

    return distScore * angleScore * (1.0 + 0.1 * magScore);
}

// ============================================================================
// Advanced Linking
// ============================================================================

int32_t FindBestNextPoint(const std::vector<EdgePoint>& points,
                          const SpatialGrid& grid,
                          int32_t currentIdx,
                          const std::vector<bool>& usedPoints,
                          const EdgeLinkingParams& params,
                          bool searchBackward) {
    const auto& current = points[currentIdx];

    // Search in direction perpendicular to edge direction
    double searchDir = current.direction + (searchBackward ? -M_PI / 2.0 : M_PI / 2.0);

    // Find neighbors
    auto neighbors = grid.FindNeighbors(current.x, current.y, params.maxGap);

    int32_t bestIdx = -1;
    double bestScore = 0.0;

    for (int32_t idx : neighbors) {
        if (idx == currentIdx || usedPoints[idx]) continue;

        double score = ScoreLink(current, points[idx],
                                  params.maxGap, params.maxAngleDiff);

        // Bias towards the search direction
        if (score > 0) {
            double tangent = TangentDirection(current, points[idx]);
            double dirDiff = std::abs(tangent - searchDir);
            while (dirDiff > M_PI) dirDiff = 2.0 * M_PI - dirDiff;

            // Prefer points in the search direction
            if (dirDiff < M_PI / 2.0) {
                score *= 1.0 + 0.5 * (1.0 - dirDiff / (M_PI / 2.0));
            }
        }

        if (score > bestScore) {
            bestScore = score;
            bestIdx = idx;
        }
    }

    return bestIdx;
}

EdgeChain BuildChainFromSeed(const std::vector<EdgePoint>& points,
                             const SpatialGrid& grid,
                             int32_t seedIdx,
                             std::vector<bool>& usedPoints,
                             const EdgeLinkingParams& params) {
    EdgeChain chain;
    chain.pointIds.push_back(seedIdx);
    usedPoints[seedIdx] = true;

    // Extend forward
    int32_t currentIdx = seedIdx;
    while (true) {
        int32_t nextIdx = FindBestNextPoint(points, grid, currentIdx,
                                            usedPoints, params, false);
        if (nextIdx < 0) break;

        chain.pointIds.push_back(nextIdx);
        usedPoints[nextIdx] = true;
        currentIdx = nextIdx;
    }

    // Extend backward (prepend to chain)
    currentIdx = seedIdx;
    std::vector<int32_t> backward;
    while (true) {
        int32_t prevIdx = FindBestNextPoint(points, grid, currentIdx,
                                            usedPoints, params, true);
        if (prevIdx < 0) break;

        backward.push_back(prevIdx);
        usedPoints[prevIdx] = true;
        currentIdx = prevIdx;
    }

    // Prepend backward points
    if (!backward.empty()) {
        std::reverse(backward.begin(), backward.end());
        std::vector<int32_t> newIds;
        newIds.reserve(backward.size() + chain.pointIds.size());
        for (int32_t idx : backward) {
            newIds.push_back(idx);
        }
        for (int32_t idx : chain.pointIds) {
            newIds.push_back(idx);
        }
        chain.pointIds = std::move(newIds);
    }

    return chain;
}

// ============================================================================
// Gap Bridging
// ============================================================================

std::vector<EdgeChain> BridgeGaps(const std::vector<EdgePoint>& points,
                                   std::vector<EdgeChain>& chains,
                                   double maxGap,
                                   double maxAngleDiff) {
    // Use merge chains with larger gap tolerance
    return MergeChains(points, chains, maxGap, maxAngleDiff);
}

// ============================================================================
// Contour Conversion
// ============================================================================

QContour ChainToContour(const std::vector<EdgePoint>& points,
                        const EdgeChain& chain) {
    auto contourPoints = ExtractContourPoints(points, chain);

    QContour contour;
    for (const auto& cp : contourPoints) {
        contour.AddPoint(cp);
    }
    contour.SetClosed(chain.isClosed);

    return contour;
}

std::vector<ContourPoint> ExtractContourPoints(const std::vector<EdgePoint>& points,
                                                const EdgeChain& chain) {
    std::vector<ContourPoint> result;
    result.reserve(chain.pointIds.size());

    for (int32_t idx : chain.pointIds) {
        const auto& ep = points[idx];
        ContourPoint cp;
        cp.x = ep.x;
        cp.y = ep.y;
        cp.amplitude = ep.magnitude;
        cp.direction = ep.direction;
        result.push_back(cp);
    }

    return result;
}

} // namespace Qi::Vision::Internal
