#include <QiVision/Internal/MorphBinary.h>
#include <QiVision/Internal/RLEOps.h>

#include <algorithm>
#include <unordered_set>
#include <queue>

namespace Qi::Vision::Internal {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Translate region by offset
QRegion TranslateRegion(const QRegion& region, int32_t dx, int32_t dy) {
    if (region.Empty()) return QRegion();

    auto runs = region.Runs();
    auto translated = TranslateRuns(runs, dx, dy);
    return QRegion(translated);
}

// Union two regions
QRegion UnionRegions(const QRegion& r1, const QRegion& r2) {
    if (r1.Empty()) return r2;
    if (r2.Empty()) return r1;

    auto runs = UnionRuns(r1.Runs(), r2.Runs());
    return QRegion(runs);
}

// Intersect two regions
QRegion IntersectRegions(const QRegion& r1, const QRegion& r2) {
    if (r1.Empty() || r2.Empty()) return QRegion();

    auto runs = IntersectRuns(r1.Runs(), r2.Runs());
    return QRegion(runs);
}

// Difference of two regions (r1 - r2)
QRegion DifferenceRegions(const QRegion& r1, const QRegion& r2) {
    if (r1.Empty()) return QRegion();
    if (r2.Empty()) return r1;

    auto runs = DifferenceRuns(r1.Runs(), r2.Runs());
    return QRegion(runs);
}

// Complement within bounds
QRegion ComplementRegion(const QRegion& region, const Rect2i& bounds) {
    auto runs = ComplementRuns(region.Runs(), bounds);
    return QRegion(runs);
}

} // anonymous namespace

// =============================================================================
// Basic Morphological Operations
// =============================================================================

QRegion Dilate(const QRegion& region, const StructElement& se) {
    if (region.Empty() || se.Empty()) return region;

    // Dilation: union of all translations of region by SE coordinates
    auto coords = se.GetCoordinates();
    if (coords.empty()) return region;

    QRegion result;
    for (const auto& pt : coords) {
        // SE stores (x, y) relative to anchor, translate by (-x, -y) for dilation
        // Actually for dilation we use SE directly (not reflected)
        QRegion translated = TranslateRegion(region, pt.x, pt.y);
        result = UnionRegions(result, translated);
    }

    return result;
}

QRegion Erode(const QRegion& region, const StructElement& se) {
    if (region.Empty() || se.Empty()) return QRegion();

    // Erosion: intersection of all translations of region by reflected SE coordinates
    auto coords = se.GetCoordinates();
    if (coords.empty()) return region;

    // Start with first translation
    QRegion result = TranslateRegion(region, -coords[0].x, -coords[0].y);

    // Intersect with remaining translations
    for (size_t i = 1; i < coords.size(); ++i) {
        QRegion translated = TranslateRegion(region, -coords[i].x, -coords[i].y);
        result = IntersectRegions(result, translated);
        if (result.Empty()) break;  // Early exit if empty
    }

    return result;
}

QRegion DilateRect(const QRegion& region, int32_t width, int32_t height) {
    if (width <= 0 || height <= 0) return region;

    // For separable rectangular SE, do 1D dilation in each direction
    // This is more efficient: O(w+h) instead of O(w*h)

    // Horizontal dilation
    QRegion result = region;
    int32_t halfW = width / 2;
    for (int32_t dx = -halfW; dx <= halfW; ++dx) {
        if (dx != 0) {
            result = UnionRegions(result, TranslateRegion(region, dx, 0));
        }
    }

    // Vertical dilation
    QRegion temp = result;
    int32_t halfH = height / 2;
    for (int32_t dy = -halfH; dy <= halfH; ++dy) {
        if (dy != 0) {
            result = UnionRegions(result, TranslateRegion(temp, 0, dy));
        }
    }

    return result;
}

QRegion ErodeRect(const QRegion& region, int32_t width, int32_t height) {
    if (width <= 0 || height <= 0) return region;

    // Separable erosion
    // Horizontal erosion
    QRegion result = region;
    int32_t halfW = width / 2;
    for (int32_t dx = -halfW; dx <= halfW; ++dx) {
        result = IntersectRegions(result, TranslateRegion(region, dx, 0));
        if (result.Empty()) return QRegion();
    }

    // Vertical erosion
    QRegion temp = result;
    int32_t halfH = height / 2;
    for (int32_t dy = -halfH; dy <= halfH; ++dy) {
        result = IntersectRegions(result, TranslateRegion(temp, 0, dy));
        if (result.Empty()) return QRegion();
    }

    return result;
}

QRegion DilateCircle(const QRegion& region, int32_t radius) {
    return Dilate(region, StructElement::Circle(radius));
}

QRegion ErodeCircle(const QRegion& region, int32_t radius) {
    return Erode(region, StructElement::Circle(radius));
}

// =============================================================================
// Compound Operations
// =============================================================================

QRegion Opening(const QRegion& region, const StructElement& se) {
    // Opening = Dilate(Erode(region, se), se)
    QRegion eroded = Erode(region, se);
    return Dilate(eroded, se);
}

QRegion Closing(const QRegion& region, const StructElement& se) {
    // Closing = Erode(Dilate(region, se), se)
    QRegion dilated = Dilate(region, se);
    return Erode(dilated, se);
}

QRegion OpeningRect(const QRegion& region, int32_t width, int32_t height) {
    QRegion eroded = ErodeRect(region, width, height);
    return DilateRect(eroded, width, height);
}

QRegion ClosingRect(const QRegion& region, int32_t width, int32_t height) {
    QRegion dilated = DilateRect(region, width, height);
    return ErodeRect(dilated, width, height);
}

QRegion OpeningCircle(const QRegion& region, int32_t radius) {
    auto se = StructElement::Circle(radius);
    return Opening(region, se);
}

QRegion ClosingCircle(const QRegion& region, int32_t radius) {
    auto se = StructElement::Circle(radius);
    return Closing(region, se);
}

// =============================================================================
// Derived Operations
// =============================================================================

QRegion MorphGradient(const QRegion& region, const StructElement& se) {
    // Gradient = Dilate - Erode
    QRegion dilated = Dilate(region, se);
    QRegion eroded = Erode(region, se);
    return DifferenceRegions(dilated, eroded);
}

QRegion InternalGradient(const QRegion& region, const StructElement& se) {
    // Internal = region - Erode
    QRegion eroded = Erode(region, se);
    return DifferenceRegions(region, eroded);
}

QRegion ExternalGradient(const QRegion& region, const StructElement& se) {
    // External = Dilate - region
    QRegion dilated = Dilate(region, se);
    return DifferenceRegions(dilated, region);
}

QRegion TopHat(const QRegion& region, const StructElement& se) {
    // TopHat = region - Opening
    QRegion opened = Opening(region, se);
    return DifferenceRegions(region, opened);
}

QRegion BlackHat(const QRegion& region, const StructElement& se) {
    // BlackHat = Closing - region
    QRegion closed = Closing(region, se);
    return DifferenceRegions(closed, region);
}

// =============================================================================
// Hit-or-Miss Transform
// =============================================================================

QRegion HitOrMiss(const QRegion& region,
                  const StructElement& hit,
                  const StructElement& miss,
                  const Rect2i& bounds) {
    // HitOrMiss = Erode(region, hit) ∩ Erode(complement(region), miss)
    QRegion hitResult = Erode(region, hit);
    if (hitResult.Empty()) return QRegion();

    QRegion complement = ComplementRegion(region, bounds);
    QRegion missResult = Erode(complement, miss);

    return IntersectRegions(hitResult, missResult);
}

QRegion HitOrMiss(const QRegion& region,
                  const std::pair<StructElement, StructElement>& sePair,
                  const Rect2i& bounds) {
    return HitOrMiss(region, sePair.first, sePair.second, bounds);
}

// =============================================================================
// Thinning and Skeleton
// =============================================================================

QRegion ThinOnce(const QRegion& region,
                 const StructElement& hit,
                 const StructElement& miss,
                 const Rect2i& bounds) {
    // Thin = region - HitOrMiss(region, hit, miss)
    QRegion matched = HitOrMiss(region, hit, miss, bounds);
    return DifferenceRegions(region, matched);
}

// Standard thinning structuring elements (8 rotations)
namespace {

std::vector<std::pair<StructElement, StructElement>> GetThinningKernels() {
    std::vector<std::pair<StructElement, StructElement>> kernels;

    // 8 directional thinning kernels
    // Kernel 1: North
    kernels.push_back({
        StructElement::FromCoordinates({{0, 0}, {-1, -1}, {-1, 0}, {-1, 1}}),
        StructElement::FromCoordinates({{1, -1}, {1, 0}, {1, 1}})
    });

    // Kernel 2: Northeast
    kernels.push_back({
        StructElement::FromCoordinates({{0, 0}, {-1, 0}, {-1, 1}, {0, 1}}),
        StructElement::FromCoordinates({{1, 0}, {1, -1}, {0, -1}})
    });

    // Kernel 3: East
    kernels.push_back({
        StructElement::FromCoordinates({{0, 0}, {-1, 1}, {0, 1}, {1, 1}}),
        StructElement::FromCoordinates({{-1, -1}, {0, -1}, {1, -1}})
    });

    // Kernel 4: Southeast
    kernels.push_back({
        StructElement::FromCoordinates({{0, 0}, {0, 1}, {1, 1}, {1, 0}}),
        StructElement::FromCoordinates({{-1, 0}, {-1, -1}, {0, -1}})
    });

    // Kernel 5: South
    kernels.push_back({
        StructElement::FromCoordinates({{0, 0}, {1, -1}, {1, 0}, {1, 1}}),
        StructElement::FromCoordinates({{-1, -1}, {-1, 0}, {-1, 1}})
    });

    // Kernel 6: Southwest
    kernels.push_back({
        StructElement::FromCoordinates({{0, 0}, {1, 0}, {1, -1}, {0, -1}}),
        StructElement::FromCoordinates({{-1, 0}, {-1, 1}, {0, 1}})
    });

    // Kernel 7: West
    kernels.push_back({
        StructElement::FromCoordinates({{0, 0}, {-1, -1}, {0, -1}, {1, -1}}),
        StructElement::FromCoordinates({{-1, 1}, {0, 1}, {1, 1}})
    });

    // Kernel 8: Northwest
    kernels.push_back({
        StructElement::FromCoordinates({{0, 0}, {-1, 0}, {-1, -1}, {0, -1}}),
        StructElement::FromCoordinates({{1, 0}, {1, 1}, {0, 1}})
    });

    return kernels;
}

} // anonymous namespace

QRegion Thin(const QRegion& region, int maxIterations) {
    if (region.Empty()) return region;

    Rect2i bounds = region.BoundingBox();
    // Expand bounds a bit
    bounds.x -= 1;
    bounds.y -= 1;
    bounds.width += 2;
    bounds.height += 2;

    auto kernels = GetThinningKernels();

    QRegion result = region;
    int iteration = 0;

    while (maxIterations == 0 || iteration < maxIterations) {
        QRegion prev = result;

        // Apply all 8 thinning kernels
        for (const auto& kernel : kernels) {
            result = ThinOnce(result, kernel.first, kernel.second, bounds);
        }

        // Check for convergence
        if (result.Area() == prev.Area()) {
            break;
        }

        ++iteration;
    }

    return result;
}

QRegion ThickenOnce(const QRegion& region,
                    const StructElement& hit,
                    const StructElement& miss,
                    const Rect2i& bounds) {
    // Thicken = region ∪ HitOrMiss(complement(region), hit, miss)
    QRegion complement = ComplementRegion(region, bounds);
    QRegion matched = HitOrMiss(complement, hit, miss, bounds);
    return UnionRegions(region, matched);
}

QRegion Skeleton(const QRegion& region) {
    if (region.Empty()) return region;

    // Skeleton using morphological thinning
    // Alternative: Skeleton = Union of (Erode^n - Open(Erode^n)) for all n

    // Simple approach: just thin until stable
    return Thin(region, 0);
}

QRegion PruneSkeleton(const QRegion& skeleton, int iterations) {
    if (skeleton.Empty() || iterations <= 0) return skeleton;

    Rect2i bounds = skeleton.BoundingBox();
    bounds.x -= 1;
    bounds.y -= 1;
    bounds.width += 2;
    bounds.height += 2;

    // Pruning kernels (remove endpoints)
    std::vector<std::pair<StructElement, StructElement>> pruneKernels;

    // Endpoint patterns (pixel with only one neighbor)
    pruneKernels.push_back({
        StructElement::FromCoordinates({{0, 0}, {0, 1}}),
        StructElement::FromCoordinates({{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}})
    });

    QRegion result = skeleton;
    for (int i = 0; i < iterations; ++i) {
        for (const auto& kernel : pruneKernels) {
            result = ThinOnce(result, kernel.first, kernel.second, bounds);
        }
    }

    return result;
}

// =============================================================================
// Iterative Operations
// =============================================================================

QRegion DilateN(const QRegion& region, const StructElement& se, int iterations) {
    QRegion result = region;
    for (int i = 0; i < iterations; ++i) {
        result = Dilate(result, se);
    }
    return result;
}

QRegion ErodeN(const QRegion& region, const StructElement& se, int iterations) {
    QRegion result = region;
    for (int i = 0; i < iterations; ++i) {
        result = Erode(result, se);
        if (result.Empty()) break;
    }
    return result;
}

QRegion OpeningN(const QRegion& region, const StructElement& se, int iterations) {
    QRegion result = region;
    for (int i = 0; i < iterations; ++i) {
        result = Opening(result, se);
    }
    return result;
}

QRegion ClosingN(const QRegion& region, const StructElement& se, int iterations) {
    QRegion result = region;
    for (int i = 0; i < iterations; ++i) {
        result = Closing(result, se);
    }
    return result;
}

// =============================================================================
// Geodesic Operations
// =============================================================================

QRegion GeodesicDilate(const QRegion& marker,
                       const QRegion& mask,
                       const StructElement& se) {
    // GeodesicDilate = Dilate(marker, se) ∩ mask
    QRegion dilated = Dilate(marker, se);
    return IntersectRegions(dilated, mask);
}

QRegion GeodesicErode(const QRegion& marker,
                      const QRegion& mask,
                      const StructElement& se) {
    // GeodesicErode = Erode(marker, se) ∪ complement(mask)
    // Or equivalently: result is largest subset of eroded that stays within mask
    QRegion eroded = Erode(marker, se);
    return UnionRegions(eroded, mask);
}

QRegion ReconstructByDilation(const QRegion& marker, const QRegion& mask) {
    if (marker.Empty() || mask.Empty()) return QRegion();

    auto se = StructElement::Square(3);  // 3x3 connectivity

    QRegion result = IntersectRegions(marker, mask);  // Ensure marker ⊆ mask
    QRegion prev;

    // Iterate until stable
    int maxIter = 10000;  // Safety limit
    for (int i = 0; i < maxIter; ++i) {
        prev = result;
        result = GeodesicDilate(result, mask, se);

        if (result.Area() == prev.Area()) {
            break;
        }
    }

    return result;
}

QRegion ReconstructByErosion(const QRegion& marker, const QRegion& mask) {
    if (marker.Empty()) return QRegion();

    auto se = StructElement::Square(3);

    QRegion result = UnionRegions(marker, mask);  // Ensure marker ⊇ mask
    QRegion prev;

    int maxIter = 10000;
    for (int i = 0; i < maxIter; ++i) {
        prev = result;
        result = GeodesicErode(result, mask, se);

        if (result.Area() == prev.Area()) {
            break;
        }
    }

    return result;
}

QRegion FillHolesByReconstruction(const QRegion& region) {
    if (region.Empty()) return region;

    Rect2i bbox = region.BoundingBox();
    // Expand bounds
    Rect2i bounds(bbox.x - 1, bbox.y - 1, bbox.width + 2, bbox.height + 2);

    // Create marker from border
    QRegion border = ComplementRegion(region, bounds);

    // Reconstruct from border - this fills holes
    QRegion filled = ReconstructByDilation(border, ComplementRegion(region, bounds));

    // Result is complement of reconstruction
    return ComplementRegion(filled, bounds);
}

QRegion ClearBorder(const QRegion& region, const Rect2i& bounds) {
    if (region.Empty()) return region;

    // Create border marker (1-pixel wide frame)
    std::vector<QRegion::Run> borderRuns;

    // Top edge
    borderRuns.emplace_back(bounds.y, bounds.x, bounds.x + bounds.width);
    // Bottom edge
    borderRuns.emplace_back(bounds.y + bounds.height - 1, bounds.x, bounds.x + bounds.width);
    // Left and right edges
    for (int32_t r = bounds.y + 1; r < bounds.y + bounds.height - 1; ++r) {
        borderRuns.emplace_back(r, bounds.x, bounds.x + 1);
        borderRuns.emplace_back(r, bounds.x + bounds.width - 1, bounds.x + bounds.width);
    }

    QRegion borderMarker(borderRuns);

    // Intersect with region to get border-touching seed
    QRegion seed = IntersectRegions(region, borderMarker);
    if (seed.Empty()) return region;

    // Reconstruct border-touching components
    QRegion borderComponents = ReconstructByDilation(seed, region);

    // Remove border components from region
    return DifferenceRegions(region, borderComponents);
}

} // namespace Qi::Vision::Internal
