#include <QiVision/Core/QRegion.h>
#include <QiVision/Core/Exception.h>

#include <algorithm>
#include <cmath>
#include <mutex>

namespace Qi::Vision {

// =============================================================================
// Implementation class
// =============================================================================

class QRegion::Impl {
public:
    std::vector<Run> runs_;

    // Cached values
    mutable int64_t cachedArea_ = -1;
    mutable Rect2i cachedBBox_;
    mutable bool bboxValid_ = false;
    mutable Point2d cachedCentroid_;
    mutable bool centroidValid_ = false;
    mutable std::once_flag areaOnce_;
    mutable std::once_flag bboxOnce_;
    mutable std::once_flag centroidOnce_;

    void InvalidateCache() {
        cachedArea_ = -1;
        bboxValid_ = false;
        centroidValid_ = false;
        // Reset once_flags by reconstructing
        new (&areaOnce_) std::once_flag();
        new (&bboxOnce_) std::once_flag();
        new (&centroidOnce_) std::once_flag();
    }

    void SortAndMerge() {
        if (runs_.empty()) return;

        // Sort by (row, colBegin)
        std::sort(runs_.begin(), runs_.end(),
            [](const Run& a, const Run& b) {
                if (a.row != b.row) return a.row < b.row;
                return a.colBegin < b.colBegin;
            });

        // Merge overlapping/adjacent runs
        std::vector<Run> merged;
        merged.reserve(runs_.size());
        merged.push_back(runs_[0]);

        for (size_t i = 1; i < runs_.size(); ++i) {
            Run& last = merged.back();
            const Run& curr = runs_[i];

            if (curr.row == last.row && curr.colBegin <= last.colEnd) {
                // Merge
                last.colEnd = std::max(last.colEnd, curr.colEnd);
            } else {
                merged.push_back(curr);
            }
        }

        runs_ = std::move(merged);
        InvalidateCache();
    }

    void ComputeArea() const {
        cachedArea_ = 0;
        for (const auto& run : runs_) {
            cachedArea_ += run.Length();
        }
    }

    void ComputeBBox() const {
        if (runs_.empty()) {
            cachedBBox_ = Rect2i();
            bboxValid_ = true;
            return;
        }

        int32_t minRow = runs_.front().row;
        int32_t maxRow = runs_.back().row;
        int32_t minCol = std::numeric_limits<int32_t>::max();
        int32_t maxCol = std::numeric_limits<int32_t>::min();

        for (const auto& run : runs_) {
            minCol = std::min(minCol, run.colBegin);
            maxCol = std::max(maxCol, run.colEnd);
        }

        cachedBBox_ = Rect2i(minCol, minRow, maxCol - minCol, maxRow - minRow + 1);
        bboxValid_ = true;
    }

    void ComputeCentroid() const {
        if (runs_.empty()) {
            cachedCentroid_ = Point2d();
            centroidValid_ = true;
            return;
        }

        double sumX = 0, sumY = 0;
        int64_t count = 0;

        for (const auto& run : runs_) {
            int32_t len = run.Length();
            double midX = (run.colBegin + run.colEnd - 1) / 2.0;
            sumX += midX * len;
            sumY += run.row * len;
            count += len;
        }

        if (count > 0) {
            cachedCentroid_ = Point2d(sumX / count, sumY / count);
        }
        centroidValid_ = true;
    }
};

// =============================================================================
// Constructors
// =============================================================================

QRegion::QRegion() : impl_(std::make_shared<Impl>()) {}

QRegion::QRegion(const Rect2i& rect) : impl_(std::make_shared<Impl>()) {
    if (!rect.IsValid()) {
        throw InvalidArgumentException("QRegion::QRegion: invalid rectangle");
    }
    if (rect.width > 0 && rect.height > 0) {
        impl_->runs_.reserve(rect.height);
        for (int32_t y = rect.y; y < rect.y + rect.height; ++y) {
            impl_->runs_.emplace_back(y, rect.x, rect.x + rect.width);
        }
    }
}

QRegion::QRegion(const std::vector<Run>& runs) : impl_(std::make_shared<Impl>()) {
    for (const auto& run : runs) {
        if (run.colEnd <= run.colBegin) {
            throw InvalidArgumentException("QRegion::QRegion: invalid run");
        }
    }
    impl_->runs_ = runs;
    impl_->SortAndMerge();
}

QRegion::QRegion(const QRegion& other) = default;
QRegion::QRegion(QRegion&& other) noexcept = default;
QRegion::~QRegion() = default;
QRegion& QRegion::operator=(const QRegion& other) = default;
QRegion& QRegion::operator=(QRegion&& other) noexcept = default;

// =============================================================================
// Factory Methods
// =============================================================================

QRegion QRegion::Rectangle(int32_t x, int32_t y, int32_t width, int32_t height) {
    if (width < 0 || height < 0) {
        throw InvalidArgumentException("QRegion::Rectangle: width/height must be >= 0");
    }
    return QRegion(Rect2i(x, y, width, height));
}

QRegion QRegion::Circle(int32_t cx, int32_t cy, int32_t radius) {
    QRegion region;
    if (radius < 0) {
        throw InvalidArgumentException("QRegion::Circle: radius must be >= 0");
    }
    if (radius == 0) return region;

    region.impl_->runs_.reserve(2 * radius + 1);

    for (int32_t dy = -radius; dy <= radius; ++dy) {
        int32_t dx = static_cast<int32_t>(
            std::sqrt(static_cast<double>(radius * radius - dy * dy)));
        if (dx >= 0) {
            region.impl_->runs_.emplace_back(cy + dy, cx - dx, cx + dx + 1);
        }
    }

    return region;
}

QRegion QRegion::Ellipse(int32_t cx, int32_t cy, int32_t radiusX, int32_t radiusY) {
    QRegion region;
    if (radiusX < 0 || radiusY < 0) {
        throw InvalidArgumentException("QRegion::Ellipse: radius must be >= 0");
    }
    if (radiusX == 0 || radiusY == 0) return region;

    region.impl_->runs_.reserve(2 * radiusY + 1);

    double ry2 = static_cast<double>(radiusY * radiusY);

    for (int32_t dy = -radiusY; dy <= radiusY; ++dy) {
        double factor = 1.0 - (dy * dy) / ry2;
        if (factor >= 0) {
            int32_t dx = static_cast<int32_t>(radiusX * std::sqrt(factor));
            region.impl_->runs_.emplace_back(cy + dy, cx - dx, cx + dx + 1);
        }
    }

    return region;
}

// =============================================================================
// Properties
// =============================================================================

bool QRegion::Empty() const {
    return impl_->runs_.empty();
}

size_t QRegion::RunCount() const {
    return impl_->runs_.size();
}

int64_t QRegion::Area() const {
    std::call_once(impl_->areaOnce_, [this]() {
        impl_->ComputeArea();
    });
    return impl_->cachedArea_;
}

Rect2i QRegion::BoundingBox() const {
    std::call_once(impl_->bboxOnce_, [this]() {
        impl_->ComputeBBox();
    });
    return impl_->cachedBBox_;
}

Point2d QRegion::Centroid() const {
    std::call_once(impl_->centroidOnce_, [this]() {
        impl_->ComputeCentroid();
    });
    return impl_->cachedCentroid_;
}

// =============================================================================
// Data Access
// =============================================================================

const std::vector<QRegion::Run>& QRegion::Runs() const {
    return impl_->runs_;
}

bool QRegion::Contains(int32_t x, int32_t y) const {
    // Binary search by row
    auto it = std::lower_bound(impl_->runs_.begin(), impl_->runs_.end(), y,
        [](const Run& run, int32_t row) { return run.row < row; });

    while (it != impl_->runs_.end() && it->row == y) {
        if (x >= it->colBegin && x < it->colEnd) {
            return true;
        }
        ++it;
    }
    return false;
}

bool QRegion::Contains(const Point2i& point) const {
    return Contains(point.x, point.y);
}

// =============================================================================
// Set Operations
// =============================================================================

QRegion QRegion::Union(const QRegion& other) const {
    std::vector<Run> combined;
    combined.reserve(impl_->runs_.size() + other.impl_->runs_.size());
    combined.insert(combined.end(), impl_->runs_.begin(), impl_->runs_.end());
    combined.insert(combined.end(), other.impl_->runs_.begin(), other.impl_->runs_.end());
    return QRegion(combined);
}

QRegion QRegion::Intersection(const QRegion& other) const {
    std::vector<Run> result;

    size_t i = 0, j = 0;
    while (i < impl_->runs_.size() && j < other.impl_->runs_.size()) {
        const Run& a = impl_->runs_[i];
        const Run& b = other.impl_->runs_[j];

        if (a.row < b.row) {
            ++i;
        } else if (a.row > b.row) {
            ++j;
        } else {
            // Same row - check column overlap
            int32_t start = std::max(a.colBegin, b.colBegin);
            int32_t end = std::min(a.colEnd, b.colEnd);

            if (start < end) {
                result.emplace_back(a.row, start, end);
            }

            if (a.colEnd <= b.colEnd) ++i;
            if (b.colEnd <= a.colEnd) ++j;
        }
    }

    return QRegion(result);
}

QRegion QRegion::Difference(const QRegion& other) const {
    // Simplified implementation - can be optimized
    std::vector<Run> result;

    for (const auto& run : impl_->runs_) {
        int32_t colStart = run.colBegin;

        for (const auto& otherRun : other.impl_->runs_) {
            if (otherRun.row != run.row) continue;
            if (otherRun.colEnd <= colStart) continue;
            if (otherRun.colBegin >= run.colEnd) break;

            if (otherRun.colBegin > colStart) {
                result.emplace_back(run.row, colStart,
                                   std::min(otherRun.colBegin, run.colEnd));
            }
            colStart = std::max(colStart, otherRun.colEnd);
        }

        if (colStart < run.colEnd) {
            result.emplace_back(run.row, colStart, run.colEnd);
        }
    }

    return QRegion(result);
}

QRegion QRegion::Complement() const {
    return Complement(BoundingBox());
}

QRegion QRegion::Complement(const Rect2i& bounds) const {
    if (!bounds.IsValid()) {
        throw InvalidArgumentException("QRegion::Complement: invalid bounds");
    }
    QRegion boundsRegion(bounds);
    return boundsRegion.Difference(*this);
}

// =============================================================================
// Morphological Operations (simplified implementations)
// =============================================================================

QRegion QRegion::Dilate(int32_t width, int32_t height) const {
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("QRegion::Dilate: width/height must be > 0");
    }
    std::vector<Run> result;
    int32_t halfW = width / 2;
    int32_t halfH = height / 2;

    for (const auto& run : impl_->runs_) {
        for (int32_t dy = -halfH; dy <= halfH; ++dy) {
            result.emplace_back(
                run.row + dy,
                run.colBegin - halfW,
                run.colEnd + halfW
            );
        }
    }

    return QRegion(result);
}

QRegion QRegion::Erode(int32_t width, int32_t height) const {
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("QRegion::Erode: width/height must be > 0");
    }
    // Simplified: erode = complement of dilate of complement
    // This is not optimal but correct
    Rect2i bbox = BoundingBox();
    Rect2i expandedBBox(bbox.x - width, bbox.y - height,
                        bbox.width + 2 * width, bbox.height + 2 * height);

    QRegion comp = Complement(expandedBBox);
    QRegion dilatedComp = comp.Dilate(width, height);
    return Difference(dilatedComp);
}

QRegion QRegion::Opening(int32_t width, int32_t height) const {
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("QRegion::Opening: width/height must be > 0");
    }
    return Erode(width, height).Dilate(width, height);
}

QRegion QRegion::Closing(int32_t width, int32_t height) const {
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("QRegion::Closing: width/height must be > 0");
    }
    return Dilate(width, height).Erode(width, height);
}

// =============================================================================
// Transformations
// =============================================================================

QRegion QRegion::Translate(int32_t dx, int32_t dy) const {
    std::vector<Run> result;
    result.reserve(impl_->runs_.size());

    for (const auto& run : impl_->runs_) {
        result.emplace_back(run.row + dy, run.colBegin + dx, run.colEnd + dx);
    }

    QRegion region;
    region.impl_->runs_ = std::move(result);
    return region;
}

QRegion QRegion::Scale(double sx, double sy) const {
    if (!std::isfinite(sx) || !std::isfinite(sy) || sx <= 0.0 || sy <= 0.0) {
        throw InvalidArgumentException("QRegion::Scale: scale must be > 0");
    }
    std::vector<Run> result;

    for (const auto& run : impl_->runs_) {
        int32_t newRow = static_cast<int32_t>(run.row * sy);
        int32_t newColBegin = static_cast<int32_t>(run.colBegin * sx);
        int32_t newColEnd = static_cast<int32_t>(run.colEnd * sx);

        if (newColEnd > newColBegin) {
            result.emplace_back(newRow, newColBegin, newColEnd);
        }
    }

    return QRegion(result);
}

} // namespace Qi::Vision
