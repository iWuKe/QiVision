#pragma once

/**
 * @file QRegion.h
 * @brief RLE-encoded region class
 */

#include <QiVision/Core/Types.h>

#include <vector>
#include <memory>

namespace Qi::Vision {

/**
 * @brief Run-length encoded region
 *
 * Key features:
 * - Uses int32_t for coordinates (supports >32K resolution)
 * - Runs sorted by (row, colBegin)
 * - Adjacent/overlapping runs auto-merged
 * - Cached computed properties (Area, BoundingBox)
 * - Thread-safe caching with std::call_once
 */
class QRegion {
public:
    /**
     * @brief Single horizontal run
     * @note Uses int32_t to support high-resolution images (>32K)
     */
    struct Run {
        int32_t row;        ///< Row index
        int32_t colBegin;   ///< Start column (inclusive)
        int32_t colEnd;     ///< End column (exclusive)

        Run() : row(0), colBegin(0), colEnd(0) {}
        Run(int32_t r, int32_t cb, int32_t ce)
            : row(r), colBegin(cb), colEnd(ce) {}

        int32_t Length() const { return colEnd - colBegin; }
    };

    // =========================================================================
    // Constructors
    // =========================================================================

    /// Default constructor (empty region)
    QRegion();

    /// Create from rectangle
    QRegion(const Rect2i& rect);

    /// Create from runs
    QRegion(const std::vector<Run>& runs);

    /// Copy constructor
    QRegion(const QRegion& other);

    /// Move constructor
    QRegion(QRegion&& other) noexcept;

    /// Destructor
    ~QRegion();

    /// Copy assignment
    QRegion& operator=(const QRegion& other);

    /// Move assignment
    QRegion& operator=(QRegion&& other) noexcept;

    // =========================================================================
    // Factory Methods
    // =========================================================================

    /// Create rectangular region
    static QRegion Rectangle(int32_t x, int32_t y, int32_t width, int32_t height);

    /// Create circular region
    static QRegion Circle(int32_t cx, int32_t cy, int32_t radius);

    /// Create elliptical region
    static QRegion Ellipse(int32_t cx, int32_t cy, int32_t radiusX, int32_t radiusY);

    // =========================================================================
    // Properties
    // =========================================================================

    /// Check if region is empty
    bool Empty() const;

    /// Number of runs
    size_t RunCount() const;

    /// Area in pixels (cached)
    int64_t Area() const;

    /// Bounding box (cached)
    Rect2i BoundingBox() const;

    /// Centroid (cached)
    Point2d Centroid() const;

    // =========================================================================
    // Data Access
    // =========================================================================

    /// Get all runs
    const std::vector<Run>& Runs() const;

    /// Check if point is inside region
    bool Contains(int32_t x, int32_t y) const;
    bool Contains(const Point2i& point) const;

    // =========================================================================
    // Set Operations
    // =========================================================================

    /// Union of two regions
    QRegion Union(const QRegion& other) const;

    /// Intersection of two regions
    QRegion Intersection(const QRegion& other) const;

    /// Difference (this - other)
    QRegion Difference(const QRegion& other) const;

    /// Complement within bounding box
    QRegion Complement() const;

    /// Complement within specified rectangle
    QRegion Complement(const Rect2i& bounds) const;

    // =========================================================================
    // Morphological Operations
    // =========================================================================

    /// Dilate by rectangular structuring element
    QRegion Dilate(int32_t width, int32_t height) const;

    /// Erode by rectangular structuring element
    QRegion Erode(int32_t width, int32_t height) const;

    /// Opening (erode then dilate)
    QRegion Opening(int32_t width, int32_t height) const;

    /// Closing (dilate then erode)
    QRegion Closing(int32_t width, int32_t height) const;

    // =========================================================================
    // Transformations
    // =========================================================================

    /// Translate region
    QRegion Translate(int32_t dx, int32_t dy) const;

    /// Scale region (around origin)
    QRegion Scale(double sx, double sy) const;

    // =========================================================================
    // Operators
    // =========================================================================

    /// Union operator
    QRegion operator|(const QRegion& other) const { return Union(other); }

    /// Intersection operator
    QRegion operator&(const QRegion& other) const { return Intersection(other); }

    /// Difference operator
    QRegion operator-(const QRegion& other) const { return Difference(other); }

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace Qi::Vision
