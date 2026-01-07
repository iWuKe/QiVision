#pragma once

/**
 * @file QContourArray.h
 * @brief Container for multiple XLD contours with hierarchy management
 *
 * QContourArray manages multiple QContour objects and maintains their
 * hierarchical relationships (parent-child for holes).
 */

#include <QiVision/Core/QContour.h>
#include <vector>
#include <functional>

namespace Qi::Vision {

// Forward declarations
class QMatrix;

/**
 * @brief Container for multiple XLD contours with hierarchy support
 *
 * Manages a collection of contours extracted from an image, including:
 * - Outer contours
 * - Holes (inner contours, children of outer contours)
 * - Nested hierarchies
 */
class QContourArray {
public:
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Default constructor (empty array)
    QContourArray();

    /// Construct with reserved capacity
    explicit QContourArray(size_t capacity);

    /// Construct from a single contour
    explicit QContourArray(const QContour& contour);

    /// Construct from vector of contours
    explicit QContourArray(const std::vector<QContour>& contours);

    /// Move construct from vector of contours
    explicit QContourArray(std::vector<QContour>&& contours);

    // =========================================================================
    // Container Operations
    // =========================================================================

    /// Number of contours
    size_t Size() const { return contours_.size(); }

    /// Check if empty
    bool Empty() const { return contours_.empty(); }

    /// Access contour by index
    const QContour& At(size_t index) const;
    QContour& At(size_t index);

    /// Operator[] access
    const QContour& operator[](size_t index) const { return contours_[index]; }
    QContour& operator[](size_t index) { return contours_[index]; }

    /// Get all contours
    const std::vector<QContour>& GetContours() const { return contours_; }
    std::vector<QContour>& GetContours() { return contours_; }

    /// First contour (throws if empty)
    const QContour& Front() const;
    QContour& Front();

    /// Last contour (throws if empty)
    const QContour& Back() const;
    QContour& Back();

    // =========================================================================
    // Modification
    // =========================================================================

    /// Add a contour
    void Add(const QContour& contour);
    void Add(QContour&& contour);

    /// Add multiple contours
    void Add(const QContourArray& other);

    /// Insert contour at index
    void Insert(size_t index, const QContour& contour);

    /// Remove contour at index
    void Remove(size_t index);

    /// Remove contours matching predicate
    void RemoveIf(std::function<bool(const QContour&)> predicate);

    /// Clear all contours
    void Clear();

    /// Reserve capacity
    void Reserve(size_t capacity);

    // =========================================================================
    // Hierarchy Management
    // =========================================================================

    /// Build hierarchy from contour containment
    /// (outer contours contain inner contours as children)
    void BuildHierarchy();

    /// Get root contours (no parent)
    std::vector<size_t> GetRootContours() const;

    /// Get children of a contour
    std::vector<size_t> GetChildren(size_t index) const;

    /// Get parent of a contour (-1 if none)
    int32_t GetParent(size_t index) const;

    /// Get depth in hierarchy (0 = root)
    int GetDepth(size_t index) const;

    /// Flatten hierarchy (set all parents to -1)
    void FlattenHierarchy();

    // =========================================================================
    // Selection / Filtering
    // =========================================================================

    /// Select contours by length
    QContourArray SelectByLength(double minLength, double maxLength = 1e9) const;

    /// Select contours by area
    QContourArray SelectByArea(double minArea, double maxArea = 1e9) const;

    /// Select contours by circularity
    QContourArray SelectByCircularity(double minCirc, double maxCirc = 1.0) const;

    /// Select closed contours only
    QContourArray SelectClosed() const;

    /// Select open contours only
    QContourArray SelectOpen() const;

    /// Select root contours (no parent)
    QContourArray SelectRoots() const;

    /// Select contours by custom predicate
    QContourArray Select(std::function<bool(const QContour&)> predicate) const;

    /// Select contours by index
    QContourArray SelectByIndex(const std::vector<size_t>& indices) const;

    // =========================================================================
    // Geometric Properties (Aggregate)
    // =========================================================================

    /// Total length of all contours
    double TotalLength() const;

    /// Total area of all closed contours
    double TotalArea() const;

    /// Bounding box of all contours
    Rect2d BoundingBox() const;

    /// Centroid of all contour points
    Point2d Centroid() const;

    // =========================================================================
    // Transformations (Apply to All)
    // =========================================================================

    /// Translate all contours
    QContourArray Translate(double dx, double dy) const;
    QContourArray Translate(const Point2d& offset) const;

    /// Scale all contours
    QContourArray Scale(double factor) const;
    QContourArray Scale(double sx, double sy) const;
    QContourArray Scale(double sx, double sy, const Point2d& center) const;

    /// Rotate all contours
    QContourArray Rotate(double angle) const;
    QContourArray Rotate(double angle, const Point2d& center) const;

    /// Apply transformation matrix to all contours
    QContourArray Transform(const QMatrix& matrix) const;

    // =========================================================================
    // Processing (Apply to All)
    // =========================================================================

    /// Smooth all contours
    QContourArray Smooth(double sigma) const;

    /// Simplify all contours
    QContourArray Simplify(double tolerance) const;

    /// Resample all contours
    QContourArray Resample(double interval) const;

    /// Close all contours
    void CloseAll();

    /// Open all contours
    void OpenAll();

    /// Reverse all contours
    void ReverseAll();

    // =========================================================================
    // Merging / Splitting
    // =========================================================================

    /// Concatenate all contours into a single contour
    /// (useful for drawing, but loses hierarchy)
    QContour Concatenate() const;

    /// Split contours at high curvature points
    QContourArray SplitAtCorners(double maxCurvature) const;

    // =========================================================================
    // Iterators
    // =========================================================================

    using iterator = std::vector<QContour>::iterator;
    using const_iterator = std::vector<QContour>::const_iterator;

    iterator begin() { return contours_.begin(); }
    iterator end() { return contours_.end(); }
    const_iterator begin() const { return contours_.begin(); }
    const_iterator end() const { return contours_.end(); }
    const_iterator cbegin() const { return contours_.cbegin(); }
    const_iterator cend() const { return contours_.cend(); }

    // =========================================================================
    // Utilities
    // =========================================================================

    /// Deep copy
    QContourArray Clone() const;

    /// Sort contours by length (descending)
    void SortByLength(bool descending = true);

    /// Sort contours by area (descending)
    void SortByArea(bool descending = true);

    /// Sort contours by position (top-left first)
    void SortByPosition();

private:
    std::vector<QContour> contours_;
};

// =============================================================================
// Type Aliases
// =============================================================================

/// Alias for XLD contour array
using QXldArray = QContourArray;

} // namespace Qi::Vision
