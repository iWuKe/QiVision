#pragma once

/**
 * @file StructElement.h
 * @brief Structuring elements for morphological operations
 *
 * This module provides:
 * - Predefined structuring elements (rectangle, ellipse, cross, diamond)
 * - Custom structuring elements from regions or masks
 * - Structuring element transformations (rotate, scale, reflect)
 * - Decomposition for efficient morphology
 *
 * Reference Halcon operators:
 * - gen_rectangle1, gen_ellipse, gen_cross, gen_star
 * - gen_struct_elem, disp_struct_elem
 * - transpose_struct_elem, reflect_struct_elem
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QRegion.h>
#include <QiVision/Core/QImage.h>

#include <vector>
#include <memory>

namespace Qi::Vision::Internal {

// =============================================================================
// Types
// =============================================================================

/// Structuring element shape type
enum class StructElementShape {
    Rectangle,      ///< Rectangular shape
    Ellipse,        ///< Elliptical shape
    Cross,          ///< Cross (plus sign) shape
    Diamond,        ///< Diamond (rhombus) shape
    Line,           ///< Line segment shape
    Octagon,        ///< Octagonal shape
    Custom          ///< Custom user-defined shape
};

/**
 * @brief Structuring element for morphological operations
 *
 * Stores a binary mask representing the structuring element shape.
 * The origin (anchor point) is at the center by default.
 */
class StructElement {
public:
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Default constructor (empty structuring element)
    StructElement();

    /// Copy constructor
    StructElement(const StructElement& other);

    /// Move constructor
    StructElement(StructElement&& other) noexcept;

    /// Destructor
    ~StructElement();

    /// Copy assignment
    StructElement& operator=(const StructElement& other);

    /// Move assignment
    StructElement& operator=(StructElement&& other) noexcept;

    // =========================================================================
    // Factory Methods - Basic Shapes
    // =========================================================================

    /**
     * @brief Create rectangular structuring element
     *
     * @param width Width (must be odd for symmetric origin)
     * @param height Height (must be odd for symmetric origin)
     * @return Rectangular structuring element
     */
    static StructElement Rectangle(int32_t width, int32_t height);

    /**
     * @brief Create square structuring element
     *
     * @param size Side length (must be odd)
     * @return Square structuring element
     */
    static StructElement Square(int32_t size);

    /**
     * @brief Create elliptical structuring element
     *
     * @param radiusX Horizontal radius
     * @param radiusY Vertical radius
     * @return Elliptical structuring element
     */
    static StructElement Ellipse(int32_t radiusX, int32_t radiusY);

    /**
     * @brief Create circular structuring element
     *
     * @param radius Circle radius
     * @return Circular structuring element
     */
    static StructElement Circle(int32_t radius);

    /**
     * @brief Create cross-shaped structuring element
     *
     * @param armLength Length of each arm from center
     * @param thickness Arm thickness
     * @return Cross-shaped structuring element
     */
    static StructElement Cross(int32_t armLength, int32_t thickness = 1);

    /**
     * @brief Create diamond-shaped structuring element
     *
     * @param radius Distance from center to vertex
     * @return Diamond structuring element
     */
    static StructElement Diamond(int32_t radius);

    /**
     * @brief Create line structuring element
     *
     * @param length Line length
     * @param angle Line angle in radians (0 = horizontal)
     * @return Line structuring element
     */
    static StructElement Line(int32_t length, double angle);

    /**
     * @brief Create octagonal structuring element
     *
     * @param radius Approximate radius
     * @return Octagonal structuring element
     */
    static StructElement Octagon(int32_t radius);

    // =========================================================================
    // Factory Methods - From Data
    // =========================================================================

    /**
     * @brief Create structuring element from binary mask
     *
     * @param mask Binary image (non-zero = part of element)
     * @param anchorX X coordinate of anchor (origin)
     * @param anchorY Y coordinate of anchor (origin)
     * @return Custom structuring element
     */
    static StructElement FromMask(const QImage& mask,
                                   int32_t anchorX = -1,
                                   int32_t anchorY = -1);

    /**
     * @brief Create structuring element from region
     *
     * @param region Input region
     * @param anchorX X coordinate of anchor
     * @param anchorY Y coordinate of anchor
     * @return Custom structuring element
     */
    static StructElement FromRegion(const QRegion& region,
                                     int32_t anchorX = -1,
                                     int32_t anchorY = -1);

    /**
     * @brief Create structuring element from coordinate list
     *
     * @param coords Vector of (row, col) offsets from anchor
     * @return Custom structuring element
     */
    static StructElement FromCoordinates(const std::vector<Point2i>& coords);

    // =========================================================================
    // Properties
    // =========================================================================

    /// Check if structuring element is empty
    bool Empty() const;

    /// Get width
    int32_t Width() const;

    /// Get height
    int32_t Height() const;

    /// Get size (width, height)
    Size2i Size() const;

    /// Get anchor X coordinate (relative to top-left)
    int32_t AnchorX() const;

    /// Get anchor Y coordinate (relative to top-left)
    int32_t AnchorY() const;

    /// Get anchor point
    Point2i Anchor() const;

    /// Get shape type
    StructElementShape Shape() const;

    /// Get number of pixels in element
    size_t PixelCount() const;

    /// Check if element is separable (for optimization)
    bool IsSeparable() const;

    /// Check if element is symmetric around anchor
    bool IsSymmetric() const;

    // =========================================================================
    // Data Access
    // =========================================================================

    /**
     * @brief Check if coordinate is part of element
     *
     * @param row Row relative to anchor
     * @param col Column relative to anchor
     * @return True if (row, col) is in element
     */
    bool Contains(int32_t row, int32_t col) const;

    /**
     * @brief Get all coordinates relative to anchor
     *
     * @return Vector of (row, col) offsets
     */
    std::vector<Point2i> GetCoordinates() const;

    /**
     * @brief Get binary mask
     *
     * @return Binary image of structuring element
     */
    QImage GetMask() const;

    /**
     * @brief Get as region
     *
     * @return Region representation
     */
    QRegion GetRegion() const;

    // =========================================================================
    // Transformations
    // =========================================================================

    /**
     * @brief Reflect structuring element (180 degree rotation)
     *
     * Required for dilation: dilation = erosion with reflected SE
     *
     * @return Reflected structuring element
     */
    StructElement Reflect() const;

    /**
     * @brief Transpose structuring element
     *
     * @return Transposed structuring element
     */
    StructElement Transpose() const;

    /**
     * @brief Rotate structuring element
     *
     * @param angle Rotation angle in radians
     * @return Rotated structuring element
     */
    StructElement Rotate(double angle) const;

    /**
     * @brief Scale structuring element
     *
     * @param scaleX Horizontal scale factor
     * @param scaleY Vertical scale factor
     * @return Scaled structuring element
     */
    StructElement Scale(double scaleX, double scaleY) const;

    // =========================================================================
    // Decomposition
    // =========================================================================

    /**
     * @brief Check if element can be decomposed into separable components
     *
     * @return True if separable decomposition exists
     */
    bool CanDecompose() const;

    /**
     * @brief Decompose into horizontal and vertical components
     *
     * For separable elements like rectangles.
     *
     * @param horizontal Output horizontal component
     * @param vertical Output vertical component
     * @return True if decomposition successful
     */
    bool Decompose(StructElement& horizontal, StructElement& vertical) const;

    /**
     * @brief Decompose into sequence of smaller elements
     *
     * For faster morphology via iterated operations.
     *
     * @return Vector of smaller structuring elements
     */
    std::vector<StructElement> DecomposeToSequence() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * @brief Create 3x3 cross structuring element
 *
 * Commonly used for 4-connected operations.
 *
 * @return 3x3 cross element
 */
StructElement SE_Cross3();

/**
 * @brief Create 3x3 square structuring element
 *
 * Commonly used for 8-connected operations.
 *
 * @return 3x3 square element
 */
StructElement SE_Square3();

/**
 * @brief Create 5x5 circle structuring element
 *
 * @return 5x5 disk element
 */
StructElement SE_Disk5();

/**
 * @brief Create structuring element for hit-or-miss transform
 *
 * @param hit Pixels that must be foreground
 * @param miss Pixels that must be background
 * @return Pair of structuring elements for hit-or-miss
 */
std::pair<StructElement, StructElement> CreateHitMissSE(
    const std::vector<Point2i>& hit,
    const std::vector<Point2i>& miss);

} // namespace Qi::Vision::Internal
