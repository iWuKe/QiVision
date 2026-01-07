#pragma once

/**
 * @file QImage.h
 * @brief Image class with Domain support
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>

#include <memory>
#include <string>

namespace Qi::Vision {

// Forward declaration
class QRegion;

/**
 * @brief Image class with Halcon-style Domain support
 *
 * Key features:
 * - Domain support for ROI processing
 * - Multiple pixel types (UInt8, UInt16, Float32)
 * - 64-byte row alignment for SIMD
 * - Shallow copy by default, Clone() for deep copy
 */
class QImage {
public:
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Default constructor (empty image)
    QImage();

    /// Create image with specified dimensions and type
    QImage(int32_t width, int32_t height,
           PixelType type = PixelType::UInt8,
           ChannelType channels = ChannelType::Gray);

    /// Copy constructor (shallow copy)
    QImage(const QImage& other);

    /// Move constructor
    QImage(QImage&& other) noexcept;

    /// Destructor
    ~QImage();

    /// Copy assignment (shallow copy)
    QImage& operator=(const QImage& other);

    /// Move assignment
    QImage& operator=(QImage&& other) noexcept;

    // =========================================================================
    // Factory Methods
    // =========================================================================

    /// Load image from file
    static QImage FromFile(const std::string& path);

    /// Create from raw data (copies data)
    static QImage FromData(const void* data, int32_t width, int32_t height,
                          PixelType type = PixelType::UInt8,
                          ChannelType channels = ChannelType::Gray);

    // =========================================================================
    // Basic Properties
    // =========================================================================

    /// Image width in pixels
    int32_t Width() const;

    /// Image height in pixels
    int32_t Height() const;

    /// Number of channels
    int Channels() const;

    /// Pixel type
    PixelType Type() const;

    /// Channel type
    ChannelType GetChannelType() const;

    /// Row stride in bytes (includes alignment padding)
    size_t Stride() const;

    /// Check if image is empty
    bool Empty() const;

    /// Check if image is valid (allocated)
    bool IsValid() const;

    // =========================================================================
    // Data Access
    // =========================================================================

    /// Get pointer to raw data
    void* Data();
    const void* Data() const;

    /// Get pointer to specific row
    void* RowPtr(int32_t row);
    const void* RowPtr(int32_t row) const;

    /// Get pixel value at (x, y) - for single channel UInt8
    uint8_t At(int32_t x, int32_t y) const;

    /// Set pixel value at (x, y) - for single channel UInt8
    void SetAt(int32_t x, int32_t y, uint8_t value);

    // =========================================================================
    // Domain Operations
    // =========================================================================

    /// Check if domain covers full image
    bool IsFullDomain() const;

    /// Get current domain
    const QRegion* GetDomain() const;

    /// Set domain (replaces current)
    void SetDomain(const QRegion& region);

    /// Reduce domain (intersect with current)
    void ReduceDomain(const QRegion& region);

    /// Reset to full domain
    void ResetDomain();

    /// Get bounding box of domain
    Rect2i GetDomainBoundingBox() const;

    // =========================================================================
    // Image Operations
    // =========================================================================

    /// Deep copy
    QImage Clone() const;

    /// Create sub-image (shares data, no copy)
    QImage SubImage(int32_t x, int32_t y, int32_t width, int32_t height) const;

    /// Save image to file
    bool SaveToFile(const std::string& path) const;

    /// Convert to different pixel type
    QImage ConvertTo(PixelType targetType) const;

    /// Convert to grayscale
    QImage ToGray() const;

    // =========================================================================
    // Metadata
    // =========================================================================

    /// Pixel size in physical units (mm/pixel), 0 if not set
    double PixelSizeX() const;
    double PixelSizeY() const;
    void SetPixelSize(double sizeX, double sizeY);

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace Qi::Vision
