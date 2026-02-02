#pragma once

/**
 * @file ImageIO.h
 * @brief Image I/O operations (Halcon-style API)
 *
 * Halcon reference operators:
 * - read_image, write_image
 * - read_sequence, write_sequence
 *
 * Supported formats: PNG, JPG/JPEG, BMP, TIFF, PGM/PPM, RAW
 * Supported types: UInt8, UInt16, Float32
 * Supported channels: Gray, RGB, RGBA
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Core/Export.h>

#include <cstdint>
#include <string>
#include <vector>

namespace Qi::Vision::IO {

// =============================================================================
// Image Write Flags (OpenCV-style vector<int> params)
// =============================================================================

/**
 * @brief Image write parameter flags (OpenCV imwrite style)
 *
 * Usage: WriteImage(img, "file.jpg", ImageFormat::Auto,
 *                   {QIWRITE_JPEG_QUALITY, 85, QIWRITE_PNG_COMPRESSION, 6});
 */
enum ImageWriteFlag {
    QIWRITE_JPEG_QUALITY = 1,        ///< JPEG quality [0-100], default 95
    QIWRITE_PNG_COMPRESSION = 2,     ///< PNG compression level [0-9], default 6
    QIWRITE_TIFF_COMPRESSION = 3     ///< TIFF LZW compression: 0=off, 1=on (default)
};

// =============================================================================
// Image Format Enumeration
// =============================================================================

/**
 * @brief Supported image file formats
 */
enum class ImageFormat {
    Auto,       ///< Auto-detect from extension
    PNG,        ///< PNG (lossless, supports 8/16-bit, alpha)
    JPEG,       ///< JPEG (lossy, 8-bit only)
    BMP,        ///< BMP (uncompressed, 8-bit)
    TIFF,       ///< TIFF (supports 8/16-bit, float)
    PGM,        ///< PGM/PNM (portable graymap)
    PPM,        ///< PPM (portable pixmap, RGB)
    RAW         ///< Raw binary data (requires dimensions)
};

// =============================================================================
// Parameter Structures
// =============================================================================

/**
 * @brief Image metadata structure
 */
struct QIVISION_API ImageMetadata {
    int32_t width = 0;
    int32_t height = 0;
    int32_t channels = 0;
    int32_t bitsPerChannel = 0;
    PixelType pixelType = PixelType::UInt8;
    ChannelType channelType = ChannelType::Gray;

    // Optional metadata
    double dpiX = 0.0;              ///< Horizontal DPI (0 = unknown)
    double dpiY = 0.0;              ///< Vertical DPI (0 = unknown)
    std::string colorProfile;       ///< Color profile name (if available)
    std::string description;        ///< Image description
    std::string software;           ///< Creating software
    std::string dateTime;           ///< Creation date/time
};

// =============================================================================
// Image Read Functions
// =============================================================================

/**
 * @brief Read image from file
 *
 * Equivalent to Halcon's read_image operator.
 *
 * @param filename Input file path
 * @param[out] image Loaded image
 * @throws IOException if file cannot be read
 *
 * @code
 * QImage img;
 * ReadImage("test.png", img);
 * @endcode
 */
QIVISION_API void ReadImage(const std::string& filename, QImage& image);

/**
 * @brief Read image with format hint
 *
 * @param filename Input file path
 * @param[out] image Loaded image
 * @param format Force specific format (Auto = detect from extension)
 */
QIVISION_API void ReadImage(const std::string& filename, QImage& image, ImageFormat format);

/**
 * @brief Read raw binary image data
 *
 * @param filename Input file path
 * @param[out] image Loaded image
 * @param width Image width (required)
 * @param height Image height (required)
 * @param pixelType Pixel type (default: UInt8)
 * @param channelType Channel type (default: Gray)
 * @param headerBytes Bytes to skip at file start (default: 0)
 * @param bigEndian Byte order for >8-bit data (default: false)
 */
QIVISION_API void ReadImageRaw(const std::string& filename, QImage& image,
                  int32_t width, int32_t height,
                  PixelType pixelType = PixelType::UInt8,
                  ChannelType channelType = ChannelType::Gray,
                  int32_t headerBytes = 0,
                  bool bigEndian = false);

/**
 * @brief Read image metadata without loading full image
 *
 * Useful for getting dimensions before loading large images.
 *
 * @param filename Input file path
 * @param metadata [out] Image metadata
 * @return true if metadata was successfully read
 */
QIVISION_API bool ReadImageMetadata(const std::string& filename, ImageMetadata& metadata);

/**
 * @brief Read image and convert to specified type
 *
 * @param filename Input file path
 * @param[out] image Loaded and converted image
 * @param targetType Target pixel type for conversion
 */
QIVISION_API void ReadImageAs(const std::string& filename, QImage& image, PixelType targetType);

/**
 * @brief Read image and convert to grayscale
 *
 * @param filename Input file path
 * @param[out] image Grayscale image
 */
QIVISION_API void ReadImageGray(const std::string& filename, QImage& image);

// =============================================================================
// Image Write Functions
// =============================================================================

/**
 * @brief Write image to file
 *
 * Equivalent to Halcon's write_image operator.
 *
 * @param image Image to write
 * @param filename Output file path
 * @return true if successful
 *
 * @code
 * WriteImage(img, "output.png");
 * WriteImage(img, "output.jpg");
 * @endcode
 */
QIVISION_API bool WriteImage(const QImage& image, const std::string& filename);

/**
 * @brief Write image with format and parameters (OpenCV imwrite style)
 *
 * @param image Image to write
 * @param filename Output file path
 * @param format Output format (Auto = detect from extension)
 * @param params Key-value pairs: {QIWRITE_JPEG_QUALITY, 85, ...}
 * @return true if successful
 *
 * @code
 * WriteImage(img, "out.jpg", ImageFormat::Auto, {QIWRITE_JPEG_QUALITY, 85});
 * WriteImage(img, "out.png", ImageFormat::PNG, {QIWRITE_PNG_COMPRESSION, 9});
 * @endcode
 */
QIVISION_API bool WriteImage(const QImage& image, const std::string& filename,
                ImageFormat format, const std::vector<int>& params);

/**
 * @brief Write image as raw binary data
 *
 * @param image Image to write
 * @param filename Output file path
 * @param bigEndian Byte order for >8-bit
 * @return true if successful
 */
QIVISION_API bool WriteImageRaw(const QImage& image, const std::string& filename,
                   bool bigEndian = false);

// =============================================================================
// Image Sequence Functions
// =============================================================================

/**
 * @brief Read image sequence (multiple files)
 *
 * Equivalent to Halcon's read_sequence operator.
 *
 * @param pattern File pattern with printf-style placeholder (e.g., "img_%03d.png")
 * @param[out] images Vector of loaded images
 * @param startIndex Starting index
 * @param endIndex Ending index (inclusive)
 * @param step Index step (default 1)
 *
 * @code
 * std::vector<QImage> images;
 * ReadSequence("frame_%04d.png", images, 0, 99);
 * @endcode
 */
QIVISION_API void ReadSequence(const std::string& pattern,
                  std::vector<QImage>& images,
                  int32_t startIndex,
                  int32_t endIndex,
                  int32_t step = 1);

/**
 * @brief Read all images from directory
 *
 * @param directory Directory path
 * @param[out] images Vector of loaded images
 * @param extensions Filter by extensions (empty = all supported)
 */
QIVISION_API void ReadDirectory(const std::string& directory,
                   std::vector<QImage>& images,
                   const std::vector<std::string>& extensions = {});

/**
 * @brief Write image sequence
 *
 * @param images Vector of images to write
 * @param pattern File pattern with printf-style placeholder
 * @param startIndex Starting index
 * @param params Optional key-value pairs: {QIWRITE_JPEG_QUALITY, 85, ...}
 * @return Number of images successfully written
 */
QIVISION_API int32_t WriteSequence(const std::vector<QImage>& images,
                       const std::string& pattern,
                       int32_t startIndex = 0,
                       const std::vector<int>& params = {});

// =============================================================================
// Format Utility Functions
// =============================================================================

/**
 * @brief Get image format from filename extension
 *
 * @param filename File path
 * @return Detected format (Auto if unknown)
 */
QIVISION_API ImageFormat GetFormatFromFilename(const std::string& filename);

/**
 * @brief Get file extension for format
 *
 * @param format Image format
 * @return Extension string (e.g., ".png")
 */
QIVISION_API std::string GetExtensionForFormat(ImageFormat format);

/**
 * @brief Check if format supports 16-bit depth
 */
QIVISION_API bool FormatSupports16Bit(ImageFormat format);

/**
 * @brief Check if format supports alpha channel
 */
QIVISION_API bool FormatSupportsAlpha(ImageFormat format);

/**
 * @brief Check if format is lossless
 */
QIVISION_API bool FormatIsLossless(ImageFormat format);

/**
 * @brief List supported file extensions
 *
 * @return Vector of supported extensions (e.g., {".png", ".jpg", ".bmp"})
 */
QIVISION_API std::vector<std::string> GetSupportedExtensions();

/**
 * @brief Check if file is a supported image format
 */
QIVISION_API bool IsSupportedImageFile(const std::string& filename);

} // namespace Qi::Vision::IO
