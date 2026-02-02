#pragma once

/**
 * @file Barcode.h
 * @brief Barcode and QR code reading (powered by ZXing-cpp)
 *
 * Supports:
 * - 1D: Code128, Code39, Code93, EAN-13, EAN-8, UPC-A, UPC-E, ITF, Codabar
 * - 2D: QR Code, Data Matrix, PDF417, Aztec
 *
 * API Style: QiVision style wrapping ZXing-cpp (Apache 2.0 license)
 */

#include <QiVision/Core/Export.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Types.h>

#include <string>
#include <vector>

namespace Qi::Vision::Barcode {

// =============================================================================
// Enums
// =============================================================================

/**
 * @brief Barcode format types
 */
enum class BarcodeFormat {
    None = 0,

    // 1D Barcodes
    Code128     = (1 << 0),
    Code39      = (1 << 1),
    Code93      = (1 << 2),
    Codabar     = (1 << 3),
    EAN8        = (1 << 4),
    EAN13       = (1 << 5),
    ITF         = (1 << 6),   // Interleaved 2 of 5
    UPCA        = (1 << 7),
    UPCE        = (1 << 8),

    // 2D Barcodes
    QRCode      = (1 << 9),
    DataMatrix  = (1 << 10),
    PDF417      = (1 << 11),
    Aztec       = (1 << 12),

    // Groups
    Linear      = Code128 | Code39 | Code93 | Codabar | EAN8 | EAN13 | ITF | UPCA | UPCE,
    Matrix      = QRCode | DataMatrix | PDF417 | Aztec,
    All         = Linear | Matrix
};

// Bitwise operators for BarcodeFormat
inline BarcodeFormat operator|(BarcodeFormat a, BarcodeFormat b) {
    return static_cast<BarcodeFormat>(static_cast<int>(a) | static_cast<int>(b));
}

inline BarcodeFormat operator&(BarcodeFormat a, BarcodeFormat b) {
    return static_cast<BarcodeFormat>(static_cast<int>(a) & static_cast<int>(b));
}

inline bool HasFormat(BarcodeFormat flags, BarcodeFormat format) {
    return (static_cast<int>(flags) & static_cast<int>(format)) != 0;
}

/**
 * @brief Binarization method for barcode detection
 */
enum class Binarizer {
    LocalAverage,   ///< Local average thresholding (default, good for most cases)
    GlobalHistogram,///< Global histogram-based (faster, less robust)
    FixedThreshold, ///< Fixed threshold (for high-contrast images)
    BoolCast        ///< Simple bool cast (for binary images)
};

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Single barcode detection result
 */
struct QIVISION_API BarcodeResult {
    bool valid = false;             ///< Whether decoding was successful
    BarcodeFormat format = BarcodeFormat::None;  ///< Detected barcode format
    std::string text;               ///< Decoded text content
    std::string formatName;         ///< Format name as string

    // Position information
    Point2d position;               ///< Center position
    double angle = 0.0;             ///< Rotation angle (radians)
    std::vector<Point2d> corners;   ///< Four corner points (clockwise from top-left)

    // Quality metrics
    int symbolVersion = 0;          ///< QR/DataMatrix version (0 if N/A)
    std::string ecLevel;            ///< Error correction level (if applicable)
    bool isMirrored = false;        ///< Whether the code was mirrored

    /// Check if result is valid
    explicit operator bool() const { return valid; }
};

/**
 * @brief Parameters for barcode reading
 */
struct QIVISION_API BarcodeParams {
    BarcodeFormat formats = BarcodeFormat::All;  ///< Formats to search for
    Binarizer binarizer = Binarizer::LocalAverage; ///< Binarization method
    bool tryHarder = false;         ///< Try harder to find barcodes (slower)
    bool tryRotate = true;          ///< Try rotated orientations
    bool tryInvert = false;         ///< Try inverted (white-on-black) codes
    bool tryDownscale = true;       ///< Try downscaled image for large codes
    int maxNumberOfSymbols = 1;     ///< Maximum number of codes to find (0=unlimited)
    int minLineCount = 2;           ///< Minimum line count for 1D codes

    /// Default parameters
    static BarcodeParams Default() { return BarcodeParams(); }

    /// Parameters optimized for QR codes
    static BarcodeParams QR() {
        BarcodeParams p;
        p.formats = BarcodeFormat::QRCode;
        return p;
    }

    /// Parameters optimized for Data Matrix
    static BarcodeParams DataMatrix() {
        BarcodeParams p;
        p.formats = BarcodeFormat::DataMatrix;
        return p;
    }

    /// Parameters optimized for 1D codes
    static BarcodeParams Linear() {
        BarcodeParams p;
        p.formats = BarcodeFormat::Linear;
        return p;
    }

    /// Parameters for challenging conditions
    static BarcodeParams Robust() {
        BarcodeParams p;
        p.tryHarder = true;
        p.tryRotate = true;
        p.tryInvert = true;
        p.tryDownscale = true;
        return p;
    }
};

// =============================================================================
// Main API Functions
// =============================================================================

/**
 * @brief Read barcodes from image
 *
 * @param image Input image (grayscale or color)
 * @param params Reading parameters
 * @return Vector of detected barcodes
 *
 * @code
 * auto results = ReadBarcodes(image);
 * for (const auto& r : results) {
 *     std::cout << r.formatName << ": " << r.text << std::endl;
 * }
 * @endcode
 */
QIVISION_API std::vector<BarcodeResult> ReadBarcodes(
    const QImage& image,
    const BarcodeParams& params = BarcodeParams::Default());

/**
 * @brief Read single barcode from image
 *
 * @param image Input image
 * @param params Reading parameters
 * @return First detected barcode (check .valid)
 */
QIVISION_API BarcodeResult ReadBarcode(
    const QImage& image,
    const BarcodeParams& params = BarcodeParams::Default());

/**
 * @brief Read barcodes of specific format
 *
 * @param image Input image
 * @param format Barcode format to search
 * @return Vector of detected barcodes
 */
QIVISION_API std::vector<BarcodeResult> ReadBarcodes(
    const QImage& image,
    BarcodeFormat format);

/**
 * @brief Read single barcode of specific format
 *
 * @param image Input image
 * @param format Barcode format to search
 * @return First detected barcode
 */
QIVISION_API BarcodeResult ReadBarcode(
    const QImage& image,
    BarcodeFormat format);

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * @brief Read QR codes from image
 */
QIVISION_API std::vector<BarcodeResult> ReadQRCodes(const QImage& image);

/**
 * @brief Read Data Matrix codes from image
 */
QIVISION_API std::vector<BarcodeResult> ReadDataMatrix(const QImage& image);

/**
 * @brief Read 1D barcodes from image
 */
QIVISION_API std::vector<BarcodeResult> ReadLinearCodes(const QImage& image);

/**
 * @brief Read Code128 barcodes from image
 */
QIVISION_API std::vector<BarcodeResult> ReadCode128(const QImage& image);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Get format name as string
 */
QIVISION_API std::string FormatToString(BarcodeFormat format);

/**
 * @brief Parse format from string
 */
QIVISION_API BarcodeFormat StringToFormat(const std::string& name);

/**
 * @brief Check if ZXing-cpp is available
 * @return Always true (linked at compile time)
 */
QIVISION_API bool IsAvailable();

/**
 * @brief Get ZXing-cpp version string
 */
QIVISION_API std::string GetVersion();

} // namespace Qi::Vision::Barcode
