/**
 * @file Barcode.cpp
 * @brief Barcode reading implementation using ZXing-cpp
 */

#include <QiVision/Barcode/Barcode.h>
#include <QiVision/Core/Exception.h>

// ZXing-cpp headers
#include <ReadBarcode.h>
#include <BarcodeFormat.h>
#include <ImageView.h>
#include <Result.h>

#include <algorithm>
#include <cstdlib>

namespace Qi::Vision::Barcode {

// =============================================================================
// Format Conversion Helpers
// =============================================================================

namespace {

ZXing::BarcodeFormats ToZXingFormats(BarcodeFormat format) {
    ZXing::BarcodeFormats zf;

    if (HasFormat(format, BarcodeFormat::Code128))
        zf |= ZXing::BarcodeFormat::Code128;
    if (HasFormat(format, BarcodeFormat::Code39))
        zf |= ZXing::BarcodeFormat::Code39;
    if (HasFormat(format, BarcodeFormat::Code93))
        zf |= ZXing::BarcodeFormat::Code93;
    if (HasFormat(format, BarcodeFormat::Codabar))
        zf |= ZXing::BarcodeFormat::Codabar;
    if (HasFormat(format, BarcodeFormat::EAN8))
        zf |= ZXing::BarcodeFormat::EAN8;
    if (HasFormat(format, BarcodeFormat::EAN13))
        zf |= ZXing::BarcodeFormat::EAN13;
    if (HasFormat(format, BarcodeFormat::ITF))
        zf |= ZXing::BarcodeFormat::ITF;
    if (HasFormat(format, BarcodeFormat::UPCA))
        zf |= ZXing::BarcodeFormat::UPCA;
    if (HasFormat(format, BarcodeFormat::UPCE))
        zf |= ZXing::BarcodeFormat::UPCE;
    if (HasFormat(format, BarcodeFormat::QRCode))
        zf |= ZXing::BarcodeFormat::QRCode;
    if (HasFormat(format, BarcodeFormat::DataMatrix))
        zf |= ZXing::BarcodeFormat::DataMatrix;
    if (HasFormat(format, BarcodeFormat::PDF417))
        zf |= ZXing::BarcodeFormat::PDF417;
    if (HasFormat(format, BarcodeFormat::Aztec))
        zf |= ZXing::BarcodeFormat::Aztec;

    return zf;
}

BarcodeFormat FromZXingFormat(ZXing::BarcodeFormat zf) {
    switch (zf) {
        case ZXing::BarcodeFormat::Code128:    return BarcodeFormat::Code128;
        case ZXing::BarcodeFormat::Code39:     return BarcodeFormat::Code39;
        case ZXing::BarcodeFormat::Code93:     return BarcodeFormat::Code93;
        case ZXing::BarcodeFormat::Codabar:    return BarcodeFormat::Codabar;
        case ZXing::BarcodeFormat::EAN8:       return BarcodeFormat::EAN8;
        case ZXing::BarcodeFormat::EAN13:      return BarcodeFormat::EAN13;
        case ZXing::BarcodeFormat::ITF:        return BarcodeFormat::ITF;
        case ZXing::BarcodeFormat::UPCA:       return BarcodeFormat::UPCA;
        case ZXing::BarcodeFormat::UPCE:       return BarcodeFormat::UPCE;
        case ZXing::BarcodeFormat::QRCode:     return BarcodeFormat::QRCode;
        case ZXing::BarcodeFormat::DataMatrix: return BarcodeFormat::DataMatrix;
        case ZXing::BarcodeFormat::PDF417:     return BarcodeFormat::PDF417;
        case ZXing::BarcodeFormat::Aztec:      return BarcodeFormat::Aztec;
        default:                               return BarcodeFormat::None;
    }
}

ZXing::Binarizer ToZXingBinarizer(Binarizer b) {
    switch (b) {
        case Binarizer::LocalAverage:    return ZXing::Binarizer::LocalAverage;
        case Binarizer::GlobalHistogram: return ZXing::Binarizer::GlobalHistogram;
        case Binarizer::FixedThreshold:  return ZXing::Binarizer::FixedThreshold;
        case Binarizer::BoolCast:        return ZXing::Binarizer::BoolCast;
    }
    return ZXing::Binarizer::LocalAverage;
}

BarcodeResult ConvertResult(const ZXing::Result& zr) {
    BarcodeResult result;

    result.valid = zr.isValid();
    if (!result.valid) {
        return result;
    }

    result.format = FromZXingFormat(zr.format());
    result.text = zr.text();
    result.formatName = ZXing::ToString(zr.format());

    // Position (use center of bounding box)
    auto pos = zr.position();
    double cx = 0, cy = 0;
    for (int i = 0; i < 4; ++i) {
        cx += pos[i].x;
        cy += pos[i].y;
    }
    result.position = Point2d(cx / 4.0, cy / 4.0);

    // Corners
    result.corners.resize(4);
    for (int i = 0; i < 4; ++i) {
        result.corners[i] = Point2d(pos[i].x, pos[i].y);
    }

    // Angle (from top edge)
    double dx = pos[1].x - pos[0].x;
    double dy = pos[1].y - pos[0].y;
    result.angle = std::atan2(dy, dx);

    // Additional info
    result.isMirrored = zr.isMirrored();

    // Symbol version (for QR/DataMatrix) - parse from string
    if (zr.format() == ZXing::BarcodeFormat::QRCode ||
        zr.format() == ZXing::BarcodeFormat::DataMatrix) {
        std::string ver = zr.version();
        if (!ver.empty()) {
            result.symbolVersion = std::atoi(ver.c_str());
        }
    }

    // Error correction level
    result.ecLevel = zr.ecLevel();

    return result;
}

} // anonymous namespace

// =============================================================================
// Main API Implementation
// =============================================================================

std::vector<BarcodeResult> ReadBarcodes(const QImage& image, const BarcodeParams& params) {
    std::vector<BarcodeResult> results;

    if (image.Empty()) {
        return results;
    }

    // Get image data
    int32_t width = image.Width();
    int32_t height = image.Height();
    int channels = image.Channels();
    const uint8_t* data = static_cast<const uint8_t*>(image.Data());
    size_t stride = image.Stride();

    // Create ZXing ImageView
    ZXing::ImageFormat imgFormat;
    if (channels == 1) {
        imgFormat = ZXing::ImageFormat::Lum;
    } else if (channels == 3) {
        imgFormat = ZXing::ImageFormat::RGB;
    } else if (channels == 4) {
        imgFormat = ZXing::ImageFormat::RGBX;
    } else {
        throw UnsupportedException("ReadBarcodes: unsupported channel count");
    }

    ZXing::ImageView imageView(data, width, height, imgFormat, static_cast<int>(stride));

    // Set up reader options
    ZXing::ReaderOptions options;
    options.setFormats(ToZXingFormats(params.formats));
    options.setBinarizer(ToZXingBinarizer(params.binarizer));
    options.setTryHarder(params.tryHarder);
    options.setTryRotate(params.tryRotate);
    options.setTryInvert(params.tryInvert);
    options.setTryDownscale(params.tryDownscale);
    options.setMaxNumberOfSymbols(params.maxNumberOfSymbols);
    options.setMinLineCount(params.minLineCount);

    // Read barcodes
    auto zxResults = ZXing::ReadBarcodes(imageView, options);

    // Convert results
    results.reserve(zxResults.size());
    for (const auto& zr : zxResults) {
        BarcodeResult r = ConvertResult(zr);
        if (r.valid) {
            results.push_back(std::move(r));
        }
    }

    return results;
}

BarcodeResult ReadBarcode(const QImage& image, const BarcodeParams& params) {
    BarcodeParams singleParams = params;
    singleParams.maxNumberOfSymbols = 1;

    auto results = ReadBarcodes(image, singleParams);
    if (!results.empty()) {
        return results[0];
    }
    return BarcodeResult();
}

std::vector<BarcodeResult> ReadBarcodes(const QImage& image, BarcodeFormat format) {
    BarcodeParams params;
    params.formats = format;
    return ReadBarcodes(image, params);
}

BarcodeResult ReadBarcode(const QImage& image, BarcodeFormat format) {
    BarcodeParams params;
    params.formats = format;
    params.maxNumberOfSymbols = 1;
    return ReadBarcode(image, params);
}

// =============================================================================
// Convenience Functions
// =============================================================================

std::vector<BarcodeResult> ReadQRCodes(const QImage& image) {
    return ReadBarcodes(image, BarcodeFormat::QRCode);
}

std::vector<BarcodeResult> ReadDataMatrix(const QImage& image) {
    return ReadBarcodes(image, BarcodeFormat::DataMatrix);
}

std::vector<BarcodeResult> ReadLinearCodes(const QImage& image) {
    return ReadBarcodes(image, BarcodeFormat::Linear);
}

std::vector<BarcodeResult> ReadCode128(const QImage& image) {
    return ReadBarcodes(image, BarcodeFormat::Code128);
}

// =============================================================================
// Utility Functions
// =============================================================================

std::string FormatToString(BarcodeFormat format) {
    switch (format) {
        case BarcodeFormat::Code128:    return "Code128";
        case BarcodeFormat::Code39:     return "Code39";
        case BarcodeFormat::Code93:     return "Code93";
        case BarcodeFormat::Codabar:    return "Codabar";
        case BarcodeFormat::EAN8:       return "EAN-8";
        case BarcodeFormat::EAN13:      return "EAN-13";
        case BarcodeFormat::ITF:        return "ITF";
        case BarcodeFormat::UPCA:       return "UPC-A";
        case BarcodeFormat::UPCE:       return "UPC-E";
        case BarcodeFormat::QRCode:     return "QR Code";
        case BarcodeFormat::DataMatrix: return "Data Matrix";
        case BarcodeFormat::PDF417:     return "PDF417";
        case BarcodeFormat::Aztec:      return "Aztec";
        case BarcodeFormat::Linear:     return "Linear (1D)";
        case BarcodeFormat::Matrix:     return "Matrix (2D)";
        case BarcodeFormat::All:        return "All";
        default:                        return "Unknown";
    }
}

BarcodeFormat StringToFormat(const std::string& name) {
    if (name == "Code128" || name == "CODE_128")   return BarcodeFormat::Code128;
    if (name == "Code39" || name == "CODE_39")     return BarcodeFormat::Code39;
    if (name == "Code93" || name == "CODE_93")     return BarcodeFormat::Code93;
    if (name == "Codabar" || name == "CODABAR")    return BarcodeFormat::Codabar;
    if (name == "EAN-8" || name == "EAN8")         return BarcodeFormat::EAN8;
    if (name == "EAN-13" || name == "EAN13")       return BarcodeFormat::EAN13;
    if (name == "ITF")                             return BarcodeFormat::ITF;
    if (name == "UPC-A" || name == "UPCA")         return BarcodeFormat::UPCA;
    if (name == "UPC-E" || name == "UPCE")         return BarcodeFormat::UPCE;
    if (name == "QR Code" || name == "QRCode")     return BarcodeFormat::QRCode;
    if (name == "Data Matrix" || name == "DataMatrix") return BarcodeFormat::DataMatrix;
    if (name == "PDF417")                          return BarcodeFormat::PDF417;
    if (name == "Aztec")                           return BarcodeFormat::Aztec;
    return BarcodeFormat::None;
}

bool IsAvailable() {
    return true;  // ZXing-cpp is linked at compile time
}

std::string GetVersion() {
    return "ZXing-cpp 2.2.1";
}

} // namespace Qi::Vision::Barcode
