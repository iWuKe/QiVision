#pragma once

/**
 * @file ColorDetect.h
 * @brief Color detection and dominant color classification
 */

#include <QiVision/Core/Export.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
#include <QiVision/Color/ColorConvert.h>

#include <cstdint>
#include <string>
#include <vector>

namespace Qi::Vision::Color {

/**
 * @brief 3-channel threshold range in selected color space
 *
 * Channel0 usually represents Hue for HSV and may wrap around.
 */
struct QIVISION_API ColorRange {
    uint8_t c0Min = 0;
    uint8_t c0Max = 255;
    uint8_t c1Min = 0;
    uint8_t c1Max = 255;
    uint8_t c2Min = 0;
    uint8_t c2Max = 255;
    bool c0WrapAround = false;  ///< true for hue ranges crossing 0/255
};

/**
 * @brief Parameters for color region detection
 */
struct QIVISION_API ColorDetectParams {
    ColorSpace colorSpace = ColorSpace::HSV;  ///< Working color space (default: HSV)
    double valueGain = 1.0;                   ///< Optional V channel gain for HSV [0.1, 10.0]
    int32_t minArea = 0;                      ///< Minimum detected area in pixels (0 disables)
};

/**
 * @brief Output of color detection
 */
struct QIVISION_API ColorDetectResult {
    QRegion region;            ///< Detected color region (RLE)
    int64_t pixelCount = 0;    ///< Number of pixels in region
    Rect2i boundingBox;        ///< Region bounding box (valid when found=true)
    double coverage = 0.0;     ///< pixelCount / (image width * image height)
    bool found = false;        ///< True when valid region found
};

/**
 * @brief Create preset color range for common colors
 *
 * Supported names:
 * - "red", "green", "blue", "yellow", "orange"
 * - "white", "black", "gray"
 */
QIVISION_API ColorRange CreateColorRangePreset(
    const std::string& colorName,
    ColorSpace colorSpace = ColorSpace::HSV);

/**
 * @brief Detect color region by threshold range
 */
QIVISION_API void FindColorRegion(
    const QImage& image,
    QRegion& region,
    const ColorRange& range,
    const ColorDetectParams& params = {});

/**
 * @brief Detect color and return stats
 */
QIVISION_API ColorDetectResult FindColor(
    const QImage& image,
    const ColorRange& range,
    const ColorDetectParams& params = {});

/**
 * @brief Detect color using preset name
 */
QIVISION_API ColorDetectResult FindColor(
    const QImage& image,
    const std::string& colorName,
    const ColorDetectParams& params = {});

} // namespace Qi::Vision::Color

