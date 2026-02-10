/**
 * @file ColorDetect.cpp
 * @brief Color detection implementation
 */

#include <QiVision/Color/ColorDetect.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>

#include <algorithm>
#include <cctype>
#include <cmath>

namespace Qi::Vision::Color {

namespace {

inline uint8_t ClampU8FromDouble(double v) {
    if (v <= 0.0) return 0;
    if (v >= 255.0) return 255;
    return static_cast<uint8_t>(v + 0.5);
}

inline bool InRange(uint8_t v, uint8_t minV, uint8_t maxV) {
    return v >= minV && v <= maxV;
}

inline bool InRangeWrap(uint8_t v, uint8_t minV, uint8_t maxV) {
    if (minV <= maxV) {
        return v >= minV && v <= maxV;
    }
    return v >= minV || v <= maxV;
}

inline ColorSpace InferInputColorSpace(const QImage& image) {
    switch (image.GetChannelType()) {
        case ChannelType::RGB: return ColorSpace::RGB;
        case ChannelType::BGR: return ColorSpace::BGR;
        case ChannelType::RGBA: return ColorSpace::RGBA;
        case ChannelType::BGRA: return ColorSpace::BGRA;
        case ChannelType::Gray: return ColorSpace::Gray;
    }
    return ColorSpace::RGB;
}

inline void ToRgbU8(const QImage& image, QImage& rgb) {
    if (image.Empty()) {
        rgb = QImage();
        return;
    }
    if (image.GetChannelType() == ChannelType::Gray) {
        GrayToRgb(image, rgb);
        return;
    }

    ColorSpace from = InferInputColorSpace(image);
    if (from == ColorSpace::RGB) {
        rgb = image;
    } else {
        ConvertColorSpace(image, rgb, from, ColorSpace::RGB);
    }
}

inline std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

inline void ValidateParams(const QImage& image, const ColorDetectParams& params, const char* funcName) {
    if (!Validate::RequireImageU8(image, funcName)) {
        return;
    }
    if (!std::isfinite(params.valueGain) || params.valueGain <= 0.0) {
        throw InvalidArgumentException(std::string(funcName) + ": valueGain must be > 0");
    }
    if (params.minArea < 0) {
        throw InvalidArgumentException(std::string(funcName) + ": minArea must be >= 0");
    }
    if (params.colorSpace != ColorSpace::HSV && params.colorSpace != ColorSpace::Lab) {
        throw InvalidArgumentException(std::string(funcName) + ": colorSpace must be HSV or Lab");
    }
}

inline bool MatchPixel(uint8_t c0, uint8_t c1, uint8_t c2, const ColorRange& range) {
    bool c0Match = range.c0WrapAround
        ? InRangeWrap(c0, range.c0Min, range.c0Max)
        : InRange(c0, range.c0Min, range.c0Max);
    return c0Match &&
           InRange(c1, range.c1Min, range.c1Max) &&
           InRange(c2, range.c2Min, range.c2Max);
}

} // namespace

ColorRange CreateColorRangePreset(const std::string& colorName, ColorSpace colorSpace) {
    std::string lower = ToLower(colorName);
    if (colorSpace == ColorSpace::HSV) {
        if (lower == "red")     return {230, 20,  70, 255,  40, 255, true};
        if (lower == "green")   return { 60, 120, 60, 255,  40, 255, false};
        if (lower == "blue")    return {135, 190, 60, 255,  30, 255, false};
        if (lower == "yellow")  return { 28,  55, 60, 255,  40, 255, false};
        if (lower == "orange")  return { 15,  35, 60, 255,  40, 255, false};
        if (lower == "white")   return {  0, 255,  0,  60, 170, 255, false};
        if (lower == "black")   return {  0, 255,  0, 255,   0,  60, false};
        if (lower == "gray")    return {  0, 255,  0,  50,  60, 190, false};
    } else if (colorSpace == ColorSpace::Lab) {
        // Lab channel storage in this project follows uint8 packed format.
        if (lower == "red")     return { 90, 220, 140, 255, 130, 220, false};
        if (lower == "green")   return { 70, 220,  60, 140, 120, 220, false};
        if (lower == "blue")    return { 40, 200, 120, 220,  40, 130, false};
        if (lower == "yellow")  return {140, 255, 110, 170, 140, 230, false};
        if (lower == "orange")  return {120, 240, 140, 210, 140, 230, false};
        if (lower == "white")   return {180, 255, 108, 148, 108, 148, false};
        if (lower == "black")   return {  0,  55, 108, 148, 108, 148, false};
        if (lower == "gray")    return { 55, 190, 108, 148, 108, 148, false};
    }
    throw InvalidArgumentException("CreateColorRangePreset: unknown colorName or unsupported colorSpace: " + colorName);
}

void FindColorRegion(const QImage& image,
                     QRegion& region,
                     const ColorRange& range,
                     const ColorDetectParams& params) {
    ValidateParams(image, params, "FindColorRegion");
    if (image.Empty()) {
        region = QRegion();
        return;
    }

    QImage rgb;
    ToRgbU8(image, rgb);

    QImage colorImage;
    ConvertColorSpace(rgb, colorImage, ColorSpace::RGB, params.colorSpace);

    const int32_t w = colorImage.Width();
    const int32_t h = colorImage.Height();
    const size_t stride = colorImage.Stride();
    const uint8_t* data = static_cast<const uint8_t*>(colorImage.Data());

    std::vector<QRegion::Run> runs;
    runs.reserve(static_cast<size_t>(h) * 4);

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* row = data + static_cast<size_t>(y) * stride;
        int32_t runStart = -1;
        for (int32_t x = 0; x < w; ++x) {
            uint8_t c0 = row[x * 3 + 0];
            uint8_t c1 = row[x * 3 + 1];
            uint8_t c2 = row[x * 3 + 2];

            if (params.colorSpace == ColorSpace::HSV && std::abs(params.valueGain - 1.0) > 1e-9) {
                c2 = ClampU8FromDouble(static_cast<double>(c2) * params.valueGain);
            }

            bool hit = MatchPixel(c0, c1, c2, range);
            if (hit) {
                if (runStart < 0) runStart = x;
            } else if (runStart >= 0) {
                runs.emplace_back(y, runStart, x);
                runStart = -1;
            }
        }
        if (runStart >= 0) {
            runs.emplace_back(y, runStart, w);
        }
    }

    region = QRegion(runs);
    if (params.minArea > 0 && region.Area() < params.minArea) {
        region = QRegion();
    }
}

ColorDetectResult FindColor(const QImage& image,
                            const ColorRange& range,
                            const ColorDetectParams& params) {
    ColorDetectResult result;
    if (image.Empty()) {
        return result;
    }

    FindColorRegion(image, result.region, range, params);
    result.pixelCount = result.region.Area();
    result.found = !result.region.Empty();
    if (result.found) {
        result.boundingBox = result.region.BoundingBox();
        double denom = static_cast<double>(image.Width()) * static_cast<double>(image.Height());
        result.coverage = (denom > 0.0) ? static_cast<double>(result.pixelCount) / denom : 0.0;
    }
    return result;
}

ColorDetectResult FindColor(const QImage& image,
                            const std::string& colorName,
                            const ColorDetectParams& params) {
    ColorRange range = CreateColorRangePreset(colorName, params.colorSpace);
    return FindColor(image, range, params);
}

} // namespace Qi::Vision::Color

