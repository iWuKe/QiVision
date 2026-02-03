/**
 * @file ColorConvert.cpp
 * @brief Color space conversion implementation
 */

#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace Qi::Vision::Color {

// =============================================================================
// Constants
// =============================================================================

namespace {

// sRGB to XYZ (D65) conversion matrix (reserved for future Lab/XYZ support)
[[maybe_unused]] constexpr double RGB_TO_XYZ[3][3] = {
    {0.4124564, 0.3575761, 0.1804375},
    {0.2126729, 0.7151522, 0.0721750},
    {0.0193339, 0.1191920, 0.9503041}
};

// XYZ to sRGB (D65) conversion matrix (reserved for future Lab/XYZ support)
[[maybe_unused]] constexpr double XYZ_TO_RGB[3][3] = {
    { 3.2404542, -1.5371385, -0.4985314},
    {-0.9692660,  1.8760108,  0.0415560},
    { 0.0556434, -0.2040259,  1.0572252}
};

// D65 white point (reserved for future Lab/XYZ support)
[[maybe_unused]] constexpr double D65_X = 0.95047;
[[maybe_unused]] constexpr double D65_Y = 1.00000;
[[maybe_unused]] constexpr double D65_Z = 1.08883;

// Lab/Luv constants
constexpr double LAB_EPSILON = 0.008856;    // (6/29)^3
constexpr double LAB_KAPPA = 903.3;         // (29/3)^3
[[maybe_unused]] constexpr double LAB_16_116 = 16.0 / 116.0;

// D65 white point chromaticity for Luv
// u'_n = 4*X_n / (X_n + 15*Y_n + 3*Z_n)
// v'_n = 9*Y_n / (X_n + 15*Y_n + 3*Z_n)
constexpr double D65_UN = 0.19783000664283681;  // 4*0.95047 / 19.21696
constexpr double D65_VN = 0.46831999493879100;  // 9*1.0 / 19.21696

// Helper functions
inline double Clamp(double val, double minVal, double maxVal) {
    return std::max(minVal, std::min(maxVal, val));
}

inline uint8_t ClampU8(double val) {
    return static_cast<uint8_t>(Clamp(val, 0.0, 255.0));
}

[[maybe_unused]] inline double SrgbToLinear(double val) {
    return (val <= 0.04045) ? val / 12.92 : std::pow((val + 0.055) / 1.055, 2.4);
}

[[maybe_unused]] inline double LinearToSrgb(double val) {
    return (val <= 0.0031308) ? val * 12.92 : 1.055 * std::pow(val, 1.0/2.4) - 0.055;
}

[[maybe_unused]] inline double LabF(double t) {
    return (t > LAB_EPSILON) ? std::cbrt(t) : (LAB_KAPPA * t + 16.0) / 116.0;
}

[[maybe_unused]] inline double LabFInv(double t) {
    double t3 = t * t * t;
    return (t3 > LAB_EPSILON) ? t3 : (116.0 * t - 16.0) / LAB_KAPPA;
}

// Color conversion requires UInt8 images
inline bool RequireImage(const QImage& image, const char* funcName) {
    return Validate::RequireImageU8(image, funcName);
}

// Grayscale UInt8 image validation
inline bool RequireGrayU8(const QImage& image, const char* funcName) {
    return Validate::RequireImageU8Gray(image, funcName);
}

inline bool RequireImages(const QImage& image1, const QImage& image2, const char* funcName) {
    return RequireImage(image1, funcName) && RequireImage(image2, funcName);
}

inline bool RequireImages(const QImage& image1, const QImage& image2, const QImage& image3,
                          const char* funcName) {
    return RequireImage(image1, funcName) && RequireImage(image2, funcName) &&
           RequireImage(image3, funcName);
}

inline bool RequireImages(const QImage& image1, const QImage& image2, const QImage& image3,
                          const QImage& image4, const char* funcName) {
    return RequireImage(image1, funcName) && RequireImage(image2, funcName) &&
           RequireImage(image3, funcName) && RequireImage(image4, funcName);
}

// =============================================================================
// RGB <-> XYZ Conversion (sRGB, D65 illuminant)
// =============================================================================

// Convert sRGB [0,255] to XYZ
// XYZ values are scaled: X,Y,Z in [0,1] range normalized to D65
void RgbToXyz(uint8_t r, uint8_t g, uint8_t b, double& x, double& y, double& z) {
    // sRGB to linear RGB
    double rLin = SrgbToLinear(r / 255.0);
    double gLin = SrgbToLinear(g / 255.0);
    double bLin = SrgbToLinear(b / 255.0);

    // Linear RGB to XYZ (D65)
    x = RGB_TO_XYZ[0][0] * rLin + RGB_TO_XYZ[0][1] * gLin + RGB_TO_XYZ[0][2] * bLin;
    y = RGB_TO_XYZ[1][0] * rLin + RGB_TO_XYZ[1][1] * gLin + RGB_TO_XYZ[1][2] * bLin;
    z = RGB_TO_XYZ[2][0] * rLin + RGB_TO_XYZ[2][1] * gLin + RGB_TO_XYZ[2][2] * bLin;
}

// Convert XYZ to sRGB [0,255]
void XyzToRgb(double x, double y, double z, uint8_t& r, uint8_t& g, uint8_t& b) {
    // XYZ to linear RGB
    double rLin = XYZ_TO_RGB[0][0] * x + XYZ_TO_RGB[0][1] * y + XYZ_TO_RGB[0][2] * z;
    double gLin = XYZ_TO_RGB[1][0] * x + XYZ_TO_RGB[1][1] * y + XYZ_TO_RGB[1][2] * z;
    double bLin = XYZ_TO_RGB[2][0] * x + XYZ_TO_RGB[2][1] * y + XYZ_TO_RGB[2][2] * z;

    // Linear RGB to sRGB
    r = ClampU8(LinearToSrgb(rLin) * 255.0);
    g = ClampU8(LinearToSrgb(gLin) * 255.0);
    b = ClampU8(LinearToSrgb(bLin) * 255.0);
}

// RGB to XYZ with uint8_t output (scaled to 0-255)
// X: 0-0.95047 -> 0-255, Y: 0-1.0 -> 0-255, Z: 0-1.08883 -> 0-255
void RgbToXyzU8(uint8_t r, uint8_t g, uint8_t b, uint8_t& xOut, uint8_t& yOut, uint8_t& zOut) {
    double x, y, z;
    RgbToXyz(r, g, b, x, y, z);

    // Scale to 0-255 (using D65 white point as max)
    xOut = ClampU8(x / D65_X * 255.0);
    yOut = ClampU8(y / D65_Y * 255.0);
    zOut = ClampU8(z / D65_Z * 255.0);
}

// XYZ (uint8_t, scaled) to RGB
void XyzU8ToRgb(uint8_t xIn, uint8_t yIn, uint8_t zIn, uint8_t& r, uint8_t& g, uint8_t& b) {
    // Unscale from 0-255 to actual XYZ values
    double x = (xIn / 255.0) * D65_X;
    double y = (yIn / 255.0) * D65_Y;
    double z = (zIn / 255.0) * D65_Z;

    XyzToRgb(x, y, z, r, g, b);
}

// =============================================================================
// RGB <-> Lab Conversion (CIE L*a*b*, D65 illuminant)
// =============================================================================

// Convert sRGB [0,255] to Lab
// L: 0-100, a: -128 to +127, b: -128 to +127
void RgbToLab(uint8_t r, uint8_t g, uint8_t b, double& L, double& a, double& labB) {
    // First convert to XYZ
    double x, y, z;
    RgbToXyz(r, g, b, x, y, z);

    // Normalize by D65 white point
    double xn = x / D65_X;
    double yn = y / D65_Y;
    double zn = z / D65_Z;

    // Apply Lab transfer function
    double fx = LabF(xn);
    double fy = LabF(yn);
    double fz = LabF(zn);

    // Compute Lab values
    L = 116.0 * fy - 16.0;      // L: 0-100
    a = 500.0 * (fx - fy);      // a: typically -128 to +127
    labB = 200.0 * (fy - fz);   // b: typically -128 to +127
}

// Convert Lab to sRGB [0,255]
void LabToRgb(double L, double a, double labB, uint8_t& r, uint8_t& g, uint8_t& b) {
    // Lab to XYZ
    double fy = (L + 16.0) / 116.0;
    double fx = a / 500.0 + fy;
    double fz = fy - labB / 200.0;

    double xn = LabFInv(fx);
    double yn = LabFInv(fy);
    double zn = LabFInv(fz);

    // Denormalize by D65 white point
    double x = xn * D65_X;
    double y = yn * D65_Y;
    double z = zn * D65_Z;

    // XYZ to RGB
    XyzToRgb(x, y, z, r, g, b);
}

// RGB to Lab with uint8_t output
// L: 0-100 -> 0-255 (scale by 2.55)
// a: -128 to +127 -> 0-255 (offset by 128)
// b: -128 to +127 -> 0-255 (offset by 128)
void RgbToLabU8(uint8_t r, uint8_t g, uint8_t b, uint8_t& lOut, uint8_t& aOut, uint8_t& bOut) {
    double L, a, labB;
    RgbToLab(r, g, b, L, a, labB);

    // Scale L from 0-100 to 0-255
    lOut = ClampU8(L * 2.55);
    // Offset a and b from [-128, 127] to [0, 255]
    aOut = ClampU8(a + 128.0);
    bOut = ClampU8(labB + 128.0);
}

// Lab (uint8_t, scaled) to RGB
void LabU8ToRgb(uint8_t lIn, uint8_t aIn, uint8_t bIn, uint8_t& r, uint8_t& g, uint8_t& b) {
    // Unscale L from 0-255 to 0-100
    double L = lIn / 2.55;
    // Unoffset a and b from [0, 255] to [-128, 127]
    double a = aIn - 128.0;
    double labB = bIn - 128.0;

    LabToRgb(L, a, labB, r, g, b);
}

// =============================================================================
// RGB <-> Luv Conversion (CIE L*u*v*, D65 illuminant)
// =============================================================================

// Convert sRGB [0,255] to Luv
// L: 0-100, u: ~-134 to +220, v: ~-140 to +122 (for sRGB gamut)
void RgbToLuv(uint8_t r, uint8_t g, uint8_t b, double& L, double& u, double& v) {
    // First convert to XYZ
    double x, y, z;
    RgbToXyz(r, g, b, x, y, z);

    // Compute u' and v' chromaticity coordinates
    double denom = x + 15.0 * y + 3.0 * z;
    double uPrime, vPrime;

    if (denom < 1e-10) {
        // Black or very dark - set chromaticity to white point
        uPrime = D65_UN;
        vPrime = D65_VN;
    } else {
        uPrime = 4.0 * x / denom;
        vPrime = 9.0 * y / denom;
    }

    // Compute L* (same as Lab)
    double yr = y / D65_Y;
    if (yr > LAB_EPSILON) {
        L = 116.0 * std::cbrt(yr) - 16.0;
    } else {
        L = LAB_KAPPA * yr;
    }

    // Compute u* and v*
    u = 13.0 * L * (uPrime - D65_UN);
    v = 13.0 * L * (vPrime - D65_VN);
}

// Convert Luv to sRGB [0,255]
void LuvToRgb(double L, double u, double v, uint8_t& r, uint8_t& g, uint8_t& b) {
    // Handle black case
    if (L < 1e-10) {
        r = g = b = 0;
        return;
    }

    // Compute u' and v' from u* and v*
    double uPrime = u / (13.0 * L) + D65_UN;
    double vPrime = v / (13.0 * L) + D65_VN;

    // Compute Y from L*
    double y;
    if (L > 8.0) {  // LAB_KAPPA * LAB_EPSILON ≈ 8
        double t = (L + 16.0) / 116.0;
        y = D65_Y * t * t * t;
    } else {
        y = D65_Y * L / LAB_KAPPA;
    }

    // Handle degenerate v' case
    if (std::abs(vPrime) < 1e-10) {
        r = g = b = ClampU8(y * 255.0);
        return;
    }

    // Compute X and Z from chromaticity
    double x = y * 9.0 * uPrime / (4.0 * vPrime);
    double z = y * (12.0 - 3.0 * uPrime - 20.0 * vPrime) / (4.0 * vPrime);

    // XYZ to RGB
    XyzToRgb(x, y, z, r, g, b);
}

// RGB to Luv with uint8_t output (OpenCV compatible)
// L: 0-100 -> 0-255, scale by 255/100
// u: -134 to +220 -> 0-255, (u + 134) * 255/354
// v: -140 to +122 -> 0-255, (v + 140) * 255/262
void RgbToLuvU8(uint8_t r, uint8_t g, uint8_t b, uint8_t& lOut, uint8_t& uOut, uint8_t& vOut) {
    double L, u, v;
    RgbToLuv(r, g, b, L, u, v);

    lOut = ClampU8(L * 255.0 / 100.0);
    uOut = ClampU8((u + 134.0) * 255.0 / 354.0);
    vOut = ClampU8((v + 140.0) * 255.0 / 262.0);
}

// Luv (uint8_t) to RGB (OpenCV compatible)
void LuvU8ToRgb(uint8_t lIn, uint8_t uIn, uint8_t vIn, uint8_t& r, uint8_t& g, uint8_t& b) {
    double L = lIn * 100.0 / 255.0;
    double u = uIn * 354.0 / 255.0 - 134.0;
    double v = vIn * 262.0 / 255.0 - 140.0;

    LuvToRgb(L, u, v, r, g, b);
}

} // anonymous namespace

// =============================================================================
// Utility Functions
// =============================================================================

std::string GetColorSpaceName(ColorSpace space) {
    switch (space) {
        case ColorSpace::Gray: return "gray";
        case ColorSpace::RGB: return "rgb";
        case ColorSpace::BGR: return "bgr";
        case ColorSpace::RGBA: return "rgba";
        case ColorSpace::BGRA: return "bgra";
        case ColorSpace::HSV: return "hsv";
        case ColorSpace::HSL: return "hsl";
        case ColorSpace::Lab: return "lab";
        case ColorSpace::Luv: return "luv";
        case ColorSpace::XYZ: return "xyz";
        case ColorSpace::YCrCb: return "ycrcb";
        case ColorSpace::YUV: return "yuv";
        default: return "unknown";
    }
}

ColorSpace ParseColorSpace(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "gray" || lower == "grey") return ColorSpace::Gray;
    if (lower == "rgb") return ColorSpace::RGB;
    if (lower == "bgr") return ColorSpace::BGR;
    if (lower == "rgba") return ColorSpace::RGBA;
    if (lower == "bgra") return ColorSpace::BGRA;
    if (lower == "hsv") return ColorSpace::HSV;
    if (lower == "hsl" || lower == "hls") return ColorSpace::HSL;
    if (lower == "lab" || lower == "cielab") return ColorSpace::Lab;
    if (lower == "luv" || lower == "cieluv") return ColorSpace::Luv;
    if (lower == "xyz" || lower == "ciexyz") return ColorSpace::XYZ;
    if (lower == "ycrcb" || lower == "ycbcr") return ColorSpace::YCrCb;
    if (lower == "yuv") return ColorSpace::YUV;

    throw InvalidArgumentException("Unknown color space: " + name);
}

int32_t GetChannelCount(ColorSpace space) {
    switch (space) {
        case ColorSpace::Gray: return 1;
        case ColorSpace::RGB:
        case ColorSpace::BGR:
        case ColorSpace::HSV:
        case ColorSpace::HSL:
        case ColorSpace::Lab:
        case ColorSpace::Luv:
        case ColorSpace::XYZ:
        case ColorSpace::YCrCb:
        case ColorSpace::YUV: return 3;
        case ColorSpace::RGBA:
        case ColorSpace::BGRA: return 4;
        default: return 1;
    }
}

bool HasAlphaChannel(ColorSpace space) {
    return space == ColorSpace::RGBA || space == ColorSpace::BGRA;
}

int32_t CountChannels(const QImage& image) {
    if (!RequireImage(image, "CountChannels")) {
        return 0;
    }
    return image.Channels();
}

// =============================================================================
// Grayscale Conversion
// =============================================================================

void Rgb1ToGray(const QImage& image, QImage& output, const std::string& method) {
    if (!RequireImage(image, "Rgb1ToGray")) {
        output = QImage();
        return;
    }

    // Already grayscale
    if (image.GetChannelType() == ChannelType::Gray) {
        output = image.Clone();
        return;
    }

    int srcChannels = image.Channels();
    if (srcChannels < 3) {
        throw InvalidArgumentException("Input must have at least 3 channels");
    }

    output = QImage(image.Width(), image.Height(), PixelType::UInt8, ChannelType::Gray);

    // Determine weights based on method
    double rWeight, gWeight, bWeight;
    bool useMinMax = false;
    bool useMax = false;

    std::string lowerMethod = method;
    std::transform(lowerMethod.begin(), lowerMethod.end(), lowerMethod.begin(), ::tolower);

    if (lowerMethod.empty()) {
        lowerMethod = "luminosity";
    }

    if (lowerMethod == "luminosity" || lowerMethod == "bt601") {
        rWeight = 0.299; gWeight = 0.587; bWeight = 0.114;
    } else if (lowerMethod == "bt709") {
        rWeight = 0.2126; gWeight = 0.7152; bWeight = 0.0722;
    } else if (lowerMethod == "average") {
        rWeight = gWeight = bWeight = 1.0 / 3.0;
    } else if (lowerMethod == "lightness" || lowerMethod == "desaturate") {
        useMinMax = true;
    } else if (lowerMethod == "max") {
        useMinMax = true;
        useMax = true;
    } else if (lowerMethod == "min") {
        useMinMax = true;
    } else {
        throw InvalidArgumentException("Unknown grayscale method: " + method);
    }

    for (int32_t y = 0; y < image.Height(); ++y) {
        const uint8_t* src = static_cast<const uint8_t*>(image.RowPtr(y));
        uint8_t* dst = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < image.Width(); ++x) {
            int r = src[x * srcChannels + 0];
            int g = src[x * srcChannels + 1];
            int b = src[x * srcChannels + 2];

            uint8_t gray;
            if (useMinMax) {
                int maxVal = std::max({r, g, b});
                int minVal = std::min({r, g, b});
                gray = useMax ? static_cast<uint8_t>(maxVal)
                              : static_cast<uint8_t>((maxVal + minVal) / 2);
            } else {
                gray = ClampU8(r * rWeight + g * gWeight + b * bWeight);
            }
            dst[x] = gray;
        }
    }
}

void Rgb3ToGray(const QImage& red, const QImage& green, const QImage& blue,
                QImage& output, const std::string& method) {
    // Compose then convert
    QImage rgb;
    Compose3(red, green, blue, rgb, ChannelType::RGB);
    Rgb1ToGray(rgb, output, method);
}

void GrayToRgb(const QImage& gray, QImage& output) {
    if (!RequireGrayU8(gray, "GrayToRgb")) {
        output = QImage();
        return;
    }

    output = QImage(gray.Width(), gray.Height(), gray.Type(), ChannelType::RGB);

    for (int32_t y = 0; y < gray.Height(); ++y) {
        const uint8_t* src = static_cast<const uint8_t*>(gray.RowPtr(y));
        uint8_t* dst = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < gray.Width(); ++x) {
            uint8_t val = src[x];
            dst[x * 3 + 0] = val;
            dst[x * 3 + 1] = val;
            dst[x * 3 + 2] = val;
        }
    }
}

// =============================================================================
// Channel Operations
// =============================================================================

void Decompose3(const QImage& image, QImage& ch1, QImage& ch2, QImage& ch3) {
    if (!RequireImage(image, "Decompose3")) {
        ch1 = ch2 = ch3 = QImage();
        return;
    }

    if (image.Channels() < 3) {
        throw InvalidArgumentException("Input must have at least 3 channels");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    PixelType type = image.Type();
    int channels = image.Channels();

    ch1 = QImage(w, h, type, ChannelType::Gray);
    ch2 = QImage(w, h, type, ChannelType::Gray);
    ch3 = QImage(w, h, type, ChannelType::Gray);

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* src = static_cast<const uint8_t*>(image.RowPtr(y));
        uint8_t* d1 = static_cast<uint8_t*>(ch1.RowPtr(y));
        uint8_t* d2 = static_cast<uint8_t*>(ch2.RowPtr(y));
        uint8_t* d3 = static_cast<uint8_t*>(ch3.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            d1[x] = src[x * channels + 0];
            d2[x] = src[x * channels + 1];
            d3[x] = src[x * channels + 2];
        }
    }
}

void Decompose4(const QImage& image, QImage& ch1, QImage& ch2, QImage& ch3, QImage& ch4) {
    if (!RequireImage(image, "Decompose4")) {
        ch1 = ch2 = ch3 = ch4 = QImage();
        return;
    }

    if (image.Channels() < 4) {
        throw InvalidArgumentException("Input must have 4 channels");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    PixelType type = image.Type();

    ch1 = QImage(w, h, type, ChannelType::Gray);
    ch2 = QImage(w, h, type, ChannelType::Gray);
    ch3 = QImage(w, h, type, ChannelType::Gray);
    ch4 = QImage(w, h, type, ChannelType::Gray);

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* src = static_cast<const uint8_t*>(image.RowPtr(y));
        uint8_t* d1 = static_cast<uint8_t*>(ch1.RowPtr(y));
        uint8_t* d2 = static_cast<uint8_t*>(ch2.RowPtr(y));
        uint8_t* d3 = static_cast<uint8_t*>(ch3.RowPtr(y));
        uint8_t* d4 = static_cast<uint8_t*>(ch4.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            d1[x] = src[x * 4 + 0];
            d2[x] = src[x * 4 + 1];
            d3[x] = src[x * 4 + 2];
            d4[x] = src[x * 4 + 3];
        }
    }
}

void Compose3(const QImage& ch1, const QImage& ch2, const QImage& ch3,
              QImage& output, ChannelType channelType) {
    if (!RequireImages(ch1, ch2, ch3, "Compose3")) {
        output = QImage();
        return;
    }

    if (ch1.Width() != ch2.Width() || ch1.Width() != ch3.Width() ||
        ch1.Height() != ch2.Height() || ch1.Height() != ch3.Height()) {
        throw InvalidArgumentException("All channels must have same dimensions");
    }
    if (ch1.Type() != PixelType::UInt8 || ch2.Type() != PixelType::UInt8 ||
        ch3.Type() != PixelType::UInt8) {
        throw UnsupportedException("Compose3 only supports UInt8 images");
    }

    int32_t w = ch1.Width();
    int32_t h = ch1.Height();
    PixelType type = ch1.Type();

    output = QImage(w, h, type, channelType);

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* s1 = static_cast<const uint8_t*>(ch1.RowPtr(y));
        const uint8_t* s2 = static_cast<const uint8_t*>(ch2.RowPtr(y));
        const uint8_t* s3 = static_cast<const uint8_t*>(ch3.RowPtr(y));
        uint8_t* dst = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            dst[x * 3 + 0] = s1[x];
            dst[x * 3 + 1] = s2[x];
            dst[x * 3 + 2] = s3[x];
        }
    }
}

void Compose4(const QImage& ch1, const QImage& ch2,
              const QImage& ch3, const QImage& ch4,
              QImage& output, ChannelType channelType) {
    if (!RequireImages(ch1, ch2, ch3, ch4, "Compose4")) {
        output = QImage();
        return;
    }
    if (ch1.Width() != ch2.Width() || ch1.Width() != ch3.Width() || ch1.Width() != ch4.Width() ||
        ch1.Height() != ch2.Height() || ch1.Height() != ch3.Height() || ch1.Height() != ch4.Height()) {
        throw InvalidArgumentException("All channels must have same dimensions");
    }
    if (ch1.Type() != PixelType::UInt8 || ch2.Type() != PixelType::UInt8 ||
        ch3.Type() != PixelType::UInt8 || ch4.Type() != PixelType::UInt8) {
        throw UnsupportedException("Compose4 only supports UInt8 images");
    }

    int32_t w = ch1.Width();
    int32_t h = ch1.Height();
    PixelType type = ch1.Type();

    output = QImage(w, h, type, channelType);

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* s1 = static_cast<const uint8_t*>(ch1.RowPtr(y));
        const uint8_t* s2 = static_cast<const uint8_t*>(ch2.RowPtr(y));
        const uint8_t* s3 = static_cast<const uint8_t*>(ch3.RowPtr(y));
        const uint8_t* s4 = static_cast<const uint8_t*>(ch4.RowPtr(y));
        uint8_t* dst = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            dst[x * 4 + 0] = s1[x];
            dst[x * 4 + 1] = s2[x];
            dst[x * 4 + 2] = s3[x];
            dst[x * 4 + 3] = s4[x];
        }
    }
}

void AccessChannel(const QImage& image, QImage& output, int32_t channelIndex) {
    if (!RequireImage(image, "AccessChannel")) {
        output = QImage();
        return;
    }

    int channels = image.Channels();
    if (channelIndex < 0 || channelIndex >= channels) {
        throw InvalidArgumentException("Channel index out of range");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    PixelType type = image.Type();

    output = QImage(w, h, type, ChannelType::Gray);

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* src = static_cast<const uint8_t*>(image.RowPtr(y));
        uint8_t* dst = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            dst[x] = src[x * channels + channelIndex];
        }
    }
}

void SplitChannels(const QImage& image, std::vector<QImage>& outputs) {
    outputs.clear();

    if (!RequireImage(image, "SplitChannels")) {
        return;
    }
    int numChannels = image.Channels();
    for (int i = 0; i < numChannels; ++i) {
        QImage ch;
        AccessChannel(image, ch, i);
        outputs.push_back(std::move(ch));
    }
}

void MergeChannels(const std::vector<QImage>& channels, QImage& output, ChannelType channelType) {
    if (channels.empty()) {
        output = QImage();
        return;
    }

    if (channels.size() == 3) {
        Compose3(channels[0], channels[1], channels[2], output, channelType);
    } else if (channels.size() == 4) {
        Compose4(channels[0], channels[1], channels[2], channels[3], output, channelType);
    } else if (channels.size() == 1) {
        output = channels[0].Clone();
    } else {
        throw InvalidArgumentException("Unsupported number of channels");
    }
}

// =============================================================================
// Channel Swapping
// =============================================================================

void RgbToBgr(const QImage& image, QImage& output) {
    SwapChannels(image, output, 0, 2);
}

void BgrToRgb(const QImage& image, QImage& output) {
    SwapChannels(image, output, 0, 2);
}

void SwapChannels(const QImage& image, QImage& output, int32_t ch1, int32_t ch2) {
    if (!RequireImage(image, "SwapChannels")) {
        output = QImage();
        return;
    }

    int channels = image.Channels();
    if (ch1 < 0 || ch1 >= channels || ch2 < 0 || ch2 >= channels) {
        throw InvalidArgumentException("Channel index out of range");
    }

    if (ch1 == ch2) {
        output = image.Clone();
        return;
    }

    output = image.Clone();
    int32_t w = image.Width();
    int32_t h = image.Height();

    if (image.Type() == PixelType::UInt8) {
        for (int32_t y = 0; y < h; ++y) {
            uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                std::swap(row[x * channels + ch1], row[x * channels + ch2]);
            }
        }
    }
}

void ReorderChannels(const QImage& image, QImage& output, const std::vector<int32_t>& order) {
    if (!RequireImage(image, "ReorderChannels")) {
        output = QImage();
        return;
    }

    int channels = image.Channels();
    if (static_cast<int>(order.size()) != channels) {
        throw InvalidArgumentException("Order size must match channel count");
    }
    for (int idx : order) {
        if (idx < 0 || idx >= channels) {
            throw InvalidArgumentException("ReorderChannels: order index out of range");
        }
    }

    output = QImage(image.Width(), image.Height(), image.Type(), image.GetChannelType());
    int32_t w = image.Width();
    int32_t h = image.Height();

    if (image.Type() == PixelType::UInt8) {
        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* src = static_cast<const uint8_t*>(image.RowPtr(y));
            uint8_t* dst = static_cast<uint8_t*>(output.RowPtr(y));

            for (int32_t x = 0; x < w; ++x) {
                for (int c = 0; c < channels; ++c) {
                    dst[x * channels + c] = src[x * channels + order[c]];
                }
            }
        }
    }
}

// =============================================================================
// Color Space Conversion - RGB <-> HSV
// =============================================================================

namespace {

void RgbToHsv(uint8_t r, uint8_t g, uint8_t b, uint8_t& h, uint8_t& s, uint8_t& v) {
    double rd = r / 255.0;
    double gd = g / 255.0;
    double bd = b / 255.0;

    double maxVal = std::max({rd, gd, bd});
    double minVal = std::min({rd, gd, bd});
    double diff = maxVal - minVal;

    // Value
    v = ClampU8(maxVal * 255.0);

    // Saturation
    if (maxVal == 0) {
        s = 0;
    } else {
        s = ClampU8((diff / maxVal) * 255.0);
    }

    // Hue
    double hue = 0;
    if (diff > 0) {
        if (maxVal == rd) {
            hue = 60.0 * std::fmod((gd - bd) / diff + 6.0, 6.0);
        } else if (maxVal == gd) {
            hue = 60.0 * ((bd - rd) / diff + 2.0);
        } else {
            hue = 60.0 * ((rd - gd) / diff + 4.0);
        }
    }
    h = ClampU8(hue * 255.0 / 360.0);
}

void HsvToRgb(uint8_t h, uint8_t s, uint8_t v, uint8_t& r, uint8_t& g, uint8_t& b) {
    double hd = h * 360.0 / 255.0;
    double sd = s / 255.0;
    double vd = v / 255.0;

    if (sd == 0) {
        r = g = b = v;
        return;
    }

    double c = vd * sd;
    double x = c * (1.0 - std::fabs(std::fmod(hd / 60.0, 2.0) - 1.0));
    double m = vd - c;

    double rd, gd, bd;
    if (hd < 60) { rd = c; gd = x; bd = 0; }
    else if (hd < 120) { rd = x; gd = c; bd = 0; }
    else if (hd < 180) { rd = 0; gd = c; bd = x; }
    else if (hd < 240) { rd = 0; gd = x; bd = c; }
    else if (hd < 300) { rd = x; gd = 0; bd = c; }
    else { rd = c; gd = 0; bd = x; }

    r = ClampU8((rd + m) * 255.0);
    g = ClampU8((gd + m) * 255.0);
    b = ClampU8((bd + m) * 255.0);
}

void RgbToHsl(uint8_t r, uint8_t g, uint8_t b, uint8_t& h, uint8_t& s, uint8_t& l) {
    double rd = r / 255.0;
    double gd = g / 255.0;
    double bd = b / 255.0;

    double maxVal = std::max({rd, gd, bd});
    double minVal = std::min({rd, gd, bd});
    double diff = maxVal - minVal;

    // Lightness
    double ld = (maxVal + minVal) / 2.0;
    l = ClampU8(ld * 255.0);

    // Saturation
    if (diff == 0) {
        s = 0;
        h = 0;
        return;
    }

    double sd = diff / (1.0 - std::fabs(2.0 * ld - 1.0));
    s = ClampU8(sd * 255.0);

    // Hue
    double hue;
    if (maxVal == rd) {
        hue = 60.0 * std::fmod((gd - bd) / diff + 6.0, 6.0);
    } else if (maxVal == gd) {
        hue = 60.0 * ((bd - rd) / diff + 2.0);
    } else {
        hue = 60.0 * ((rd - gd) / diff + 4.0);
    }
    h = ClampU8(hue * 255.0 / 360.0);
}

void HslToRgb(uint8_t h, uint8_t s, uint8_t l, uint8_t& r, uint8_t& g, uint8_t& b) {
    double hd = h * 360.0 / 255.0;
    double sd = s / 255.0;
    double ld = l / 255.0;

    if (sd == 0) {
        r = g = b = l;
        return;
    }

    double c = (1.0 - std::fabs(2.0 * ld - 1.0)) * sd;
    double x = c * (1.0 - std::fabs(std::fmod(hd / 60.0, 2.0) - 1.0));
    double m = ld - c / 2.0;

    double rd, gd, bd;
    if (hd < 60) { rd = c; gd = x; bd = 0; }
    else if (hd < 120) { rd = x; gd = c; bd = 0; }
    else if (hd < 180) { rd = 0; gd = c; bd = x; }
    else if (hd < 240) { rd = 0; gd = x; bd = c; }
    else if (hd < 300) { rd = x; gd = 0; bd = c; }
    else { rd = c; gd = 0; bd = x; }

    r = ClampU8((rd + m) * 255.0);
    g = ClampU8((gd + m) * 255.0);
    b = ClampU8((bd + m) * 255.0);
}

void RgbToYCrCb(uint8_t r, uint8_t g, uint8_t b, uint8_t& y, uint8_t& cr, uint8_t& cb) {
    // ITU-R BT.601 conversion
    y  = ClampU8(0.299 * r + 0.587 * g + 0.114 * b);
    cb = ClampU8(128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b);
    cr = ClampU8(128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b);
}

void YCrCbToRgb(uint8_t y, uint8_t cr, uint8_t cb, uint8_t& r, uint8_t& g, uint8_t& b) {
    double yd = y;
    double crd = cr - 128.0;
    double cbd = cb - 128.0;

    r = ClampU8(yd + 1.402 * crd);
    g = ClampU8(yd - 0.344136 * cbd - 0.714136 * crd);
    b = ClampU8(yd + 1.772 * cbd);
}

} // anonymous namespace

// =============================================================================
// Color Space Conversion - Main Functions
// =============================================================================

void TransFromRgb(const QImage& image, QImage& output, ColorSpace toSpace) {
    if (!RequireImage(image, "TransFromRgb")) {
        output = QImage();
        return;
    }

    if (image.Channels() < 3) {
        throw InvalidArgumentException("Input must have at least 3 channels");
    }

    // Handle simple cases
    if (toSpace == ColorSpace::RGB) {
        output = image.Clone();
        return;
    }
    if (toSpace == ColorSpace::Gray) {
        Rgb1ToGray(image, output, "luminosity");
        return;
    }
    if (toSpace == ColorSpace::BGR) {
        RgbToBgr(image, output);
        return;
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int srcChannels = image.Channels();

    output = QImage(w, h, PixelType::UInt8, ChannelType::RGB);

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* src = static_cast<const uint8_t*>(image.RowPtr(y));
        uint8_t* dst = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            uint8_t r = src[x * srcChannels + 0];
            uint8_t g = src[x * srcChannels + 1];
            uint8_t b = src[x * srcChannels + 2];

            uint8_t c1, c2, c3;

            switch (toSpace) {
                case ColorSpace::HSV:
                    RgbToHsv(r, g, b, c1, c2, c3);
                    break;
                case ColorSpace::HSL:
                    RgbToHsl(r, g, b, c1, c2, c3);
                    break;
                case ColorSpace::YCrCb:
                    RgbToYCrCb(r, g, b, c1, c3, c2);  // Y, Cr, Cb order
                    break;
                case ColorSpace::YUV:
                    // YUV is similar to YCrCb for our purposes
                    RgbToYCrCb(r, g, b, c1, c3, c2);
                    break;
                case ColorSpace::Luv:
                    RgbToLuvU8(r, g, b, c1, c2, c3);
                    break;
                case ColorSpace::Lab:
                    RgbToLabU8(r, g, b, c1, c2, c3);
                    break;
                case ColorSpace::XYZ:
                    RgbToXyzU8(r, g, b, c1, c2, c3);
                    break;
                default:
                    c1 = r; c2 = g; c3 = b;
                    break;
            }

            dst[x * 3 + 0] = c1;
            dst[x * 3 + 1] = c2;
            dst[x * 3 + 2] = c3;
        }
    }
}

void TransFromRgb(const QImage& image, QImage& output, const std::string& colorSpace) {
    TransFromRgb(image, output, ParseColorSpace(colorSpace));
}

void TransToRgb(const QImage& image, QImage& output, ColorSpace fromSpace) {
    if (!RequireImage(image, "TransToRgb")) {
        output = QImage();
        return;
    }

    // Handle simple cases
    if (fromSpace == ColorSpace::RGB) {
        output = image.Clone();
        return;
    }
    if (fromSpace == ColorSpace::Gray) {
        GrayToRgb(image, output);
        return;
    }
    if (fromSpace == ColorSpace::BGR) {
        BgrToRgb(image, output);
        return;
    }

    if (image.Channels() < 3) {
        throw InvalidArgumentException("Input must have at least 3 channels");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int srcChannels = image.Channels();

    output = QImage(w, h, PixelType::UInt8, ChannelType::RGB);

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* src = static_cast<const uint8_t*>(image.RowPtr(y));
        uint8_t* dst = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            uint8_t c1 = src[x * srcChannels + 0];
            uint8_t c2 = src[x * srcChannels + 1];
            uint8_t c3 = src[x * srcChannels + 2];

            uint8_t r, g, b;

            switch (fromSpace) {
                case ColorSpace::HSV:
                    HsvToRgb(c1, c2, c3, r, g, b);
                    break;
                case ColorSpace::HSL:
                    HslToRgb(c1, c2, c3, r, g, b);
                    break;
                case ColorSpace::YCrCb:
                    YCrCbToRgb(c1, c3, c2, r, g, b);
                    break;
                case ColorSpace::YUV:
                    YCrCbToRgb(c1, c3, c2, r, g, b);
                    break;
                case ColorSpace::Lab:
                    LabU8ToRgb(c1, c2, c3, r, g, b);
                    break;
                case ColorSpace::XYZ:
                    XyzU8ToRgb(c1, c2, c3, r, g, b);
                    break;
                case ColorSpace::Luv:
                    LuvU8ToRgb(c1, c2, c3, r, g, b);
                    break;
                default:
                    r = c1; g = c2; b = c3;
                    break;
            }

            dst[x * 3 + 0] = r;
            dst[x * 3 + 1] = g;
            dst[x * 3 + 2] = b;
        }
    }
}

void TransToRgb(const QImage& image, QImage& output, const std::string& colorSpace) {
    TransToRgb(image, output, ParseColorSpace(colorSpace));
}

void ConvertColorSpace(const QImage& image, QImage& output,
                       ColorSpace fromSpace, ColorSpace toSpace) {
    if (!RequireImage(image, "ConvertColorSpace")) {
        output = QImage();
        return;
    }
    if (fromSpace == toSpace) {
        output = image.Clone();
        return;
    }

    // Convert via RGB as intermediate
    if (fromSpace != ColorSpace::RGB) {
        QImage rgb;
        TransToRgb(image, rgb, fromSpace);
        TransFromRgb(rgb, output, toSpace);
    } else {
        TransFromRgb(image, output, toSpace);
    }
}

// =============================================================================
// Color Adjustment
// =============================================================================

void AdjustBrightness(const QImage& image, QImage& output, double brightness) {
    if (!RequireImage(image, "AdjustBrightness")) {
        output = QImage();
        return;
    }
    if (!std::isfinite(brightness)) {
        throw InvalidArgumentException("AdjustBrightness: invalid brightness");
    }

    output = image.Clone();
    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    if (image.Type() == PixelType::UInt8) {
        for (int32_t y = 0; y < h; ++y) {
            uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                row[x] = ClampU8(row[x] + brightness);
            }
        }
    }
}

void AdjustContrast(const QImage& image, QImage& output, double contrast) {
    if (!RequireImage(image, "AdjustContrast")) {
        output = QImage();
        return;
    }
    if (!std::isfinite(contrast)) {
        throw InvalidArgumentException("AdjustContrast: invalid contrast");
    }

    output = image.Clone();
    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    double factor = contrast;

    if (image.Type() == PixelType::UInt8) {
        for (int32_t y = 0; y < h; ++y) {
            uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                double val = (row[x] - 128.0) * factor + 128.0;
                row[x] = ClampU8(val);
            }
        }
    }
}

void AdjustSaturation(const QImage& image, QImage& output, double saturation) {
    if (!RequireImage(image, "AdjustSaturation")) {
        output = QImage();
        return;
    }
    if (!std::isfinite(saturation)) {
        throw InvalidArgumentException("AdjustSaturation: invalid saturation");
    }

    if (image.Channels() < 3) {
        throw InvalidArgumentException("AdjustSaturation requires at least 3 channels");
    }

    QImage hsv;
    TransFromRgb(image, hsv, ColorSpace::HSV);
    int32_t w = hsv.Width();
    int32_t h = hsv.Height();

    for (int32_t y = 0; y < h; ++y) {
        uint8_t* row = static_cast<uint8_t*>(hsv.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            double s = row[x * 3 + 1] * saturation;
            row[x * 3 + 1] = ClampU8(s);
        }
    }

    TransToRgb(hsv, output, ColorSpace::HSV);
}

void AdjustHue(const QImage& image, QImage& output, double hueShift) {
    if (!RequireImage(image, "AdjustHue")) {
        output = QImage();
        return;
    }
    if (!std::isfinite(hueShift)) {
        throw InvalidArgumentException("AdjustHue: invalid hueShift");
    }

    if (image.Channels() < 3) {
        throw InvalidArgumentException("AdjustHue requires at least 3 channels");
    }

    QImage hsv;
    TransFromRgb(image, hsv, ColorSpace::HSV);
    int32_t w = hsv.Width();
    int32_t h = hsv.Height();

    double shift = hueShift * 255.0 / 360.0;  // Convert degrees to 0-255 range

    for (int32_t y = 0; y < h; ++y) {
        uint8_t* row = static_cast<uint8_t*>(hsv.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            double hue = row[x * 3 + 0] + shift;
            while (hue < 0) hue += 256;
            while (hue >= 256) hue -= 256;
            row[x * 3 + 0] = static_cast<uint8_t>(hue);
        }
    }

    TransToRgb(hsv, output, ColorSpace::HSV);
}

void AdjustGamma(const QImage& image, QImage& output, double gamma) {
    if (!RequireImage(image, "AdjustGamma")) {
        output = QImage();
        return;
    }
    if (!std::isfinite(gamma)) {
        throw InvalidArgumentException("AdjustGamma: invalid gamma");
    }
    if (gamma <= 0.0) {
        throw InvalidArgumentException("AdjustGamma: gamma must be > 0");
    }

    output = image.Clone();
    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    // Build lookup table
    uint8_t lut[256];
    double invGamma = 1.0 / gamma;
    for (int i = 0; i < 256; ++i) {
        lut[i] = ClampU8(std::pow(i / 255.0, invGamma) * 255.0);
    }

    for (int32_t y = 0; y < h; ++y) {
        uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
        for (int32_t x = 0; x < w * channels; ++x) {
            row[x] = lut[row[x]];
        }
    }
}

void InvertColors(const QImage& image, QImage& output) {
    if (!RequireImage(image, "InvertColors")) {
        output = QImage();
        return;
    }

    output = image.Clone();
    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    for (int32_t y = 0; y < h; ++y) {
        uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
        for (int32_t x = 0; x < w * channels; ++x) {
            row[x] = 255 - row[x];
        }
    }
}

void ScaleImage(const QImage& image, QImage& output, double mult, double add) {
    if (!RequireImage(image, "ScaleImage")) {
        output = QImage();
        return;
    }
    if (!std::isfinite(mult) || !std::isfinite(add)) {
        throw InvalidArgumentException("ScaleImage: invalid mult/add");
    }
    if (image.Type() != PixelType::UInt8 &&
        image.Type() != PixelType::UInt16 &&
        image.Type() != PixelType::Float32) {
        throw UnsupportedException("ScaleImage only supports UInt8/UInt16/Float32 images");
    }

    output = QImage(image.Width(), image.Height(), image.Type(), image.GetChannelType());
    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    if (image.Type() == PixelType::UInt8) {
        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* srcRow = static_cast<const uint8_t*>(image.RowPtr(y));
            uint8_t* dstRow = static_cast<uint8_t*>(output.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                double val = srcRow[x] * mult + add;
                dstRow[x] = static_cast<uint8_t>(std::clamp(val, 0.0, 255.0));
            }
        }
    } else if (image.Type() == PixelType::UInt16) {
        for (int32_t y = 0; y < h; ++y) {
            const uint16_t* srcRow = static_cast<const uint16_t*>(image.RowPtr(y));
            uint16_t* dstRow = static_cast<uint16_t*>(output.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                double val = srcRow[x] * mult + add;
                dstRow[x] = static_cast<uint16_t>(std::clamp(val, 0.0, 65535.0));
            }
        }
    } else if (image.Type() == PixelType::Float32) {
        for (int32_t y = 0; y < h; ++y) {
            const float* srcRow = static_cast<const float*>(image.RowPtr(y));
            float* dstRow = static_cast<float*>(output.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                dstRow[x] = static_cast<float>(srcRow[x] * mult + add);
            }
        }
    }
}

void ScaleImageMax(const QImage& image, QImage& output) {
    if (!RequireImage(image, "ScaleImageMax")) {
        output = QImage();
        return;
    }
    if (image.Type() != PixelType::UInt8 && image.Type() != PixelType::UInt16) {
        throw UnsupportedException("ScaleImageMax only supports UInt8/UInt16 images");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    if (image.Type() == PixelType::UInt8) {
        // Find min and max
        uint8_t minVal = 255, maxVal = 0;
        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                if (row[x] < minVal) minVal = row[x];
                if (row[x] > maxVal) maxVal = row[x];
            }
        }

        if (minVal == maxVal) {
            output = image.Clone();
            return;
        }

        double scale = 255.0 / (maxVal - minVal);
        ScaleImage(image, output, scale, -minVal * scale);
    } else if (image.Type() == PixelType::UInt16) {
        uint16_t minVal = 65535, maxVal = 0;
        for (int32_t y = 0; y < h; ++y) {
            const uint16_t* row = static_cast<const uint16_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                if (row[x] < minVal) minVal = row[x];
                if (row[x] > maxVal) maxVal = row[x];
            }
        }

        if (minVal == maxVal) {
            output = image.Clone();
            return;
        }

        double scale = 65535.0 / (maxVal - minVal);
        ScaleImage(image, output, scale, -minVal * scale);
    }
}

void EquHistoImage(const QImage& image, QImage& output) {
    if (!RequireGrayU8(image, "EquHistoImage")) {
        output = QImage();
        return;
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int64_t totalPixels = static_cast<int64_t>(w) * h;

    // Compute histogram
    std::array<int64_t, 256> histogram = {};
    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            histogram[row[x]]++;
        }
    }

    // Compute cumulative distribution function (CDF)
    std::array<int64_t, 256> cdf = {};
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Find minimum non-zero CDF value
    int64_t cdfMin = 0;
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] > 0) {
            cdfMin = cdf[i];
            break;
        }
    }

    // Build lookup table
    std::array<uint8_t, 256> lut = {};
    double scale = 255.0 / (totalPixels - cdfMin);
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] <= cdfMin) {
            lut[i] = 0;
        } else {
            lut[i] = static_cast<uint8_t>(std::clamp((cdf[i] - cdfMin) * scale, 0.0, 255.0));
        }
    }

    // Apply LUT
    output = QImage(w, h, PixelType::UInt8, ChannelType::Gray);
    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* srcRow = static_cast<const uint8_t*>(image.RowPtr(y));
        uint8_t* dstRow = static_cast<uint8_t*>(output.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            dstRow[x] = lut[srcRow[x]];
        }
    }
}

// =============================================================================
// Histogram Analysis
// =============================================================================

void GrayHisto(const QImage& image,
               std::vector<int64_t>& absoluteHisto,
               std::vector<double>& relativeHisto) {
    absoluteHisto.assign(256, 0);
    relativeHisto.assign(256, 0.0);

    if (!RequireImage(image, "GrayHisto")) {
        return;
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();
    int64_t totalPixels = 0;

    if (image.Type() == PixelType::UInt8) {
        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                // For multi-channel, use first channel or compute luminance
                uint8_t val = row[x * channels];
                absoluteHisto[val]++;
                totalPixels++;
            }
        }
    }

    if (totalPixels > 0) {
        for (int i = 0; i < 256; ++i) {
            relativeHisto[i] = static_cast<double>(absoluteHisto[i]) / totalPixels;
        }
    }
}

std::vector<int64_t> GrayHistoAbs(const QImage& image) {
    std::vector<int64_t> absHisto;
    std::vector<double> relHisto;
    GrayHisto(image, absHisto, relHisto);
    return absHisto;
}

void MinMaxGray(const QImage& image, double& minGray, double& maxGray, double& range) {
    minGray = 255.0;
    maxGray = 0.0;
    range = 0.0;

    if (!RequireImage(image, "MinMaxGray")) {
        return;
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    if (image.Type() == PixelType::UInt8) {
        uint8_t minVal = 255, maxVal = 0;
        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                if (row[x] < minVal) minVal = row[x];
                if (row[x] > maxVal) maxVal = row[x];
            }
        }
        minGray = minVal;
        maxGray = maxVal;
    } else if (image.Type() == PixelType::UInt16) {
        uint16_t minVal = 65535, maxVal = 0;
        for (int32_t y = 0; y < h; ++y) {
            const uint16_t* row = static_cast<const uint16_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                if (row[x] < minVal) minVal = row[x];
                if (row[x] > maxVal) maxVal = row[x];
            }
        }
        minGray = minVal;
        maxGray = maxVal;
    } else if (image.Type() == PixelType::Float32) {
        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::lowest();
        for (int32_t y = 0; y < h; ++y) {
            const float* row = static_cast<const float*>(image.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                if (row[x] < minVal) minVal = row[x];
                if (row[x] > maxVal) maxVal = row[x];
            }
        }
        minGray = minVal;
        maxGray = maxVal;
    } else {
        throw UnsupportedException("MinMaxGray only supports UInt8/UInt16/Float32 images");
    }

    range = maxGray - minGray;
}

void Intensity(const QImage& image, double& mean, double& deviation) {
    mean = 0.0;
    deviation = 0.0;

    if (!RequireImage(image, "Intensity")) {
        return;
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();
    int64_t totalPixels = static_cast<int64_t>(w) * h * channels;

    if (totalPixels == 0) return;

    // First pass: compute mean
    double sum = 0.0;
    if (image.Type() == PixelType::UInt8) {
        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                sum += row[x];
            }
        }
    } else if (image.Type() == PixelType::UInt16) {
        for (int32_t y = 0; y < h; ++y) {
            const uint16_t* row = static_cast<const uint16_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                sum += row[x];
            }
        }
    } else if (image.Type() == PixelType::Float32) {
        for (int32_t y = 0; y < h; ++y) {
            const float* row = static_cast<const float*>(image.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                sum += row[x];
            }
        }
    } else {
        throw UnsupportedException("Intensity only supports UInt8/UInt16/Float32 images");
    }

    mean = sum / totalPixels;

    // Second pass: compute standard deviation
    double sumSqDiff = 0.0;
    if (image.Type() == PixelType::UInt8) {
        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                double diff = row[x] - mean;
                sumSqDiff += diff * diff;
            }
        }
    } else if (image.Type() == PixelType::UInt16) {
        for (int32_t y = 0; y < h; ++y) {
            const uint16_t* row = static_cast<const uint16_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                double diff = row[x] - mean;
                sumSqDiff += diff * diff;
            }
        }
    } else if (image.Type() == PixelType::Float32) {
        for (int32_t y = 0; y < h; ++y) {
            const float* row = static_cast<const float*>(image.RowPtr(y));
            for (int32_t x = 0; x < w * channels; ++x) {
                double diff = row[x] - mean;
                sumSqDiff += diff * diff;
            }
        }
    } else {
        throw UnsupportedException("Intensity only supports UInt8/UInt16/Float32 images");
    }

    deviation = std::sqrt(sumSqDiff / totalPixels);
}

double EntropyGray(const QImage& image) {
    if (!RequireImage(image, "EntropyGray")) {
        return 0.0;
    }

    std::vector<int64_t> absHisto;
    std::vector<double> relHisto;
    GrayHisto(image, absHisto, relHisto);

    double entropy = 0.0;
    for (int i = 0; i < 256; ++i) {
        if (relHisto[i] > 0.0) {
            entropy -= relHisto[i] * std::log2(relHisto[i]);
        }
    }

    return entropy;
}

double GrayHistoPercentile(const QImage& image, double percentile) {
    if (!std::isfinite(percentile)) {
        throw InvalidArgumentException("GrayHistoPercentile: invalid percentile");
    }
    if (percentile < 0.0 || percentile > 100.0) {
        throw InvalidArgumentException("GrayHistoPercentile: percentile must be in [0, 100]");
    }
    if (!RequireImage(image, "GrayHistoPercentile")) {
        return 0.0;
    }

    std::vector<int64_t> absHisto = GrayHistoAbs(image);

    int64_t total = 0;
    for (int i = 0; i < 256; ++i) {
        total += absHisto[i];
    }

    if (total == 0) return 0.0;

    int64_t target = static_cast<int64_t>(total * percentile / 100.0);
    int64_t cumulative = 0;

    for (int i = 0; i < 256; ++i) {
        cumulative += absHisto[i];
        if (cumulative >= target) {
            return static_cast<double>(i);
        }
    }

    return 255.0;
}

// =============================================================================
// White Balance
// =============================================================================

void AutoWhiteBalance(const QImage& image, QImage& output, const std::string& method) {
    if (!RequireImage(image, "AutoWhiteBalance")) {
        output = QImage();
        return;
    }
    if (image.Channels() < 3) {
        throw InvalidArgumentException("AutoWhiteBalance requires at least 3 channels");
    }

    std::string lowerMethod = method;
    std::transform(lowerMethod.begin(), lowerMethod.end(), lowerMethod.begin(), ::tolower);
    if (lowerMethod.empty()) {
        lowerMethod = "gray_world";
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    if (lowerMethod == "gray_world") {
        // Calculate average of each channel
        double sumR = 0, sumG = 0, sumB = 0;
        int64_t count = w * h;

        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                sumR += row[x * channels + 0];
                sumG += row[x * channels + 1];
                sumB += row[x * channels + 2];
            }
        }

        double avgR = sumR / count;
        double avgG = sumG / count;
        double avgB = sumB / count;
        double avgGray = (avgR + avgG + avgB) / 3.0;

        ApplyWhiteBalance(image, output, avgGray / avgR, avgGray / avgG, avgGray / avgB);
    } else if (lowerMethod == "white_patch") {
        // Find max of each channel
        uint8_t maxR = 0, maxG = 0, maxB = 0;

        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                maxR = std::max(maxR, row[x * channels + 0]);
                maxG = std::max(maxG, row[x * channels + 1]);
                maxB = std::max(maxB, row[x * channels + 2]);
            }
        }

        double scaleR = maxR > 0 ? 255.0 / maxR : 1.0;
        double scaleG = maxG > 0 ? 255.0 / maxG : 1.0;
        double scaleB = maxB > 0 ? 255.0 / maxB : 1.0;

        ApplyWhiteBalance(image, output, scaleR, scaleG, scaleB);
    } else if (lowerMethod == "none" || lowerMethod == "identity") {
        output = image.Clone();
    } else {
        throw InvalidArgumentException("Unknown white balance method: " + method);
    }
}

void ApplyWhiteBalance(const QImage& image, QImage& output,
                       double whiteR, double whiteG, double whiteB) {
    if (!RequireImage(image, "ApplyWhiteBalance")) {
        output = QImage();
        return;
    }
    if (!std::isfinite(whiteR) || !std::isfinite(whiteG) || !std::isfinite(whiteB)) {
        throw InvalidArgumentException("ApplyWhiteBalance: invalid white balance factors");
    }
    if (image.Channels() < 3) {
        throw InvalidArgumentException("ApplyWhiteBalance requires at least 3 channels");
    }

    output = image.Clone();
    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    for (int32_t y = 0; y < h; ++y) {
        uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            row[x * channels + 0] = ClampU8(row[x * channels + 0] * whiteR);
            row[x * channels + 1] = ClampU8(row[x * channels + 1] * whiteG);
            row[x * channels + 2] = ClampU8(row[x * channels + 2] * whiteB);
        }
    }
}

// =============================================================================
// Color Transform LUT
// =============================================================================

ColorTransLut::ColorTransLut() = default;
ColorTransLut::~ColorTransLut() = default;
ColorTransLut::ColorTransLut(ColorTransLut&& other) noexcept = default;
ColorTransLut& ColorTransLut::operator=(ColorTransLut&& other) noexcept = default;

bool ColorTransLut::IsValid() const {
    return !lut_.empty();
}

ColorTransLut CreateColorTransLut(const std::string& colorSpace,
                                   const std::string& transDirection,
                                   int32_t numBits) {
    if (numBits != 8) {
        throw UnsupportedException("CreateColorTransLut only supports 8-bit images");
    }

    ColorTransLut lut;
    ColorSpace targetSpace = ParseColorSpace(colorSpace);

    std::string lowerDir = transDirection;
    std::transform(lowerDir.begin(), lowerDir.end(), lowerDir.begin(), ::tolower);
    if (lowerDir.empty()) {
        lowerDir = "from_rgb";
    }
    if (lowerDir != "from_rgb" && lowerDir != "to_rgb") {
        throw InvalidArgumentException("Unknown transDirection: " + transDirection);
    }
    bool fromRgb = (lowerDir == "from_rgb");

    if (targetSpace != ColorSpace::HSV && targetSpace != ColorSpace::HSL &&
        targetSpace != ColorSpace::YCrCb && targetSpace != ColorSpace::YUV &&
        targetSpace != ColorSpace::Lab && targetSpace != ColorSpace::Luv &&
        targetSpace != ColorSpace::XYZ) {
        throw UnsupportedException("CreateColorTransLut: target color space not implemented");
    }
    lut.fromSpace_ = fromRgb ? ColorSpace::RGB : targetSpace;
    lut.toSpace_ = fromRgb ? targetSpace : ColorSpace::RGB;

    // Allocate 48MB LUT: 256 * 256 * 256 * 3 bytes
    constexpr size_t LUT_SIZE = 256 * 256 * 256 * 3;
    lut.lut_.resize(LUT_SIZE);

    // Pre-compute all conversions
    for (int r = 0; r < 256; ++r) {
        for (int g = 0; g < 256; ++g) {
            for (int b = 0; b < 256; ++b) {
                size_t idx = (static_cast<size_t>(r) * 256 * 256 + g * 256 + b) * 3;

                uint8_t c1, c2, c3;

                if (fromRgb) {
                    // RGB -> target
                    switch (targetSpace) {
                        case ColorSpace::HSV:
                            RgbToHsv(r, g, b, c1, c2, c3);
                            break;
                        case ColorSpace::HSL:
                            RgbToHsl(r, g, b, c1, c2, c3);
                            break;
                        case ColorSpace::YCrCb:
                        case ColorSpace::YUV:
                            RgbToYCrCb(r, g, b, c1, c3, c2);
                            break;
                        case ColorSpace::Lab:
                            RgbToLabU8(r, g, b, c1, c2, c3);
                            break;
                        case ColorSpace::XYZ:
                            RgbToXyzU8(r, g, b, c1, c2, c3);
                            break;
                        case ColorSpace::Luv:
                            RgbToLuvU8(r, g, b, c1, c2, c3);
                            break;
                        default:
                            c1 = r; c2 = g; c3 = b;
                            break;
                    }
                } else {
                    // target -> RGB
                    switch (targetSpace) {
                        case ColorSpace::HSV:
                            HsvToRgb(r, g, b, c1, c2, c3);
                            break;
                        case ColorSpace::HSL:
                            HslToRgb(r, g, b, c1, c2, c3);
                            break;
                        case ColorSpace::YCrCb:
                        case ColorSpace::YUV:
                            YCrCbToRgb(r, b, g, c1, c2, c3);
                            break;
                        case ColorSpace::Lab:
                            LabU8ToRgb(r, g, b, c1, c2, c3);
                            break;
                        case ColorSpace::XYZ:
                            XyzU8ToRgb(r, g, b, c1, c2, c3);
                            break;
                        case ColorSpace::Luv:
                            LuvU8ToRgb(r, g, b, c1, c2, c3);
                            break;
                        default:
                            c1 = r; c2 = g; c3 = b;
                            break;
                    }
                }

                lut.lut_[idx + 0] = c1;
                lut.lut_[idx + 1] = c2;
                lut.lut_[idx + 2] = c3;
            }
        }
    }

    return lut;
}

void ApplyColorTransLut(const QImage& image1, const QImage& image2, const QImage& image3,
                        QImage& result1, QImage& result2, QImage& result3,
                        const ColorTransLut& lut) {
    if (!lut.IsValid()) {
        throw InvalidArgumentException("Invalid ColorTransLut handle");
    }

    if (!RequireImages(image1, image2, image3, "ApplyColorTransLut")) {
        result1 = QImage();
        result2 = QImage();
        result3 = QImage();
        return;
    }
    if (image1.Type() != PixelType::UInt8 || image2.Type() != PixelType::UInt8 ||
        image3.Type() != PixelType::UInt8) {
        throw UnsupportedException("ApplyColorTransLut only supports UInt8 images");
    }

    int32_t w = image1.Width();
    int32_t h = image1.Height();

    if (image2.Width() != w || image2.Height() != h ||
        image3.Width() != w || image3.Height() != h) {
        throw InvalidArgumentException("All input images must have the same size");
    }

    // Create output images
    result1 = QImage(w, h, PixelType::UInt8, ChannelType::Gray);
    result2 = QImage(w, h, PixelType::UInt8, ChannelType::Gray);
    result3 = QImage(w, h, PixelType::UInt8, ChannelType::Gray);

    const uint8_t* lutData = lut.lut_.data();

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* src1 = static_cast<const uint8_t*>(image1.RowPtr(y));
        const uint8_t* src2 = static_cast<const uint8_t*>(image2.RowPtr(y));
        const uint8_t* src3 = static_cast<const uint8_t*>(image3.RowPtr(y));
        uint8_t* dst1 = static_cast<uint8_t*>(result1.RowPtr(y));
        uint8_t* dst2 = static_cast<uint8_t*>(result2.RowPtr(y));
        uint8_t* dst3 = static_cast<uint8_t*>(result3.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            size_t idx = (static_cast<size_t>(src1[x]) * 256 * 256 +
                          src2[x] * 256 + src3[x]) * 3;
            dst1[x] = lutData[idx + 0];
            dst2[x] = lutData[idx + 1];
            dst3[x] = lutData[idx + 2];
        }
    }
}

void ApplyColorTransLut(const QImage& image, QImage& output, const ColorTransLut& lut) {
    if (!lut.IsValid()) {
        throw InvalidArgumentException("Invalid ColorTransLut handle");
    }

    if (!RequireImage(image, "ApplyColorTransLut")) {
        output = QImage();
        return;
    }
    if (image.Channels() != 3) {
        throw InvalidArgumentException("Input must be a 3-channel image");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();

    output = QImage(w, h, PixelType::UInt8, ChannelType::RGB);
    const uint8_t* lutData = lut.lut_.data();

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* src = static_cast<const uint8_t*>(image.RowPtr(y));
        uint8_t* dst = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            size_t idx = (static_cast<size_t>(src[x * 3]) * 256 * 256 +
                          src[x * 3 + 1] * 256 + src[x * 3 + 2]) * 3;
            dst[x * 3 + 0] = lutData[idx + 0];
            dst[x * 3 + 1] = lutData[idx + 1];
            dst[x * 3 + 2] = lutData[idx + 2];
        }
    }
}

void ClearColorTransLut(ColorTransLut& lut) {
    lut.lut_.clear();
    lut.lut_.shrink_to_fit();
    lut.fromSpace_ = ColorSpace::RGB;
    lut.toSpace_ = ColorSpace::RGB;
}

// =============================================================================
// Bayer Pattern (CFA) Conversion
// =============================================================================

namespace {

// Bayer pattern offsets
enum class BayerPattern { RGGB, GRBG, GBRG, BGGR };

BayerPattern ParseBayerPattern(const std::string& cfaType) {
    std::string lower = cfaType;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "bayer_rg" || lower == "rggb") return BayerPattern::RGGB;
    if (lower == "bayer_gr" || lower == "grbg") return BayerPattern::GRBG;
    if (lower == "bayer_gb" || lower == "gbrg") return BayerPattern::GBRG;
    if (lower == "bayer_bg" || lower == "bggr") return BayerPattern::BGGR;

    if (lower.empty()) {
        return BayerPattern::GBRG;  // Default
    }
    throw InvalidArgumentException("Unknown CFA pattern: " + cfaType);
}

// Get Bayer color at position (for RGGB pattern at origin)
// 0=R, 1=G, 2=B
inline int GetBayerColor(int x, int y, BayerPattern pattern) {
    int baseX = x & 1;
    int baseY = y & 1;
    int pos = baseY * 2 + baseX;

    // Pattern encoding: [0,0]=TL, [0,1]=TR, [1,0]=BL, [1,1]=BR
    static const int patterns[4][4] = {
        {0, 1, 1, 2},  // RGGB: R G / G B
        {1, 0, 2, 1},  // GRBG: G R / B G
        {1, 2, 0, 1},  // GBRG: G B / R G
        {2, 1, 1, 0}   // BGGR: B G / G R
    };

    return patterns[static_cast<int>(pattern)][pos];
}

} // anonymous namespace

void CfaToRgb(const QImage& cfaImage, QImage& output,
              const std::string& cfaType,
              const std::string& interpolation) {
    if (!RequireImage(cfaImage, "CfaToRgb")) {
        output = QImage();
        return;
    }
    if (cfaImage.Channels() != 1) {
        throw InvalidArgumentException("CfaToRgb requires a single-channel image");
    }
    if (cfaImage.Type() != PixelType::UInt8 && cfaImage.Type() != PixelType::UInt16) {
        throw UnsupportedException("CfaToRgb only supports UInt8/UInt16 images");
    }

    int32_t w = cfaImage.Width();
    int32_t h = cfaImage.Height();

    if (w < 2 || h < 2) {
        throw InvalidArgumentException("Image too small for Bayer interpolation");
    }

    BayerPattern pattern = ParseBayerPattern(cfaType);
    std::string lowerInterp = interpolation;
    std::transform(lowerInterp.begin(), lowerInterp.end(), lowerInterp.begin(), ::tolower);
    if (lowerInterp.empty()) {
        lowerInterp = "bilinear";
    }
    bool useDirectional = false;
    if (lowerInterp == "bilinear") {
        useDirectional = false;
    } else if (lowerInterp == "bilinear_dir") {
        useDirectional = true;
    } else {
        throw InvalidArgumentException("Unknown interpolation: " + interpolation);
    }

    output = QImage(w, h, cfaImage.Type(),
                    cfaImage.Type() == PixelType::UInt16 ? ChannelType::RGB : ChannelType::RGB);

    if (cfaImage.Type() == PixelType::UInt8) {
        // 8-bit processing
        for (int32_t y = 1; y < h - 1; ++y) {
            const uint8_t* prevRow = static_cast<const uint8_t*>(cfaImage.RowPtr(y - 1));
            const uint8_t* currRow = static_cast<const uint8_t*>(cfaImage.RowPtr(y));
            const uint8_t* nextRow = static_cast<const uint8_t*>(cfaImage.RowPtr(y + 1));
            uint8_t* dst = static_cast<uint8_t*>(output.RowPtr(y));

            for (int32_t x = 1; x < w - 1; ++x) {
                int color = GetBayerColor(x, y, pattern);
                uint8_t r, g, b;

                // Get the known color at this position
                uint8_t center = currRow[x];

                if (color == 0) {  // Red pixel
                    r = center;
                    // Green: average of 4 neighbors
                    g = (currRow[x-1] + currRow[x+1] + prevRow[x] + nextRow[x]) / 4;
                    // Blue: average of 4 diagonal neighbors
                    b = (prevRow[x-1] + prevRow[x+1] + nextRow[x-1] + nextRow[x+1]) / 4;
                } else if (color == 2) {  // Blue pixel
                    b = center;
                    g = (currRow[x-1] + currRow[x+1] + prevRow[x] + nextRow[x]) / 4;
                    r = (prevRow[x-1] + prevRow[x+1] + nextRow[x-1] + nextRow[x+1]) / 4;
                } else {  // Green pixel
                    g = center;
                    // Determine if R or B is on the same row
                    int leftColor = GetBayerColor(x - 1, y, pattern);
                    if (leftColor == 0) {  // R on left/right
                        r = (currRow[x-1] + currRow[x+1]) / 2;
                        b = (prevRow[x] + nextRow[x]) / 2;
                    } else {  // B on left/right
                        b = (currRow[x-1] + currRow[x+1]) / 2;
                        r = (prevRow[x] + nextRow[x]) / 2;
                    }
                }

                dst[x * 3 + 0] = r;
                dst[x * 3 + 1] = g;
                dst[x * 3 + 2] = b;
            }
        }

        // Handle borders (simple replication)
        // Top row
        const uint8_t* row1 = static_cast<const uint8_t*>(output.RowPtr(1));
        uint8_t* row0 = static_cast<uint8_t*>(output.RowPtr(0));
        std::memcpy(row0, row1, w * 3);

        // Bottom row
        const uint8_t* rowN1 = static_cast<const uint8_t*>(output.RowPtr(h - 2));
        uint8_t* rowN = static_cast<uint8_t*>(output.RowPtr(h - 1));
        std::memcpy(rowN, rowN1, w * 3);

        // Left and right columns
        for (int32_t y = 0; y < h; ++y) {
            uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
            // Left
            row[0] = row[3];
            row[1] = row[4];
            row[2] = row[5];
            // Right
            row[(w-1) * 3 + 0] = row[(w-2) * 3 + 0];
            row[(w-1) * 3 + 1] = row[(w-2) * 3 + 1];
            row[(w-1) * 3 + 2] = row[(w-2) * 3 + 2];
        }
    } else if (cfaImage.Type() == PixelType::UInt16) {
        // 16-bit processing (similar logic)
        for (int32_t y = 1; y < h - 1; ++y) {
            const uint16_t* prevRow = static_cast<const uint16_t*>(cfaImage.RowPtr(y - 1));
            const uint16_t* currRow = static_cast<const uint16_t*>(cfaImage.RowPtr(y));
            const uint16_t* nextRow = static_cast<const uint16_t*>(cfaImage.RowPtr(y + 1));
            uint16_t* dst = static_cast<uint16_t*>(output.RowPtr(y));

            for (int32_t x = 1; x < w - 1; ++x) {
                int color = GetBayerColor(x, y, pattern);
                uint16_t r, g, b;
                uint16_t center = currRow[x];

                if (color == 0) {
                    r = center;
                    g = (currRow[x-1] + currRow[x+1] + prevRow[x] + nextRow[x]) / 4;
                    b = (prevRow[x-1] + prevRow[x+1] + nextRow[x-1] + nextRow[x+1]) / 4;
                } else if (color == 2) {
                    b = center;
                    g = (currRow[x-1] + currRow[x+1] + prevRow[x] + nextRow[x]) / 4;
                    r = (prevRow[x-1] + prevRow[x+1] + nextRow[x-1] + nextRow[x+1]) / 4;
                } else {
                    g = center;
                    int leftColor = GetBayerColor(x - 1, y, pattern);
                    if (leftColor == 0) {
                        r = (currRow[x-1] + currRow[x+1]) / 2;
                        b = (prevRow[x] + nextRow[x]) / 2;
                    } else {
                        b = (currRow[x-1] + currRow[x+1]) / 2;
                        r = (prevRow[x] + nextRow[x]) / 2;
                    }
                }

                dst[x * 3 + 0] = r;
                dst[x * 3 + 1] = g;
                dst[x * 3 + 2] = b;
            }
        }

        // Border handling for 16-bit
        const uint16_t* row1 = static_cast<const uint16_t*>(output.RowPtr(1));
        uint16_t* row0 = static_cast<uint16_t*>(output.RowPtr(0));
        std::memcpy(row0, row1, w * 3 * sizeof(uint16_t));

        const uint16_t* rowN1 = static_cast<const uint16_t*>(output.RowPtr(h - 2));
        uint16_t* rowN = static_cast<uint16_t*>(output.RowPtr(h - 1));
        std::memcpy(rowN, rowN1, w * 3 * sizeof(uint16_t));

        for (int32_t y = 0; y < h; ++y) {
            uint16_t* row = static_cast<uint16_t*>(output.RowPtr(y));
            row[0] = row[3]; row[1] = row[4]; row[2] = row[5];
            row[(w-1)*3+0] = row[(w-2)*3+0];
            row[(w-1)*3+1] = row[(w-2)*3+1];
            row[(w-1)*3+2] = row[(w-2)*3+2];
        }
    }

    // Suppress unused warning
    (void)useDirectional;
}

// =============================================================================
// Linear Color Transformation
// =============================================================================

void LinearTransColor(const QImage& image, QImage& output,
                      const std::vector<double>& transMat,
                      int32_t numOutputChannels) {
    if (!RequireImage(image, "LinearTransColor")) {
        output = QImage();
        return;
    }
    if (numOutputChannels <= 0) {
        throw InvalidArgumentException("LinearTransColor: numOutputChannels must be > 0");
    }
    for (double v : transMat) {
        if (!std::isfinite(v)) {
            throw InvalidArgumentException("LinearTransColor: invalid transMat value");
        }
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int32_t inChannels = image.Channels();

    // Matrix is m x (n+1) where m = output channels, n = input channels
    int32_t expectedSize = numOutputChannels * (inChannels + 1);
    if (static_cast<int32_t>(transMat.size()) != expectedSize) {
        throw InvalidArgumentException(
            "transMat size must be numOutputChannels * (numInputChannels + 1)");
    }

    // Output is always Float32
    ChannelType outChannelType = ChannelType::Gray;
    if (numOutputChannels == 3) outChannelType = ChannelType::RGB;
    else if (numOutputChannels == 4) outChannelType = ChannelType::RGBA;

    output = QImage(w, h, PixelType::Float32, outChannelType);

    // Convert input to float if needed
    QImage floatInput = image.Type() == PixelType::Float32 ?
                        image : image.ConvertTo(PixelType::Float32);

    for (int32_t y = 0; y < h; ++y) {
        const float* src = static_cast<const float*>(floatInput.RowPtr(y));
        float* dst = static_cast<float*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            for (int32_t outCh = 0; outCh < numOutputChannels; ++outCh) {
                double sum = 0.0;
                int32_t rowOffset = outCh * (inChannels + 1);

                // Linear combination of input channels
                for (int32_t inCh = 0; inCh < inChannels; ++inCh) {
                    sum += transMat[rowOffset + inCh] * src[x * inChannels + inCh];
                }
                // Add offset (last column)
                sum += transMat[rowOffset + inChannels];

                dst[x * numOutputChannels + outCh] = static_cast<float>(sum);
            }
        }
    }
}

void ApplyColorMatrix(const QImage& image, QImage& output, const std::vector<double>& matrix) {
    if (!RequireImage(image, "ApplyColorMatrix")) {
        output = QImage();
        return;
    }
    if (matrix.size() != 9) {
        throw InvalidArgumentException("ApplyColorMatrix requires a 3x3 matrix (9 elements)");
    }
    for (double v : matrix) {
        if (!std::isfinite(v)) {
            throw InvalidArgumentException("ApplyColorMatrix: invalid matrix value");
        }
    }

    // Convert 3x3 matrix to 3x4 (add zero offsets)
    std::vector<double> transMat = {
        matrix[0], matrix[1], matrix[2], 0.0,
        matrix[3], matrix[4], matrix[5], 0.0,
        matrix[6], matrix[7], matrix[8], 0.0
    };

    LinearTransColor(image, output, transMat, 3);
}

// =============================================================================
// Principal Component Analysis
// =============================================================================

void PrincipalComp(const QImage& image, QImage& output, int32_t numComponents) {
    if (!RequireImage(image, "PrincipalComp")) {
        output = QImage();
        return;
    }

    int32_t channels = image.Channels();
    if (numComponents <= 0 || numComponents > channels) {
        numComponents = channels;
    }

    // Get transformation matrix
    std::vector<double> transMat, mean, eigenvalues;
    GenPrincipalCompTrans(image, transMat, mean, eigenvalues);

    // Apply transformation (only first numComponents rows)
    int32_t w = image.Width();
    int32_t h = image.Height();

    ChannelType outType = ChannelType::Gray;
    if (numComponents == 3) outType = ChannelType::RGB;
    else if (numComponents == 4) outType = ChannelType::RGBA;

    output = QImage(w, h, PixelType::Float32, outType);

    QImage floatInput = image.Type() == PixelType::Float32 ?
                        image : image.ConvertTo(PixelType::Float32);

    for (int32_t y = 0; y < h; ++y) {
        const float* src = static_cast<const float*>(floatInput.RowPtr(y));
        float* dst = static_cast<float*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            for (int32_t outCh = 0; outCh < numComponents; ++outCh) {
                double sum = 0.0;
                for (int32_t inCh = 0; inCh < channels; ++inCh) {
                    double centered = src[x * channels + inCh] - mean[inCh];
                    sum += transMat[outCh * channels + inCh] * centered;
                }
                dst[x * numComponents + outCh] = static_cast<float>(sum);
            }
        }
    }
}

void GenPrincipalCompTrans(const QImage& image,
                           std::vector<double>& transMat,
                           std::vector<double>& mean,
                           std::vector<double>& eigenvalues) {
    if (!RequireImage(image, "GenPrincipalCompTrans")) {
        transMat.clear();
        mean.clear();
        eigenvalues.clear();
        return;
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int32_t channels = image.Channels();
    int64_t n = static_cast<int64_t>(w) * h;

    // Convert to float
    QImage floatInput = image.Type() == PixelType::Float32 ?
                        image : image.ConvertTo(PixelType::Float32);

    // Compute mean
    mean.resize(channels, 0.0);
    for (int32_t y = 0; y < h; ++y) {
        const float* row = static_cast<const float*>(floatInput.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            for (int32_t c = 0; c < channels; ++c) {
                mean[c] += row[x * channels + c];
            }
        }
    }
    for (int32_t c = 0; c < channels; ++c) {
        mean[c] /= n;
    }

    // Compute covariance matrix
    std::vector<double> cov(channels * channels, 0.0);
    for (int32_t y = 0; y < h; ++y) {
        const float* row = static_cast<const float*>(floatInput.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            for (int32_t i = 0; i < channels; ++i) {
                double vi = row[x * channels + i] - mean[i];
                for (int32_t j = i; j < channels; ++j) {
                    double vj = row[x * channels + j] - mean[j];
                    cov[i * channels + j] += vi * vj;
                }
            }
        }
    }

    // Make symmetric and normalize
    for (int32_t i = 0; i < channels; ++i) {
        for (int32_t j = i; j < channels; ++j) {
            cov[i * channels + j] /= (n - 1);
            cov[j * channels + i] = cov[i * channels + j];
        }
    }

    // Simple power iteration for eigenvalue decomposition
    // (For small channel counts like 3-4, this is sufficient)
    transMat.resize(channels * channels);
    eigenvalues.resize(channels);

    // Initialize with identity
    for (int32_t i = 0; i < channels; ++i) {
        for (int32_t j = 0; j < channels; ++j) {
            transMat[i * channels + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Jacobi iteration (simplified)
    std::vector<double> tempCov = cov;

    for (int iter = 0; iter < 50; ++iter) {
        // Find largest off-diagonal element
        int32_t p = 0, q = 1;
        double maxVal = 0.0;
        for (int32_t i = 0; i < channels; ++i) {
            for (int32_t j = i + 1; j < channels; ++j) {
                if (std::abs(tempCov[i * channels + j]) > maxVal) {
                    maxVal = std::abs(tempCov[i * channels + j]);
                    p = i;
                    q = j;
                }
            }
        }

        if (maxVal < 1e-10) break;

        // Compute rotation angle
        double theta = 0.5 * std::atan2(2.0 * tempCov[p * channels + q],
                                         tempCov[q * channels + q] - tempCov[p * channels + p]);
        double c = std::cos(theta);
        double s = std::sin(theta);

        // Apply rotation to covariance matrix
        for (int32_t i = 0; i < channels; ++i) {
            double temp1 = tempCov[i * channels + p];
            double temp2 = tempCov[i * channels + q];
            tempCov[i * channels + p] = c * temp1 - s * temp2;
            tempCov[i * channels + q] = s * temp1 + c * temp2;
        }
        for (int32_t j = 0; j < channels; ++j) {
            double temp1 = tempCov[p * channels + j];
            double temp2 = tempCov[q * channels + j];
            tempCov[p * channels + j] = c * temp1 - s * temp2;
            tempCov[q * channels + j] = s * temp1 + c * temp2;
        }

        // Update eigenvector matrix
        for (int32_t i = 0; i < channels; ++i) {
            double temp1 = transMat[i * channels + p];
            double temp2 = transMat[i * channels + q];
            transMat[i * channels + p] = c * temp1 - s * temp2;
            transMat[i * channels + q] = s * temp1 + c * temp2;
        }
    }

    // Extract eigenvalues (diagonal of transformed covariance)
    for (int32_t i = 0; i < channels; ++i) {
        eigenvalues[i] = tempCov[i * channels + i];
    }

    // Sort eigenvectors by eigenvalue (descending)
    for (int32_t i = 0; i < channels - 1; ++i) {
        for (int32_t j = i + 1; j < channels; ++j) {
            if (eigenvalues[j] > eigenvalues[i]) {
                std::swap(eigenvalues[i], eigenvalues[j]);
                for (int32_t k = 0; k < channels; ++k) {
                    std::swap(transMat[k * channels + i], transMat[k * channels + j]);
                }
            }
        }
    }

    // Transpose to get row-major eigenvectors
    std::vector<double> temp = transMat;
    for (int32_t i = 0; i < channels; ++i) {
        for (int32_t j = 0; j < channels; ++j) {
            transMat[i * channels + j] = temp[j * channels + i];
        }
    }
}

} // namespace Qi::Vision::Color
