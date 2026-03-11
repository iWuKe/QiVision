/**
 * @file ShapeModel.cpp
 * @brief Halcon-style ShapeModel API implementation
 *
 * Provides Halcon-compatible free functions that wrap the internal
 * ShapeModelImpl class.
 */

#include "ShapeModelImpl.h"
#include "DiagnosticFlags.h"
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>
#include <QiVision/Core/QContourArray.h>
#include <QiVision/Platform/FileIO.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace Qi::Vision::Matching {

// Import Internal types for pyramid operations
using Qi::Vision::Internal::AnglePyramid;
using Qi::Vision::Internal::AnglePyramidParams;

namespace {
bool g_debugCreateModel = false;
}

// =============================================================================
// String to Enum Conversion Helpers
// =============================================================================

namespace {

OptimizationMode ParseOptimization(const std::string& str) {
    std::string lower;
    lower.reserve(str.size());
    for (char c : str) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower.empty() || lower == "none") return OptimizationMode::None;
    if (lower == "auto") return OptimizationMode::Auto;
    if (lower == "point_reduction_low") return OptimizationMode::PointReductionLow;
    if (lower == "point_reduction_medium") return OptimizationMode::PointReductionMedium;
    if (lower == "point_reduction_high") return OptimizationMode::PointReductionHigh;
    throw InvalidArgumentException("Unknown optimization: " + str);
}

MetricMode ParseMetric(const std::string& str) {
    std::string lower;
    lower.reserve(str.size());
    for (char c : str) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower.empty() || lower == "use_polarity") return MetricMode::UsePolarity;
    if (lower == "ignore_global_polarity") return MetricMode::IgnoreGlobalPolarity;
    if (lower == "ignore_local_polarity") return MetricMode::IgnoreLocalPolarity;
    if (lower == "ignore_color_polarity") return MetricMode::IgnoreColorPolarity;
    throw InvalidArgumentException("Unknown metric: " + str);
}

SubpixelMethod ParseSubpixel(const std::string& str) {
    std::string lower;
    lower.reserve(str.size());
    for (char c : str) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower.empty() || lower == "least_squares") return SubpixelMethod::LeastSquares;
    if (lower == "none" || lower == "false") return SubpixelMethod::None;
    if (lower == "interpolation" || lower == "true") return SubpixelMethod::Parabolic;
    if (lower == "least_squares_high") return SubpixelMethod::LeastSquaresHigh;
    if (lower == "least_squares_very_high") return SubpixelMethod::LeastSquaresVeryHigh;
    // Decompiled numeric mode (a12 % 10) → SubpixelModeFromDecompiled
    if (!lower.empty() && lower[0] >= '0' && lower[0] <= '9') {
        int a12 = std::stoi(lower);
        return SubpixelModeFromDecompiled(a12 % 10);
    }
    throw InvalidArgumentException("Unknown subpixel mode: " + str);
}

/**
 * @brief Decode decompiled a12 parameter encoding
 *
 * Decompiled encoding (FindScaledShapeModel_Decompiled.md §2.2):
 *   subPixel = a12 % 10   (mode: 0=none, 1=LS, 2=bresenham, 3=jacobian, clamp >3→3)
 *   searchRadiusBase = a12 / 10   (per-level search radius base, clamp to 32)
 *
 * For string-based inputs ("least_squares", "interpolation", etc.),
 * only subpixel mode is extracted; searchRadiusBase defaults to 0
 * (matching decompiled a12<10 behavior: halving chain from 0 → intermediate=1, final=4).
 *
 * @param subPixel Input parameter string
 * @param[out] method  Decoded subpixel method
 * @param[out] searchRadiusBase  Decoded search radius base (0 = halving chain from 0)
 */
void DecodeA12Param(const std::string& subPixel,
                    SubpixelMethod& method, int32_t& searchRadiusBase) {
    searchRadiusBase = 0;

    std::string lower;
    lower.reserve(subPixel.size());
    for (char c : subPixel) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (!lower.empty() && lower[0] >= '0' && lower[0] <= '9') {
        // Decompiled a12 encoding: mode + searchRadiusBase * 10
        int a12 = std::stoi(lower);
        int mode = a12 % 10;
        if (mode > 3) mode = 3;  // clamp per decompiled
        method = SubpixelModeFromDecompiled(mode);
        searchRadiusBase = std::min(a12 / 10, 32);  // clamp to 32
    } else {
        // String-based mode (backward compatible)
        method = ParseSubpixel(subPixel);
    }
}

// Shape search accepts any valid image (empty = no results)
inline bool RequireValidImage(const QImage& image, const char* funcName) {
    return Validate::RequireImageValid(image, funcName);
}

// Shape model creation requires non-empty grayscale image (UInt8 or Float32)
inline void RequireTemplateImage(const QImage& image, const char* funcName) {
    Validate::RequireImageNonEmpty(image, funcName);
    static const PixelType allowed[] = { PixelType::UInt8, PixelType::Float32 };
    Validate::RequireImageTypeOneOf(image, allowed, 2, funcName);
    Validate::RequireChannelCountExact(image, 1, funcName);
}

std::string MetricToString(MetricMode mode) {
    switch (mode) {
        case MetricMode::UsePolarity: return "use_polarity";
        case MetricMode::IgnoreGlobalPolarity: return "ignore_global_polarity";
        case MetricMode::IgnoreLocalPolarity: return "ignore_local_polarity";
        case MetricMode::IgnoreColorPolarity: return "ignore_color_polarity";
        default: return "use_polarity";
    }
}

double EstimateAutoMinContrastFromPyramid(const AnglePyramid& pyramid) {
    const auto& level0 = pyramid.GetLevel(0);
    const QImage& mag = level0.gradMag;
    if (!mag.IsValid() || mag.Empty()) {
        return 10.0;
    }

    const int32_t width = mag.Width();
    const int32_t height = mag.Height();
    if (width <= 0 || height <= 0) {
        return 10.0;
    }

    const double targetSamples = 20000.0;
    const double total = static_cast<double>(width) * static_cast<double>(height);
    const int32_t step = std::max(1, static_cast<int32_t>(std::sqrt(total / targetSamples)));

    std::vector<float> samples;
    samples.reserve(static_cast<size_t>((width / step + 1) * (height / step + 1)));

    for (int32_t y = 0; y < height; y += step) {
        const float* row = static_cast<const float*>(mag.RowPtr(y));
        for (int32_t x = 0; x < width; x += step) {
            samples.push_back(row[x]);
        }
    }

    if (samples.empty()) {
        return 10.0;
    }

    const size_t idx90 = static_cast<size_t>(samples.size() * 0.90);
    std::nth_element(samples.begin(), samples.begin() + idx90, samples.end());
    const float p90 = samples[idx90];

    const size_t idx50 = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + idx50, samples.end());
    const float p50 = samples[idx50];

    double minContrast = std::max(3.0, std::min(static_cast<double>(p90) * 0.35,
                                                static_cast<double>(p90) * 0.70));
    minContrast = std::max(minContrast, static_cast<double>(p50) * 0.80);
    if (!std::isfinite(minContrast) || minContrast <= 0.0) {
        minContrast = 10.0;
    }

    return minContrast;
}

inline void ValidateLevels(int32_t numLevels, const char* funcName) {
    Validate::RequireNonNegative(numLevels, "numLevels", funcName);
}

inline void ValidateAngleStep(double angleStep, const char* funcName) {
    Validate::RequireNonNegative(angleStep, "angleStep", funcName);
}



/**
 * @brief Parse contrast parameter string
 *
 * Supports Halcon-style contrast formats:
 * - "auto" / "auto_contrast" / "auto_contrast_hyst" / "auto_min_size":
 *   All map to ContrastMode::Auto with contrastHigh=0. ExtractEdgeLevels
 *   uses decompiled default thresholds (high=2.0, low=1.0).
 * - Numeric string (e.g., "30"): Manual single threshold
 * - "[low,high]" (e.g., "[10,30]"): Manual hysteresis thresholds
 * - "[low,high,minSize]" (e.g., "[10,30,15]"): Hysteresis + min component filter
 *
 * @param str Contrast parameter string
 * @param[out] mode Contrast mode
 * @param[out] contrastHigh High threshold value (0 for auto modes)
 * @param[out] contrastLow Low threshold value (for hysteresis)
 * @param[out] minComponentSize Minimum component size (for filtering)
 */
void ParseContrast(const std::string& str, ContrastMode& mode,
                   double& contrastHigh, double& contrastLow, int32_t& minComponentSize) {
    // Default values
    mode = ContrastMode::Manual;
    contrastHigh = 30.0;
    contrastLow = 0.0;
    minComponentSize = 3;

    if (str.empty()) {
        return;
    }
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Check for auto modes
    // Note: all auto variants currently resolve to the same behavior —
    // contrastHigh=0 → ExtractEdgeLevels uses decompiled default thresholds (high=2, low=1).
    // Separate ContrastMode enums are preserved for future auto-contrast implementation.
    if (lower == "auto" || lower == "auto_contrast") {
        mode = ContrastMode::Auto;
        contrastHigh = 0.0;
        contrastLow = 0.0;
        return;
    }

    if (lower == "auto_contrast_hyst" || lower == "auto_hyst") {
        mode = ContrastMode::Auto;  // same as "auto" (no independent hysteresis auto-detect yet)
        contrastHigh = 0.0;
        contrastLow = 0.0;
        return;
    }

    if (lower == "auto_min_size") {
        mode = ContrastMode::Auto;  // same as "auto" (no independent min-size auto-detect yet)
        contrastHigh = 0.0;
        contrastLow = 0.0;
        return;
    }

    // Check for [low,high] or [low,high,minSize] format
    if (lower.size() > 2 && lower.front() == '[' && lower.back() == ']') {
        std::string inner = lower.substr(1, lower.size() - 2);

        // Split by commas
        std::vector<std::string> parts;
        size_t start = 0;
        size_t pos;
        while ((pos = inner.find(',', start)) != std::string::npos) {
            parts.push_back(inner.substr(start, pos - start));
            start = pos + 1;
        }
        parts.push_back(inner.substr(start));

        try {
            if (parts.size() >= 2) {
                contrastLow = std::stod(parts[0]);
                contrastHigh = std::stod(parts[1]);
                mode = ContrastMode::Manual;

                if (parts.size() >= 3) {
                    minComponentSize = std::stoi(parts[2]);
                }
                return;
            }
        } catch (...) {
            // Parse error, fall through to single value
        }
    }

    // Try to parse as single numeric value
    try {
        contrastHigh = std::stod(lower);
        contrastLow = 0.0;
        mode = ContrastMode::Manual;
    } catch (...) {
        throw InvalidArgumentException("Invalid contrast format: " + str);
    }
}

} // anonymous namespace

// =============================================================================
// ShapeModel Class (Handle)
// =============================================================================

ShapeModel::ShapeModel() : impl_(std::make_unique<Internal::ShapeModelImpl>()) {}

ShapeModel::~ShapeModel() = default;

ShapeModel::ShapeModel(const ShapeModel& other)
    : impl_(other.impl_ ? std::make_unique<Internal::ShapeModelImpl>(*other.impl_) : nullptr) {}

ShapeModel::ShapeModel(ShapeModel&& other) noexcept = default;

ShapeModel& ShapeModel::operator=(const ShapeModel& other) {
    if (this != &other) {
        impl_ = other.impl_ ? std::make_unique<Internal::ShapeModelImpl>(*other.impl_) : nullptr;
    }
    return *this;
}

ShapeModel& ShapeModel::operator=(ShapeModel&& other) noexcept = default;

bool ShapeModel::IsValid() const {
    return impl_ && impl_->valid_;
}

// =============================================================================
// Model Creation Functions
// =============================================================================

void CreateShapeModel(
    const QImage& templateImage,
    ShapeModel& model,
    int32_t numLevels,
    double angleStart,
    double angleExtent,
    double angleStep,
    const std::string& optimization,
    const std::string& metric,
    const std::string& contrast,
    double minContrast,
    int32_t numAngleBins)
{
    RequireTemplateImage(templateImage, "CreateShapeModel");
    ValidateLevels(numLevels, "CreateShapeModel");
    ValidateAngleStep(angleStep, "CreateShapeModel");

    // No-ROI version: pass empty QRegion (CreateModel handles full-image path internally)
    model = ShapeModel();

    ModelParams params;
    params.numLevels = numLevels;
    params.angleStart = angleStart;
    params.angleExtent = angleExtent;
    params.angleStep = angleStep;
    params.optimization = ParseOptimization(optimization);
    params.metric = ParseMetric(metric);

    ParseContrast(contrast, params.contrastMode, params.contrastHigh, params.contrastLow, params.minComponentSize);
    params.minContrast = minContrast;
    params.numAngleBins = std::clamp(numAngleBins, 1, 128);

    model.Impl()->params_ = params;
    model.Impl()->timingParams_.debugCreateModel = g_debugCreateModel;

    Point2d origin{templateImage.Width() / 2.0, templateImage.Height() / 2.0};
    if (!model.Impl()->CreateModel(templateImage, QRegion{}, origin)) {
        model = ShapeModel();
    }
}

void CreateShapeModel(
    const QImage& templateImage,
    const QRegion& region,
    ShapeModel& model,
    int32_t numLevels,
    double angleStart,
    double angleExtent,
    double angleStep,
    const std::string& optimization,
    const std::string& metric,
    const std::string& contrast,
    double minContrast,
    int32_t numAngleBins)
{
    RequireTemplateImage(templateImage, "CreateShapeModel");
    if (region.Empty()) {
        throw InvalidArgumentException("CreateShapeModel: empty region");
    }
    ValidateLevels(numLevels, "CreateShapeModel");
    ValidateAngleStep(angleStep, "CreateShapeModel");
    model = ShapeModel();

    // Set up parameters
    ModelParams params;
    params.numLevels = numLevels;
    params.angleStart = angleStart;
    params.angleExtent = angleExtent;
    params.angleStep = angleStep;
    params.optimization = ParseOptimization(optimization);
    params.metric = ParseMetric(metric);

    // Parse contrast parameter
    ParseContrast(contrast, params.contrastMode, params.contrastHigh, params.contrastLow, params.minComponentSize);
    params.minContrast = minContrast;
    params.numAngleBins = std::clamp(numAngleBins, 1, 128);

    model.Impl()->params_ = params;

    // Region path uses local template coordinates (bbox-cropped), so default
    // origin should be the local template center instead of full-image center.
    const Rect2i bbox = region.BoundingBox();
    Point2d origin{bbox.width / 2.0, bbox.height / 2.0};
    if (!model.Impl()->CreateModel(templateImage, region, origin)) {
        model = ShapeModel();  // Set to invalid model
    }
    if (g_debugCreateModel && model.IsValid()) {
        auto* impl = model.Impl();
        std::printf("[CreateShapeModel] contrast=[%.2f, %.2f] minContrast=%.2f points=%zu\n",
                    impl->params_.contrastLow, impl->params_.contrastHigh, impl->params_.minContrast,
                    impl->levels_.empty() ? 0u : impl->levels_[0].points.size());
        std::fflush(stdout);
    }
}

void CreateScaledShapeModel(
    const QImage& templateImage,
    ShapeModel& model,
    int32_t numLevels,
    double angleStart,
    double angleExtent,
    double angleStep,
    double scaleMin,
    double scaleMax,
    double scaleStep,
    const std::string& optimization,
    const std::string& metric,
    const std::string& contrast,
    double minContrast,
    int32_t numAngleBins)
{
    RequireTemplateImage(templateImage, "CreateScaledShapeModel");
    ValidateLevels(numLevels, "CreateScaledShapeModel");
    ValidateAngleStep(angleStep, "CreateScaledShapeModel");
    if (scaleMin <= 0.0 || scaleMax <= 0.0 || scaleStep < 0.0 || scaleMax < scaleMin) {
        throw InvalidArgumentException("CreateScaledShapeModel: invalid scale range");
    }
    // No-ROI version: pass empty QRegion (CreateModel handles full-image path internally)
    model = ShapeModel();

    double actualScaleStep = scaleStep;
    if (actualScaleStep <= 0.0) {
        if (scaleMax > scaleMin) {
            actualScaleStep = (scaleMax - scaleMin) / 10.0;
            actualScaleStep = std::max(0.01, actualScaleStep);
        } else {
            actualScaleStep = 0.01;
        }
    }

    ModelParams params;
    params.numLevels = numLevels;
    params.angleStart = angleStart;
    params.angleExtent = angleExtent;
    params.angleStep = angleStep;
    params.scaleMin = scaleMin;
    params.scaleMax = scaleMax;
    params.scaleStep = actualScaleStep;
    params.optimization = ParseOptimization(optimization);
    params.metric = ParseMetric(metric);

    ParseContrast(contrast, params.contrastMode, params.contrastHigh, params.contrastLow, params.minComponentSize);
    params.minContrast = minContrast;
    params.numAngleBins = std::clamp(numAngleBins, 1, 128);

    model.Impl()->params_ = params;
    model.Impl()->timingParams_.debugCreateModel = g_debugCreateModel;

    Point2d origin{templateImage.Width() / 2.0, templateImage.Height() / 2.0};
    if (!model.Impl()->CreateModel(templateImage, QRegion{}, origin)) {
        model = ShapeModel();
    }
}

void CreateScaledShapeModel(
    const QImage& templateImage,
    const QRegion& region,
    ShapeModel& model,
    int32_t numLevels,
    double angleStart,
    double angleExtent,
    double angleStep,
    double scaleMin,
    double scaleMax,
    double scaleStep,
    const std::string& optimization,
    const std::string& metric,
    const std::string& contrast,
    double minContrast,
    int32_t numAngleBins)
{
    RequireTemplateImage(templateImage, "CreateScaledShapeModel");
    if (region.Empty()) {
        throw InvalidArgumentException("CreateScaledShapeModel: empty region");
    }
    ValidateLevels(numLevels, "CreateScaledShapeModel");
    ValidateAngleStep(angleStep, "CreateScaledShapeModel");
    if (scaleMin <= 0.0 || scaleMax <= 0.0 || scaleStep < 0.0 || scaleMax < scaleMin) {
        throw InvalidArgumentException("CreateScaledShapeModel: invalid scale range");
    }
    model = ShapeModel();

    // Auto-compute scaleStep if 0 (Halcon convention: 0 = auto)
    double actualScaleStep = scaleStep;
    if (actualScaleStep <= 0.0) {
        if (scaleMax > scaleMin) {
            actualScaleStep = (scaleMax - scaleMin) / 10.0;
            actualScaleStep = std::max(0.01, actualScaleStep);
        } else {
            actualScaleStep = 0.01;
        }
    }

    ModelParams params;
    params.numLevels = numLevels;
    params.angleStart = angleStart;
    params.angleExtent = angleExtent;
    params.angleStep = angleStep;
    params.scaleMin = scaleMin;
    params.scaleMax = scaleMax;
    params.scaleStep = actualScaleStep;
    params.optimization = ParseOptimization(optimization);
    params.metric = ParseMetric(metric);

    // Parse contrast parameter (supports "auto", numeric, "[low,high]", or "[low,high,minSize]")
    ParseContrast(contrast, params.contrastMode, params.contrastHigh, params.contrastLow, params.minComponentSize);
    params.minContrast = minContrast;
    params.numAngleBins = std::clamp(numAngleBins, 1, 128);

    model.Impl()->params_ = params;
    model.Impl()->timingParams_.debugCreateModel = g_debugCreateModel;

    const Rect2i bbox = region.BoundingBox();
    Point2d origin{bbox.width / 2.0, bbox.height / 2.0};
    if (!model.Impl()->CreateModel(templateImage, region, origin)) {
        model = ShapeModel();
    }
    if (g_debugCreateModel && model.IsValid()) {
        auto* impl = model.Impl();
        std::printf("[CreateScaledShapeModel] contrast=[%.2f, %.2f] minContrast=%.2f points=%zu\n",
                    impl->params_.contrastLow, impl->params_.contrastHigh, impl->params_.minContrast,
                    impl->levels_.empty() ? 0u : impl->levels_[0].points.size());
        std::fflush(stdout);
    }
}

// =============================================================================
// Search Mask Helpers (decompiled find_shape_model_2: sub_180038450)
// =============================================================================

namespace {

// Simple 2x downsample for uint8 mask:
// 2x2 block sum, threshold at 256 (need ≥2 of 4 pixels non-zero to keep).
// Approximation of decompiled cv::pyrDown (5x5 Gaussian); differs by ~1px at mask boundary.
static QImage DownsampleMask(const QImage& mask, int32_t dstW, int32_t dstH) {
    QImage dst(dstW, dstH, PixelType::UInt8);
    const uint8_t* src = static_cast<const uint8_t*>(mask.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());
    const int32_t srcW = mask.Width();
    const int32_t srcH = mask.Height();
    const int32_t srcStride = mask.Stride();
    const int32_t dstStride = dst.Stride();

    for (int32_t dy = 0; dy < dstH; ++dy) {
        const int32_t sy = dy * 2;
        for (int32_t dx = 0; dx < dstW; ++dx) {
            const int32_t sx = dx * 2;
            // 2x2 block: conservative — if ANY source pixel is 0, output is 0
            int32_t sum = 0;
            sum += src[sy * srcStride + sx];
            if (sx + 1 < srcW) sum += src[sy * srcStride + sx + 1];
            if (sy + 1 < srcH) sum += src[(sy + 1) * srcStride + sx];
            if (sx + 1 < srcW && sy + 1 < srcH) sum += src[(sy + 1) * srcStride + sx + 1];
            dstData[dy * dstStride + dx] = (sum >= 128 * 2) ? 255 : 0;
        }
    }
    return dst;
}

// Zero out gradient (cosGrad/sinGrad) where mask pixel == 0.
static void MaskGradientLevel(Internal::AnglePyramid& pyramid,
                               int32_t level, const QImage& mask) {
    const float* gxConst; const float* gyConst;
    int32_t w, h, stride;
    if (!pyramid.GetGradientData(level, gxConst, gyConst, w, h, stride)) return;
    if (mask.Width() != w || mask.Height() != h) return;

    // GetGradientData returns const pointers, but we need to modify in-place.
    // The pyramid is a local variable in FindShapeModel, safe to const_cast.
    float* gx = const_cast<float*>(gxConst);
    float* gy = const_cast<float*>(gyConst);
    const int32_t maskStride = mask.Stride();
    const uint8_t* maskData = static_cast<const uint8_t*>(mask.Data());

    for (int32_t row = 0; row < h; ++row) {
        const uint8_t* maskRow = maskData + row * maskStride;
        float* gxRow = gx + row * stride;
        float* gyRow = gy + row * stride;
        for (int32_t col = 0; col < w; ++col) {
            if (maskRow[col] == 0) {
                gxRow[col] = 0.0f;
                gyRow[col] = 0.0f;
            }
        }
    }
}

// Decompiled find_shape_model_2: sub_180038450 -> sub_1800385C0
// Zero out gradient where mask pixel == 0.
// Mask pyramid: max 2 levels (decompiled: min(numLevels, 2)).
static void ApplySearchMask(Internal::AnglePyramid& pyramid,
                             const QImage& mask, int32_t numLevels) {
    const int32_t maskLevels = std::min(numLevels, 2);

    // Level 0: apply original mask directly
    if (maskLevels >= 1) {
        MaskGradientLevel(pyramid, 0, mask);
    }

    // Level 1: downsample mask by 2x, then apply
    if (maskLevels >= 2) {
        const float* gx1; const float* gy1;
        int32_t w1, h1, s1;
        if (pyramid.GetGradientData(1, gx1, gy1, w1, h1, s1)) {
            QImage maskDown = DownsampleMask(mask, w1, h1);
            MaskGradientLevel(pyramid, 1, maskDown);
        }
    }
}

} // anonymous namespace

// =============================================================================
// Model Search Functions
// =============================================================================

// Unified FindShapeModel: supports optional searchMask and startLevel
void FindShapeModel(
    const QImage& image,
    const ShapeModel& model,
    double angleStart,
    double angleExtent,
    double minScore,
    int32_t numMatches,
    double maxOverlap,
    const std::string& subPixel,
    int32_t numLevels,
    double greediness,
    std::vector<double>& rows,
    std::vector<double>& cols,
    std::vector<double>& angles,
    std::vector<double>& scores,
    const QImage& searchMask,
    int32_t startLevel)
{
    // Clear outputs
    rows.clear();
    cols.clear();
    angles.clear();
    scores.clear();

    if (!RequireValidImage(image, "FindShapeModel")) {
        return;
    }

    if (!model.IsValid()) {
        throw InvalidArgumentException("FindShapeModel: invalid ShapeModel");
    }
    if (!std::isfinite(angleStart) || !std::isfinite(angleExtent)) {
        throw InvalidArgumentException("FindShapeModel: invalid angle range");
    }
    if (!std::isfinite(minScore) || minScore < 0.0) {
        throw InvalidArgumentException("FindShapeModel: minScore must be >= 0");
    }
    if (!std::isfinite(maxOverlap) || maxOverlap < 0.0 || maxOverlap > 1.0) {
        throw InvalidArgumentException("FindShapeModel: maxOverlap must be in [0,1]");
    }
    if (!std::isfinite(greediness) || greediness <= 0.0 || greediness > 1.0) {
        throw InvalidArgumentException("FindShapeModel: greediness must be in (0,1]");
    }
    if (startLevel < 0) {
        throw InvalidArgumentException("FindShapeModel: startLevel must be >= 0");
    }

    // Validate search mask if provided
    if (!searchMask.Empty()) {
        if (searchMask.Width() != image.Width() || searchMask.Height() != image.Height()) {
            throw InvalidArgumentException(
                "FindShapeModel: searchMask must have same size as image");
        }
        if (searchMask.Type() != PixelType::UInt8) {
            throw InvalidArgumentException(
                "FindShapeModel: searchMask must be UInt8");
        }
    }

    ValidateLevels(numLevels, "FindShapeModel");

    const auto* impl = model.Impl();

    // Decode decompiled a12 parameter: subpixel mode + searchRadiusBase
    SubpixelMethod subpixelMethod;
    int32_t decodedSearchRadius;
    DecodeA12Param(subPixel, subpixelMethod, decodedSearchRadius);

    // Set up search parameters
    SearchParams params;
    params.angleStart = angleStart;
    params.angleExtent = angleExtent;
    params.minScore = minScore;
    params.maxMatches = numMatches;
    params.maxOverlap = maxOverlap;
    params.subpixelMethod = subpixelMethod;
    params.greediness = greediness;
    params.numLevels = numLevels;
    params.startLevel = startLevel;                 // decompiled a13[1]
    params.searchRadiusBase = decodedSearchRadius;  // decompiled a12/10

    // Build target pyramid
    Internal::AnglePyramidParams pyramidParams;
    int32_t modelLevels = static_cast<int32_t>(impl->levels_.size());
    pyramidParams.numLevels = (numLevels > 0) ? std::min(numLevels, modelLevels) : modelLevels;

    // Decompiled mode-3 special case (FindShapeModel only):
    // if (v30 == 3 && v33 <= 0 && v31 - v34 > 1) v504 = v35 + 1;
    // When subPixel mode 3 and startLevel <= 0 and >1 levels of refinement,
    // bump startLevel by 1 (skip finest level in pyramid refinement)
    if (subpixelMethod == SubpixelMethod::LeastSquaresHigh &&
        params.startLevel <= 0 &&
        pyramidParams.numLevels - 1 - params.startLevel > 1) {
        params.startLevel += 1;
    }

    pyramidParams.minContrast = 1.0;
    pyramidParams.smoothSigma = 0.5;
    pyramidParams.extractEdgePoints = false;
    pyramidParams.storeDirection = false;

    Internal::AnglePyramid targetPyramid;
    if (!targetPyramid.Build(image, pyramidParams)) {
        return;
    }

    // Apply search mask to gradient data (decompiled find_shape_model_2: sub_180038450)
    // Mask pyramid: max 2 levels. Zero out gradX/gradY where mask==0.
    // This ensures masked regions don't contribute to refinement scoring.
    // Coarse search uses angleBinImage (computed during Build, before masking), so unaffected.
    if (!searchMask.Empty()) {
        ApplySearchMask(targetPyramid, searchMask, pyramidParams.numLevels);
    }

    if (impl->params_.minContrast <= 0.0) {
        impl->searchMinContrast_ = EstimateAutoMinContrastFromPyramid(targetPyramid);
        if (impl->timingParams_.debugCreateModel) {
            std::printf("[Find] auto minContrast=%.2f\n", impl->searchMinContrast_);
        }
    } else {
        impl->searchMinContrast_ = impl->params_.minContrast;
    }

    std::vector<MatchResult> results = impl->SearchPyramid(targetPyramid, params);

    impl->searchMinContrast_ = -1.0;  // Reset override

    // Convert results to output vectors
    rows.reserve(results.size());
    cols.reserve(results.size());
    angles.reserve(results.size());
    scores.reserve(results.size());

    for (const auto& r : results) {
        rows.push_back(r.y);    // Halcon uses row (y) first
        cols.push_back(r.x);    // then column (x)
        angles.push_back(r.angle);
        scores.push_back(r.score);
    }
}

// =============================================================================
// SpatialNMSCluster — Decompiled sub_18004B100
// Spatial NMS + angle/scale distance suppression + clustering
// Scaled path post-processing (replaces NonMaxSuppressionOverlap)
// =============================================================================
namespace {

std::vector<MatchResult> SpatialNMSCluster(
    const std::vector<MatchResult>& matches,
    int32_t imageWidth,
    double overlapAngle,     // decompiled a5: angle distance threshold factor
    double overlapScale,     // decompiled a6: scale distance threshold factor
    int32_t maxMatchesPerCluster)
{
    if (matches.empty()) return {};

    // Decompiled constants (IDA-confirmed values)
    constexpr double ANGLE_DIST_SCALE = 2.5;   // qword_1800D6B48
    constexpr double SCALE_DIST_SCALE = 1.1;   // qword_1800D6B10
    constexpr double RAD2DEG = 180.0 / 3.14159265358979323846;

    const double angleThreshold = overlapAngle * ANGLE_DIST_SCALE;  // degrees
    const double scaleThreshold = overlapScale * SCALE_DIST_SCALE;

    // Spatial key: col + row * imageWidth
    // Decompiled uses integer col/row fields from Match40B struct (truncated, not rounded).
    const int64_t W = static_cast<int64_t>(imageWidth);
    auto toKey = [W](double x, double y) -> int64_t {
        int32_t col = static_cast<int32_t>(x);
        int32_t row = static_cast<int32_t>(y);
        return static_cast<int64_t>(col) + static_cast<int64_t>(row) * W;
    };

    // --- Phase 1: Build spatial hash ---
    std::unordered_multimap<int64_t, size_t> posMap;
    std::unordered_map<int64_t, double> bestScoreMap;

    for (size_t i = 0; i < matches.size(); ++i) {
        int64_t key = toKey(matches[i].x, matches[i].y);
        posMap.emplace(key, i);
        auto it = bestScoreMap.find(key);
        if (it == bestScoreMap.end() || matches[i].score > it->second) {
            bestScoreMap[key] = matches[i].score;
        }
    }

    // 8-neighbor offset table (decompiled v130)
    const int64_t offsets[9] = { 0, -W, 1-W, 1, W+1, W, W-1, -1, -(W+1) };

    // --- Phase 2: 8-neighbor local maximum suppression ---
    std::vector<std::pair<int64_t, double>> localMaxima;
    for (const auto& [key, score] : bestScoreMap) {
        bool isMax = true;
        for (int n = 1; n <= 8; ++n) {
            auto nit = bestScoreMap.find(key + offsets[n]);
            if (nit != bestScoreMap.end() && nit->second > score) {
                isMax = false;
                break;
            }
        }
        if (isMax) {
            localMaxima.emplace_back(key, score);
        }
    }

    // Process highest-scoring local maxima first
    std::sort(localMaxima.begin(), localMaxima.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // --- Phase 3 + 4: Per-neighborhood angle/scale suppression + clustering ---
    std::vector<std::vector<MatchResult>> clusters;

    for (const auto& [maxKey, maxScore] : localMaxima) {
        // Collect all matches from 9-cell neighborhood and consume (erase) entries
        // Decompiled sub_18004B100: collected matches are removed from the hash to prevent
        // the same match being processed by a subsequent local maximum's neighborhood.
        std::vector<std::pair<size_t, double>> neighborhood;
        for (int n = 0; n < 9; ++n) {
            int64_t nKey = maxKey + offsets[n];
            auto range = posMap.equal_range(nKey);
            for (auto it = range.first; it != range.second; ++it) {
                neighborhood.emplace_back(it->second, matches[it->second].score);
            }
            posMap.erase(nKey);
        }
        if (neighborhood.empty()) continue;

        // Sort by score descending
        std::sort(neighborhood.begin(), neighborhood.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        // Sequential angle+scale suppression (greedy NMS within cluster)
        std::vector<MatchResult> survivors;
        for (const auto& [idx, score] : neighborhood) {
            bool suppress = false;
            for (const auto& kept : survivors) {
                // Angle difference in degrees, normalized to [-180, 180)
                double angleDiff = (matches[idx].angle - kept.angle) * RAD2DEG;
                while (angleDiff < -180.0) angleDiff += 360.0;
                while (angleDiff >= 180.0) angleDiff -= 360.0;
                angleDiff = std::abs(angleDiff);

                if (!::Qi::Vision::Internal::g_scaleDiag.disableAngleScaleSuppress) {
                    if (angleThreshold > angleDiff) {
                        double scaleDiff = std::abs(matches[idx].scaleX - kept.scaleX);
                        if (scaleThreshold > scaleDiff) {
                            suppress = true;
                            break;
                        }
                    }
                }
            }
            if (!suppress) {
                survivors.push_back(matches[idx]);
            }
        }

        if (!survivors.empty()) {
            clusters.push_back(std::move(survivors));
        }
    }

    // --- Phase 5: Output collection (per-cluster truncation) ---
    std::vector<MatchResult> output;
    for (auto& cluster : clusters) {
        int32_t count = static_cast<int32_t>(cluster.size());
        int32_t take = (maxMatchesPerCluster > 0)
            ? std::min(maxMatchesPerCluster, count) : count;
        for (int32_t i = 0; i < take; ++i) {
            output.push_back(std::move(cluster[i]));
        }
    }

    return output;
}

} // anonymous namespace

// Overload with explicit startLevel (decompiled a13[1])
// Unified FindScaledShapeModel: supports optional searchMask and startLevel
void FindScaledShapeModel(
    const QImage& image,
    const ShapeModel& model,
    double angleStart,
    double angleExtent,
    double scaleMin,
    double scaleMax,
    double minScore,
    int32_t numMatches,
    double maxOverlap,
    const std::string& subPixel,
    int32_t numLevels,
    double greediness,
    std::vector<double>& rows,
    std::vector<double>& cols,
    std::vector<double>& angles,
    std::vector<double>& scales,
    std::vector<double>& scores,
    const QImage& searchMask,
    int32_t startLevel)
{
    // Clear outputs
    rows.clear();
    cols.clear();
    angles.clear();
    scales.clear();
    scores.clear();

    if (!RequireValidImage(image, "FindScaledShapeModel")) {
        return;
    }

    if (!model.IsValid()) {
        throw InvalidArgumentException("FindScaledShapeModel: invalid ShapeModel");
    }
    if (!std::isfinite(angleStart) || !std::isfinite(angleExtent)) {
        throw InvalidArgumentException("FindScaledShapeModel: invalid angle range");
    }
    if (!std::isfinite(scaleMin) || !std::isfinite(scaleMax) || scaleMin <= 0.0 || scaleMax <= 0.0 ||
        scaleMax < scaleMin) {
        throw InvalidArgumentException("FindScaledShapeModel: invalid scale range");
    }
    if (!std::isfinite(minScore) || minScore < 0.0) {
        throw InvalidArgumentException("FindScaledShapeModel: minScore must be >= 0");
    }
    if (!std::isfinite(maxOverlap) || maxOverlap < 0.0 || maxOverlap > 1.0) {
        throw InvalidArgumentException("FindScaledShapeModel: maxOverlap must be in [0,1]");
    }
    if (!std::isfinite(greediness) || greediness <= 0.0 || greediness > 1.0) {
        throw InvalidArgumentException("FindScaledShapeModel: greediness must be in (0,1]");
    }
    if (startLevel < 0) {
        throw InvalidArgumentException("FindScaledShapeModel: startLevel must be >= 0");
    }

    // Validate search mask if provided
    if (!searchMask.Empty()) {
        if (searchMask.Width() != image.Width() || searchMask.Height() != image.Height()) {
            throw InvalidArgumentException(
                "FindScaledShapeModel: searchMask must have same size as image");
        }
        if (searchMask.Type() != PixelType::UInt8) {
            throw InvalidArgumentException(
                "FindScaledShapeModel: searchMask must be UInt8");
        }
    }

    ValidateLevels(numLevels, "FindScaledShapeModel");

    // Get model implementation
    const auto* impl = model.Impl();

    // Build pyramid for search image
    AnglePyramidParams pyramidParams;
    int32_t modelLevels = static_cast<int32_t>(impl->levels_.size());
    pyramidParams.numLevels = (numLevels > 0) ? std::min(numLevels, modelLevels) : modelLevels;
    pyramidParams.smoothSigma = 0.5;
    pyramidParams.minContrast = 1.0;
    pyramidParams.useNMS = true;
    pyramidParams.extractEdgePoints = false;
    pyramidParams.storeDirection = false;

    AnglePyramid targetPyramid;
    if (!targetPyramid.Build(image, pyramidParams)) {
        return;
    }

    // Apply search mask to gradient data (decompiled find_scaled_shape_model_2: sub_180038450)
    // Mask pyramid: max 2 levels. Zero out gradX/gradY where mask==0.
    if (!searchMask.Empty()) {
        ApplySearchMask(targetPyramid, searchMask, pyramidParams.numLevels);
    }

    if (impl->params_.minContrast <= 0.0) {
        impl->searchMinContrast_ = EstimateAutoMinContrastFromPyramid(targetPyramid);
        if (impl->timingParams_.debugCreateModel) {
            std::printf("[FindScaled] auto minContrast=%.2f\n", impl->searchMinContrast_);
        }
    } else {
        impl->searchMinContrast_ = impl->params_.minContrast;
    }

    // Decode decompiled a12 parameter: subpixel mode + searchRadiusBase
    SubpixelMethod subpixelMethod;
    int32_t decodedSearchRadius;
    DecodeA12Param(subPixel, subpixelMethod, decodedSearchRadius);

    SearchParams params;
    params.angleStart = angleStart;
    params.angleExtent = angleExtent;
    params.minScore = minScore;
    params.maxMatches = 0;              // Don't truncate inside FinalizeResults; truncate here
    params.maxOverlap = maxOverlap;
    params.subpixelMethod = subpixelMethod;
    params.numLevels = numLevels;
    params.startLevel = startLevel;                 // decompiled a13[1]
    params.searchRadiusBase = decodedSearchRadius;  // decompiled a12/10
    params.greediness = greediness;
    params.scaleMin = scaleMin;
    params.scaleMax = scaleMax;

    // Decompiled mode-3 special case: if (v35 == 3 && startLevel <= 0 && numLevels - startLevel > 1)
    // Bump startLevel by 1 (skip finest level in pyramid refinement, subPixel handles it)
    if (subpixelMethod == SubpixelMethod::LeastSquaresHigh &&
        params.startLevel <= 0 &&
        pyramidParams.numLevels - 1 - params.startLevel > 1) {
        params.startLevel += 1;
    }

    // Decompiled flow: PyramidRefine → sub_18004B100 (NMS) → SubPixelRefine → sort+truncate
    // SearchPyramid(skipSubPixel=true) returns raw PyramidRefine output (no FinalizeResults).
    auto allResults = impl->SearchPyramid(targetPyramid, params,
                                           /*skipSubPixel=*/true);

    // Step 7: sub_18004B100 — Spatial NMS + angle/scale distance suppression + clustering
    // Decompiled: overlapAngle=a11, overlapScale=a11 (Halcon API has single MaxOverlap param)
    int32_t imgWidth = image.Width();

    // Switch D: pre-NMS minScore gate
    if (Qi::Vision::Internal::g_scaleDiag.preNmsMinScoreGate) {
        allResults.erase(
            std::remove_if(allResults.begin(), allResults.end(),
                            [&](const MatchResult& m) { return m.score < params.minScore; }),
            allResults.end());
    }

    // Switch A: bypass SpatialNMSCluster
    if (!Qi::Vision::Internal::g_scaleDiag.bypassS7Cluster) {
        allResults = SpatialNMSCluster(allResults, imgWidth,
                                        maxOverlap, maxOverlap, numMatches);
    }

    // Step 8: SubPixelRefine — after NMS to avoid wasting compute on suppressed candidates
    int32_t spStartLevel = std::min(static_cast<int32_t>(impl->levels_.size()) - 1,
                                     targetPyramid.NumLevels() - 1);
    if (params.numLevels > 0) spStartLevel = std::min(spStartLevel, params.numLevels - 1);
    allResults = impl->SubPixelRefine(targetPyramid, spStartLevel,
                                       std::move(allResults), params);

    // Step 9: minScore filter + angle normalization + sort + truncate
    // (Decompiled: minScore filtering is implicit in PyramidRefine's levelThreshold;
    //  SubPixelRefine may change scores, so re-filter here)
    allResults.erase(
        std::remove_if(allResults.begin(), allResults.end(),
            [minScore](const MatchResult& m) { return m.score < minScore; }),
        allResults.end());
    for (auto& m : allResults) {
        while (m.angle > PI) m.angle -= 2.0 * PI;
        while (m.angle < -PI) m.angle += 2.0 * PI;
    }
    std::sort(allResults.begin(), allResults.end());
    if (numMatches > 0 && static_cast<int32_t>(allResults.size()) > numMatches) {
        allResults.resize(numMatches);
    }

    // Convert results to output vectors
    for (const auto& r : allResults) {
        rows.push_back(r.y);
        cols.push_back(r.x);
        angles.push_back(r.angle);
        scales.push_back(r.scaleX);
        scores.push_back(r.score);
    }

    impl->searchMinContrast_ = -1.0;  // Reset override
}

// =============================================================================
// FindShapeModels — Multi-model simultaneous search
// Halcon equivalent: find_shape_models
// =============================================================================

void FindShapeModels(
    const QImage& image,
    const std::vector<ShapeModel>& models,
    double angleStart,
    double angleExtent,
    double minScore,
    int32_t numMatches,
    double maxOverlap,
    const std::string& subPixel,
    const std::vector<int32_t>& numLevels,
    double greediness,
    std::vector<double>& rows,
    std::vector<double>& cols,
    std::vector<double>& angles,
    std::vector<double>& scores,
    std::vector<int32_t>& modelIndices,
    const QImage& searchMask)
{
    // Clear outputs
    rows.clear();
    cols.clear();
    angles.clear();
    scores.clear();
    modelIndices.clear();

    // Phase 0: Input validation
    if (!RequireValidImage(image, "FindShapeModels")) return;
    if (models.empty()) return;

    const int32_t modelCount = static_cast<int32_t>(models.size());

    // Validate all models
    std::vector<const Internal::ShapeModelImpl*> impls(modelCount);
    for (int32_t i = 0; i < modelCount; ++i) {
        if (!models[i].IsValid()) {
            throw InvalidArgumentException(
                "FindShapeModels: model[" + std::to_string(i) + "] is invalid");
        }
        impls[i] = models[i].Impl();
    }

    if (!std::isfinite(angleStart) || !std::isfinite(angleExtent))
        throw InvalidArgumentException("FindShapeModels: invalid angle range");
    if (!std::isfinite(minScore) || minScore < 0.0)
        throw InvalidArgumentException("FindShapeModels: minScore must be >= 0");
    if (!std::isfinite(maxOverlap) || maxOverlap < 0.0 || maxOverlap > 1.0)
        throw InvalidArgumentException("FindShapeModels: maxOverlap must be in [0,1]");
    if (!std::isfinite(greediness) || greediness <= 0.0 || greediness > 1.0)
        throw InvalidArgumentException("FindShapeModels: greediness must be in (0,1]");

    // Decode subpixel parameter (shared for all models)
    SubpixelMethod subpixelMethod;
    int32_t decodedSearchRadius;
    DecodeA12Param(subPixel, subpixelMethod, decodedSearchRadius);

    // Phase 0c: Compute per-model effective numLevels + global maxPyrLevels
    std::vector<int32_t> effectiveLevels(modelCount);
    int32_t maxPyrLevels = 0;
    for (int32_t i = 0; i < modelCount; ++i) {
        int32_t modelLvls = static_cast<int32_t>(impls[i]->levels_.size());
        int32_t userLvls = (i < static_cast<int32_t>(numLevels.size())) ? numLevels[i] : 0;
        if (userLvls > 0 && userLvls <= modelLvls)
            effectiveLevels[i] = userLvls;
        else
            effectiveLevels[i] = modelLvls;
        maxPyrLevels = std::max(maxPyrLevels, effectiveLevels[i]);
    }

    // Phase 1: Build shared pyramid (one build for all models)
    AnglePyramidParams pyramidParams;
    pyramidParams.numLevels = maxPyrLevels;
    pyramidParams.minContrast = 1.0;
    pyramidParams.smoothSigma = 0.5;
    pyramidParams.extractEdgePoints = false;
    pyramidParams.storeDirection = false;

    AnglePyramid sharedPyramid;
    if (!sharedPyramid.Build(image, pyramidParams)) {
        return;
    }

    // Apply search mask to gradient data (decompiled find_shape_models_2)
    // Mask pyramid: max 2 levels. Zero out gradX/gradY where mask==0.
    if (!searchMask.Empty()) {
        ApplySearchMask(sharedPyramid, searchMask, pyramidParams.numLevels);
    }

    // Phase 2-4: Per-model search using shared pyramid
    std::vector<MatchResult> allResults;

    for (int32_t mi = 0; mi < modelCount; ++mi) {
        SearchParams params;
        params.angleStart = angleStart;
        params.angleExtent = angleExtent;
        params.minScore = minScore;
        params.maxMatches = 0;         // Don't limit per-model (limit globally later)
        params.maxOverlap = 1.0;       // Don't NMS per-model (NMS globally later)
        params.subpixelMethod = subpixelMethod;
        params.greediness = greediness;
        params.numLevels = effectiveLevels[mi];
        params.startLevel = 0;
        params.searchRadiusBase = decodedSearchRadius;

        // Decompiled mode-3 special case: bump startLevel by 1
        if (subpixelMethod == SubpixelMethod::LeastSquaresHigh &&
            params.startLevel <= 0 &&
            effectiveLevels[mi] - 1 - params.startLevel > 1) {
            params.startLevel += 1;
        }

        // Set search-time minContrast override (mutable, thread-safe per-call)
        if (impls[mi]->params_.minContrast <= 0.0) {
            impls[mi]->searchMinContrast_ =
                EstimateAutoMinContrastFromPyramid(sharedPyramid);
        } else {
            impls[mi]->searchMinContrast_ = impls[mi]->params_.minContrast;
        }

        auto results = impls[mi]->SearchPyramid(sharedPyramid, params);

        impls[mi]->searchMinContrast_ = -1.0;  // Reset override

        // Tag modelIndex
        for (auto& r : results) {
            r.modelIndex = mi;
        }

        allResults.insert(allResults.end(),
                          std::make_move_iterator(results.begin()),
                          std::make_move_iterator(results.end()));
    }

    if (allResults.empty()) return;

    // Phase 5: Sort by score (descending) - global across all models
    std::sort(allResults.begin(), allResults.end());

    // Phase 6: Cross-model NMS using OBB overlap
    if (maxOverlap < 1.0 && allResults.size() > 1) {
        std::vector<MatchResult> nmsResults;
        nmsResults.reserve(allResults.size());

        struct CachedOBB {
            nms_detail::Vec2 corners[4];
            double area;
            double diagSq;   // half-diagonal squared for distance prefilter
            double cx, cy;
        };
        std::vector<CachedOBB> keptOBBs;

        for (const auto& match : allResults) {
            auto* impl = impls[match.modelIndex];

            // Model bounding box from impl's cached bounds
            double modelW = impl->modelMaxX_ - impl->modelMinX_;
            double modelH = impl->modelMaxY_ - impl->modelMinY_;
            double w = modelW * match.scaleX;
            double h = modelH * match.scaleY;
            double hw = w * 0.5, hh = h * 0.5;
            double area = w * h;

            if (area <= 0.0) {
                nmsResults.push_back(match);
                continue;
            }

            nms_detail::Vec2 corners[4];
            bool cornersComputed = false;
            bool suppress = false;

            // Distance prefilter: sum of both boxes' diagonal half-lengths squared
            double diagSq = hw * hw + hh * hh;

            for (size_t k = 0; k < keptOBBs.size(); ++k) {
                // Distance prefilter: if centers are farther than sum of diagonals, no overlap possible
                double dx = match.x - keptOBBs[k].cx;
                double dy = match.y - keptOBBs[k].cy;
                double distSq = dx * dx + dy * dy;
                double maxDist = diagSq + keptOBBs[k].diagSq + 2.0 * std::sqrt(diagSq * keptOBBs[k].diagSq);
                if (distSq >= maxDist) continue;

                // Lazy compute OBB corners
                if (!cornersComputed) {
                    nms_detail::GetOBBCorners(match.x, match.y, hw, hh,
                                              match.angle, corners);
                    cornersComputed = true;
                }

                double minArea = std::min(area, keptOBBs[k].area);
                if (minArea <= 0.0) continue;

                double interArea = nms_detail::OBBIntersectionArea(
                    corners, keptOBBs[k].corners);
                double overlapRatio = interArea / minArea;

                if (overlapRatio > maxOverlap) {
                    suppress = true;
                    break;
                }
            }

            if (!suppress) {
                nmsResults.push_back(match);
                CachedOBB obb;
                if (!cornersComputed) {
                    nms_detail::GetOBBCorners(match.x, match.y, hw, hh,
                                              match.angle, corners);
                }
                for (int i = 0; i < 4; ++i) obb.corners[i] = corners[i];
                obb.area = area;
                obb.diagSq = diagSq;
                obb.cx = match.x;
                obb.cy = match.y;
                keptOBBs.push_back(obb);
            }
        }

        allResults = std::move(nmsResults);
    }

    // Phase 7: Truncate to numMatches
    if (numMatches > 0 &&
        static_cast<int32_t>(allResults.size()) > numMatches) {
        allResults.resize(numMatches);
    }

    // Phase 8: Angle normalization + output
    rows.reserve(allResults.size());
    cols.reserve(allResults.size());
    angles.reserve(allResults.size());
    scores.reserve(allResults.size());
    modelIndices.reserve(allResults.size());

    for (auto& r : allResults) {
        r.angle = std::remainder(r.angle, 2.0 * PI);
        rows.push_back(r.y);
        cols.push_back(r.x);
        angles.push_back(r.angle);
        scores.push_back(r.score);
        modelIndices.push_back(r.modelIndex);
    }
}

// =============================================================================
// GetModelTransform (aligned with decompiled get_model_transform)
// =============================================================================

std::vector<ModelPoint> GetModelTransform(
    const ShapeModel& model,
    int32_t level,
    double angle,
    double scale)
{
    if (!model.IsValid()) {
        throw InvalidArgumentException("GetModelTransform: invalid ShapeModel");
    }

    auto* impl = model.Impl();
    int32_t actualLevel = (level >= 1) ? level - 1 : 0;

    if (actualLevel < 0 || actualLevel >= static_cast<int32_t>(impl->levels_.size())) {
        throw InvalidArgumentException("GetModelTransform: level out of range");
    }

    const auto& src = impl->levels_[actualLevel].points;
    if (src.empty()) return {};

    // Identity case: no transform needed
    if (std::abs(angle) < 1e-12 && std::abs(scale - 1.0) < 1e-12) {
        return src;
    }

    double cosA = std::cos(angle);
    double sinA = std::sin(angle);

    std::vector<ModelPoint> out;
    out.reserve(src.size());

    for (const auto& pt : src) {
        // Position: standard CCW rotation [cosA, -sinA; sinA, cosA]
        // Note: decompiled uses CW convention with degrees, but QiVision search kernel
        // returns CCW angles in radians, so GetModelTransform must match that convention.
        double rx = (cosA * pt.x - sinA * pt.y) * scale;
        double ry = (sinA * pt.x + cosA * pt.y) * scale;
        double ra = pt.angle + angle;
        out.emplace_back(rx, ry, ra, pt.magnitude, pt.angleBin, pt.weight);
    }

    return out;
}

// =============================================================================
// Model Property Functions
// =============================================================================

void GetShapeModelContours(
    const ShapeModel& model,
    int32_t level,
    std::vector<double>& contourRows,
    std::vector<double>& contourCols)
{
    contourRows.clear();
    contourCols.clear();

    if (!model.IsValid()) {
        throw InvalidArgumentException("GetShapeModelContours: invalid ShapeModel");
    }

    auto* impl = model.Impl();
    int32_t actualLevel = (level >= 1) ? level - 1 : 0;  // Halcon uses 1-based

    if (actualLevel < 0 || actualLevel >= static_cast<int32_t>(impl->levels_.size())) {
        throw InvalidArgumentException("GetShapeModelContours: level out of range");
    }

    const auto& points = impl->levels_[actualLevel].points;
    contourRows.reserve(points.size());
    contourCols.reserve(points.size());

    for (const auto& pt : points) {
        contourRows.push_back(pt.y);  // Halcon: row = y
        contourCols.push_back(pt.x);  // Halcon: col = x
    }
}

void GetShapeModelXLD(
    const ShapeModel& model,
    int32_t level,
    QContourArray& contours)
{
    contours = QContourArray();

    if (!model.IsValid()) {
        throw InvalidArgumentException("GetShapeModelXLD: invalid ShapeModel");
    }

    auto* impl = model.Impl();
    int32_t actualLevel = (level >= 1) ? level - 1 : 0;

    if (actualLevel < 0 || actualLevel >= static_cast<int32_t>(impl->levels_.size())) {
        throw InvalidArgumentException("GetShapeModelXLD: level out of range");
    }

    const auto& levelModel = impl->levels_[actualLevel];
    const auto& points = levelModel.points;
    const auto& contourStarts = levelModel.contourStarts;
    const auto& contourClosed = levelModel.contourClosed;

    if (points.empty()) {
        return;
    }

    // Use stored contour topology (from XLD tracing during model creation)
    if (!contourStarts.empty() && contourStarts.size() > 1) {
        // Proper contour topology available
        size_t numContours = contourStarts.size() - 1;  // Last value is sentinel

        for (size_t c = 0; c < numContours; ++c) {
            int32_t startIdx = contourStarts[c];
            int32_t endIdx = contourStarts[c + 1];

            if (endIdx <= startIdx) continue;

            QContour contour;
            for (int32_t i = startIdx; i < endIdx; ++i) {
                contour.AddPoint(points[i].x, points[i].y);
            }

            // Close contour if marked as closed
            if (c < contourClosed.size() && contourClosed[c] && contour.Size() > 2) {
                contour.AddPoint(points[startIdx].x, points[startIdx].y);
            }

            if (contour.Size() > 0) {
                contours.Add(contour);
            }
        }
    } else {
        // Fallback: no topology info, return points as single contour
        QContour contour;
        for (const auto& pt : points) {
            contour.AddPoint(pt.x, pt.y);
        }
        if (contour.Size() > 0) {
            contours.Add(contour);
        }
    }
}

void GetShapeModelParams(
    const ShapeModel& model,
    int32_t& numLevels,
    double& angleStart,
    double& angleExtent,
    double& angleStep,
    double& scaleMin,
    double& scaleMax,
    double& scaleStep,
    std::string& metric)
{
    if (!model.IsValid()) {
        throw InvalidArgumentException("GetShapeModelParams: invalid ShapeModel");
    }

    auto* impl = model.Impl();
    numLevels = static_cast<int32_t>(impl->levels_.size());
    angleStart = impl->params_.angleStart;
    angleExtent = impl->params_.angleExtent;
    angleStep = impl->params_.angleStep;
    scaleMin = impl->params_.scaleMin;
    scaleMax = impl->params_.scaleMax;
    scaleStep = impl->params_.scaleStep;
    metric = MetricToString(impl->params_.metric);
}

void GetShapeModelOrigin(
    const ShapeModel& model,
    double& row,
    double& col)
{
    if (!model.IsValid()) {
        throw InvalidArgumentException("GetShapeModelOrigin: invalid ShapeModel");
    }

    auto* impl = model.Impl();
    row = impl->origin_.y;  // Halcon: row = y
    col = impl->origin_.x;  // Halcon: col = x
}

void SetShapeModelOrigin(
    ShapeModel& model,
    double row,
    double col)
{
    if (!model.IsValid()) {
        throw InvalidArgumentException("SetShapeModelOrigin: invalid ShapeModel");
    }
    if (!std::isfinite(row) || !std::isfinite(col)) {
        throw InvalidArgumentException("SetShapeModelOrigin: invalid origin");
    }

    auto* impl = model.Impl();
    impl->origin_.y = row;
    impl->origin_.x = col;
}

// =============================================================================
// Model I/O Functions
// =============================================================================

void WriteShapeModel(
    const ShapeModel& model,
    const std::string& filename)
{
    if (!model.IsValid()) {
        throw InvalidArgumentException("WriteShapeModel: invalid ShapeModel");
    }
    if (filename.empty()) {
        throw InvalidArgumentException("WriteShapeModel: filename is empty");
    }

    auto* impl = model.Impl();

    using Platform::BinaryWriter;
    BinaryWriter writer(filename);
    if (!writer.IsOpen()) {
        throw IOException("Cannot open file for writing: " + filename);
    }

    // Magic number and version
    const uint32_t MAGIC = 0x4D495351;  // "QISM"
    const uint32_t VERSION = 5;
    writer.Write(MAGIC);
    writer.Write(VERSION);

    // Model parameters
    writer.Write(static_cast<int32_t>(impl->params_.contrastMode));
    writer.Write(impl->params_.contrastHigh);
    writer.Write(impl->params_.contrastLow);
    writer.Write(impl->params_.contrastMax);
    writer.Write(impl->params_.minComponentSize);
    writer.Write(impl->params_.minContrast);
    writer.Write(static_cast<int32_t>(impl->params_.optimization));
    writer.Write(impl->params_.pregeneration);
    writer.Write(static_cast<int32_t>(impl->params_.metric));
    writer.Write(impl->params_.numLevels);
    writer.Write(impl->params_.startLevel);
    writer.Write(impl->params_.angleStart);
    writer.Write(impl->params_.angleExtent);
    writer.Write(impl->params_.angleStep);
    writer.Write(impl->params_.scaleMin);
    writer.Write(impl->params_.scaleMax);
    writer.Write(impl->params_.scaleStep);
    writer.Write(static_cast<int32_t>(impl->params_.polarity));
    writer.Write(impl->numAngleBins_);  // v5: angle quantization bins

    // Origin and template size
    writer.Write(impl->origin_.x);
    writer.Write(impl->origin_.y);
    writer.Write(impl->templateSize_.width);
    writer.Write(impl->templateSize_.height);

    // Model bounds
    writer.Write(impl->modelMinX_);
    writer.Write(impl->modelMaxX_);
    writer.Write(impl->modelMinY_);
    writer.Write(impl->modelMaxY_);

    // Pyramid levels
    uint32_t numLevels = static_cast<uint32_t>(impl->levels_.size());
    writer.Write(numLevels);

    for (const auto& level : impl->levels_) {
        writer.Write(level.width);
        writer.Write(level.height);
        writer.Write(level.scale);

        uint32_t numPoints = static_cast<uint32_t>(level.points.size());
        writer.Write(numPoints);

        for (const auto& pt : level.points) {
            writer.Write(pt.x);
            writer.Write(pt.y);
            writer.Write(pt.angle);
            writer.Write(pt.magnitude);
            writer.Write(pt.angleBin);
            writer.Write(pt.weight);
            writer.Write(pt.cosAngle);
            writer.Write(pt.sinAngle);
        }
    }

    writer.Close();
}

void ReadShapeModel(
    const std::string& filename,
    ShapeModel& model)
{
    if (filename.empty()) {
        throw InvalidArgumentException("ReadShapeModel: filename is empty");
    }
    using Platform::BinaryReader;
    BinaryReader reader(filename);
    if (!reader.IsOpen()) {
        throw IOException("Cannot open file for reading: " + filename);
    }

    const uint32_t MAGIC = 0x4D495351;
    uint32_t magic = reader.Read<uint32_t>();
    if (magic != MAGIC) {
        throw InvalidArgumentException("Invalid shape model file format");
    }

    uint32_t version = reader.Read<uint32_t>();
    if (version < 1 || version > 5) {
        throw VersionMismatchException("Unsupported shape model version");
    }

    model = ShapeModel();
    auto* impl = model.Impl();

    if (version >= 3) {
        impl->params_.contrastMode = static_cast<ContrastMode>(reader.Read<int32_t>());
        impl->params_.contrastHigh = reader.Read<double>();
        impl->params_.contrastLow = reader.Read<double>();
        impl->params_.contrastMax = reader.Read<double>();
        impl->params_.minComponentSize = reader.Read<int32_t>();
        impl->params_.minContrast = reader.Read<double>();
        impl->params_.optimization = static_cast<OptimizationMode>(reader.Read<int32_t>());
        impl->params_.pregeneration = reader.Read<bool>();
        impl->params_.metric = static_cast<MetricMode>(reader.Read<int32_t>());
        impl->params_.numLevels = reader.Read<int32_t>();
        impl->params_.startLevel = reader.Read<int32_t>();
        impl->params_.angleStart = reader.Read<double>();
        impl->params_.angleExtent = reader.Read<double>();
        impl->params_.angleStep = reader.Read<double>();
        impl->params_.scaleMin = reader.Read<double>();
        impl->params_.scaleMax = reader.Read<double>();
        if (version >= 4) {
            impl->params_.scaleStep = reader.Read<double>();
        } else {
            impl->params_.scaleStep = 0.0;
        }
        impl->params_.polarity = static_cast<MatchPolarity>(reader.Read<int32_t>());
        if (version >= 5) {
            impl->numAngleBins_ = std::clamp(reader.Read<int32_t>(), 1, 128);
        } else {
            impl->numAngleBins_ = 16;  // default for v3/v4
        }
        impl->params_.numAngleBins = impl->numAngleBins_;
    } else {
        // Legacy v1/v2 format
        double minContrast = reader.Read<double>();
        double hysteresisContrast = 0.0;
        if (version >= 2) {
            hysteresisContrast = reader.Read<double>();
        }
        double maxContrast = reader.Read<double>();

        impl->params_.contrastMode = ContrastMode::Manual;
        impl->params_.contrastHigh = minContrast;
        impl->params_.contrastLow = hysteresisContrast;
        impl->params_.contrastMax = maxContrast;
        impl->params_.minContrast = minContrast;

        impl->params_.numLevels = reader.Read<int32_t>();
        impl->params_.startLevel = reader.Read<int32_t>();
        impl->params_.angleStart = reader.Read<double>();
        impl->params_.angleExtent = reader.Read<double>();
        impl->params_.scaleMin = reader.Read<double>();
        impl->params_.scaleMax = reader.Read<double>();
        impl->params_.scaleStep = 0.0;

        bool optimizeModel = reader.Read<bool>();
        (void)reader.Read<int32_t>();  // maxModelPoints
        (void)reader.Read<double>();   // modelPointSpacing

        impl->params_.optimization = optimizeModel
            ? OptimizationMode::Auto : OptimizationMode::None;

        impl->params_.polarity = static_cast<MatchPolarity>(reader.Read<int32_t>());
        impl->params_.metric = MetricMode::UsePolarity;
        impl->numAngleBins_ = 16;  // default for v1/v2
        impl->params_.numAngleBins = 16;
    }

    impl->origin_.x = reader.Read<double>();
    impl->origin_.y = reader.Read<double>();
    impl->templateSize_.width = reader.Read<int32_t>();
    impl->templateSize_.height = reader.Read<int32_t>();

    impl->modelMinX_ = reader.Read<double>();
    impl->modelMaxX_ = reader.Read<double>();
    impl->modelMinY_ = reader.Read<double>();
    impl->modelMaxY_ = reader.Read<double>();

    uint32_t numLevels = reader.Read<uint32_t>();
    impl->levels_.resize(numLevels);

    for (uint32_t i = 0; i < numLevels; ++i) {
        auto& level = impl->levels_[i];

        level.width = reader.Read<int32_t>();
        level.height = reader.Read<int32_t>();
        level.scale = reader.Read<double>();

        uint32_t numPoints = reader.Read<uint32_t>();
        level.points.resize(numPoints);

        for (uint32_t j = 0; j < numPoints; ++j) {
            auto& pt = level.points[j];
            pt.x = reader.Read<double>();
            pt.y = reader.Read<double>();
            pt.angle = reader.Read<double>();
            pt.magnitude = reader.Read<double>();
            pt.angleBin = reader.Read<int32_t>();
            pt.weight = reader.Read<double>();
            pt.cosAngle = reader.Read<double>();
            pt.sinAngle = reader.Read<double>();
        }

        level.BuildSoA();
    }

    impl->valid_ = true;

    // Rebuild search structures from loaded parameters
    impl->BuildCosLUT(impl->numAngleBins_);
    double angleExtent = (impl->params_.angleExtent > 0) ? impl->params_.angleExtent : 2.0 * 3.14159265358979323846;
    impl->BuildSearchAngleCache(impl->params_.angleStart, angleExtent, impl->params_.angleStep);
}

void ClearShapeModel(ShapeModel& model)
{
    if (model.Impl()) {
        model.Impl()->levels_.clear();
        model.Impl()->valid_ = false;
    }
}

void InspectShapeModel(
    const ShapeModel& model,
    int32_t level,
    QImage& contrastImage,
    int32_t& numPoints)
{
    contrastImage = QImage();
    numPoints = 0;

    if (!model.IsValid()) {
        return;
    }

    auto* impl = model.Impl();
    int32_t actualLevel = (level >= 1) ? level - 1 : 0;

    if (actualLevel < 0 || actualLevel >= static_cast<int32_t>(impl->levels_.size())) {
        return;
    }

    const auto& levelData = impl->levels_[actualLevel];
    numPoints = static_cast<int32_t>(levelData.points.size());

    // Create contrast visualization image
    int32_t w = levelData.width;
    int32_t h = levelData.height;
    if (w <= 0 || h <= 0) {
        return;
    }

    contrastImage = QImage(w, h, PixelType::UInt8, ChannelType::Gray);
    std::memset(contrastImage.Data(), 0, contrastImage.Height() * contrastImage.Stride());

    uint8_t* data = static_cast<uint8_t*>(contrastImage.Data());
    int32_t stride = static_cast<int32_t>(contrastImage.Stride());

    // Mark model points
    for (const auto& pt : levelData.points) {
        int32_t px = static_cast<int32_t>(pt.x + w / 2);
        int32_t py = static_cast<int32_t>(pt.y + h / 2);
        if (px >= 0 && px < w && py >= 0 && py < h) {
            data[py * stride + px] = 255;
        }
    }
}

void SetShapeModelDebugCreate(ShapeModel& model, bool enable)
{
    if (model.Impl()) {
        model.Impl()->timingParams_.debugCreateModel = enable;
    }
    g_debugCreateModel = enable;
}

void SetShapeModelDebugCreateGlobal(bool enable)
{
    g_debugCreateModel = enable;
    std::printf("[ShapeModel] debugCreateModel=%d\n", enable ? 1 : 0);
    std::fflush(stdout);
}

} // namespace Qi::Vision::Matching
