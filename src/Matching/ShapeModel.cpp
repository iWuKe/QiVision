/**
 * @file ShapeModel.cpp
 * @brief Halcon-style ShapeModel API implementation
 *
 * Provides Halcon-compatible free functions that wrap the internal
 * ShapeModelImpl class.
 */

#include "ShapeModelImpl.h"
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>
#include <QiVision/Core/QContourArray.h>
#include <QiVision/Platform/FileIO.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

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
 * only subpixel mode is extracted; searchRadiusBase defaults to 0.
 *
 * @param subPixel Input parameter string
 * @param[out] method  Decoded subpixel method
 * @param[out] searchRadiusBase  Decoded search radius base (0 = use model table)
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
// Model Search Functions
// =============================================================================

// Original API: delegates to overload with startLevel=0 (decompiled a13[1] default)
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
    std::vector<double>& scores)
{
    FindShapeModel(image, model, angleStart, angleExtent,
                   minScore, numMatches, maxOverlap,
                   subPixel, numLevels, /*startLevel=*/0, greediness,
                   rows, cols, angles, scores);
}

// Overload with explicit startLevel (decompiled a13[1])
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
    int32_t startLevel,
    double greediness,
    std::vector<double>& rows,
    std::vector<double>& cols,
    std::vector<double>& angles,
    std::vector<double>& scores)
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

    ValidateLevels(numLevels, "FindShapeModel");

    auto* impl = const_cast<Internal::ShapeModelImpl*>(model.Impl());

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

    // Build target pyramid (with timing)
    auto tPyramidStart = std::chrono::high_resolution_clock::now();

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

    // Keep pyramid minContrast low; scoring applies search-time thresholds
    pyramidParams.minContrast = 1.0;
    pyramidParams.smoothSigma = 0.5;
    pyramidParams.extractEdgePoints = false;
    pyramidParams.storeDirection = false;  // Search mode: skip storing gradDir

    Internal::AnglePyramid targetPyramid;
    if (!targetPyramid.Build(image, pyramidParams)) {
        return;
    }

    const double savedMinContrast = impl->params_.minContrast;
    if (savedMinContrast <= 0.0) {
        impl->params_.minContrast = EstimateAutoMinContrastFromPyramid(targetPyramid);
        if (impl->timingParams_.debugCreateModel) {
            std::printf("[Find] auto minContrast=%.2f\n", impl->params_.minContrast);
        }
    }

    auto tPyramidEnd = std::chrono::high_resolution_clock::now();

    std::vector<MatchResult> results = impl->SearchPyramid(targetPyramid, params);

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

    impl->params_.minContrast = savedMinContrast;
}

// Original API: delegates to overload with startLevel=0 (decompiled a13[1] default)
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
    std::vector<double>& scores)
{
    FindScaledShapeModel(image, model, angleStart, angleExtent,
                         scaleMin, scaleMax, minScore, numMatches, maxOverlap,
                         subPixel, numLevels, /*startLevel=*/0, greediness,
                         rows, cols, angles, scales, scores);
}

// Overload with explicit startLevel (decompiled a13[1])
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
    int32_t startLevel,
    double greediness,
    std::vector<double>& rows,
    std::vector<double>& cols,
    std::vector<double>& angles,
    std::vector<double>& scales,
    std::vector<double>& scores)
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

    ValidateLevels(numLevels, "FindScaledShapeModel");

    // Get model implementation
    auto* impl = const_cast<Internal::ShapeModelImpl*>(model.Impl());

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

    const double savedMinContrast = impl->params_.minContrast;
    if (savedMinContrast <= 0.0) {
        impl->params_.minContrast = EstimateAutoMinContrastFromPyramid(targetPyramid);
        if (impl->timingParams_.debugCreateModel) {
            std::printf("[FindScaled] auto minContrast=%.2f\n", impl->params_.minContrast);
        }
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

    // Decompiled: scaled path does GreedyNMS (sub_18004C8C0) then sort+truncate (sub_1800B9C20)
    // FinalizeResults does NOT do overlap-NMS for scaled path (applyNMS=false)
    auto allResults = impl->SearchPyramid(targetPyramid, params, /*applyNMS=*/false);

    // Decompiled sub_18004C8C0: GreedyNMS with rotated rectangle overlap
    // Sort by score → distance prefilter → OBB overlap → suppress if > maxOverlap
    std::sort(allResults.begin(), allResults.end());  // sort before greedy NMS
    double modelW = impl->templateSize_.width;
    double modelH = impl->templateSize_.height;
    allResults = NonMaxSuppressionOverlap(allResults, maxOverlap, modelW, modelH);
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

    impl->params_.minContrast = savedMinContrast;
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
