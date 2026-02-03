/**
 * @file NCCModel.cpp
 * @brief NCCModel class implementation and public API functions
 *
 * This file contains:
 * - NCCModel class (handle) implementation
 * - Public API functions (CreateNCCModel, FindNCCModel, etc.)
 * - Model I/O functions
 *
 * Actual algorithm implementation is in:
 * - NCCModelCreate.cpp: Model creation
 * - NCCModelSearch.cpp: Search functions
 * - NCCModelScore.cpp: NCC score computation
 */

#include <QiVision/Matching/NCCModel.h>
#include "NCCModelImpl.h"

#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <cmath>

namespace Qi::Vision::Matching {

// =============================================================================
// NCCModel Class Implementation
// =============================================================================

namespace {

MetricMode ParseMetric(const std::string& metric) {
    std::string lower;
    lower.reserve(metric.size());
    for (char c : metric) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower.empty() || lower == "use_polarity") return MetricMode::UsePolarity;
    if (lower == "ignore_global_polarity") return MetricMode::IgnoreGlobalPolarity;
    throw InvalidArgumentException("Unknown NCC metric: " + metric);
}

SubpixelMethod ParseSubPixel(const std::string& subPixel) {
    std::string lower;
    lower.reserve(subPixel.size());
    for (char c : subPixel) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower.empty() || lower == "interpolation" || lower == "true") {
        return SubpixelMethod::Parabolic;
    }
    if (lower == "none" || lower == "false") return SubpixelMethod::None;
    throw InvalidArgumentException("Unknown subpixel mode: " + subPixel);
}

inline void ValidateLevels(int32_t numLevels, const char* funcName) {
    Validate::RequireNonNegative(numLevels, "numLevels", funcName);
}

inline void ValidateAngleStep(double angleStep, const char* funcName) {
    Validate::RequireNonNegative(angleStep, "angleStep", funcName);
}

inline void ValidateScale(double scaleMin, double scaleMax, double scaleStep, const char* funcName) {
    Validate::RequirePositive(scaleMin, "scaleMin", funcName);
    Validate::RequirePositive(scaleMax, "scaleMax", funcName);
    Validate::RequirePositive(scaleStep, "scaleStep", funcName);
    if (scaleMax < scaleMin) {
        throw InvalidArgumentException(std::string(funcName) + ": scaleMax must be >= scaleMin");
    }
}

// NCC matching accepts any valid image
inline bool RequireValidImage(const QImage& image, const char* funcName) {
    return Validate::RequireImageValid(image, funcName);
}

} // anonymous namespace

NCCModel::NCCModel()
    : impl_(std::make_unique<Internal::NCCModelImpl>())
{
}

NCCModel::~NCCModel() = default;

NCCModel::NCCModel(const NCCModel& other)
    : impl_(nullptr)
{
    if (other.impl_) {
        // Deep copy by creating new impl
        // Note: Full copy not implemented yet
        impl_ = std::make_unique<Internal::NCCModelImpl>();
        // TODO: Copy implementation data
    }
}

NCCModel::NCCModel(NCCModel&& other) noexcept = default;

NCCModel& NCCModel::operator=(const NCCModel& other)
{
    if (this != &other) {
        if (other.impl_) {
            impl_ = std::make_unique<Internal::NCCModelImpl>();
            // TODO: Copy implementation data
        } else {
            impl_.reset();
        }
    }
    return *this;
}

NCCModel& NCCModel::operator=(NCCModel&& other) noexcept = default;

bool NCCModel::IsValid() const
{
    return impl_ && impl_->valid_;
}

// =============================================================================
// Model Creation Functions
// =============================================================================

void CreateNCCModel(
    const QImage& templateImage,
    NCCModel& model,
    int32_t numLevels,
    double angleStart,
    double angleExtent,
    double angleStep,
    const std::string& metric)
{
    if (!templateImage.IsValid()) {
        throw InvalidArgumentException("CreateNCCModel: invalid template image");
    }
    ValidateLevels(numLevels, "CreateNCCModel");
    ValidateAngleStep(angleStep, "CreateNCCModel");

    // Set parameters
    auto* impl = model.Impl();
    impl->params_.numLevels = numLevels;
    impl->params_.angleStart = angleStart;
    impl->params_.angleExtent = angleExtent;
    impl->params_.angleStep = angleStep;

    // Parse metric
    impl->metric_ = ParseMetric(metric);

    // Create model
    if (!impl->CreateModel(templateImage)) {
        throw InsufficientDataException("CreateNCCModel: failed to create model");
    }
}

void CreateNCCModel(
    const QImage& templateImage,
    const Rect2i& roi,
    NCCModel& model,
    int32_t numLevels,
    double angleStart,
    double angleExtent,
    double angleStep,
    const std::string& metric)
{
    if (!templateImage.IsValid()) {
        throw InvalidArgumentException("CreateNCCModel: invalid template image");
    }
    ValidateLevels(numLevels, "CreateNCCModel");
    ValidateAngleStep(angleStep, "CreateNCCModel");
    if (roi.width <= 0 || roi.height <= 0) {
        throw InvalidArgumentException("CreateNCCModel: ROI width/height must be > 0");
    }
    if (roi.x < 0 || roi.y < 0 ||
        roi.x + roi.width > templateImage.Width() ||
        roi.y + roi.height > templateImage.Height()) {
        throw InvalidArgumentException("CreateNCCModel: ROI out of bounds");
    }

    // Set parameters
    auto* impl = model.Impl();
    impl->params_.numLevels = numLevels;
    impl->params_.angleStart = angleStart;
    impl->params_.angleExtent = angleExtent;
    impl->params_.angleStep = angleStep;

    // Parse metric
    impl->metric_ = ParseMetric(metric);

    // Create model with ROI
    if (!impl->CreateModel(templateImage, roi)) {
        throw InsufficientDataException("CreateNCCModel: failed to create model from ROI");
    }
}

void CreateNCCModel(
    const QImage& templateImage,
    const QRegion& region,
    NCCModel& model,
    int32_t numLevels,
    double angleStart,
    double angleExtent,
    double angleStep,
    const std::string& metric)
{
    if (!templateImage.IsValid()) {
        throw InvalidArgumentException("CreateNCCModel: invalid template image");
    }

    if (region.Empty()) {
        throw InvalidArgumentException("CreateNCCModel: empty region");
    }
    ValidateLevels(numLevels, "CreateNCCModel");
    ValidateAngleStep(angleStep, "CreateNCCModel");

    // Set parameters
    auto* impl = model.Impl();
    impl->params_.numLevels = numLevels;
    impl->params_.angleStart = angleStart;
    impl->params_.angleExtent = angleExtent;
    impl->params_.angleStep = angleStep;

    // Parse metric
    impl->metric_ = ParseMetric(metric);

    // Create model with QRegion
    if (!impl->CreateModel(templateImage, region)) {
        throw InsufficientDataException("CreateNCCModel: failed to create model from region");
    }
}

void CreateScaledNCCModel(
    const QImage& templateImage,
    NCCModel& model,
    int32_t numLevels,
    double angleStart,
    double angleExtent,
    double angleStep,
    double scaleMin,
    double scaleMax,
    double scaleStep,
    const std::string& metric)
{
    if (!templateImage.IsValid()) {
        throw InvalidArgumentException("CreateScaledNCCModel: invalid template image");
    }
    ValidateLevels(numLevels, "CreateScaledNCCModel");
    ValidateAngleStep(angleStep, "CreateScaledNCCModel");
    ValidateScale(scaleMin, scaleMax, scaleStep, "CreateScaledNCCModel");

    // Set parameters
    auto* impl = model.Impl();
    impl->params_.numLevels = numLevels;
    impl->params_.angleStart = angleStart;
    impl->params_.angleExtent = angleExtent;
    impl->params_.angleStep = angleStep;
    impl->params_.scaleMin = scaleMin;
    impl->params_.scaleMax = scaleMax;

    // Parse metric
    impl->metric_ = ParseMetric(metric);

    // Create model (scale search done during Find)
    if (!impl->CreateModel(templateImage)) {
        throw InsufficientDataException("CreateScaledNCCModel: failed to create model");
    }
}

// =============================================================================
// Model Search Functions
// =============================================================================

void FindNCCModel(
    const QImage& image,
    const NCCModel& model,
    double angleStart,
    double angleExtent,
    double minScore,
    int32_t numMatches,
    double maxOverlap,
    const std::string& subPixel,
    int32_t numLevels,
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

    if (!RequireValidImage(image, "FindNCCModel")) {
        return;
    }

    if (!model.IsValid()) {
        throw InvalidArgumentException("FindNCCModel: invalid model");
    }
    if (!std::isfinite(angleStart) || !std::isfinite(angleExtent)) {
        throw InvalidArgumentException("FindNCCModel: invalid angle range");
    }
    if (!std::isfinite(minScore) || minScore < 0.0) {
        throw InvalidArgumentException("FindNCCModel: minScore must be >= 0");
    }
    if (!std::isfinite(maxOverlap) || maxOverlap < 0.0) {
        throw InvalidArgumentException("FindNCCModel: maxOverlap must be >= 0");
    }
    if (numMatches < 0) {
        throw InvalidArgumentException("FindNCCModel: numMatches must be >= 0");
    }

    ValidateLevels(numLevels, "FindNCCModel");

    // Build search parameters
    SearchParams params;
    params.minScore = minScore;
    params.maxMatches = numMatches;
    params.maxOverlap = maxOverlap;
    params.angleStart = angleStart;
    params.angleExtent = angleExtent;
    params.numLevels = numLevels;

    // Parse subpixel mode
    params.subpixelMethod = ParseSubPixel(subPixel);

    // Find matches
    const auto* impl = model.Impl();
    auto matches = impl->Find(image, params);

    // Convert to output format
    rows.reserve(matches.size());
    cols.reserve(matches.size());
    angles.reserve(matches.size());
    scores.reserve(matches.size());

    for (const auto& m : matches) {
        rows.push_back(m.y);
        cols.push_back(m.x);
        angles.push_back(m.angle);
        scores.push_back(m.score);
    }
}

void FindScaledNCCModel(
    const QImage& image,
    const NCCModel& model,
    double angleStart,
    double angleExtent,
    double scaleMin,
    double scaleMax,
    double minScore,
    int32_t numMatches,
    double maxOverlap,
    const std::string& subPixel,
    int32_t numLevels,
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

    if (!RequireValidImage(image, "FindScaledNCCModel")) {
        return;
    }

    if (!model.IsValid()) {
        throw InvalidArgumentException("FindScaledNCCModel: invalid model");
    }
    if (!std::isfinite(angleStart) || !std::isfinite(angleExtent)) {
        throw InvalidArgumentException("FindScaledNCCModel: invalid angle range");
    }
    if (!std::isfinite(scaleMin) || !std::isfinite(scaleMax) || scaleMin <= 0.0 || scaleMax <= 0.0 ||
        scaleMax < scaleMin) {
        throw InvalidArgumentException("FindScaledNCCModel: invalid scale range");
    }
    if (!std::isfinite(minScore) || minScore < 0.0) {
        throw InvalidArgumentException("FindScaledNCCModel: minScore must be >= 0");
    }
    if (!std::isfinite(maxOverlap) || maxOverlap < 0.0) {
        throw InvalidArgumentException("FindScaledNCCModel: maxOverlap must be >= 0");
    }
    if (numMatches < 0) {
        throw InvalidArgumentException("FindScaledNCCModel: numMatches must be >= 0");
    }

    ValidateLevels(numLevels, "FindScaledNCCModel");
    ValidateScale(scaleMin, scaleMax, 1.0, "FindScaledNCCModel");

    // Build search parameters
    SearchParams params;
    params.minScore = minScore;
    params.maxMatches = numMatches;
    params.maxOverlap = maxOverlap;
    params.angleStart = angleStart;
    params.angleExtent = angleExtent;
    params.scaleMode = ScaleSearchMode::Uniform;
    params.scaleMin = scaleMin;
    params.scaleMax = scaleMax;
    params.numLevels = numLevels;

    // Parse subpixel mode
    params.subpixelMethod = ParseSubPixel(subPixel);

    // Find matches
    const auto* impl = model.Impl();
    auto matches = impl->Find(image, params);

    // Convert to output format
    rows.reserve(matches.size());
    cols.reserve(matches.size());
    angles.reserve(matches.size());
    scales.reserve(matches.size());
    scores.reserve(matches.size());

    for (const auto& m : matches) {
        rows.push_back(m.y);
        cols.push_back(m.x);
        angles.push_back(m.angle);
        scales.push_back(m.scaleX);  // Uniform scale
        scores.push_back(m.score);
    }
}

// =============================================================================
// Model Property Functions
// =============================================================================

void GetNCCModelParams(
    const NCCModel& model,
    int32_t& numLevels,
    double& angleStart,
    double& angleExtent,
    double& angleStep,
    std::string& metric)
{
    if (!model.IsValid()) {
        throw InvalidArgumentException("GetNCCModelParams: invalid model");
    }

    const auto* impl = model.Impl();
    numLevels = static_cast<int32_t>(impl->levels_.size());
    angleStart = impl->params_.angleStart;
    angleExtent = impl->params_.angleExtent;
    angleStep = impl->params_.angleStep;

    switch (impl->metric_) {
        case MetricMode::UsePolarity:
            metric = "use_polarity";
            break;
        case MetricMode::IgnoreGlobalPolarity:
            metric = "ignore_global_polarity";
            break;
        default:
            metric = "use_polarity";
            break;
    }
}

void GetNCCModelOrigin(
    const NCCModel& model,
    double& row,
    double& col)
{
    if (!model.IsValid()) {
        throw InvalidArgumentException("GetNCCModelOrigin: invalid model");
    }

    const auto* impl = model.Impl();
    row = impl->origin_.y;
    col = impl->origin_.x;
}

void SetNCCModelOrigin(
    NCCModel& model,
    double row,
    double col)
{
    if (!model.IsValid()) {
        throw InvalidArgumentException("SetNCCModelOrigin: invalid model");
    }
    if (!std::isfinite(row) || !std::isfinite(col)) {
        throw InvalidArgumentException("SetNCCModelOrigin: invalid origin");
    }

    auto* impl = model.Impl();
    impl->origin_.y = row;
    impl->origin_.x = col;
}

void GetNCCModelSize(
    const NCCModel& model,
    int32_t& width,
    int32_t& height)
{
    if (!model.IsValid()) {
        throw InvalidArgumentException("GetNCCModelSize: invalid model");
    }

    const auto* impl = model.Impl();
    width = impl->templateSize_.width;
    height = impl->templateSize_.height;
}

// =============================================================================
// Model I/O Functions
// =============================================================================

void WriteNCCModel(
    const NCCModel& model,
    const std::string& filename)
{
    if (!model.IsValid()) {
        throw InvalidArgumentException("WriteNCCModel: invalid model");
    }
    if (filename.empty()) {
        throw InvalidArgumentException("WriteNCCModel: filename is empty");
    }

    // TODO: Implement model serialization
    throw UnsupportedException("WriteNCCModel: Not implemented yet");
}

void ReadNCCModel(
    const std::string& filename,
    NCCModel& model)
{
    (void)model;
    if (filename.empty()) {
        throw InvalidArgumentException("ReadNCCModel: filename is empty");
    }
    // TODO: Implement model deserialization
    throw UnsupportedException("ReadNCCModel: Not implemented yet");
}

void ClearNCCModel(
    NCCModel& model)
{
    auto* impl = model.Impl();
    if (impl) {
        impl->levels_.clear();
        impl->rotatedTemplatesCoarse_.clear();
        impl->rotatedTemplatesFine_.clear();
        impl->searchAnglesCoarse_.clear();
        impl->searchAnglesFine_.clear();
        impl->valid_ = false;
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

void DetermineNCCModelParams(
    const QImage& templateImage,
    const Rect2i& roi,
    int32_t& numLevels,
    double& angleStep)
{
    if (!templateImage.IsValid()) {
        throw InvalidArgumentException("DetermineNCCModelParams: invalid template image");
    }

    // Get template dimensions
    int32_t width = templateImage.Width();
    int32_t height = templateImage.Height();

    if (roi.width > 0 && roi.height > 0) {
        width = roi.width;
        height = roi.height;
    }

    // Compute optimal pyramid levels
    int32_t minDim = std::min(width, height);
    numLevels = 1;
    while (minDim >= 16 && numLevels < 6) {
        minDim /= 2;
        numLevels++;
    }

    // Compute optimal angle step (~1 pixel arc at edge)
    double radius = std::max(width, height) * 0.5;
    if (radius < 10) radius = 10;
    angleStep = std::atan(1.0 / radius);

    // Clamp to reasonable range
    constexpr double MIN_ANGLE_STEP = 0.005;  // ~0.3 degrees
    constexpr double MAX_ANGLE_STEP = 0.1;    // ~5.7 degrees
    angleStep = std::clamp(angleStep, MIN_ANGLE_STEP, MAX_ANGLE_STEP);
}

} // namespace Qi::Vision::Matching
