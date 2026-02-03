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
#include <QiVision/Platform/FileIO.h>
#include <algorithm>
#include <cmath>
#include <fstream>

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

// NCC search accepts any valid image (empty = no results)
inline bool RequireValidImage(const QImage& image, const char* funcName) {
    return Validate::RequireImageValid(image, funcName);
}

// NCC model creation requires non-empty UInt8 grayscale image
inline void RequireTemplateImage(const QImage& image, const char* funcName) {
    Validate::RequireImageNonEmpty(image, funcName);
    Validate::RequireImageType(image, PixelType::UInt8, funcName);
    Validate::RequireChannelCountExact(image, 1, funcName);
}

} // anonymous namespace

// =============================================================================
// NCCModelImpl::Clone() Implementation
// =============================================================================

std::unique_ptr<Internal::NCCModelImpl> Internal::NCCModelImpl::Clone() const {
    auto clone = std::make_unique<NCCModelImpl>();

    // Copy all member data
    clone->levels_ = levels_;
    clone->rotatedTemplatesCoarse_ = rotatedTemplatesCoarse_;
    clone->rotatedTemplatesFine_ = rotatedTemplatesFine_;
    clone->params_ = params_;
    clone->origin_ = origin_;
    clone->templateSize_ = templateSize_;
    clone->valid_ = valid_;
    clone->hasMask_ = hasMask_;
    clone->metric_ = metric_;
    clone->searchAnglesCoarse_ = searchAnglesCoarse_;
    clone->searchAnglesFine_ = searchAnglesFine_;
    clone->fineAngleLevels_ = fineAngleLevels_;

    return clone;
}

// =============================================================================
// NCCModel Class Implementation
// =============================================================================

NCCModel::NCCModel()
    : impl_(std::make_unique<Internal::NCCModelImpl>())
{
}

NCCModel::~NCCModel() = default;

NCCModel::NCCModel(const NCCModel& other)
    : impl_(nullptr)
{
    if (other.impl_) {
        impl_ = other.impl_->Clone();
    }
}

NCCModel::NCCModel(NCCModel&& other) noexcept = default;

NCCModel& NCCModel::operator=(const NCCModel& other)
{
    if (this != &other) {
        if (other.impl_) {
            impl_ = other.impl_->Clone();
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
    RequireTemplateImage(templateImage, "CreateNCCModel");
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
    RequireTemplateImage(templateImage, "CreateNCCModel");
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
    RequireTemplateImage(templateImage, "CreateNCCModel");
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
    RequireTemplateImage(templateImage, "CreateScaledNCCModel");
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

namespace {

// File format constants
constexpr uint32_t NCC_MODEL_MAGIC = 0x434E4951;  // "QINC" - QiVision NCC
constexpr uint32_t NCC_MODEL_VERSION = 1;

// Security limits for deserialization (prevent malicious files)
constexpr int32_t MAX_PYRAMID_LEVELS = 10;
constexpr int32_t MAX_TEMPLATE_SIZE = 10000;          // Max width/height
constexpr uint64_t MAX_ARRAY_SIZE = 100 * 1024 * 1024; // 100MB per array
constexpr uint64_t MAX_ROTATED_TEMPLATES = 10000;      // Max templates per level

// Serialization helpers for NCCLevelModel
void WriteNCCLevelModel(Platform::BinaryWriter& writer, const Internal::NCCLevelModel& level) {
    writer.Write<int32_t>(level.width);
    writer.Write<int32_t>(level.height);
    writer.Write<double>(level.scale);
    writer.Write<double>(level.mean);
    writer.Write<double>(level.stddev);
    writer.Write<double>(level.sumSq);
    writer.Write<int32_t>(level.numPixels);
    writer.WriteVector(level.data);
    writer.WriteVector(level.mask);
    writer.WriteVector(level.zeroMean);
}

void ReadNCCLevelModel(Platform::BinaryReader& reader, Internal::NCCLevelModel& level) {
    level.width = reader.Read<int32_t>();
    level.height = reader.Read<int32_t>();
    if (level.width < 0 || level.width > MAX_TEMPLATE_SIZE ||
        level.height < 0 || level.height > MAX_TEMPLATE_SIZE) {
        throw IOException("ReadNCCModel: invalid level dimensions");
    }
    level.scale = reader.Read<double>();
    level.mean = reader.Read<double>();
    level.stddev = reader.Read<double>();
    level.sumSq = reader.Read<double>();
    level.numPixels = reader.Read<int32_t>();
    if (level.numPixels < 0 || static_cast<uint64_t>(level.numPixels) > MAX_ARRAY_SIZE) {
        throw IOException("ReadNCCModel: invalid numPixels");
    }

    // Read with size validation
    uint64_t dataSize = reader.Read<uint64_t>();
    if (dataSize > MAX_ARRAY_SIZE / sizeof(float)) {
        throw IOException("ReadNCCModel: data array too large");
    }
    level.data.resize(static_cast<size_t>(dataSize));
    reader.ReadArray(level.data.data(), level.data.size());

    uint64_t maskSize = reader.Read<uint64_t>();
    if (maskSize > MAX_ARRAY_SIZE) {
        throw IOException("ReadNCCModel: mask array too large");
    }
    level.mask.resize(static_cast<size_t>(maskSize));
    reader.ReadArray(level.mask.data(), level.mask.size());

    uint64_t zeroMeanSize = reader.Read<uint64_t>();
    if (zeroMeanSize > MAX_ARRAY_SIZE / sizeof(float)) {
        throw IOException("ReadNCCModel: zeroMean array too large");
    }
    level.zeroMean.resize(static_cast<size_t>(zeroMeanSize));
    reader.ReadArray(level.zeroMean.data(), level.zeroMean.size());
}

// Serialization helpers for RotatedTemplate
void WriteRotatedTemplate(Platform::BinaryWriter& writer, const Internal::RotatedTemplate& tmpl) {
    writer.Write<double>(tmpl.angle);
    writer.Write<double>(tmpl.mean);
    writer.Write<double>(tmpl.stddev);
    writer.Write<int32_t>(tmpl.width);
    writer.Write<int32_t>(tmpl.height);
    writer.Write<int32_t>(tmpl.offsetX);
    writer.Write<int32_t>(tmpl.offsetY);
    writer.Write<int32_t>(tmpl.numPixels);
    writer.WriteVector(tmpl.data);
    writer.WriteVector(tmpl.mask);
}

void ReadRotatedTemplate(Platform::BinaryReader& reader, Internal::RotatedTemplate& tmpl) {
    tmpl.angle = reader.Read<double>();
    tmpl.mean = reader.Read<double>();
    tmpl.stddev = reader.Read<double>();
    tmpl.width = reader.Read<int32_t>();
    tmpl.height = reader.Read<int32_t>();
    if (tmpl.width < 0 || tmpl.width > MAX_TEMPLATE_SIZE ||
        tmpl.height < 0 || tmpl.height > MAX_TEMPLATE_SIZE) {
        throw IOException("ReadNCCModel: invalid rotated template dimensions");
    }
    tmpl.offsetX = reader.Read<int32_t>();
    tmpl.offsetY = reader.Read<int32_t>();
    tmpl.numPixels = reader.Read<int32_t>();
    if (tmpl.numPixels < 0 || static_cast<uint64_t>(tmpl.numPixels) > MAX_ARRAY_SIZE) {
        throw IOException("ReadNCCModel: invalid rotated template numPixels");
    }

    // Read with size validation
    uint64_t dataSize = reader.Read<uint64_t>();
    if (dataSize > MAX_ARRAY_SIZE / sizeof(float)) {
        throw IOException("ReadNCCModel: rotated template data too large");
    }
    tmpl.data.resize(static_cast<size_t>(dataSize));
    reader.ReadArray(tmpl.data.data(), tmpl.data.size());

    uint64_t maskSize = reader.Read<uint64_t>();
    if (maskSize > MAX_ARRAY_SIZE) {
        throw IOException("ReadNCCModel: rotated template mask too large");
    }
    tmpl.mask.resize(static_cast<size_t>(maskSize));
    reader.ReadArray(tmpl.mask.data(), tmpl.mask.size());
}

// Write vector of RotatedTemplates for a level
void WriteRotatedTemplatesLevel(Platform::BinaryWriter& writer,
                                 const std::vector<Internal::RotatedTemplate>& templates) {
    writer.Write<uint64_t>(templates.size());
    for (const auto& tmpl : templates) {
        WriteRotatedTemplate(writer, tmpl);
    }
}

// Read vector of RotatedTemplates for a level
void ReadRotatedTemplatesLevel(Platform::BinaryReader& reader,
                                std::vector<Internal::RotatedTemplate>& templates) {
    uint64_t count = reader.Read<uint64_t>();
    if (count > MAX_ROTATED_TEMPLATES) {
        throw IOException("ReadNCCModel: too many rotated templates in level");
    }
    templates.resize(static_cast<size_t>(count));
    for (auto& tmpl : templates) {
        ReadRotatedTemplate(reader, tmpl);
    }
}

} // anonymous namespace

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

    Platform::BinaryWriter writer(filename);
    if (!writer.IsOpen()) {
        throw IOException("WriteNCCModel: failed to open file: " + filename);
    }

    const auto* impl = model.Impl();

    // Header
    writer.Write<uint32_t>(NCC_MODEL_MAGIC);
    writer.Write<uint32_t>(NCC_MODEL_VERSION);

    // Model parameters
    writer.Write<int32_t>(impl->params_.numLevels);
    writer.Write<double>(impl->params_.angleStart);
    writer.Write<double>(impl->params_.angleExtent);
    writer.Write<double>(impl->params_.angleStep);
    writer.Write<double>(impl->params_.scaleMin);
    writer.Write<double>(impl->params_.scaleMax);

    // Model state
    writer.Write<double>(impl->origin_.x);
    writer.Write<double>(impl->origin_.y);
    writer.Write<int32_t>(impl->templateSize_.width);
    writer.Write<int32_t>(impl->templateSize_.height);
    writer.Write<uint8_t>(impl->valid_ ? 1 : 0);
    writer.Write<uint8_t>(impl->hasMask_ ? 1 : 0);
    writer.Write<int32_t>(static_cast<int32_t>(impl->metric_));
    writer.Write<int32_t>(impl->fineAngleLevels_);

    // Search angles
    writer.WriteVector(impl->searchAnglesCoarse_);
    writer.WriteVector(impl->searchAnglesFine_);

    // Pyramid levels
    writer.Write<uint64_t>(impl->levels_.size());
    for (const auto& level : impl->levels_) {
        WriteNCCLevelModel(writer, level);
    }

    // Rotated templates - coarse
    writer.Write<uint64_t>(impl->rotatedTemplatesCoarse_.size());
    for (const auto& levelTemplates : impl->rotatedTemplatesCoarse_) {
        WriteRotatedTemplatesLevel(writer, levelTemplates);
    }

    // Rotated templates - fine
    writer.Write<uint64_t>(impl->rotatedTemplatesFine_.size());
    for (const auto& levelTemplates : impl->rotatedTemplatesFine_) {
        WriteRotatedTemplatesLevel(writer, levelTemplates);
    }

    writer.Close();
}

void ReadNCCModel(
    const std::string& filename,
    NCCModel& model)
{
    if (filename.empty()) {
        throw InvalidArgumentException("ReadNCCModel: filename is empty");
    }

    Platform::BinaryReader reader(filename);
    if (!reader.IsOpen()) {
        throw IOException("ReadNCCModel: failed to open file: " + filename);
    }

    // Header verification
    uint32_t magic = reader.Read<uint32_t>();
    if (magic != NCC_MODEL_MAGIC) {
        throw IOException("ReadNCCModel: invalid file format (magic mismatch)");
    }

    uint32_t version = reader.Read<uint32_t>();
    if (version != NCC_MODEL_VERSION) {
        throw VersionMismatchException("ReadNCCModel: unsupported version: " + std::to_string(version));
    }

    auto* impl = model.Impl();

    // Clear existing data
    impl->levels_.clear();
    impl->rotatedTemplatesCoarse_.clear();
    impl->rotatedTemplatesFine_.clear();
    impl->searchAnglesCoarse_.clear();
    impl->searchAnglesFine_.clear();

    // Model parameters
    impl->params_.numLevels = reader.Read<int32_t>();
    if (impl->params_.numLevels < 1 || impl->params_.numLevels > MAX_PYRAMID_LEVELS) {
        throw IOException("ReadNCCModel: invalid numLevels: " + std::to_string(impl->params_.numLevels));
    }
    impl->params_.angleStart = reader.Read<double>();
    impl->params_.angleExtent = reader.Read<double>();
    impl->params_.angleStep = reader.Read<double>();
    impl->params_.scaleMin = reader.Read<double>();
    impl->params_.scaleMax = reader.Read<double>();

    // Model state
    impl->origin_.x = reader.Read<double>();
    impl->origin_.y = reader.Read<double>();
    impl->templateSize_.width = reader.Read<int32_t>();
    impl->templateSize_.height = reader.Read<int32_t>();
    if (impl->templateSize_.width < 1 || impl->templateSize_.width > MAX_TEMPLATE_SIZE ||
        impl->templateSize_.height < 1 || impl->templateSize_.height > MAX_TEMPLATE_SIZE) {
        throw IOException("ReadNCCModel: invalid template size");
    }
    impl->valid_ = reader.Read<uint8_t>() != 0;
    impl->hasMask_ = reader.Read<uint8_t>() != 0;
    impl->metric_ = static_cast<MetricMode>(reader.Read<int32_t>());
    impl->fineAngleLevels_ = reader.Read<int32_t>();

    // Search angles (with size limits)
    uint64_t numCoarseAngles = reader.Read<uint64_t>();
    if (numCoarseAngles > MAX_ROTATED_TEMPLATES) {
        throw IOException("ReadNCCModel: too many coarse angles");
    }
    impl->searchAnglesCoarse_.resize(static_cast<size_t>(numCoarseAngles));
    reader.ReadArray(impl->searchAnglesCoarse_.data(), impl->searchAnglesCoarse_.size());

    uint64_t numFineAngles = reader.Read<uint64_t>();
    if (numFineAngles > MAX_ROTATED_TEMPLATES) {
        throw IOException("ReadNCCModel: too many fine angles");
    }
    impl->searchAnglesFine_.resize(static_cast<size_t>(numFineAngles));
    reader.ReadArray(impl->searchAnglesFine_.data(), impl->searchAnglesFine_.size());

    // Pyramid levels
    uint64_t numLevels = reader.Read<uint64_t>();
    if (numLevels > MAX_PYRAMID_LEVELS) {
        throw IOException("ReadNCCModel: too many pyramid levels");
    }
    impl->levels_.resize(static_cast<size_t>(numLevels));
    for (auto& level : impl->levels_) {
        ReadNCCLevelModel(reader, level);
    }

    // Rotated templates - coarse
    uint64_t numCoarseLevels = reader.Read<uint64_t>();
    if (numCoarseLevels > MAX_PYRAMID_LEVELS) {
        throw IOException("ReadNCCModel: too many coarse template levels");
    }
    impl->rotatedTemplatesCoarse_.resize(static_cast<size_t>(numCoarseLevels));
    for (auto& levelTemplates : impl->rotatedTemplatesCoarse_) {
        ReadRotatedTemplatesLevel(reader, levelTemplates);
    }

    // Rotated templates - fine
    uint64_t numFineLevels = reader.Read<uint64_t>();
    if (numFineLevels > MAX_PYRAMID_LEVELS) {
        throw IOException("ReadNCCModel: too many fine template levels");
    }
    impl->rotatedTemplatesFine_.resize(static_cast<size_t>(numFineLevels));
    for (auto& levelTemplates : impl->rotatedTemplatesFine_) {
        ReadRotatedTemplatesLevel(reader, levelTemplates);
    }

    reader.Close();
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
    RequireTemplateImage(templateImage, "DetermineNCCModelParams");

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
