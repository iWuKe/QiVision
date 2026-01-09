/**
 * @file AnglePyramid.cpp
 * @brief Implementation of angle pyramid for shape-based matching
 */

#include <QiVision/Internal/AnglePyramid.h>
#include <QiVision/Internal/Gradient.h>
#include <QiVision/Internal/Gaussian.h>
#include <QiVision/Internal/Convolution.h>
#include <QiVision/Internal/Pyramid.h>
#include <QiVision/Internal/Interpolate.h>
#include <QiVision/Internal/Geometry2d.h>
#include <QiVision/Core/Constants.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Qi::Vision::Internal {

// =============================================================================
// Implementation Class
// =============================================================================

class AnglePyramid::Impl {
public:
    AnglePyramidParams params_;
    std::vector<PyramidLevelData> levels_;
    int32_t originalWidth_ = 0;
    int32_t originalHeight_ = 0;
    bool valid_ = false;

    bool BuildLevel(const std::vector<float>& srcData, int32_t width, int32_t height,
                    int32_t level, double scale);
    void ExtractEdgePointsForLevel(int32_t level);
};

bool AnglePyramid::Impl::BuildLevel(const std::vector<float>& srcData, int32_t width, int32_t height,
                                    int32_t level, double scale) {
    PyramidLevelData levelData;
    levelData.level = level;
    levelData.width = width;
    levelData.height = height;
    levelData.scale = scale;

    // Allocate contiguous buffers for gradients
    std::vector<float> gxBuffer(static_cast<size_t>(width) * height);
    std::vector<float> gyBuffer(static_cast<size_t>(width) * height);

    // Use Sobel operator for gradient computation (contiguous memory)
    Gradient<float, float>(srcData.data(), gxBuffer.data(), gyBuffer.data(),
                           width, height,
                           GradientOperator::Sobel3x3, BorderMode::Reflect101);

    // Copy gradients to QImage for storage
    QImage gx(width, height, PixelType::Float32, ChannelType::Gray);
    QImage gy(width, height, PixelType::Float32, ChannelType::Gray);

    float* gxDst = static_cast<float*>(gx.Data());
    float* gyDst = static_cast<float*>(gy.Data());
    int32_t gxStride = gx.Stride() / sizeof(float);
    int32_t gyStride = gy.Stride() / sizeof(float);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            gxDst[y * gxStride + x] = gxBuffer[y * width + x];
            gyDst[y * gyStride + x] = gyBuffer[y * width + x];
        }
    }

    if (gx.Empty() || gy.Empty()) {
        return false;
    }

    levelData.gradX = std::move(gx);
    levelData.gradY = std::move(gy);

    // Compute magnitude and direction
    levelData.gradMag = ComputeGradientMagnitude(levelData.gradX, levelData.gradY);
    levelData.gradDir = ComputeGradientDirection(levelData.gradX, levelData.gradY);

    // Quantize direction to angle bins
    levelData.angleBinImage = QuantizeGradientDirection(levelData.gradDir, params_.angleBins);

    levels_.push_back(std::move(levelData));
    return true;
}

void AnglePyramid::Impl::ExtractEdgePointsForLevel(int32_t level) {
    if (level < 0 || level >= static_cast<int32_t>(levels_.size())) {
        return;
    }

    auto& levelData = levels_[level];
    levelData.edgePoints.clear();

    const float* magData = static_cast<const float*>(levelData.gradMag.Data());
    const float* dirData = static_cast<const float*>(levelData.gradDir.Data());
    const int16_t* binData = static_cast<const int16_t*>(levelData.angleBinImage.Data());

    int32_t magStride = levelData.gradMag.Stride() / sizeof(float);
    int32_t dirStride = levelData.gradDir.Stride() / sizeof(float);
    int32_t binStride = levelData.angleBinImage.Stride() / sizeof(int16_t);

    double minContrast = params_.minContrast;

    for (int32_t y = 1; y < levelData.height - 1; ++y) {
        for (int32_t x = 1; x < levelData.width - 1; ++x) {
            float mag = magData[y * magStride + x];

            if (mag >= minContrast) {
                float dir = dirData[y * dirStride + x];
                int16_t bin = binData[y * binStride + x];

                levelData.edgePoints.emplace_back(
                    static_cast<double>(x),
                    static_cast<double>(y),
                    static_cast<double>(dir),
                    static_cast<double>(mag),
                    static_cast<int32_t>(bin)
                );
            }
        }
    }
}

// =============================================================================
// AnglePyramid Implementation
// =============================================================================

AnglePyramid::AnglePyramid() : impl_(std::make_unique<Impl>()) {}

AnglePyramid::~AnglePyramid() = default;

AnglePyramid::AnglePyramid(const AnglePyramid& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

AnglePyramid::AnglePyramid(AnglePyramid&& other) noexcept = default;

AnglePyramid& AnglePyramid::operator=(const AnglePyramid& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

AnglePyramid& AnglePyramid::operator=(AnglePyramid&& other) noexcept = default;

bool AnglePyramid::Build(const QImage& image, const AnglePyramidParams& params) {
    Clear();

    if (image.Empty()) {
        return false;
    }

    // Validate parameters
    impl_->params_ = params;
    impl_->params_.numLevels = std::clamp(params.numLevels, 1, ANGLE_PYRAMID_MAX_LEVELS);
    impl_->params_.angleBins = std::clamp(params.angleBins, MIN_ANGLE_BINS, MAX_ANGLE_BINS);

    impl_->originalWidth_ = image.Width();
    impl_->originalHeight_ = image.Height();

    // Convert to grayscale if needed
    QImage grayImage = image;
    if (image.Channels() > 1) {
        // TODO: Convert to grayscale
        return false;
    }

    // Convert to float for processing
    QImage floatImage;
    if (image.Type() == PixelType::Float32) {
        floatImage = grayImage;
    } else {
        floatImage = QImage(grayImage.Width(), grayImage.Height(),
                            PixelType::Float32, ChannelType::Gray);
        const uint8_t* srcData = static_cast<const uint8_t*>(grayImage.Data());
        float* dstData = static_cast<float*>(floatImage.Data());
        int32_t srcStride = grayImage.Stride();
        int32_t dstStride = floatImage.Stride() / sizeof(float);

        for (int32_t y = 0; y < grayImage.Height(); ++y) {
            for (int32_t x = 0; x < grayImage.Width(); ++x) {
                dstData[y * dstStride + x] = static_cast<float>(srcData[y * srcStride + x]);
            }
        }
    }

    // Build Gaussian pyramid using the existing Pyramid module
    // The pyramid's sigma parameter handles smoothing before each level
    PyramidParams pyramidParams;
    pyramidParams.numLevels = impl_->params_.numLevels;
    pyramidParams.sigma = std::max(1.0, params.smoothSigma);  // Use smoothSigma for pyramid smoothing
    pyramidParams.minDimension = 16;

    // Extract float data from the float QImage (handle stride correctly)
    int32_t floatWidth = floatImage.Width();
    int32_t floatHeight = floatImage.Height();
    int32_t floatStride = floatImage.Stride() / sizeof(float);
    const float* floatSrcData = static_cast<const float*>(floatImage.Data());

    std::vector<float> floatContiguous(floatWidth * floatHeight);
    for (int32_t y = 0; y < floatHeight; ++y) {
        for (int32_t x = 0; x < floatWidth; ++x) {
            floatContiguous[y * floatWidth + x] = floatSrcData[y * floatStride + x];
        }
    }

    // Use the float overload of BuildGaussianPyramid
    ImagePyramid gaussPyramid = BuildGaussianPyramid(floatContiguous.data(),
                                                      floatWidth, floatHeight, pyramidParams);

    // Update actual number of levels
    impl_->params_.numLevels = gaussPyramid.NumLevels();

    // Build angle pyramid for each level using PyramidLevel data directly
    impl_->levels_.reserve(gaussPyramid.NumLevels());
    double scale = 1.0;

    for (int32_t level = 0; level < gaussPyramid.NumLevels(); ++level) {
        const auto& pyramidLevel = gaussPyramid.GetLevel(level);
        if (!impl_->BuildLevel(pyramidLevel.data, pyramidLevel.width, pyramidLevel.height,
                               level, scale)) {
            Clear();
            return false;
        }
        scale *= 0.5;
    }

    // Extract edge points for each level
    for (int32_t level = 0; level < static_cast<int32_t>(impl_->levels_.size()); ++level) {
        impl_->ExtractEdgePointsForLevel(level);
    }

    impl_->valid_ = true;
    return true;
}

void AnglePyramid::Clear() {
    impl_->levels_.clear();
    impl_->originalWidth_ = 0;
    impl_->originalHeight_ = 0;
    impl_->valid_ = false;
}

bool AnglePyramid::IsValid() const {
    return impl_->valid_;
}

int32_t AnglePyramid::NumLevels() const {
    return static_cast<int32_t>(impl_->levels_.size());
}

int32_t AnglePyramid::AngleBins() const {
    return impl_->params_.angleBins;
}

int32_t AnglePyramid::OriginalWidth() const {
    return impl_->originalWidth_;
}

int32_t AnglePyramid::OriginalHeight() const {
    return impl_->originalHeight_;
}

const AnglePyramidParams& AnglePyramid::GetParams() const {
    return impl_->params_;
}

const PyramidLevelData& AnglePyramid::GetLevel(int32_t level) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        throw std::out_of_range("Pyramid level out of range");
    }
    return impl_->levels_[level];
}

int32_t AnglePyramid::GetWidth(int32_t level) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return 0;
    }
    return impl_->levels_[level].width;
}

int32_t AnglePyramid::GetHeight(int32_t level) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return 0;
    }
    return impl_->levels_[level].height;
}

double AnglePyramid::GetScale(int32_t level) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return 0.0;
    }
    return impl_->levels_[level].scale;
}

double AnglePyramid::GetAngleAt(int32_t level, double x, double y) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return -1.0;
    }

    const auto& levelData = impl_->levels_[level];

    if (x < 0 || x >= levelData.width - 1 || y < 0 || y >= levelData.height - 1) {
        return -1.0;
    }

    // Bilinear interpolation of angle (careful with wrap-around)
    int32_t x0 = static_cast<int32_t>(x);
    int32_t y0 = static_cast<int32_t>(y);
    double fx = x - x0;
    double fy = y - y0;

    const float* dirData = static_cast<const float*>(levelData.gradDir.Data());
    int32_t stride = levelData.gradDir.Stride() / sizeof(float);

    // Get four corner angles
    double a00 = dirData[y0 * stride + x0];
    double a10 = dirData[y0 * stride + x0 + 1];
    double a01 = dirData[(y0 + 1) * stride + x0];
    double a11 = dirData[(y0 + 1) * stride + x0 + 1];

    // Handle angle wrap-around: convert to unit vectors and interpolate
    double cx = (1 - fx) * (1 - fy) * std::cos(a00) +
                fx * (1 - fy) * std::cos(a10) +
                (1 - fx) * fy * std::cos(a01) +
                fx * fy * std::cos(a11);

    double cy = (1 - fx) * (1 - fy) * std::sin(a00) +
                fx * (1 - fy) * std::sin(a10) +
                (1 - fx) * fy * std::sin(a01) +
                fx * fy * std::sin(a11);

    return NormalizeAngle0To2PI(std::atan2(cy, cx));
}

double AnglePyramid::GetMagnitudeAt(int32_t level, double x, double y) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return 0.0;
    }

    const auto& levelData = impl_->levels_[level];

    if (x < 0 || x >= levelData.width - 1 || y < 0 || y >= levelData.height - 1) {
        return 0.0;
    }

    // Bilinear interpolation
    int32_t x0 = static_cast<int32_t>(x);
    int32_t y0 = static_cast<int32_t>(y);
    double fx = x - x0;
    double fy = y - y0;

    const float* magData = static_cast<const float*>(levelData.gradMag.Data());
    int32_t stride = levelData.gradMag.Stride() / sizeof(float);

    double m00 = magData[y0 * stride + x0];
    double m10 = magData[y0 * stride + x0 + 1];
    double m01 = magData[(y0 + 1) * stride + x0];
    double m11 = magData[(y0 + 1) * stride + x0 + 1];

    return (1 - fx) * (1 - fy) * m00 +
           fx * (1 - fy) * m10 +
           (1 - fx) * fy * m01 +
           fx * fy * m11;
}

int32_t AnglePyramid::GetAngleBinAt(int32_t level, int32_t x, int32_t y) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return -1;
    }

    const auto& levelData = impl_->levels_[level];

    if (x < 0 || x >= levelData.width || y < 0 || y >= levelData.height) {
        return -1;
    }

    const int16_t* binData = static_cast<const int16_t*>(levelData.angleBinImage.Data());
    int32_t stride = levelData.angleBinImage.Stride() / sizeof(int16_t);

    return binData[y * stride + x];
}

bool AnglePyramid::GetGradientAt(int32_t level, double x, double y,
                                  double& gx, double& gy) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return false;
    }

    const auto& levelData = impl_->levels_[level];

    if (x < 0 || x >= levelData.width - 1 || y < 0 || y >= levelData.height - 1) {
        return false;
    }

    // Bilinear interpolation
    int32_t x0 = static_cast<int32_t>(x);
    int32_t y0 = static_cast<int32_t>(y);
    double fx = x - x0;
    double fy = y - y0;

    const float* gxData = static_cast<const float*>(levelData.gradX.Data());
    const float* gyData = static_cast<const float*>(levelData.gradY.Data());
    int32_t strideX = levelData.gradX.Stride() / sizeof(float);
    int32_t strideY = levelData.gradY.Stride() / sizeof(float);

    gx = (1 - fx) * (1 - fy) * gxData[y0 * strideX + x0] +
         fx * (1 - fy) * gxData[y0 * strideX + x0 + 1] +
         (1 - fx) * fy * gxData[(y0 + 1) * strideX + x0] +
         fx * fy * gxData[(y0 + 1) * strideX + x0 + 1];

    gy = (1 - fx) * (1 - fy) * gyData[y0 * strideY + x0] +
         fx * (1 - fy) * gyData[y0 * strideY + x0 + 1] +
         (1 - fx) * fy * gyData[(y0 + 1) * strideY + x0] +
         fx * fy * gyData[(y0 + 1) * strideY + x0 + 1];

    return true;
}

bool AnglePyramid::GetGradientAtFast(int32_t level, int32_t x, int32_t y,
                                      float& gx, float& gy) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return false;
    }

    const auto& levelData = impl_->levels_[level];

    if (x < 0 || x >= levelData.width || y < 0 || y >= levelData.height) {
        return false;
    }

    const float* gxData = static_cast<const float*>(levelData.gradX.Data());
    const float* gyData = static_cast<const float*>(levelData.gradY.Data());
    int32_t stride = levelData.gradX.Stride() / sizeof(float);

    gx = gxData[y * stride + x];
    gy = gyData[y * stride + x];

    return true;
}

bool AnglePyramid::GetGradientData(int32_t level, const float*& gxData, const float*& gyData,
                                    int32_t& width, int32_t& height, int32_t& stride) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return false;
    }

    const auto& levelData = impl_->levels_[level];
    gxData = static_cast<const float*>(levelData.gradX.Data());
    gyData = static_cast<const float*>(levelData.gradY.Data());
    width = levelData.width;
    height = levelData.height;
    stride = levelData.gradX.Stride() / sizeof(float);

    return true;
}

std::vector<EdgePoint> AnglePyramid::ExtractEdgePoints(int32_t level,
                                                        const Rect2i& roi,
                                                        double minContrast) const {
    std::vector<EdgePoint> result;

    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return result;
    }

    const auto& levelData = impl_->levels_[level];

    // Determine ROI
    int32_t x0 = 1, y0 = 1;
    int32_t x1 = levelData.width - 1;
    int32_t y1 = levelData.height - 1;

    if (roi.width > 0 && roi.height > 0) {
        x0 = std::max(1, roi.x);
        y0 = std::max(1, roi.y);
        x1 = std::min(levelData.width - 1, roi.x + roi.width);
        y1 = std::min(levelData.height - 1, roi.y + roi.height);
    }

    double threshold = (minContrast > 0) ? minContrast : impl_->params_.minContrast;

    const float* magData = static_cast<const float*>(levelData.gradMag.Data());
    const float* dirData = static_cast<const float*>(levelData.gradDir.Data());
    const int16_t* binData = static_cast<const int16_t*>(levelData.angleBinImage.Data());

    int32_t magStride = levelData.gradMag.Stride() / sizeof(float);
    int32_t dirStride = levelData.gradDir.Stride() / sizeof(float);
    int32_t binStride = levelData.angleBinImage.Stride() / sizeof(int16_t);

    for (int32_t y = y0; y < y1; ++y) {
        for (int32_t x = x0; x < x1; ++x) {
            float mag = magData[y * magStride + x];

            if (mag >= threshold) {
                float dir = dirData[y * dirStride + x];
                int16_t bin = binData[y * binStride + x];

                result.emplace_back(
                    static_cast<double>(x),
                    static_cast<double>(y),
                    static_cast<double>(dir),
                    static_cast<double>(mag),
                    static_cast<int32_t>(bin)
                );
            }
        }
    }

    return result;
}

const std::vector<EdgePoint>& AnglePyramid::GetEdgePoints(int32_t level) const {
    static const std::vector<EdgePoint> empty;
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return empty;
    }
    return impl_->levels_[level].edgePoints;
}

Point2d AnglePyramid::ToLevelCoords(int32_t level, const Point2d& original) const {
    double scale = GetScale(level);
    if (scale <= 0) return original;
    return Point2d{original.x * scale, original.y * scale};
}

Point2d AnglePyramid::ToOriginalCoords(int32_t level, const Point2d& levelCoords) const {
    double scale = GetScale(level);
    if (scale <= 0) return levelCoords;
    return Point2d{levelCoords.x / scale, levelCoords.y / scale};
}

int32_t AnglePyramid::AngleToBin(double angle) const {
    double normalized = NormalizeAngle0To2PI(angle);
    int32_t bin = static_cast<int32_t>(normalized * impl_->params_.angleBins / (2.0 * PI));
    return std::clamp(bin, 0, impl_->params_.angleBins - 1);
}

double AnglePyramid::BinToAngle(int32_t bin) const {
    return (bin + 0.5) * 2.0 * PI / impl_->params_.angleBins;
}

double AnglePyramid::AngleDifference(double angle1, double angle2) {
    double diff = std::abs(angle1 - angle2);
    if (diff > PI) {
        diff = 2.0 * PI - diff;
    }
    return diff;
}

double AnglePyramid::AngleSimilarity(double angle1, double angle2) {
    return std::cos(angle1 - angle2);
}

// =============================================================================
// Utility Functions
// =============================================================================

QImage ComputeGradientDirection(const QImage& gradX, const QImage& gradY) {
    if (gradX.Empty() || gradY.Empty()) {
        return QImage();
    }

    if (gradX.Width() != gradY.Width() || gradX.Height() != gradY.Height()) {
        return QImage();
    }

    int32_t width = gradX.Width();
    int32_t height = gradX.Height();

    QImage result(width, height, PixelType::Float32, ChannelType::Gray);

    const float* gxData = static_cast<const float*>(gradX.Data());
    const float* gyData = static_cast<const float*>(gradY.Data());
    float* outData = static_cast<float*>(result.Data());

    int32_t gxStride = gradX.Stride() / sizeof(float);
    int32_t gyStride = gradY.Stride() / sizeof(float);
    int32_t outStride = result.Stride() / sizeof(float);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            float gx = gxData[y * gxStride + x];
            float gy = gyData[y * gyStride + x];
            float angle = std::atan2(gy, gx);
            if (angle < 0) angle += 2.0f * static_cast<float>(PI);
            outData[y * outStride + x] = angle;
        }
    }

    return result;
}

QImage ComputeGradientMagnitude(const QImage& gradX, const QImage& gradY) {
    if (gradX.Empty() || gradY.Empty()) {
        return QImage();
    }

    if (gradX.Width() != gradY.Width() || gradX.Height() != gradY.Height()) {
        return QImage();
    }

    int32_t width = gradX.Width();
    int32_t height = gradX.Height();

    QImage result(width, height, PixelType::Float32, ChannelType::Gray);

    const float* gxData = static_cast<const float*>(gradX.Data());
    const float* gyData = static_cast<const float*>(gradY.Data());
    float* outData = static_cast<float*>(result.Data());

    int32_t gxStride = gradX.Stride() / sizeof(float);
    int32_t gyStride = gradY.Stride() / sizeof(float);
    int32_t outStride = result.Stride() / sizeof(float);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            float gx = gxData[y * gxStride + x];
            float gy = gyData[y * gyStride + x];
            outData[y * outStride + x] = std::sqrt(gx * gx + gy * gy);
        }
    }

    return result;
}

QImage QuantizeGradientDirection(const QImage& gradDir, int32_t numBins) {
    if (gradDir.Empty() || numBins <= 0) {
        return QImage();
    }

    int32_t width = gradDir.Width();
    int32_t height = gradDir.Height();

    QImage result(width, height, PixelType::Int16, ChannelType::Gray);

    const float* dirData = static_cast<const float*>(gradDir.Data());
    int16_t* outData = static_cast<int16_t*>(result.Data());

    int32_t dirStride = gradDir.Stride() / sizeof(float);
    int32_t outStride = result.Stride() / sizeof(int16_t);

    double binScale = numBins / (2.0 * PI);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            float angle = dirData[y * dirStride + x];
            int32_t bin = static_cast<int32_t>(angle * binScale);
            bin = std::clamp(bin, 0, numBins - 1);
            outData[y * outStride + x] = static_cast<int16_t>(bin);
        }
    }

    return result;
}

int32_t ComputeOptimalPyramidLevels(int32_t width, int32_t height, int32_t minSize) {
    int32_t minDim = std::min(width, height);
    int32_t levels = 1;

    while (minDim >= minSize * 2) {
        minDim /= 2;
        levels++;
    }

    return std::min(levels, ANGLE_PYRAMID_MAX_LEVELS);
}

} // namespace Qi::Vision::Internal
