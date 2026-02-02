/**
 * @file IntegralImage.cpp
 * @brief Implementation of integral image for fast region computations
 */

#include <QiVision/Internal/IntegralImage.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// Implementation Class
// =============================================================================

class IntegralImage::Impl {
public:
    std::vector<double> integral_;          // Sum integral (double)
    std::vector<double> squaredIntegral_;   // Squared sum integral (double)
    int32_t width_ = 0;        // Original image width
    int32_t height_ = 0;       // Original image height
    int32_t integralWidth_ = 0;  // Integral image width (width + 1)
    int32_t integralHeight_ = 0; // Integral image height (height + 1)
    bool valid_ = false;
    bool hasSquared_ = false;

    double GetValueAt(const std::vector<double>& data, int32_t x, int32_t y) const {
        if (x < 0 || y < 0) return 0.0;
        if (x >= integralWidth_ || y >= integralHeight_) return 0.0;
        return data[y * integralWidth_ + x];
    }
};

// =============================================================================
// IntegralImage Implementation
// =============================================================================

IntegralImage::IntegralImage() : impl_(std::make_unique<Impl>()) {}

IntegralImage::~IntegralImage() = default;

IntegralImage::IntegralImage(const IntegralImage& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

IntegralImage::IntegralImage(IntegralImage&& other) noexcept = default;

IntegralImage& IntegralImage::operator=(const IntegralImage& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

IntegralImage& IntegralImage::operator=(IntegralImage&& other) noexcept = default;

bool IntegralImage::Compute(const QImage& image, bool computeSquared) {
    Clear();

    if (image.Empty()) {
        return false;
    }

    impl_->width_ = image.Width();
    impl_->height_ = image.Height();

    // Integral image is (width+1) x (height+1) to handle boundary cases
    // integral_[y][x] = sum of all pixels (0,0) to (x-1, y-1)
    impl_->integralWidth_ = impl_->width_ + 1;
    impl_->integralHeight_ = impl_->height_ + 1;

    // Create integral images (double precision for accuracy)
    size_t totalSize = static_cast<size_t>(impl_->integralWidth_) * impl_->integralHeight_;
    impl_->integral_.resize(totalSize, 0.0);

    double* integralData = impl_->integral_.data();
    int32_t integralStride = impl_->integralWidth_;

    double* squaredData = nullptr;
    int32_t squaredStride = 0;

    if (computeSquared) {
        impl_->squaredIntegral_.resize(totalSize, 0.0);
        squaredData = impl_->squaredIntegral_.data();
        squaredStride = impl_->integralWidth_;
    }

    // Compute integral image
    // I(x,y) = i(x,y) + I(x-1,y) + I(x,y-1) - I(x-1,y-1)
    // where i(x,y) is the original pixel value

    if (image.Type() == PixelType::UInt8) {
        const uint8_t* srcData = static_cast<const uint8_t*>(image.Data());
        int32_t srcStride = image.Stride();

        for (int32_t y = 1; y < impl_->integralHeight_; ++y) {
            for (int32_t x = 1; x < impl_->integralWidth_; ++x) {
                double pixel = static_cast<double>(srcData[(y - 1) * srcStride + (x - 1)]);

                integralData[y * integralStride + x] =
                    pixel +
                    integralData[y * integralStride + (x - 1)] +
                    integralData[(y - 1) * integralStride + x] -
                    integralData[(y - 1) * integralStride + (x - 1)];

                if (computeSquared) {
                    squaredData[y * squaredStride + x] =
                        pixel * pixel +
                        squaredData[y * squaredStride + (x - 1)] +
                        squaredData[(y - 1) * squaredStride + x] -
                        squaredData[(y - 1) * squaredStride + (x - 1)];
                }
            }
        }
    } else if (image.Type() == PixelType::Float32) {
        const float* srcData = static_cast<const float*>(image.Data());
        int32_t srcStride = image.Stride() / sizeof(float);

        for (int32_t y = 1; y < impl_->integralHeight_; ++y) {
            for (int32_t x = 1; x < impl_->integralWidth_; ++x) {
                double pixel = static_cast<double>(srcData[(y - 1) * srcStride + (x - 1)]);

                integralData[y * integralStride + x] =
                    pixel +
                    integralData[y * integralStride + (x - 1)] +
                    integralData[(y - 1) * integralStride + x] -
                    integralData[(y - 1) * integralStride + (x - 1)];

                if (computeSquared) {
                    squaredData[y * squaredStride + x] =
                        pixel * pixel +
                        squaredData[y * squaredStride + (x - 1)] +
                        squaredData[(y - 1) * squaredStride + x] -
                        squaredData[(y - 1) * squaredStride + (x - 1)];
                }
            }
        }
    } else {
        // Unsupported pixel type
        Clear();
        return false;
    }

    impl_->valid_ = true;
    impl_->hasSquared_ = computeSquared;
    return true;
}

void IntegralImage::Clear() {
    impl_->integral_.clear();
    impl_->squaredIntegral_.clear();
    impl_->width_ = 0;
    impl_->height_ = 0;
    impl_->integralWidth_ = 0;
    impl_->integralHeight_ = 0;
    impl_->valid_ = false;
    impl_->hasSquared_ = false;
}

bool IntegralImage::IsValid() const {
    return impl_->valid_;
}

bool IntegralImage::HasSquaredIntegral() const {
    return impl_->hasSquared_;
}

int32_t IntegralImage::Width() const {
    return impl_->width_;
}

int32_t IntegralImage::Height() const {
    return impl_->height_;
}

double IntegralImage::GetRectSum(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const {
    if (!impl_->valid_) return 0.0;

    // Clamp to valid range
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(impl_->width_ - 1, x2);
    y2 = std::min(impl_->height_ - 1, y2);

    if (x1 > x2 || y1 > y2) return 0.0;

    // Convert to integral image coordinates (offset by 1)
    // Sum = I(x2+1, y2+1) - I(x1, y2+1) - I(x2+1, y1) + I(x1, y1)
    double A = impl_->GetValueAt(impl_->integral_, x1, y1);
    double B = impl_->GetValueAt(impl_->integral_, x2 + 1, y1);
    double C = impl_->GetValueAt(impl_->integral_, x1, y2 + 1);
    double D = impl_->GetValueAt(impl_->integral_, x2 + 1, y2 + 1);

    return D - B - C + A;
}

double IntegralImage::GetRectSumSquared(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const {
    if (!impl_->valid_ || !impl_->hasSquared_) return 0.0;

    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(impl_->width_ - 1, x2);
    y2 = std::min(impl_->height_ - 1, y2);

    if (x1 > x2 || y1 > y2) return 0.0;

    double A = impl_->GetValueAt(impl_->squaredIntegral_, x1, y1);
    double B = impl_->GetValueAt(impl_->squaredIntegral_, x2 + 1, y1);
    double C = impl_->GetValueAt(impl_->squaredIntegral_, x1, y2 + 1);
    double D = impl_->GetValueAt(impl_->squaredIntegral_, x2 + 1, y2 + 1);

    return D - B - C + A;
}

double IntegralImage::GetRectMean(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const {
    int64_t count = GetRectCount(x1, y1, x2, y2);
    if (count == 0) return 0.0;
    return GetRectSum(x1, y1, x2, y2) / count;
}

double IntegralImage::GetRectVariance(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const {
    if (!impl_->hasSquared_) return 0.0;

    int64_t count = GetRectCount(x1, y1, x2, y2);
    if (count == 0) return 0.0;

    double sum = GetRectSum(x1, y1, x2, y2);
    double sumSq = GetRectSumSquared(x1, y1, x2, y2);

    double mean = sum / count;
    double meanSq = sumSq / count;

    // Variance = E[X^2] - E[X]^2
    return std::max(0.0, meanSq - mean * mean);
}

double IntegralImage::GetRectStdDev(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const {
    return std::sqrt(GetRectVariance(x1, y1, x2, y2));
}

int64_t IntegralImage::GetRectCount(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const {
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(impl_->width_ - 1, x2);
    y2 = std::min(impl_->height_ - 1, y2);

    if (x1 > x2 || y1 > y2) return 0;

    return static_cast<int64_t>(x2 - x1 + 1) * (y2 - y1 + 1);
}

double IntegralImage::GetIntegralAt(int32_t x, int32_t y) const {
    return impl_->GetValueAt(impl_->integral_, x, y);
}

double IntegralImage::GetSquaredIntegralAt(int32_t x, int32_t y) const {
    return impl_->GetValueAt(impl_->squaredIntegral_, x, y);
}

const double* IntegralImage::GetIntegralData() const {
    if (!impl_->valid_ || impl_->integral_.empty()) return nullptr;
    return impl_->integral_.data();
}

const double* IntegralImage::GetSquaredIntegralData() const {
    if (!impl_->valid_ || !impl_->hasSquared_ || impl_->squaredIntegral_.empty()) return nullptr;
    return impl_->squaredIntegral_.data();
}

int32_t IntegralImage::IntegralWidth() const {
    return impl_->integralWidth_;
}

int32_t IntegralImage::IntegralHeight() const {
    return impl_->integralHeight_;
}

// =============================================================================
// Utility Functions
// =============================================================================

template<typename T>
void ComputeIntegralImage(const T* src, int32_t srcWidth, int32_t srcHeight,
                          int32_t srcStride, double* integral, int32_t intStride) {
    int32_t intWidth = srcWidth + 1;
    int32_t intHeight = srcHeight + 1;

    // Initialize first row and column to zero
    for (int32_t x = 0; x < intWidth; ++x) {
        integral[x] = 0.0;
    }
    for (int32_t y = 0; y < intHeight; ++y) {
        integral[y * intStride] = 0.0;
    }

    // Compute integral image
    for (int32_t y = 1; y < intHeight; ++y) {
        for (int32_t x = 1; x < intWidth; ++x) {
            double pixel = static_cast<double>(src[(y - 1) * srcStride + (x - 1)]);
            integral[y * intStride + x] =
                pixel +
                integral[y * intStride + (x - 1)] +
                integral[(y - 1) * intStride + x] -
                integral[(y - 1) * intStride + (x - 1)];
        }
    }
}

// Explicit template instantiations
template void ComputeIntegralImage<uint8_t>(const uint8_t*, int32_t, int32_t, int32_t, double*, int32_t);
template void ComputeIntegralImage<float>(const float*, int32_t, int32_t, int32_t, double*, int32_t);
template void ComputeIntegralImage<double>(const double*, int32_t, int32_t, int32_t, double*, int32_t);

double GetRectSumFromIntegral(const double* integral, int32_t intWidth, int32_t intHeight,
                               int32_t x1, int32_t y1, int32_t x2, int32_t y2) {
    if (!integral) return 0.0;

    auto getValue = [&](int32_t x, int32_t y) -> double {
        if (x < 0 || y < 0 || x >= intWidth || y >= intHeight) return 0.0;
        return integral[y * intWidth + x];
    };

    // Note: integral image is 1 larger than source, so coordinates are offset
    double A = getValue(x1, y1);
    double B = getValue(x2 + 1, y1);
    double C = getValue(x1, y2 + 1);
    double D = getValue(x2 + 1, y2 + 1);

    return D - B - C + A;
}

} // namespace Qi::Vision::Internal
