#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Platform/Memory.h>

#include <cstring>
#include <algorithm>

// stb_image for file I/O
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

namespace Qi::Vision {

// =============================================================================
// Implementation class
// =============================================================================

class QImage::Impl {
public:
    int32_t width_ = 0;
    int32_t height_ = 0;
    PixelType type_ = PixelType::UInt8;
    ChannelType channelType_ = ChannelType::Gray;
    size_t stride_ = 0;
    std::shared_ptr<uint8_t> data_;
    std::shared_ptr<QRegion> domain_;

    // Metadata
    double pixelSizeX_ = 0.0;
    double pixelSizeY_ = 0.0;

    size_t BytesPerPixel() const {
        size_t channelSize = 1;
        switch (type_) {
            case PixelType::UInt8: channelSize = 1; break;
            case PixelType::UInt16:
            case PixelType::Int16: channelSize = 2; break;
            case PixelType::Float32: channelSize = 4; break;
        }

        int numChannels = 1;
        switch (channelType_) {
            case ChannelType::Gray: numChannels = 1; break;
            case ChannelType::RGB:
            case ChannelType::BGR: numChannels = 3; break;
            case ChannelType::RGBA:
            case ChannelType::BGRA: numChannels = 4; break;
        }

        return channelSize * numChannels;
    }

    void Allocate(int32_t w, int32_t h) {
        width_ = w;
        height_ = h;

        size_t bpp = BytesPerPixel();
        stride_ = Platform::AlignedSize(w * bpp, MEMORY_ALIGNMENT);
        size_t totalSize = stride_ * h;

        uint8_t* ptr = static_cast<uint8_t*>(
            Platform::AlignedAlloc(totalSize, MEMORY_ALIGNMENT));

        if (!ptr) {
            throw std::bad_alloc();
        }

        data_ = std::shared_ptr<uint8_t>(ptr, Platform::AlignedDeleter{});
        std::memset(ptr, 0, totalSize);
    }
};

// =============================================================================
// Constructors
// =============================================================================

QImage::QImage() : impl_(std::make_shared<Impl>()) {}

QImage::QImage(int32_t width, int32_t height, PixelType type, ChannelType channels)
    : impl_(std::make_shared<Impl>())
{
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("Image dimensions must be positive");
    }

    impl_->type_ = type;
    impl_->channelType_ = channels;
    impl_->Allocate(width, height);
}

QImage::QImage(const QImage& other) = default;
QImage::QImage(QImage&& other) noexcept = default;
QImage::~QImage() = default;
QImage& QImage::operator=(const QImage& other) = default;
QImage& QImage::operator=(QImage&& other) noexcept = default;

// =============================================================================
// Factory Methods
// =============================================================================

QImage QImage::FromFile(const std::string& path) {
    int w, h, channels;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &channels, 0);

    if (!data) {
        throw IOException("Failed to load image: " + path);
    }

    QImage img;
    img.impl_->type_ = PixelType::UInt8;

    switch (channels) {
        case 1: img.impl_->channelType_ = ChannelType::Gray; break;
        case 3: img.impl_->channelType_ = ChannelType::RGB; break;
        case 4: img.impl_->channelType_ = ChannelType::RGBA; break;
        default:
            stbi_image_free(data);
            throw UnsupportedException("Unsupported channel count: " +
                                       std::to_string(channels));
    }

    img.impl_->Allocate(w, h);

    // Copy row by row (handle stride)
    size_t srcStride = w * channels;
    for (int32_t y = 0; y < h; ++y) {
        std::memcpy(
            static_cast<uint8_t*>(img.RowPtr(y)),
            data + y * srcStride,
            srcStride
        );
    }

    stbi_image_free(data);
    return img;
}

QImage QImage::FromData(const void* data, int32_t width, int32_t height,
                       PixelType type, ChannelType channels) {
    QImage img(width, height, type, channels);

    size_t bpp = img.impl_->BytesPerPixel();
    size_t srcStride = width * bpp;

    const uint8_t* src = static_cast<const uint8_t*>(data);
    for (int32_t y = 0; y < height; ++y) {
        std::memcpy(img.RowPtr(y), src + y * srcStride, srcStride);
    }

    return img;
}

// =============================================================================
// Basic Properties
// =============================================================================

int32_t QImage::Width() const { return impl_->width_; }
int32_t QImage::Height() const { return impl_->height_; }
PixelType QImage::Type() const { return impl_->type_; }
ChannelType QImage::GetChannelType() const { return impl_->channelType_; }
size_t QImage::Stride() const { return impl_->stride_; }
bool QImage::Empty() const { return impl_->width_ == 0 || impl_->height_ == 0; }
bool QImage::IsValid() const { return impl_->data_ != nullptr && !Empty(); }

int QImage::Channels() const {
    switch (impl_->channelType_) {
        case ChannelType::Gray: return 1;
        case ChannelType::RGB:
        case ChannelType::BGR: return 3;
        case ChannelType::RGBA:
        case ChannelType::BGRA: return 4;
    }
    return 1;
}

// =============================================================================
// Data Access
// =============================================================================

void* QImage::Data() { return impl_->data_.get(); }
const void* QImage::Data() const { return impl_->data_.get(); }

void* QImage::RowPtr(int32_t row) {
    return impl_->data_.get() + row * impl_->stride_;
}

const void* QImage::RowPtr(int32_t row) const {
    return impl_->data_.get() + row * impl_->stride_;
}

uint8_t QImage::At(int32_t x, int32_t y) const {
    if (impl_->type_ != PixelType::UInt8 ||
        impl_->channelType_ != ChannelType::Gray) {
        throw UnsupportedException("At() only supports UInt8 grayscale");
    }
    return static_cast<const uint8_t*>(RowPtr(y))[x];
}

void QImage::SetAt(int32_t x, int32_t y, uint8_t value) {
    if (impl_->type_ != PixelType::UInt8 ||
        impl_->channelType_ != ChannelType::Gray) {
        throw UnsupportedException("SetAt() only supports UInt8 grayscale");
    }
    static_cast<uint8_t*>(RowPtr(y))[x] = value;
}

// =============================================================================
// Domain Operations
// =============================================================================

bool QImage::IsFullDomain() const {
    return impl_->domain_ == nullptr;
}

const QRegion* QImage::GetDomain() const {
    return impl_->domain_.get();
}

void QImage::SetDomain(const QRegion& region) {
    impl_->domain_ = std::make_shared<QRegion>(region);
}

void QImage::ReduceDomain(const QRegion& region) {
    if (impl_->domain_) {
        impl_->domain_ = std::make_shared<QRegion>(
            impl_->domain_->Intersection(region));
    } else {
        impl_->domain_ = std::make_shared<QRegion>(region);
    }
}

void QImage::ResetDomain() {
    impl_->domain_.reset();
}

Rect2i QImage::GetDomainBoundingBox() const {
    if (IsFullDomain()) {
        return Rect2i(0, 0, impl_->width_, impl_->height_);
    }
    return impl_->domain_->BoundingBox();
}

// =============================================================================
// Image Operations
// =============================================================================

QImage QImage::Clone() const {
    QImage copy(impl_->width_, impl_->height_,
                impl_->type_, impl_->channelType_);

    // Copy pixel data
    for (int32_t y = 0; y < impl_->height_; ++y) {
        std::memcpy(copy.RowPtr(y), RowPtr(y),
                   impl_->width_ * impl_->BytesPerPixel());
    }

    // Copy domain
    if (impl_->domain_) {
        copy.impl_->domain_ = std::make_shared<QRegion>(*impl_->domain_);
    }

    // Copy metadata
    copy.impl_->pixelSizeX_ = impl_->pixelSizeX_;
    copy.impl_->pixelSizeY_ = impl_->pixelSizeY_;

    return copy;
}

QImage QImage::SubImage(int32_t x, int32_t y, int32_t width, int32_t height) const {
    // Validate parameters
    if (Empty()) return QImage();
    if (x < 0 || y < 0 || width <= 0 || height <= 0) return QImage();
    if (x + width > impl_->width_ || y + height > impl_->height_) return QImage();

    // Create new image
    QImage sub(width, height, impl_->type_, impl_->channelType_);

    // Copy pixel data
    int32_t bytesPerRow = width * impl_->BytesPerPixel();
    for (int32_t row = 0; row < height; ++row) {
        const uint8_t* srcRow = static_cast<const uint8_t*>(RowPtr(y + row));
        uint8_t* dstRow = static_cast<uint8_t*>(sub.RowPtr(row));
        std::memcpy(dstRow, srcRow + x * impl_->BytesPerPixel(), bytesPerRow);
    }

    // Copy metadata
    sub.impl_->pixelSizeX_ = impl_->pixelSizeX_;
    sub.impl_->pixelSizeY_ = impl_->pixelSizeY_;

    return sub;
}

bool QImage::SaveToFile(const std::string& path) const {
    if (Empty()) return false;

    // Only support UInt8 for now
    if (impl_->type_ != PixelType::UInt8) {
        return false;
    }

    int channels = Channels();

    // Create contiguous buffer
    std::vector<uint8_t> buffer(impl_->width_ * impl_->height_ * channels);
    size_t srcStride = impl_->width_ * channels;

    for (int32_t y = 0; y < impl_->height_; ++y) {
        std::memcpy(buffer.data() + y * srcStride, RowPtr(y), srcStride);
    }

    // Determine format from extension
    if (path.size() >= 4) {
        std::string ext = path.substr(path.size() - 4);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".png") {
            return stbi_write_png(path.c_str(), impl_->width_, impl_->height_,
                                  channels, buffer.data(), srcStride) != 0;
        } else if (ext == ".jpg" || ext == "jpeg") {
            return stbi_write_jpg(path.c_str(), impl_->width_, impl_->height_,
                                  channels, buffer.data(), 95) != 0;
        } else if (ext == ".bmp") {
            return stbi_write_bmp(path.c_str(), impl_->width_, impl_->height_,
                                  channels, buffer.data()) != 0;
        }
    }

    // Default to PNG
    return stbi_write_png(path.c_str(), impl_->width_, impl_->height_,
                          channels, buffer.data(), srcStride) != 0;
}

// =============================================================================
// Pixel Type Conversion
// =============================================================================

QImage QImage::ConvertTo(PixelType targetType) const {
    if (Empty()) return QImage();

    // Same type - return clone
    if (impl_->type_ == targetType) {
        return Clone();
    }

    // Create output image
    QImage result(impl_->width_, impl_->height_, targetType, impl_->channelType_);

    int numChannels = Channels();
    PixelType srcType = impl_->type_;

    for (int32_t y = 0; y < impl_->height_; ++y) {
        const uint8_t* srcRow = static_cast<const uint8_t*>(RowPtr(y));
        uint8_t* dstRow = static_cast<uint8_t*>(result.RowPtr(y));

        for (int32_t x = 0; x < impl_->width_; ++x) {
            for (int c = 0; c < numChannels; ++c) {
                // Read source value as double
                double value = 0.0;

                switch (srcType) {
                    case PixelType::UInt8: {
                        value = static_cast<double>(srcRow[x * numChannels + c]) / 255.0;
                        break;
                    }
                    case PixelType::UInt16: {
                        const uint16_t* src16 = reinterpret_cast<const uint16_t*>(srcRow);
                        value = static_cast<double>(src16[x * numChannels + c]) / 65535.0;
                        break;
                    }
                    case PixelType::Int16: {
                        const int16_t* srcS16 = reinterpret_cast<const int16_t*>(srcRow);
                        value = (static_cast<double>(srcS16[x * numChannels + c]) + 32768.0) / 65535.0;
                        break;
                    }
                    case PixelType::Float32: {
                        const float* srcF = reinterpret_cast<const float*>(srcRow);
                        value = static_cast<double>(srcF[x * numChannels + c]);
                        break;
                    }
                }

                // Clamp to [0, 1]
                value = std::max(0.0, std::min(1.0, value));

                // Write to target
                switch (targetType) {
                    case PixelType::UInt8: {
                        dstRow[x * numChannels + c] = static_cast<uint8_t>(value * 255.0 + 0.5);
                        break;
                    }
                    case PixelType::UInt16: {
                        uint16_t* dst16 = reinterpret_cast<uint16_t*>(dstRow);
                        dst16[x * numChannels + c] = static_cast<uint16_t>(value * 65535.0 + 0.5);
                        break;
                    }
                    case PixelType::Int16: {
                        int16_t* dstS16 = reinterpret_cast<int16_t*>(dstRow);
                        dstS16[x * numChannels + c] = static_cast<int16_t>(value * 65535.0 - 32768.0 + 0.5);
                        break;
                    }
                    case PixelType::Float32: {
                        float* dstF = reinterpret_cast<float*>(dstRow);
                        dstF[x * numChannels + c] = static_cast<float>(value);
                        break;
                    }
                }
            }
        }
    }

    // Copy metadata
    result.impl_->pixelSizeX_ = impl_->pixelSizeX_;
    result.impl_->pixelSizeY_ = impl_->pixelSizeY_;

    // Copy domain if present
    if (impl_->domain_) {
        result.impl_->domain_ = std::make_shared<QRegion>(*impl_->domain_);
    }

    return result;
}

// =============================================================================
// Color Conversion
// =============================================================================

QImage QImage::ToGray() const {
    if (Empty()) return QImage();

    // Already grayscale - return copy
    if (impl_->channelType_ == ChannelType::Gray) {
        return Clone();
    }

    // Only support UInt8 RGB/RGBA
    if (impl_->type_ != PixelType::UInt8) {
        return QImage();
    }

    int srcChannels = Channels();
    if (srcChannels < 3) {
        return QImage();
    }

    // Create grayscale output
    QImage result(impl_->width_, impl_->height_, PixelType::UInt8, ChannelType::Gray);

    // RGB to Gray: Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601)
    const int32_t rWeight = 77;   // 0.299 * 256
    const int32_t gWeight = 150;  // 0.587 * 256
    const int32_t bWeight = 29;   // 0.114 * 256

    for (int32_t y = 0; y < impl_->height_; ++y) {
        const uint8_t* src = static_cast<const uint8_t*>(RowPtr(y));
        uint8_t* dst = static_cast<uint8_t*>(result.RowPtr(y));

        for (int32_t x = 0; x < impl_->width_; ++x) {
            int r = src[x * srcChannels + 0];
            int g = src[x * srcChannels + 1];
            int b = src[x * srcChannels + 2];
            dst[x] = static_cast<uint8_t>((r * rWeight + g * gWeight + b * bWeight) >> 8);
        }
    }

    return result;
}

// =============================================================================
// Metadata
// =============================================================================

double QImage::PixelSizeX() const { return impl_->pixelSizeX_; }
double QImage::PixelSizeY() const { return impl_->pixelSizeY_; }

void QImage::SetPixelSize(double sizeX, double sizeY) {
    impl_->pixelSizeX_ = sizeX;
    impl_->pixelSizeY_ = sizeY;
}

} // namespace Qi::Vision
