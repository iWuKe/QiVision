/**
 * @file VariationModel.cpp
 * @brief Implementation of Variation Model for defect detection
 */

#include <QiVision/Defect/VariationModel.h>
#include <QiVision/Filter/Filter.h>
#include <QiVision/Internal/RLEOps.h>
#include <QiVision/Internal/MorphBinary.h>
#include <QiVision/Internal/StructElement.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Platform/Memory.h>

#include <cmath>
#include <fstream>
#include <algorithm>
#include <cstring>

namespace Qi::Vision::Defect {

using namespace Internal;

// =============================================================================
// Implementation class
// =============================================================================

class VariationModel::Impl {
public:
    int32_t width_ = 0;
    int32_t height_ = 0;

    // Accumulated statistics for training
    QImage sumImage_;      // Sum of all training images (Float32)
    QImage sumSqImage_;    // Sum of squared values (Float32)
    int32_t trainCount_ = 0;

    // Final model
    QImage meanImage_;     // Mean image (Float32)
    QImage varImage_;      // Variance image (Float32)
    double minVariance_ = 1.0;
    bool isReady_ = false;

    void InitAccumulators(int32_t w, int32_t h) {
        width_ = w;
        height_ = h;
        sumImage_ = QImage(w, h, PixelType::Float32);
        sumSqImage_ = QImage(w, h, PixelType::Float32);

        // Initialize to zero
        std::memset(sumImage_.Data(), 0, sumImage_.Stride() * h);
        std::memset(sumSqImage_.Data(), 0, sumSqImage_.Stride() * h);

        trainCount_ = 0;
        isReady_ = false;
    }

    void AccumulateImage(const QImage& image) {
        // Convert to float if needed
        QImage floatImg;
        if (image.Type() == PixelType::Float32) {
            floatImg = image;
        } else {
            floatImg = QImage(width_, height_, PixelType::Float32);
            const uint8_t* src = static_cast<const uint8_t*>(image.Data());
            float* dst = static_cast<float*>(floatImg.Data());
            int32_t srcStride = image.Stride();
            int32_t dstStride = floatImg.Stride() / sizeof(float);

            for (int32_t y = 0; y < height_; ++y) {
                for (int32_t x = 0; x < width_; ++x) {
                    dst[y * dstStride + x] = static_cast<float>(src[y * srcStride + x]);
                }
            }
        }

        // Accumulate sum and sum of squares
        float* sumPtr = static_cast<float*>(sumImage_.Data());
        float* sumSqPtr = static_cast<float*>(sumSqImage_.Data());
        const float* imgPtr = static_cast<const float*>(floatImg.Data());

        int32_t sumStride = sumImage_.Stride() / sizeof(float);
        int32_t imgStride = floatImg.Stride() / sizeof(float);

        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                float val = imgPtr[y * imgStride + x];
                sumPtr[y * sumStride + x] += val;
                sumSqPtr[y * sumStride + x] += val * val;
            }
        }

        trainCount_++;
    }

    void ComputeStatistics() {
        if (trainCount_ == 0) {
            throw Exception("No training images provided");
        }

        meanImage_ = QImage(width_, height_, PixelType::Float32);
        varImage_ = QImage(width_, height_, PixelType::Float32);

        const float* sumPtr = static_cast<const float*>(sumImage_.Data());
        const float* sumSqPtr = static_cast<const float*>(sumSqImage_.Data());
        float* meanPtr = static_cast<float*>(meanImage_.Data());
        float* varPtr = static_cast<float*>(varImage_.Data());

        int32_t stride = sumImage_.Stride() / sizeof(float);
        float n = static_cast<float>(trainCount_);
        float minVar = static_cast<float>(minVariance_);

        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                int32_t idx = y * stride + x;
                float sum = sumPtr[idx];
                float sumSq = sumSqPtr[idx];

                float mean = sum / n;
                // Var = E[X²] - E[X]²
                float variance = (sumSq / n) - (mean * mean);
                variance = std::max(variance, minVar);

                meanPtr[idx] = mean;
                varPtr[idx] = variance;
            }
        }

        // Free accumulators
        sumImage_ = QImage();
        sumSqImage_ = QImage();

        isReady_ = true;
    }

    void CreateFromSingleImageImpl(
        const QImage& golden,
        double edgeTolerance,
        double flatTolerance,
        double edgeSigma,
        int32_t edgeDilateRadius
    ) {
        width_ = golden.Width();
        height_ = golden.Height();

        // Create mean image (just the golden image as float)
        meanImage_ = QImage(width_, height_, PixelType::Float32);

        if (golden.Type() == PixelType::Float32) {
            std::memcpy(meanImage_.Data(), golden.Data(),
                        meanImage_.Stride() * height_);
        } else {
            const uint8_t* src = static_cast<const uint8_t*>(golden.Data());
            float* dst = static_cast<float*>(meanImage_.Data());
            int32_t srcStride = golden.Stride();
            int32_t dstStride = meanImage_.Stride() / sizeof(float);

            for (int32_t y = 0; y < height_; ++y) {
                for (int32_t x = 0; x < width_; ++x) {
                    dst[y * dstStride + x] = static_cast<float>(src[y * srcStride + x]);
                }
            }
        }

        // Compute gradient magnitude for edge detection using Sobel
        QImage gradMag;
        Filter::SobelAmp(golden, gradMag, "sum_abs", 3);

        // Create variance image
        varImage_ = QImage(width_, height_, PixelType::Float32);
        float* varPtr = static_cast<float*>(varImage_.Data());
        int32_t varStride = varImage_.Stride() / sizeof(float);

        // Find max gradient magnitude
        const uint8_t* magPtr = static_cast<const uint8_t*>(gradMag.Data());
        int32_t magStride = gradMag.Stride();

        float maxMag = 0.0f;
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                float mag = static_cast<float>(magPtr[y * magStride + x]);
                maxMag = std::max(maxMag, mag);
            }
        }

        // Determine edge threshold (e.g., 10% of max gradient)
        float edgeThresh = maxMag * 0.1f;

        // Create edge mask (binary)
        QImage edgeMask(width_, height_, PixelType::UInt8);
        uint8_t* maskPtr = static_cast<uint8_t*>(edgeMask.Data());
        int32_t maskStride2 = edgeMask.Stride();

        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                float mag = static_cast<float>(magPtr[y * magStride + x]);
                maskPtr[y * maskStride2 + x] = (mag > edgeThresh) ? 255 : 0;
            }
        }

        // Convert to region and dilate
        QRegion edgeRegion = ThresholdToRegion(edgeMask, 128, 255);

        if (edgeDilateRadius > 0) {
            auto se = StructElement::Circle(edgeDilateRadius);
            edgeRegion = Dilate(edgeRegion, se);
        }

        // Create variance image based on edge/flat regions
        float edgeVar = static_cast<float>(edgeTolerance * edgeTolerance);
        float flatVar = static_cast<float>(flatTolerance * flatTolerance);

        // Initialize with flat variance
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                varPtr[y * varStride + x] = flatVar;
            }
        }

        // Set edge regions to edge variance
        const auto& runs = edgeRegion.Runs();
        for (const auto& run : runs) {
            if (run.row >= 0 && run.row < height_) {
                int32_t xStart = std::max(0, run.colBegin);
                int32_t xEnd = std::min(width_, run.colEnd);
                for (int32_t x = xStart; x < xEnd; ++x) {
                    varPtr[run.row * varStride + x] = edgeVar;
                }
            }
        }

        trainCount_ = 1;
        isReady_ = true;
    }

    QRegion CompareImpl(const QImage& testImage, double threshold) const {
        if (!isReady_) {
            throw Exception("Model not ready. Call Prepare() or CreateFromSingleImage() first.");
        }

        if (testImage.Width() != width_ || testImage.Height() != height_) {
            throw Exception("Test image size doesn't match model size");
        }

        // Convert test image to float if needed
        QImage testFloat;
        if (testImage.Type() == PixelType::Float32) {
            testFloat = testImage;
        } else {
            testFloat = QImage(width_, height_, PixelType::Float32);
            const uint8_t* src = static_cast<const uint8_t*>(testImage.Data());
            float* dst = static_cast<float*>(testFloat.Data());
            int32_t srcStride = testImage.Stride();
            int32_t dstStride = testFloat.Stride() / sizeof(float);

            for (int32_t y = 0; y < height_; ++y) {
                for (int32_t x = 0; x < width_; ++x) {
                    dst[y * dstStride + x] = static_cast<float>(src[y * srcStride + x]);
                }
            }
        }

        // Compare: |test - mean| > threshold * sqrt(var)
        const float* testPtr = static_cast<const float*>(testFloat.Data());
        const float* meanPtr = static_cast<const float*>(meanImage_.Data());
        const float* varPtr = static_cast<const float*>(varImage_.Data());

        int32_t testStride = testFloat.Stride() / sizeof(float);
        int32_t meanStride = meanImage_.Stride() / sizeof(float);
        int32_t varStride = varImage_.Stride() / sizeof(float);

        float thresh = static_cast<float>(threshold);

        // Build defect region using RLE
        std::vector<Run> runs;

        for (int32_t y = 0; y < height_; ++y) {
            int32_t runStart = -1;

            for (int32_t x = 0; x < width_; ++x) {
                float test = testPtr[y * testStride + x];
                float mean = meanPtr[y * meanStride + x];
                float var = varPtr[y * varStride + x];

                float diff = std::abs(test - mean);
                float sigma = std::sqrt(var);
                bool isDefect = (diff > thresh * sigma);

                if (isDefect) {
                    if (runStart < 0) {
                        runStart = x;
                    }
                } else {
                    if (runStart >= 0) {
                        runs.push_back({y, runStart, x});
                        runStart = -1;
                    }
                }
            }

            // Close run at end of row
            if (runStart >= 0) {
                runs.push_back({y, runStart, width_});
            }
        }

        return QRegion(std::move(runs));
    }

    void GetDiffImageImpl(const QImage& testImage, QImage& diffImage) const {
        if (!isReady_) {
            throw Exception("Model not ready");
        }

        diffImage = QImage(width_, height_, PixelType::Float32);

        // Convert test to float
        QImage testFloat;
        if (testImage.Type() == PixelType::Float32) {
            testFloat = testImage;
        } else {
            testFloat = QImage(width_, height_, PixelType::Float32);
            const uint8_t* src = static_cast<const uint8_t*>(testImage.Data());
            float* dst = static_cast<float*>(testFloat.Data());
            int32_t srcStride = testImage.Stride();
            int32_t dstStride = testFloat.Stride() / sizeof(float);

            for (int32_t y = 0; y < height_; ++y) {
                for (int32_t x = 0; x < width_; ++x) {
                    dst[y * dstStride + x] = static_cast<float>(src[y * srcStride + x]);
                }
            }
        }

        const float* testPtr = static_cast<const float*>(testFloat.Data());
        const float* meanPtr = static_cast<const float*>(meanImage_.Data());
        const float* varPtr = static_cast<const float*>(varImage_.Data());
        float* diffPtr = static_cast<float*>(diffImage.Data());

        int32_t testStride = testFloat.Stride() / sizeof(float);
        int32_t meanStride = meanImage_.Stride() / sizeof(float);
        int32_t varStride = varImage_.Stride() / sizeof(float);
        int32_t diffStride = diffImage.Stride() / sizeof(float);

        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                float test = testPtr[y * testStride + x];
                float mean = meanPtr[y * meanStride + x];
                float var = varPtr[y * varStride + x];

                float diff = std::abs(test - mean);
                float sigma = std::sqrt(var);

                diffPtr[y * diffStride + x] = diff / sigma;
            }
        }
    }
};

// =============================================================================
// VariationModel implementation
// =============================================================================

VariationModel::VariationModel(int32_t width, int32_t height)
    : impl_(std::make_unique<Impl>())
{
    if (width > 0 && height > 0) {
        impl_->InitAccumulators(width, height);
    }
}

VariationModel::~VariationModel() = default;

VariationModel::VariationModel(VariationModel&& other) noexcept = default;
VariationModel& VariationModel::operator=(VariationModel&& other) noexcept = default;

void VariationModel::Train(const QImage& goodImage) {
    if (impl_->width_ == 0) {
        impl_->InitAccumulators(goodImage.Width(), goodImage.Height());
    }

    if (goodImage.Width() != impl_->width_ || goodImage.Height() != impl_->height_) {
        throw Exception("Training image size doesn't match model size");
    }

    impl_->AccumulateImage(goodImage);
}

void VariationModel::Prepare() {
    impl_->ComputeStatistics();
}

void VariationModel::CreateFromSingleImage(
    const QImage& golden,
    double edgeTolerance,
    double flatTolerance,
    double edgeSigma,
    int32_t edgeDilateRadius
) {
    impl_->CreateFromSingleImageImpl(golden, edgeTolerance, flatTolerance,
                                      edgeSigma, edgeDilateRadius);
}

void VariationModel::CreateFromImages(const QImage& golden, const QImage& varImage) {
    impl_->width_ = golden.Width();
    impl_->height_ = golden.Height();

    // Copy mean image
    impl_->meanImage_ = QImage(impl_->width_, impl_->height_, PixelType::Float32);
    if (golden.Type() == PixelType::Float32) {
        std::memcpy(impl_->meanImage_.Data(), golden.Data(),
                    impl_->meanImage_.Stride() * impl_->height_);
    } else {
        const uint8_t* src = static_cast<const uint8_t*>(golden.Data());
        float* dst = static_cast<float*>(impl_->meanImage_.Data());
        int32_t srcStride = golden.Stride();
        int32_t dstStride = impl_->meanImage_.Stride() / sizeof(float);

        for (int32_t y = 0; y < impl_->height_; ++y) {
            for (int32_t x = 0; x < impl_->width_; ++x) {
                dst[y * dstStride + x] = static_cast<float>(src[y * srcStride + x]);
            }
        }
    }

    // Copy variance image
    impl_->varImage_ = varImage.Clone();
    impl_->trainCount_ = 1;
    impl_->isReady_ = true;
}

QRegion VariationModel::Compare(const QImage& testImage, double threshold) const {
    return impl_->CompareImpl(testImage, threshold);
}

QRegion VariationModel::Compare(const QImage& testImage, const QRegion& roi,
                                 double threshold) const {
    QRegion fullDefects = impl_->CompareImpl(testImage, threshold);
    return fullDefects.Intersection(roi);
}

void VariationModel::GetDiffImage(const QImage& testImage, QImage& diffImage) const {
    impl_->GetDiffImageImpl(testImage, diffImage);
}

QImage VariationModel::GetMeanImage() const {
    return impl_->meanImage_;
}

QImage VariationModel::GetVarImage() const {
    return impl_->varImage_;
}

void VariationModel::SetVarImage(const QImage& varImage) {
    if (varImage.Width() != impl_->width_ || varImage.Height() != impl_->height_) {
        throw Exception("Variance image size doesn't match model size");
    }
    impl_->varImage_ = varImage.Clone();
}

void VariationModel::SetMinVariance(double minVar) {
    impl_->minVariance_ = minVar;
}

int32_t VariationModel::Width() const {
    return impl_->width_;
}

int32_t VariationModel::Height() const {
    return impl_->height_;
}

bool VariationModel::IsReady() const {
    return impl_->isReady_;
}

int32_t VariationModel::TrainingCount() const {
    return impl_->trainCount_;
}

void VariationModel::Write(const std::string& filename) const {
    if (!impl_->isReady_) {
        throw Exception("Model not ready for serialization");
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw Exception("Failed to open file for writing: " + filename);
    }

    // Magic number and version
    const char magic[] = "QIVM";
    int32_t version = 1;
    file.write(magic, 4);
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Dimensions
    file.write(reinterpret_cast<const char*>(&impl_->width_), sizeof(impl_->width_));
    file.write(reinterpret_cast<const char*>(&impl_->height_), sizeof(impl_->height_));
    file.write(reinterpret_cast<const char*>(&impl_->trainCount_), sizeof(impl_->trainCount_));
    file.write(reinterpret_cast<const char*>(&impl_->minVariance_), sizeof(impl_->minVariance_));

    // Mean image data
    int32_t stride = impl_->meanImage_.Stride();
    file.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
    file.write(reinterpret_cast<const char*>(impl_->meanImage_.Data()),
               stride * impl_->height_);

    // Variance image data
    stride = impl_->varImage_.Stride();
    file.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
    file.write(reinterpret_cast<const char*>(impl_->varImage_.Data()),
               stride * impl_->height_);
}

VariationModel VariationModel::Read(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw Exception("Failed to open file for reading: " + filename);
    }

    // Check magic number
    char magic[5] = {0};
    file.read(magic, 4);
    if (std::string(magic) != "QIVM") {
        throw Exception("Invalid variation model file format");
    }

    int32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        throw Exception("Unsupported variation model version");
    }

    VariationModel model;
    auto& impl = *model.impl_;

    // Dimensions
    file.read(reinterpret_cast<char*>(&impl.width_), sizeof(impl.width_));
    file.read(reinterpret_cast<char*>(&impl.height_), sizeof(impl.height_));
    file.read(reinterpret_cast<char*>(&impl.trainCount_), sizeof(impl.trainCount_));
    file.read(reinterpret_cast<char*>(&impl.minVariance_), sizeof(impl.minVariance_));

    // Mean image
    int32_t stride;
    file.read(reinterpret_cast<char*>(&stride), sizeof(stride));
    impl.meanImage_ = QImage(impl.width_, impl.height_, PixelType::Float32);
    file.read(reinterpret_cast<char*>(impl.meanImage_.Data()),
              stride * impl.height_);

    // Variance image
    file.read(reinterpret_cast<char*>(&stride), sizeof(stride));
    impl.varImage_ = QImage(impl.width_, impl.height_, PixelType::Float32);
    file.read(reinterpret_cast<char*>(impl.varImage_.Data()),
              stride * impl.height_);

    impl.isReady_ = true;
    return model;
}

VariationModel VariationModel::Clone() const {
    VariationModel clone;
    clone.impl_->width_ = impl_->width_;
    clone.impl_->height_ = impl_->height_;
    clone.impl_->trainCount_ = impl_->trainCount_;
    clone.impl_->minVariance_ = impl_->minVariance_;
    clone.impl_->isReady_ = impl_->isReady_;

    if (!impl_->meanImage_.Empty()) {
        clone.impl_->meanImage_ = impl_->meanImage_.Clone();
    }
    if (!impl_->varImage_.Empty()) {
        clone.impl_->varImage_ = impl_->varImage_.Clone();
    }

    return clone;
}

// =============================================================================
// Convenience functions
// =============================================================================

QRegion CompareImages(
    const QImage& golden,
    const QImage& test,
    double tolerance
) {
    VariationModel model;
    model.CreateFromSingleImage(golden, tolerance, tolerance, 1.5, 0);
    return model.Compare(test, 1.0);  // threshold=1 since tolerance is already in variance
}

QRegion CompareImagesEdgeAware(
    const QImage& golden,
    const QImage& test,
    double edgeTolerance,
    double flatTolerance
) {
    VariationModel model;
    model.CreateFromSingleImage(golden, edgeTolerance, flatTolerance);
    return model.Compare(test, 1.0);
}

QRegion AbsDiffThreshold(
    const QImage& image1,
    const QImage& image2,
    double threshold
) {
    if (image1.Width() != image2.Width() || image1.Height() != image2.Height()) {
        throw Exception("Image sizes don't match");
    }

    int32_t width = image1.Width();
    int32_t height = image1.Height();
    int32_t thresh = static_cast<int32_t>(threshold);

    std::vector<Run> runs;

    // Handle UInt8 images directly
    if (image1.Type() == PixelType::UInt8 && image2.Type() == PixelType::UInt8) {
        const uint8_t* ptr1 = static_cast<const uint8_t*>(image1.Data());
        const uint8_t* ptr2 = static_cast<const uint8_t*>(image2.Data());
        int32_t stride1 = image1.Stride();
        int32_t stride2 = image2.Stride();

        for (int32_t y = 0; y < height; ++y) {
            int32_t runStart = -1;

            for (int32_t x = 0; x < width; ++x) {
                int32_t diff = std::abs(static_cast<int32_t>(ptr1[y * stride1 + x]) -
                                        static_cast<int32_t>(ptr2[y * stride2 + x]));
                bool isDefect = (diff > thresh);

                if (isDefect) {
                    if (runStart < 0) runStart = x;
                } else {
                    if (runStart >= 0) {
                        runs.push_back({y, runStart, x});
                        runStart = -1;
                    }
                }
            }
            if (runStart >= 0) {
                runs.push_back({y, runStart, width});
            }
        }
    } else {
        throw Exception("AbsDiffThreshold only supports UInt8 images currently");
    }

    return QRegion(std::move(runs));
}

void AbsDiffImage(
    const QImage& image1,
    const QImage& image2,
    QImage& diffImage
) {
    if (image1.Width() != image2.Width() || image1.Height() != image2.Height()) {
        throw Exception("Image sizes don't match");
    }

    int32_t width = image1.Width();
    int32_t height = image1.Height();

    diffImage = QImage(width, height, PixelType::UInt8);

    if (image1.Type() == PixelType::UInt8 && image2.Type() == PixelType::UInt8) {
        const uint8_t* ptr1 = static_cast<const uint8_t*>(image1.Data());
        const uint8_t* ptr2 = static_cast<const uint8_t*>(image2.Data());
        uint8_t* dst = static_cast<uint8_t*>(diffImage.Data());
        int32_t stride1 = image1.Stride();
        int32_t stride2 = image2.Stride();
        int32_t dstStride = diffImage.Stride();

        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                int32_t diff = std::abs(static_cast<int32_t>(ptr1[y * stride1 + x]) -
                                        static_cast<int32_t>(ptr2[y * stride2 + x]));
                dst[y * dstStride + x] = static_cast<uint8_t>(std::min(diff, 255));
            }
        }
    } else {
        throw Exception("AbsDiffImage only supports UInt8 images currently");
    }
}

} // namespace Qi::Vision::Defect
