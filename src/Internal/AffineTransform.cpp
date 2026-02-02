#include <QiVision/Internal/AffineTransform.h>
#include <QiVision/Internal/Solver.h>
#include <QiVision/Internal/Matrix.h>
#include <QiVision/Platform/Random.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

constexpr double EPSILON = 1e-10;
constexpr double PI = 3.14159265358979323846;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Clamp coordinate to valid range
inline int32_t ClampCoord(int32_t val, int32_t maxVal) {
    return std::max(0, std::min(val, maxVal - 1));
}

// Get pixel with border handling
template<typename T>
T GetPixelWithBorder(const T* data, int32_t width, int32_t height,
                     int32_t x, int32_t y, BorderMode mode, T borderValue) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        return data[y * width + x];
    }

    switch (mode) {
        case BorderMode::Constant:
            return borderValue;
        case BorderMode::Replicate:
            x = ClampCoord(x, width);
            y = ClampCoord(y, height);
            return data[y * width + x];
        case BorderMode::Reflect:
            if (x < 0) x = -x;
            if (x >= width) x = 2 * width - x - 2;
            if (y < 0) y = -y;
            if (y >= height) y = 2 * height - y - 2;
            x = ClampCoord(x, width);
            y = ClampCoord(y, height);
            return data[y * width + x];
        case BorderMode::Reflect101:
            if (x < 0) x = -x - 1;
            if (x >= width) x = 2 * width - x - 1;
            if (y < 0) y = -y - 1;
            if (y >= height) y = 2 * height - y - 1;
            x = ClampCoord(x, width);
            y = ClampCoord(y, height);
            return data[y * width + x];
        case BorderMode::Wrap:
            x = ((x % width) + width) % width;
            y = ((y % height) + height) % height;
            return data[y * width + x];
        default:
            return borderValue;
    }
}

// Bilinear interpolation helper
template<typename T>
double BilinearSample(const T* data, int32_t width, int32_t height,
                      double x, double y, BorderMode mode, double borderValue) {
    int32_t x0 = static_cast<int32_t>(std::floor(x));
    int32_t y0 = static_cast<int32_t>(std::floor(y));
    int32_t x1 = x0 + 1;
    int32_t y1 = y0 + 1;

    double fx = x - x0;
    double fy = y - y0;

    T bv = static_cast<T>(borderValue);
    double v00 = GetPixelWithBorder(data, width, height, x0, y0, mode, bv);
    double v10 = GetPixelWithBorder(data, width, height, x1, y0, mode, bv);
    double v01 = GetPixelWithBorder(data, width, height, x0, y1, mode, bv);
    double v11 = GetPixelWithBorder(data, width, height, x1, y1, mode, bv);

    return v00 * (1 - fx) * (1 - fy) +
           v10 * fx * (1 - fy) +
           v01 * (1 - fx) * fy +
           v11 * fx * fy;
}

// Bicubic kernel (Catmull-Rom, a = -0.5)
inline double CubicKernel(double x) {
    x = std::abs(x);
    if (x < 1.0) {
        return (1.5 * x - 2.5) * x * x + 1.0;
    } else if (x < 2.0) {
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    }
    return 0.0;
}

// Bicubic interpolation helper
template<typename T>
double BicubicSample(const T* data, int32_t width, int32_t height,
                     double x, double y, BorderMode mode, double borderValue) {
    int32_t x0 = static_cast<int32_t>(std::floor(x)) - 1;
    int32_t y0 = static_cast<int32_t>(std::floor(y)) - 1;

    double fx = x - std::floor(x);
    double fy = y - std::floor(y);

    T bv = static_cast<T>(borderValue);
    double result = 0.0;

    for (int j = 0; j < 4; ++j) {
        double wy = CubicKernel(fy - (j - 1));
        for (int i = 0; i < 4; ++i) {
            double wx = CubicKernel(fx - (i - 1));
            double v = GetPixelWithBorder(data, width, height, x0 + i, y0 + j, mode, bv);
            result += v * wx * wy;
        }
    }

    return result;
}

// Nearest neighbor helper
template<typename T>
double NearestSample(const T* data, int32_t width, int32_t height,
                     double x, double y, BorderMode mode, double borderValue) {
    int32_t xi = static_cast<int32_t>(std::round(x));
    int32_t yi = static_cast<int32_t>(std::round(y));
    T bv = static_cast<T>(borderValue);
    return GetPixelWithBorder(data, width, height, xi, yi, mode, bv);
}

// Generic warp function
template<typename T>
void WarpImageGeneric(const T* src, int32_t srcWidth, int32_t srcHeight,
                      T* dst, int32_t dstWidth, int32_t dstHeight,
                      const QMatrix& invMatrix,
                      InterpolationMethod method,
                      BorderMode borderMode, double borderValue) {
    for (int32_t y = 0; y < dstHeight; ++y) {
        for (int32_t x = 0; x < dstWidth; ++x) {
            // Map destination to source
            Point2d srcPt = invMatrix.Transform(static_cast<double>(x),
                                                 static_cast<double>(y));

            double value;
            switch (method) {
                case InterpolationMethod::Nearest:
                    value = NearestSample(src, srcWidth, srcHeight,
                                         srcPt.x, srcPt.y, borderMode, borderValue);
                    break;
                case InterpolationMethod::Bicubic:
                    value = BicubicSample(src, srcWidth, srcHeight,
                                         srcPt.x, srcPt.y, borderMode, borderValue);
                    break;
                case InterpolationMethod::Bilinear:
                default:
                    value = BilinearSample(src, srcWidth, srcHeight,
                                          srcPt.x, srcPt.y, borderMode, borderValue);
                    break;
            }

            // Clamp and store
            if constexpr (std::is_integral_v<T>) {
                value = std::round(value);
                value = std::max(static_cast<double>(std::numeric_limits<T>::min()),
                                std::min(static_cast<double>(std::numeric_limits<T>::max()), value));
            }
            dst[y * dstWidth + x] = static_cast<T>(value);
        }
    }
}

// RANSAC helper to count inliers
int32_t CountInliers(const std::vector<Point2d>& srcPoints,
                     const std::vector<Point2d>& dstPoints,
                     const QMatrix& matrix,
                     double threshold,
                     std::vector<bool>* mask = nullptr) {
    int32_t count = 0;
    double thresholdSq = threshold * threshold;

    if (mask) {
        mask->resize(srcPoints.size());
    }

    for (size_t i = 0; i < srcPoints.size(); ++i) {
        Point2d transformed = matrix.Transform(srcPoints[i]);
        double dx = transformed.x - dstPoints[i].x;
        double dy = transformed.y - dstPoints[i].y;
        double distSq = dx * dx + dy * dy;

        bool isInlier = distSq <= thresholdSq;
        if (isInlier) count++;
        if (mask) (*mask)[i] = isInlier;
    }

    return count;
}

// Calculate required RANSAC iterations
int32_t CalculateRANSACIterations(int32_t sampleSize, double confidence,
                                   double inlierRatio, int32_t maxIterations) {
    if (inlierRatio >= 1.0) return 1;
    if (inlierRatio <= 0.0) return maxIterations;

    double num = std::log(1.0 - confidence);
    double denom = std::log(1.0 - std::pow(inlierRatio, sampleSize));

    if (std::abs(denom) < EPSILON) return maxIterations;

    int32_t iterations = static_cast<int32_t>(std::ceil(num / denom));
    return std::min(iterations, maxIterations);
}

} // anonymous namespace

// =============================================================================
// Image Warping
// =============================================================================

void ComputeAffineOutputSize(int32_t srcWidth, int32_t srcHeight,
                             const QMatrix& matrix,
                             int32_t& dstWidth, int32_t& dstHeight,
                             double& offsetX, double& offsetY) {
    // Transform corners
    Point2d corners[4] = {
        matrix.Transform(0, 0),
        matrix.Transform(srcWidth - 1, 0),
        matrix.Transform(srcWidth - 1, srcHeight - 1),
        matrix.Transform(0, srcHeight - 1)
    };

    // Find bounding box
    double minX = corners[0].x, maxX = corners[0].x;
    double minY = corners[0].y, maxY = corners[0].y;

    for (int i = 1; i < 4; ++i) {
        minX = std::min(minX, corners[i].x);
        maxX = std::max(maxX, corners[i].x);
        minY = std::min(minY, corners[i].y);
        maxY = std::max(maxY, corners[i].y);
    }

    dstWidth = static_cast<int32_t>(std::ceil(maxX - minX)) + 1;
    dstHeight = static_cast<int32_t>(std::ceil(maxY - minY)) + 1;
    offsetX = -minX;
    offsetY = -minY;
}

QImage WarpAffine(const QImage& src,
                  const QMatrix& matrix,
                  int32_t dstWidth,
                  int32_t dstHeight,
                  InterpolationMethod method,
                  BorderMode borderMode,
                  double borderValue) {
    if (src.Empty()) return QImage();

    // Calculate output size if not specified
    double offsetX = 0, offsetY = 0;
    if (dstWidth <= 0 || dstHeight <= 0) {
        ComputeAffineOutputSize(src.Width(), src.Height(), matrix,
                               dstWidth, dstHeight, offsetX, offsetY);
    }

    // Create adjusted matrix with offset
    QMatrix adjustedMatrix = QMatrix::Translation(offsetX, offsetY) * matrix;

    // Compute inverse matrix for backward mapping
    QMatrix invMatrix = adjustedMatrix.Inverse();
    if (!adjustedMatrix.IsInvertible()) {
        return QImage();
    }

    // Convert channel count to ChannelType
    ChannelType chType = src.GetChannelType();

    // Create output image
    QImage dst(dstWidth, dstHeight, src.Type(), chType);

    int32_t srcW = src.Width();
    int32_t srcH = src.Height();
    int32_t channels = src.Channels();

    // For simplicity, process entire image as single channel if grayscale
    // For multi-channel, we process channel-by-channel (planar assumed)
    // Note: QImage uses planar format internally

    // Process each channel
    for (int32_t c = 0; c < channels; ++c) {
        switch (src.Type()) {
            case PixelType::UInt8: {
                const uint8_t* srcBase = static_cast<const uint8_t*>(src.Data());
                uint8_t* dstBase = static_cast<uint8_t*>(dst.Data());
                const uint8_t* srcData = srcBase + c * srcW * srcH;
                uint8_t* dstData = dstBase + c * dstWidth * dstHeight;
                WarpImageGeneric(srcData, srcW, srcH, dstData, dstWidth, dstHeight,
                                invMatrix, method, borderMode, borderValue);
                break;
            }
            case PixelType::UInt16: {
                const uint16_t* srcBase = static_cast<const uint16_t*>(src.Data());
                uint16_t* dstBase = static_cast<uint16_t*>(dst.Data());
                const uint16_t* srcData = srcBase + c * srcW * srcH;
                uint16_t* dstData = dstBase + c * dstWidth * dstHeight;
                WarpImageGeneric(srcData, srcW, srcH, dstData, dstWidth, dstHeight,
                                invMatrix, method, borderMode, borderValue);
                break;
            }
            case PixelType::Int16: {
                const int16_t* srcBase = static_cast<const int16_t*>(src.Data());
                int16_t* dstBase = static_cast<int16_t*>(dst.Data());
                const int16_t* srcData = srcBase + c * srcW * srcH;
                int16_t* dstData = dstBase + c * dstWidth * dstHeight;
                WarpImageGeneric(srcData, srcW, srcH, dstData, dstWidth, dstHeight,
                                invMatrix, method, borderMode, borderValue);
                break;
            }
            case PixelType::Float32: {
                const float* srcBase = static_cast<const float*>(src.Data());
                float* dstBase = static_cast<float*>(dst.Data());
                const float* srcData = srcBase + c * srcW * srcH;
                float* dstData = dstBase + c * dstWidth * dstHeight;
                WarpImageGeneric(srcData, srcW, srcH, dstData, dstWidth, dstHeight,
                                invMatrix, method, borderMode, borderValue);
                break;
            }
        }
    }

    return dst;
}

QImage RotateImage(const QImage& src,
                   double angle,
                   bool resize,
                   InterpolationMethod method,
                   BorderMode borderMode,
                   double borderValue) {
    if (src.Empty()) return QImage();

    double cx = (src.Width() - 1) / 2.0;
    double cy = (src.Height() - 1) / 2.0;

    return RotateImage(src, angle, cx, cy, resize, method, borderMode, borderValue);
}

QImage RotateImage(const QImage& src,
                   double angle,
                   double centerX,
                   double centerY,
                   bool resize,
                   InterpolationMethod method,
                   BorderMode borderMode,
                   double borderValue) {
    if (src.Empty()) return QImage();

    // Create rotation matrix
    QMatrix matrix = QMatrix::Rotation(angle, centerX, centerY);

    int32_t dstWidth = 0, dstHeight = 0;
    if (resize) {
        // Auto-calculate to fit entire rotated image
        double offsetX, offsetY;
        ComputeAffineOutputSize(src.Width(), src.Height(), matrix,
                               dstWidth, dstHeight, offsetX, offsetY);
    } else {
        // Keep same size
        dstWidth = src.Width();
        dstHeight = src.Height();
    }

    return WarpAffine(src, matrix, dstWidth, dstHeight, method, borderMode, borderValue);
}

QImage ScaleImage(const QImage& src,
                  int32_t dstWidth,
                  int32_t dstHeight,
                  InterpolationMethod method) {
    if (src.Empty() || dstWidth <= 0 || dstHeight <= 0) return QImage();

    double scaleX = static_cast<double>(dstWidth) / src.Width();
    double scaleY = static_cast<double>(dstHeight) / src.Height();

    QMatrix matrix = QMatrix::Scaling(scaleX, scaleY);

    return WarpAffine(src, matrix, dstWidth, dstHeight, method,
                     BorderMode::Replicate, 0.0);
}

QImage ScaleImageFactor(const QImage& src,
                        double scaleX,
                        double scaleY,
                        InterpolationMethod method) {
    if (src.Empty() || scaleX <= 0 || scaleY <= 0) return QImage();

    int32_t dstWidth = static_cast<int32_t>(std::round(src.Width() * scaleX));
    int32_t dstHeight = static_cast<int32_t>(std::round(src.Height() * scaleY));

    return ScaleImage(src, dstWidth, dstHeight, method);
}

QImage CropRotatedRect(const QImage& src,
                       const RotatedRect2d& rect,
                       InterpolationMethod method) {
    if (src.Empty()) return QImage();

    // Create matrix to map rotated rect to axis-aligned output
    QMatrix matrix = RotatedRectToAxisAligned(rect);
    QMatrix invMatrix = matrix.Inverse();

    int32_t outWidth = static_cast<int32_t>(std::ceil(rect.width));
    int32_t outHeight = static_cast<int32_t>(std::ceil(rect.height));

    // For crop, we need inverse: from output coords to source coords
    return WarpAffine(src, invMatrix.Inverse(), outWidth, outHeight, method,
                     BorderMode::Constant, 0.0);
}

// =============================================================================
// Transform Estimation
// =============================================================================

std::optional<QMatrix> EstimateAffine(const std::vector<Point2d>& srcPoints,
                                       const std::vector<Point2d>& dstPoints) {
    if (srcPoints.size() < 3 || srcPoints.size() != dstPoints.size()) {
        return std::nullopt;
    }

    size_t n = srcPoints.size();

    // Build overdetermined system: A * x = b
    // For each point: [x y 1 0 0 0] * [a b tx c d ty]^T = x'
    //                 [0 0 0 x y 1]                      = y'

    // Using normal equations: A^T * A * x = A^T * b
    // A is 2n x 6, we need to solve 6x6 system

    // Accumulate A^T * A (6x6) and A^T * b (6x1)
    double AtA[36] = {0};
    double Atb[6] = {0};

    for (size_t i = 0; i < n; ++i) {
        double x = srcPoints[i].x;
        double y = srcPoints[i].y;
        double xp = dstPoints[i].x;
        double yp = dstPoints[i].y;

        // Row 1: [x y 1 0 0 0]
        double r1[6] = {x, y, 1, 0, 0, 0};
        // Row 2: [0 0 0 x y 1]
        double r2[6] = {0, 0, 0, x, y, 1};

        // Accumulate A^T * A
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                AtA[j * 6 + k] += r1[j] * r1[k] + r2[j] * r2[k];
            }
            // Accumulate A^T * b
            Atb[j] += r1[j] * xp + r2[j] * yp;
        }
    }

    // Convert to MatX and VecX for solving
    MatX AtAMat(6, 6);
    VecX AtbVec(6);
    for (int i = 0; i < 6; ++i) {
        AtbVec[i] = Atb[i];
        for (int j = 0; j < 6; ++j) {
            AtAMat(i, j) = AtA[i * 6 + j];
        }
    }

    // Solve using LU decomposition
    VecX solution = SolveLU(AtAMat, AtbVec);
    if (solution.Size() != 6) {
        return std::nullopt;
    }

    return QMatrix(solution[0], solution[1], solution[2],
                   solution[3], solution[4], solution[5]);
}

std::optional<QMatrix> EstimateAffineRANSAC(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    double threshold,
    double confidence,
    int32_t maxIterations,
    std::vector<bool>* inlierMask) {

    if (srcPoints.size() < 3 || srcPoints.size() != dstPoints.size()) {
        return std::nullopt;
    }

    size_t n = srcPoints.size();
    auto& rng = Platform::Random::Instance();

    QMatrix bestMatrix;
    int32_t bestInliers = 0;
    std::vector<bool> bestMask;

    int32_t iterations = maxIterations;
    const int32_t sampleSize = 3;

    for (int32_t iter = 0; iter < iterations; ++iter) {
        // Random sample of 3 points
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) indices[i] = i;

        // Fisher-Yates shuffle for first 3
        for (int i = 0; i < sampleSize; ++i) {
            size_t j = i + rng.Index(n - i);
            std::swap(indices[i], indices[j]);
        }

        // Extract sample points
        std::vector<Point2d> sampleSrc(sampleSize), sampleDst(sampleSize);
        for (int i = 0; i < sampleSize; ++i) {
            sampleSrc[i] = srcPoints[indices[i]];
            sampleDst[i] = dstPoints[indices[i]];
        }

        // Estimate from minimal sample
        auto matrix = EstimateAffine(sampleSrc, sampleDst);
        if (!matrix) continue;

        // Count inliers
        std::vector<bool> mask;
        int32_t numInliers = CountInliers(srcPoints, dstPoints, *matrix, threshold, &mask);

        if (numInliers > bestInliers) {
            bestInliers = numInliers;
            bestMatrix = *matrix;
            bestMask = std::move(mask);

            // Update iteration count based on inlier ratio
            double inlierRatio = static_cast<double>(numInliers) / n;
            iterations = CalculateRANSACIterations(sampleSize, confidence,
                                                    inlierRatio, maxIterations);
        }
    }

    if (bestInliers < 3) {
        return std::nullopt;
    }

    // Refine with all inliers
    std::vector<Point2d> inlierSrc, inlierDst;
    for (size_t i = 0; i < n; ++i) {
        if (bestMask[i]) {
            inlierSrc.push_back(srcPoints[i]);
            inlierDst.push_back(dstPoints[i]);
        }
    }

    auto refinedMatrix = EstimateAffine(inlierSrc, inlierDst);
    if (refinedMatrix) {
        bestMatrix = *refinedMatrix;
        // Update mask with refined matrix
        CountInliers(srcPoints, dstPoints, bestMatrix, threshold, &bestMask);
    }

    if (inlierMask) {
        *inlierMask = std::move(bestMask);
    }

    return bestMatrix;
}

std::optional<QMatrix> EstimateRigid(const std::vector<Point2d>& srcPoints,
                                      const std::vector<Point2d>& dstPoints) {
    if (srcPoints.size() < 2 || srcPoints.size() != dstPoints.size()) {
        return std::nullopt;
    }

    size_t n = srcPoints.size();

    // Compute centroids
    Point2d srcCentroid{0, 0}, dstCentroid{0, 0};
    for (size_t i = 0; i < n; ++i) {
        srcCentroid.x += srcPoints[i].x;
        srcCentroid.y += srcPoints[i].y;
        dstCentroid.x += dstPoints[i].x;
        dstCentroid.y += dstPoints[i].y;
    }
    srcCentroid.x /= n;
    srcCentroid.y /= n;
    dstCentroid.x /= n;
    dstCentroid.y /= n;

    // Compute rotation using Procrustes
    double sumXX = 0, sumXY = 0, sumYX = 0, sumYY = 0;
    for (size_t i = 0; i < n; ++i) {
        double sx = srcPoints[i].x - srcCentroid.x;
        double sy = srcPoints[i].y - srcCentroid.y;
        double dx = dstPoints[i].x - dstCentroid.x;
        double dy = dstPoints[i].y - dstCentroid.y;

        sumXX += sx * dx;
        sumXY += sx * dy;
        sumYX += sy * dx;
        sumYY += sy * dy;
    }

    // Angle from cross-covariance
    double angle = std::atan2(sumXY - sumYX, sumXX + sumYY);

    double cosA = std::cos(angle);
    double sinA = std::sin(angle);

    // Compute translation
    double tx = dstCentroid.x - (cosA * srcCentroid.x - sinA * srcCentroid.y);
    double ty = dstCentroid.y - (sinA * srcCentroid.x + cosA * srcCentroid.y);

    return QMatrix(cosA, -sinA, tx,
                   sinA, cosA, ty);
}

std::optional<QMatrix> EstimateRigidRANSAC(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    double threshold,
    double confidence,
    int32_t maxIterations,
    std::vector<bool>* inlierMask) {

    if (srcPoints.size() < 2 || srcPoints.size() != dstPoints.size()) {
        return std::nullopt;
    }

    size_t n = srcPoints.size();
    auto& rng = Platform::Random::Instance();

    QMatrix bestMatrix;
    int32_t bestInliers = 0;
    std::vector<bool> bestMask;

    int32_t iterations = maxIterations;
    const int32_t sampleSize = 2;

    for (int32_t iter = 0; iter < iterations; ++iter) {
        // Random sample
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) indices[i] = i;

        for (int i = 0; i < sampleSize; ++i) {
            size_t j = i + rng.Index(n - i);
            std::swap(indices[i], indices[j]);
        }

        std::vector<Point2d> sampleSrc(sampleSize), sampleDst(sampleSize);
        for (int i = 0; i < sampleSize; ++i) {
            sampleSrc[i] = srcPoints[indices[i]];
            sampleDst[i] = dstPoints[indices[i]];
        }

        auto matrix = EstimateRigid(sampleSrc, sampleDst);
        if (!matrix) continue;

        std::vector<bool> mask;
        int32_t numInliers = CountInliers(srcPoints, dstPoints, *matrix, threshold, &mask);

        if (numInliers > bestInliers) {
            bestInliers = numInliers;
            bestMatrix = *matrix;
            bestMask = std::move(mask);

            double inlierRatio = static_cast<double>(numInliers) / n;
            iterations = CalculateRANSACIterations(sampleSize, confidence,
                                                    inlierRatio, maxIterations);
        }
    }

    if (bestInliers < 2) {
        return std::nullopt;
    }

    // Refine with all inliers
    std::vector<Point2d> inlierSrc, inlierDst;
    for (size_t i = 0; i < n; ++i) {
        if (bestMask[i]) {
            inlierSrc.push_back(srcPoints[i]);
            inlierDst.push_back(dstPoints[i]);
        }
    }

    auto refinedMatrix = EstimateRigid(inlierSrc, inlierDst);
    if (refinedMatrix) {
        bestMatrix = *refinedMatrix;
        CountInliers(srcPoints, dstPoints, bestMatrix, threshold, &bestMask);
    }

    if (inlierMask) {
        *inlierMask = std::move(bestMask);
    }

    return bestMatrix;
}

std::optional<QMatrix> EstimateSimilarity(const std::vector<Point2d>& srcPoints,
                                           const std::vector<Point2d>& dstPoints) {
    if (srcPoints.size() < 2 || srcPoints.size() != dstPoints.size()) {
        return std::nullopt;
    }

    size_t n = srcPoints.size();

    // Compute centroids
    Point2d srcCentroid{0, 0}, dstCentroid{0, 0};
    for (size_t i = 0; i < n; ++i) {
        srcCentroid.x += srcPoints[i].x;
        srcCentroid.y += srcPoints[i].y;
        dstCentroid.x += dstPoints[i].x;
        dstCentroid.y += dstPoints[i].y;
    }
    srcCentroid.x /= n;
    srcCentroid.y /= n;
    dstCentroid.x /= n;
    dstCentroid.y /= n;

    // Compute cross-covariance and source variance
    double sumXX = 0, sumXY = 0, sumYX = 0, sumYY = 0;
    double srcVar = 0;
    for (size_t i = 0; i < n; ++i) {
        double sx = srcPoints[i].x - srcCentroid.x;
        double sy = srcPoints[i].y - srcCentroid.y;
        double dx = dstPoints[i].x - dstCentroid.x;
        double dy = dstPoints[i].y - dstCentroid.y;

        sumXX += sx * dx;
        sumXY += sx * dy;
        sumYX += sy * dx;
        sumYY += sy * dy;
        srcVar += sx * sx + sy * sy;
    }

    if (srcVar < EPSILON) {
        return std::nullopt;
    }

    // Angle
    double angle = std::atan2(sumXY - sumYX, sumXX + sumYY);

    // Scale
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    double scale = (sumXX * cosA + sumXY * sinA + sumYX * (-sinA) + sumYY * cosA) / srcVar;

    if (scale < EPSILON) {
        return std::nullopt;
    }

    double a = scale * cosA;
    double b = -scale * sinA;

    // Translation
    double tx = dstCentroid.x - (a * srcCentroid.x + b * srcCentroid.y);
    double ty = dstCentroid.y - (-b * srcCentroid.x + a * srcCentroid.y);

    return QMatrix(a, b, tx,
                   -b, a, ty);
}

std::optional<QMatrix> EstimateSimilarityRANSAC(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    double threshold,
    double confidence,
    int32_t maxIterations,
    std::vector<bool>* inlierMask) {

    if (srcPoints.size() < 2 || srcPoints.size() != dstPoints.size()) {
        return std::nullopt;
    }

    size_t n = srcPoints.size();
    auto& rng = Platform::Random::Instance();

    QMatrix bestMatrix;
    int32_t bestInliers = 0;
    std::vector<bool> bestMask;

    int32_t iterations = maxIterations;
    const int32_t sampleSize = 2;

    for (int32_t iter = 0; iter < iterations; ++iter) {
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) indices[i] = i;

        for (int i = 0; i < sampleSize; ++i) {
            size_t j = i + rng.Index(n - i);
            std::swap(indices[i], indices[j]);
        }

        std::vector<Point2d> sampleSrc(sampleSize), sampleDst(sampleSize);
        for (int i = 0; i < sampleSize; ++i) {
            sampleSrc[i] = srcPoints[indices[i]];
            sampleDst[i] = dstPoints[indices[i]];
        }

        auto matrix = EstimateSimilarity(sampleSrc, sampleDst);
        if (!matrix) continue;

        std::vector<bool> mask;
        int32_t numInliers = CountInliers(srcPoints, dstPoints, *matrix, threshold, &mask);

        if (numInliers > bestInliers) {
            bestInliers = numInliers;
            bestMatrix = *matrix;
            bestMask = std::move(mask);

            double inlierRatio = static_cast<double>(numInliers) / n;
            iterations = CalculateRANSACIterations(sampleSize, confidence,
                                                    inlierRatio, maxIterations);
        }
    }

    if (bestInliers < 2) {
        return std::nullopt;
    }

    std::vector<Point2d> inlierSrc, inlierDst;
    for (size_t i = 0; i < n; ++i) {
        if (bestMask[i]) {
            inlierSrc.push_back(srcPoints[i]);
            inlierDst.push_back(dstPoints[i]);
        }
    }

    auto refinedMatrix = EstimateSimilarity(inlierSrc, inlierDst);
    if (refinedMatrix) {
        bestMatrix = *refinedMatrix;
        CountInliers(srcPoints, dstPoints, bestMatrix, threshold, &bestMask);
    }

    if (inlierMask) {
        *inlierMask = std::move(bestMask);
    }

    return bestMatrix;
}

double ComputeTransformError(const std::vector<Point2d>& srcPoints,
                              const std::vector<Point2d>& dstPoints,
                              const QMatrix& matrix) {
    if (srcPoints.empty() || srcPoints.size() != dstPoints.size()) {
        return std::numeric_limits<double>::max();
    }

    double sumSqError = 0;
    for (size_t i = 0; i < srcPoints.size(); ++i) {
        Point2d transformed = matrix.Transform(srcPoints[i]);
        double dx = transformed.x - dstPoints[i].x;
        double dy = transformed.y - dstPoints[i].y;
        sumSqError += dx * dx + dy * dy;
    }

    return std::sqrt(sumSqError / srcPoints.size());
}

std::vector<double> ComputePointErrors(const std::vector<Point2d>& srcPoints,
                                        const std::vector<Point2d>& dstPoints,
                                        const QMatrix& matrix) {
    std::vector<double> errors;
    if (srcPoints.size() != dstPoints.size()) {
        return errors;
    }

    errors.reserve(srcPoints.size());
    for (size_t i = 0; i < srcPoints.size(); ++i) {
        Point2d transformed = matrix.Transform(srcPoints[i]);
        double dx = transformed.x - dstPoints[i].x;
        double dy = transformed.y - dstPoints[i].y;
        errors.push_back(std::sqrt(dx * dx + dy * dy));
    }

    return errors;
}

// =============================================================================
// Region Transformation
// =============================================================================

QRegion AffineTransformRegion(const QRegion& region, const QMatrix& matrix) {
    if (region.Empty()) return QRegion();

    // Get bounding box and expand
    Rect2i bbox = region.BoundingBox();

    // Transform corners to get output bounding box
    Point2d corners[4] = {
        matrix.Transform(bbox.x, bbox.y),
        matrix.Transform(bbox.x + bbox.width, bbox.y),
        matrix.Transform(bbox.x + bbox.width, bbox.y + bbox.height),
        matrix.Transform(bbox.x, bbox.y + bbox.height)
    };

    double minX = corners[0].x, maxX = corners[0].x;
    double minY = corners[0].y, maxY = corners[0].y;
    for (int i = 1; i < 4; ++i) {
        minX = std::min(minX, corners[i].x);
        maxX = std::max(maxX, corners[i].x);
        minY = std::min(minY, corners[i].y);
        maxY = std::max(maxY, corners[i].y);
    }

    int32_t outMinX = static_cast<int32_t>(std::floor(minX));
    int32_t outMaxX = static_cast<int32_t>(std::ceil(maxX));
    int32_t outMinY = static_cast<int32_t>(std::floor(minY));
    int32_t outMaxY = static_cast<int32_t>(std::ceil(maxY));

    // Inverse matrix for backward mapping
    QMatrix invMatrix = matrix.Inverse();
    if (!matrix.IsInvertible()) {
        return QRegion();
    }

    // Build new runs
    std::vector<QRegion::Run> newRuns;

    for (int32_t y = outMinY; y <= outMaxY; ++y) {
        int32_t runStart = -1;

        for (int32_t x = outMinX; x <= outMaxX; ++x) {
            // Map back to source
            Point2d srcPt = invMatrix.Transform(static_cast<double>(x),
                                                 static_cast<double>(y));

            int32_t srcX = static_cast<int32_t>(std::round(srcPt.x));
            int32_t srcY = static_cast<int32_t>(std::round(srcPt.y));

            bool inside = region.Contains(srcX, srcY);

            if (inside && runStart < 0) {
                runStart = x;
            } else if (!inside && runStart >= 0) {
                newRuns.push_back({y, runStart, x});
                runStart = -1;
            }
        }

        // Close run at end of row
        if (runStart >= 0) {
            newRuns.push_back({y, runStart, outMaxX + 1});
        }
    }

    return QRegion(newRuns);
}

std::vector<QRegion> AffineTransformRegions(const std::vector<QRegion>& regions,
                                             const QMatrix& matrix) {
    std::vector<QRegion> result;
    result.reserve(regions.size());

    for (const auto& region : regions) {
        result.push_back(AffineTransformRegion(region, matrix));
    }

    return result;
}

// =============================================================================
// Contour Transformation
// =============================================================================

QContour AffineTransformContour(const QContour& contour, const QMatrix& matrix) {
    return contour.Transform(matrix);
}

std::vector<QContour> AffineTransformContours(const std::vector<QContour>& contours,
                                               const QMatrix& matrix) {
    std::vector<QContour> result;
    result.reserve(contours.size());

    for (const auto& contour : contours) {
        result.push_back(contour.Transform(matrix));
    }

    return result;
}

// =============================================================================
// Transform Analysis
// =============================================================================

bool DecomposeAffine(const QMatrix& matrix,
                     double& tx, double& ty,
                     double& angle,
                     double& scaleX, double& scaleY,
                     double& shear) {
    // Extract translation
    tx = matrix.M02();
    ty = matrix.M12();

    // Linear part
    double a = matrix.M00();
    double b = matrix.M01();
    double c = matrix.M10();
    double d = matrix.M11();

    // Determinant
    double det = a * d - b * c;
    if (std::abs(det) < EPSILON) {
        return false;
    }

    // QR decomposition of linear part
    // [a b] = [cos -sin] * [sx  shear*sy]
    // [c d]   [sin  cos]   [0   sy      ]

    scaleX = std::sqrt(a * a + c * c);
    if (scaleX < EPSILON) {
        return false;
    }

    angle = std::atan2(c, a);
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);

    // Rotate back to get scale/shear
    double r00 = cosA * a + sinA * c;
    double r01 = cosA * b + sinA * d;
    double r11 = -sinA * b + cosA * d;

    scaleX = r00;
    scaleY = r11;
    shear = r01 / r11;

    // Check sign of determinant (flip)
    if (det < 0) {
        scaleY = -scaleY;
    }

    return true;
}

bool IsRigidTransform(const QMatrix& matrix, double tolerance) {
    double a = matrix.M00();
    double b = matrix.M01();
    double c = matrix.M10();
    double d = matrix.M11();

    // For rigid: a = cos(θ), b = -sin(θ), c = sin(θ), d = cos(θ)
    // So: a = d, b = -c, and a² + c² = 1

    if (std::abs(a - d) > tolerance) return false;
    if (std::abs(b + c) > tolerance) return false;
    if (std::abs(a * a + c * c - 1.0) > tolerance) return false;

    return true;
}

bool IsSimilarityTransform(const QMatrix& matrix, double tolerance) {
    double a = matrix.M00();
    double b = matrix.M01();
    double c = matrix.M10();
    double d = matrix.M11();

    // For similarity: a = s*cos(θ), b = -s*sin(θ), c = s*sin(θ), d = s*cos(θ)
    // So: a = d, b = -c

    if (std::abs(a - d) > tolerance) return false;
    if (std::abs(b + c) > tolerance) return false;

    return true;
}

QMatrix InterpolateTransform(const QMatrix& m1, const QMatrix& m2, double t) {
    // Simple linear interpolation of matrix elements
    // For better results with rotations, use decomposition-based interpolation

    double e1[6], e2[6];
    m1.GetElements(e1);
    m2.GetElements(e2);

    double result[6];
    for (int i = 0; i < 6; ++i) {
        result[i] = e1[i] * (1.0 - t) + e2[i] * t;
    }

    return QMatrix(result[0], result[1], result[2],
                   result[3], result[4], result[5]);
}

// =============================================================================
// Utility Functions
// =============================================================================

QMatrix RectToRectTransform(const Rect2d& srcRect, const Rect2d& dstRect) {
    double scaleX = dstRect.width / srcRect.width;
    double scaleY = dstRect.height / srcRect.height;

    double tx = dstRect.x - srcRect.x * scaleX;
    double ty = dstRect.y - srcRect.y * scaleY;

    return QMatrix(scaleX, 0, tx,
                   0, scaleY, ty);
}

QMatrix RotatedRectToAxisAligned(const RotatedRect2d& rotRect) {
    // Transform that maps rotRect to [0, width] x [0, height]
    // First translate center to origin, then rotate, then translate to output center

    double cx = rotRect.center.x;
    double cy = rotRect.center.y;
    double angle = -rotRect.angle;  // Negative to undo rotation

    double cosA = std::cos(angle);
    double sinA = std::sin(angle);

    // New center after rotation is at origin, then shift to output center
    double outCx = rotRect.width / 2.0;
    double outCy = rotRect.height / 2.0;

    // Combined matrix: T(outCx, outCy) * R(-angle) * T(-cx, -cy)
    double tx = outCx - (cosA * cx - sinA * cy);
    double ty = outCy - (sinA * cx + cosA * cy);

    return QMatrix(cosA, -sinA, tx,
                   sinA, cosA, ty);
}

Rect2d TransformBoundingBox(const Rect2d& bbox, const QMatrix& matrix) {
    Point2d corners[4] = {
        matrix.Transform(bbox.x, bbox.y),
        matrix.Transform(bbox.x + bbox.width, bbox.y),
        matrix.Transform(bbox.x + bbox.width, bbox.y + bbox.height),
        matrix.Transform(bbox.x, bbox.y + bbox.height)
    };

    double minX = corners[0].x, maxX = corners[0].x;
    double minY = corners[0].y, maxY = corners[0].y;

    for (int i = 1; i < 4; ++i) {
        minX = std::min(minX, corners[i].x);
        maxX = std::max(maxX, corners[i].x);
        minY = std::min(minY, corners[i].y);
        maxY = std::max(maxY, corners[i].y);
    }

    return Rect2d(minX, minY, maxX - minX, maxY - minY);
}

Rect2d TransformPointsBoundingBox(const std::vector<Point2d>& points,
                                   const QMatrix& matrix) {
    if (points.empty()) {
        return Rect2d(0, 0, 0, 0);
    }

    Point2d first = matrix.Transform(points[0]);
    double minX = first.x, maxX = first.x;
    double minY = first.y, maxY = first.y;

    for (size_t i = 1; i < points.size(); ++i) {
        Point2d p = matrix.Transform(points[i]);
        minX = std::min(minX, p.x);
        maxX = std::max(maxX, p.x);
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
    }

    return Rect2d(minX, minY, maxX - minX, maxY - minY);
}

} // namespace Qi::Vision::Internal
