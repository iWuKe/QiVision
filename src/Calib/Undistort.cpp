/**
 * @file Undistort.cpp
 * @brief Image undistortion implementation
 */

#include <QiVision/Calib/Undistort.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>

#include <cmath>
#include <algorithm>
#include <limits>

namespace Qi::Vision::Calib {

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

/**
 * @brief Get pixel value with interpolation
 */
template<typename T>
double GetInterpolatedPixel(
    const T* data,
    int32_t width,
    int32_t height,
    size_t stride,
    double x,
    double y,
    Internal::InterpolationMethod method,
    Internal::BorderMode borderMode,
    double borderValue)
{
    return Internal::InterpolateStrided<T>(
        data, width, height, stride, x, y, method, borderMode, borderValue);
}

} // anonymous namespace

// ============================================================================
// Undistort Implementation
// ============================================================================

void Undistort(
    const QImage& src,
    QImage& dst,
    const CameraModel& camera,
    Internal::InterpolationMethod method)
{
    if (!Validate::RequireImageValid(src, "Undistort")) {
        dst = QImage();
        return;
    }
    if (!camera.IsValid()) {
        throw InvalidArgumentException("Undistort: invalid camera model");
    }

    // Use original camera matrix and image size
    Undistort(src, dst, camera, camera.Intrinsics(),
              Size2i(src.Width(), src.Height()), method);
}

void Undistort(
    const QImage& src,
    QImage& dst,
    const CameraModel& camera,
    const CameraIntrinsics& newCameraMatrix,
    const Size2i& outputSize,
    Internal::InterpolationMethod method)
{
    if (!Validate::RequireImageValid(src, "Undistort")) {
        dst = QImage();
        return;
    }
    if (!camera.IsValid()) {
        throw InvalidArgumentException("Undistort: invalid camera model");
    }
    if (!std::isfinite(newCameraMatrix.fx) || !std::isfinite(newCameraMatrix.fy) ||
        newCameraMatrix.fx <= 0.0 || newCameraMatrix.fy <= 0.0) {
        throw InvalidArgumentException("Undistort: invalid newCameraMatrix");
    }

    if (outputSize.width < 0 || outputSize.height < 0) {
        throw InvalidArgumentException("Undistort: outputSize must be >= 0");
    }

    // Determine output size
    int32_t outW = (outputSize.width > 0) ? outputSize.width : src.Width();
    int32_t outH = (outputSize.height > 0) ? outputSize.height : src.Height();

    // Initialize undistortion map
    UndistortMap map = InitUndistortMap(camera, Size2i(outW, outH), &newCameraMatrix);

    // Apply remapping
    Remap(src, dst, map, method, Internal::BorderMode::Constant, 0.0);
}

UndistortMap InitUndistortMap(
    const CameraModel& camera,
    const Size2i& outputSize,
    const CameraIntrinsics* newCameraMatrix)
{
    if (!camera.IsValid()) {
        throw InvalidArgumentException("InitUndistortMap: invalid camera model");
    }
    UndistortMap map;
    map.width = outputSize.width;
    map.height = outputSize.height;

    if (map.width <= 0 || map.height <= 0) {
        throw InvalidArgumentException("InitUndistortMap: outputSize must be positive");
    }

    size_t totalPixels = static_cast<size_t>(map.width) * map.height;
    map.mapX.resize(totalPixels);
    map.mapY.resize(totalPixels);

    // Use provided new camera matrix or the original
    CameraIntrinsics newK = newCameraMatrix ? *newCameraMatrix : camera.Intrinsics();
    if (!std::isfinite(newK.fx) || !std::isfinite(newK.fy) || newK.fx <= 0.0 || newK.fy <= 0.0) {
        throw InvalidArgumentException("InitUndistortMap: invalid new camera intrinsics");
    }
    const CameraIntrinsics& origK = camera.Intrinsics();
    const DistortionCoeffs& dist = camera.Distortion();

    // Pre-compute inverse of new camera matrix parameters
    double fx_inv = 1.0 / newK.fx;
    double fy_inv = 1.0 / newK.fy;

    // For each pixel in destination image, compute source coordinates
    #pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < map.height; ++y) {
        size_t rowOffset = static_cast<size_t>(y) * map.width;

        for (int32_t x = 0; x < map.width; ++x) {
            // Convert destination pixel to normalized coordinates (undistorted)
            double x_norm = (x - newK.cx) * fx_inv;
            double y_norm = (y - newK.cy) * fy_inv;

            // Apply distortion to get source normalized coordinates
            double k1 = dist.k1;
            double k2 = dist.k2;
            double k3 = dist.k3;
            double p1 = dist.p1;
            double p2 = dist.p2;

            double r2 = x_norm * x_norm + y_norm * y_norm;
            double r4 = r2 * r2;
            double r6 = r4 * r2;

            // Radial distortion
            double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

            // Tangential distortion
            double dx_tang = 2.0 * p1 * x_norm * y_norm + p2 * (r2 + 2.0 * x_norm * x_norm);
            double dy_tang = p1 * (r2 + 2.0 * y_norm * y_norm) + 2.0 * p2 * x_norm * y_norm;

            // Distorted normalized coordinates
            double x_dist = x_norm * radial + dx_tang;
            double y_dist = y_norm * radial + dy_tang;

            // Convert to source pixel coordinates
            double src_x = origK.fx * x_dist + origK.cx;
            double src_y = origK.fy * y_dist + origK.cy;

            // Store in map
            map.mapX[rowOffset + x] = static_cast<float>(src_x);
            map.mapY[rowOffset + x] = static_cast<float>(src_y);
        }
    }

    return map;
}

void Remap(
    const QImage& src,
    QImage& dst,
    const UndistortMap& map,
    Internal::InterpolationMethod method,
    Internal::BorderMode borderMode,
    double borderValue)
{
    if (!Validate::RequireImageValid(src, "Remap")) {
        dst = QImage();
        return;
    }
    if (!std::isfinite(borderValue)) {
        throw InvalidArgumentException("Remap: invalid borderValue");
    }

    if (!map.IsValid()) {
        throw InvalidArgumentException("Remap: invalid undistort map");
    }

    // Ensure dst has correct size
    if (dst.Width() != map.width || dst.Height() != map.height ||
        dst.Type() != src.Type() || dst.GetChannelType() != src.GetChannelType()) {
        dst = QImage(map.width, map.height, src.Type(), src.GetChannelType());
    }

    int32_t srcW = src.Width();
    int32_t srcH = src.Height();
    int32_t dstW = map.width;
    int32_t dstH = map.height;

    // Get source data info
    size_t srcStride = src.Stride();
    size_t dstStride = dst.Stride();
    int numChannels = src.Channels();

    // Handle based on pixel type
    if (src.Type() == PixelType::UInt8) {
        const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
        uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

        // Element stride (for multi-channel)
        size_t srcElemStride = srcStride;  // bytes per row
        size_t dstElemStride = dstStride;

        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < dstH; ++y) {
            size_t mapRowOffset = static_cast<size_t>(y) * dstW;
            uint8_t* dstRow = dstData + y * dstElemStride;

            for (int32_t x = 0; x < dstW; ++x) {
                float srcX = map.mapX[mapRowOffset + x];
                float srcY = map.mapY[mapRowOffset + x];

                for (int c = 0; c < numChannels; ++c) {
                    // For multi-channel, compute per-channel
                    // Assuming interleaved format (RGBRGBRGB...)
                    double val;
                    if (numChannels == 1) {
                        val = Internal::InterpolateStrided<uint8_t>(
                            srcData, srcW, srcH, srcElemStride,
                            srcX, srcY, method, borderMode, borderValue);
                    } else {
                        // For multi-channel images, interpolate each channel
                        // Stride in elements = srcStride (bytes per row)
                        // Element step = numChannels for interleaved
                        // This is simplified - assumes packed RGB
                        val = Internal::InterpolateBilinear<uint8_t>(
                            srcData + c, srcW * numChannels, srcH,
                            srcX * numChannels + c, srcY, borderMode, borderValue);
                        // Note: For proper multi-channel support, need channel-aware interpolation
                        // For now, handle grayscale primarily
                    }
                    dstRow[x * numChannels + c] = static_cast<uint8_t>(
                        std::clamp(val, 0.0, 255.0));
                }
            }
        }
    } else if (src.Type() == PixelType::UInt16) {
        const uint16_t* srcData = static_cast<const uint16_t*>(src.Data());
        uint16_t* dstData = static_cast<uint16_t*>(dst.Data());

        size_t srcElemStride = srcStride / sizeof(uint16_t);
        size_t dstElemStride = dstStride / sizeof(uint16_t);

        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < dstH; ++y) {
            size_t mapRowOffset = static_cast<size_t>(y) * dstW;
            uint16_t* dstRow = dstData + y * dstElemStride;

            for (int32_t x = 0; x < dstW; ++x) {
                float srcX = map.mapX[mapRowOffset + x];
                float srcY = map.mapY[mapRowOffset + x];

                double val = Internal::InterpolateStrided<uint16_t>(
                    srcData, srcW, srcH, srcElemStride,
                    srcX, srcY, method, borderMode, borderValue);
                dstRow[x] = static_cast<uint16_t>(
                    std::clamp(val, 0.0, 65535.0));
            }
        }
    } else if (src.Type() == PixelType::Float32) {
        const float* srcData = static_cast<const float*>(src.Data());
        float* dstData = static_cast<float*>(dst.Data());

        size_t srcElemStride = srcStride / sizeof(float);
        size_t dstElemStride = dstStride / sizeof(float);

        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < dstH; ++y) {
            size_t mapRowOffset = static_cast<size_t>(y) * dstW;
            float* dstRow = dstData + y * dstElemStride;

            for (int32_t x = 0; x < dstW; ++x) {
                float srcX = map.mapX[mapRowOffset + x];
                float srcY = map.mapY[mapRowOffset + x];

                double val = Internal::InterpolateStrided<float>(
                    srcData, srcW, srcH, srcElemStride,
                    srcX, srcY, method, borderMode, static_cast<float>(borderValue));
                dstRow[x] = static_cast<float>(val);
            }
        }
    }
}

CameraIntrinsics GetOptimalNewCameraMatrix(
    const CameraModel& camera,
    double alpha,
    const Size2i& newImageSize)
{
    if (newImageSize.width < 0 || newImageSize.height < 0) {
        throw InvalidArgumentException("GetOptimalNewCameraMatrix: newImageSize must be >= 0");
    }

    const CameraIntrinsics& K = camera.Intrinsics();
    const DistortionCoeffs& dist = camera.Distortion();

    int32_t imgW = (newImageSize.width > 0) ? newImageSize.width : camera.ImageSize().width;
    int32_t imgH = (newImageSize.height > 0) ? newImageSize.height : camera.ImageSize().height;

    if (imgW <= 0 || imgH <= 0) {
        throw InvalidArgumentException("GetOptimalNewCameraMatrix: image size must be positive");
    }

    // If no distortion, return original
    if (dist.IsZero()) {
        return K;
    }

    // Sample the image boundary to find valid region
    const int numSamples = 100;
    std::vector<Point2d> boundaryPoints;
    boundaryPoints.reserve(numSamples * 4);

    // Top and bottom edges
    for (int i = 0; i < numSamples; ++i) {
        double x = static_cast<double>(i) / (numSamples - 1) * (imgW - 1);
        boundaryPoints.emplace_back(x, 0);
        boundaryPoints.emplace_back(x, imgH - 1);
    }
    // Left and right edges
    for (int i = 0; i < numSamples; ++i) {
        double y = static_cast<double>(i) / (numSamples - 1) * (imgH - 1);
        boundaryPoints.emplace_back(0, y);
        boundaryPoints.emplace_back(imgW - 1, y);
    }

    // Undistort boundary points
    std::vector<Point2d> undistortedBoundary = UndistortPoints(boundaryPoints, camera);

    // Find bounding box of undistorted boundary
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();

    for (const auto& pt : undistortedBoundary) {
        minX = std::min(minX, pt.x);
        maxX = std::max(maxX, pt.x);
        minY = std::min(minY, pt.y);
        maxY = std::max(maxY, pt.y);
    }

    // Compute inner rectangle (all pixels valid) at alpha=0
    // and outer rectangle (keep all source pixels) at alpha=1
    double innerMinX = minX, innerMaxX = maxX;
    double innerMinY = minY, innerMaxY = maxY;
    double outerMinX = 0, outerMaxX = imgW - 1;
    double outerMinY = 0, outerMaxY = imgH - 1;

    // Blend based on alpha
    double newMinX = innerMinX + alpha * (outerMinX - innerMinX);
    double newMaxX = innerMaxX + alpha * (outerMaxX - innerMaxX);
    double newMinY = innerMinY + alpha * (outerMinY - innerMinY);
    double newMaxY = innerMaxY + alpha * (outerMaxY - innerMaxY);

    // Compute new intrinsics that maps this region to [0, imgW) x [0, imgH)
    double scaleX = static_cast<double>(imgW - 1) / (newMaxX - newMinX);
    double scaleY = static_cast<double>(imgH - 1) / (newMaxY - newMinY);

    // Use uniform scale to avoid aspect ratio distortion
    double scale = std::min(scaleX, scaleY);

    CameraIntrinsics newK;
    newK.fx = K.fx * scale;
    newK.fy = K.fy * scale;
    newK.cx = K.cx - newMinX * scale;
    newK.cy = K.cy - newMinY * scale;

    return newK;
}

Point2d UndistortPoint(const Point2d& point, const CameraModel& camera) {
    if (!point.IsValid()) {
        throw InvalidArgumentException("UndistortPoint: invalid point");
    }
    const CameraIntrinsics& K = camera.Intrinsics();

    // Convert to normalized coordinates
    double x_norm = (point.x - K.cx) / K.fx;
    double y_norm = (point.y - K.cy) / K.fy;

    // Remove distortion
    Point2d undistorted = camera.Undistort(Point2d(x_norm, y_norm));

    // Convert back to pixel coordinates
    return Point2d(
        undistorted.x * K.fx + K.cx,
        undistorted.y * K.fy + K.cy
    );
}

std::vector<Point2d> UndistortPoints(
    const std::vector<Point2d>& points,
    const CameraModel& camera)
{
    std::vector<Point2d> result;
    result.reserve(points.size());

    for (const auto& pt : points) {
        if (!pt.IsValid()) {
            throw InvalidArgumentException("UndistortPoints: invalid point");
        }
        result.push_back(UndistortPoint(pt, camera));
    }

    return result;
}

Point2d DistortPoint(const Point2d& point, const CameraModel& camera) {
    if (!point.IsValid()) {
        throw InvalidArgumentException("DistortPoint: invalid point");
    }
    const CameraIntrinsics& K = camera.Intrinsics();

    // Convert to normalized coordinates (these are ideal/undistorted)
    double x_norm = (point.x - K.cx) / K.fx;
    double y_norm = (point.y - K.cy) / K.fy;

    // Apply distortion
    Point2d distorted = camera.Distort(Point2d(x_norm, y_norm));

    // Convert back to pixel coordinates
    return Point2d(
        distorted.x * K.fx + K.cx,
        distorted.y * K.fy + K.cy
    );
}

} // namespace Qi::Vision::Calib
