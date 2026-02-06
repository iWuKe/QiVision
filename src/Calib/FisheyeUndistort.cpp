/**
 * @file FisheyeUndistort.cpp
 * @brief Fisheye undistortion implementation
 */

#include <QiVision/Calib/FisheyeUndistort.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace Qi::Vision::Calib {

namespace {

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

inline bool IsFinitePoint(const Point2d& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

} // namespace

// =============================================================================
// Core Undistortion
// =============================================================================

void FisheyeUndistort(
    const QImage& src,
    QImage& dst,
    const FisheyeCameraModel& camera,
    const FisheyeUndistortParams& params,
    Internal::InterpolationMethod method)
{
    if (!Validate::RequireImageValid(src, "FisheyeUndistort")) {
        dst = QImage();
        return;
    }
    if (!camera.IsValid()) {
        throw InvalidArgumentException("FisheyeUndistort: invalid camera model");
    }

    Size2i outSize = params.outputSize;
    if (outSize.width <= 0 || outSize.height <= 0) {
        outSize = Size2i(src.Width(), src.Height());
    }

    FisheyeUndistortMap map = InitFisheyeUndistortMap(camera, params);
    if (map.width != outSize.width || map.height != outSize.height) {
        map.width = outSize.width;
        map.height = outSize.height;
    }

    FisheyeRemap(src, dst, map, method, Internal::BorderMode::Constant, 0.0);
}

void FisheyeUndistort(
    const QImage& src,
    QImage& dst,
    const FisheyeCameraModel& camera,
    const CameraIntrinsics& newCameraMatrix,
    const Size2i& outputSize,
    Internal::InterpolationMethod method)
{
    if (!Validate::RequireImageValid(src, "FisheyeUndistort")) {
        dst = QImage();
        return;
    }
    if (!camera.IsValid()) {
        throw InvalidArgumentException("FisheyeUndistort: invalid camera model");
    }
    if (!std::isfinite(newCameraMatrix.fx) || !std::isfinite(newCameraMatrix.fy) ||
        newCameraMatrix.fx <= 0.0 || newCameraMatrix.fy <= 0.0) {
        throw InvalidArgumentException("FisheyeUndistort: invalid newCameraMatrix");
    }

    Size2i outSize = outputSize;
    if (outSize.width <= 0 || outSize.height <= 0) {
        outSize = Size2i(src.Width(), src.Height());
    }

    FisheyeUndistortMap map = InitFisheyeUndistortMap(camera, outSize, &newCameraMatrix);
    FisheyeRemap(src, dst, map, method, Internal::BorderMode::Constant, 0.0);
}

// =============================================================================
// Map Initialization
// =============================================================================

FisheyeUndistortMap InitFisheyeUndistortMap(
    const FisheyeCameraModel& camera,
    const FisheyeUndistortParams& params)
{
    if (!camera.IsValid()) {
        throw InvalidArgumentException("InitFisheyeUndistortMap: invalid camera model");
    }

    Size2i outSize = params.outputSize;
    if (outSize.width <= 0 || outSize.height <= 0) {
        outSize = camera.ImageSize();
    }
    if (outSize.width <= 0 || outSize.height <= 0) {
        throw InvalidArgumentException("InitFisheyeUndistortMap: outputSize must be positive");
    }

    if (params.projection == FisheyeProjection::Perspective) {
        CameraIntrinsics newK = GetOptimalFisheyeNewCameraMatrix(
            camera, params.balance, outSize, params.fovScale);
        return InitFisheyeUndistortMap(camera, outSize, &newK);
    }

    // Equirectangular projection map
    FisheyeUndistortMap map;
    map.width = outSize.width;
    map.height = outSize.height;
    size_t total = static_cast<size_t>(map.width) * map.height;
    map.mapX.resize(total);
    map.mapY.resize(total);

    const CameraIntrinsics& origK = camera.Intrinsics();

    #pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < map.height; ++y) {
        size_t rowOffset = static_cast<size_t>(y) * map.width;
        double v = (map.height > 1) ? static_cast<double>(y) / (map.height - 1) : 0.0;
        double lat = (0.5 - v) * PI;  // +pi/2 to -pi/2
        double cosLat = std::cos(lat);
        double sinLat = std::sin(lat);

        for (int32_t x = 0; x < map.width; ++x) {
            double u = (map.width > 1) ? static_cast<double>(x) / (map.width - 1) : 0.0;
            double lon = (u - 0.5) * TWO_PI;  // -pi to +pi

            double sinLon = std::sin(lon);
            double cosLon = std::cos(lon);

            // Ray direction on unit sphere
            double X = sinLon * cosLat;
            double Y = sinLat;
            double Z = cosLon * cosLat;

            if (Z <= 0.0) {
                map.mapX[rowOffset + x] = -1.0f;
                map.mapY[rowOffset + x] = -1.0f;
                continue;
            }

            Point2d normalized(X / Z, Y / Z);
            Point2d distorted = camera.Distort(normalized);

            double srcX = origK.fx * distorted.x + origK.cx;
            double srcY = origK.fy * distorted.y + origK.cy;

            map.mapX[rowOffset + x] = static_cast<float>(srcX);
            map.mapY[rowOffset + x] = static_cast<float>(srcY);
        }
    }

    return map;
}

FisheyeUndistortMap InitFisheyeUndistortMap(
    const FisheyeCameraModel& camera,
    const Size2i& outputSize,
    const CameraIntrinsics* newCameraMatrix)
{
    if (!camera.IsValid()) {
        throw InvalidArgumentException("InitFisheyeUndistortMap: invalid camera model");
    }

    FisheyeUndistortMap map;
    map.width = outputSize.width;
    map.height = outputSize.height;
    if (map.width <= 0 || map.height <= 0) {
        throw InvalidArgumentException("InitFisheyeUndistortMap: outputSize must be positive");
    }

    size_t total = static_cast<size_t>(map.width) * map.height;
    map.mapX.resize(total);
    map.mapY.resize(total);

    CameraIntrinsics newK = newCameraMatrix ? *newCameraMatrix : camera.Intrinsics();
    if (!std::isfinite(newK.fx) || !std::isfinite(newK.fy) || newK.fx <= 0.0 || newK.fy <= 0.0) {
        throw InvalidArgumentException("InitFisheyeUndistortMap: invalid new camera intrinsics");
    }

    const CameraIntrinsics& origK = camera.Intrinsics();

    const double fxInv = 1.0 / newK.fx;
    const double fyInv = 1.0 / newK.fy;

    #pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < map.height; ++y) {
        size_t rowOffset = static_cast<size_t>(y) * map.width;
        for (int32_t x = 0; x < map.width; ++x) {
            double xNorm = (x - newK.cx) * fxInv;
            double yNorm = (y - newK.cy) * fyInv;

            Point2d distorted = camera.Distort(Point2d(xNorm, yNorm));
            double srcX = origK.fx * distorted.x + origK.cx;
            double srcY = origK.fy * distorted.y + origK.cy;

            map.mapX[rowOffset + x] = static_cast<float>(srcX);
            map.mapY[rowOffset + x] = static_cast<float>(srcY);
        }
    }

    return map;
}

// =============================================================================
// Remap
// =============================================================================

void FisheyeRemap(
    const QImage& src,
    QImage& dst,
    const FisheyeUndistortMap& map,
    Internal::InterpolationMethod method,
    Internal::BorderMode borderMode,
    double borderValue)
{
    if (!Validate::RequireImageValid(src, "FisheyeRemap")) {
        dst = QImage();
        return;
    }
    if (!map.IsValid()) {
        throw InvalidArgumentException("FisheyeRemap: invalid map");
    }
    if (!std::isfinite(borderValue)) {
        throw InvalidArgumentException("FisheyeRemap: invalid borderValue");
    }

    if (dst.Width() != map.width || dst.Height() != map.height ||
        dst.Type() != src.Type() || dst.GetChannelType() != src.GetChannelType()) {
        dst = QImage(map.width, map.height, src.Type(), src.GetChannelType());
    }

    int32_t srcW = src.Width();
    int32_t srcH = src.Height();
    int32_t dstW = map.width;
    int32_t dstH = map.height;

    size_t srcStride = src.Stride();
    size_t dstStride = dst.Stride();
    int channels = src.Channels();

    if (src.Type() == PixelType::UInt8) {
        const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
        uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < dstH; ++y) {
            size_t mapRowOffset = static_cast<size_t>(y) * dstW;
            uint8_t* dstRow = dstData + y * dstStride;

            for (int32_t x = 0; x < dstW; ++x) {
                float srcX = map.mapX[mapRowOffset + x];
                float srcY = map.mapY[mapRowOffset + x];

                for (int c = 0; c < channels; ++c) {
                    double val;
                    if (channels == 1) {
                        val = GetInterpolatedPixel<uint8_t>(
                            srcData, srcW, srcH, srcStride,
                            srcX, srcY, method, borderMode, borderValue);
                        dstRow[x] = static_cast<uint8_t>(std::clamp(val, 0.0, 255.0));
                    } else {
                        const uint8_t* srcChan = srcData + c;
                        val = GetInterpolatedPixel<uint8_t>(
                            srcChan, srcW, srcH, srcStride,
                            srcX * channels, srcY, method, borderMode, borderValue);
                        dstRow[x * channels + c] = static_cast<uint8_t>(std::clamp(val, 0.0, 255.0));
                    }
                }
            }
        }
    } else if (src.Type() == PixelType::Float32) {
        const float* srcData = static_cast<const float*>(src.Data());
        float* dstData = static_cast<float*>(dst.Data());

        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < dstH; ++y) {
            size_t mapRowOffset = static_cast<size_t>(y) * dstW;
            float* dstRow = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(dstData) + y * dstStride);

            for (int32_t x = 0; x < dstW; ++x) {
                float srcX = map.mapX[mapRowOffset + x];
                float srcY = map.mapY[mapRowOffset + x];

                for (int c = 0; c < channels; ++c) {
                    double val;
                    if (channels == 1) {
                        val = GetInterpolatedPixel<float>(
                            srcData, srcW, srcH, srcStride,
                            srcX, srcY, method, borderMode, borderValue);
                        dstRow[x] = static_cast<float>(val);
                    } else {
                        const float* srcChan = reinterpret_cast<const float*>(
                            reinterpret_cast<const uint8_t*>(srcData) + c * sizeof(float));
                        val = GetInterpolatedPixel<float>(
                            srcChan, srcW, srcH, srcStride,
                            srcX * channels, srcY, method, borderMode, borderValue);
                        dstRow[x * channels + c] = static_cast<float>(val);
                    }
                }
            }
        }
    } else {
        throw InvalidArgumentException("FisheyeRemap: unsupported pixel type");
    }
}

// =============================================================================
// Camera Matrix Utilities
// =============================================================================

CameraIntrinsics GetOptimalFisheyeNewCameraMatrix(
    const FisheyeCameraModel& camera,
    double balance,
    const Size2i& newImageSize,
    double fovScale)
{
    if (!camera.IsValid()) {
        throw InvalidArgumentException("GetOptimalFisheyeNewCameraMatrix: invalid camera model");
    }

    Size2i size = newImageSize;
    if (size.width <= 0 || size.height <= 0) {
        size = camera.ImageSize();
    }
    if (size.width <= 0 || size.height <= 0) {
        throw InvalidArgumentException("GetOptimalFisheyeNewCameraMatrix: invalid image size");
    }

    balance = std::clamp(balance, 0.0, 1.0);
    if (!std::isfinite(fovScale) || fovScale <= 0.0) {
        fovScale = 1.0;
    }

    const CameraIntrinsics& K = camera.Intrinsics();

    // Sample points around the image border to estimate undistorted bounds
    const int samples = 32;
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();

    auto updateBounds = [&](double u, double v) {
        double x = (u - K.cx) / K.fx;
        double y = (v - K.cy) / K.fy;
        Point2d und = camera.Undistort(Point2d(x, y));
        if (!IsFinitePoint(und)) {
            return;
        }
        minX = std::min(minX, und.x);
        minY = std::min(minY, und.y);
        maxX = std::max(maxX, und.x);
        maxY = std::max(maxY, und.y);
    };

    for (int i = 0; i < samples; ++i) {
        double t = static_cast<double>(i) / (samples - 1);
        updateBounds(t * (size.width - 1), 0.0);
        updateBounds(t * (size.width - 1), size.height - 1);
        updateBounds(0.0, t * (size.height - 1));
        updateBounds(size.width - 1, t * (size.height - 1));
    }

    double spanX = maxX - minX;
    double spanY = maxY - minY;
    if (!(spanX > 0.0 && spanY > 0.0)) {
        return CameraIntrinsics(K.fx, K.fy, size.width * 0.5, size.height * 0.5);
    }

    double fxFull = (size.width - 1) / spanX;
    double fyFull = (size.height - 1) / spanY;

    double fFull = std::min(fxFull, fyFull);   // keep all pixels (may add borders)
    double fCrop = std::max(fxFull, fyFull);   // fill output (may crop)

    double f = fCrop * (1.0 - balance) + fFull * balance;
    f /= fovScale;

    CameraIntrinsics newK(f, f, size.width * 0.5, size.height * 0.5);
    return newK;
}

// =============================================================================
// FOV Scale Estimation
// =============================================================================

double EstimateFisheyeFovScale(
    const FisheyeCameraModel& camera,
    double targetFovDegrees)
{
    if (!camera.IsValid()) {
        throw InvalidArgumentException("EstimateFisheyeFovScale: invalid camera model");
    }
    if (!std::isfinite(targetFovDegrees) || targetFovDegrees <= 0.0) {
        throw InvalidArgumentException("EstimateFisheyeFovScale: invalid targetFovDegrees");
    }

    double targetFov = targetFovDegrees * PI / 180.0;
    double currentFov = std::max(camera.HorizontalFOV(), camera.VerticalFOV());
    if (currentFov <= 0.0) {
        return 1.0;
    }

    return currentFov / targetFov;
}

// =============================================================================
// Point Undistortion/Distortion
// =============================================================================

Point2d FisheyeUndistortPoint(
    const Point2d& point,
    const FisheyeCameraModel& camera)
{
    if (!camera.IsValid()) {
        throw InvalidArgumentException("FisheyeUndistortPoint: invalid camera model");
    }

    const auto& K = camera.Intrinsics();
    double x = (point.x - K.cx) / K.fx;
    double y = (point.y - K.cy) / K.fy;
    Point2d und = camera.Undistort(Point2d(x, y));

    return Point2d(und.x * K.fx + K.cx, und.y * K.fy + K.cy);
}

Point2d FisheyeUndistortPoint(
    const Point2d& point,
    const FisheyeCameraModel& camera,
    const CameraIntrinsics& newCameraMatrix)
{
    if (!camera.IsValid()) {
        throw InvalidArgumentException("FisheyeUndistortPoint: invalid camera model");
    }
    if (!std::isfinite(newCameraMatrix.fx) || !std::isfinite(newCameraMatrix.fy) ||
        newCameraMatrix.fx <= 0.0 || newCameraMatrix.fy <= 0.0) {
        throw InvalidArgumentException("FisheyeUndistortPoint: invalid newCameraMatrix");
    }

    const auto& K = camera.Intrinsics();
    double x = (point.x - K.cx) / K.fx;
    double y = (point.y - K.cy) / K.fy;
    Point2d und = camera.Undistort(Point2d(x, y));

    return Point2d(und.x * newCameraMatrix.fx + newCameraMatrix.cx,
                   und.y * newCameraMatrix.fy + newCameraMatrix.cy);
}

std::vector<Point2d> FisheyeUndistortPoints(
    const std::vector<Point2d>& points,
    const FisheyeCameraModel& camera)
{
    std::vector<Point2d> out;
    out.reserve(points.size());
    for (const auto& p : points) {
        out.push_back(FisheyeUndistortPoint(p, camera));
    }
    return out;
}

std::vector<Point2d> FisheyeUndistortPoints(
    const std::vector<Point2d>& points,
    const FisheyeCameraModel& camera,
    const CameraIntrinsics& newCameraMatrix)
{
    std::vector<Point2d> out;
    out.reserve(points.size());
    for (const auto& p : points) {
        out.push_back(FisheyeUndistortPoint(p, camera, newCameraMatrix));
    }
    return out;
}

Point2d FisheyeDistortPoint(
    const Point2d& point,
    const FisheyeCameraModel& camera)
{
    if (!camera.IsValid()) {
        throw InvalidArgumentException("FisheyeDistortPoint: invalid camera model");
    }

    const auto& K = camera.Intrinsics();
    double x = (point.x - K.cx) / K.fx;
    double y = (point.y - K.cy) / K.fy;
    Point2d dist = camera.Distort(Point2d(x, y));

    return Point2d(dist.x * K.fx + K.cx, dist.y * K.fy + K.cy);
}

std::vector<Point2d> FisheyeDistortPoints(
    const std::vector<Point2d>& points,
    const FisheyeCameraModel& camera)
{
    std::vector<Point2d> out;
    out.reserve(points.size());
    for (const auto& p : points) {
        out.push_back(FisheyeDistortPoint(p, camera));
    }
    return out;
}

// =============================================================================
// Projection Conversion
// =============================================================================

Point2d ConvertFisheyeProjection(
    const Point2d& point,
    const FisheyeCameraModel& camera,
    FisheyeProjection fromProjection,
    FisheyeProjection toProjection)
{
    if (fromProjection == toProjection) {
        return point;
    }

    if (!camera.IsValid()) {
        throw InvalidArgumentException("ConvertFisheyeProjection: invalid camera model");
    }

    // Convert input point to a ray direction in camera coordinates
    Point3d ray;
    if (fromProjection == FisheyeProjection::Perspective) {
        ray = camera.UnprojectPixel(point);
    } else if (fromProjection == FisheyeProjection::Equirectangular) {
        // Interpret point as (lon, lat) in radians
        double lon = point.x;
        double lat = point.y;
        double cosLat = std::cos(lat);
        ray = Point3d(std::sin(lon) * cosLat, std::sin(lat), std::cos(lon) * cosLat);
    } else {
        return point;
    }

    if (toProjection == FisheyeProjection::Perspective) {
        return camera.ProjectPoint(ray);
    }

    // Convert ray to equirectangular (lon, lat)
    double lon = std::atan2(ray.x, ray.z);
    double lat = std::atan2(ray.y, std::sqrt(ray.x * ray.x + ray.z * ray.z));
    return Point2d(lon, lat);
}

Rect2i GetFisheyeValidRegion(
    const FisheyeCameraModel& camera,
    const FisheyeUndistortParams& params)
{
    Size2i outSize = params.outputSize;
    if (outSize.width <= 0 || outSize.height <= 0) {
        outSize = camera.ImageSize();
    }
    if (outSize.width <= 0 || outSize.height <= 0) {
        return Rect2i(0, 0, 0, 0);
    }

    // Conservative: full image for now (future: compute real valid mask)
    return Rect2i(0, 0, outSize.width, outSize.height);
}

} // namespace Qi::Vision::Calib
