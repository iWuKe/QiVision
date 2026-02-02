#pragma once

/**
 * @file Undistort.h
 * @brief Image undistortion functions
 *
 * Provides:
 * - Direct image undistortion
 * - Pre-computed undistortion maps for efficient batch processing
 * - Remap function using pre-computed maps
 *
 * Reference: OpenCV undistort, initUndistortRectifyMap
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Calib/CameraModel.h>
#include <QiVision/Internal/Interpolate.h>
#include <QiVision/Core/Export.h>

#include <vector>

namespace Qi::Vision::Calib {

/**
 * @brief Pre-computed undistortion map for efficient remapping
 *
 * Stores (x, y) coordinates in source image for each destination pixel.
 * Using float for memory efficiency while maintaining sub-pixel accuracy.
 */
struct QIVISION_API UndistortMap {
    std::vector<float> mapX;    ///< Source X coordinates
    std::vector<float> mapY;    ///< Source Y coordinates
    int32_t width = 0;          ///< Map width
    int32_t height = 0;         ///< Map height

    bool IsValid() const {
        return width > 0 && height > 0 &&
               mapX.size() == static_cast<size_t>(width * height) &&
               mapY.size() == static_cast<size_t>(width * height);
    }
};

/**
 * @brief Remove lens distortion from image
 *
 * @param src Source (distorted) image
 * @param dst Output (undistorted) image
 * @param camera Camera model with intrinsics and distortion
 * @param method Interpolation method
 */
QIVISION_API void Undistort(
    const QImage& src,
    QImage& dst,
    const CameraModel& camera,
    Internal::InterpolationMethod method = Internal::InterpolationMethod::Bilinear
);

/**
 * @brief Remove lens distortion with new camera matrix
 *
 * @param src Source (distorted) image
 * @param dst Output (undistorted) image
 * @param camera Original camera model
 * @param newCameraMatrix New intrinsic matrix (can adjust FOV)
 * @param outputSize Output image size (0 = same as source)
 * @param method Interpolation method
 */
QIVISION_API void Undistort(
    const QImage& src,
    QImage& dst,
    const CameraModel& camera,
    const CameraIntrinsics& newCameraMatrix,
    const Size2i& outputSize = Size2i(0, 0),
    Internal::InterpolationMethod method = Internal::InterpolationMethod::Bilinear
);

/**
 * @brief Initialize undistortion map for efficient batch processing
 *
 * Pre-computes the remapping coordinates once, then use Remap() for
 * fast undistortion of multiple images with same camera parameters.
 *
 * @param camera Camera model
 * @param outputSize Output image size
 * @param newCameraMatrix Optional new camera matrix (null = use original)
 * @return Undistortion map
 */
QIVISION_API UndistortMap InitUndistortMap(
    const CameraModel& camera,
    const Size2i& outputSize,
    const CameraIntrinsics* newCameraMatrix = nullptr
);

/**
 * @brief Apply remapping using pre-computed map
 *
 * @param src Source image
 * @param dst Output image
 * @param map Pre-computed undistortion map
 * @param method Interpolation method
 * @param borderMode Border handling
 * @param borderValue Border value for constant mode
 */
QIVISION_API void Remap(
    const QImage& src,
    QImage& dst,
    const UndistortMap& map,
    Internal::InterpolationMethod method = Internal::InterpolationMethod::Bilinear,
    Internal::BorderMode borderMode = Internal::BorderMode::Constant,
    double borderValue = 0.0
);

/**
 * @brief Compute optimal new camera matrix for undistortion
 *
 * @param camera Original camera model
 * @param alpha Free scaling parameter (0 = all pixels valid, 1 = keep all source pixels)
 * @param newImageSize New image size (0,0 = same as original)
 * @return Optimal camera matrix that balances FOV and valid pixels
 */
QIVISION_API CameraIntrinsics GetOptimalNewCameraMatrix(
    const CameraModel& camera,
    double alpha = 1.0,
    const Size2i& newImageSize = Size2i(0, 0)
);

/**
 * @brief Undistort a single point
 *
 * @param point Distorted pixel coordinate
 * @param camera Camera model
 * @return Undistorted pixel coordinate
 */
QIVISION_API Point2d UndistortPoint(const Point2d& point, const CameraModel& camera);

/**
 * @brief Undistort multiple points
 *
 * @param points Distorted pixel coordinates
 * @param camera Camera model
 * @return Undistorted pixel coordinates
 */
QIVISION_API std::vector<Point2d> UndistortPoints(
    const std::vector<Point2d>& points,
    const CameraModel& camera
);

/**
 * @brief Distort a single point (for simulation)
 *
 * @param point Ideal (undistorted) pixel coordinate
 * @param camera Camera model
 * @return Distorted pixel coordinate
 */
QIVISION_API Point2d DistortPoint(const Point2d& point, const CameraModel& camera);

} // namespace Qi::Vision::Calib
