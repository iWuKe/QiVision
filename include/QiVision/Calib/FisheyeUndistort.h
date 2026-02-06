#pragma once

/**
 * @file FisheyeUndistort.h
 * @brief Fisheye image undistortion and point transformation
 *
 * Provides comprehensive fisheye undistortion capabilities including:
 * - Multiple output projection types (perspective, equirectangular, etc.)
 * - Direct image undistortion
 * - Pre-computed undistortion maps for efficient batch processing
 * - Point-level undistortion and distortion operations
 *
 * The undistortion process converts fisheye (equidistant projection) images
 * to various standard projections, enabling compatibility with algorithms
 * designed for standard pinhole cameras.
 *
 * Reference:
 * - OpenCV fisheye undistortion (cv::fisheye::undistortImage)
 * - Various projection models for omnidirectional cameras
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Calib/FisheyeModel.h>
#include <QiVision/Calib/CameraModel.h>
#include <QiVision/Internal/Interpolate.h>
#include <QiVision/Core/Export.h>

#include <vector>

namespace Qi::Vision::Calib {

/**
 * @brief Output projection type for fisheye undistortion
 *
 * Different projection models produce different output images with
 * various geometric properties:
 *
 * - Perspective: Standard pinhole (rectilinear) projection
 *   - Preserves straight lines
 *   - Limited FOV (typically < 120 degrees)
 *   - Best for machine vision algorithms expecting standard camera model
 *
 * - Equirectangular: Cylindrical panoramic projection
 *   - Maps longitude to x, latitude to y
 *   - Good for 360-degree panoramas
 *   - Horizontal lines at equator remain straight
 */
enum class FisheyeProjection {
    Perspective,        ///< Standard rectilinear (pinhole) projection
    Equirectangular     ///< Cylindrical panoramic projection
};

/**
 * @brief Parameters for fisheye undistortion
 *
 * Controls the output projection type, field of view balance,
 * and output image dimensions.
 */
struct QIVISION_API FisheyeUndistortParams {
    /**
     * @brief Output projection type
     *
     * Determines how the fisheye image is mapped to the output.
     * Default: Perspective (standard pinhole model)
     */
    FisheyeProjection projection = FisheyeProjection::Perspective;

    /**
     * @brief Balance between cropping and black borders
     *
     * Controls the trade-off between:
     * - balance = 0.0: All output pixels are valid (may crop corners)
     * - balance = 1.0: All input pixels are visible (may have black borders)
     *
     * Values between 0 and 1 provide intermediate results.
     * Default: 0.0 (prefer valid pixels)
     */
    double balance = 0.0;

    /**
     * @brief Field of view scaling factor
     *
     * Adjusts the effective field of view of the output:
     * - fovScale > 1.0: Zoom out (wider FOV, smaller objects)
     * - fovScale < 1.0: Zoom in (narrower FOV, larger objects)
     * - fovScale = 1.0: Natural mapping based on focal length
     *
     * Default: 1.0
     */
    double fovScale = 1.0;

    /**
     * @brief Output image size
     *
     * Size2i(0, 0) means use the same size as the input image.
     */
    Size2i outputSize{0, 0};

    /// Default constructor
    FisheyeUndistortParams() = default;

    /**
     * @brief Construct with main parameters
     * @param proj Output projection type
     * @param bal Balance factor [0, 1]
     * @param fov FOV scaling factor
     */
    FisheyeUndistortParams(FisheyeProjection proj, double bal = 0.0, double fov = 1.0)
        : projection(proj), balance(bal), fovScale(fov) {}
};

/**
 * @brief Pre-computed fisheye undistortion map
 *
 * Stores source (x, y) coordinates for each destination pixel.
 * Using float for memory efficiency while maintaining sub-pixel accuracy.
 *
 * For batch processing, compute the map once then reuse for multiple images.
 */
struct QIVISION_API FisheyeUndistortMap {
    std::vector<float> mapX;    ///< Source X coordinates (row-major)
    std::vector<float> mapY;    ///< Source Y coordinates (row-major)
    int32_t width = 0;          ///< Map width (output image width)
    int32_t height = 0;         ///< Map height (output image height)

    /**
     * @brief Check if map is valid and ready for use
     * @return true if map contains valid data
     */
    bool IsValid() const {
        return width > 0 && height > 0 &&
               mapX.size() == static_cast<size_t>(width * height) &&
               mapY.size() == static_cast<size_t>(width * height);
    }

    /**
     * @brief Get source coordinates for destination pixel
     * @param dstX Destination x coordinate
     * @param dstY Destination y coordinate
     * @return Source (x, y) coordinates as Point2d
     */
    Point2d GetSourceCoord(int32_t dstX, int32_t dstY) const {
        size_t idx = static_cast<size_t>(dstY * width + dstX);
        return Point2d(mapX[idx], mapY[idx]);
    }
};

// =============================================================================
// Image Undistortion Functions
// =============================================================================

/**
 * @brief Remove fisheye distortion from image
 *
 * Undistorts a fisheye image using the specified projection model and
 * parameters. For processing multiple images with the same camera,
 * use InitFisheyeUndistortMap() + FisheyeRemap() for better performance.
 *
 * @param src Source (distorted) fisheye image
 * @param dst Output (undistorted) image
 * @param camera Fisheye camera model with intrinsics and distortion
 * @param params Undistortion parameters (projection type, balance, etc.)
 * @param method Interpolation method (default: Bilinear)
 *
 * @code
 * // Example: Undistort to perspective view
 * FisheyeUndistortParams params(FisheyeProjection::Perspective, 0.5);
 * FisheyeUndistort(fisheyeImage, undistortedImage, camera, params);
 *
 * // Example: Create panoramic view
 * FisheyeUndistortParams panoParams(FisheyeProjection::Equirectangular);
 * panoParams.outputSize = Size2i(1920, 960);
 * FisheyeUndistort(fisheyeImage, panorama, camera, panoParams);
 * @endcode
 */
QIVISION_API void FisheyeUndistort(
    const QImage& src,
    QImage& dst,
    const FisheyeCameraModel& camera,
    const FisheyeUndistortParams& params = FisheyeUndistortParams(),
    Internal::InterpolationMethod method = Internal::InterpolationMethod::Bilinear
);

/**
 * @brief Remove fisheye distortion with custom output camera matrix
 *
 * Allows specifying a custom output camera matrix for precise control
 * over the undistorted image geometry. The output will use standard
 * pinhole projection with the specified intrinsics.
 *
 * @param src Source (distorted) fisheye image
 * @param dst Output (undistorted) image
 * @param camera Original fisheye camera model
 * @param newCameraMatrix Camera intrinsics for the output image
 * @param outputSize Output image size (0,0 = same as source)
 * @param method Interpolation method
 *
 * @code
 * // Create output with custom focal length
 * CameraIntrinsics newK(500.0, 500.0, 960.0, 540.0);
 * FisheyeUndistort(fisheyeImg, outputImg, camera, newK, Size2i(1920, 1080));
 * @endcode
 */
QIVISION_API void FisheyeUndistort(
    const QImage& src,
    QImage& dst,
    const FisheyeCameraModel& camera,
    const CameraIntrinsics& newCameraMatrix,
    const Size2i& outputSize = Size2i(0, 0),
    Internal::InterpolationMethod method = Internal::InterpolationMethod::Bilinear
);

// =============================================================================
// Undistortion Map Functions
// =============================================================================

/**
 * @brief Initialize fisheye undistortion map for efficient batch processing
 *
 * Pre-computes the source-to-destination coordinate mapping once.
 * Use this map with FisheyeRemap() for efficient processing of multiple
 * images with the same camera parameters.
 *
 * @param camera Fisheye camera model
 * @param params Undistortion parameters
 * @return Undistortion map for use with FisheyeRemap()
 *
 * @code
 * // Batch processing example
 * auto map = InitFisheyeUndistortMap(camera, params);
 *
 * for (const auto& inputImage : images) {
 *     QImage output;
 *     FisheyeRemap(inputImage, output, map);
 *     // Process output...
 * }
 * @endcode
 */
QIVISION_API FisheyeUndistortMap InitFisheyeUndistortMap(
    const FisheyeCameraModel& camera,
    const FisheyeUndistortParams& params = FisheyeUndistortParams()
);

/**
 * @brief Initialize fisheye undistortion map with custom output camera
 *
 * Creates an undistortion map that transforms the fisheye image to
 * a standard pinhole projection with the specified camera matrix.
 *
 * @param camera Original fisheye camera model
 * @param outputSize Output image size
 * @param newCameraMatrix New camera intrinsics (null = compute optimal)
 * @return Undistortion map
 */
QIVISION_API FisheyeUndistortMap InitFisheyeUndistortMap(
    const FisheyeCameraModel& camera,
    const Size2i& outputSize,
    const CameraIntrinsics* newCameraMatrix = nullptr
);

/**
 * @brief Apply remapping using pre-computed undistortion map
 *
 * Performs the actual image transformation using a pre-computed map.
 * This is more efficient than FisheyeUndistort() when processing
 * multiple images with the same camera parameters.
 *
 * @param src Source fisheye image
 * @param dst Output undistorted image
 * @param map Pre-computed undistortion map from InitFisheyeUndistortMap()
 * @param method Interpolation method
 * @param borderMode Border handling mode
 * @param borderValue Border value for constant mode
 */
QIVISION_API void FisheyeRemap(
    const QImage& src,
    QImage& dst,
    const FisheyeUndistortMap& map,
    Internal::InterpolationMethod method = Internal::InterpolationMethod::Bilinear,
    Internal::BorderMode borderMode = Internal::BorderMode::Constant,
    double borderValue = 0.0
);

// =============================================================================
// Optimal Camera Matrix Computation
// =============================================================================

/**
 * @brief Compute optimal new camera matrix for fisheye undistortion
 *
 * Computes camera intrinsics that balance between:
 * - All output pixels being valid (no undefined regions)
 * - All input pixels being visible (no cropping)
 *
 * The balance parameter controls this trade-off:
 * - balance = 0: Maximize valid output pixels (may crop input edges)
 * - balance = 1: Keep all input pixels (may have black borders)
 *
 * @param camera Original fisheye camera model
 * @param balance Balance parameter [0, 1]
 * @param newImageSize Desired output image size (0,0 = same as original)
 * @param fovScale Field of view scaling factor
 * @return Optimal camera intrinsics for the output image
 *
 * @code
 * // Get optimal camera matrix with 50% balance
 * auto newK = GetOptimalFisheyeNewCameraMatrix(camera, 0.5, Size2i(1920, 1080));
 *
 * // Use with undistortion
 * FisheyeUndistort(src, dst, camera, newK, Size2i(1920, 1080));
 * @endcode
 */
QIVISION_API CameraIntrinsics GetOptimalFisheyeNewCameraMatrix(
    const FisheyeCameraModel& camera,
    double balance = 0.0,
    const Size2i& newImageSize = Size2i(0, 0),
    double fovScale = 1.0
);

/**
 * @brief Estimate FOV scaling factor to achieve target field of view
 *
 * Calculates the fovScale parameter needed to achieve a specific
 * field of view in the undistorted output image.
 *
 * @param camera Fisheye camera model
 * @param targetFovDegrees Target diagonal FOV in degrees (default: 90)
 * @return FOV scale factor for use in undistortion parameters
 *
 * @code
 * double scale = EstimateFisheyeFovScale(camera, 120.0);  // 120 degree output
 * FisheyeUndistortParams params(FisheyeProjection::Perspective, 0.0, scale);
 * @endcode
 */
QIVISION_API double EstimateFisheyeFovScale(
    const FisheyeCameraModel& camera,
    double targetFovDegrees = 90.0
);

// =============================================================================
// Point Undistortion/Distortion Functions
// =============================================================================

/**
 * @brief Undistort a single point from fisheye image
 *
 * Transforms a pixel coordinate from the distorted fisheye image to
 * the corresponding coordinate in an ideal pinhole camera image.
 *
 * @param point Distorted pixel coordinate in fisheye image
 * @param camera Fisheye camera model
 * @return Undistorted pixel coordinate (ideal pinhole projection)
 */
QIVISION_API Point2d FisheyeUndistortPoint(
    const Point2d& point,
    const FisheyeCameraModel& camera
);

/**
 * @brief Undistort a point with custom output camera matrix
 *
 * Transforms a point from fisheye image coordinates to coordinates
 * in an image with the specified pinhole camera intrinsics.
 *
 * @param point Distorted pixel coordinate in fisheye image
 * @param camera Original fisheye camera model
 * @param newCameraMatrix Output camera intrinsics
 * @return Undistorted pixel coordinate in new camera frame
 */
QIVISION_API Point2d FisheyeUndistortPoint(
    const Point2d& point,
    const FisheyeCameraModel& camera,
    const CameraIntrinsics& newCameraMatrix
);

/**
 * @brief Undistort multiple points from fisheye image
 *
 * Batch version of FisheyeUndistortPoint() for efficiency when
 * processing many points.
 *
 * @param points Distorted pixel coordinates
 * @param camera Fisheye camera model
 * @return Undistorted pixel coordinates
 */
QIVISION_API std::vector<Point2d> FisheyeUndistortPoints(
    const std::vector<Point2d>& points,
    const FisheyeCameraModel& camera
);

/**
 * @brief Undistort multiple points with custom output camera
 *
 * @param points Distorted pixel coordinates
 * @param camera Original fisheye camera model
 * @param newCameraMatrix Output camera intrinsics
 * @return Undistorted pixel coordinates in new camera frame
 */
QIVISION_API std::vector<Point2d> FisheyeUndistortPoints(
    const std::vector<Point2d>& points,
    const FisheyeCameraModel& camera,
    const CameraIntrinsics& newCameraMatrix
);

/**
 * @brief Distort a point using fisheye model (inverse operation)
 *
 * Transforms an ideal pinhole coordinate to the corresponding
 * coordinate in the fisheye image. Useful for:
 * - Simulating fisheye images
 * - Mapping overlays onto fisheye images
 * - Testing undistortion accuracy (round-trip test)
 *
 * @param point Ideal (undistorted) pixel coordinate
 * @param camera Fisheye camera model
 * @return Distorted pixel coordinate in fisheye image
 */
QIVISION_API Point2d FisheyeDistortPoint(
    const Point2d& point,
    const FisheyeCameraModel& camera
);

/**
 * @brief Distort multiple points using fisheye model
 *
 * Batch version of FisheyeDistortPoint().
 *
 * @param points Ideal pixel coordinates
 * @param camera Fisheye camera model
 * @return Distorted pixel coordinates
 */
QIVISION_API std::vector<Point2d> FisheyeDistortPoints(
    const std::vector<Point2d>& points,
    const FisheyeCameraModel& camera
);

// =============================================================================
// Projection Conversion
// =============================================================================

/**
 * @brief Convert point between fisheye projection types
 *
 * Transforms a point from one fisheye projection model to another.
 * Useful for converting between different panoramic representations.
 *
 * @param point Input point in source projection
 * @param camera Fisheye camera model
 * @param fromProjection Source projection type
 * @param toProjection Target projection type
 * @return Point in target projection coordinates
 *
 * @code
 * // Convert from perspective to equirectangular
 * Point2d equiPt = ConvertFisheyeProjection(perspPt, camera,
 *     FisheyeProjection::Perspective, FisheyeProjection::Equirectangular);
 * @endcode
 */
QIVISION_API Point2d ConvertFisheyeProjection(
    const Point2d& point,
    const FisheyeCameraModel& camera,
    FisheyeProjection fromProjection,
    FisheyeProjection toProjection
);

/**
 * @brief Get valid image region after undistortion
 *
 * Computes the rectangular region in the output image that contains
 * only valid (non-black) pixels after undistortion.
 *
 * @param camera Fisheye camera model
 * @param params Undistortion parameters
 * @return Valid region as rectangle (may be smaller than output size)
 */
QIVISION_API Rect2i GetFisheyeValidRegion(
    const FisheyeCameraModel& camera,
    const FisheyeUndistortParams& params
);

} // namespace Qi::Vision::Calib
