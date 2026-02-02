#pragma once

/**
 * @file CalibBoard.h
 * @brief Calibration board detection (chessboard corners)
 *
 * Provides:
 * - Chessboard corner detection
 * - Subpixel corner refinement
 *
 * Reference: OpenCV findChessboardCorners, cornerSubPix
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Export.h>

#include <vector>

namespace Qi::Vision::Calib {

/**
 * @brief Result of chessboard corner detection
 */
struct QIVISION_API CornerGrid {
    std::vector<Point2d> corners;   ///< Detected corners in row-major order
    int32_t rows = 0;               ///< Pattern rows (inner corners)
    int32_t cols = 0;               ///< Pattern columns (inner corners)
    bool found = false;             ///< Whether pattern was fully detected

    /// Get corner at (row, col)
    Point2d At(int32_t row, int32_t col) const {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            return Point2d();
        }
        return corners[row * cols + col];
    }

    /// Total corner count
    size_t Count() const { return corners.size(); }

    /// Check if grid is valid
    bool IsValid() const { return found && corners.size() == static_cast<size_t>(rows * cols); }
};

/**
 * @brief Chessboard detection flags
 */
enum class ChessboardFlags : uint32_t {
    None = 0,
    AdaptiveThresh = 1,     ///< Use adaptive thresholding
    NormalizeImage = 2,     ///< Normalize image before detection
    FilterQuads = 4,        ///< Filter out false quads
    FastCheck = 8           ///< Quick check if chessboard present
};

// Bitwise operators for flags
inline ChessboardFlags operator|(ChessboardFlags a, ChessboardFlags b) {
    return static_cast<ChessboardFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline bool operator&(ChessboardFlags a, ChessboardFlags b) {
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
}

/**
 * @brief Find chessboard corners in image
 *
 * Detects the inner corners of a chessboard pattern.
 * Pattern size is (patternCols x patternRows) inner corners.
 *
 * Algorithm:
 * 1. Apply adaptive thresholding to binarize image
 * 2. Find contours and approximate to quadrilaterals
 * 3. Find corner intersections from quad corners
 * 4. Cluster and organize corners into grid
 * 5. Refine corners to subpixel accuracy
 *
 * @param image Input grayscale image
 * @param patternCols Number of inner corners per row
 * @param patternRows Number of inner corners per column
 * @param flags Detection flags
 * @return CornerGrid with detected corners (empty if not found)
 */
QIVISION_API CornerGrid FindChessboardCorners(
    const QImage& image,
    int32_t patternCols,
    int32_t patternRows,
    ChessboardFlags flags = ChessboardFlags::AdaptiveThresh | ChessboardFlags::NormalizeImage
);

/**
 * @brief Refine corner positions to subpixel accuracy
 *
 * Uses gradient-based corner refinement within a local window.
 * The algorithm iteratively adjusts the corner position to minimize
 * the sum of squared dot products between gradient and position vectors.
 *
 * @param image Input grayscale image
 * @param corners Input/Output corner positions (refined in place)
 * @param winSize Half-size of search window (actual size = 2*winSize+1)
 * @param maxIterations Maximum refinement iterations
 * @param epsilon Convergence threshold (pixel movement)
 */
QIVISION_API void CornerSubPix(
    const QImage& image,
    std::vector<Point2d>& corners,
    int32_t winSize = 5,
    int32_t maxIterations = 30,
    double epsilon = 0.001
);

/**
 * @brief Generate ideal chessboard corner positions in object space
 *
 * Creates a grid of 3D points at z=0 with specified square size.
 *
 * @param patternCols Number of inner corners per row
 * @param patternRows Number of inner corners per column
 * @param squareSize Size of each square (same units as world coordinates)
 * @return Vector of 3D corner positions
 */
QIVISION_API std::vector<Point3d> GenerateChessboardPoints(
    int32_t patternCols,
    int32_t patternRows,
    double squareSize = 1.0
);

/**
 * @brief Draw detected chessboard corners on image
 *
 * Draws crosses at each corner location. If drawOrder is true,
 * also draws lines connecting corners in row-major order and
 * uses color gradient to show corner order.
 *
 * @param image Input/Output color image (will be modified)
 * @param grid Detected corner grid
 * @param drawOrder Draw lines connecting corners in order
 */
QIVISION_API void DrawChessboardCorners(
    QImage& image,
    const CornerGrid& grid,
    bool drawOrder = true
);

} // namespace Qi::Vision::Calib
