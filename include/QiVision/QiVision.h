#pragma once

/**
 * @file QiVision.h
 * @brief Main header file for QiVision library
 *
 * QiVision is an industrial machine vision library providing
 * Halcon-like functionality with sub-pixel accuracy.
 *
 * @author QiVision Team
 * @version 0.1.0
 */

// Configuration and export macros
#include <QiVision/QiVisionConfig.h>
#include <QiVision/Core/Export.h>

// Core types and utilities
#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Core/Exception.h>

// Core data structures
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
#include <QiVision/Core/QContour.h>
#include <QiVision/Core/QContourArray.h>
#include <QiVision/Core/QMatrix.h>

// Platform abstraction
#include <QiVision/Platform/Memory.h>
#include <QiVision/Platform/SIMD.h>

// Feature modules
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Filter/Filter.h>
#include <QiVision/Segment/Segment.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Morphology/Morphology.h>
#include <QiVision/Display/Display.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/Calib/CameraModel.h>
#include <QiVision/Calib/Undistort.h>
#include <QiVision/Calib/CalibBoard.h>
#include <QiVision/Calib/CameraCalib.h>
#include <QiVision/Transform/PolarTransform.h>
#include <QiVision/Transform/AffineTransform.h>
#include <QiVision/Transform/Homography.h>
#include <QiVision/Edge/Edge.h>
#include <QiVision/Hough/Hough.h>
#include <QiVision/Contour/Contour.h>
#include <QiVision/Defect/VariationModel.h>
#include <QiVision/Texture/Texture.h>

namespace Qi::Vision {

/**
 * @brief Get library version string
 * @return Version string in format "major.minor.patch"
 */
inline const char* GetVersion() {
    return QIVISION_VERSION_STRING;
}

/**
 * @brief Get library version as integers
 */
inline void GetVersion(int& major, int& minor, int& patch) {
    major = QIVISION_VERSION_MAJOR;
    minor = QIVISION_VERSION_MINOR;
    patch = QIVISION_VERSION_PATCH;
}

} // namespace Qi::Vision
