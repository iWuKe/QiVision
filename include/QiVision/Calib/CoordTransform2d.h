#pragma once

/**
 * @file CoordTransform2d.h
 * @brief 2D coordinate transformation utilities
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Calib/QHomMat2d.h>
#include <QiVision/Core/Export.h>

#include <vector>

namespace Qi::Vision::Calib {

QIVISION_API Point2d TransformPoint2d(const QHomMat2d& H, const Point2d& p);

QIVISION_API std::vector<Point2d> TransformPoints2d(
    const QHomMat2d& H,
    const std::vector<Point2d>& points);

} // namespace Qi::Vision::Calib
