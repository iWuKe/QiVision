#pragma once

/**
 * @file CoordTransform3d.h
 * @brief 3D coordinate transformation utilities
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Calib/QHomMat3d.h>
#include <QiVision/Core/Export.h>

#include <vector>

namespace Qi::Vision::Calib {

QIVISION_API Point3d TransformPoint3d(const QHomMat3d& H, const Point3d& p);

QIVISION_API std::vector<Point3d> TransformPoints3d(
    const QHomMat3d& H,
    const std::vector<Point3d>& points);

} // namespace Qi::Vision::Calib
