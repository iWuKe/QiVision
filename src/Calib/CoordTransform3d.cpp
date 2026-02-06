/**
 * @file CoordTransform3d.cpp
 * @brief 3D coordinate transform implementation
 */

#include <QiVision/Calib/CoordTransform3d.h>

namespace Qi::Vision::Calib {

Point3d TransformPoint3d(const QHomMat3d& H, const Point3d& p) {
    return H.TransformPoint(p);
}

std::vector<Point3d> TransformPoints3d(
    const QHomMat3d& H,
    const std::vector<Point3d>& points)
{
    std::vector<Point3d> out;
    out.reserve(points.size());
    for (const auto& p : points) {
        out.push_back(H.TransformPoint(p));
    }
    return out;
}

} // namespace Qi::Vision::Calib
