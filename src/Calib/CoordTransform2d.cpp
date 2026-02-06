/**
 * @file CoordTransform2d.cpp
 * @brief 2D coordinate transform implementation
 */

#include <QiVision/Calib/CoordTransform2d.h>

namespace Qi::Vision::Calib {

Point2d TransformPoint2d(const QHomMat2d& H, const Point2d& p) {
    double x = H.M00() * p.x + H.M01() * p.y + H.M02();
    double y = H.M10() * p.x + H.M11() * p.y + H.M12();
    return Point2d(x, y);
}

std::vector<Point2d> TransformPoints2d(
    const QHomMat2d& H,
    const std::vector<Point2d>& points)
{
    std::vector<Point2d> out;
    out.reserve(points.size());
    for (const auto& p : points) {
        out.push_back(TransformPoint2d(H, p));
    }
    return out;
}

} // namespace Qi::Vision::Calib
