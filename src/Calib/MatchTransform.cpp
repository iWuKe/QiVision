/**
 * @file MatchTransform.cpp
 * @brief Match transform implementation
 */

#include <QiVision/Calib/MatchTransform.h>

namespace Qi::Vision::Calib {

Point3d MatchToWorld(
    const Matching::MatchResult& match,
    const QHomMat3d& imageToWorld,
    double z)
{
    Point3d p(match.x, match.y, z);
    return imageToWorld.TransformPoint(p);
}

} // namespace Qi::Vision::Calib
