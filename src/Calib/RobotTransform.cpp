/**
 * @file RobotTransform.cpp
 * @brief Robot transform utilities
 */

#include <QiVision/Calib/RobotTransform.h>

namespace Qi::Vision::Calib {

QHomMat3d ComposeTransform(const QHomMat3d& a, const QHomMat3d& b) {
    return a * b;
}

QHomMat3d InvertTransform(const QHomMat3d& a) {
    return a.InverseRigid();
}

} // namespace Qi::Vision::Calib
