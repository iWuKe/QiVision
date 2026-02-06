#pragma once

/**
 * @file RobotTransform.h
 * @brief Robot coordinate transform utilities
 */

#include <QiVision/Calib/QHomMat3d.h>
#include <QiVision/Core/Export.h>

namespace Qi::Vision::Calib {

QIVISION_API QHomMat3d ComposeTransform(const QHomMat3d& a, const QHomMat3d& b);
QIVISION_API QHomMat3d InvertTransform(const QHomMat3d& a);

} // namespace Qi::Vision::Calib
