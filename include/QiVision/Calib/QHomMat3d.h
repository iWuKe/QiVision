#pragma once

/**
 * @file QHomMat3d.h
 * @brief 3D homogeneous transformation matrix (4x4)
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Internal/Matrix.h>
#include <QiVision/Core/Export.h>

namespace Qi::Vision::Calib {

class QIVISION_API QHomMat3d {
public:
    QHomMat3d();
    explicit QHomMat3d(const Internal::Mat44& mat);

    static QHomMat3d Identity();
    static QHomMat3d Translation(double tx, double ty, double tz);
    static QHomMat3d RotationZYX(double roll, double pitch, double yaw);

    const Internal::Mat44& Matrix() const { return mat_; }
    Internal::Mat44& Matrix() { return mat_; }

    QHomMat3d operator*(const QHomMat3d& other) const;

    QHomMat3d InverseRigid() const;

    Point3d TransformPoint(const Point3d& p) const;

private:
    Internal::Mat44 mat_;
};

} // namespace Qi::Vision::Calib
