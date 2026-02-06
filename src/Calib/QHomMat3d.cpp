/**
 * @file QHomMat3d.cpp
 * @brief QHomMat3d implementation
 */

#include <QiVision/Calib/QHomMat3d.h>

namespace Qi::Vision::Calib {

QHomMat3d::QHomMat3d() : mat_(Internal::Mat44::Identity()) {}

QHomMat3d::QHomMat3d(const Internal::Mat44& mat) : mat_(mat) {}

QHomMat3d QHomMat3d::Identity() {
    return QHomMat3d(Internal::Mat44::Identity());
}

QHomMat3d QHomMat3d::Translation(double tx, double ty, double tz) {
    return QHomMat3d(Internal::Translation3D(tx, ty, tz));
}

QHomMat3d QHomMat3d::RotationZYX(double roll, double pitch, double yaw) {
    return QHomMat3d(Internal::RotationEulerZYX(roll, pitch, yaw));
}

QHomMat3d QHomMat3d::operator*(const QHomMat3d& other) const {
    return QHomMat3d(mat_ * other.mat_);
}

QHomMat3d QHomMat3d::InverseRigid() const {
    // Assume rigid transform: inverse = [R^T, -R^T t]
    Internal::Mat33 R;
    Internal::Vec3 t;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R(r, c) = mat_(r, c);
        }
        t[r] = mat_(r, 3);
    }

    Internal::Mat33 Rt = R.Transpose();
    Internal::Vec3 tInv = -(Rt * t);

    Internal::Mat44 inv = Internal::Mat44::Identity();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            inv(r, c) = Rt(r, c);
        }
        inv(r, 3) = tInv[r];
    }

    return QHomMat3d(inv);
}

Point3d QHomMat3d::TransformPoint(const Point3d& p) const {
    Internal::Vec4 v{p.x, p.y, p.z, 1.0};
    Internal::Vec4 r = mat_ * v;
    return Point3d(r[0], r[1], r[2]);
}

} // namespace Qi::Vision::Calib
