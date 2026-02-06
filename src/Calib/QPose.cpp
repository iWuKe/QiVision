/**
 * @file QPose.cpp
 * @brief QPose implementation
 */

#include <QiVision/Calib/QPose.h>
#include <QiVision/Internal/Matrix.h>

namespace Qi::Vision::Calib {

Internal::Mat44 QPose::ToMatrix() const {
    Internal::Mat44 R = Internal::RotationEulerZYX(roll, pitch, yaw);
    Internal::Mat44 T = Internal::Translation3D(x, y, z);
    return T * R;
}

QPose QPose::FromMatrix(const Internal::Mat44& T) {
    QPose pose;
    pose.x = T(0, 3);
    pose.y = T(1, 3);
    pose.z = T(2, 3);

    Internal::Mat33 R;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R(r, c) = T(r, c);
        }
    }

    Internal::Vec3 euler = Internal::ExtractEulerZYX(R);
    pose.roll = euler[0];
    pose.pitch = euler[1];
    pose.yaw = euler[2];
    return pose;
}

} // namespace Qi::Vision::Calib
