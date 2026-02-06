#pragma once

/**
 * @file QPose.h
 * @brief 6DOF pose representation (ZYX Euler)
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Internal/Matrix.h>
#include <QiVision/Core/Export.h>

namespace Qi::Vision::Calib {

/**
 * @brief 6DOF pose (x,y,z + roll,pitch,yaw)
 * Euler order: ZYX (yaw, pitch, roll)
 */
struct QIVISION_API QPose {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double roll = 0.0;   ///< rotation around X
    double pitch = 0.0;  ///< rotation around Y
    double yaw = 0.0;    ///< rotation around Z

    QPose() = default;
    QPose(double x_, double y_, double z_, double roll_, double pitch_, double yaw_)
        : x(x_), y(y_), z(z_), roll(roll_), pitch(pitch_), yaw(yaw_) {}

    /// Convert to 4x4 homogeneous transform (world = T * local)
    Internal::Mat44 ToMatrix() const;

    /// Create pose from 4x4 homogeneous transform
    static QPose FromMatrix(const Internal::Mat44& T);
};

} // namespace Qi::Vision::Calib
