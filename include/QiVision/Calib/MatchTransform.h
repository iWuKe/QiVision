#pragma once

/**
 * @file MatchTransform.h
 * @brief Transform template matching results to world coordinates
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Calib/QHomMat3d.h>
#include <QiVision/Matching/MatchTypes.h>
#include <QiVision/Core/Export.h>

namespace Qi::Vision::Calib {

/**
 * @brief Convert match result (image coordinates) to world coordinates
 * @param match Match result (x,y in image)
 * @param imageToWorld 4x4 transform (image -> world)
 * @param z Image plane z (default 0)
 */
QIVISION_API Point3d MatchToWorld(
    const Matching::MatchResult& match,
    const QHomMat3d& imageToWorld,
    double z = 0.0
);

} // namespace Qi::Vision::Calib
