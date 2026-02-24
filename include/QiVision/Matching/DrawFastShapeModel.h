#pragma once

/**
 * @file DrawFastShapeModel.h
 * @brief Drawing utilities for FastShapeModel match results.
 *
 * Provides DrawFastShapeModelResults which visualizes FindFastShapeModel output
 * directly: rotated bounding box + feature X-marks + model display contours.
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/Matching/FastShapeModel.h>

#include <vector>

namespace Qi::Vision::Matching {

/**
 * @brief Draw FastShapeModel match results onto an RGB image.
 *
 * For each match, draws:
 *  1. Rotated bounding rectangle (template outline)
 *  2. Green contour polylines from model display contours
 *  3. X-shaped cross marks at each feature point
 *
 * Parameters correspond directly to FindFastShapeModel outputs, so results
 * can be passed without any conversion:
 * @code
 *   std::vector<double> rows, cols, angles, scores, scales;
 *   FindFastShapeModel(image, model, minScore, maxMatches, overlap, greediness,
 *                      rows, cols, angles, scores, &scales);
 *   DrawFastShapeModelResults(vis, model, rows, cols, angles, scores, scales);
 * @endcode
 *
 * @param image      RGB image to draw on (must be 3-channel)
 * @param model      Model handle (provides feature points, contours, and template size)
 * @param rows       Match row positions (from FindFastShapeModel)
 * @param cols       Match column positions
 * @param angles     Match angles in radians
 * @param scores     Match scores (currently unused in drawing, reserved)
 * @param scales     Match scales (empty = all 1.0)
 * @param color      Drawing color (default green)
 * @param thickness  Line/marker thickness in pixels (default 1)
 */
QIVISION_API void DrawFastShapeModelResults(
    QImage& image,
    const FastShapeModel& model,
    const std::vector<double>& rows,
    const std::vector<double>& cols,
    const std::vector<double>& angles,
    const std::vector<double>& scores,
    const std::vector<double>& scales,
    const Scalar& color = Scalar::Green(),
    int32_t thickness = 1
);

} // namespace Qi::Vision::Matching
