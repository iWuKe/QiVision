#pragma once

/**
 * @file FastShapeModel.h
 * @brief Fast shape matching (independent pipeline, meiqua-style direction matching)
 *
 * This module is intentionally implemented independently from ShapeModel.
 * It targets high-speed rotation-invariant matching with a compact feature template.
 */

#include <QiVision/Core/Export.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Types.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace Qi::Vision::Matching {

namespace Internal {
class FastShapeModelImpl;
}

class QIVISION_API FastShapeModel {
public:
    FastShapeModel();
    ~FastShapeModel();
    FastShapeModel(const FastShapeModel& other);
    FastShapeModel(FastShapeModel&& other) noexcept;
    FastShapeModel& operator=(const FastShapeModel& other);
    FastShapeModel& operator=(FastShapeModel&& other) noexcept;

    bool IsValid() const;

    Internal::FastShapeModelImpl* Impl() { return impl_.get(); }
    const Internal::FastShapeModelImpl* Impl() const { return impl_.get(); }

private:
    std::unique_ptr<Internal::FastShapeModelImpl> impl_;
};

/**
 * @brief Strategy parameters aligned with line2Dup/LINEMOD.
 */
struct QIVISION_API FastShapeModelStrategy {
    int32_t numFeatures = 63;               ///< Feature count at level 0
    std::vector<int32_t> tAtLevel{4, 8};   ///< Pyramid step (T) per level
    double weakThreshold = 10.0;            ///< Search-side magnitude threshold
    double strongThreshold = 55.0;          ///< Template-side magnitude threshold
};

/**
 * @brief Create a fast shape model from rectangular ROI.
 *
 * @param image         Template/source image (grayscale)
 * @param roi           Template ROI
 * @param model         [out] Model handle
 * @param numLevels     Pyramid levels (0 = strategy.tAtLevel.size())
 * @param angleStart    Minimum search angle in radians
 * @param angleExtent   Search extent in radians
 * @param angleStep     Search step in radians (0 = auto)
 * @param strategy      line2Dup-style strategy (numFeatures/tAtLevel/weak/strong)
 */
QIVISION_API void CreateFastShapeModel(
    const QImage& image,
    const Rect2i& roi,
    FastShapeModel& model,
    int32_t numLevels,
    double angleStart,
    double angleExtent,
    double angleStep,
    const FastShapeModelStrategy& strategy = FastShapeModelStrategy()
);

/**
 * @brief Search fast shape model in image.
 *
 * @param image         Search image (grayscale)
 * @param model         Model handle
 * @param minScore      Minimum score [0,1]
 * @param numMatches    Max matches (0 = all)
 * @param maxOverlap    Max overlap [0,1]
 * @param greediness    Search pruning [0,1]
 * @param rows          [out] Match rows
 * @param cols          [out] Match cols
 * @param angles        [out] Match angles (rad)
 * @param scores        [out] Match scores [0,1]
 */
QIVISION_API void FindFastShapeModel(
    const QImage& image,
    const FastShapeModel& model,
    double minScore,
    int32_t numMatches,
    double maxOverlap,
    double greediness,
    std::vector<double>& rows,
    std::vector<double>& cols,
    std::vector<double>& angles,
    std::vector<double>& scores
);

/**
 * @brief Get model feature points at level 0 (template-relative coordinates).
 *
 * Returned points are centered at template origin (0,0). To visualize on match:
 * rotate by match angle and translate by (col,row).
 */
QIVISION_API void GetFastShapeModelFeaturePoints(
    const FastShapeModel& model,
    std::vector<Point2d>& points
);

/**
 * @brief Get template size stored in the model.
 */
QIVISION_API void GetFastShapeModelTemplateSize(
    const FastShapeModel& model,
    int32_t& width,
    int32_t& height
);

} // namespace Qi::Vision::Matching
