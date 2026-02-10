#pragma once

/**
 * @file FastShapeDetector.h
 * @brief Multi-template detector built on FastShapeModel.
 */

#include <QiVision/Core/Export.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Matching/FastShapeModel.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace Qi::Vision::Matching {

namespace Internal {
class FastShapeDetectorImpl;
}

class QIVISION_API FastShapeDetector {
public:
    FastShapeDetector();
    ~FastShapeDetector();
    FastShapeDetector(const FastShapeDetector& other);
    FastShapeDetector(FastShapeDetector&& other) noexcept;
    FastShapeDetector& operator=(const FastShapeDetector& other);
    FastShapeDetector& operator=(FastShapeDetector&& other) noexcept;

    bool IsValid() const;

    Internal::FastShapeDetectorImpl* Impl() { return impl_.get(); }
    const Internal::FastShapeDetectorImpl* Impl() const { return impl_.get(); }

private:
    std::unique_ptr<Internal::FastShapeDetectorImpl> impl_;
};

struct QIVISION_API FastShapeDetection {
    int32_t templateId = -1;     ///< Matched template ID in detector
    double row = 0.0;            ///< Match center row
    double col = 0.0;            ///< Match center col
    double angle = 0.0;          ///< Match angle (rad)
    double score = 0.0;          ///< Match score [0,1]
    int32_t templateWidth = 0;   ///< Template width in pixels
    int32_t templateHeight = 0;  ///< Template height in pixels
};

/**
 * @brief Initialize detector and clear all templates.
 */
QIVISION_API void CreateFastShapeDetector(FastShapeDetector& detector);

/**
 * @brief Build a template from image ROI and add to detector.
 * @return Assigned template ID.
 */
QIVISION_API int32_t AddFastShapeTemplate(
    FastShapeDetector& detector,
    const QImage& image,
    const Rect2i& roi,
    int32_t numLevels,
    double angleStart,
    double angleExtent,
    double angleStep,
    const FastShapeModelStrategy& strategy = FastShapeModelStrategy()
);

/**
 * @brief Add an existing model as template.
 * @return Assigned template ID.
 */
QIVISION_API int32_t AddFastShapeTemplate(
    FastShapeDetector& detector,
    const FastShapeModel& model
);

/**
 * @brief Remove template by ID.
 */
QIVISION_API void RemoveFastShapeTemplate(
    FastShapeDetector& detector,
    int32_t templateId
);

/**
 * @brief Clear all templates and reset ID allocator.
 */
QIVISION_API void ClearFastShapeDetector(FastShapeDetector& detector);

/**
 * @brief Get current template count.
 */
QIVISION_API int32_t GetFastShapeTemplateCount(const FastShapeDetector& detector);

/**
 * @brief Match all templates and return merged detections.
 *
 * @param minScore      Minimum score [0,1]
 * @param numMatches    Maximum output count (0 = all)
 * @param maxOverlap    Overlap threshold [0,1]
 * @param greediness    Pruning strength [0,1]
 */
QIVISION_API void FindFastShapeDetector(
    const QImage& image,
    const FastShapeDetector& detector,
    double minScore,
    int32_t numMatches,
    double maxOverlap,
    double greediness,
    std::vector<FastShapeDetection>& detections
);

} // namespace Qi::Vision::Matching
