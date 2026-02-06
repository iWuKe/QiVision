#pragma once

/**
 * @file ComponentModel.h
 * @brief Multi-component template matching with spatial constraints
 *
 * ComponentModel combines multiple ShapeModel/NCCModel components and
 * enforces relative position/angle/scale constraints during matching.
 *
 * Typical usage:
 * - Add a root component (reference)
 * - Add child components with expected offsets and tolerances
 * - FindComponentModel to get grouped matches
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Core/Export.h>
#include <QiVision/Matching/MatchTypes.h>
#include <QiVision/Matching/ShapeModel.h>
#include <QiVision/Matching/NCCModel.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace Qi::Vision::Matching {

namespace Internal {
class ComponentModelImpl;
}

// =============================================================================
// Component Types
// =============================================================================

/**
 * @brief Component model type
 */
enum class ComponentType {
    Shape,      ///< ShapeModel component
    NCC         ///< NCCModel component
};

// =============================================================================
// Component Constraints and Results
// =============================================================================

/**
 * @brief Relative constraint for a component
 *
 * The offset is defined in the reference component's model coordinate system.
 * The expected position is obtained by applying the reference match transform
 * to the offset.
 */
struct QIVISION_API ComponentConstraint {
    Point2d offset{0.0, 0.0};   ///< Expected offset from reference component
    double angleOffset = 0.0;  ///< Expected relative angle [rad]
    double scale = 1.0;        ///< Expected relative scale (uniform)

    double positionTolerance = 0.0;  ///< Position tolerance (pixels, 0 = ignore)
    double angleTolerance = 0.0;     ///< Angle tolerance (radians, 0 = ignore)
    double scaleTolerance = 0.0;     ///< Scale tolerance (absolute, 0 = ignore)

    double weight = 1.0;       ///< Score weight for this component
};

/**
 * @brief Grouped match result for a ComponentModel
 */
struct QIVISION_API ComponentMatch {
    std::vector<MatchResult> components;  ///< Per-component match results
    double score = 0.0;                   ///< Aggregated score

    bool Empty() const { return components.empty(); }
    size_t Size() const { return components.size(); }
};

// =============================================================================
// ComponentModel Class (Model Handle)
// =============================================================================

class QIVISION_API ComponentModel {
public:
    ComponentModel();
    ~ComponentModel();
    ComponentModel(const ComponentModel& other);
    ComponentModel(ComponentModel&& other) noexcept;
    ComponentModel& operator=(const ComponentModel& other);
    ComponentModel& operator=(ComponentModel&& other) noexcept;

    bool IsValid() const;

    Internal::ComponentModelImpl* Impl() { return impl_.get(); }
    const Internal::ComponentModelImpl* Impl() const { return impl_.get(); }

private:
    friend class Internal::ComponentModelImpl;
    std::unique_ptr<Internal::ComponentModelImpl> impl_;
};

// =============================================================================
// Component Model API
// =============================================================================

/**
 * @brief Initialize/clear a component model
 */
QIVISION_API void CreateComponentModel(ComponentModel& model);

/**
 * @brief Add a ShapeModel component
 * @return Component index
 */
QIVISION_API int32_t AddComponent(
    ComponentModel& model,
    const ShapeModel& shapeModel,
    const SearchParams& params
);

/**
 * @brief Add an NCCModel component
 * @return Component index
 */
QIVISION_API int32_t AddComponent(
    ComponentModel& model,
    const NCCModel& nccModel,
    const SearchParams& params
);

/**
 * @brief Set the root (reference) component
 */
QIVISION_API void SetComponentRoot(ComponentModel& model, int32_t componentIndex);

/**
 * @brief Set relative constraint for a component (root-anchored)
 *
 * @param componentIndex Index of the constrained component
 * @param referenceIndex Reference component index (can be any component)
 * @param constraint     Relative constraint parameters
 */
QIVISION_API void SetComponentRelation(
    ComponentModel& model,
    int32_t componentIndex,
    int32_t referenceIndex,
    const ComponentConstraint& constraint
);

/**
 * @brief Find grouped matches for a component model
 *
 * @param image      Search image (grayscale)
 * @param model      Component model
 * @param minScore   Minimum aggregated score to accept
 * @param maxMatches Maximum number of grouped matches (0 = all)
 * @param matches    [out] Grouped matches
 */
QIVISION_API void FindComponentModel(
    const QImage& image,
    const ComponentModel& model,
    double minScore,
    int32_t maxMatches,
    std::vector<ComponentMatch>& matches
);

/**
 * @brief Clear a component model
 */
QIVISION_API void ClearComponentModel(ComponentModel& model);

/**
 * @brief Get number of components
 */
QIVISION_API int32_t GetComponentCount(const ComponentModel& model);

} // namespace Qi::Vision::Matching
