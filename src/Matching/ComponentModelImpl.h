#pragma once

#include <QiVision/Matching/ComponentModel.h>

#include <vector>

namespace Qi::Vision::Matching::Internal {

struct ComponentEntry {
    ComponentType type = ComponentType::Shape;
    ShapeModel shapeModel;
    NCCModel nccModel;
    SearchParams params;
    ComponentConstraint constraint;
    int32_t referenceIndex = -1;
};

class ComponentModelImpl {
public:
    ComponentModelImpl() = default;
    ~ComponentModelImpl() = default;

    ComponentModelImpl(const ComponentModelImpl&) = default;
    ComponentModelImpl& operator=(const ComponentModelImpl&) = default;
    ComponentModelImpl(ComponentModelImpl&&) noexcept = default;
    ComponentModelImpl& operator=(ComponentModelImpl&&) noexcept = default;

    void Clear();
    bool IsValid() const;

    int32_t AddComponent(const ShapeModel& model, const SearchParams& params);
    int32_t AddComponent(const NCCModel& model, const SearchParams& params);

    void SetRoot(int32_t index);
    void SetRelation(int32_t componentIndex, int32_t referenceIndex, const ComponentConstraint& constraint);

    int32_t GetCount() const { return static_cast<int32_t>(components_.size()); }
    int32_t GetRoot() const { return rootIndex_; }

    std::vector<ComponentMatch> Find(const QImage& image, double minScore, int32_t maxMatches) const;

private:
    std::vector<ComponentEntry> components_;
    int32_t rootIndex_ = 0;
};

} // namespace Qi::Vision::Matching::Internal
