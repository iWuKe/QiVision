/**
 * @file ComponentModel.cpp
 * @brief ComponentModel API implementation
 */

#include "ComponentModelImpl.h"

#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>

#include <algorithm>
#include <cmath>
#include <functional>

namespace Qi::Vision::Matching {

namespace {

constexpr double kTwoPi = 6.28318530717958647692;

inline double WrapAngle(double angle) {
    while (angle > M_PI) angle -= kTwoPi;
    while (angle < -M_PI) angle += kTwoPi;
    return angle;
}

inline double AngleDiff(double a, double b) {
    return std::abs(WrapAngle(a - b));
}

std::string SubpixelToShapeString(SubpixelMethod method) {
    switch (method) {
        case SubpixelMethod::None: return "none";
        case SubpixelMethod::Parabolic: return "interpolation";
        case SubpixelMethod::LeastSquares: return "least_squares";
        case SubpixelMethod::LeastSquaresHigh: return "least_squares_high";
        case SubpixelMethod::LeastSquaresVeryHigh: return "least_squares_very_high";
        default: return "least_squares";
    }
}

std::string SubpixelToNccString(SubpixelMethod method) {
    switch (method) {
        case SubpixelMethod::None: return "false";
        case SubpixelMethod::Parabolic: return "interpolation";
        default: return "true";
    }
}

void ComputeAngleRange(const SearchParams& params, double& angleStart, double& angleExtent) {
    if (params.angleMode == AngleSearchMode::Full) {
        angleStart = 0.0;
        angleExtent = kTwoPi;
        return;
    }
    angleStart = params.angleStart;
    angleExtent = params.angleExtent;
}

std::vector<MatchResult> FindCandidates(const QImage& image, const Internal::ComponentEntry& entry) {
    std::vector<MatchResult> candidates;

    double angleStart = 0.0;
    double angleExtent = kTwoPi;
    ComputeAngleRange(entry.params, angleStart, angleExtent);

    const double minScore = entry.params.minScore;
    const int32_t maxMatches = entry.params.maxMatches;
    const double maxOverlap = entry.params.maxOverlap;
    const int32_t numLevels = entry.params.numLevels;

    if (entry.type == ComponentType::Shape) {
        std::vector<double> rows, cols, angles, scores, scales;
        if (entry.params.scaleMode == ScaleSearchMode::Uniform) {
            FindScaledShapeModel(image, entry.shapeModel,
                                 angleStart, angleExtent,
                                 entry.params.scaleMin, entry.params.scaleMax,
                                 minScore, maxMatches, maxOverlap,
                                 SubpixelToShapeString(entry.params.subpixelMethod),
                                 numLevels, entry.params.greediness,
                                 rows, cols, angles, scales, scores);
        } else {
            FindShapeModel(image, entry.shapeModel,
                           angleStart, angleExtent,
                           minScore, maxMatches, maxOverlap,
                           SubpixelToShapeString(entry.params.subpixelMethod),
                           numLevels, entry.params.greediness,
                           rows, cols, angles, scores);
        }

        const size_t count = scores.size();
        candidates.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            MatchResult m;
            m.y = rows[i];
            m.x = cols[i];
            m.angle = angles[i];
            if (entry.params.scaleMode == ScaleSearchMode::Uniform && i < scales.size()) {
                m.scaleX = scales[i];
                m.scaleY = scales[i];
            }
            m.score = scores[i];
            candidates.push_back(m);
        }
    } else {
        std::vector<double> rows, cols, angles, scores, scales;
        if (entry.params.scaleMode == ScaleSearchMode::Uniform) {
            FindScaledNCCModel(image, entry.nccModel,
                               angleStart, angleExtent,
                               entry.params.scaleMin, entry.params.scaleMax,
                               minScore, maxMatches, maxOverlap,
                               SubpixelToNccString(entry.params.subpixelMethod),
                               numLevels,
                               rows, cols, angles, scales, scores);
        } else {
            FindNCCModel(image, entry.nccModel,
                         angleStart, angleExtent,
                         minScore, maxMatches, maxOverlap,
                         SubpixelToNccString(entry.params.subpixelMethod),
                         numLevels,
                         rows, cols, angles, scores);
        }

        const size_t count = scores.size();
        candidates.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            MatchResult m;
            m.y = rows[i];
            m.x = cols[i];
            m.angle = angles[i];
            if (entry.params.scaleMode == ScaleSearchMode::Uniform && i < scales.size()) {
                m.scaleX = scales[i];
                m.scaleY = scales[i];
            }
            m.score = scores[i];
            candidates.push_back(m);
        }
    }

    return candidates;
}

MatchResult SelectBestCandidate(const std::vector<MatchResult>& candidates,
                                const MatchResult& reference,
                                const ComponentConstraint& constraint,
                                bool& found) {
    found = false;
    MatchResult best;

    const Point2d expected = reference.TransformPoint(constraint.offset);
    const double expectedAngle = reference.angle + constraint.angleOffset;
    const double expectedScale = reference.scaleX * constraint.scale;

    double bestScore = -1.0;
    for (const auto& cand : candidates) {
        if (constraint.positionTolerance > 0.0) {
            const double dx = cand.x - expected.x;
            const double dy = cand.y - expected.y;
            const double dist = std::sqrt(dx * dx + dy * dy);
            if (dist > constraint.positionTolerance) {
                continue;
            }
        }
        if (constraint.angleTolerance > 0.0) {
            if (AngleDiff(cand.angle, expectedAngle) > constraint.angleTolerance) {
                continue;
            }
        }
        if (constraint.scaleTolerance > 0.0) {
            if (std::abs(cand.scaleX - expectedScale) > constraint.scaleTolerance) {
                continue;
            }
        }

        if (cand.score > bestScore) {
            best = cand;
            bestScore = cand.score;
            found = true;
        }
    }

    return best;
}

bool BuildTraversalOrder(int32_t root,
                         const std::vector<int32_t>& parent,
                         std::vector<int32_t>& order) {
    const int32_t count = static_cast<int32_t>(parent.size());
    order.clear();
    order.reserve(static_cast<size_t>(count));

    enum class State { Unvisited, Visiting, Done };
    std::vector<State> state(static_cast<size_t>(count), State::Unvisited);

    std::function<bool(int32_t)> dfs = [&](int32_t node) -> bool {
        State& st = state[static_cast<size_t>(node)];
        if (st == State::Visiting) {
            return false; // cycle
        }
        if (st == State::Done) {
            return true;
        }
        st = State::Visiting;

        int32_t p = parent[static_cast<size_t>(node)];
        if (node != root) {
            if (p < 0 || p >= count) {
                return false;
            }
            if (!dfs(p)) {
                return false;
            }
        }

        st = State::Done;
        order.push_back(node);
        return true;
    };

    for (int32_t i = 0; i < count; ++i) {
        if (!dfs(i)) {
            return false;
        }
    }

    return true;
}

} // namespace

namespace Internal {

void ComponentModelImpl::Clear() {
    components_.clear();
    rootIndex_ = 0;
}

bool ComponentModelImpl::IsValid() const {
    if (components_.empty()) {
        return false;
    }
    for (const auto& entry : components_) {
        if (entry.type == ComponentType::Shape) {
            if (!entry.shapeModel.IsValid()) return false;
        } else {
            if (!entry.nccModel.IsValid()) return false;
        }
    }
    return true;
}

int32_t ComponentModelImpl::AddComponent(const ShapeModel& model, const SearchParams& params) {
    if (!model.IsValid()) {
        throw InvalidArgumentException("AddComponent: ShapeModel is not valid");
    }
    ComponentEntry entry;
    entry.type = ComponentType::Shape;
    entry.shapeModel = model;
    entry.params = params;
    entry.referenceIndex = -1;
    components_.push_back(entry);
    return static_cast<int32_t>(components_.size() - 1);
}

int32_t ComponentModelImpl::AddComponent(const NCCModel& model, const SearchParams& params) {
    if (!model.IsValid()) {
        throw InvalidArgumentException("AddComponent: NCCModel is not valid");
    }
    ComponentEntry entry;
    entry.type = ComponentType::NCC;
    entry.nccModel = model;
    entry.params = params;
    entry.referenceIndex = -1;
    components_.push_back(entry);
    return static_cast<int32_t>(components_.size() - 1);
}

void ComponentModelImpl::SetRoot(int32_t index) {
    if (index < 0 || index >= static_cast<int32_t>(components_.size())) {
        throw InvalidArgumentException("SetComponentRoot: index out of range");
    }
    rootIndex_ = index;
}

void ComponentModelImpl::SetRelation(int32_t componentIndex, int32_t referenceIndex,
                                     const ComponentConstraint& constraint) {
    if (componentIndex < 0 || componentIndex >= static_cast<int32_t>(components_.size())) {
        throw InvalidArgumentException("SetComponentRelation: componentIndex out of range");
    }
    if (referenceIndex < 0 || referenceIndex >= static_cast<int32_t>(components_.size())) {
        throw InvalidArgumentException("SetComponentRelation: referenceIndex out of range");
    }
    if (componentIndex == referenceIndex) {
        throw InvalidArgumentException("SetComponentRelation: componentIndex == referenceIndex");
    }

    components_[componentIndex].referenceIndex = referenceIndex;
    components_[componentIndex].constraint = constraint;
}

std::vector<ComponentMatch> ComponentModelImpl::Find(const QImage& image,
                                                     double minScore,
                                                     int32_t maxMatches) const {
    if (components_.empty()) {
        return {};
    }

    if (!Validate::RequireImageValid(image, __func__)) {
        return {};
    }

    const int32_t count = static_cast<int32_t>(components_.size());
    int32_t root = rootIndex_;
    if (root < 0 || root >= count) {
        root = 0;
    }

    std::vector<std::vector<MatchResult>> candidateSets;
    candidateSets.reserve(static_cast<size_t>(count));
    for (const auto& entry : components_) {
        candidateSets.push_back(FindCandidates(image, entry));
    }

    if (candidateSets[root].empty()) {
        return {};
    }

    std::vector<int32_t> parent(static_cast<size_t>(count), root);
    for (int32_t i = 0; i < count; ++i) {
        if (i == root) {
            parent[static_cast<size_t>(i)] = root;
            continue;
        }
        int32_t ref = components_[static_cast<size_t>(i)].referenceIndex;
        parent[static_cast<size_t>(i)] = (ref >= 0) ? ref : root;
    }

    std::vector<int32_t> order;
    if (!BuildTraversalOrder(root, parent, order)) {
        throw InvalidArgumentException("FindComponentModel: invalid component relations (cycle or missing parent)");
    }

    std::vector<ComponentMatch> results;
    for (const auto& rootMatch : candidateSets[root]) {
        ComponentMatch group;
        group.components.resize(static_cast<size_t>(count));
        group.components[static_cast<size_t>(root)] = rootMatch;

        bool ok = true;
        double scoreSum = 0.0;
        double weightSum = 0.0;

        const double rootWeight = std::max(components_[root].constraint.weight, 0.0);
        scoreSum += rootWeight * rootMatch.score;
        weightSum += rootWeight;

        for (int32_t node : order) {
            if (node == root) {
                continue;
            }
            const auto& entry = components_[static_cast<size_t>(node)];
            const int32_t referenceIndex = parent[static_cast<size_t>(node)];

            bool found = false;
            MatchResult selected = SelectBestCandidate(candidateSets[static_cast<size_t>(node)],
                                                       group.components[static_cast<size_t>(referenceIndex)],
                                                       entry.constraint,
                                                       found);
            if (!found) {
                ok = false;
                break;
            }

            group.components[static_cast<size_t>(node)] = selected;
            const double weight = std::max(entry.constraint.weight, 0.0);
            scoreSum += weight * selected.score;
            weightSum += weight;
        }

        if (!ok) {
            continue;
        }

        if (weightSum > 0.0) {
            group.score = scoreSum / weightSum;
        }

        if (group.score >= minScore) {
            results.push_back(group);
        }
    }

    std::sort(results.begin(), results.end(),
              [](const ComponentMatch& a, const ComponentMatch& b) { return a.score > b.score; });

    if (maxMatches > 0 && static_cast<int32_t>(results.size()) > maxMatches) {
        results.resize(static_cast<size_t>(maxMatches));
    }

    return results;
}

} // namespace Internal

// =============================================================================
// Public API
// =============================================================================

ComponentModel::ComponentModel() : impl_(std::make_unique<Internal::ComponentModelImpl>()) {}
ComponentModel::~ComponentModel() = default;
ComponentModel::ComponentModel(const ComponentModel& other)
    : impl_(other.impl_ ? std::make_unique<Internal::ComponentModelImpl>(*other.impl_) : nullptr) {}
ComponentModel::ComponentModel(ComponentModel&& other) noexcept = default;
ComponentModel& ComponentModel::operator=(const ComponentModel& other) {
    if (this != &other) {
        impl_ = other.impl_ ? std::make_unique<Internal::ComponentModelImpl>(*other.impl_) : nullptr;
    }
    return *this;
}
ComponentModel& ComponentModel::operator=(ComponentModel&& other) noexcept = default;

bool ComponentModel::IsValid() const {
    return impl_ && impl_->IsValid();
}

void CreateComponentModel(ComponentModel& model) {
    if (!model.Impl()) {
        model = ComponentModel();
    }
    model.Impl()->Clear();
}

int32_t AddComponent(ComponentModel& model, const ShapeModel& shapeModel, const SearchParams& params) {
    if (!model.Impl()) {
        model = ComponentModel();
    }
    return model.Impl()->AddComponent(shapeModel, params);
}

int32_t AddComponent(ComponentModel& model, const NCCModel& nccModel, const SearchParams& params) {
    if (!model.Impl()) {
        model = ComponentModel();
    }
    return model.Impl()->AddComponent(nccModel, params);
}

void SetComponentRoot(ComponentModel& model, int32_t componentIndex) {
    if (!model.Impl()) {
        throw InvalidArgumentException("SetComponentRoot: model is not initialized");
    }
    model.Impl()->SetRoot(componentIndex);
}

void SetComponentRelation(ComponentModel& model, int32_t componentIndex, int32_t referenceIndex,
                          const ComponentConstraint& constraint) {
    if (!model.Impl()) {
        throw InvalidArgumentException("SetComponentRelation: model is not initialized");
    }
    model.Impl()->SetRelation(componentIndex, referenceIndex, constraint);
}

void FindComponentModel(const QImage& image, const ComponentModel& model,
                        double minScore, int32_t maxMatches,
                        std::vector<ComponentMatch>& matches) {
    matches.clear();
    if (!model.Impl()) {
        return;
    }
    matches = model.Impl()->Find(image, minScore, maxMatches);
}

void ClearComponentModel(ComponentModel& model) {
    if (!model.Impl()) {
        model = ComponentModel();
    }
    model.Impl()->Clear();
}

int32_t GetComponentCount(const ComponentModel& model) {
    if (!model.Impl()) {
        return 0;
    }
    return model.Impl()->GetCount();
}

} // namespace Qi::Vision::Matching
