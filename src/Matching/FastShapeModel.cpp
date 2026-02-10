#include <QiVision/Matching/FastShapeModel.h>

#include <QiVision/Core/Exception.h>
#include <QiVision/Internal/LinemodPyramid.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace Qi::Vision::Matching {

namespace {

constexpr double K_PI = 3.14159265358979323846;
constexpr int32_t K_NEIGHBOR_THRESHOLD = 5;
constexpr double K_SMOOTH_SIGMA = 1.0;
constexpr float K_FEATURE_MIN_DISTANCE = 2.0f;

constexpr double K_COARSE_MIN_SCORE_FLOOR = 0.40;
constexpr double K_COARSE_MIN_SCORE_SCALE = 0.68;
constexpr double K_COARSE_DIVERSITY_DIST_SCALE = 0.10;
constexpr double K_COARSE_DIVERSITY_ANGLE_DEG = 8.0;
constexpr int32_t K_COARSE_CANDIDATE_CAP = 420;
constexpr double K_FINE_ANGLE_RADIUS_DEG = 5.0;
constexpr double K_FINE_ANGLE_STEP_DEG = 2.0;
constexpr int32_t K_REFINE_SEARCH_RADIUS_FINE = 3;
constexpr int32_t K_REFINE_SEARCH_RADIUS_COARSE = 2;
constexpr double K_REFINE_ANGLE_RADIUS_FINE = 0.05;
constexpr double K_REFINE_ANGLE_RADIUS_COARSE = 0.10;
constexpr double K_REFINE_ANGLE_STEP_FINE = 0.01;
constexpr double K_REFINE_ANGLE_STEP_COARSE = 0.025;
constexpr int32_t K_REFINE_CANDIDATE_CAP_FINE = 120;
constexpr int32_t K_REFINE_CANDIDATE_CAP_COARSE = 240;
constexpr double K_REFINE_ACCEPT_SCALE = 0.72;
constexpr double K_REFINE_DIVERSITY_DIST_SCALE = 0.08;
constexpr double K_REFINE_DIVERSITY_ANGLE_DEG = 6.0;
constexpr double K_NMS_MIN_DIST_SCALE = 0.20;
constexpr double K_NMS_OVERLAP_MIN_AREA = 0.45;

struct Candidate {
    int32_t x = 0;
    int32_t y = 0;
    double angle = 0.0;
    double score = 0.0;
};

inline double NormalizeAngle(double a) {
    while (a < 0.0) a += 2.0 * K_PI;
    while (a >= 2.0 * K_PI) a -= 2.0 * K_PI;
    return a;
}

inline double AngleDistance(double a, double b) {
    double d = std::abs(NormalizeAngle(a) - NormalizeAngle(b));
    if (d > K_PI) d = 2.0 * K_PI - d;
    return d;
}

void ComputeFeatureBounds(const std::vector<Qi::Vision::Internal::LinemodFeature>& features,
                          int32_t& minX, int32_t& maxX,
                          int32_t& minY, int32_t& maxY) {
    if (features.empty()) {
        minX = 0;
        maxX = -1;
        minY = 0;
        maxY = -1;
        return;
    }

    minX = std::numeric_limits<int32_t>::max();
    maxX = std::numeric_limits<int32_t>::min();
    minY = std::numeric_limits<int32_t>::max();
    maxY = std::numeric_limits<int32_t>::min();

    for (const auto& f : features) {
        minX = std::min(minX, static_cast<int32_t>(f.x));
        maxX = std::max(maxX, static_cast<int32_t>(f.x));
        minY = std::min(minY, static_cast<int32_t>(f.y));
        maxY = std::max(maxY, static_cast<int32_t>(f.y));
    }
}

double ComputeIoUAxisAligned(const Candidate& a, const Candidate& b, double w, double h) {
    double ax1 = a.x - w * 0.5;
    double ay1 = a.y - h * 0.5;
    double ax2 = a.x + w * 0.5;
    double ay2 = a.y + h * 0.5;
    double bx1 = b.x - w * 0.5;
    double by1 = b.y - h * 0.5;
    double bx2 = b.x + w * 0.5;
    double by2 = b.y + h * 0.5;

    double ix1 = std::max(ax1, bx1);
    double iy1 = std::max(ay1, by1);
    double ix2 = std::min(ax2, bx2);
    double iy2 = std::min(ay2, by2);

    if (ix2 <= ix1 || iy2 <= iy1) return 0.0;
    double inter = (ix2 - ix1) * (iy2 - iy1);
    double areaA = (ax2 - ax1) * (ay2 - ay1);
    double areaB = (bx2 - bx1) * (by2 - by1);
    double uni = areaA + areaB - inter;
    return (uni > 1e-9) ? (inter / uni) : 0.0;
}

double ComputeOverlapMinArea(const Candidate& a, const Candidate& b, double w, double h) {
    double ax1 = a.x - w * 0.5;
    double ay1 = a.y - h * 0.5;
    double ax2 = a.x + w * 0.5;
    double ay2 = a.y + h * 0.5;
    double bx1 = b.x - w * 0.5;
    double by1 = b.y - h * 0.5;
    double bx2 = b.x + w * 0.5;
    double by2 = b.y + h * 0.5;

    double ix1 = std::max(ax1, bx1);
    double iy1 = std::max(ay1, by1);
    double ix2 = std::min(ax2, bx2);
    double iy2 = std::min(ay2, by2);

    if (ix2 <= ix1 || iy2 <= iy1) return 0.0;
    double inter = (ix2 - ix1) * (iy2 - iy1);
    double area = w * h;
    return (area > 1e-9) ? (inter / area) : 0.0;
}

void ApplyNMS(std::vector<Candidate>& cands,
              double maxOverlap,
              double modelW,
              double modelH,
              double nmsMinDistScale,
              double nmsOverlapMinArea) {
    std::sort(cands.begin(), cands.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    std::vector<Candidate> keep;
    keep.reserve(cands.size());

    // LINEMOD archived search used distance-based NMS (~10 px). Keep this behavior
    // as primary suppression to avoid duplicate detections on the same object.
    const double diag = std::sqrt(modelW * modelW + modelH * modelH);
    const double minDist = std::max(10.0, diag * nmsMinDistScale);
    const double minDistSq = minDist * minDist;

    for (const auto& c : cands) {
        bool suppressed = false;
        for (const auto& k : keep) {
            double dx = static_cast<double>(c.x - k.x);
            double dy = static_cast<double>(c.y - k.y);
            if (dx * dx + dy * dy < minDistSq) {
                suppressed = true;
                break;
            }

            // Keep overlap-based check as secondary rule for close-size duplicates.
            if (ComputeIoUAxisAligned(c, k, modelW, modelH) > maxOverlap) {
                suppressed = true;
                break;
            }
            // Additional suppression for near-duplicate detections even when IoU is
            // not high enough under very loose maxOverlap settings.
            if (ComputeOverlapMinArea(c, k, modelW, modelH) > nmsOverlapMinArea) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            keep.push_back(c);
        }
    }

    cands.swap(keep);
}

void KeepTopDiverse(std::vector<Candidate>& cands,
                    size_t cap,
                    double minDist,
                    double minAngleDiff) {
    if (cands.empty()) return;
    std::sort(cands.begin(), cands.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    if (cap == 0) {
        cands.clear();
        return;
    }

    const double minDistSq = minDist * minDist;
    std::vector<Candidate> keep;
    keep.reserve(std::min(cap, cands.size()));

    for (const auto& c : cands) {
        bool nearDuplicate = false;
        for (const auto& k : keep) {
            double dx = static_cast<double>(c.x - k.x);
            double dy = static_cast<double>(c.y - k.y);
            if (dx * dx + dy * dy < minDistSq &&
                AngleDistance(c.angle, k.angle) < minAngleDiff) {
                nearDuplicate = true;
                break;
            }
        }
        if (!nearDuplicate) {
            keep.push_back(c);
            if (keep.size() >= cap) break;
        }
    }

    cands.swap(keep);
}

} // namespace

namespace Internal {

class FastShapeModelImpl {
public:
    bool valid = false;
    int32_t numLevels = 0;
    int32_t templateWidth = 0;
    int32_t templateHeight = 0;

    double angleStart = 0.0;
    double angleExtent = 2.0 * K_PI;
    double coarseAngleStep = K_PI / 18.0; // 10 degrees

    FastShapeModelStrategy strategy;
    float weakThreshold = 10.0f;
    float strongThreshold = 55.0f;
    int32_t numFeatures = 63;

    std::vector<std::vector<Qi::Vision::Internal::LinemodFeature>> levelFeatures;
    std::vector<double> templateAngles;
    std::vector<std::vector<std::vector<Qi::Vision::Internal::LinemodFeature>>> rotatedTemplates;

    std::unique_ptr<FastShapeModelImpl> Clone() const {
        auto out = std::make_unique<FastShapeModelImpl>();
        *out = *this;
        return out;
    }
};

} // namespace Internal

// =============================================================================
// FastShapeModel handle
// =============================================================================

FastShapeModel::FastShapeModel() : impl_(std::make_unique<Internal::FastShapeModelImpl>()) {}
FastShapeModel::~FastShapeModel() = default;
FastShapeModel::FastShapeModel(const FastShapeModel& other)
    : impl_(other.impl_ ? other.impl_->Clone() : std::make_unique<Internal::FastShapeModelImpl>()) {}
FastShapeModel::FastShapeModel(FastShapeModel&& other) noexcept = default;
FastShapeModel& FastShapeModel::operator=(const FastShapeModel& other) {
    if (this != &other) {
        impl_ = other.impl_ ? other.impl_->Clone() : std::make_unique<Internal::FastShapeModelImpl>();
    }
    return *this;
}
FastShapeModel& FastShapeModel::operator=(FastShapeModel&& other) noexcept = default;
bool FastShapeModel::IsValid() const { return impl_ && impl_->valid; }

// =============================================================================
// Public API
// =============================================================================

void CreateFastShapeModel(
    const QImage& image,
    const Rect2i& roi,
    FastShapeModel& model,
    int32_t numLevels,
    double angleStart,
    double angleExtent,
    double angleStep,
    const FastShapeModelStrategy& strategy)
{
    if (image.Empty()) {
        throw InvalidArgumentException("CreateFastShapeModel: image is empty");
    }
    if (!roi.IsValid() || roi.width <= 0 || roi.height <= 0) {
        throw InvalidArgumentException("CreateFastShapeModel: invalid roi");
    }
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > image.Width() || roi.y + roi.height > image.Height()) {
        throw InvalidArgumentException("CreateFastShapeModel: roi out of image bounds");
    }
    if (strategy.weakThreshold <= 0.0 || strategy.strongThreshold <= 0.0) {
        throw InvalidArgumentException("CreateFastShapeModel: weak/strong threshold must be > 0");
    }
    if (strategy.numFeatures < 16) {
        throw InvalidArgumentException("CreateFastShapeModel: numFeatures must be >= 16");
    }
    if (strategy.tAtLevel.empty()) {
        throw InvalidArgumentException("CreateFastShapeModel: tAtLevel must not be empty");
    }

    FastShapeModelStrategy st = strategy;
    st.numFeatures = std::max(16, st.numFeatures);
    for (auto& t : st.tAtLevel) {
        t = std::max(1, t);
    }
    st.weakThreshold = std::max(1.0, st.weakThreshold);
    st.strongThreshold = std::max(st.weakThreshold, st.strongThreshold);

    QImage templ = image.SubImage(roi.x, roi.y, roi.width, roi.height);
    if (templ.Empty()) {
        throw InvalidArgumentException("CreateFastShapeModel: failed to extract template roi");
    }

    auto* impl = model.Impl();
    impl->valid = false;
    impl->templateWidth = roi.width;
    impl->templateHeight = roi.height;
    impl->angleStart = angleStart;
    impl->angleExtent = (angleExtent > 0.0) ? angleExtent : (2.0 * K_PI);

    if (numLevels <= 0) {
        numLevels = static_cast<int32_t>(st.tAtLevel.size());
    }
    impl->numLevels = std::max(1, numLevels);
    if (static_cast<int32_t>(st.tAtLevel.size()) < impl->numLevels) {
        int32_t fillT = st.tAtLevel.empty() ? 4 : st.tAtLevel.back();
        st.tAtLevel.resize(static_cast<size_t>(impl->numLevels), std::max(1, fillT));
    } else if (static_cast<int32_t>(st.tAtLevel.size()) > impl->numLevels) {
        st.tAtLevel.resize(static_cast<size_t>(impl->numLevels));
    }
    impl->strategy = st;

    if (angleStep <= 0.0) {
        angleStep = K_PI / 18.0; // 10 degrees, aligned with coarse template strategy
    }
    impl->coarseAngleStep = std::max(angleStep, K_PI / 180.0);

    impl->weakThreshold = static_cast<float>(st.weakThreshold);
    impl->strongThreshold = static_cast<float>(st.strongThreshold);
    impl->numFeatures = st.numFeatures;

    Qi::Vision::Internal::LinemodPyramid templatePyramid;
    Qi::Vision::Internal::LinemodPyramidParams pyramidParams;
    pyramidParams.numLevels = impl->numLevels;
    pyramidParams.minMagnitude = impl->strongThreshold;
    pyramidParams.spreadT = st.tAtLevel.front();
    pyramidParams.neighborThreshold = K_NEIGHBOR_THRESHOLD;
    pyramidParams.smoothSigma = K_SMOOTH_SIGMA;
    pyramidParams.extractFeatures = true;

    if (!templatePyramid.Build(templ, pyramidParams)) {
        throw InvalidArgumentException("CreateFastShapeModel: failed to build LINEMOD pyramid");
    }

    impl->numLevels = templatePyramid.NumLevels();
    if (static_cast<int32_t>(impl->strategy.tAtLevel.size()) < impl->numLevels) {
        int32_t fillT = impl->strategy.tAtLevel.empty() ? 4 : impl->strategy.tAtLevel.back();
        impl->strategy.tAtLevel.resize(static_cast<size_t>(impl->numLevels), std::max(1, fillT));
    } else if (static_cast<int32_t>(impl->strategy.tAtLevel.size()) > impl->numLevels) {
        impl->strategy.tAtLevel.resize(static_cast<size_t>(impl->numLevels));
    }
    impl->levelFeatures.assign(static_cast<size_t>(impl->numLevels), {});

    for (int32_t level = 0; level < impl->numLevels; ++level) {
        int32_t levelMaxFeatures = std::max(16, impl->numFeatures >> level);
        impl->levelFeatures[level] = templatePyramid.ExtractFeatures(
            level, Rect2i(), levelMaxFeatures, K_FEATURE_MIN_DISTANCE);
    }

    if (impl->levelFeatures.empty() || impl->levelFeatures[0].size() < 24) {
        throw InvalidArgumentException("CreateFastShapeModel: insufficient template features");
    }

    impl->templateAngles.clear();
    for (double angle = impl->angleStart;
         angle <= impl->angleStart + impl->angleExtent + 1e-9;
         angle += impl->coarseAngleStep) {
        impl->templateAngles.push_back(angle);
    }
    if (impl->templateAngles.empty()) {
        impl->templateAngles.push_back(impl->angleStart);
    }

    impl->rotatedTemplates.clear();
    impl->rotatedTemplates.resize(impl->templateAngles.size());
    for (size_t ai = 0; ai < impl->templateAngles.size(); ++ai) {
        auto& byLevel = impl->rotatedTemplates[ai];
        byLevel.resize(static_cast<size_t>(impl->numLevels));
        for (int32_t level = 0; level < impl->numLevels; ++level) {
            byLevel[static_cast<size_t>(level)] =
                Qi::Vision::Internal::LinemodPyramid::RotateFeatures(
                    impl->levelFeatures[static_cast<size_t>(level)],
                    impl->templateAngles[ai]);
        }
    }

    impl->valid = true;
}

void FindFastShapeModel(
    const QImage& image,
    const FastShapeModel& model,
    double minScore,
    int32_t numMatches,
    double maxOverlap,
    double greediness,
    std::vector<double>& rows,
    std::vector<double>& cols,
    std::vector<double>& angles,
    std::vector<double>& scores)
{
    rows.clear();
    cols.clear();
    angles.clear();
    scores.clear();

    if (image.Empty()) {
        throw InvalidArgumentException("FindFastShapeModel: image is empty");
    }
    if (!model.IsValid() || model.Impl() == nullptr || !model.Impl()->valid) {
        throw InvalidArgumentException("FindFastShapeModel: invalid model");
    }
    if (minScore < 0.0 || minScore > 1.0) {
        throw InvalidArgumentException("FindFastShapeModel: minScore must be in [0,1]");
    }
    if (numMatches < 0) {
        throw InvalidArgumentException("FindFastShapeModel: numMatches must be >= 0");
    }
    if (maxOverlap < 0.0 || maxOverlap > 1.0) {
        throw InvalidArgumentException("FindFastShapeModel: maxOverlap must be in [0,1]");
    }
    greediness = std::clamp(greediness, 0.0, 1.0);

    const auto* impl = model.Impl();
    const auto& st = impl->strategy;
    Qi::Vision::Internal::LinemodPyramid targetPyramid;
    Qi::Vision::Internal::LinemodPyramidParams targetParams;
    targetParams.numLevels = impl->numLevels;
    targetParams.minMagnitude = impl->weakThreshold;
    targetParams.spreadT = st.tAtLevel.empty() ? 4 : st.tAtLevel.front();
    targetParams.neighborThreshold = K_NEIGHBOR_THRESHOLD;
    targetParams.smoothSigma = K_SMOOTH_SIGMA;
    targetParams.extractFeatures = false;

    if (!targetPyramid.Build(image, targetParams)) {
        throw InvalidArgumentException("FindFastShapeModel: failed to build target LINEMOD pyramid");
    }

    const int32_t levels = std::min<int32_t>(impl->numLevels, targetPyramid.NumLevels());
    if (levels <= 0) {
        return;
    }

    int32_t startLevel = levels - 1;
    while (startLevel > 0 &&
           (startLevel >= static_cast<int32_t>(impl->levelFeatures.size()) ||
            static_cast<int32_t>(impl->levelFeatures[static_cast<size_t>(startLevel)].size()) < 32)) {
        startLevel--;
    }

    if (startLevel < 0 ||
        startLevel >= static_cast<int32_t>(impl->levelFeatures.size()) ||
        impl->levelFeatures[static_cast<size_t>(startLevel)].empty()) {
        return;
    }

    int32_t kStepSize = 4;
    if (!st.tAtLevel.empty()) {
        int32_t stepIndex = std::clamp(startLevel, 0, static_cast<int32_t>(st.tAtLevel.size()) - 1);
        kStepSize = std::max(1, st.tAtLevel[static_cast<size_t>(stepIndex)]);
    }
    const double coarseThreshold = std::max(
        K_COARSE_MIN_SCORE_FLOOR,
        minScore * (K_COARSE_MIN_SCORE_SCALE + 0.08 * greediness));

    std::vector<Candidate> candidates;
    candidates.reserve(2048);

    for (size_t ai = 0; ai < impl->templateAngles.size(); ++ai) {
        if (ai >= impl->rotatedTemplates.size() ||
            startLevel >= static_cast<int32_t>(impl->rotatedTemplates[ai].size())) {
            continue;
        }

        const auto& rotatedFeatures = impl->rotatedTemplates[ai][static_cast<size_t>(startLevel)];
        if (rotatedFeatures.empty()) {
            continue;
        }

        int32_t rotMinX = 0, rotMaxX = 0, rotMinY = 0, rotMaxY = 0;
        ComputeFeatureBounds(rotatedFeatures, rotMinX, rotMaxX, rotMinY, rotMaxY);
        if (rotMinX > rotMaxX || rotMinY > rotMaxY) {
            continue;
        }

        int32_t targetW = targetPyramid.GetWidth(startLevel);
        int32_t targetH = targetPyramid.GetHeight(startLevel);

        int32_t searchXMin = std::max(0, -rotMinX);
        int32_t searchXMax = std::min(targetW - 1, targetW - 1 - rotMaxX);
        int32_t searchYMin = std::max(0, -rotMinY);
        int32_t searchYMax = std::min(targetH - 1, targetH - 1 - rotMaxY);

        if (searchXMin > searchXMax || searchYMin > searchYMax) {
            continue;
        }

        for (int32_t y = searchYMin; y <= searchYMax; y += kStepSize) {
            std::vector<int32_t> xPositions;
            std::vector<double> rowScores;
            targetPyramid.ComputeScoresRow(rotatedFeatures, startLevel,
                                           searchXMin, searchXMax, y,
                                           kStepSize, coarseThreshold,
                                           xPositions, rowScores);

            for (size_t i = 0; i < xPositions.size() && i < rowScores.size(); ++i) {
                candidates.push_back({xPositions[i], y, impl->templateAngles[ai], rowScores[i]});
            }
        }
    }

    if (candidates.empty()) {
        return;
    }

    const double modelWCoarse =
        std::max(8.0, static_cast<double>(impl->templateWidth) / static_cast<double>(1 << startLevel));
    const double modelHCoarse =
        std::max(8.0, static_cast<double>(impl->templateHeight) / static_cast<double>(1 << startLevel));
    const double coarseDist = std::max(
        5.0, std::sqrt(modelWCoarse * modelWCoarse + modelHCoarse * modelHCoarse) * K_COARSE_DIVERSITY_DIST_SCALE);
    const size_t coarseCap = static_cast<size_t>(std::max(
        16, static_cast<int32_t>(std::lround(K_COARSE_CANDIDATE_CAP * (1.15 - 0.55 * greediness)))));
    KeepTopDiverse(candidates, coarseCap, coarseDist, K_COARSE_DIVERSITY_ANGLE_DEG * K_PI / 180.0);

    // Fine angle refinement around coarse candidates (aligned with archived LINEMOD flow).
    const double kFineAngleRadius = K_FINE_ANGLE_RADIUS_DEG * K_PI / 180.0;
    const double kFineAngleStep = K_FINE_ANGLE_STEP_DEG * K_PI / 180.0;
    const auto& startLevelFeatures = impl->levelFeatures[static_cast<size_t>(startLevel)];

    std::vector<Candidate> refinedCandidates;
    refinedCandidates.reserve(candidates.size());

    for (const auto& c : candidates) {
        Candidate best = c;

        for (double dAngle = -kFineAngleRadius; dAngle <= kFineAngleRadius + 1e-9; dAngle += kFineAngleStep) {
            if (std::abs(dAngle) < 1e-9) {
                continue;
            }

            double fineAngle = c.angle + dAngle;
            auto rotated = Qi::Vision::Internal::LinemodPyramid::RotateFeatures(startLevelFeatures, fineAngle);

            for (int32_t dy = -1; dy <= 1; ++dy) {
                for (int32_t dx = -1; dx <= 1; ++dx) {
                    int32_t px = c.x + dx;
                    int32_t py = c.y + dy;

                    if (px < 0 || px >= targetPyramid.GetWidth(startLevel) ||
                        py < 0 || py >= targetPyramid.GetHeight(startLevel)) {
                        continue;
                    }

                    double s = targetPyramid.ComputeScorePrecomputed(rotated, startLevel, px, py);
                    if (s > best.score) {
                        best.x = px;
                        best.y = py;
                        best.angle = fineAngle;
                        best.score = s;
                    }
                }
            }
        }

        refinedCandidates.push_back(best);
    }

    candidates.swap(refinedCandidates);

    // Coarse-to-fine pyramid refinement.
    for (int32_t level = startLevel - 1; level >= 0; --level) {
        if (level >= static_cast<int32_t>(impl->levelFeatures.size()) ||
            impl->levelFeatures[static_cast<size_t>(level)].empty()) {
            continue;
        }

        const auto& levelFeatures = impl->levelFeatures[static_cast<size_t>(level)];
        int32_t levelW = targetPyramid.GetWidth(level);
        int32_t levelH = targetPyramid.GetHeight(level);

        int32_t searchRadius = (level == 0) ? K_REFINE_SEARCH_RADIUS_FINE : K_REFINE_SEARCH_RADIUS_COARSE;
        double angleRadius = (level == 0) ? K_REFINE_ANGLE_RADIUS_FINE : K_REFINE_ANGLE_RADIUS_COARSE;
        double levelAngleStep = (level == 0) ? K_REFINE_ANGLE_STEP_FINE : K_REFINE_ANGLE_STEP_COARSE;

        std::vector<double> refineAngles;
        for (double da = -angleRadius; da <= angleRadius + 1e-9; da += levelAngleStep) {
            refineAngles.push_back(da);
        }

        std::vector<Candidate> refined;
        refined.reserve(candidates.size());

        for (const auto& c : candidates) {
            double baseX = c.x * 2.0;
            double baseY = c.y * 2.0;
            double baseAngle = c.angle;

            std::vector<std::vector<Qi::Vision::Internal::LinemodFeature>> rotatedTemplates;
            rotatedTemplates.resize(refineAngles.size());
            for (size_t ai = 0; ai < refineAngles.size(); ++ai) {
                rotatedTemplates[ai] = Qi::Vision::Internal::LinemodPyramid::RotateFeatures(
                    levelFeatures, baseAngle + refineAngles[ai]);
            }

            Candidate best;
            best.score = -1.0;

            for (int32_t dy = -searchRadius; dy <= searchRadius; ++dy) {
                for (int32_t dx = -searchRadius; dx <= searchRadius; ++dx) {
                    int32_t px = static_cast<int32_t>(baseX) + dx;
                    int32_t py = static_cast<int32_t>(baseY) + dy;

                    if (px < 0 || px >= levelW || py < 0 || py >= levelH) {
                        continue;
                    }

                    for (size_t ai = 0; ai < refineAngles.size(); ++ai) {
                        double s = targetPyramid.ComputeScorePrecomputed(rotatedTemplates[ai], level, px, py);
                        if (s > best.score) {
                            best.x = px;
                            best.y = py;
                            best.angle = baseAngle + refineAngles[ai];
                            best.score = s;
                        }
                    }
                }
            }

            const double refineAccept = std::clamp(K_REFINE_ACCEPT_SCALE + 0.08 * greediness, 0.0, 1.0);
            if (best.score >= minScore * refineAccept) {
                refined.push_back(best);
            }
        }

        candidates.swap(refined);
        if (candidates.empty()) {
            break;
        }

        const double levelModelW =
            std::max(8.0, static_cast<double>(impl->templateWidth) / static_cast<double>(1 << level));
        const double levelModelH =
            std::max(8.0, static_cast<double>(impl->templateHeight) / static_cast<double>(1 << level));
        const double levelDist = std::max(
            4.0, std::sqrt(levelModelW * levelModelW + levelModelH * levelModelH) * K_REFINE_DIVERSITY_DIST_SCALE);
        int32_t capBase = (level == 0) ? K_REFINE_CANDIDATE_CAP_FINE : K_REFINE_CANDIDATE_CAP_COARSE;
        size_t limit = static_cast<size_t>(std::max(
            8, static_cast<int32_t>(std::lround(capBase * (1.15 - 0.50 * greediness)))));
        KeepTopDiverse(candidates, limit, levelDist, K_REFINE_DIVERSITY_ANGLE_DEG * K_PI / 180.0);
    }

    if (candidates.empty()) {
        return;
    }

    std::vector<Candidate> results;
    results.reserve(candidates.size());
    for (const auto& c : candidates) {
        if (c.score >= minScore) {
            Candidate out = c;
            out.angle = NormalizeAngle(out.angle);
            results.push_back(out);
        }
    }

    if (results.empty()) {
        return;
    }

    ApplyNMS(results, maxOverlap,
             static_cast<double>(impl->templateWidth),
             static_cast<double>(impl->templateHeight),
             K_NMS_MIN_DIST_SCALE,
             K_NMS_OVERLAP_MIN_AREA);

    std::sort(results.begin(), results.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    if (numMatches > 0 && static_cast<int32_t>(results.size()) > numMatches) {
        results.resize(static_cast<size_t>(numMatches));
    }

    rows.reserve(results.size());
    cols.reserve(results.size());
    angles.reserve(results.size());
    scores.reserve(results.size());

    for (const auto& r : results) {
        rows.push_back(r.y);
        cols.push_back(r.x);
        angles.push_back(r.angle);
        scores.push_back(r.score);
    }
}

void GetFastShapeModelFeaturePoints(
    const FastShapeModel& model,
    std::vector<Point2d>& points)
{
    points.clear();
    if (!model.IsValid() || model.Impl() == nullptr || !model.Impl()->valid) {
        return;
    }

    const auto* impl = model.Impl();
    if (impl->levelFeatures.empty()) {
        return;
    }

    const auto& feats = impl->levelFeatures[0];
    points.reserve(feats.size());
    for (const auto& f : feats) {
        points.emplace_back(static_cast<double>(f.x), static_cast<double>(f.y));
    }
}

void GetFastShapeModelTemplateSize(
    const FastShapeModel& model,
    int32_t& width,
    int32_t& height)
{
    width = 0;
    height = 0;
    if (!model.IsValid() || model.Impl() == nullptr || !model.Impl()->valid) {
        return;
    }

    const auto* impl = model.Impl();
    width = impl->templateWidth;
    height = impl->templateHeight;
}

} // namespace Qi::Vision::Matching
