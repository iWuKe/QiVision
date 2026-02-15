#include <QiVision/Matching/FastShapeModel.h>

#include <QiVision/Core/Exception.h>
#include <QiVision/Internal/LinemodPyramid.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Qi::Vision::Matching {

namespace {

constexpr double K_PI = 3.14159265358979323846;
constexpr int32_t K_NEIGHBOR_THRESHOLD = 5;
constexpr double K_SMOOTH_SIGMA = 1.0;
constexpr float K_FEATURE_MIN_DISTANCE = 2.0f;

constexpr double K_COARSE_MIN_SCORE_FLOOR = 0.40;
constexpr double K_COARSE_MIN_SCORE_SCALE = 0.65;
constexpr int32_t K_COARSE_STEP_SIZE = 1;
constexpr int32_t K_REFINE_SEARCH_RADIUS_FINE = 3;
constexpr int32_t K_REFINE_SEARCH_RADIUS_COARSE = 2;
constexpr double K_REFINE_ANGLE_RADIUS_FINE = 0.08;
constexpr double K_REFINE_ANGLE_RADIUS_COARSE = 0.14;
constexpr double K_REFINE_ANGLE_STEP_FINE = 0.015;
constexpr double K_REFINE_ANGLE_STEP_COARSE = 0.03;
constexpr int32_t K_REFINE_CANDIDATE_CAP_FINE = 80;
constexpr int32_t K_REFINE_CANDIDATE_CAP_COARSE = 180;
constexpr double K_REFINE_ACCEPT_SCALE = 0.70;
constexpr int32_t K_COARSE_CANDIDATE_CAP = 420;
constexpr int32_t K_MIN_START_LEVEL_FEATURES = 32;

struct Candidate {
    int32_t x = 0;
    int32_t y = 0;
    double angle = 0.0;
    double scale = 1.0;
    double score = 0.0;
};

inline bool CompareCandidate(const Candidate& a, const Candidate& b) {
    if (a.score != b.score) {
        return a.score > b.score;
    }
    if (a.y != b.y) {
        return a.y < b.y;
    }
    if (a.x != b.x) {
        return a.x < b.x;
    }
    return a.angle < b.angle;
}

inline std::vector<double> BuildScaleList(double scaleMin, double scaleMax, double scaleStep) {
    const double sMin = std::max(0.05, std::min(scaleMin, scaleMax));
    const double sMax = std::max(0.05, std::max(scaleMin, scaleMax));
    if (scaleStep <= 0.0 || (sMax - sMin) < 1e-9) {
        return {std::clamp(1.0, sMin, sMax)};
    }

    const double step = std::max(1e-4, scaleStep);
    std::vector<double> scales;
    for (double s = sMin; s <= sMax + 1e-9; s += step) {
        scales.push_back(s);
    }
    if (scales.empty()) {
        scales.push_back(std::clamp(1.0, sMin, sMax));
    }
    return scales;
}

std::vector<Qi::Vision::Internal::LinemodFeature> ScaleFeatures(
    const std::vector<Qi::Vision::Internal::LinemodFeature>& features,
    double scale)
{
    if (features.empty()) {
        return {};
    }
    if (std::abs(scale - 1.0) < 1e-9) {
        return features;
    }

    std::vector<Qi::Vision::Internal::LinemodFeature> out;
    out.reserve(features.size());
    for (const auto& f : features) {
        const int32_t sx = static_cast<int32_t>(std::lround(static_cast<double>(f.x) * scale));
        const int32_t sy = static_cast<int32_t>(std::lround(static_cast<double>(f.y) * scale));
        out.emplace_back(
            static_cast<int16_t>(std::clamp(sx, -32768, 32767)),
            static_cast<int16_t>(std::clamp(sy, -32768, 32767)),
            f.ori);
    }
    return out;
}

inline bool IsFastShapeProfileEnabled() {
    static int cached = -1;
    if (cached >= 0) {
        return cached != 0;
    }
    const char* env = std::getenv("QIVISION_FAST_SHAPE_PROFILE");
    if (env == nullptr || env[0] == '\0' || env[0] == '0') {
        cached = 0;
    } else {
        cached = 1;
    }
    return cached != 0;
}

inline double DurationMs(const std::chrono::steady_clock::time_point& t0,
                         const std::chrono::steady_clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

inline double NormalizeAngle(double a) {
    while (a < 0.0) a += 2.0 * K_PI;
    while (a >= 2.0 * K_PI) a -= 2.0 * K_PI;
    return a;
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

double ComputeIoUAabb(const Candidate& a, const Candidate& b,
                      double wa, double ha, double wb, double hb) {
    double ax1 = a.x - wa * 0.5;
    double ay1 = a.y - ha * 0.5;
    double ax2 = a.x + wa * 0.5;
    double ay2 = a.y + ha * 0.5;
    double bx1 = b.x - wb * 0.5;
    double by1 = b.y - hb * 0.5;
    double bx2 = b.x + wb * 0.5;
    double by2 = b.y + hb * 0.5;

    double ix1 = std::max(ax1, bx1);
    double iy1 = std::max(ay1, by1);
    double ix2 = std::min(ax2, bx2);
    double iy2 = std::min(ay2, by2);
    if (ix2 <= ix1 || iy2 <= iy1) {
        return 0.0;
    }
    double inter = (ix2 - ix1) * (iy2 - iy1);
    double areaA = (ax2 - ax1) * (ay2 - ay1);
    double areaB = (bx2 - bx1) * (by2 - by1);
    double uni = areaA + areaB - inter;
    return (uni > 1e-9) ? (inter / uni) : 0.0;
}

double ComputeOverlapMinAreaAabb(const Candidate& a, const Candidate& b,
                                 double wa, double ha, double wb, double hb) {
    double ax1 = a.x - wa * 0.5;
    double ay1 = a.y - ha * 0.5;
    double ax2 = a.x + wa * 0.5;
    double ay2 = a.y + ha * 0.5;
    double bx1 = b.x - wb * 0.5;
    double by1 = b.y - hb * 0.5;
    double bx2 = b.x + wb * 0.5;
    double by2 = b.y + hb * 0.5;

    double ix1 = std::max(ax1, bx1);
    double iy1 = std::max(ay1, by1);
    double ix2 = std::min(ax2, bx2);
    double iy2 = std::min(ay2, by2);
    if (ix2 <= ix1 || iy2 <= iy1) {
        return 0.0;
    }
    double inter = (ix2 - ix1) * (iy2 - iy1);
    const double areaA = std::max(1.0, wa * ha);
    const double areaB = std::max(1.0, wb * hb);
    double area = std::max(1.0, std::min(areaA, areaB));
    return inter / area;
}

void ApplyNMS(std::vector<Candidate>& cands,
              double maxOverlap,
              double baseModelW,
              double baseModelH) {
    std::sort(cands.begin(), cands.end(), CompareCandidate);

    std::vector<Candidate> keep;
    keep.reserve(cands.size());

    const bool enableOverlap = (maxOverlap < 1.0 - 1e-9);

    for (const auto& c : cands) {
        bool suppressed = false;
        for (const auto& k : keep) {
            const double wc = std::max(2.0, baseModelW * c.scale);
            const double hc = std::max(2.0, baseModelH * c.scale);
            const double wk = std::max(2.0, baseModelW * k.scale);
            const double hk = std::max(2.0, baseModelH * k.scale);
            double dx = static_cast<double>(c.x - k.x);
            double dy = static_cast<double>(c.y - k.y);
            double d2 = dx * dx + dy * dy;
            const double minDist =
                std::max(4.0, std::max({wc, hc, wk, hk}) * 0.5 *
                                   std::clamp(1.0 - maxOverlap, 0.0, 1.0));
            const double minDistSq = minDist * minDist;
            if (d2 < minDistSq) {
                suppressed = true;
                break;
            }

            if (enableOverlap) {
                if (ComputeOverlapMinAreaAabb(c, k, wc, hc, wk, hk) > maxOverlap) {
                    suppressed = true;
                    break;
                }
                if (ComputeIoUAabb(c, k, wc, hc, wk, hk) > std::min(0.90, maxOverlap + 0.15)) {
                    suppressed = true;
                    break;
                }
            }
        }
        if (!suppressed) {
            keep.push_back(c);
        }
    }

    cands.swap(keep);
}

void ApplyFinalDedup(std::vector<Candidate>& cands,
                     double maxOverlap,
                     double baseModelW,
                     double baseModelH) {
    if (cands.size() <= 1) {
        return;
    }

    std::sort(cands.begin(), cands.end(), CompareCandidate);

    const double strictOverlap = std::clamp(maxOverlap + 0.05, 0.35, 0.75);
    const double strictIou = std::clamp(maxOverlap + 0.12, 0.45, 0.85);

    std::vector<Candidate> keep;
    keep.reserve(cands.size());

    for (const auto& c : cands) {
        bool merged = false;
        for (const auto& k : keep) {
            const double wc = std::max(2.0, baseModelW * c.scale);
            const double hc = std::max(2.0, baseModelH * c.scale);
            const double wk = std::max(2.0, baseModelW * k.scale);
            const double hk = std::max(2.0, baseModelH * k.scale);
            const double strictDist = std::max(
                8.0, std::min({wc, hc, wk, hk}) *
                         (0.28 + 0.30 * (1.0 - std::clamp(maxOverlap, 0.0, 1.0))));
            const double strictDistSq = strictDist * strictDist;
            const double dx = static_cast<double>(c.x - k.x);
            const double dy = static_cast<double>(c.y - k.y);
            if (dx * dx + dy * dy < strictDistSq) {
                merged = true;
                break;
            }
            if (ComputeOverlapMinAreaAabb(c, k, wc, hc, wk, hk) > strictOverlap) {
                merged = true;
                break;
            }
            if (ComputeIoUAabb(c, k, wc, hc, wk, hk) > strictIou) {
                merged = true;
                break;
            }
        }
        if (!merged) {
            keep.push_back(c);
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
    float weakThreshold = 20.0f;
    float strongThreshold = 60.0f;
    int32_t numFeatures = 63;

    std::vector<std::vector<Qi::Vision::Internal::LinemodFeature>> levelFeatures;
    std::vector<double> templateAngles;
    std::vector<double> templateScales;
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
    if (strategy.scaleMin <= 0.0 || strategy.scaleMax <= 0.0) {
        throw InvalidArgumentException("CreateFastShapeModel: scaleMin/scaleMax must be > 0");
    }

    FastShapeModelStrategy st = strategy;
    st.numFeatures = std::max(16, st.numFeatures);
    for (auto& t : st.tAtLevel) {
        t = std::max(1, t);
    }
    st.weakThreshold = std::max(1.0, st.weakThreshold);
    st.strongThreshold = std::max(st.weakThreshold, st.strongThreshold);
    st.scaleMin = std::max(0.05, st.scaleMin);
    st.scaleMax = std::max(0.05, st.scaleMax);
    if (st.scaleStep > 0.0) {
        st.scaleStep = std::max(1e-4, st.scaleStep);
    }

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
        angleStep = K_PI / 18.0; // 10 degrees
    }
    impl->coarseAngleStep = std::max(angleStep, K_PI / 360.0);

    impl->weakThreshold = static_cast<float>(st.weakThreshold);
    impl->strongThreshold = static_cast<float>(st.strongThreshold);
    impl->numFeatures = st.numFeatures;

    // Build pyramid on original (angle=0) template for levelFeatures
    // (used by GetFastShapeModelFeaturePoints and refinement)
    Qi::Vision::Internal::LinemodPyramid templatePyramid;
    Qi::Vision::Internal::LinemodPyramidParams pyramidParams;
    pyramidParams.numLevels = impl->numLevels;
    pyramidParams.minMagnitude = impl->strongThreshold;
    pyramidParams.spreadT = st.tAtLevel.front();
    pyramidParams.spreadTAtLevel = st.tAtLevel;
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
    const int32_t numAngleTemplates = std::max(
        1, static_cast<int32_t>(std::ceil(impl->angleExtent / impl->coarseAngleStep)));
    impl->templateAngles.reserve(static_cast<size_t>(numAngleTemplates));
    for (int32_t ai = 0; ai < numAngleTemplates; ++ai) {
        impl->templateAngles.push_back(impl->angleStart + ai * impl->coarseAngleStep);
    }
    if (impl->templateAngles.empty()) {
        impl->templateAngles.push_back(impl->angleStart);
    }

    const std::vector<double> scaleList = BuildScaleList(st.scaleMin, st.scaleMax, st.scaleStep);
    impl->templateScales.clear();
    impl->rotatedTemplates.clear();
    impl->templateScales.reserve(scaleList.size() * impl->templateAngles.size());
    impl->rotatedTemplates.reserve(scaleList.size() * impl->templateAngles.size());

    for (double scale : scaleList) {
        for (size_t ai = 0; ai < impl->templateAngles.size(); ++ai) {
            impl->templateScales.push_back(scale);
            impl->rotatedTemplates.push_back({});
            auto& byLevel = impl->rotatedTemplates.back();
            byLevel.resize(static_cast<size_t>(impl->numLevels));
            for (int32_t level = 0; level < impl->numLevels; ++level) {
                auto scaledFeatures = ScaleFeatures(
                    impl->levelFeatures[static_cast<size_t>(level)],
                    scale);
                byLevel[static_cast<size_t>(level)] =
                    Qi::Vision::Internal::LinemodPyramid::RotateFeatures(
                        scaledFeatures,
                        impl->templateAngles[ai]);
            }
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
    std::vector<double>& scores,
    std::vector<double>* scales)
{
    using Clock = std::chrono::steady_clock;
    const bool enableProfile = IsFastShapeProfileEnabled();
    const auto tAll0 = Clock::now();
    auto tBuild1 = tAll0;
    auto tCoarse1 = tAll0;
    auto tRefine1 = tAll0;
    auto tPost1 = tAll0;
    size_t coarseRawCount = 0;
    size_t coarseNmsCount = 0;
    size_t finalPreNmsCount = 0;
    size_t finalCount = 0;

    auto printTiming = [&](const char* status,
                           int32_t startLevel,
                           int32_t stepSize,
                           size_t angleCount) {
        if (!enableProfile) {
            return;
        }
        const auto tNow = Clock::now();
        std::printf("[FastShapeTiming] status=%s build=%.3fms coarse=%.3fms refine=%.3fms post=%.3fms total=%.3fms | "
                    "startLevel=%d step=%d angles=%zu coarse=%zu->%zu final=%zu->%zu\n",
                    status,
                    DurationMs(tAll0, tBuild1),
                    DurationMs(tBuild1, tCoarse1),
                    DurationMs(tCoarse1, tRefine1),
                    DurationMs(tRefine1, tPost1),
                    DurationMs(tAll0, tNow),
                    startLevel,
                    stepSize,
                    angleCount,
                    coarseRawCount,
                    coarseNmsCount,
                    finalPreNmsCount,
                    finalCount);
    };

    rows.clear();
    cols.clear();
    angles.clear();
    scores.clear();
    if (scales != nullptr) {
        scales->clear();
    }

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
    const double greedy = std::clamp(greediness, 0.0, 1.0);

    const auto* impl = model.Impl();
    const auto& st = impl->strategy;
    const size_t templateCount = impl->rotatedTemplates.size();
    Qi::Vision::Internal::LinemodPyramid targetPyramid;
    Qi::Vision::Internal::LinemodPyramidParams targetParams;
    targetParams.numLevels = impl->numLevels;
    targetParams.minMagnitude = impl->weakThreshold;
    targetParams.spreadT = st.tAtLevel.empty() ? 4 : st.tAtLevel.front();
    targetParams.spreadTAtLevel = st.tAtLevel;
    targetParams.neighborThreshold = K_NEIGHBOR_THRESHOLD;
    targetParams.smoothSigma = K_SMOOTH_SIGMA;
    targetParams.extractFeatures = false;

    if (!targetPyramid.Build(image, targetParams)) {
        throw InvalidArgumentException("FindFastShapeModel: failed to build target LINEMOD pyramid");
    }
    tBuild1 = Clock::now();

    const int32_t levels = std::min<int32_t>(impl->numLevels, targetPyramid.NumLevels());
    if (levels <= 0) {
        tCoarse1 = tBuild1;
        tRefine1 = tCoarse1;
        tPost1 = tRefine1;
        printTiming("no_levels", -1, 0, templateCount);
        return;
    }

    int32_t startLevel = levels - 1;
    while (startLevel > 0 &&
           (startLevel >= static_cast<int32_t>(impl->levelFeatures.size()) ||
            static_cast<int32_t>(impl->levelFeatures[static_cast<size_t>(startLevel)].size()) <
                K_MIN_START_LEVEL_FEATURES)) {
        startLevel--;
    }

    if (startLevel < 0 ||
        startLevel >= static_cast<int32_t>(impl->levelFeatures.size()) ||
        impl->levelFeatures[static_cast<size_t>(startLevel)].empty()) {
        tCoarse1 = tBuild1;
        tRefine1 = tCoarse1;
        tPost1 = tRefine1;
        printTiming("no_start_level", startLevel, 0, templateCount);
        return;
    }

    int32_t stepSize = K_COARSE_STEP_SIZE;
    if (!st.tAtLevel.empty()) {
        const int32_t idx = std::clamp(startLevel, 0, static_cast<int32_t>(st.tAtLevel.size()) - 1);
        // Faster coarse scan: keep alignment with T but preserve recall with /2.
        stepSize = std::max(stepSize, std::max(1, st.tAtLevel[static_cast<size_t>(idx)] / 2));
    }
    const double coarseScale = std::clamp(K_COARSE_MIN_SCORE_SCALE + (greedy - 0.5) * 0.12, 0.55, 0.75);
    const double coarseThreshold = std::max(
        K_COARSE_MIN_SCORE_FLOOR,
        minScore * coarseScale);

    std::vector<Candidate> candidates;
    candidates.reserve(2048);

    auto coarseSearchForAngle = [&](size_t ai, std::vector<Candidate>& out) {
        if (ai >= templateCount ||
            startLevel >= static_cast<int32_t>(impl->rotatedTemplates[ai].size())) {
            return;
        }

        const auto& rotatedFeatures = impl->rotatedTemplates[ai][static_cast<size_t>(startLevel)];
        if (rotatedFeatures.empty()) {
            return;
        }

        int32_t rotMinX = 0, rotMaxX = 0, rotMinY = 0, rotMaxY = 0;
        ComputeFeatureBounds(rotatedFeatures, rotMinX, rotMaxX, rotMinY, rotMaxY);
        if (rotMinX > rotMaxX || rotMinY > rotMaxY) {
            return;
        }

        int32_t targetW = targetPyramid.GetWidth(startLevel);
        int32_t targetH = targetPyramid.GetHeight(startLevel);

        int32_t searchXMin = std::max(0, -rotMinX);
        int32_t searchXMax = std::min(targetW - 1, targetW - 1 - rotMaxX);
        int32_t searchYMin = std::max(0, -rotMinY);
        int32_t searchYMax = std::min(targetH - 1, targetH - 1 - rotMaxY);

        if (searchXMin > searchXMax || searchYMin > searchYMax) {
            return;
        }

        alignas(32) double scores8[8];
        for (int32_t y = searchYMin; y <= searchYMax; y += stepSize) {
            int32_t x = searchXMin;

            for (; x + 7 <= searchXMax; x += 8) {
                targetPyramid.ComputeScoresBatch8(rotatedFeatures, startLevel, x, y, scores8, coarseThreshold);
                for (int32_t i = 0; i < 8; ++i) {
                    if (scores8[i] >= coarseThreshold) {
                        out.push_back({x + i, y, impl->templateAngles[ai], impl->templateScales[ai], scores8[i]});
                    }
                }
            }

            for (; x <= searchXMax; ++x) {
                double score = targetPyramid.ComputeScorePrecomputed(rotatedFeatures, startLevel, x, y, coarseThreshold);
                if (score >= coarseThreshold) {
                    out.push_back({x, y, impl->templateAngles[ai], impl->templateScales[ai], score});
                }
            }
        }
    };

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<Candidate> localCandidates;
        localCandidates.reserve(1024);

#pragma omp for schedule(dynamic)
        for (int32_t ai = 0; ai < static_cast<int32_t>(templateCount); ++ai) {
            coarseSearchForAngle(static_cast<size_t>(ai), localCandidates);
        }

#pragma omp critical
        {
            candidates.insert(candidates.end(),
                              localCandidates.begin(), localCandidates.end());
        }
    }
#else
    for (size_t ai = 0; ai < templateCount; ++ai) {
        coarseSearchForAngle(ai, candidates);
    }
#endif

    if (candidates.empty()) {
        tCoarse1 = Clock::now();
        tRefine1 = tCoarse1;
        tPost1 = tRefine1;
        printTiming("no_coarse_candidate", startLevel, stepSize, templateCount);
        return;
    }

    coarseRawCount = candidates.size();
    const double startScale = std::ldexp(1.0, -startLevel);
    ApplyNMS(candidates, maxOverlap,
             std::max(4.0, static_cast<double>(impl->templateWidth) * startScale),
             std::max(4.0, static_cast<double>(impl->templateHeight) * startScale));
    if (static_cast<int32_t>(candidates.size()) > K_COARSE_CANDIDATE_CAP) {
        candidates.resize(static_cast<size_t>(K_COARSE_CANDIDATE_CAP));
    }
    coarseNmsCount = candidates.size();
    tCoarse1 = Clock::now();

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
        const double acceptScale = std::clamp(K_REFINE_ACCEPT_SCALE + (greedy - 0.5) * 0.10, 0.62, 0.82);
        const double acceptThreshold = minScore * acceptScale;

        auto refineOne = [&](const Candidate& c,
                             std::vector<std::vector<Qi::Vision::Internal::LinemodFeature>>& rotatedTemplates,
                             std::vector<Candidate>& out) {
            double baseX = c.x * 2.0;
            double baseY = c.y * 2.0;
            double baseAngle = c.angle;
            double baseScale = c.scale;
            auto scaledLevelFeatures = ScaleFeatures(levelFeatures, baseScale);

            for (size_t ai = 0; ai < refineAngles.size(); ++ai) {
                rotatedTemplates[ai] = Qi::Vision::Internal::LinemodPyramid::RotateFeatures(
                    scaledLevelFeatures, baseAngle + refineAngles[ai]);
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
                        double s = targetPyramid.ComputeScorePrecomputed(rotatedTemplates[ai], level, px, py, acceptThreshold);
                        if (s > best.score) {
                            best.x = px;
                            best.y = py;
                            best.angle = baseAngle + refineAngles[ai];
                            best.scale = baseScale;
                            best.score = s;
                        }
                    }
                }
            }

            if (best.score >= acceptThreshold) {
                out.push_back(best);
            }
        };

#ifdef _OPENMP
#pragma omp parallel
        {
            std::vector<Candidate> localRefined;
            localRefined.reserve(candidates.size() / 4 + 8);
            std::vector<std::vector<Qi::Vision::Internal::LinemodFeature>> rotatedTemplates(refineAngles.size());

#pragma omp for schedule(static)
            for (int32_t ci = 0; ci < static_cast<int32_t>(candidates.size()); ++ci) {
                refineOne(candidates[static_cast<size_t>(ci)], rotatedTemplates, localRefined);
            }

#pragma omp critical
            {
                refined.insert(refined.end(), localRefined.begin(), localRefined.end());
            }
        }
#else
        std::vector<std::vector<Qi::Vision::Internal::LinemodFeature>> rotatedTemplates(refineAngles.size());
        for (const auto& c : candidates) {
            refineOne(c, rotatedTemplates, refined);
        }
#endif

        candidates.swap(refined);
        if (candidates.empty()) {
            break;
        }

        const double levelScale = std::ldexp(1.0, -level);
        ApplyNMS(candidates, maxOverlap,
                 std::max(4.0, static_cast<double>(impl->templateWidth) * levelScale),
                 std::max(4.0, static_cast<double>(impl->templateHeight) * levelScale));
        const int32_t levelCap = (level == 0) ? K_REFINE_CANDIDATE_CAP_FINE : K_REFINE_CANDIDATE_CAP_COARSE;
        if (static_cast<int32_t>(candidates.size()) > levelCap) {
            candidates.resize(static_cast<size_t>(levelCap));
        }
    }
    tRefine1 = Clock::now();

    if (candidates.empty()) {
        tPost1 = tRefine1;
        printTiming("empty_after_refine", startLevel, stepSize, templateCount);
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
        tPost1 = Clock::now();
        finalPreNmsCount = 0;
        finalCount = 0;
        printTiming("below_min_score", startLevel, stepSize, templateCount);
        return;
    }
    finalPreNmsCount = results.size();

    ApplyNMS(results, maxOverlap,
             static_cast<double>(impl->templateWidth),
             static_cast<double>(impl->templateHeight));
    ApplyFinalDedup(results, maxOverlap,
                    static_cast<double>(impl->templateWidth),
                    static_cast<double>(impl->templateHeight));
    finalCount = results.size();
    tPost1 = Clock::now();

    if (numMatches > 0 && static_cast<int32_t>(results.size()) > numMatches) {
        results.resize(static_cast<size_t>(numMatches));
    }

    rows.reserve(results.size());
    cols.reserve(results.size());
    angles.reserve(results.size());
    scores.reserve(results.size());
    if (scales != nullptr) {
        scales->reserve(results.size());
    }

    for (const auto& r : results) {
        rows.push_back(r.y);
        cols.push_back(r.x);
        angles.push_back(r.angle);
        scores.push_back(r.score);
        if (scales != nullptr) {
            scales->push_back(r.scale);
        }
    }

    printTiming("ok", startLevel, stepSize, templateCount);
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
