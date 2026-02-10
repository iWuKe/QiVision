#include <QiVision/Matching/FastShapeDetector.h>

#include <QiVision/Core/Exception.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace Qi::Vision::Matching {

namespace {

double ComputeIoU(const FastShapeDetection& a, const FastShapeDetection& b) {
    double aw = static_cast<double>(a.templateWidth);
    double ah = static_cast<double>(a.templateHeight);
    double bw = static_cast<double>(b.templateWidth);
    double bh = static_cast<double>(b.templateHeight);

    if (aw <= 0.0 || ah <= 0.0 || bw <= 0.0 || bh <= 0.0) {
        return 0.0;
    }

    double ax1 = a.col - aw * 0.5;
    double ay1 = a.row - ah * 0.5;
    double ax2 = a.col + aw * 0.5;
    double ay2 = a.row + ah * 0.5;

    double bx1 = b.col - bw * 0.5;
    double by1 = b.row - bh * 0.5;
    double bx2 = b.col + bw * 0.5;
    double by2 = b.row + bh * 0.5;

    double ix1 = std::max(ax1, bx1);
    double iy1 = std::max(ay1, by1);
    double ix2 = std::min(ax2, bx2);
    double iy2 = std::min(ay2, by2);

    if (ix2 <= ix1 || iy2 <= iy1) {
        return 0.0;
    }

    double inter = (ix2 - ix1) * (iy2 - iy1);
    double areaA = aw * ah;
    double areaB = bw * bh;
    double uni = areaA + areaB - inter;
    return (uni > 1e-9) ? (inter / uni) : 0.0;
}

double ComputeOverlapMinArea(const FastShapeDetection& a, const FastShapeDetection& b) {
    double aw = static_cast<double>(a.templateWidth);
    double ah = static_cast<double>(a.templateHeight);
    double bw = static_cast<double>(b.templateWidth);
    double bh = static_cast<double>(b.templateHeight);

    if (aw <= 0.0 || ah <= 0.0 || bw <= 0.0 || bh <= 0.0) {
        return 0.0;
    }

    double ax1 = a.col - aw * 0.5;
    double ay1 = a.row - ah * 0.5;
    double ax2 = a.col + aw * 0.5;
    double ay2 = a.row + ah * 0.5;

    double bx1 = b.col - bw * 0.5;
    double by1 = b.row - bh * 0.5;
    double bx2 = b.col + bw * 0.5;
    double by2 = b.row + bh * 0.5;

    double ix1 = std::max(ax1, bx1);
    double iy1 = std::max(ay1, by1);
    double ix2 = std::min(ax2, bx2);
    double iy2 = std::min(ay2, by2);

    if (ix2 <= ix1 || iy2 <= iy1) {
        return 0.0;
    }

    double inter = (ix2 - ix1) * (iy2 - iy1);
    double minArea = std::min(aw * ah, bw * bh);
    return (minArea > 1e-9) ? (inter / minArea) : 0.0;
}

void ApplyCrossTemplateNms(std::vector<FastShapeDetection>& detections, double maxOverlap) {
    std::sort(detections.begin(), detections.end(),
              [](const FastShapeDetection& a, const FastShapeDetection& b) {
                  return a.score > b.score;
              });

    std::vector<FastShapeDetection> keep;
    keep.reserve(detections.size());

    for (const auto& d : detections) {
        bool suppressed = false;
        for (const auto& k : keep) {
            double iou = ComputeIoU(d, k);
            if (iou > maxOverlap) {
                suppressed = true;
                break;
            }

            double overlapMinArea = ComputeOverlapMinArea(d, k);
            if (overlapMinArea > 0.45) {
                suppressed = true;
                break;
            }
        }

        if (!suppressed) {
            keep.push_back(d);
        }
    }

    detections.swap(keep);
}

} // namespace

namespace Internal {

struct FastShapeTemplateEntry {
    int32_t templateId = -1;
    FastShapeModel model;
    int32_t width = 0;
    int32_t height = 0;
};

class FastShapeDetectorImpl {
public:
    int32_t nextTemplateId = 1;
    std::vector<FastShapeTemplateEntry> templates;

    std::unique_ptr<FastShapeDetectorImpl> Clone() const {
        auto out = std::make_unique<FastShapeDetectorImpl>();
        *out = *this;
        return out;
    }
};

} // namespace Internal

FastShapeDetector::FastShapeDetector()
    : impl_(std::make_unique<Internal::FastShapeDetectorImpl>()) {}
FastShapeDetector::~FastShapeDetector() = default;
FastShapeDetector::FastShapeDetector(const FastShapeDetector& other)
    : impl_(other.impl_
          ? other.impl_->Clone()
          : std::make_unique<Internal::FastShapeDetectorImpl>()) {}
FastShapeDetector::FastShapeDetector(FastShapeDetector&& other) noexcept = default;
FastShapeDetector& FastShapeDetector::operator=(const FastShapeDetector& other) {
    if (this != &other) {
        impl_ = other.impl_
            ? other.impl_->Clone()
            : std::make_unique<Internal::FastShapeDetectorImpl>();
    }
    return *this;
}
FastShapeDetector& FastShapeDetector::operator=(FastShapeDetector&& other) noexcept = default;
bool FastShapeDetector::IsValid() const { return impl_ != nullptr; }

void CreateFastShapeDetector(FastShapeDetector& detector) {
    if (detector.Impl() == nullptr) {
        detector = FastShapeDetector();
        return;
    }

    detector.Impl()->templates.clear();
    detector.Impl()->nextTemplateId = 1;
}

int32_t AddFastShapeTemplate(
    FastShapeDetector& detector,
    const QImage& image,
    const Rect2i& roi,
    int32_t numLevels,
    double angleStart,
    double angleExtent,
    double angleStep,
    const FastShapeModelStrategy& strategy)
{
    if (detector.Impl() == nullptr) {
        detector = FastShapeDetector();
    }

    FastShapeModel model;
    CreateFastShapeModel(
        image,
        roi,
        model,
        numLevels,
        angleStart,
        angleExtent,
        angleStep,
        strategy);

    return AddFastShapeTemplate(detector, model);
}

int32_t AddFastShapeTemplate(
    FastShapeDetector& detector,
    const FastShapeModel& model)
{
    if (!model.IsValid()) {
        throw InvalidArgumentException("AddFastShapeTemplate: model is invalid");
    }

    if (detector.Impl() == nullptr) {
        detector = FastShapeDetector();
    }

    int32_t width = 0;
    int32_t height = 0;
    GetFastShapeModelTemplateSize(model, width, height);

    Internal::FastShapeTemplateEntry entry;
    entry.templateId = detector.Impl()->nextTemplateId++;
    entry.model = model;
    entry.width = width;
    entry.height = height;
    detector.Impl()->templates.push_back(entry);
    return entry.templateId;
}

void RemoveFastShapeTemplate(
    FastShapeDetector& detector,
    int32_t templateId)
{
    if (detector.Impl() == nullptr) {
        throw InvalidArgumentException("RemoveFastShapeTemplate: detector is invalid");
    }

    auto& templates = detector.Impl()->templates;
    auto it = std::find_if(templates.begin(), templates.end(),
                           [templateId](const Internal::FastShapeTemplateEntry& e) {
                               return e.templateId == templateId;
                           });
    if (it == templates.end()) {
        throw InvalidArgumentException("RemoveFastShapeTemplate: templateId not found");
    }

    templates.erase(it);
}

void ClearFastShapeDetector(FastShapeDetector& detector) {
    if (detector.Impl() == nullptr) {
        return;
    }
    detector.Impl()->templates.clear();
    detector.Impl()->nextTemplateId = 1;
}

int32_t GetFastShapeTemplateCount(const FastShapeDetector& detector) {
    if (detector.Impl() == nullptr) {
        return 0;
    }
    return static_cast<int32_t>(detector.Impl()->templates.size());
}

void FindFastShapeDetector(
    const QImage& image,
    const FastShapeDetector& detector,
    double minScore,
    int32_t numMatches,
    double maxOverlap,
    double greediness,
    std::vector<FastShapeDetection>& detections)
{
    detections.clear();

    if (image.Empty()) {
        throw InvalidArgumentException("FindFastShapeDetector: image is empty");
    }
    if (detector.Impl() == nullptr) {
        throw InvalidArgumentException("FindFastShapeDetector: detector is invalid");
    }
    if (minScore < 0.0 || minScore > 1.0) {
        throw InvalidArgumentException("FindFastShapeDetector: minScore must be in [0,1]");
    }
    if (numMatches < 0) {
        throw InvalidArgumentException("FindFastShapeDetector: numMatches must be >= 0");
    }
    if (maxOverlap < 0.0 || maxOverlap > 1.0) {
        throw InvalidArgumentException("FindFastShapeDetector: maxOverlap must be in [0,1]");
    }
    greediness = std::clamp(greediness, 0.0, 1.0);

    const auto& templates = detector.Impl()->templates;
    if (templates.empty()) {
        return;
    }

    std::vector<FastShapeDetection> allDetections;
    for (const auto& entry : templates) {
        std::vector<double> rows;
        std::vector<double> cols;
        std::vector<double> angles;
        std::vector<double> scores;
        FindFastShapeModel(
            image,
            entry.model,
            minScore,
            0,
            maxOverlap,
            greediness,
            rows,
            cols,
            angles,
            scores);

        for (size_t i = 0; i < scores.size(); ++i) {
            FastShapeDetection d;
            d.templateId = entry.templateId;
            d.row = rows[i];
            d.col = cols[i];
            d.angle = angles[i];
            d.score = scores[i];
            d.templateWidth = entry.width;
            d.templateHeight = entry.height;
            allDetections.push_back(d);
        }
    }

    if (allDetections.empty()) {
        return;
    }

    ApplyCrossTemplateNms(allDetections, maxOverlap);
    std::sort(allDetections.begin(), allDetections.end(),
              [](const FastShapeDetection& a, const FastShapeDetection& b) {
                  return a.score > b.score;
              });

    if (numMatches > 0 && static_cast<int32_t>(allDetections.size()) > numMatches) {
        allDetections.resize(static_cast<size_t>(numMatches));
    }

    detections.swap(allDetections);
}

} // namespace Qi::Vision::Matching
