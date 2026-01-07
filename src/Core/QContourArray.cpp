#include <QiVision/Core/QContourArray.h>
#include <QiVision/Core/QMatrix.h>
#include <QiVision/Core/Constants.h>
#include <algorithm>
#include <stdexcept>
#include <limits>

namespace Qi::Vision {

// =============================================================================
// Constructors
// =============================================================================

QContourArray::QContourArray() = default;

QContourArray::QContourArray(size_t capacity) {
    contours_.reserve(capacity);
}

QContourArray::QContourArray(const QContour& contour) {
    contours_.push_back(contour);
}

QContourArray::QContourArray(const std::vector<QContour>& contours)
    : contours_(contours) {}

QContourArray::QContourArray(std::vector<QContour>&& contours)
    : contours_(std::move(contours)) {}

// =============================================================================
// Container Operations
// =============================================================================

const QContour& QContourArray::At(size_t index) const {
    if (index >= contours_.size()) {
        throw std::out_of_range("Contour array index out of range");
    }
    return contours_[index];
}

QContour& QContourArray::At(size_t index) {
    if (index >= contours_.size()) {
        throw std::out_of_range("Contour array index out of range");
    }
    return contours_[index];
}

const QContour& QContourArray::Front() const {
    if (contours_.empty()) {
        throw std::out_of_range("Contour array is empty");
    }
    return contours_.front();
}

QContour& QContourArray::Front() {
    if (contours_.empty()) {
        throw std::out_of_range("Contour array is empty");
    }
    return contours_.front();
}

const QContour& QContourArray::Back() const {
    if (contours_.empty()) {
        throw std::out_of_range("Contour array is empty");
    }
    return contours_.back();
}

QContour& QContourArray::Back() {
    if (contours_.empty()) {
        throw std::out_of_range("Contour array is empty");
    }
    return contours_.back();
}

// =============================================================================
// Modification
// =============================================================================

void QContourArray::Add(const QContour& contour) {
    contours_.push_back(contour);
}

void QContourArray::Add(QContour&& contour) {
    contours_.push_back(std::move(contour));
}

void QContourArray::Add(const QContourArray& other) {
    contours_.insert(contours_.end(), other.contours_.begin(), other.contours_.end());
}

void QContourArray::Insert(size_t index, const QContour& contour) {
    if (index > contours_.size()) {
        throw std::out_of_range("Insert index out of range");
    }
    contours_.insert(contours_.begin() + static_cast<ptrdiff_t>(index), contour);
}

void QContourArray::Remove(size_t index) {
    if (index >= contours_.size()) {
        throw std::out_of_range("Remove index out of range");
    }
    contours_.erase(contours_.begin() + static_cast<ptrdiff_t>(index));
}

void QContourArray::RemoveIf(std::function<bool(const QContour&)> predicate) {
    contours_.erase(
        std::remove_if(contours_.begin(), contours_.end(), predicate),
        contours_.end());
}

void QContourArray::Clear() {
    contours_.clear();
}

void QContourArray::Reserve(size_t capacity) {
    contours_.reserve(capacity);
}

// =============================================================================
// Hierarchy Management
// =============================================================================

void QContourArray::BuildHierarchy() {
    // Reset all hierarchy info
    for (auto& contour : contours_) {
        contour.SetParent(-1);
        contour.ClearChildren();
    }

    size_t n = contours_.size();
    if (n == 0) return;

    // Precompute areas
    std::vector<double> areas(n);
    for (size_t i = 0; i < n; ++i) {
        if (contours_[i].IsClosed() && contours_[i].Size() >= 3) {
            areas[i] = contours_[i].Area();
        } else {
            areas[i] = 0.0;
        }
    }

    // For each closed contour, find the smallest closed contour that:
    // 1. Contains the center of contour i
    // 2. Has larger area than contour i (to prevent cycles)
    for (size_t i = 0; i < n; ++i) {
        if (!contours_[i].IsClosed() || contours_[i].Size() < 3) {
            continue;
        }

        double minParentArea = std::numeric_limits<double>::max();
        int32_t bestParent = -1;

        Point2d center = contours_[i].Centroid();
        double areaI = areas[i];

        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            if (!contours_[j].IsClosed() || contours_[j].Size() < 3) continue;

            double areaJ = areas[j];

            // Parent must have larger area to prevent cycles
            if (areaJ <= areaI) continue;

            // Check if contour j contains the center of contour i
            if (contours_[j].Contains(center)) {
                if (areaJ < minParentArea) {
                    minParentArea = areaJ;
                    bestParent = static_cast<int32_t>(j);
                }
            }
        }

        if (bestParent >= 0) {
            contours_[i].SetParent(bestParent);
            contours_[static_cast<size_t>(bestParent)].AddChild(static_cast<int32_t>(i));
        }
    }
}

std::vector<size_t> QContourArray::GetRootContours() const {
    std::vector<size_t> roots;
    for (size_t i = 0; i < contours_.size(); ++i) {
        if (contours_[i].GetParent() < 0) {
            roots.push_back(i);
        }
    }
    return roots;
}

std::vector<size_t> QContourArray::GetChildren(size_t index) const {
    if (index >= contours_.size()) {
        return {};
    }

    const auto& children = contours_[index].GetChildren();
    std::vector<size_t> result;
    result.reserve(children.size());
    for (int32_t child : children) {
        if (child >= 0 && static_cast<size_t>(child) < contours_.size()) {
            result.push_back(static_cast<size_t>(child));
        }
    }
    return result;
}

int32_t QContourArray::GetParent(size_t index) const {
    if (index >= contours_.size()) {
        return -1;
    }
    return contours_[index].GetParent();
}

int QContourArray::GetDepth(size_t index) const {
    if (index >= contours_.size()) {
        return 0;
    }

    int depth = 0;
    int32_t parent = contours_[index].GetParent();

    while (parent >= 0 && static_cast<size_t>(parent) < contours_.size()) {
        depth++;
        parent = contours_[static_cast<size_t>(parent)].GetParent();
    }

    return depth;
}

void QContourArray::FlattenHierarchy() {
    for (auto& contour : contours_) {
        contour.SetParent(-1);
        contour.ClearChildren();
    }
}

// =============================================================================
// Selection / Filtering
// =============================================================================

QContourArray QContourArray::SelectByLength(double minLength, double maxLength) const {
    return Select([minLength, maxLength](const QContour& c) {
        double len = c.Length();
        return len >= minLength && len <= maxLength;
    });
}

QContourArray QContourArray::SelectByArea(double minArea, double maxArea) const {
    return Select([minArea, maxArea](const QContour& c) {
        if (!c.IsClosed()) return false;
        double area = c.Area();
        return area >= minArea && area <= maxArea;
    });
}

QContourArray QContourArray::SelectByCircularity(double minCirc, double maxCirc) const {
    return Select([minCirc, maxCirc](const QContour& c) {
        if (!c.IsClosed() || c.Size() < 3) return false;
        double circ = c.Circularity();
        return circ >= minCirc && circ <= maxCirc;
    });
}

QContourArray QContourArray::SelectClosed() const {
    return Select([](const QContour& c) { return c.IsClosed(); });
}

QContourArray QContourArray::SelectOpen() const {
    return Select([](const QContour& c) { return !c.IsClosed(); });
}

QContourArray QContourArray::SelectRoots() const {
    return Select([](const QContour& c) { return !c.HasParent(); });
}

QContourArray QContourArray::Select(std::function<bool(const QContour&)> predicate) const {
    QContourArray result;
    for (const auto& contour : contours_) {
        if (predicate(contour)) {
            result.Add(contour);
        }
    }
    return result;
}

QContourArray QContourArray::SelectByIndex(const std::vector<size_t>& indices) const {
    QContourArray result;
    for (size_t idx : indices) {
        if (idx < contours_.size()) {
            result.Add(contours_[idx]);
        }
    }
    return result;
}

// =============================================================================
// Geometric Properties
// =============================================================================

double QContourArray::TotalLength() const {
    double total = 0.0;
    for (const auto& contour : contours_) {
        total += contour.Length();
    }
    return total;
}

double QContourArray::TotalArea() const {
    double total = 0.0;
    for (const auto& contour : contours_) {
        if (contour.IsClosed()) {
            total += contour.Area();
        }
    }
    return total;
}

Rect2d QContourArray::BoundingBox() const {
    if (contours_.empty()) {
        return Rect2d(0, 0, 0, 0);
    }

    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double maxY = std::numeric_limits<double>::lowest();

    for (const auto& contour : contours_) {
        if (contour.Empty()) continue;
        Rect2d bbox = contour.BoundingBox();
        minX = std::min(minX, bbox.x);
        minY = std::min(minY, bbox.y);
        maxX = std::max(maxX, bbox.x + bbox.width);
        maxY = std::max(maxY, bbox.y + bbox.height);
    }

    if (minX > maxX) {
        return Rect2d(0, 0, 0, 0);
    }

    return Rect2d(minX, minY, maxX - minX, maxY - minY);
}

Point2d QContourArray::Centroid() const {
    if (contours_.empty()) {
        return {0.0, 0.0};
    }

    double totalX = 0.0, totalY = 0.0;
    size_t totalPoints = 0;

    for (const auto& contour : contours_) {
        for (size_t i = 0; i < contour.Size(); ++i) {
            totalX += contour[i].x;
            totalY += contour[i].y;
        }
        totalPoints += contour.Size();
    }

    if (totalPoints == 0) {
        return {0.0, 0.0};
    }

    return {totalX / static_cast<double>(totalPoints),
            totalY / static_cast<double>(totalPoints)};
}

// =============================================================================
// Transformations
// =============================================================================

QContourArray QContourArray::Translate(double dx, double dy) const {
    QContourArray result;
    result.Reserve(contours_.size());
    for (const auto& contour : contours_) {
        result.Add(contour.Translate(dx, dy));
    }
    return result;
}

QContourArray QContourArray::Translate(const Point2d& offset) const {
    return Translate(offset.x, offset.y);
}

QContourArray QContourArray::Scale(double factor) const {
    return Scale(factor, factor);
}

QContourArray QContourArray::Scale(double sx, double sy) const {
    Point2d center = Centroid();
    return Scale(sx, sy, center);
}

QContourArray QContourArray::Scale(double sx, double sy, const Point2d& center) const {
    QContourArray result;
    result.Reserve(contours_.size());
    for (const auto& contour : contours_) {
        result.Add(contour.Scale(sx, sy, center));
    }
    return result;
}

QContourArray QContourArray::Rotate(double angle) const {
    Point2d center = Centroid();
    return Rotate(angle, center);
}

QContourArray QContourArray::Rotate(double angle, const Point2d& center) const {
    QContourArray result;
    result.Reserve(contours_.size());
    for (const auto& contour : contours_) {
        result.Add(contour.Rotate(angle, center));
    }
    return result;
}

QContourArray QContourArray::Transform(const QMatrix& matrix) const {
    QContourArray result;
    result.Reserve(contours_.size());
    for (const auto& contour : contours_) {
        result.Add(contour.Transform(matrix));
    }
    return result;
}

// =============================================================================
// Processing
// =============================================================================

QContourArray QContourArray::Smooth(double sigma) const {
    QContourArray result;
    result.Reserve(contours_.size());
    for (const auto& contour : contours_) {
        result.Add(contour.Smooth(sigma));
    }
    return result;
}

QContourArray QContourArray::Simplify(double tolerance) const {
    QContourArray result;
    result.Reserve(contours_.size());
    for (const auto& contour : contours_) {
        result.Add(contour.Simplify(tolerance));
    }
    return result;
}

QContourArray QContourArray::Resample(double interval) const {
    QContourArray result;
    result.Reserve(contours_.size());
    for (const auto& contour : contours_) {
        result.Add(contour.Resample(interval));
    }
    return result;
}

void QContourArray::CloseAll() {
    for (auto& contour : contours_) {
        contour.Close();
    }
}

void QContourArray::OpenAll() {
    for (auto& contour : contours_) {
        contour.Open();
    }
}

void QContourArray::ReverseAll() {
    for (auto& contour : contours_) {
        contour.Reverse();
    }
}

// =============================================================================
// Merging / Splitting
// =============================================================================

QContour QContourArray::Concatenate() const {
    QContour result;

    size_t totalPoints = 0;
    for (const auto& contour : contours_) {
        totalPoints += contour.Size();
    }
    result.Reserve(totalPoints);

    for (const auto& contour : contours_) {
        for (size_t i = 0; i < contour.Size(); ++i) {
            result.AddPoint(contour[i]);
        }
    }

    return result;
}

QContourArray QContourArray::SplitAtCorners(double maxCurvature) const {
    QContourArray result;

    for (const auto& contour : contours_) {
        if (contour.Size() < 3) {
            result.Add(contour);
            continue;
        }

        // Compute curvature if not already done
        QContour copy = contour.Clone();
        copy.ComputeCurvature(5);

        // Find split points
        std::vector<size_t> splitPoints;
        for (size_t i = 0; i < copy.Size(); ++i) {
            if (std::abs(copy.GetCurvature(i)) > maxCurvature) {
                splitPoints.push_back(i);
            }
        }

        if (splitPoints.empty()) {
            result.Add(contour);
        } else {
            // Split at each corner
            size_t start = 0;
            for (size_t splitIdx : splitPoints) {
                if (splitIdx > start) {
                    QContour segment;
                    for (size_t i = start; i <= splitIdx; ++i) {
                        segment.AddPoint(contour[i]);
                    }
                    if (segment.Size() >= 2) {
                        result.Add(segment);
                    }
                }
                start = splitIdx;
            }

            // Add final segment
            if (start < contour.Size() - 1) {
                QContour segment;
                for (size_t i = start; i < contour.Size(); ++i) {
                    segment.AddPoint(contour[i]);
                }
                if (segment.Size() >= 2) {
                    result.Add(segment);
                }
            }
        }
    }

    return result;
}

// =============================================================================
// Utilities
// =============================================================================

QContourArray QContourArray::Clone() const {
    return QContourArray(contours_);
}

void QContourArray::SortByLength(bool descending) {
    if (descending) {
        std::sort(contours_.begin(), contours_.end(),
                  [](const QContour& a, const QContour& b) {
                      return a.Length() > b.Length();
                  });
    } else {
        std::sort(contours_.begin(), contours_.end(),
                  [](const QContour& a, const QContour& b) {
                      return a.Length() < b.Length();
                  });
    }
}

void QContourArray::SortByArea(bool descending) {
    if (descending) {
        std::sort(contours_.begin(), contours_.end(),
                  [](const QContour& a, const QContour& b) {
                      return a.Area() > b.Area();
                  });
    } else {
        std::sort(contours_.begin(), contours_.end(),
                  [](const QContour& a, const QContour& b) {
                      return a.Area() < b.Area();
                  });
    }
}

void QContourArray::SortByPosition() {
    std::sort(contours_.begin(), contours_.end(),
              [](const QContour& a, const QContour& b) {
                  if (a.Empty() && b.Empty()) return false;
                  if (a.Empty()) return false;
                  if (b.Empty()) return true;
                  Rect2d bboxA = a.BoundingBox();
                  Rect2d bboxB = b.BoundingBox();
                  if (std::abs(bboxA.y - bboxB.y) > 10) {
                      return bboxA.y < bboxB.y;
                  }
                  return bboxA.x < bboxB.x;
              });
}

} // namespace Qi::Vision
