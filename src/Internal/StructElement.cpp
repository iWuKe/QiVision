#include <QiVision/Internal/StructElement.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

namespace Qi::Vision::Internal {

// =============================================================================
// Implementation Details
// =============================================================================

struct StructElement::Impl {
    int32_t width = 0;
    int32_t height = 0;
    int32_t anchorX = 0;
    int32_t anchorY = 0;
    StructElementShape shape = StructElementShape::Custom;
    std::vector<uint8_t> mask;  // Row-major, 1 = in element
    std::vector<Point2i> coords;  // Coordinates relative to anchor

    Impl() = default;

    Impl(int32_t w, int32_t h, int32_t ax, int32_t ay, StructElementShape s)
        : width(w), height(h), anchorX(ax), anchorY(ay), shape(s) {
        mask.resize(static_cast<size_t>(w) * h, 0);
    }

    bool At(int32_t row, int32_t col) const {
        if (row < 0 || row >= height || col < 0 || col >= width) return false;
        return mask[static_cast<size_t>(row) * width + col] != 0;
    }

    void Set(int32_t row, int32_t col, bool value) {
        if (row < 0 || row >= height || col < 0 || col >= width) return;
        mask[static_cast<size_t>(row) * width + col] = value ? 1 : 0;
    }

    void UpdateCoords() {
        coords.clear();
        for (int32_t r = 0; r < height; ++r) {
            for (int32_t c = 0; c < width; ++c) {
                if (At(r, c)) {
                    // Store relative to anchor
                    coords.push_back({c - anchorX, r - anchorY});
                }
            }
        }
    }
};

// =============================================================================
// Constructors
// =============================================================================

StructElement::StructElement() : impl_(std::make_unique<Impl>()) {}

StructElement::StructElement(const StructElement& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

StructElement::StructElement(StructElement&& other) noexcept = default;

StructElement::~StructElement() = default;

StructElement& StructElement::operator=(const StructElement& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

StructElement& StructElement::operator=(StructElement&& other) noexcept = default;

// =============================================================================
// Factory Methods - Basic Shapes
// =============================================================================

StructElement StructElement::Rectangle(int32_t width, int32_t height) {
    if (width <= 0 || height <= 0) return StructElement();

    StructElement se;
    se.impl_ = std::make_unique<Impl>(width, height, width / 2, height / 2,
                                       StructElementShape::Rectangle);

    // Fill all pixels
    std::fill(se.impl_->mask.begin(), se.impl_->mask.end(), 1);
    se.impl_->UpdateCoords();

    return se;
}

StructElement StructElement::Square(int32_t size) {
    return Rectangle(size, size);
}

StructElement StructElement::Ellipse(int32_t radiusX, int32_t radiusY) {
    if (radiusX <= 0 || radiusY <= 0) return StructElement();

    int32_t width = 2 * radiusX + 1;
    int32_t height = 2 * radiusY + 1;

    StructElement se;
    se.impl_ = std::make_unique<Impl>(width, height, radiusX, radiusY,
                                       StructElementShape::Ellipse);

    // Fill ellipse: (x/rx)^2 + (y/ry)^2 <= 1
    double rxSq = static_cast<double>(radiusX) * radiusX;
    double rySq = static_cast<double>(radiusY) * radiusY;

    for (int32_t r = 0; r < height; ++r) {
        for (int32_t c = 0; c < width; ++c) {
            double dx = c - radiusX;
            double dy = r - radiusY;
            if ((dx * dx / rxSq) + (dy * dy / rySq) <= 1.0) {
                se.impl_->Set(r, c, true);
            }
        }
    }

    se.impl_->UpdateCoords();
    return se;
}

StructElement StructElement::Circle(int32_t radius) {
    return Ellipse(radius, radius);
}

StructElement StructElement::Cross(int32_t armLength, int32_t thickness) {
    if (armLength <= 0 || thickness <= 0) return StructElement();

    int32_t size = 2 * armLength + 1;
    int32_t halfThick = thickness / 2;

    StructElement se;
    se.impl_ = std::make_unique<Impl>(size, size, armLength, armLength,
                                       StructElementShape::Cross);

    // Horizontal arm
    for (int32_t r = armLength - halfThick; r <= armLength + halfThick; ++r) {
        if (r >= 0 && r < size) {
            for (int32_t c = 0; c < size; ++c) {
                se.impl_->Set(r, c, true);
            }
        }
    }

    // Vertical arm
    for (int32_t c = armLength - halfThick; c <= armLength + halfThick; ++c) {
        if (c >= 0 && c < size) {
            for (int32_t r = 0; r < size; ++r) {
                se.impl_->Set(r, c, true);
            }
        }
    }

    se.impl_->UpdateCoords();
    return se;
}

StructElement StructElement::Diamond(int32_t radius) {
    if (radius <= 0) return StructElement();

    int32_t size = 2 * radius + 1;

    StructElement se;
    se.impl_ = std::make_unique<Impl>(size, size, radius, radius,
                                       StructElementShape::Diamond);

    // Diamond: |x| + |y| <= radius
    for (int32_t r = 0; r < size; ++r) {
        for (int32_t c = 0; c < size; ++c) {
            int32_t dx = std::abs(c - radius);
            int32_t dy = std::abs(r - radius);
            if (dx + dy <= radius) {
                se.impl_->Set(r, c, true);
            }
        }
    }

    se.impl_->UpdateCoords();
    return se;
}

StructElement StructElement::Line(int32_t length, double angle) {
    if (length <= 0) return StructElement();

    double cosA = std::cos(angle);
    double sinA = std::sin(angle);

    // Compute bounding box based on line endpoints
    int32_t halfLen = length / 2;
    double endX = halfLen * cosA;
    double endY = halfLen * sinA;

    // Find bounding box that contains both endpoints
    int32_t minX = static_cast<int32_t>(std::floor(-std::abs(endX)));
    int32_t maxX = static_cast<int32_t>(std::ceil(std::abs(endX)));
    int32_t minY = static_cast<int32_t>(std::floor(-std::abs(endY)));
    int32_t maxY = static_cast<int32_t>(std::ceil(std::abs(endY)));

    int32_t width = maxX - minX + 1;
    int32_t height = maxY - minY + 1;
    int32_t cx = -minX;  // Anchor so that (0,0) relative maps to line center
    int32_t cy = -minY;

    StructElement se;
    se.impl_ = std::make_unique<Impl>(width, height, cx, cy,
                                       StructElementShape::Line);

    // Draw line by stepping through parametric form
    // Ensure we hit integer points along the line
    int32_t numPoints = length + 1;
    for (int i = 0; i < numPoints; ++i) {
        // t goes from -halfLen to +halfLen
        double t = -halfLen + static_cast<double>(i) * length / (numPoints - 1);
        int32_t c = cx + static_cast<int32_t>(std::round(t * cosA));
        int32_t r = cy + static_cast<int32_t>(std::round(t * sinA));
        if (r >= 0 && r < height && c >= 0 && c < width) {
            se.impl_->Set(r, c, true);
        }
    }

    se.impl_->UpdateCoords();
    return se;
}

StructElement StructElement::Octagon(int32_t radius) {
    if (radius <= 0) return StructElement();

    int32_t size = 2 * radius + 1;

    StructElement se;
    se.impl_ = std::make_unique<Impl>(size, size, radius, radius,
                                       StructElementShape::Octagon);

    // Octagon approximation
    double cutoff = radius * 0.414;  // tan(22.5)

    for (int32_t r = 0; r < size; ++r) {
        for (int32_t c = 0; c < size; ++c) {
            int32_t dx = std::abs(c - radius);
            int32_t dy = std::abs(r - radius);
            // Inside if within square and cut corners
            if (dx <= radius && dy <= radius &&
                (dx + dy) <= radius + cutoff) {
                se.impl_->Set(r, c, true);
            }
        }
    }

    se.impl_->UpdateCoords();
    return se;
}

// =============================================================================
// Factory Methods - From Data
// =============================================================================

StructElement StructElement::FromMask(const QImage& mask,
                                        int32_t anchorX,
                                        int32_t anchorY) {
    if (mask.Empty()) return StructElement();

    int32_t width = mask.Width();
    int32_t height = mask.Height();

    // Default anchor to center
    if (anchorX < 0) anchorX = width / 2;
    if (anchorY < 0) anchorY = height / 2;

    StructElement se;
    se.impl_ = std::make_unique<Impl>(width, height, anchorX, anchorY,
                                       StructElementShape::Custom);

    // Copy non-zero pixels
    for (int32_t r = 0; r < height; ++r) {
        const uint8_t* row = static_cast<const uint8_t*>(mask.RowPtr(r));
        for (int32_t c = 0; c < width; ++c) {
            se.impl_->Set(r, c, row[c] != 0);
        }
    }

    se.impl_->UpdateCoords();
    return se;
}

StructElement StructElement::FromRegion(const QRegion& region,
                                          int32_t anchorX,
                                          int32_t anchorY) {
    if (region.Empty()) return StructElement();

    Rect2i bbox = region.BoundingBox();

    // Default anchor to center
    if (anchorX < 0) anchorX = bbox.width / 2;
    if (anchorY < 0) anchorY = bbox.height / 2;

    StructElement se;
    se.impl_ = std::make_unique<Impl>(bbox.width, bbox.height, anchorX, anchorY,
                                       StructElementShape::Custom);

    // Set pixels from runs
    for (const auto& run : region.Runs()) {
        int32_t r = run.row - bbox.y;
        for (int32_t c = run.colBegin - bbox.x; c < run.colEnd - bbox.x; ++c) {
            se.impl_->Set(r, c, true);
        }
    }

    se.impl_->UpdateCoords();
    return se;
}

StructElement StructElement::FromCoordinates(const std::vector<Point2i>& coords) {
    if (coords.empty()) return StructElement();

    // Find bounding box
    int32_t minX = coords[0].x, maxX = coords[0].x;
    int32_t minY = coords[0].y, maxY = coords[0].y;
    for (const auto& pt : coords) {
        minX = std::min(minX, pt.x);
        maxX = std::max(maxX, pt.x);
        minY = std::min(minY, pt.y);
        maxY = std::max(maxY, pt.y);
    }

    int32_t width = maxX - minX + 1;
    int32_t height = maxY - minY + 1;

    // Anchor at (0,0) offset
    int32_t anchorX = -minX;
    int32_t anchorY = -minY;

    StructElement se;
    se.impl_ = std::make_unique<Impl>(width, height, anchorX, anchorY,
                                       StructElementShape::Custom);

    for (const auto& pt : coords) {
        int32_t r = pt.y - minY;
        int32_t c = pt.x - minX;
        se.impl_->Set(r, c, true);
    }

    se.impl_->UpdateCoords();
    return se;
}

// =============================================================================
// Properties
// =============================================================================

bool StructElement::Empty() const {
    return impl_->width == 0 || impl_->height == 0;
}

int32_t StructElement::Width() const { return impl_->width; }
int32_t StructElement::Height() const { return impl_->height; }
Size2i StructElement::Size() const { return {impl_->width, impl_->height}; }

int32_t StructElement::AnchorX() const { return impl_->anchorX; }
int32_t StructElement::AnchorY() const { return impl_->anchorY; }
Point2i StructElement::Anchor() const { return {impl_->anchorX, impl_->anchorY}; }

StructElementShape StructElement::Shape() const { return impl_->shape; }

size_t StructElement::PixelCount() const {
    return impl_->coords.size();
}

bool StructElement::IsSeparable() const {
    // Only rectangles are easily separable
    return impl_->shape == StructElementShape::Rectangle;
}

bool StructElement::IsSymmetric() const {
    if (Empty()) return true;

    // Check if element is symmetric around anchor
    for (const auto& pt : impl_->coords) {
        // Check if (-x, -y) also exists
        bool found = false;
        for (const auto& other : impl_->coords) {
            if (other.x == -pt.x && other.y == -pt.y) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

// =============================================================================
// Data Access
// =============================================================================

bool StructElement::Contains(int32_t row, int32_t col) const {
    // Convert from anchor-relative to mask coordinates
    int32_t maskRow = row + impl_->anchorY;
    int32_t maskCol = col + impl_->anchorX;
    return impl_->At(maskRow, maskCol);
}

std::vector<Point2i> StructElement::GetCoordinates() const {
    return impl_->coords;
}

QImage StructElement::GetMask() const {
    if (Empty()) return QImage();

    QImage mask(impl_->width, impl_->height, PixelType::UInt8, ChannelType::Gray);
    for (int32_t r = 0; r < impl_->height; ++r) {
        uint8_t* row = static_cast<uint8_t*>(mask.RowPtr(r));
        for (int32_t c = 0; c < impl_->width; ++c) {
            row[c] = impl_->At(r, c) ? 255 : 0;
        }
    }
    return mask;
}

QRegion StructElement::GetRegion() const {
    if (Empty()) return QRegion();

    std::vector<QRegion::Run> runs;
    for (int32_t r = 0; r < impl_->height; ++r) {
        int32_t runStart = -1;
        for (int32_t c = 0; c < impl_->width; ++c) {
            if (impl_->At(r, c)) {
                if (runStart < 0) runStart = c;
            } else if (runStart >= 0) {
                runs.emplace_back(r, runStart, c);
                runStart = -1;
            }
        }
        if (runStart >= 0) {
            runs.emplace_back(r, runStart, impl_->width);
        }
    }
    return QRegion(runs);
}

// =============================================================================
// Transformations
// =============================================================================

StructElement StructElement::Reflect() const {
    if (Empty()) return StructElement();

    StructElement se;
    se.impl_ = std::make_unique<Impl>(impl_->width, impl_->height,
                                       impl_->width - 1 - impl_->anchorX,
                                       impl_->height - 1 - impl_->anchorY,
                                       impl_->shape);

    // Reflect: (r, c) -> (h-1-r, w-1-c)
    for (int32_t r = 0; r < impl_->height; ++r) {
        for (int32_t c = 0; c < impl_->width; ++c) {
            se.impl_->Set(impl_->height - 1 - r, impl_->width - 1 - c,
                          impl_->At(r, c));
        }
    }

    se.impl_->UpdateCoords();
    return se;
}

StructElement StructElement::Transpose() const {
    if (Empty()) return StructElement();

    StructElement se;
    se.impl_ = std::make_unique<Impl>(impl_->height, impl_->width,
                                       impl_->anchorY, impl_->anchorX,
                                       impl_->shape);

    // Transpose: (r, c) -> (c, r)
    for (int32_t r = 0; r < impl_->height; ++r) {
        for (int32_t c = 0; c < impl_->width; ++c) {
            se.impl_->Set(c, r, impl_->At(r, c));
        }
    }

    se.impl_->UpdateCoords();
    return se;
}

StructElement StructElement::Rotate(double angle) const {
    if (Empty()) return StructElement();

    double cosA = std::cos(angle);
    double sinA = std::sin(angle);

    // Compute rotated coordinates
    std::vector<Point2i> rotatedCoords;
    for (const auto& pt : impl_->coords) {
        double nx = pt.x * cosA - pt.y * sinA;
        double ny = pt.x * sinA + pt.y * cosA;
        rotatedCoords.push_back({
            static_cast<int32_t>(std::round(nx)),
            static_cast<int32_t>(std::round(ny))
        });
    }

    return FromCoordinates(rotatedCoords);
}

StructElement StructElement::Scale(double scaleX, double scaleY) const {
    if (Empty() || scaleX <= 0 || scaleY <= 0) return StructElement();

    // Scale coordinates
    std::vector<Point2i> scaledCoords;
    for (const auto& pt : impl_->coords) {
        int32_t nx = static_cast<int32_t>(std::round(pt.x * scaleX));
        int32_t ny = static_cast<int32_t>(std::round(pt.y * scaleY));
        scaledCoords.push_back({nx, ny});
    }

    // Remove duplicates
    std::sort(scaledCoords.begin(), scaledCoords.end(),
              [](const Point2i& a, const Point2i& b) {
                  if (a.y != b.y) return a.y < b.y;
                  return a.x < b.x;
              });
    scaledCoords.erase(std::unique(scaledCoords.begin(), scaledCoords.end(),
                                    [](const Point2i& a, const Point2i& b) {
                                        return a.x == b.x && a.y == b.y;
                                    }),
                       scaledCoords.end());

    return FromCoordinates(scaledCoords);
}

// =============================================================================
// Decomposition
// =============================================================================

bool StructElement::CanDecompose() const {
    return impl_->shape == StructElementShape::Rectangle;
}

bool StructElement::Decompose(StructElement& horizontal, StructElement& vertical) const {
    if (impl_->shape != StructElementShape::Rectangle) {
        return false;
    }

    horizontal = Rectangle(impl_->width, 1);
    vertical = Rectangle(1, impl_->height);
    return true;
}

std::vector<StructElement> StructElement::DecomposeToSequence() const {
    std::vector<StructElement> result;

    if (impl_->shape == StructElementShape::Rectangle) {
        // Decompose large rectangles into smaller ones
        int32_t w = impl_->width;
        int32_t h = impl_->height;

        // Simple decomposition: 1D horizontal + 1D vertical
        if (w > 1) result.push_back(Rectangle(w, 1));
        if (h > 1) result.push_back(Rectangle(1, h));

        if (result.empty()) {
            result.push_back(*this);
        }
    } else {
        // No decomposition for other shapes
        result.push_back(*this);
    }

    return result;
}

// =============================================================================
// Convenience Functions
// =============================================================================

StructElement SE_Cross3() {
    return StructElement::Cross(1, 1);
}

StructElement SE_Square3() {
    return StructElement::Square(3);
}

StructElement SE_Disk5() {
    return StructElement::Circle(2);
}

std::pair<StructElement, StructElement> CreateHitMissSE(
    const std::vector<Point2i>& hit,
    const std::vector<Point2i>& miss) {
    return {StructElement::FromCoordinates(hit),
            StructElement::FromCoordinates(miss)};
}

} // namespace Qi::Vision::Internal
