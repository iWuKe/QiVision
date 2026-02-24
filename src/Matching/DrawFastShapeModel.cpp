#include <QiVision/Matching/DrawFastShapeModel.h>

#include <QiVision/Display/Draw.h>
#include <QiVision/Matching/FastShapeModel.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace Qi::Vision::Matching {

void DrawFastShapeModelResults(
    QImage& image,
    const FastShapeModel& model,
    const std::vector<double>& rows,
    const std::vector<double>& cols,
    const std::vector<double>& angles,
    const std::vector<double>& /*scores*/,
    const std::vector<double>& scales,
    const Scalar& color,
    int32_t thickness)
{
    if (image.Empty() || !model.IsValid()) {
        return;
    }

    std::vector<Point2d> modelFeat;
    GetFastShapeModelFeaturePoints(model, modelFeat);

    int32_t tmplW = 0, tmplH = 0;
    GetFastShapeModelTemplateSize(model, tmplW, tmplH);

    constexpr int32_t kCrossSize = 4;
    constexpr double  kPi4       = 0.7853981633974483; // π/4 — rotates + into X

    const size_t N = std::min(rows.size(), std::min(cols.size(), angles.size()));
    for (size_t i = 0; i < N; ++i) {
        const double matchCX = cols[i];
        const double matchCY = rows[i];
        const double ang     = angles[i];
        const double scale   = (i < scales.size()) ? scales[i] : 1.0;
        const double cosA    = std::cos(ang);
        const double sinA    = std::sin(ang);

        auto transform = [&](const Point2d& p) -> Point2d {
            const double fx = p.x * scale;
            const double fy = p.y * scale;
            return {cosA * fx - sinA * fy + matchCX,
                    sinA * fx + cosA * fy + matchCY};
        };

        // 1. Rotated bounding rectangle
        Draw::RotatedRectangle(image, Point2d(matchCX, matchCY),
                               static_cast<double>(tmplW) * scale,
                               static_cast<double>(tmplH) * scale,
                               ang, color, thickness);

        // 2. X-shaped cross at each feature point
        for (const auto& p : modelFeat) {
            Draw::Cross(image, transform(p), kCrossSize, kPi4, color, thickness);
        }
    }
}

} // namespace Qi::Vision::Matching
