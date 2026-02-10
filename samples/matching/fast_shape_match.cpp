#include <QiVision/Matching/FastShapeModel.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Display/Draw.h>

#include <cstdio>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;
using namespace Qi::Vision::IO;

int main() {
    constexpr double kPi = 3.14159265358979323846;
    auto rad = [](double deg) { return deg * kPi / 180.0; };
    auto deg = [](double radVal) { return radVal * 180.0 / kPi; };

    std::printf("=== FastShapeModel Demo ===\n");

    QImage gray;
    ReadImageGray("tests/data/halcon_images/rings/mixed_01.png", gray);
    if (gray.Empty()) {
        std::printf("Failed to load image.\n");
        return 1;
    }

    // Same ROI style used in existing matching samples.
    Rect2i roi{367, 213, 89, 87};

    FastShapeModel model;
    FastShapeModelStrategy strategy;
    strategy.tAtLevel = {4, 8};
    strategy.weakThreshold = 10.0;
    strategy.strongThreshold = 55.0;
    strategy.numFeatures = 63;
    CreateFastShapeModel(
        gray, roi, model,
        0,             // auto levels
        0, rad(360),   // full rotation
        0,             // auto angle step
        strategy
    );

    if (!model.IsValid()) {
        std::printf("CreateFastShapeModel failed.\n");
        return 1;
    }

    std::vector<double> rows, cols, angles, scores;
    FindFastShapeModel(
        gray, model,
        0.35, 20, 0.5, 0.8,
        rows, cols, angles, scores
    );

    std::printf("Matches: %zu\n", rows.size());
    for (size_t i = 0; i < rows.size() && i < 5; ++i) {
        std::printf("  #%zu: row=%.2f col=%.2f angle=%.2fdeg score=%.4f\n",
                    i, rows[i], cols[i], deg(angles[i]), scores[i]);
    }

    QImage vis;
    Color::GrayToRgb(gray, vis);
    Draw::Rectangle(vis, roi, Scalar::Cyan(), 1);
    for (size_t i = 0; i < rows.size(); ++i) {
        Draw::Cross(vis, Point2d(cols[i], rows[i]), 10, angles[i], Scalar::Green(), 2);
    }

    WriteImage(vis, "tests/output/fast_shape_match_result.png");
    std::printf("Saved: tests/output/fast_shape_match_result.png\n");
    return 0;
}
