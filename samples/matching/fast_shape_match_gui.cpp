/**
 * @file fast_shape_match_gui.cpp
 * @brief Interactive fast shape matching demo.
 *
 * Default behavior:
 * - Load tests/data/halcon_images/clip.png
 * - Draw ROI interactively to create template
 * - Match in same image (maxMatches=20, minScore=0.6)
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/GUI/Window.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Matching/FastShapeModel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::GUI;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::Matching;

namespace {

std::string NormalizePath(std::string path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    const std::string prefix = "//wsl.localhost/Ubuntu";
    if (path.rfind(prefix, 0) == 0) {
        path = path.substr(prefix.size());
        if (path.empty() || path[0] != '/') {
            path = "/" + path;
        }
    }
    return path;
}

Rect2i ToRect(const ROIResult& roi) {
    Rect2i r;
    r.x = static_cast<int32_t>(std::floor(std::min(roi.col1, roi.col2)));
    r.y = static_cast<int32_t>(std::floor(std::min(roi.row1, roi.row2)));
    r.width = static_cast<int32_t>(std::ceil(std::abs(roi.col2 - roi.col1)));
    r.height = static_cast<int32_t>(std::ceil(std::abs(roi.row2 - roi.row1)));
    return r;
}

double Cross2D(const Point2d& o, const Point2d& a, const Point2d& b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

std::vector<Point2d> ConvexHull(std::vector<Point2d> pts) {
    if (pts.size() < 3) return pts;
    std::sort(pts.begin(), pts.end(), [](const Point2d& a, const Point2d& b) {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    });
    std::vector<Point2d> h;
    h.reserve(pts.size() * 2);
    for (const auto& p : pts) {
        while (h.size() >= 2 &&
               Cross2D(h[h.size() - 2], h[h.size() - 1], p) <= 0.0) {
            h.pop_back();
        }
        h.push_back(p);
    }
    size_t lowerSize = h.size();
    for (int i = static_cast<int>(pts.size()) - 2; i >= 0; --i) {
        const auto& p = pts[static_cast<size_t>(i)];
        while (h.size() > lowerSize &&
               Cross2D(h[h.size() - 2], h[h.size() - 1], p) <= 0.0) {
            h.pop_back();
        }
        h.push_back(p);
    }
    if (!h.empty()) h.pop_back();
    return h;
}

} // namespace

int main(int argc, char* argv[]) {
    constexpr double kPi = 3.14159265358979323846;
    auto rad = [](double deg) { return deg * kPi / 180.0; };

    std::string imagePath = "tests/data/halcon_images/clip.png";
    if (argc > 1) {
        imagePath = argv[1];
    }
    imagePath = NormalizePath(imagePath);

    QImage gray;
    ReadImageGray(imagePath, gray);
    if (gray.Empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        std::cerr << "Usage: " << argv[0] << " [image_path]" << std::endl;
        return 1;
    }

    Window win("Fast Shape Match - Draw ROI", 1200, 860);
    win.SetAutoResize(false);
    win.Move(60, 60);
    win.Resize(1200, 860);
    win.EnablePixelInfo(true);
    win.DispImage(gray, ScaleMode::Fit);
    for (int i = 0; i < 20 && win.IsOpen(); ++i) {
        win.WaitKey(10);
    }

    if (!win.IsOpen()) {
        std::cerr << "Window failed to open (check DISPLAY/X11)." << std::endl;
        return 1;
    }

    std::cout << "Image: " << imagePath << "\n";
    std::cout << "Step1: Drag mouse to draw ROI. ESC to cancel.\n";
    ROIResult roiRes = win.DrawRectangle();
    if (!roiRes.valid) {
        std::cerr << "ROI drawing cancelled." << std::endl;
        return 1;
    }

    Rect2i roi = ToRect(roiRes);
    if (roi.width < 8 || roi.height < 8) {
        std::cerr << "ROI too small. Need at least 8x8." << std::endl;
        return 1;
    }

    FastShapeModel model;
    FastShapeModelStrategy strategy;
    strategy.tAtLevel = {4, 8};
    strategy.weakThreshold = 10.0;
    strategy.strongThreshold = 55.0;
    strategy.numFeatures = 63;
    CreateFastShapeModel(
        gray, roi, model,
        0,             // auto levels
        0.0, rad(360), // full rotation
        0.0,           // auto angle step
        strategy
    );
    if (!model.IsValid()) {
        std::cerr << "CreateFastShapeModel failed." << std::endl;
        return 1;
    }

    std::vector<double> rows, cols, angles, scores;
    FindFastShapeModel(
        gray, model,
        0.6, // default min score
        20,  // max matches
        0.9,
        0.9,
        rows, cols, angles, scores
    );

    QImage vis;
    Color::GrayToRgb(gray, vis);
    Draw::Rectangle(vis, roi, Scalar::Green(), 1);
    std::vector<Point2d> modelFeat;
    GetFastShapeModelFeaturePoints(model, modelFeat);

    double displayScoreFloor = 0.6;
    if (!scores.empty()) {
        // Keep only visually reliable matches to avoid non-fitting overlapping contours.
        displayScoreFloor = std::max(0.6, scores[0] * 0.78);
    }

    for (size_t i = 0; i < rows.size(); ++i) {
        if (scores[i] < displayScoreFloor) {
            continue;
        }

        Draw::RotatedRectangle(vis, Point2d(cols[i], rows[i]),
                               static_cast<double>(roi.width),
                               static_cast<double>(roi.height),
                               angles[i], Scalar::Green(), 1);

        double c = std::cos(angles[i]);
        double s = std::sin(angles[i]);
        std::vector<Point2d> transformed;
        transformed.reserve(modelFeat.size());
        for (const auto& p : modelFeat) {
            double x = c * p.x - s * p.y + cols[i];
            double y = s * p.x + c * p.y + rows[i];
            transformed.emplace_back(x, y);
        }

        if (transformed.size() >= 3) {
            auto hull = ConvexHull(std::move(transformed));
            Draw::Polyline(vis, hull, Scalar::Green(), 1, true);
        }
    }

    std::cout << "Matches: " << rows.size() << " (max 20, minScore 0.6)\n";
    if (!scores.empty()) {
        std::cout << "Top score: " << scores[0] << "\n";
    }

    WriteImage(vis, "tests/output/fast_shape_match_gui_result.png");
    std::cout << "Saved: tests/output/fast_shape_match_gui_result.png\n";
    win.SetTitle("Fast Shape Match Result");
    win.DispImage(vis, ScaleMode::Fit);
    for (int i = 0; i < 10 && win.IsOpen(); ++i) {
        win.WaitKey(10);
    }
    std::cout << "Press any key to close..." << std::endl;
    win.WaitKey(0);
    return 0;
}
