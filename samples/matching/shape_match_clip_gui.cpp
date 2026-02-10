/**
 * @file shape_match_clip_gui.cpp
 * @brief Interactive shape matching on clip image
 *
 * Default behavior:
 * - Load tests/data/halcon_images/clip.png
 * - Draw ROI interactively
 * - Create shape model from ROI
 * - Match in the same image (numMatches=20, minScore=0.9)
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/GUI/Window.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Matching/MatchTypes.h>
#include <QiVision/Matching/ShapeModel.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::GUI;
using namespace Qi::Vision::Matching;

namespace {

std::string NormalizePath(std::string path) {
    // Support Windows UNC path style:
    // \\wsl.localhost\Ubuntu\home\zq\QiVision\...
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

} // namespace

int main(int argc, char* argv[]) {
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

    Window win("Shape Match Clip - Draw ROI");
    win.SetAutoResize(true);
    win.EnablePixelInfo(true);
    win.DispImage(gray);

    std::cout << "Image: " << imagePath << "\n";
    std::cout << "Step1: Drag mouse to draw ROI. ESC to cancel.\n";
    ROIResult roiResult = win.DrawRectangle();
    if (!roiResult.valid) {
        std::cerr << "ROI drawing cancelled." << std::endl;
        return 1;
    }

    Rect2i roi = ToRect(roiResult);
    if (roi.width < 5 || roi.height < 5) {
        std::cerr << "ROI too small. Need at least 5x5." << std::endl;
        return 1;
    }

    ShapeModel model;
    CreateShapeModel(
        gray, roi, model,
        4,                  // numLevels
        0.0, RAD(360), 0.0, // full angle range, auto step
        "auto",
        "use_polarity",
        "auto", 8.0
    );
    if (!model.IsValid()) {
        std::cerr << "CreateShapeModel failed." << std::endl;
        return 1;
    }

    std::vector<double> rows, cols, angles, scores;
    FindShapeModel(
        gray, model,
        0.0, RAD(360),
        0.9,                // default min score
        20,                 // max matches
        0.9,                // max overlap
        "least_squares",
        0,
        0.9,
        rows, cols, angles, scores
    );

    QImage vis;
    Color::GrayToRgb(gray, vis);
    Draw::Rectangle(vis, roi, Scalar::Cyan(), 1);

    std::vector<MatchResult> matches;
    matches.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        MatchResult m;
        m.x = cols[i];
        m.y = rows[i];
        m.angle = angles[i];
        m.score = scores[i];
        matches.push_back(m);
    }
    Draw::ShapeMatchingResults(vis, model, matches, Scalar::Green(), Scalar::Red(), 1, 0.5);

    std::cout << "Matches: " << rows.size() << " (max 20, minScore 0.9, maxOverlap 0.9)\n";
    if (!scores.empty()) {
        std::cout << "Top score: " << scores[0] << "\n";
    }

    WriteImage(vis, "tests/output/shape_match_clip_gui_result.png");
    win.SetTitle("Shape Match Result - clip");
    win.DispImage(vis);
    std::cout << "Saved: tests/output/shape_match_clip_gui_result.png\n";
    std::cout << "Press any key to close..." << std::endl;
    win.WaitKey(0);
    return 0;
}
