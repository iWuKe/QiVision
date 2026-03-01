/**
 * @file test_create_m1.cpp
 * @brief Test CreateShapeModel with multiple templates - visualize feature points
 *
 * Visualization matches the original decompiled demo style:
 * - Single green pixels at each feature point position (like demo.cpp ShowFeatures)
 * - Yellow cross at image center for reference
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Matching/ShapeModel.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/GUI/Window.h>

#include <cstdio>
#include <cmath>
#include <string>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::GUI;

// Set a single pixel in an RGB image
static void SetPixelRgb(QImage& img, int32_t x, int32_t y, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || y < 0 || x >= img.Width() || y >= img.Height()) return;
    uint8_t* row = static_cast<uint8_t*>(img.Data()) + y * img.Stride();
    row[x * 3 + 0] = r;
    row[x * 3 + 1] = g;
    row[x * 3 + 2] = b;
}

// Draw a 3x3 cross at (x,y) for better visibility
static void DrawPixelCross(QImage& img, int32_t x, int32_t y, uint8_t r, uint8_t g, uint8_t b) {
    SetPixelRgb(img, x, y, r, g, b);
    SetPixelRgb(img, x - 1, y, r, g, b);
    SetPixelRgb(img, x + 1, y, r, g, b);
    SetPixelRgb(img, x, y - 1, r, g, b);
    SetPixelRgb(img, x, y + 1, r, g, b);
}

static void VisualizeModel(const char* imagePath) {
    QImage gray;
    ReadImageGray(imagePath, gray);
    if (gray.Empty()) {
        std::fprintf(stderr, "Failed to load: %s\n", imagePath);
        return;
    }
    std::printf("\n=== %s (%d x %d) ===\n", imagePath, gray.Width(), gray.Height());

    ShapeModel model;
    CreateShapeModel(gray, model, 3, 0, RAD(360), 0,
                     "auto", "use_polarity", "40", 10);

    if (!model.IsValid()) {
        std::printf("  Model creation FAILED\n");
        return;
    }

    // Extract short name (without extension) from path
    std::string path(imagePath);
    std::string nameExt = path.substr(path.find_last_of("/\\") + 1);
    std::string name = nameExt.substr(0, nameExt.find_last_of('.'));

    // Only visualize Level 1 (original resolution)
    std::vector<ModelPoint> features;
    try {
        features = GetModelTransform(model, 1, 0.0, 1.0);
    } catch (...) {}

    if (features.empty()) {
        std::printf("  No features at Level 1\n");
        return;
    }

    double cx = gray.Width() * 0.5;
    double cy = gray.Height() * 0.5;

    QImage vis;
    Color::GrayToRgb(gray, vis);

    // Draw feature points as green 3x3 crosses for visibility
    for (const auto& f : features) {
        int32_t px = static_cast<int32_t>(std::round(f.x + cx));
        int32_t py = static_cast<int32_t>(std::round(f.y + cy));
        DrawPixelCross(vis, px, py, 0, 255, 0);
    }

    // Yellow cross at image center for reference
    Draw::Cross(vis, Point2d{cx, cy}, 5, 0, Scalar::Yellow(), 1);

    std::printf("  Level 1 (%dx%d): %zu features\n",
                gray.Width(), gray.Height(), features.size());

    // Show in GUI window
    std::string title = name + " (" + std::to_string(features.size()) + " pts)";
    Window::ShowImage(vis, title);
}

int main() {
    SetShapeModelDebugCreateGlobal(true);

    const char* templates[] = {
        "tests/data/imgs/m1.bmp",
        "tests/data/imgs/m_2.bmp",
        "tests/data/imgs/m_3.bmp",
        "tests/data/imgs/m2.bmp",
    };

    for (const char* path : templates) {
        VisualizeModel(path);
    }
    std::fflush(stdout);

    // Wait for user to close windows
    Window::ShowImage(QImage(), "Press any key to exit");

    return 0;
}
