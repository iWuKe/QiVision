/**
 * @file test_shape_find.cpp
 * @brief Test ShapeModel creation + FindShapeModel with GUI visualization
 *
 * - m1.bmp as template: creates model, shows feature points in a window
 * - s1.png as search image: runs FindShapeModel, draws match bounding boxes
 *
 * Both windows display simultaneously. Press any key to exit.
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
#include <vector>

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

int main() {
    // =========================================================================
    // 1. Load template image and create model
    // =========================================================================
    const char* templatePath = "tests/data/imgs/m1.bmp";
    const char* searchPath   = "tests/data/imgs/s1.png";

    QImage templateGray;
    ReadImageGray(templatePath, templateGray);
    if (templateGray.Empty()) {
        std::fprintf(stderr, "Failed to load template: %s\n", templatePath);
        return 1;
    }
    std::printf("Template: %s (%d x %d)\n", templatePath,
                templateGray.Width(), templateGray.Height());

    SetShapeModelDebugCreateGlobal(true);

    ShapeModel model;
    CreateShapeModel(templateGray, model, 3, 0, RAD(360), 0,
                     "auto", "use_polarity", "40", 10);

    if (!model.IsValid()) {
        std::fprintf(stderr, "Model creation FAILED\n");
        return 1;
    }
    std::printf("Model created successfully\n");

    // Get model parameters
    int32_t numLevels;
    double angleStart, angleExtent, angleStep, scaleMin, scaleMax, scaleStep;
    std::string metric;
    GetShapeModelParams(model, numLevels, angleStart, angleExtent, angleStep,
                        scaleMin, scaleMax, scaleStep, metric);
    std::printf("  numLevels=%d, angleStep=%.2f deg, metric=%s\n",
                numLevels, angleStep * 180.0 / PI, metric.c_str());

    // =========================================================================
    // 2. Visualize model features (Level 1 = original resolution)
    // =========================================================================
    std::vector<ModelPoint> features = GetModelTransform(model, 1, 0.0, 1.0);
    std::printf("  Level 1 features: %zu points\n", features.size());

    double cx = templateGray.Width() * 0.5;
    double cy = templateGray.Height() * 0.5;

    QImage templateVis;
    Color::GrayToRgb(templateGray, templateVis);

    // Draw feature points as green pixels
    for (const auto& f : features) {
        int32_t px = static_cast<int32_t>(std::round(f.x + cx));
        int32_t py = static_cast<int32_t>(std::round(f.y + cy));
        SetPixelRgb(templateVis, px, py, 0, 255, 0);
    }

    // Yellow cross at template center
    Draw::Cross(templateVis, Point2d{cx, cy}, 5, 0, Scalar::Yellow(), 1);

    // Compute model bounding box from features (for drawing on search results)
    double fMinX = 1e9, fMaxX = -1e9, fMinY = 1e9, fMaxY = -1e9;
    for (const auto& f : features) {
        if (f.x < fMinX) fMinX = f.x;
        if (f.x > fMaxX) fMaxX = f.x;
        if (f.y < fMinY) fMinY = f.y;
        if (f.y > fMaxY) fMaxY = f.y;
    }
    double modelW = fMaxX - fMinX;
    double modelH = fMaxY - fMinY;
    std::printf("  Model bounding box: %.1f x %.1f (offset: %.1f, %.1f)\n",
                modelW, modelH, fMinX, fMinY);

    // Save template visualization to file
    WriteImage(templateVis, "result_template.png");
    std::printf("  Saved: result_template.png\n");

    // Display template in named window (non-blocking)
    DispImage(templateVis, "Template Features");

    // =========================================================================
    // 3. Load search image and run FindShapeModel
    // =========================================================================
    QImage searchGray;
    ReadImageGray(searchPath, searchGray);
    if (searchGray.Empty()) {
        std::fprintf(stderr, "Failed to load search image: %s\n", searchPath);
        return 1;
    }
    std::printf("\nSearch: %s (%d x %d)\n", searchPath,
                searchGray.Width(), searchGray.Height());

    std::vector<double> rows, cols, angles, scores;
    FindShapeModel(searchGray, model, 0, RAD(360), 0.5, 0, 0.5,
                   "least_squares", 0, 0.7,
                   rows, cols, angles, scores);

    std::printf("Found %zu matches:\n", rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        std::printf("  [%zu] row=%.2f col=%.2f angle=%.2f deg score=%.4f\n",
                    i, rows[i], cols[i], angles[i] * 180.0 / PI, scores[i]);
    }

    // =========================================================================
    // 4. Visualize search results
    // =========================================================================
    QImage searchVis;
    Color::GrayToRgb(searchGray, searchVis);

    if (rows.empty()) {
        Draw::Text(searchVis, 10, 20, "No matches found", Scalar::Red(), 2);
        std::printf("  No matches found!\n");
    }

    for (size_t i = 0; i < rows.size(); ++i) {
        double matchX = cols[i];
        double matchY = rows[i];
        double matchAngle = angles[i];

        // Draw rotated bounding box around match
        double boxCenterOffX = (fMinX + fMaxX) * 0.5;
        double boxCenterOffY = (fMinY + fMaxY) * 0.5;
        double cosA = std::cos(matchAngle);
        double sinA = std::sin(matchAngle);
        double drawCx = matchX + boxCenterOffX * cosA - boxCenterOffY * sinA;
        double drawCy = matchY + boxCenterOffX * sinA + boxCenterOffY * cosA;

        Draw::RotatedRectangle(searchVis, Point2d{drawCx, drawCy},
                               modelW + 4, modelH + 4, matchAngle,
                               Scalar::Green(), 2);

        // Cross at match center
        Draw::Cross(searchVis, Point2d{matchX, matchY}, 8, matchAngle,
                    Scalar::Yellow(), 1);

        // Draw rotated model feature points at match location
        std::vector<ModelPoint> rotFeatures = GetModelTransform(model, 1, matchAngle, 1.0);
        for (const auto& f : rotFeatures) {
            int32_t px = static_cast<int32_t>(std::round(f.x + matchX));
            int32_t py = static_cast<int32_t>(std::round(f.y + matchY));
            SetPixelRgb(searchVis, px, py, 0, 255, 0);
        }

        // Score label
        char label[64];
        std::snprintf(label, sizeof(label), "#%zu s=%.3f a=%.1f",
                      i, scores[i], angles[i] * 180.0 / PI);
        Draw::Text(searchVis, static_cast<int32_t>(matchX - modelW * 0.5),
                   static_cast<int32_t>(matchY - modelH * 0.5 - 12),
                   label, Scalar::Cyan(), 1);
    }

    // Save search results to file
    WriteImage(searchVis, "result_search.png");
    std::printf("  Saved: result_search.png\n");

    // Display search results in named window (non-blocking)
    DispImage(searchVis, "Search Results");

    // Wait for key press on any window to exit
    std::printf("\nPress any key in a window to exit...\n");
    std::fflush(stdout);
    WaitKey();

    return 0;
}
