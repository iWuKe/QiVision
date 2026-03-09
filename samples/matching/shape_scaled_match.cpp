/**
 * @file shape_scaled_match.cpp
 * @brief ShapeModel scaled matching demo
 *
 * - m1.bmp as template: creates model, shows feature points
 * - s1.png as search image: runs FindScaledShapeModel (scale 0.5~1.5)
 *
 * Results are saved to tests/data/imgs/results/
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Matching/ShapeModel.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/GUI/Window.h>

#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::GUI;
namespace fs = std::filesystem;

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

int main() {
    std::printf("========================================\n");
    std::printf("  ShapeModel Scaled Matching Demo\n");
    std::printf("========================================\n\n");

    const char* templatePath = "tests/data/imgs/m1.bmp";
    const char* searchPath   = "tests/data/imgs/s1.png";

    const std::string resultsFolder = "tests/data/imgs/results";
    fs::create_directories(resultsFolder);
    std::printf("Results will be saved to: %s\n\n", resultsFolder.c_str());

    // =========================================================================
    // 1. Load template image and create model
    // =========================================================================
    QImage templateGray;
    ReadImageGray(templatePath, templateGray);
    if (templateGray.Empty()) {
        std::fprintf(stderr, "Failed to load template: %s\n", templatePath);
        return 1;
    }
    std::printf("Template: %s (%d x %d)\n", templatePath,
                templateGray.Width(), templateGray.Height());

    ShapeModel model;
    CreateShapeModel(templateGray, model, 0, 0, RAD(360), 0,
                     "auto", "ignore_local_polarity", "33 44 9", 6);

    if (!model.IsValid()) {
        std::fprintf(stderr, "Model creation FAILED\n");
        return 1;
    }

    int32_t numLevels;
    double angleStart, angleExtent, angleStep, scaleMin, scaleMax, scaleStep;
    std::string metric;
    GetShapeModelParams(model, numLevels, angleStart, angleExtent, angleStep,
                        scaleMin, scaleMax, scaleStep, metric);
    std::printf("Model created: numLevels=%d, angleStep=%.2f deg, metric=%s\n",
                numLevels, angleStep * 180.0 / PI, metric.c_str());

    // =========================================================================
    // 2. Visualize model features
    // =========================================================================
    std::vector<ModelPoint> features = GetModelTransform(model, 1, 0.0, 1.0);
    std::printf("Level 1 features: %zu points\n\n", features.size());

    double cx = templateGray.Width() * 0.5;
    double cy = templateGray.Height() * 0.5;

    QImage templateVis;
    Color::GrayToRgb(templateGray, templateVis);

    for (const auto& f : features) {
        int32_t px = static_cast<int32_t>(std::round(f.x + cx));
        int32_t py = static_cast<int32_t>(std::round(f.y + cy));
        DrawPixelCross(templateVis, px, py, 0, 255, 0);
    }
    Draw::Cross(templateVis, Point2d{cx, cy}, 5, 0, Scalar::Yellow(), 1);

    // Compute model bounding box
    double fMinX = 1e9, fMaxX = -1e9, fMinY = 1e9, fMaxY = -1e9;
    for (const auto& f : features) {
        if (f.x < fMinX) fMinX = f.x;
        if (f.x > fMaxX) fMaxX = f.x;
        if (f.y < fMinY) fMinY = f.y;
        if (f.y > fMaxY) fMaxY = f.y;
    }
    double modelW = fMaxX - fMinX;
    double modelH = fMaxY - fMinY;

    WriteImage(templateVis, resultsFolder + "/template.png");
    DispImage(templateVis, "Template Features");

    // =========================================================================
    // 3. Load search image and run FindScaledShapeModel
    // =========================================================================
    QImage searchGray;
    ReadImageGray(searchPath, searchGray);
    if (searchGray.Empty()) {
        std::fprintf(stderr, "Failed to load search image: %s\n", searchPath);
        return 1;
    }
    std::printf("Search: %s (%d x %d)\n", searchPath,
                searchGray.Width(), searchGray.Height());

    std::vector<double> rows, cols, angles, scales, scores;
    auto t0 = std::chrono::high_resolution_clock::now();
    FindScaledShapeModel(searchGray, model,
                         0, RAD(360),       // angleStart, angleExtent
                         0.5, 1.5,          // scaleMin, scaleMax
                         0.8,               // minScore
                         0,                 // numMatches (0 = all)
                         0.8,               // maxOverlap
                         "least_squares",   // subPixel
                         0,                 // numLevels (0 = auto)
                         0.75,              // greediness
                         rows, cols, angles, scales, scores);
    auto t1 = std::chrono::high_resolution_clock::now();
    double matchMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("Found %zu matches, %.1f ms\n", rows.size(), matchMs);
    for (size_t i = 0; i < rows.size(); ++i) {
        std::printf("  [%zu] row=%.2f col=%.2f angle=%.1f scale=%.3f score=%.4f\n",
                    i, rows[i], cols[i], angles[i] * 180.0 / PI, scales[i], scores[i]);
    }

    // =========================================================================
    // 4. Visualize search results
    // =========================================================================
    QImage searchVis;
    Color::GrayToRgb(searchGray, searchVis);

    // Draw match time
    char timeLabel[64];
    std::snprintf(timeLabel, sizeof(timeLabel), "%.1f ms", matchMs);
    Draw::Text(searchVis, 10, 30, timeLabel, Scalar::Green(), 2);

    if (rows.empty()) {
        Draw::Text(searchVis, 10, 60, "No matches found", Scalar::Red(), 2);
    }

    for (size_t i = 0; i < rows.size(); ++i) {
        double matchX = cols[i];
        double matchY = rows[i];
        double matchAngle = angles[i];
        double matchScale = scales[i];

        // Cross at match center
        Draw::Cross(searchVis, Point2d{matchX, matchY}, 8, matchAngle,
                    Scalar::Yellow(), 1);

        // Draw rotated+scaled model feature points
        std::vector<ModelPoint> rotFeatures = GetModelTransform(model, 1, matchAngle, matchScale);
        for (const auto& f : rotFeatures) {
            int32_t px = static_cast<int32_t>(std::round(f.x + matchX));
            int32_t py = static_cast<int32_t>(std::round(f.y + matchY));
            DrawPixelCross(searchVis, px, py, 0, 255, 0);
        }

        // Score + angle + scale label, text fits scaled model box width
        char label[64];
        std::snprintf(label, sizeof(label), "score:%.3f angle:%.1f", scores[i], angles[i] * 180.0 / PI);
        double scaledW = modelW * matchScale;
        double modelSize = std::max(scaledW, modelH * matchScale);
        int32_t numChars = static_cast<int32_t>(std::strlen(label)) + 1;
        int32_t textScale = std::max(1, static_cast<int32_t>(scaledW / (numChars * 6.0)));

        double offset = modelSize * 0.5 + textScale * 10;
        double cosA = std::cos(matchAngle), sinA = std::sin(matchAngle);
        double tx = matchX + offset * sinA;
        double ty = matchY - offset * cosA;
        Draw::Text(searchVis, static_cast<int32_t>(tx), static_cast<int32_t>(ty),
                   label, Scalar::Cyan(), textScale);
        // Degree circle
        auto [tw, th] = Draw::TextSize(label, textScale);
        int32_t degR = std::max(1, textScale);
        Draw::Circle(searchVis, static_cast<int32_t>(tx) + tw + degR + 1,
                     static_cast<int32_t>(ty) + degR + 1,
                     degR, Scalar::Cyan(), 1);

        // Scale label on second line
        char scaleLabel[32];
        std::snprintf(scaleLabel, sizeof(scaleLabel), "scale:%.3f", matchScale);
        Draw::Text(searchVis, static_cast<int32_t>(tx),
                   static_cast<int32_t>(ty) + textScale * 7 + 3,
                   scaleLabel, Scalar::Cyan(), textScale);
    }

    WriteImage(searchVis, resultsFolder + "/result_search.png");
    DispImage(searchVis, "Scaled Match Results");

    std::printf("\nResults saved to: %s\n", resultsFolder.c_str());
    std::printf("Press any key to exit...\n");
    WaitKey();

    return 0;
}
