/**
 * @file shape_multi_match.cpp
 * @brief FindShapeModels multi-model matching demo
 *
 * Uses 3 different templates (m_1.bmp, m_2.bmp, m_3.bmp) to simultaneously
 * search in s_123.png. Results are visualized with different colors per model.
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

static void SetPixelRgb(QImage& img, int32_t x, int32_t y, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || y < 0 || x >= img.Width() || y >= img.Height()) return;
    uint8_t* row = static_cast<uint8_t*>(img.Data()) + y * img.Stride();
    row[x * 3 + 0] = r;
    row[x * 3 + 1] = g;
    row[x * 3 + 2] = b;
}

static void DrawPixelCross(QImage& img, int32_t x, int32_t y, uint8_t r, uint8_t g, uint8_t b) {
    SetPixelRgb(img, x, y, r, g, b);
    SetPixelRgb(img, x - 1, y, r, g, b);
    SetPixelRgb(img, x + 1, y, r, g, b);
    SetPixelRgb(img, x, y - 1, r, g, b);
    SetPixelRgb(img, x, y + 1, r, g, b);
}

// Per-model colors (RGB)
static const uint8_t MODEL_COLORS[][3] = {
    {0, 255, 0},     // Model 0: Green
    {255, 100, 0},   // Model 1: Orange
    {0, 180, 255},   // Model 2: Cyan
};
static const Scalar MODEL_SCALARS[] = {
    Scalar(0, 255, 0),
    Scalar(255, 100, 0),
    Scalar(0, 180, 255),
};
static const char* MODEL_NAMES[] = {"Model_1", "Model_2", "Model_3"};

int main() {
    std::printf("========================================\n");
    std::printf("  FindShapeModels Multi-Model Demo\n");
    std::printf("========================================\n\n");

    const char* templatePaths[] = {
        "tests/data/imgs/m_1.bmp",
        "tests/data/imgs/m_2.bmp",
        "tests/data/imgs/m_3.bmp",
    };
    const char* searchPath = "tests/data/imgs/s_123.png";
    const int NUM_MODELS = 3;

    const std::string resultsFolder = "tests/data/imgs/results";
    fs::create_directories(resultsFolder);

    // =========================================================================
    // 1. Load templates and create models
    // =========================================================================
    std::vector<ShapeModel> models(NUM_MODELS);

    for (int i = 0; i < NUM_MODELS; ++i) {
        QImage tmplGray;
        ReadImageGray(templatePaths[i], tmplGray);
        if (tmplGray.Empty()) {
            std::fprintf(stderr, "Failed to load template: %s\n", templatePaths[i]);
            return 1;
        }
        std::printf("Template %d: %s (%d x %d)\n", i,
                    templatePaths[i], tmplGray.Width(), tmplGray.Height());

        CreateShapeModel(tmplGray, models[i], 0, 0, RAD(360), 0,
                         "auto", "use_polarity", "50", 10);

        if (!models[i].IsValid()) {
            std::fprintf(stderr, "Model %d creation FAILED\n", i);
            return 1;
        }

        int32_t nLvl;
        double aS, aE, aSt, sMin, sMax, sSt;
        std::string met;
        GetShapeModelParams(models[i], nLvl, aS, aE, aSt, sMin, sMax, sSt, met);
        std::printf("  -> numLevels=%d, angleStep=%.2f deg, metric=%s\n",
                    nLvl, aSt * 180.0 / PI, met.c_str());
    }
    std::printf("\n");

    // =========================================================================
    // 2. Load search image
    // =========================================================================
    QImage searchGray;
    ReadImageGray(searchPath, searchGray);
    if (searchGray.Empty()) {
        std::fprintf(stderr, "Failed to load search image: %s\n", searchPath);
        return 1;
    }
    std::printf("Search: %s (%d x %d)\n", searchPath,
                searchGray.Width(), searchGray.Height());

    // =========================================================================
    // 3. FindShapeModels (multi-model simultaneous search)
    // =========================================================================
    std::vector<double> rows, cols, angles, scores;
    std::vector<int32_t> modelIndices;
    std::vector<int32_t> numLevels(NUM_MODELS, 0);  // 0 = use model default

    auto t0 = std::chrono::high_resolution_clock::now();
    FindShapeModels(searchGray, models,
                    0, RAD(360),          // angleStart, angleExtent
                    0.7,                  // minScore
                    0,                    // numMatches (0 = all)
                    0.5,                  // maxOverlap
                    "least_squares",      // subPixel
                    numLevels,            // per-model levels
                    0.9,                  // greediness
                    rows, cols, angles, scores, modelIndices);
    auto t1 = std::chrono::high_resolution_clock::now();
    double matchMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("\nFound %zu matches in %.1f ms\n", rows.size(), matchMs);
    for (size_t i = 0; i < rows.size(); ++i) {
        std::printf("  [%zu] model=%d (%s) row=%.2f col=%.2f angle=%.1f score=%.4f\n",
                    i, modelIndices[i], MODEL_NAMES[modelIndices[i]],
                    rows[i], cols[i], angles[i] * 180.0 / PI, scores[i]);
    }

    // =========================================================================
    // 4. Visualize results with GUI
    // =========================================================================
    QImage searchVis;
    Color::GrayToRgb(searchGray, searchVis);

    // Time label
    char timeLabel[64];
    std::snprintf(timeLabel, sizeof(timeLabel), "FindShapeModels: %.1f ms, %zu matches",
                  matchMs, rows.size());
    Draw::Text(searchVis, 10, 30, timeLabel, Scalar::Green(), 2);

    if (rows.empty()) {
        Draw::Text(searchVis, 10, 60, "No matches found", Scalar::Red(), 2);
    }

    // Draw each match with its model's color
    for (size_t i = 0; i < rows.size(); ++i) {
        int mi = modelIndices[i];
        double matchX = cols[i];
        double matchY = rows[i];
        double matchAngle = angles[i];
        Scalar color = MODEL_SCALARS[mi];
        uint8_t cr = MODEL_COLORS[mi][0];
        uint8_t cg = MODEL_COLORS[mi][1];
        uint8_t cb = MODEL_COLORS[mi][2];

        // Draw model feature points at match position
        std::vector<ModelPoint> features = GetModelTransform(models[mi], 1, matchAngle, 1.0);
        for (const auto& f : features) {
            int32_t px = static_cast<int32_t>(std::round(f.x + matchX));
            int32_t py = static_cast<int32_t>(std::round(f.y + matchY));
            DrawPixelCross(searchVis, px, py, cr, cg, cb);
        }

        // Cross at center
        Draw::Cross(searchVis, Point2d{matchX, matchY}, 10, matchAngle,
                    Scalar::Yellow(), 2);

        // Label: model name + score
        char label[64];
        std::snprintf(label, sizeof(label), "%s %.3f %.1f",
                      MODEL_NAMES[mi], scores[i], angles[i] * 180.0 / PI);
        Draw::Text(searchVis, static_cast<int32_t>(matchX) - 30,
                   static_cast<int32_t>(matchY) - 20,
                   label, color, 1);
    }

    // Legend
    for (int i = 0; i < NUM_MODELS; ++i) {
        int32_t ly = 60 + i * 25;
        Draw::Text(searchVis, 10, ly, MODEL_NAMES[i], MODEL_SCALARS[i], 2);
    }

    WriteImage(searchVis, resultsFolder + "/result_multi_match.png");
    DispImage(searchVis, "FindShapeModels - Multi-Model Results");

    std::printf("\nResult saved to: %s/result_multi_match.png\n", resultsFolder.c_str());
    std::printf("Press 'q' to exit...\n");
    while (WaitKey(50) != 'q') {}

    return 0;
}
