/**
 * @file shape_brightness.cpp
 * @brief ShapeModel matching under varying brightness conditions
 *
 * Template: tests/data/matching/image5/m15.bmp
 * Search images: tests/data/matching/image5/rings_*.png
 * Uses "ignore_global_polarity" metric to handle brightness/polarity changes.
 *
 * Results are saved to tests/data/matching/image5/results/
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Matching/ShapeModel.h>
#include <QiVision/Matching/MatchTypes.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/GUI/Window.h>

#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::GUI;
namespace fs = std::filesystem;

// Set a single pixel in an RGB image
static void SetPixelRgb(QImage& img, int32_t x, int32_t y,
                        uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || y < 0 || x >= img.Width() || y >= img.Height()) return;
    uint8_t* row = static_cast<uint8_t*>(img.Data()) + y * img.Stride();
    row[x * 3 + 0] = r;
    row[x * 3 + 1] = g;
    row[x * 3 + 2] = b;
}

// Draw model feature points as green 3x3 crosses for visibility
static void DrawFeaturePoints(QImage& img, const std::vector<ModelPoint>& pts,
                              double cx, double cy)
{
    for (const auto& pt : pts) {
        int32_t px = static_cast<int32_t>(std::round(pt.x + cx));
        int32_t py = static_cast<int32_t>(std::round(pt.y + cy));
        SetPixelRgb(img, px, py, 0, 255, 0);
        SetPixelRgb(img, px - 1, py, 0, 255, 0);
        SetPixelRgb(img, px + 1, py, 0, 255, 0);
        SetPixelRgb(img, px, py - 1, 0, 255, 0);
        SetPixelRgb(img, px, py + 1, 0, 255, 0);
    }
}

int main()
{
    std::printf("========================================\n");
    std::printf("  ShapeModel Brightness Variation Test\n");
    std::printf("========================================\n\n");

    const std::string dataFolder = "tests/data/matching/image5";
    const std::string resultsFolder = dataFolder + "/results";
    fs::create_directories(resultsFolder);
    std::printf("Results will be saved to: %s\n\n", resultsFolder.c_str());

    // =========================================================================
    // 1. Load template and create model
    // =========================================================================
    const std::string templatePath = dataFolder + "/m15.bmp";

    QImage templateGray;
    ReadImageGray(templatePath, templateGray);
    if (templateGray.Empty()) {
        std::fprintf(stderr, "ERROR: Failed to load template: %s\n", templatePath.c_str());
        return 1;
    }
    std::printf("[Template] %s (%d x %d)\n", templatePath.c_str(),
                templateGray.Width(), templateGray.Height());

    ShapeModel model;
    CreateShapeModel(templateGray, model,
                     4,                          // numLevels
                     0, RAD(360),                // angleStart, angleExtent
                     0,                          // angleStep (auto)
                     "auto",                     // optimization
                     "ignore_global_polarity",   // metric
                     "auto",                     // contrast
                     10);                        // minContrast

    if (!model.IsValid()) {
        std::fprintf(stderr, "ERROR: Model creation failed.\n");
        return 1;
    }

    int32_t numLevels;
    double aStart, aExtent, aStep, sMin, sMax, sStep;
    std::string metric;
    GetShapeModelParams(model, numLevels, aStart, aExtent, aStep,
                        sMin, sMax, sStep, metric);
    std::printf("[Model] levels=%d, angleStep=%.3f deg, metric=%s\n\n",
                numLevels, DEG(aStep), metric.c_str());

    // =========================================================================
    // 2. Display template with feature points and save
    // =========================================================================
    QImage templateVis;
    Color::GrayToRgb(templateGray, templateVis);

    double tcx = templateGray.Width() * 0.5;
    double tcy = templateGray.Height() * 0.5;

    std::vector<ModelPoint> features = GetModelTransform(model, 1, 0.0, 1.0);
    std::printf("[Model] %zu feature points at level 1\n\n", features.size());
    DrawFeaturePoints(templateVis, features, tcx, tcy);
    Draw::Cross(templateVis, Point2d{tcx, tcy}, 5, 0, Scalar::Yellow(), 1);

    // Compute model bounding box from features
    double fMinX = 1e9, fMaxX = -1e9, fMinY = 1e9, fMaxY = -1e9;
    for (const auto& f : features) {
        if (f.x < fMinX) fMinX = f.x;
        if (f.x > fMaxX) fMaxX = f.x;
        if (f.y < fMinY) fMinY = f.y;
        if (f.y > fMaxY) fMaxY = f.y;
    }
    double modelW = fMaxX - fMinX;
    double modelH = fMaxY - fMinY;

    double modelSize = std::max(modelW, modelH);

    // Save template result
    WriteImage(templateVis, resultsFolder + "/template.png");
    std::printf("[Saved] %s/template.png\n", resultsFolder.c_str());

    Window templateWin("Template (m15.bmp)", templateGray.Width(), templateGray.Height());
    templateWin.DispImage(templateVis, ScaleMode::None);

    // =========================================================================
    // 3. Search each image, display and save results
    // =========================================================================
    const std::vector<std::string> searchFiles = {
        dataFolder + "/rings_01.png",
        dataFolder + "/rings_02.png",
        dataFolder + "/rings_04.png",
        dataFolder + "/rings_05.png",
        dataFolder + "/rings_06.png",
        dataFolder + "/rings_07.png",
        dataFolder + "/rings_08.png"
    };

    const double minScore    = 0.5;
    const int32_t maxMatches = 4;
    const double maxOverlap  = 0.75;
    const double greediness  = 0.7;

    int totalImages = 0;
    int totalMatches = 0;

    Window searchWin("Search Result", 640, 480);

    for (const auto& searchPath : searchFiles) {
        totalImages++;

        QImage searchGray;
        ReadImageGray(searchPath, searchGray);
        if (searchGray.Empty()) {
            std::fprintf(stderr, "WARNING: Failed to load %s\n", searchPath.c_str());
            continue;
        }

        // Find matches
        std::vector<double> rows, cols, angles, scores;
        auto t0 = std::chrono::high_resolution_clock::now();
        FindShapeModel(searchGray, model,
                       0, RAD(360),
                       minScore, maxMatches, maxOverlap,
                       "least_squares", 0, greediness,
                       rows, cols, angles, scores);
        auto t1 = std::chrono::high_resolution_clock::now();
        double matchMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int matchCount = static_cast<int>(rows.size());
        totalMatches += matchCount;

        std::string fname = fs::path(searchPath).stem().string();
        std::printf("--- %s (%d x %d) --- %d matches, %.1f ms\n",
                    (fname + fs::path(searchPath).extension().string()).c_str(),
                    searchGray.Width(), searchGray.Height(), matchCount, matchMs);

        // Draw results
        QImage vis;
        Color::GrayToRgb(searchGray, vis);

        // Draw match time on image
        char timeLabel[64];
        std::snprintf(timeLabel, sizeof(timeLabel), "%.1f ms", matchMs);
        Draw::Text(vis, 10, 30, timeLabel, Scalar::Green(), 2);

        for (int i = 0; i < matchCount; i++) {
            double matchX = cols[i];
            double matchY = rows[i];
            double matchAngle = angles[i];

            std::printf("  [%d] row=%.2f col=%.2f angle=%.1f deg score=%.4f\n",
                        i + 1, rows[i], cols[i], DEG(matchAngle), scores[i]);

            // Get rotated feature points and draw as green crosses
            std::vector<ModelPoint> rotPts = GetModelTransform(model, 1, matchAngle, 1.0);
            DrawFeaturePoints(vis, rotPts, matchX, matchY);

            // Yellow cross at match center
            Draw::Cross(vis, Point2d{matchX, matchY}, 10, matchAngle,
                        Scalar::Yellow(), 1);

            // Score + angle label, scale to fit model box width
            char label[64];
            std::snprintf(label, sizeof(label), "score:%.3f angle:%.1f", scores[i], DEG(matchAngle));
            int32_t numChars = static_cast<int32_t>(std::strlen(label)) + 1; // +1 for degree circle
            int32_t textScale = std::max(1, static_cast<int32_t>(modelW / (numChars * 6.0)));
            double offset = modelSize * 0.5 + textScale * 10;
            double cosA = std::cos(matchAngle), sinA = std::sin(matchAngle);
            // "Above" in rotated frame: rotate (0, -1) by angle
            double tx = matchX + offset * sinA;
            double ty = matchY - offset * cosA;
            Draw::Text(vis, static_cast<int32_t>(tx), static_cast<int32_t>(ty),
                       label, Scalar::Cyan(), textScale);
            // Draw degree symbol as small circle
            auto [tw, th] = Draw::TextSize(label, textScale);
            int32_t degR = std::max(1, textScale);
            Draw::Circle(vis, static_cast<int32_t>(tx) + tw + degR + 1,
                         static_cast<int32_t>(ty) + degR + 1,
                         degR, Scalar::Cyan(), 1);
        }

        if (matchCount == 0) {
            Draw::Text(vis, 10, 60, "No matches found", Scalar::Red(), 2);
        }

        // Save result image
        std::string outPath = resultsFolder + "/result_" + fname + ".png";
        WriteImage(vis, outPath);
        std::printf("  [Saved] %s\n", outPath.c_str());

        // Show in window
        searchWin.SetTitle(fname + " - " + std::to_string(matchCount) + " matches");
        searchWin.Resize(searchGray.Width(), searchGray.Height());
        searchWin.DispImage(vis, ScaleMode::None);

        std::printf("  >> Press any key for next image...\n");
        searchWin.WaitKey();
    }

    // =========================================================================
    // 4. Summary
    // =========================================================================
    std::printf("\n========================================\n");
    std::printf("  Summary: %d images, %d total matches\n", totalImages, totalMatches);
    std::printf("  Results saved to: %s\n", resultsFolder.c_str());
    std::printf("========================================\n");

    return 0;
}
