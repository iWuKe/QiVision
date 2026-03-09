/**
 * @file shape_scaled_match_rings.cpp
 * @brief Interactive scaled shape matching on mixed hardware images
 *
 * Workflow:
 *   1. Load mixed_01.png, user draws ROI interactively
 *   2. Create ShapeModel from ROI
 *   3. Show template with features
 *   4. FindScaledShapeModel on each mixed_*.png (scale 0.5~1.5)
 *   5. Display and save results
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
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
#include <algorithm>
#include <filesystem>
#include <chrono>

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

int main() {
    std::printf("========================================\n");
    std::printf("  Scaled Shape Matching — Rings Mixed\n");
    std::printf("========================================\n\n");

    const std::string imageFolder = "tests/data/halcon_images/rings";
    const std::string resultsFolder = imageFolder + "/results";
    fs::create_directories(resultsFolder);

    // =========================================================================
    // 1. Load mixed_01.png as template source
    // =========================================================================
    const std::string templatePath = imageFolder + "/mixed_01.png";
    QImage templateGray;
    ReadImageGray(templatePath, templateGray);
    if (templateGray.Empty()) {
        std::fprintf(stderr, "Failed to load: %s\n", templatePath.c_str());
        return 1;
    }
    std::printf("Template source: %s (%dx%d)\n", templatePath.c_str(),
                templateGray.Width(), templateGray.Height());

    // =========================================================================
    // 2. Interactive ROI drawing
    // =========================================================================
    QImage templateRgb;
    Color::GrayToRgb(templateGray, templateRgb);

    Window win("Draw ROI on template - click and drag");
    win.SetAutoResize(true);
    win.DispImage(templateRgb, ScaleMode::None);

    std::printf("Draw a rectangle ROI around the target object...\n");
    ROIResult roi = win.DrawRectangle();

    if (!roi.valid) {
        std::fprintf(stderr, "ROI cancelled. Exiting.\n");
        return 1;
    }

    int32_t roiX = static_cast<int32_t>(std::min(roi.col1, roi.col2));
    int32_t roiY = static_cast<int32_t>(std::min(roi.row1, roi.row2));
    int32_t roiW = static_cast<int32_t>(std::abs(roi.col2 - roi.col1));
    int32_t roiH = static_cast<int32_t>(std::abs(roi.row2 - roi.row1));

    std::printf("ROI: x=%d, y=%d, w=%d, h=%d\n", roiX, roiY, roiW, roiH);

    if (roiW < 10 || roiH < 10) {
        std::fprintf(stderr, "ROI too small (min 10x10). Exiting.\n");
        return 1;
    }

    // =========================================================================
    // 3. Create ShapeModel
    // =========================================================================
    QRegion region = QRegion::Rectangle(roiX, roiY, roiW, roiH);

    ShapeModel model;
    CreateShapeModel(templateGray, region, model,
                     0,              // numLevels = auto
                     0, RAD(360), 0, // full rotation, auto step
                     "auto",         // optimization
                     "use_polarity",
                     "80",           // contrast
                     20);            // minContrast

    if (!model.IsValid()) {
        std::fprintf(stderr, "Model creation FAILED!\n");
        return 1;
    }

    int32_t numLevels;
    double angleStart, angleExtent, angleStep, scaleMin, scaleMax, scaleStep;
    std::string metric;
    GetShapeModelParams(model, numLevels, angleStart, angleExtent, angleStep,
                        scaleMin, scaleMax, scaleStep, metric);
    std::printf("Model: numLevels=%d, angleStep=%.2f deg, metric=%s\n",
                numLevels, angleStep * 180.0 / PI, metric.c_str());

    // =========================================================================
    // 4. Show template with features
    // =========================================================================
    std::vector<ModelPoint> features = GetModelTransform(model, 1, 0.0, 1.0);
    std::printf("Level 1 features: %zu points\n\n", features.size());

    QImage templateVis;
    Color::GrayToRgb(templateGray, templateVis);
    Draw::Rectangle(templateVis, roiX, roiY, roiW, roiH, Scalar::Blue(), 2);

    double cx = roiX + roiW * 0.5;
    double cy = roiY + roiH * 0.5;
    for (const auto& f : features) {
        int32_t px = static_cast<int32_t>(std::round(f.x + cx));
        int32_t py = static_cast<int32_t>(std::round(f.y + cy));
        DrawPixelCross(templateVis, px, py, 0, 255, 0);
    }
    Draw::Cross(templateVis, Point2d{cx, cy}, 8, 0, Scalar::Yellow(), 1);

    WriteImage(templateVis, resultsFolder + "/template_roi.png");
    win.SetTitle("Template with features");
    win.DispImage(templateVis, ScaleMode::None);
    std::printf("Template saved. Press any key to start matching...\n");
    win.WaitKey();

    // Model bounding box for label sizing
    double fMinX = 1e9, fMaxX = -1e9, fMinY = 1e9, fMaxY = -1e9;
    for (const auto& f : features) {
        fMinX = std::min(fMinX, f.x); fMaxX = std::max(fMaxX, f.x);
        fMinY = std::min(fMinY, f.y); fMaxY = std::max(fMaxY, f.y);
    }
    double modelW = fMaxX - fMinX;
    double modelH = fMaxY - fMinY;

    // =========================================================================
    // 5. Collect mixed_*.png files
    // =========================================================================
    std::vector<std::string> searchFiles;
    for (const auto& entry : fs::directory_iterator(imageFolder)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (fname.rfind("mixed_", 0) == 0) {
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".bmp" || ext == ".jpg") {
                searchFiles.push_back(entry.path().string());
            }
        }
    }
    std::sort(searchFiles.begin(), searchFiles.end());
    std::printf("Found %zu mixed images to search\n\n", searchFiles.size());

    // =========================================================================
    // 6. Match each image with FindScaledShapeModel
    // =========================================================================
    for (size_t idx = 0; idx < searchFiles.size(); ++idx) {
        QImage searchGray;
        ReadImageGray(searchFiles[idx], searchGray);
        if (searchGray.Empty()) {
            std::printf("[%zu/%zu] SKIP: %s\n", idx + 1, searchFiles.size(),
                        searchFiles[idx].c_str());
            continue;
        }

        std::string fname = fs::path(searchFiles[idx]).filename().string();

        std::vector<double> rows, cols, angles, scales, scores;
        auto t0 = std::chrono::high_resolution_clock::now();
        FindScaledShapeModel(searchGray, model,
                             0, RAD(360),
                             0.5, 1.5,
                             0.8,
                             0, 0.0,
                             "least_squares",
                             0, 0.7,
                             rows, cols, angles, scales, scores);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::printf("--- %s (%dx%d) --- %zu matches, %.1f ms\n",
                    fname.c_str(), searchGray.Width(), searchGray.Height(),
                    rows.size(), ms);
        for (size_t i = 0; i < rows.size(); ++i) {
            std::printf("  [%zu] row=%.1f col=%.1f angle=%.1f deg scale=%.3f score=%.4f\n",
                        i, rows[i], cols[i], angles[i] * 180.0 / PI, scales[i], scores[i]);
        }

        // Visualize
        QImage vis;
        Color::GrayToRgb(searchGray, vis);

        char timeLabel[64];
        std::snprintf(timeLabel, sizeof(timeLabel), "%s  %.1f ms  %zu matches",
                      fname.c_str(), ms, rows.size());
        Draw::Text(vis, 10, 25, timeLabel, Scalar::Green(), 1);

        for (size_t i = 0; i < rows.size(); ++i) {
            double mx = cols[i], my = rows[i], ma = angles[i], ms_ = scales[i];

            Draw::Cross(vis, Point2d{mx, my}, 8, ma, Scalar::Yellow(), 1);

            // Draw scaled+rotated features
            std::vector<ModelPoint> rotF = GetModelTransform(model, 1, ma, ms_);
            for (const auto& f : rotF) {
                int32_t px = static_cast<int32_t>(std::round(f.x + mx));
                int32_t py = static_cast<int32_t>(std::round(f.y + my));
                DrawPixelCross(vis, px, py, 0, 255, 0);
            }

            // Label
            char label[80];
            std::snprintf(label, sizeof(label), "s:%.3f a:%.1f sc:%.2f",
                          scores[i], angles[i] * 180.0 / PI, ms_);
            double scaledSize = std::max(modelW, modelH) * ms_;
            int32_t numChars = static_cast<int32_t>(std::strlen(label)) + 1;
            int32_t textScale = std::max(1, static_cast<int32_t>(scaledSize / (numChars * 6.0)));
            double offset = scaledSize * 0.5 + textScale * 10;
            double cosA = std::cos(ma), sinA = std::sin(ma);
            double tx = mx + offset * sinA;
            double ty = my - offset * cosA;
            Draw::Text(vis, static_cast<int32_t>(tx), static_cast<int32_t>(ty),
                       label, Scalar::Cyan(), textScale);
        }

        std::string outName = resultsFolder + "/result_" +
                              fs::path(searchFiles[idx]).stem().string() + ".png";
        WriteImage(vis, outName);

        char title[256];
        std::snprintf(title, sizeof(title), "[%zu/%zu] %s - %zu matches",
                      idx + 1, searchFiles.size(), fname.c_str(), rows.size());
        win.SetTitle(title);
        win.DispImage(vis, ScaleMode::None);

        int32_t key = win.WaitKey();
        if (key == 27) {
            std::printf("ESC pressed, exiting.\n");
            break;
        }
    }

    std::printf("\nResults saved to: %s\n", resultsFolder.c_str());
    return 0;
}
