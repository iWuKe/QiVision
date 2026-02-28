/**
 * @file shape_match_interactive.cpp
 * @brief Interactive ShapeModel test: draw ROI on first image, match all images in folder
 *
 * Usage:
 *   ./shape_match_interactive [image_folder] [contrast]
 *
 * Default:
 *   image_folder = tests/data/matching/image1
 *   contrast     = auto
 *
 * Workflow:
 *   1. Load first image, display in window
 *   2. User draws rectangle ROI (click+drag, ESC to cancel)
 *   3. Create ShapeModel from ROI region
 *   4. Show template with feature overlay
 *   5. Match each image in folder, display results one by one (press any key for next)
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
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::GUI;
namespace fs = std::filesystem;

static std::vector<std::string> ListImageFiles(const std::string& folder) {
    std::vector<std::string> files;
    if (!fs::exists(folder) || !fs::is_directory(folder)) {
        return files;
    }
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

static void DrawMatchResults(QImage& vis, const ShapeModel& model,
                              const std::vector<double>& rows,
                              const std::vector<double>& cols,
                              const std::vector<double>& angles,
                              const std::vector<double>& scores) {
    // Get model bounding box from level 1 features
    std::vector<ModelPoint> features = GetModelTransform(model, 1, 0.0, 1.0);
    double fMinX = 1e9, fMaxX = -1e9, fMinY = 1e9, fMaxY = -1e9;
    for (const auto& f : features) {
        fMinX = std::min(fMinX, f.x);
        fMaxX = std::max(fMaxX, f.x);
        fMinY = std::min(fMinY, f.y);
        fMaxY = std::max(fMaxY, f.y);
    }
    double modelW = fMaxX - fMinX;
    double modelH = fMaxY - fMinY;

    for (size_t i = 0; i < rows.size(); ++i) {
        double mx = cols[i], my = rows[i], ma = angles[i];
        double cosA = std::cos(ma), sinA = std::sin(ma);

        // Rotated bounding box
        double boxCx = mx + ((fMinX + fMaxX) * 0.5) * cosA - ((fMinY + fMaxY) * 0.5) * sinA;
        double boxCy = my + ((fMinX + fMaxX) * 0.5) * sinA + ((fMinY + fMaxY) * 0.5) * cosA;
        Draw::RotatedRectangle(vis, Point2d{boxCx, boxCy},
                               modelW + 4, modelH + 4, ma, Scalar::Green(), 2);

        // Cross at match center
        Draw::Cross(vis, Point2d{mx, my}, 10, ma, Scalar::Yellow(), 1);

        // Draw rotated model features
        std::vector<ModelPoint> rotF = GetModelTransform(model, 1, ma, 1.0);
        for (const auto& f : rotF) {
            int32_t px = static_cast<int32_t>(std::round(f.x + mx));
            int32_t py = static_cast<int32_t>(std::round(f.y + my));
            if (px >= 0 && py >= 0 && px < vis.Width() && py < vis.Height()) {
                uint8_t* row = static_cast<uint8_t*>(vis.Data()) + py * vis.Stride();
                row[px * 3 + 0] = 0;
                row[px * 3 + 1] = 255;
                row[px * 3 + 2] = 0;
            }
        }

        // Score label
        char label[64];
        std::snprintf(label, sizeof(label), "#%zu s=%.3f a=%.1f",
                      i, scores[i], angles[i] * 180.0 / PI);
        Draw::Text(vis, static_cast<int32_t>(mx - modelW * 0.5),
                   static_cast<int32_t>(my - modelH * 0.5 - 15),
                   label, Scalar::Cyan(), 1);
    }
}

int main(int argc, char* argv[]) {
    // =========================================================================
    // Parse command line
    // =========================================================================
    std::string imageFolder = "tests/data/matching/image1";
    std::string contrast = "auto";

    if (argc >= 2) imageFolder = argv[1];
    if (argc >= 3) contrast = argv[2];

    std::printf("Image folder: %s\n", imageFolder.c_str());
    std::printf("Contrast: %s\n", contrast.c_str());

    // =========================================================================
    // 1. List all images
    // =========================================================================
    auto imageFiles = ListImageFiles(imageFolder);
    if (imageFiles.empty()) {
        std::fprintf(stderr, "No images found in: %s\n", imageFolder.c_str());
        return 1;
    }
    std::printf("Found %zu images\n", imageFiles.size());

    // =========================================================================
    // 2. Load first image and let user draw ROI
    // =========================================================================
    QImage firstGray;
    ReadImageGray(imageFiles[0], firstGray);
    if (firstGray.Empty()) {
        std::fprintf(stderr, "Failed to load: %s\n", imageFiles[0].c_str());
        return 1;
    }
    std::printf("First image: %s (%dx%d)\n", imageFiles[0].c_str(),
                firstGray.Width(), firstGray.Height());

    QImage firstRgb;
    Color::GrayToRgb(firstGray, firstRgb);

    Window win("Draw ROI - click and drag, ESC to cancel");
    win.SetAutoResize(true);
    win.DispImage(firstRgb, ScaleMode::None);

    std::printf("\nDraw a rectangle ROI on the image...\n");
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
    // 3. Create ShapeModel with ROI region
    // =========================================================================
    QRegion region = QRegion::Rectangle(roiX, roiY, roiW, roiH);

    ShapeModel model;
    CreateShapeModel(firstGray, region, model,
                     0,              // numLevels = auto
                     0, RAD(360), 0, // full rotation, auto step
                     "auto",         // optimization
                     "use_polarity", // metric
                     contrast,       // contrast
                     10);            // minContrast

    if (!model.IsValid()) {
        std::fprintf(stderr, "Model creation FAILED!\n");
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
    // 4. Show template with features
    // =========================================================================
    std::vector<ModelPoint> features = GetModelTransform(model, 1, 0.0, 1.0);
    std::printf("Level 1 features: %zu points\n", features.size());

    // Draw ROI + features on first image
    QImage templateVis;
    Color::GrayToRgb(firstGray, templateVis);
    Draw::Rectangle(templateVis, roiX, roiY, roiW, roiH, Scalar::Blue(), 2);

    double cx = roiX + roiW * 0.5;
    double cy = roiY + roiH * 0.5;
    for (const auto& f : features) {
        int32_t px = static_cast<int32_t>(std::round(f.x + cx));
        int32_t py = static_cast<int32_t>(std::round(f.y + cy));
        if (px >= 0 && py >= 0 && px < templateVis.Width() && py < templateVis.Height()) {
            uint8_t* row = static_cast<uint8_t*>(templateVis.Data()) + py * templateVis.Stride();
            row[px * 3 + 0] = 0;
            row[px * 3 + 1] = 255;
            row[px * 3 + 2] = 0;
        }
    }
    Draw::Cross(templateVis, Point2d{cx, cy}, 8, 0, Scalar::Yellow(), 1);

    win.DispImage(templateVis, ScaleMode::None);
    std::printf("Template shown. Press any key to start matching...\n");
    win.WaitKey();

    // =========================================================================
    // 5. Match each image in folder
    // =========================================================================
    for (size_t idx = 0; idx < imageFiles.size(); ++idx) {
        QImage searchGray;
        ReadImageGray(imageFiles[idx], searchGray);
        if (searchGray.Empty()) {
            std::printf("[%zu/%zu] SKIP (load failed): %s\n",
                        idx + 1, imageFiles.size(), imageFiles[idx].c_str());
            continue;
        }

        std::vector<double> rows, cols, angles, scores;
        FindShapeModel(searchGray, model,
                       0, RAD(360),    // full rotation
                       0.5,            // minScore
                       0,              // numMatches = all
                       0.5,            // maxOverlap
                       "least_squares", // subPixel
                       0,              // numLevels = all
                       0.7,            // greediness
                       rows, cols, angles, scores);

        std::printf("[%zu/%zu] %s: %zu matches",
                    idx + 1, imageFiles.size(),
                    fs::path(imageFiles[idx]).filename().c_str(),
                    rows.size());

        for (size_t i = 0; i < rows.size(); ++i) {
            std::printf("  (%.1f,%.1f s=%.3f)", cols[i], rows[i], scores[i]);
        }
        std::printf("\n");

        // Visualize
        QImage vis;
        Color::GrayToRgb(searchGray, vis);

        if (rows.empty()) {
            Draw::Text(vis, 10, 20, "No matches", Scalar::Red(), 2);
        } else {
            DrawMatchResults(vis, model, rows, cols, angles, scores);
        }

        // Title with file info
        char title[256];
        std::snprintf(title, sizeof(title), "[%zu/%zu] %s - %zu matches",
                      idx + 1, imageFiles.size(),
                      fs::path(imageFiles[idx]).filename().c_str(),
                      rows.size());
        win.SetTitle(title);
        win.DispImage(vis, ScaleMode::None);

        int32_t key = win.WaitKey();
        if (key == 27) {  // ESC to quit early
            std::printf("ESC pressed, exiting.\n");
            break;
        }
    }

    std::printf("\nDone.\n");
    return 0;
}
