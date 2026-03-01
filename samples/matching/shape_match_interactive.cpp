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
 *   6. All result images are saved to <image_folder>/results/
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

    double modelSize = std::max(modelW, modelH);

    for (size_t i = 0; i < rows.size(); ++i) {
        double mx = cols[i], my = rows[i], ma = angles[i];

        // Cross at match center
        Draw::Cross(vis, Point2d{mx, my}, 10, ma, Scalar::Yellow(), 1);

        // Draw rotated model features as 3x3 crosses
        std::vector<ModelPoint> rotF = GetModelTransform(model, 1, ma, 1.0);
        for (const auto& f : rotF) {
            int32_t px = static_cast<int32_t>(std::round(f.x + mx));
            int32_t py = static_cast<int32_t>(std::round(f.y + my));
            DrawPixelCross(vis, px, py, 0, 255, 0);
        }

        // Score + angle label, scale to fit model box width
        char label[64];
        std::snprintf(label, sizeof(label), "score:%.3f angle:%.1f", scores[i], angles[i] * 180.0 / PI);
        int32_t numChars = static_cast<int32_t>(std::strlen(label)) + 1; // +1 for degree circle
        int32_t textScale = std::max(1, static_cast<int32_t>(modelW / (numChars * 6.0)));
        double offset = modelSize * 0.5 + textScale * 10;
        double cosA = std::cos(ma), sinA = std::sin(ma);
        // "Above" in rotated frame: rotate (0, -1) by angle
        double tx = mx + offset * sinA;
        double ty = my - offset * cosA;
        Draw::Text(vis, static_cast<int32_t>(tx), static_cast<int32_t>(ty),
                   label, Scalar::Cyan(), textScale);
        // Draw degree symbol as small circle
        auto [tw, th] = Draw::TextSize(label, textScale);
        int32_t degR = std::max(1, textScale);
        Draw::Circle(vis, static_cast<int32_t>(tx) + tw + degR + 1,
                     static_cast<int32_t>(ty) + degR + 1,
                     degR, Scalar::Cyan(), 1);
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
        DrawPixelCross(templateVis, px, py, 0, 255, 0);
    }
    Draw::Cross(templateVis, Point2d{cx, cy}, 8, 0, Scalar::Yellow(), 1);

    // =========================================================================
    // 4b. Create results output folder
    // =========================================================================
    std::string resultsFolder = imageFolder + "/results";
    fs::create_directories(resultsFolder);
    std::printf("Results will be saved to: %s\n", resultsFolder.c_str());

    // Save template visualization
    WriteImage(templateVis, resultsFolder + "/template.png");

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
        auto t0 = std::chrono::high_resolution_clock::now();
        FindShapeModel(searchGray, model,
                       0, RAD(360),    // full rotation
                       0.5,            // minScore
                       0,              // numMatches = all
                       0,              // maxOverlap
                       "least_squares", // subPixel
                       0,              // numLevels = all
                       0.7,            // greediness
                       rows, cols, angles, scores);
        auto t1 = std::chrono::high_resolution_clock::now();
        double matchMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::printf("[%zu/%zu] %s: %zu matches, %.1f ms",
                    idx + 1, imageFiles.size(),
                    fs::path(imageFiles[idx]).filename().c_str(),
                    rows.size(), matchMs);

        for (size_t i = 0; i < rows.size(); ++i) {
            std::printf("  (%.1f,%.1f s=%.3f)", cols[i], rows[i], scores[i]);
        }
        std::printf("\n");

        // Visualize
        QImage vis;
        Color::GrayToRgb(searchGray, vis);

        // Draw match time on image
        char timeLabel[64];
        std::snprintf(timeLabel, sizeof(timeLabel), "%.1f ms", matchMs);
        Draw::Text(vis, 10, 30, timeLabel, Scalar::Green(), 2);

        if (rows.empty()) {
            Draw::Text(vis, 10, 60, "No matches", Scalar::Red(), 2);
        } else {
            DrawMatchResults(vis, model, rows, cols, angles, scores);
        }

        // Save result image
        std::string outName = resultsFolder + "/result_" +
                              fs::path(imageFiles[idx]).stem().string() + ".png";
        WriteImage(vis, outName);

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
