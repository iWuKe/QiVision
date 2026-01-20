/**
 * @file circle_measure.cpp
 * @brief Circle measurement sample using Metrology module
 *
 * Demonstrates:
 * - GUI display with zoom/pan
 * - Creating a MetrologyModel with fixed circle parameters
 * - Configuring measurement parameters (measure_length, num_measures, etc.)
 * - Executing measurement and retrieving results
 * - Visualizing results using Draw module
 *
 * Usage:
 * 1. Run the program: ./measure_circle [image_dir]
 *    - [image_dir]: Image directory (default: tests/data/matching/image3)
 * 2. Left drag to pan, scroll wheel to zoom, right click to reset view
 * 3. Press any key for next image, 'q' to quit
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Draw.h>
#include <QiVision/Measure/Metrology.h>
#include <QiVision/GUI/Window.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Measure;
using namespace Qi::Vision::GUI;

namespace fs = std::filesystem;

// Draw edge points with colors based on weights
// Auto-detects binary (RANSAC/Tukey: 2 colors) vs continuous (Huber: 3 colors)
void DrawEdgePointsWithWeights(QImage& image, const std::vector<Point2d>& points,
                                const std::vector<double>& weights, int markerSize = 3)
{
    if (points.empty()) return;

    // Detect if weights are binary or continuous
    bool isBinary = true;
    for (double w : weights) {
        if (w > 0.01 && w < 0.99) {
            isBinary = false;
            break;
        }
    }

    for (size_t i = 0; i < points.size(); ++i) {
        double w = (i < weights.size()) ? weights[i] : 1.0;

        Color color;
        if (isBinary) {
            // RANSAC/Tukey: green (inlier) or red (outlier)
            color = (w >= 0.5) ? Color::Green() : Color::Red();
        } else {
            // Huber: green (strong), yellow (moderate), red (weak)
            if (w >= 0.8) color = Color::Green();
            else if (w >= 0.3) color = Color::Yellow();
            else color = Color::Red();
        }

        Draw::FilledCircle(image, points[i], markerSize, color);
    }
}

// Print measurement parameters
void PrintParams(const MetrologyMeasureParams& params) {
    std::cout << "\n=== Measurement Parameters ===\n";
    std::cout << "  measureLength1 (projection length): " << params.measureLength1 << " px\n";
    std::cout << "  measureLength2 (projection width):  " << params.measureLength2 << " px\n";
    std::cout << "  numMeasures (projection count):     " << params.numMeasures << "\n";
    std::cout << "  measureSigma:                       " << params.measureSigma << "\n";
    std::cout << "  measureThreshold:                   ";
    if (params.thresholdMode == ThresholdMode::Auto) {
        std::cout << "auto\n";
    } else {
        std::cout << params.measureThreshold << "\n";
    }
    std::cout << "  numInstances:                       " << params.numInstances << "\n";
    std::cout << "  minScore:                           " << params.minScore << "\n";
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================\n";
    std::cout << "  QiVision Circle Measurement Demo\n";
    std::cout << "==============================================\n\n";

    // Default image directory
    std::string imageDir = "tests/data/matching/image3";

    // Allow override from command line
    if (argc > 1) {
        imageDir = argv[1];
    }

    // Find images in directory
    std::vector<std::string> imagePaths;
    if (fs::exists(imageDir) && fs::is_directory(imageDir)) {
        for (const auto& entry : fs::directory_iterator(imageDir)) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".bmp" || ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                imagePaths.push_back(entry.path().string());
            }
        }
        std::sort(imagePaths.begin(), imagePaths.end());
    }

    if (imagePaths.empty()) {
        std::cerr << "No images found in: " << imageDir << "\n";
        return 1;
    }

    std::cout << "Found " << imagePaths.size() << " images in " << imageDir << "\n";

    // Load first image to get dimensions
    QImage firstImage = QImage::FromFile(imagePaths[0]);
    if (!firstImage.IsValid()) {
        std::cerr << "Failed to load: " << imagePaths[0] << "\n";
        return 1;
    }

    int width = firstImage.Width();
    int height = firstImage.Height();
    std::cout << "Image size: " << width << " x " << height << "\n";

    // =========================================================================
    // Configure measurement parameters
    // =========================================================================

    MetrologyMeasureParams params;
    params.measureLength1 = 30.0;
    params.measureLength2 = 10.0;
    params.numMeasures = 36;
    params.measureSigma = 1.5;
    params.measureThreshold = 10.0;
    params.measureTransition = EdgeTransition::All;
    params.numInstances = 1;
    params.minScore = 0.5;
    params.SetThreshold("auto");

    PrintParams(params);

    // =========================================================================
    // Fixed circle parameters (adjust based on your image)
    // =========================================================================

    double centerCol = 650.0;
    double centerRow = 500.0;
    double radius = 220.0;

    std::cout << "\n=== Fixed Circle Parameters ===\n";
    std::cout << "  Center: (" << centerCol << ", " << centerRow << ")\n";
    std::cout << "  Radius: " << radius << " px\n";

    // =========================================================================
    // Create GUI Window
    // =========================================================================

    Window win("Circle Measurement - QiVision", 0, 0);
    win.SetAutoResize(true, 1400, 900);
    win.EnableZoomPan(true);

    std::string currentTitle;
    win.SetMouseCallback([&](const MouseEvent& evt) {
        if (evt.type == MouseEventType::Move) {
            char coordStr[128];
            snprintf(coordStr, sizeof(coordStr), " | X:%.1f Y:%.1f", evt.imageX, evt.imageY);
            win.SetTitle(currentTitle + coordStr);
        }
    });

    std::cout << "\n=== Controls ===\n";
    std::cout << "  Left drag: Pan image\n";
    std::cout << "  Scroll wheel: Zoom in/out\n";
    std::cout << "  Right click: Reset view\n";
    std::cout << "  Any key: Next image\n";
    std::cout << "  'q': Quit\n\n";

    // =========================================================================
    // Process images
    // =========================================================================

    int successCount = 0;
    int totalCount = 0;
    int maxImages = static_cast<int>(imagePaths.size());

    std::cout << "=== Processing " << maxImages << " images ===\n";

    for (int i = 0; i < maxImages; ++i) {
        const auto& path = imagePaths[i];
        std::string filename = fs::path(path).filename().string();

        std::cout << "\n--- Image " << (i+1) << "/" << maxImages << ": " << filename << " ---\n";

        QImage image = QImage::FromFile(path);
        if (!image.IsValid()) {
            std::cerr << "Failed to load: " << filename << "\n";
            continue;
        }
        if (image.Channels() > 1) {
            image = image.ToGray();
        }

        totalCount++;

        // =========================================================================
        // Perform measurement
        // =========================================================================

        MetrologyModel model;
        int circleIdx = model.AddCircleMeasure(centerRow, centerCol, radius, params);

        std::string title;
        QImage display = Draw::PrepareForDrawing(image);

        if (model.Apply(image)) {
            auto edgePoints = model.GetMeasuredPoints(circleIdx);
            auto result = model.GetCircleResult(circleIdx);

            std::cout << "Detected " << edgePoints.size() << " edge points\n";

            if (result.IsValid()) {
                successCount++;
                std::cout << ">>> Measurement Result <<<\n";
                std::cout << "  Center: (" << result.column << ", " << result.row << ")\n";
                std::cout << "  Radius: " << result.radius << " px\n";
                std::cout << "  Points used: " << result.numUsed << "/" << params.numMeasures << "\n";
                std::cout << "  RMS Error: " << result.rmsError << " px\n";
                std::cout << "  Score: " << result.score << "\n";
                title = filename + " - OK (" + std::to_string(edgePoints.size()) + " pts)";
            } else {
                std::cout << "Measurement failed - no valid circle found\n";
                title = filename + " - FAILED (" + std::to_string(edgePoints.size()) + " pts)";
            }

            // =========================================================================
            // Visualization using Draw module
            // =========================================================================

            // 1. Draw calipers (cyan) - each caliper is an independent rectangle
            const MetrologyObject* obj = model.GetObject(circleIdx);
            if (obj) {
                auto calipers = obj->GetCalipers();
                std::cout << "  Drawing " << calipers.size() << " calipers\n";
                std::cout << "  Caliper params: Length1=" << params.measureLength1
                          << ", Length2=" << params.measureLength2 << "\n";
                Draw::MeasureRects(display, calipers, Color::Cyan(), 1);
            }

            // 2. Draw edge points with weight-based coloring (green=inlier, red=outlier)
            auto pointWeights = model.GetPointWeights(circleIdx);
            std::cout << "  Edge points: " << edgePoints.size()
                      << ", weights: " << pointWeights.size() << "\n";
            if (!pointWeights.empty()) {
                int inliers = 0, outliers = 0;
                for (double w : pointWeights) {
                    if (w >= 0.8) inliers++;
                    else if (w < 0.5) outliers++;
                }
                std::cout << "  Inliers: " << inliers << ", Outliers: " << outliers << "\n";
            }
            DrawEdgePointsWithWeights(display, edgePoints, pointWeights, 3);
        } else {
            std::cout << "Apply failed\n";
            title = filename + " - Apply failed";
        }

        currentTitle = title;
        win.SetTitle(title);
        win.DispImage(display);

        int key = win.WaitKey(0);
        if (key == 'q' || key == 'Q') {
            std::cout << "Quit requested.\n";
            break;
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================

    std::cout << "\n=== Summary ===\n";
    std::cout << "  Success: " << successCount << " / " << totalCount << "\n";

    std::cout << "\n=== Parameter Tuning Tips ===\n";
    std::cout << "  - If no edges found: decrease measureThreshold\n";
    std::cout << "  - If noisy: increase measureSigma or measureLength2\n";
    std::cout << "  - If missing parts: increase measureLength1 (search range)\n";
    std::cout << "  - For better accuracy: increase numMeasures\n";

    return 0;
}
