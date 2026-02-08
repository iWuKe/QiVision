/**
 * @file line_metrology.cpp
 * @brief Metrology demo - measure lines with automatic edge finding and fitting
 *
 * Demonstrates:
 * - AddLineMeasure for measuring straight edges
 * - Parameter tuning (threshold modes, transition types)
 * - Result visualization
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Measure/Metrology.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Platform/Timer.h>
#include <QiVision/GUI/Window.h>

#include <iostream>
#include <iomanip>
#include <cmath>

using namespace Qi::Vision;
using namespace Qi::Vision::Measure;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::Platform;
using namespace Qi::Vision::GUI;

int main(int argc, char* argv[]) {
    // Default: use IC chip image which has clear rectangular edges
    std::string imagePath = "tests/data/halcon_images/ic.png";
    if (argc > 1) {
        imagePath = argv[1];
    }

    QImage grayImage;
    ReadImageGray(imagePath, grayImage);
    if (grayImage.Empty()) {
        std::cerr << "Failed to load: " << imagePath << std::endl;
        std::cerr << "Usage: " << argv[0] << " [image_path]" << std::endl;
        return 1;
    }

    std::cout << "=== Line Metrology Demo ===" << std::endl;
    std::cout << "Image: " << imagePath << std::endl;
    std::cout << "Size: " << grayImage.Width() << " x " << grayImage.Height() << std::endl;

    // Create metrology model
    MetrologyModel model;

    // Configure measurement parameters
    MetrologyMeasureParams lineParams;
    lineParams.SetNumMeasures(20)         // 20 calipers along the line
              .SetMeasureLength(30, 5)    // Caliper: 30px half-length, 5px half-width
              .SetThreshold("auto")       // Automatic threshold
              .SetFitMethod("huber")      // Huber is more stable on real-image texture/noise
              .SetMinScore(0.3)           // Relax acceptance threshold for demo robustness
              .SetDistanceThreshold(2.0); // 2px outlier threshold

    // Add line measurements for IC chip edges
    // NOTE: Adjust these coordinates based on your actual image content
    // The IC chip in ic.png has edges approximately at these positions
    int w = grayImage.Width();
    int h = grayImage.Height();

    // Measure the IC chip package edges (approximate positions)
    // Top edge: horizontal line around row 120
    model.AddLineMeasure(120, 100, 120, w - 100,  // row1, col1, row2, col2
                         30.0, 5.0,                // measureLength1, measureLength2
                         "positive", "strongest",  // top edge: dark->bright
                         lineParams);

    // Left edge: vertical line around col 130
    model.AddLineMeasure(150, 130, h - 150, 130,
                         30.0, 5.0,
                         "positive", "strongest",  // left edge: dark->bright
                         lineParams);

    // Bottom edge: horizontal line around row 390
    model.AddLineMeasure(h - 120, 100, h - 120, w - 100,
                         30.0, 5.0,
                         "negative", "strongest",  // bottom edge: bright->dark
                         lineParams);

    // Right edge: vertical line around col 380
    model.AddLineMeasure(150, w - 130, h - 150, w - 130,
                         30.0, 5.0,
                         "negative", "strongest",  // right edge: bright->dark
                         lineParams);

    std::cout << "\nAdded " << model.NumObjects() << " line measurements" << std::endl;

    // Apply measurement
    Timer timer;
    timer.Start();
    bool success = model.Apply(grayImage);
    double elapsed = timer.ElapsedMs();

    if (!success) {
        std::cerr << "Measurement failed!" << std::endl;
        return 1;
    }

    // Output results
    std::cout << "\n=== Results ===" << std::endl;
    const char* names[] = {"Top edge", "Left edge", "Bottom edge", "Right edge"};

    for (int i = 0; i < model.NumObjects(); ++i) {
        auto result = model.GetLineResult(i);
        auto points = model.GetMeasuredPoints(i);

        // Calculate angle from line endpoints
        double dx = result.col2 - result.col1;
        double dy = result.row2 - result.row1;
        double angle = std::atan2(dy, dx) * 180.0 / 3.14159;

        std::cout << "\n" << names[i] << " (Line " << (i+1) << "):" << std::endl;
        std::cout << "  Start: (" << std::fixed << std::setprecision(2)
                  << result.col1 << ", " << result.row1 << ")" << std::endl;
        std::cout << "  End:   (" << result.col2 << ", " << result.row2 << ")" << std::endl;
        std::cout << "  Angle: " << std::setprecision(3) << angle << " deg" << std::endl;
        std::cout << "  Points used: " << result.numUsed << "/" << points.size() << std::endl;
        std::cout << "  RMS error: " << result.rmsError << " px" << std::endl;
    }

    std::cout << "\nTotal time: " << std::setprecision(2) << elapsed << " ms" << std::endl;

    // Visualize results
    QImage colorImg;
    Color::GrayToRgb(grayImage, colorImg);
    Draw::MetrologyModelResult(colorImg, model);

    // Save result
    WriteImage(colorImg, "tests/output/line_metrology_result.png");
    std::cout << "\nResult saved to: tests/output/line_metrology_result.png" << std::endl;

    // Display
    Window win("Line Metrology Demo");
    win.SetAutoResize(true);
    win.EnablePixelInfo(true);
    win.DispImage(colorImg);
    std::cout << "Press any key to close..." << std::endl;
    win.WaitKey(0);

    return 0;
}
