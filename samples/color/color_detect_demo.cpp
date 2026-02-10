/**
 * @file color_detect_demo.cpp
 * @brief Color detection demo (preset + tunable params)
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Color/ColorDetect.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/GUI/Window.h>

#include <iostream>
#include <iomanip>

using namespace Qi::Vision;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::Color;
using namespace Qi::Vision::GUI;

int main(int argc, char* argv[]) {
    std::string imagePath = "tests/data/halcon_images/color/citrus_fruits_10.png";
    std::string colorName = "orange";
    if (argc > 1) imagePath = argv[1];
    if (argc > 2) colorName = argv[2];

    QImage image;
    ReadImage(imagePath, image);
    if (image.Empty()) {
        std::cerr << "Failed to load: " << imagePath << std::endl;
        std::cerr << "Usage: " << argv[0] << " [image_path] [color_name]" << std::endl;
        return 1;
    }

    ColorDetectParams params;
    params.colorSpace = ColorSpace::HSV;
    params.valueGain = 1.0;
    params.minArea = 200;

    ColorDetectResult result = FindColor(image, colorName, params);

    std::cout << "=== Color Detect Demo ===\n";
    std::cout << "Image: " << imagePath << "\n";
    std::cout << "Color: " << colorName << "\n";
    std::cout << "Found: " << (result.found ? "yes" : "no") << "\n";
    std::cout << "Pixel count: " << result.pixelCount << "\n";
    std::cout << "Coverage: " << std::fixed << std::setprecision(4)
              << (result.coverage * 100.0) << "%\n";
    if (result.found) {
        std::cout << "BBox: x=" << result.boundingBox.x
                  << ", y=" << result.boundingBox.y
                  << ", w=" << result.boundingBox.width
                  << ", h=" << result.boundingBox.height << "\n";
    }

    QImage display = image;
    if (display.GetChannelType() == ChannelType::Gray) {
        QImage rgb;
        GrayToRgb(display, rgb);
        display = rgb;
    }

    if (result.found) {
        Draw::RegionAlpha(display, result.region, Scalar(0, 255, 0), 0.25);
        Draw::RegionContour(display, result.region, Scalar::Green(), 1);
        Draw::Rectangle(display, result.boundingBox, Scalar::Yellow(), 1);
    }

    WriteImage(display, "tests/output/color_detect_result.png");
    std::cout << "Saved: tests/output/color_detect_result.png\n";

    Window win("Color Detect Demo");
    win.SetAutoResize(true);
    win.EnablePixelInfo(true);
    win.DispImage(display);
    std::cout << "Press any key to close..." << std::endl;
    win.WaitKey(0);
    return 0;
}

