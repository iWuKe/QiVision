/**
 * @file ellipse_fit_synthetic.cpp
 * @brief Ellipse fitting on real image contours
 *
 * Workflow:
 * 1) Read image (grayscale)
 * 2) Edge detection -> subpixel contours
 * 3) Pick longest contour
 * 4) Fit ellipse and report error metrics
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Edge/Edge.h>
#include <QiVision/Contour/Contour.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/GUI/Window.h>
#include <QiVision/Core/Constants.h>

#include <iostream>
#include <limits>

using namespace Qi::Vision;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::Edge;
using namespace Qi::Vision::Contour;
using namespace Qi::Vision::GUI;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>\n";
        return 1;
    }

    std::string imagePath = argv[1];

    QImage gray;
    ReadImageGray(imagePath, gray);
    if (gray.Empty()) {
        std::cerr << "Failed to load image: " << imagePath << "\n";
        return 1;
    }

    std::cout << "=== Ellipse Fit (Real Image) ===\n";
    std::cout << "Image: " << imagePath << " (" << gray.Width() << "x" << gray.Height() << ")\n";

    // Edge detection (subpixel contours)
    QContourArray contours;
    EdgesSubPixAuto(gray, contours, "canny", 1.0);

    if (contours.Empty()) {
        std::cerr << "No contours found.\n";
        return 1;
    }

    // Pick longest contour
    int bestIdx = -1;
    double bestLen = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < contours.Size(); ++i) {
        double len = contours[i].Length();
        if (len > bestLen) {
            bestLen = len;
            bestIdx = i;
        }
    }

    if (bestIdx < 0) {
        std::cerr << "Failed to select contour.\n";
        return 1;
    }

    const QContour& contour = contours[bestIdx];

    double row = 0, col = 0, phi = 0, ra = 0, rb = 0;
    if (!FitEllipseContourXld(contour, row, col, phi, ra, rb)) {
        std::cerr << "Ellipse fit failed.\n";
        return 1;
    }

    std::cout << "Ellipse: center=(" << col << ", " << row << ")"
              << " ra=" << ra << " rb=" << rb
              << " angle=" << (phi * 180.0 / Qi::Vision::PI) << " deg\n";

    // Visualization
    QImage display;
    Color::GrayToRgb(gray, display);
    Draw::Contour(display, contour, Scalar(0, 255, 255), 1);
    Draw::Ellipse(display, Point2d(col, row), ra, rb, phi, Scalar(0, 255, 0), 2);
    Draw::Cross(display, static_cast<int>(col), static_cast<int>(row), 8, Scalar(255, 0, 0), 2);

    Window win("Ellipse Fit - Real Image");
    win.SetAutoResize(true, 1000, 800);
    win.EnablePixelInfo(true);
    win.DispImage(display);

    std::cout << "Press any key to close...\n";
    win.WaitKey(0);

    return 0;
}
