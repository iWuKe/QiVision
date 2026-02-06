/**
 * @file polar_ocr_circular_barcode.cpp
 * @brief Demo: Caliper circle detection + polar transform + OCR + inverse overlay
 *
 * Steps:
 * 1) Detect circle using Metrology (caliper measurement)
 * 2) Polar transform around the detected circle center
 * 3) Run OCR on the polar image
 * 4) Map OCR results back to original image and draw overlays
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Measure/Metrology.h>
#include <QiVision/Transform/PolarTransform.h>
#include <QiVision/OCR/OCR.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/GUI/Window.h>

#include <cmath>
#include <iostream>
#include <iomanip>

using namespace Qi::Vision;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::Measure;
using namespace Qi::Vision::Transform;
using namespace Qi::Vision::OCR;
using namespace Qi::Vision::GUI;

namespace {

constexpr double TWO_PI = 6.28318530717958647692;

Point2d PolarPixelToCartesian(const Point2d& polarPixel,
                              int polarWidth,
                              int polarHeight,
                              double maxRadius,
                              const Point2d& center,
                              bool flipRadius) {
    if (polarWidth <= 0 || polarHeight <= 0 || maxRadius <= 0) {
        return Point2d();
    }
    double theta = polarPixel.x / static_cast<double>(polarWidth) * TWO_PI;
    double y = flipRadius ? (static_cast<double>(polarHeight - 1) - polarPixel.y) : polarPixel.y;
    double radius = y / static_cast<double>(polarHeight) * maxRadius;
    return PointPolarToCartesian(theta, radius, center);
}

Point2d AveragePoint(const std::vector<Point2d>& points) {
    if (points.empty()) {
        return Point2d();
    }
    double sumX = 0.0;
    double sumY = 0.0;
    for (const auto& p : points) {
        sumX += p.x;
        sumY += p.y;
    }
    return Point2d(sumX / points.size(), sumY / points.size());
}

} // namespace

int main(int argc, char* argv[]) {
    std::cout << "=== Polar OCR (Circular Barcode) Demo ===\n";

    std::string imagePath = "tests/data/halcon_images/circular_barcode.png";
    std::string modelDir = OCR::GetDefaultModelDir();
    int gpuIndex = -1;

    // Args: [image_path] [model_dir] [--gpu]
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gpu") {
            gpuIndex = 0;
        } else if (imagePath == "tests/data/halcon_images/circular_barcode.png") {
            imagePath = arg;
        } else {
            modelDir = arg;
        }
    }

    if (!OCR::IsAvailable()) {
        std::cerr << "OCR backend not available. Build with:\n";
        std::cerr << "  cmake -DQIVISION_BUILD_OCR=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime\n";
        return 1;
    }

    QImage gray;
    ReadImageGray(imagePath, gray);
    if (gray.Empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return 1;
    }

    std::cout << "Image: " << imagePath << " (" << gray.Width() << "x" << gray.Height() << ")\n";

    // ---------------------------------------------------------------------
    // 1) Detect circle using Metrology (calipers)
    // ---------------------------------------------------------------------
    double circleRow = 439.0;
    double circleCol = 454.0;
    double circleRadius = 340.0;

    int numCalipers = 60;
    double measureLength1 = 25;
    double measureLength2 = 8;

    MetrologyModel model;
    MetrologyMeasureParams params;
    params.SetNumMeasures(numCalipers)
          .SetThreshold("auto")
          .SetMeasureSigma(1.0);

    model.AddCircleMeasure(
        circleRow, circleCol, circleRadius,
        measureLength1, measureLength2,
        "all", "all", params
    );

    model.Apply(gray);
    auto result = model.GetCircleResult(0);

    if (result.numUsed < 5) {
        std::cerr << "Circle detection failed. Using initial estimate.\n";
    }

    Point2d center(result.numUsed >= 5 ? result.column : circleCol,
                   result.numUsed >= 5 ? result.row : circleRow);
    double radius = result.numUsed >= 5 ? result.radius : circleRadius;

    std::cout << "Circle center: (" << std::fixed << std::setprecision(2)
              << center.x << ", " << center.y << ") r=" << radius << "\n";

    // ---------------------------------------------------------------------
    // 2) Polar transform
    // ---------------------------------------------------------------------
    bool flipRadius = true; // outer ring on top
    QImage polar;
    // Use default size: width≈2πR, height≈R (correct polar sampling)
    CartesianToPolar(gray, polar, center, radius, 0, 0,
                     PolarMode::Linear, PolarInterpolation::Bilinear, flipRadius);

    std::cout << "Polar image: " << polar.Width() << "x" << polar.Height() << "\n";

    // ---------------------------------------------------------------------
    // 3) OCR on polar image
    // ---------------------------------------------------------------------
    std::cout << "Initializing OCR from: " << modelDir << "\n";
    std::cout << "Device: " << (gpuIndex >= 0 ? "GPU" : "CPU") << "\n";

    if (!OCR::InitOCR(modelDir, gpuIndex)) {
        std::cerr << "Failed to init OCR. Check model files in: " << modelDir << "\n";
        return 1;
    }

    OCRParams ocrParams = OCRParams::Default();
    ocrParams.doAngleClassify = false;

    std::cout << "Running OCR...\n";
    OCRResult ocrResult = OCR::RecognizeText(polar, ocrParams);

    std::cout << "Text blocks: " << ocrResult.Size() << "\n";

    // ---------------------------------------------------------------------
    // 4) Draw results on original image (inverse mapping)
    // ---------------------------------------------------------------------
    QImage displayOrig;
    Color::GrayToRgb(gray, displayOrig);
    Draw::MetrologyModelResult(displayOrig, model);
    Draw::Circle(displayOrig, static_cast<int>(center.x), static_cast<int>(center.y),
                 static_cast<int>(radius), Scalar(0, 255, 0), 2);

    QImage displayOcr;
    Color::GrayToRgb(gray, displayOcr);

    // Polar display
    QImage polarColor;
    Color::GrayToRgb(polar, polarColor);

    for (const auto& tb : ocrResult.textBlocks) {
        if (tb.corners.size() != 4) {
            continue;
        }

        std::vector<Point2d> mapped;
        mapped.reserve(4);
        for (const auto& p : tb.corners) {
            mapped.push_back(PolarPixelToCartesian(p, polar.Width(), polar.Height(), radius, center, flipRadius));
        }

        // Draw on original
        for (int i = 0; i < 4; ++i) {
            int j = (i + 1) % 4;
            Draw::Line(displayOcr, mapped[i], mapped[j], Scalar(0, 255, 255), 2);
        }

        // Label near center of the box
        Point2d labelPos = AveragePoint(mapped);
        Draw::Text(displayOcr, static_cast<int>(labelPos.x), static_cast<int>(labelPos.y),
                   tb.text, Scalar(255, 0, 0), 1);

        // Draw on polar image (for verification)
        for (int i = 0; i < 4; ++i) {
            int j = (i + 1) % 4;
            Draw::Line(polarColor, tb.corners[i], tb.corners[j], Scalar(0, 255, 0), 2);
        }
        Point2d polarLabel = AveragePoint(tb.corners);
        Draw::Text(polarColor, static_cast<int>(polarLabel.x), static_cast<int>(polarLabel.y),
                   tb.text, Scalar(0, 255, 0), 1);
    }

    // ---------------------------------------------------------------------
    // 5) Show
    // ---------------------------------------------------------------------
    Window winOrig("Polar OCR - Original");
    winOrig.SetAutoResize(true, 900, 900);
    winOrig.EnablePixelInfo(true);
    winOrig.DispImage(displayOrig);

    Window winPolar("Polar OCR - Polar");
    winPolar.SetAutoResize(true, 1600, 600);
    winPolar.EnablePixelInfo(true);
    winPolar.DispImage(polarColor);

    Window winOcr("Polar OCR - Inverse");
    winOcr.SetAutoResize(true, 900, 900);
    winOcr.EnablePixelInfo(true);
    winOcr.DispImage(displayOcr);

    std::cout << "Press any key to close...\n";
    winOrig.WaitKey(0);

    OCR::ReleaseOCR();

    return 0;
}
