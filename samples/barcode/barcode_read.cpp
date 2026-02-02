/**
 * @file barcode_read.cpp
 * @brief Barcode reading demonstration
 *
 * Demonstrates reading various barcode types including:
 * - 1D: Code128, Code39, EAN-13, UPC-A
 * - 2D: QR Code, Data Matrix, PDF417
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Barcode/Barcode.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/GUI/Window.h>

#include <iostream>
#include <iomanip>

using namespace Qi::Vision;

int main(int argc, char* argv[]) {
    std::cout << "=== QiVision Barcode Reading Demo ===\n\n";
    std::cout << "ZXing-cpp version: " << Barcode::GetVersion() << "\n\n";

    // Check command line arguments
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>\n\n";
        std::cout << "Supported formats:\n";
        std::cout << "  1D: Code128, Code39, Code93, Codabar, EAN-8, EAN-13, ITF, UPC-A, UPC-E\n";
        std::cout << "  2D: QR Code, Data Matrix, PDF417, Aztec\n";
        return 1;
    }

    std::string imagePath = argv[1];

    // Load image
    QImage image;
    try {
        IO::ReadImage(imagePath, image);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load image: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Loaded: " << imagePath << " (" << image.Width() << "x" << image.Height() << ")\n\n";

    // Read all barcodes with default settings
    std::cout << "Searching for barcodes...\n";
    Barcode::BarcodeParams params = Barcode::BarcodeParams::Default();
    params.maxNumberOfSymbols = 0;  // Find all

    auto results = Barcode::ReadBarcodes(image, params);

    if (results.empty()) {
        std::cout << "No barcodes found. Trying with harder settings...\n";
        params = Barcode::BarcodeParams::Robust();
        params.maxNumberOfSymbols = 0;
        results = Barcode::ReadBarcodes(image, params);
    }

    if (results.empty()) {
        std::cout << "No barcodes found in the image.\n";
        return 0;
    }

    // Print results
    std::cout << "\nFound " << results.size() << " barcode(s):\n\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << "[" << (i + 1) << "] " << r.formatName << "\n";
        std::cout << "    Text: " << r.text << "\n";
        std::cout << "    Position: (" << std::fixed << std::setprecision(1)
                  << r.position.x << ", " << r.position.y << ")\n";
        std::cout << "    Angle: " << std::setprecision(2)
                  << (r.angle * 180.0 / 3.14159265) << " deg\n";
        if (r.symbolVersion > 0) {
            std::cout << "    Version: " << r.symbolVersion << "\n";
        }
        if (!r.ecLevel.empty()) {
            std::cout << "    EC Level: " << r.ecLevel << "\n";
        }
        if (r.isMirrored) {
            std::cout << "    Mirrored: yes\n";
        }
        std::cout << "\n";
    }

    // Display results
    QImage display;
    if (image.Channels() == 1) {
        Color::GrayToRgb(image, display);
    } else {
        display = image.Clone();
    }

    // Draw bounding boxes and text for each barcode
    for (const auto& r : results) {
        // Draw corners as polygon
        if (r.corners.size() == 4) {
            for (int i = 0; i < 4; ++i) {
                int j = (i + 1) % 4;
                Draw::Line(display, r.corners[i], r.corners[j], Scalar(0, 255, 0), 2);
            }
        }

        // Draw center cross
        Draw::Cross(display, r.position, 15, 0, Scalar(255, 0, 0), 2);
    }

    // Show in window
    GUI::Window window("Barcode Detection", display.Width(), display.Height());
    window.DispImage(display);

    std::cout << "Press any key to close...\n";
    window.WaitKey(0);

    return 0;
}
