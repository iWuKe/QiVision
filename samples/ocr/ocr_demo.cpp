/**
 * @file ocr_demo.cpp
 * @brief OCR (Optical Character Recognition) demonstration
 *
 * Demonstrates:
 * - Model initialization
 * - Text detection and recognition
 * - Chinese and English text support
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/OCR/OCR.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/GUI/Window.h>

#include <iostream>
#include <iomanip>

using namespace Qi::Vision;

int main(int argc, char* argv[]) {
    std::cout << "=== QiVision OCR Demo ===\n\n";
    std::cout << "OCR Version: " << OCR::GetVersion() << "\n";
    std::cout << "Available: " << (OCR::IsAvailable() ? "Yes" : "No") << "\n\n";

    if (!OCR::IsAvailable()) {
        std::cerr << "OCR not available. Please build with:\n";
        std::cerr << "  cmake -DQIVISION_BUILD_OCR=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime\n";
        return 1;
    }

    // Check command line arguments
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path> [model_dir] [--gpu]\n\n";
        std::cout << "Arguments:\n";
        std::cout << "  image_path  - Path to image containing text\n";
        std::cout << "  model_dir   - Optional: Directory containing OCR models\n";
        std::cout << "                (default: " << OCR::GetDefaultModelDir() << ")\n";
        std::cout << "  --gpu       - Use GPU acceleration (CUDA)\n\n";
        std::cout << "Required models:\n";
        std::cout << "  - ch_PP-OCRv4_det_infer.onnx\n";
        std::cout << "  - ch_ppocr_mobile_v2.0_cls_infer.onnx\n";
        std::cout << "  - ch_PP-OCRv4_rec_infer.onnx\n";
        std::cout << "  - ppocr_keys_v1.txt\n\n";
        std::cout << "Download from: https://paddleocr.bj.bcebos.com/PP-OCRv4/\n";
        return 1;
    }

    std::string imagePath = argv[1];
    std::string modelDir = OCR::GetDefaultModelDir();
    int gpuIndex = -1;  // -1 = CPU

    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gpu") {
            gpuIndex = 0;  // Use first GPU
        } else {
            modelDir = arg;
        }
    }

    // Initialize OCR model
    std::cout << "Initializing OCR models from: " << modelDir << "\n";
    std::cout << "Device: " << (gpuIndex >= 0 ? "GPU (CUDA)" : "CPU") << "\n";
    if (!OCR::InitOCR(modelDir, gpuIndex)) {
        std::cerr << "Failed to initialize OCR models.\n";
        std::cerr << "Please ensure model files are in: " << modelDir << "\n";
        return 1;
    }
    std::cout << "OCR initialized successfully.\n\n";

    // Load image
    QImage image;
    try {
        IO::ReadImage(imagePath, image);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load image: " << e.what() << std::endl;
        OCR::ReleaseOCR();
        return 1;
    }
    std::cout << "Loaded: " << imagePath << " (" << image.Width() << "x" << image.Height() << ")\n\n";

    // Recognize text
    std::cout << "Running OCR...\n";
    OCR::OCRParams params = OCR::OCRParams::Default();

    try {
        auto result = OCR::RecognizeText(image, params);

        std::cout << "\n=== Results ===\n";
        std::cout << "Detection time: " << std::fixed << std::setprecision(1)
                  << result.detectTime << " ms\n";
        std::cout << "Recognition time: " << result.recognizeTime << " ms\n";
        std::cout << "Total time: " << result.totalTime << " ms\n\n";

        if (result.Empty()) {
            std::cout << "No text found in the image.\n";
        } else {
            std::cout << "Found " << result.Size() << " text block(s):\n\n";

            for (size_t i = 0; i < result.textBlocks.size(); ++i) {
                const auto& tb = result.textBlocks[i];
                std::cout << "[" << (i + 1) << "] " << tb.text << "\n";
                std::cout << "    Confidence: " << std::setprecision(2)
                          << (tb.confidence * 100) << "%\n";
                auto rect = tb.BoundingRect();
                std::cout << "    Position: (" << rect.x << ", " << rect.y
                          << ", " << rect.width << "x" << rect.height << ")\n\n";
            }

            std::cout << "=== Full Text ===\n";
            std::cout << result.fullText << "\n";
        }

        // Display result with annotations
        QImage display;
        if (image.Channels() == 1) {
            Color::GrayToRgb(image, display);
        } else {
            display = image.Clone();
        }

        // Draw text boxes
        for (const auto& tb : result.textBlocks) {
            if (tb.corners.size() == 4) {
                for (int j = 0; j < 4; ++j) {
                    int k = (j + 1) % 4;
                    Draw::Line(display, tb.corners[j], tb.corners[k], Scalar(0, 255, 0), 2);
                }
            }
        }

        // Show in window
        GUI::Window window("OCR Result", display.Width(), display.Height());
        window.DispImage(display);

        std::cout << "\nPress any key to close...\n";
        window.WaitKey(0);

    } catch (const std::exception& e) {
        std::cerr << "OCR failed: " << e.what() << "\n";
    }

    // Cleanup
    OCR::ReleaseOCR();

    return 0;
}
