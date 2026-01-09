/**
 * @file 08_shape_match_large.cpp
 * @brief Shape matching test for large 2048x4001 images
 *
 * Uses image2 directory with larger line scan images
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Draw.h>
#include <QiVision/Matching/ShapeModel.h>
#include <QiVision/Platform/Timer.h>
#include <QiVision/Platform/FileIO.h>

#include <iostream>
#include <cmath>
#include <cstring>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;
using namespace Qi::Vision::Platform;

int main(int argc, char* argv[]) {
    std::cout << "=== QiVision Shape Matching - Large Image Test ===" << std::endl;

    // Define paths
    std::string dataDir = "tests/data/matching/image2/";
    std::string outputDir = "tests/data/matching/";
    std::string modelFile = "shape_model_large.qism";

    // Template image first, then search images
    std::string templateFile = "2025120119482739.bmp";
    std::vector<std::string> imageFiles = {
        "2025120119482739.bmp",   // Template image (for verification)
        "20251201194802191.bmp",
        "20251201194804137.bmp",
        "20251201194805266.bmp",
        "20251201194806360.bmp",
        "2025120119482935.bmp",
        "20251201194837127.bmp",
        "20251201194840615.bmp",
        "20251201194842271.bmp",
        "20251201194843675.bmp",
        "20251201194844759.bmp"
    };

    ShapeModel model;
    Timer timer;

    // =========================================================================
    // Part 1: Create or Load Model
    // =========================================================================

    if (FileExists(modelFile)) {
        std::cout << "\n1. Loading existing model from: " << modelFile << std::endl;
        timer.Start();
        if (!model.Load(modelFile)) {
            std::cerr << "   Failed to load model!" << std::endl;
            return 1;
        }
        std::cout << "   Model loaded in " << timer.ElapsedMs() << " ms" << std::endl;

        auto stats = model.GetStats();
        std::cout << "   Model points: " << stats.numPoints << std::endl;
        std::cout << "   Pyramid levels: " << stats.numLevels << std::endl;
        std::cout << "   Model size: " << stats.Width() << " x " << stats.Height() << std::endl;

    } else {
        // Create new model from template image with specified ROI
        std::cout << "\n1. Creating new model from: " << templateFile << std::endl;

        QImage templateImg = QImage::FromFile(dataDir + templateFile);
        if (!templateImg.IsValid()) {
            std::cerr << "   Failed to load template image!" << std::endl;
            return 1;
        }

        std::cout << "   Image size: " << templateImg.Width() << " x " << templateImg.Height() << std::endl;
        std::cout << "   Channels: " << templateImg.Channels() << std::endl;

        // Convert to grayscale if needed (8-bit BMP is loaded as RGB by stb_image)
        if (templateImg.Channels() == 3) {
            QImage gray(templateImg.Width(), templateImg.Height(), PixelType::UInt8, ChannelType::Gray);
            const uint8_t* src = static_cast<const uint8_t*>(templateImg.Data());
            uint8_t* dst = static_cast<uint8_t*>(gray.Data());
            size_t srcStride = templateImg.Stride();
            size_t dstStride = gray.Stride();

            for (int32_t y = 0; y < templateImg.Height(); ++y) {
                const uint8_t* srcRow = src + y * srcStride;
                uint8_t* dstRow = dst + y * dstStride;
                for (int32_t x = 0; x < templateImg.Width(); ++x) {
                    // For indexed BMP loaded as RGB, all channels are equal
                    dstRow[x] = srcRow[x * 3];
                }
            }
            templateImg = std::move(gray);
            std::cout << "   Converted to grayscale: " << templateImg.Channels() << " channel(s)" << std::endl;
        }

        // Template ROI for the "花" (flower) pattern
        // Region: (1065, 2720) to (1200, 2920) = 135 x 200
        int roiX, roiY, roiW, roiH;

        if (argc >= 5) {
            // Use command line arguments: x y w h
            roiX = std::atoi(argv[1]);
            roiY = std::atoi(argv[2]);
            roiW = std::atoi(argv[3]);
            roiH = std::atoi(argv[4]);
        } else {
            // Default: flower pattern region from user specification
            roiX = 1065;
            roiY = 2720;
            roiW = 135;  // 1200 - 1065
            roiH = 200;  // 2920 - 2720
        }

        std::cout << "   Template ROI: (" << roiX << ", " << roiY << ", " << roiW << ", " << roiH << ")" << std::endl;

        Rect2i roi(roiX, roiY, roiW, roiH);

        // Configure model parameters
        ModelParams modelParams;
        modelParams.numLevels = 5;  // More levels for larger images
        modelParams.angleStart = 0;
        modelParams.angleExtent = 2 * M_PI;  // 360 degrees
        // Lower contrast for 8-bit grayscale BMP
        modelParams.contrastHigh = 10;
        modelParams.contrastLow = 5;
        modelParams.metric = MetricMode::IgnoreLocalPolarity;
        modelParams.optimization = OptimizationMode::Auto;

        timer.Start();
        if (!model.Create(templateImg, roi, modelParams)) {
            std::cerr << "   Failed to create model!" << std::endl;
            return 1;
        }
        double createTime = timer.ElapsedMs();

        auto stats = model.GetStats();
        std::cout << "   Model created in " << createTime << " ms" << std::endl;
        std::cout << "   Model points: " << stats.numPoints << std::endl;
        std::cout << "   Pyramid levels: " << stats.numLevels << std::endl;

        // Save model
        timer.Reset();
        timer.Start();
        if (model.Save(modelFile)) {
            std::cout << "   Model saved to: " << modelFile << " in " << timer.ElapsedMs() << " ms" << std::endl;
        }

        // Save template ROI image
        {
            QImage roiImg = templateImg.Clone();
            // Convert to RGB for drawing
            QImage colorImg(roiImg.Width(), roiImg.Height(), PixelType::UInt8, ChannelType::RGB);
            const uint8_t* src = static_cast<const uint8_t*>(roiImg.Data());
            uint8_t* dst = static_cast<uint8_t*>(colorImg.Data());
            for (int32_t y = 0; y < roiImg.Height(); ++y) {
                for (int32_t x = 0; x < roiImg.Width(); ++x) {
                    uint8_t v = src[y * roiImg.Stride() + x];
                    size_t dstIdx = y * colorImg.Stride() + x * 3;
                    dst[dstIdx + 0] = v;
                    dst[dstIdx + 1] = v;
                    dst[dstIdx + 2] = v;
                }
            }
            // Draw ROI rectangle
            Draw::Rectangle(colorImg, roi.x, roi.y, roi.width, roi.height, Color::Green(), 2);
            std::string roiPath = outputDir + "output_large_template_roi.bmp";
            colorImg.SaveToFile(roiPath);
            std::cout << "   Saved template ROI: " << roiPath << std::endl;
        }

        // Save model edge points
        {
            auto modelPoints = model.GetModelPoints(0);
            QImage edgeImg(roi.width, roi.height, PixelType::UInt8, ChannelType::RGB);
            // Fill with black
            std::memset(edgeImg.Data(), 0, edgeImg.Height() * edgeImg.Stride());

            uint8_t* dst = static_cast<uint8_t*>(edgeImg.Data());
            for (const auto& pt : modelPoints) {
                int32_t px = static_cast<int32_t>(pt.x + roi.width / 2);
                int32_t py = static_cast<int32_t>(pt.y + roi.height / 2);
                if (px >= 0 && px < roi.width && py >= 0 && py < roi.height) {
                    size_t idx = py * edgeImg.Stride() + px * 3;
                    dst[idx + 0] = 0;
                    dst[idx + 1] = 255;
                    dst[idx + 2] = 0;
                }
            }
            std::string edgePath = outputDir + "output_large_model_edges.bmp";
            edgeImg.SaveToFile(edgePath);
            std::cout << "   Saved model edges: " << edgePath << " (" << modelPoints.size() << " points)" << std::endl;
        }
    }

    // =========================================================================
    // Part 2: Search in images
    // =========================================================================

    std::cout << "\n2. Searching in images..." << std::endl;

    SearchParams searchParams;
    searchParams.angleStart = 0;
    searchParams.angleExtent = 2 * M_PI;
    searchParams.minScore = 0.7;
    searchParams.greediness = 0.9;
    searchParams.maxMatches = 10;

    double totalTime = 0;
    int numImages = 0;

    for (size_t i = 0; i < imageFiles.size(); ++i) {
        std::string imagePath = dataDir + imageFiles[i];
        QImage searchImg = QImage::FromFile(imagePath);

        if (!searchImg.IsValid()) {
            std::cerr << "   [" << (i+1) << "/" << imageFiles.size() << "] Failed to load: " << imageFiles[i] << std::endl;
            continue;
        }

        // Convert to grayscale if needed
        if (searchImg.Channels() == 3) {
            QImage gray(searchImg.Width(), searchImg.Height(), PixelType::UInt8, ChannelType::Gray);
            const uint8_t* src = static_cast<const uint8_t*>(searchImg.Data());
            uint8_t* dst = static_cast<uint8_t*>(gray.Data());
            size_t srcStride = searchImg.Stride();
            size_t dstStride = gray.Stride();

            for (int32_t y = 0; y < searchImg.Height(); ++y) {
                const uint8_t* srcRow = src + y * srcStride;
                uint8_t* dstRow = dst + y * dstStride;
                for (int32_t x = 0; x < searchImg.Width(); ++x) {
                    dstRow[x] = srcRow[x * 3];
                }
            }
            searchImg = std::move(gray);
        }

        timer.Reset();
        timer.Start();
        auto matches = model.Find(searchImg, searchParams);
        double searchTime = timer.ElapsedMs();
        timer.Stop();
        totalTime += searchTime;
        numImages++;

        std::cout << "   [" << (i+1) << "/" << imageFiles.size() << "] " << imageFiles[i] << std::endl;
        std::cout << "      Image size: " << searchImg.Width() << " x " << searchImg.Height() << std::endl;
        std::cout << "      Found " << matches.size() << " matches in " << searchTime << " ms" << std::endl;

        for (size_t j = 0; j < matches.size() && j < 3; ++j) {
            const auto& m = matches[j];
            std::cout << "      Match " << (j+1) << ": pos=(" << m.x << "," << m.y
                      << ") angle=" << (m.angle * 180.0 / M_PI) << " deg score=" << m.score << std::endl;
        }

        // Save visualization
        if (searchImg.Channels() == 1) {
            // Convert to RGB for drawing
            QImage colorImg(searchImg.Width(), searchImg.Height(), PixelType::UInt8, ChannelType::RGB);
            const uint8_t* src = static_cast<const uint8_t*>(searchImg.Data());
            uint8_t* dst = static_cast<uint8_t*>(colorImg.Data());
            for (int32_t y = 0; y < searchImg.Height(); ++y) {
                for (int32_t x = 0; x < searchImg.Width(); ++x) {
                    uint8_t v = src[y * searchImg.Stride() + x];
                    size_t dstIdx = y * colorImg.Stride() + x * 3;
                    dst[dstIdx + 0] = v;
                    dst[dstIdx + 1] = v;
                    dst[dstIdx + 2] = v;
                }
            }

            // Get model contour for drawing
            auto modelContour = model.GetModelContour();

            // Draw matches with contour and bounding box
            for (const auto& m : matches) {
                // Draw model contour at match position (green)
                Draw::MatchResultWithContour(colorImg, m, modelContour, Color::Green(), 2);

                // Draw bounding box (red)
                auto stats = model.GetStats();
                double halfW = stats.Width() / 2.0;
                double halfH = stats.Height() / 2.0;
                double cosA = std::cos(m.angle);
                double sinA = std::sin(m.angle);

                // Compute rotated corners
                double corners[4][2] = {
                    {-halfW, -halfH}, {halfW, -halfH},
                    {halfW, halfH}, {-halfW, halfH}
                };
                int32_t px[4], py[4];
                for (int k = 0; k < 4; ++k) {
                    px[k] = static_cast<int32_t>(m.x + cosA * corners[k][0] - sinA * corners[k][1]);
                    py[k] = static_cast<int32_t>(m.y + sinA * corners[k][0] + cosA * corners[k][1]);
                }
                // Draw box edges
                Draw::Line(colorImg, px[0], py[0], px[1], py[1], Color::Red(), 2);
                Draw::Line(colorImg, px[1], py[1], px[2], py[2], Color::Red(), 2);
                Draw::Line(colorImg, px[2], py[2], px[3], py[3], Color::Red(), 2);
                Draw::Line(colorImg, px[3], py[3], px[0], py[0], Color::Red(), 2);

                // Draw center cross
                Draw::Cross(colorImg, m.x, m.y, 15, Color::Yellow(), 2);
            }

            std::string outFile = outputDir + "output_large_" + std::to_string(i+1) + ".bmp";
            colorImg.SaveToFile(outFile);
            std::cout << "      Saved: " << outFile << std::endl;
        }
    }

    if (numImages > 0) {
        std::cout << "\n   Average search time: " << (totalTime / numImages) << " ms" << std::endl;
    }

    std::cout << "\n=== Demo Complete ===" << std::endl;
    return 0;
}
