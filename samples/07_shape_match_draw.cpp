/**
 * @file 07_shape_match_draw.cpp
 * @brief Demonstrates shape matching with result visualization
 *
 * Shows how to:
 * 1. Load images from files
 * 2. Create shape model from template
 * 3. Save/Load model to/from file
 * 4. Find matches in search images
 * 5. Draw results and save visualization
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Draw.h>
#include <QiVision/Matching/ShapeModel.h>
#include <QiVision/Platform/Timer.h>
#include <QiVision/Platform/FileIO.h>

#include <iostream>
#include <cmath>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;
using namespace Qi::Vision::Platform;

// Convert RGB to grayscale
QImage ToGray(const QImage& color) {
    if (color.Channels() == 1) {
        return color.Clone();
    }

    QImage gray(color.Width(), color.Height(), PixelType::UInt8, ChannelType::Gray);
    const uint8_t* src = static_cast<const uint8_t*>(color.Data());
    uint8_t* dst = static_cast<uint8_t*>(gray.Data());
    size_t srcStride = color.Stride();
    size_t dstStride = gray.Stride();

    for (int32_t y = 0; y < color.Height(); ++y) {
        const uint8_t* srcRow = src + y * srcStride;
        uint8_t* dstRow = dst + y * dstStride;
        for (int32_t x = 0; x < color.Width(); ++x) {
            uint8_t r = srcRow[x * 3 + 0];
            uint8_t g = srcRow[x * 3 + 1];
            uint8_t b = srcRow[x * 3 + 2];
            dstRow[x] = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
    return gray;
}

int main() {
    std::cout << "=== QiVision Shape Matching Demo ===" << std::endl;

    // Define paths
    std::string dataDir = "tests/data/matching/image1/";
    std::string outputDir = "tests/data/matching/";
    std::string modelFile = "shape_model.qism";  // QiVision Shape Model

    std::vector<std::string> imageFiles = {
        "052640-20210901141310.jpg",
        "052640-20210901141317.jpg",
        "052640-20210901141321.jpg",
        "052640-20210901141324.jpg",
        "052640-20210901141327.jpg"
    };

    ShapeModel model;
    Timer timer;
    Rect2i modelROI;

    // =========================================================================
    // Part 1: Create or Load Model
    // =========================================================================

    if (FileExists(modelFile)) {
        // Load existing model
        std::cout << "\n1. Loading existing model from: " << modelFile << std::endl;
        timer.Start();
        if (!model.Load(modelFile)) {
            std::cerr << "   Failed to load model!" << std::endl;
            return 1;
        }
        std::cout << "   Model loaded in " << timer.ElapsedMs() << " ms" << std::endl;

        // Get model info
        auto stats = model.GetStats();
        std::cout << "   Model points: " << stats.numPoints << std::endl;
        std::cout << "   Pyramid levels: " << stats.numLevels << std::endl;
        std::cout << "   Model size: " << stats.Width() << " x " << stats.Height() << std::endl;

        // Use stats for ROI visualization
        modelROI.x = 0;
        modelROI.y = 0;
        modelROI.width = static_cast<int32_t>(stats.Width());
        modelROI.height = static_cast<int32_t>(stats.Height());

    } else {
        // Create new model from first image
        std::cout << "\n1. Creating new model from: " << imageFiles[0] << std::endl;

        QImage templateColor = QImage::FromFile(dataDir + imageFiles[0]);
        if (templateColor.Empty()) {
            std::cerr << "   Failed to load template image!" << std::endl;
            return 1;
        }
        QImage templateGray = ToGray(templateColor);
        std::cout << "   Image size: " << templateGray.Width() << "x" << templateGray.Height() << std::endl;

        // Define ROI - user specified: top-left (180, 100), bottom-right (390, 170)
        modelROI.x = 180;
        modelROI.y = 100;
        modelROI.width = 390 - 180;   // 210
        modelROI.height = 170 - 100;  // 70
        std::cout << "   ROI: (" << modelROI.x << "," << modelROI.y << ") "
                  << modelROI.width << "x" << modelROI.height << std::endl;

        // Create model with auto hysteresis contrast detection
        // Uses Otsu + percentile + BFS propagation for better weak edge extraction
        ModelParams modelParams;
        modelParams.SetContrastAutoHysteresis();  // Auto-detect with hysteresis thresholds
        modelParams.SetNumLevels(4);

        timer.Start();
        if (!model.Create(templateGray, modelROI, modelParams)) {
            std::cerr << "   Failed to create model!" << std::endl;
            return 1;
        }
        double createTime = timer.ElapsedMs();
        std::cout << "   Model created in " << createTime << " ms" << std::endl;

        // Show model statistics
        auto stats = model.GetStats();
        std::cout << "   Model points: " << stats.numPoints << std::endl;
        std::cout << "   Points per level: ";
        for (size_t i = 0; i < stats.pointsPerLevel.size(); ++i) {
            std::cout << stats.pointsPerLevel[i];
            if (i < stats.pointsPerLevel.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;

        // Save model for future use
        std::cout << "\n2. Saving model to: " << modelFile << std::endl;
        timer.Start();
        if (model.Save(modelFile)) {
            std::cout << "   Model saved in " << timer.ElapsedMs() << " ms" << std::endl;
        } else {
            std::cerr << "   Failed to save model!" << std::endl;
        }

        // Save template visualization with ROI
        QImage templateVis = Draw::PrepareForDrawing(templateGray);
        Draw::Rectangle(templateVis, modelROI, Color::Green(), 2);
        std::string roiPath = outputDir + "output_template_roi.jpg";
        templateVis.SaveToFile(roiPath);
        std::cout << "   Saved: " << roiPath << std::endl;

        // Save model edge points visualization
        // Draw extracted edge points on template image
        QImage modelVis = Draw::PrepareForDrawing(templateGray);
        auto modelPoints = model.GetModelPoints(0);  // Level 0 (finest)
        std::cout << "   Drawing " << modelPoints.size() << " model points..." << std::endl;

        // Model points are relative to ROI center, need to offset
        double centerX = modelROI.x + modelROI.width / 2.0;
        double centerY = modelROI.y + modelROI.height / 2.0;

        for (const auto& pt : modelPoints) {
            int32_t px = static_cast<int32_t>(centerX + pt.x + 0.5);
            int32_t py = static_cast<int32_t>(centerY + pt.y + 0.5);
            Draw::Pixel(modelVis, px, py, Color::Green());
            // Also draw neighboring pixels for visibility
            Draw::Pixel(modelVis, px+1, py, Color::Green());
            Draw::Pixel(modelVis, px, py+1, Color::Green());
        }
        Draw::Rectangle(modelVis, modelROI, Color::Red(), 1);
        std::string edgePath = outputDir + "output_model_edges.jpg";
        modelVis.SaveToFile(edgePath);
        std::cout << "   Saved: " << edgePath << " (extracted edge points)" << std::endl;
    }

    // =========================================================================
    // Part 2: Search in Images
    // =========================================================================

    std::cout << "\n3. Searching in images..." << std::endl;

    // Get model contour for visualization
    auto modelContour = model.GetModelContour(0);
    std::cout << "   Model contour has " << modelContour.size() << " points" << std::endl;

    // Search parameters - full 360° rotation search
    SearchParams searchParams;
    searchParams.SetMinScore(0.5);
    searchParams.SetMaxMatches(5);
    searchParams.SetGreediness(0.9);
    searchParams.SetAngleRange(0, 2.0 * M_PI);  // Full 360° search
    searchParams.SetSubpixel(SubpixelMethod::LeastSquares);

    // Process each image
    for (size_t i = 0; i < imageFiles.size(); ++i) {
        std::cout << "\n   [" << (i+1) << "/" << imageFiles.size() << "] "
                  << imageFiles[i] << std::endl;

        // Load image
        QImage searchColor = QImage::FromFile(dataDir + imageFiles[i]);
        if (searchColor.Empty()) {
            std::cerr << "      Failed to load image!" << std::endl;
            continue;
        }
        QImage searchGray = ToGray(searchColor);

        // Search
        timer.Reset();
        timer.Start();
        auto results = model.Find(searchGray, searchParams);
        double searchTime = timer.ElapsedMs();
        timer.Stop();

        std::cout << "      Found " << results.size() << " matches in "
                  << searchTime << " ms" << std::endl;

        // Draw results
        QImage resultImage = Draw::PrepareForDrawing(searchGray);

        for (size_t j = 0; j < results.size(); ++j) {
            const auto& match = results[j];
            std::cout << "      Match " << (j+1) << ": "
                      << "pos=(" << match.x << "," << match.y << ") "
                      << "angle=" << (match.angle * 180.0 / M_PI) << " deg "
                      << "score=" << match.score << std::endl;

            // Draw model contour at match position (green)
            Draw::MatchResultWithContour(resultImage, match, modelContour, Color::Green(), 2);

            // Draw bounding box (red)
            auto stats = model.GetStats();
            double halfW = stats.Width() / 2.0;
            double halfH = stats.Height() / 2.0;
            double cosA = std::cos(match.angle);
            double sinA = std::sin(match.angle);

            // Compute rotated corners
            double corners[4][2] = {
                {-halfW, -halfH}, {halfW, -halfH},
                {halfW, halfH}, {-halfW, halfH}
            };
            int32_t px[4], py[4];
            for (int k = 0; k < 4; ++k) {
                px[k] = static_cast<int32_t>(match.x + cosA * corners[k][0] - sinA * corners[k][1]);
                py[k] = static_cast<int32_t>(match.y + sinA * corners[k][0] + cosA * corners[k][1]);
            }
            // Draw box edges
            Draw::Line(resultImage, px[0], py[0], px[1], py[1], Color::Red(), 2);
            Draw::Line(resultImage, px[1], py[1], px[2], py[2], Color::Red(), 2);
            Draw::Line(resultImage, px[2], py[2], px[3], py[3], Color::Red(), 2);
            Draw::Line(resultImage, px[3], py[3], px[0], py[0], Color::Red(), 2);

            // Draw center cross
            Draw::Cross(resultImage, match.x, match.y, 10, Color::Yellow(), 2);
        }

        // Save result
        std::string outPath = outputDir + "output_match_" + std::to_string(i + 1) + ".jpg";
        if (resultImage.SaveToFile(outPath)) {
            std::cout << "      Saved: " << outPath << std::endl;
        }
    }

    std::cout << "\n=== Demo Complete ===" << std::endl;
    return 0;
}
