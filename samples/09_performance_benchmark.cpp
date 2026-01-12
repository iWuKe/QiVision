/**
 * @file 09_performance_benchmark.cpp
 * @brief Detailed performance benchmark for ShapeModel
 *
 * Tests both small (640x512) and large (2048x4001) images with detailed timing
 * breakdown for each step in Create() and Find() operations.
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Matching/ShapeModel.h>
#include <QiVision/Internal/AnglePyramid.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;
using namespace Qi::Vision::Internal;

// Convert color image to grayscale
QImage ToGray(const QImage& color) {
    if (color.Channels() == 1) return color;

    QImage gray(color.Width(), color.Height(), PixelType::UInt8, ChannelType::Gray);
    const uint8_t* src = static_cast<const uint8_t*>(color.Data());
    uint8_t* dst = static_cast<uint8_t*>(gray.Data());
    int32_t srcStride = color.Stride();
    int32_t dstStride = gray.Stride();
    int32_t channels = color.Channels();

    for (int32_t y = 0; y < color.Height(); ++y) {
        for (int32_t x = 0; x < color.Width(); ++x) {
            int32_t srcIdx = y * srcStride + x * channels;
            int sum = src[srcIdx] + src[srcIdx + 1] + src[srcIdx + 2];
            dst[y * dstStride + x] = static_cast<uint8_t>(sum / 3);
        }
    }
    return gray;
}

void PrintSeparator(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

struct TestResult {
    std::string name;
    int32_t width = 0;
    int32_t height = 0;

    // Create timing
    ShapeModelCreateTiming createTiming;
    AnglePyramidTiming createPyramidTiming;

    // Find timing (average over multiple images)
    ShapeModelFindTiming findTiming;
    AnglePyramidTiming findPyramidTiming;
    int numFinds = 0;
};

void RunBenchmark(const std::string& name,
                  const std::string& templatePath,
                  const std::vector<std::string>& searchPaths,
                  const Rect2i& roi,
                  TestResult& result) {
    result.name = name;

    // Load template image
    QImage templateColor = QImage::FromFile(templatePath);
    if (templateColor.Empty()) {
        std::cerr << "Failed to load: " << templatePath << "\n";
        return;
    }
    QImage templateGray = ToGray(templateColor);
    result.width = templateGray.Width();
    result.height = templateGray.Height();

    std::cout << "\nImage: " << result.width << "x" << result.height << "\n";
    std::cout << "ROI: " << roi.width << "x" << roi.height << "\n";

    // ==========================================================================
    // Test Create() with detailed timing
    // ==========================================================================
    std::cout << "\n--- ShapeModel::Create() ---\n";

    // Create model with timing enabled
    ShapeModel model;
    model.SetTimingParams(ShapeModelTimingParams().SetEnableTiming(true));

    ModelParams modelParams;
    modelParams.SetContrastAutoHysteresis();
    modelParams.SetNumLevels(4);

    model.Create(templateGray, roi, modelParams);
    result.createTiming = model.GetCreateTiming();

    auto stats = model.GetStats();
    std::cout << "Model points: " << stats.numPoints << "\n\n";
    result.createTiming.Print();

    // Also get AnglePyramid timing separately for detailed breakdown
    AnglePyramidParams pyramidParams;
    pyramidParams.numLevels = 4;
    pyramidParams.enableTiming = true;

    AnglePyramid templatePyramid;
    templatePyramid.Build(templateGray, pyramidParams);
    result.createPyramidTiming = templatePyramid.GetTiming();

    std::cout << "\nAnglePyramid::Build breakdown (template):\n";
    result.createPyramidTiming.Print();

    // ==========================================================================
    // Test Find() with detailed timing
    // ==========================================================================
    std::cout << "\n--- ShapeModel::Find() ---\n";

    SearchParams searchParams;
    searchParams.SetMinScore(0.5);
    searchParams.SetMaxMatches(5);
    searchParams.SetGreediness(0.9);
    searchParams.SetAngleRange(0, 2.0 * 3.14159265358979);  // Full 360°
    searchParams.SetSubpixel(SubpixelMethod::LeastSquares);

    // Accumulate timing
    ShapeModelFindTiming avgFindTiming;
    AnglePyramidTiming avgPyramidTiming;

    for (size_t i = 0; i < searchPaths.size(); ++i) {
        QImage searchColor = QImage::FromFile(searchPaths[i]);
        if (searchColor.Empty()) continue;
        QImage searchGray = ToGray(searchColor);

        // Find with timing enabled
        auto results = model.Find(searchGray, searchParams);
        const auto& ft = model.GetFindTiming();

        // Accumulate timing
        avgFindTiming.totalMs += ft.totalMs;
        avgFindTiming.pyramidBuildMs += ft.pyramidBuildMs;
        avgFindTiming.coarseSearchMs += ft.coarseSearchMs;
        avgFindTiming.pyramidRefineMs += ft.pyramidRefineMs;
        avgFindTiming.subpixelRefineMs += ft.subpixelRefineMs;
        avgFindTiming.nmsMs += ft.nmsMs;
        avgFindTiming.numCoarseCandidates += ft.numCoarseCandidates;
        avgFindTiming.numFinalMatches += ft.numFinalMatches;

        // Also get standalone pyramid timing for detailed breakdown
        AnglePyramid searchPyramid;
        AnglePyramidParams searchPyramidParams;
        searchPyramidParams.numLevels = 4;
        searchPyramidParams.enableTiming = true;
        searchPyramid.Build(searchGray, searchPyramidParams);

        const auto& pt = searchPyramid.GetTiming();
        avgPyramidTiming.totalMs += pt.totalMs;
        avgPyramidTiming.toFloatMs += pt.toFloatMs;
        avgPyramidTiming.gaussPyramidMs += pt.gaussPyramidMs;
        avgPyramidTiming.sobelMs += pt.sobelMs;
        avgPyramidTiming.sqrtMs += pt.sqrtMs;
        avgPyramidTiming.atan2Ms += pt.atan2Ms;
        avgPyramidTiming.quantizeMs += pt.quantizeMs;
        avgPyramidTiming.extractEdgeMs += pt.extractEdgeMs;
        avgPyramidTiming.copyMs += pt.copyMs;

        result.numFinds++;

        std::cout << "[" << (i+1) << "/" << searchPaths.size() << "] "
                  << "Find: " << std::fixed << std::setprecision(1) << ft.totalMs << "ms, "
                  << "Matches: " << results.size();
        if (!results.empty()) {
            std::cout << ", Score: " << std::fixed << std::setprecision(3) << results[0].score;
        }
        std::cout << "\n";
    }

    if (result.numFinds > 0) {
        // Compute averages
        double n = result.numFinds;
        result.findTiming.totalMs = avgFindTiming.totalMs / n;
        result.findTiming.pyramidBuildMs = avgFindTiming.pyramidBuildMs / n;
        result.findTiming.coarseSearchMs = avgFindTiming.coarseSearchMs / n;
        result.findTiming.pyramidRefineMs = avgFindTiming.pyramidRefineMs / n;
        result.findTiming.subpixelRefineMs = avgFindTiming.subpixelRefineMs / n;
        result.findTiming.nmsMs = avgFindTiming.nmsMs / n;
        result.findTiming.numCoarseCandidates = avgFindTiming.numCoarseCandidates / result.numFinds;
        result.findTiming.numFinalMatches = avgFindTiming.numFinalMatches / result.numFinds;

        result.findPyramidTiming.totalMs = avgPyramidTiming.totalMs / n;
        result.findPyramidTiming.toFloatMs = avgPyramidTiming.toFloatMs / n;
        result.findPyramidTiming.gaussPyramidMs = avgPyramidTiming.gaussPyramidMs / n;
        result.findPyramidTiming.sobelMs = avgPyramidTiming.sobelMs / n;
        result.findPyramidTiming.sqrtMs = avgPyramidTiming.sqrtMs / n;
        result.findPyramidTiming.atan2Ms = avgPyramidTiming.atan2Ms / n;
        result.findPyramidTiming.quantizeMs = avgPyramidTiming.quantizeMs / n;
        result.findPyramidTiming.extractEdgeMs = avgPyramidTiming.extractEdgeMs / n;
        result.findPyramidTiming.copyMs = avgPyramidTiming.copyMs / n;

        std::cout << "\nAverage Find() breakdown:\n";
        result.findTiming.Print();

        std::cout << "\nAverage AnglePyramid::Build breakdown (search image):\n";
        result.findPyramidTiming.Print();
    }
}

int main() {
    PrintSeparator("QiVision Performance Benchmark");

    std::string dataDir = "tests/data/matching/";

    // ==========================================================================
    // Small Image Test (640x512) - in image1/
    // ==========================================================================
    TestResult smallResult;
    {
        PrintSeparator("Small Image Test (640x512)");

        std::vector<std::string> smallImages = {
            dataDir + "image1/052640-20210901141310.jpg",
            dataDir + "image1/052640-20210901141317.jpg",
            dataDir + "image1/052640-20210901141321.jpg",
            dataDir + "image1/052640-20210901141324.jpg",
            dataDir + "image1/052640-20210901141327.jpg"
        };

        Rect2i smallROI{180, 100, 210, 70};

        RunBenchmark("Small (640x512)",
                     smallImages[0],
                     smallImages,
                     smallROI,
                     smallResult);
    }

    // ==========================================================================
    // Large Image Test (2048x4001) - in image2/
    // ==========================================================================
    TestResult largeResult;
    {
        PrintSeparator("Large Image Test (2048x4001)");

        std::vector<std::string> largeImages = {
            dataDir + "image2/20251201194802191.bmp",
            dataDir + "image2/20251201194804137.bmp",
            dataDir + "image2/20251201194805266.bmp",
            dataDir + "image2/20251201194806360.bmp",
            dataDir + "image2/2025120119482739.bmp"
        };

        // Check if large images exist
        QImage testImg = QImage::FromFile(largeImages[0]);
        if (!testImg.Empty()) {
            Rect2i largeROI{900, 1800, 135, 200};

            RunBenchmark("Large (2048x4001)",
                         largeImages[0],
                         largeImages,
                         largeROI,
                         largeResult);
        } else {
            std::cout << "\nLarge test images not found at: " << largeImages[0] << "\n";
            std::cout << "Skipping large image test.\n";
        }
    }

    // ==========================================================================
    // Summary Comparison
    // ==========================================================================
    PrintSeparator("Summary Comparison");

    auto printRow = [](const std::string& name, double small, double large) {
        std::cout << std::left << std::setw(25) << name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2) << small
                  << std::setw(12) << large << "\n";
    };

    std::cout << "\n" << std::left << std::setw(25) << "Metric"
              << std::right << std::setw(12) << "Small"
              << std::setw(12) << "Large" << "\n";
    std::cout << std::string(50, '-') << "\n";

    std::cout << "\n--- Create() ---\n";
    printRow("  Total (ms)", smallResult.createTiming.totalMs, largeResult.createTiming.totalMs);
    printRow("  PyramidBuild (ms)", smallResult.createTiming.pyramidBuildMs, largeResult.createTiming.pyramidBuildMs);
    printRow("  ContrastAuto (ms)", smallResult.createTiming.contrastAutoMs, largeResult.createTiming.contrastAutoMs);
    printRow("  ExtractPoints (ms)", smallResult.createTiming.extractPointsMs, largeResult.createTiming.extractPointsMs);
    printRow("  Optimize (ms)", smallResult.createTiming.optimizeMs, largeResult.createTiming.optimizeMs);
    printRow("  BuildSoA (ms)", smallResult.createTiming.buildSoAMs, largeResult.createTiming.buildSoAMs);

    std::cout << "\n--- Find() ---\n";
    printRow("  Total (ms)", smallResult.findTiming.totalMs, largeResult.findTiming.totalMs);
    printRow("  PyramidBuild (ms)", smallResult.findTiming.pyramidBuildMs, largeResult.findTiming.pyramidBuildMs);
    printRow("  CoarseSearch (ms)", smallResult.findTiming.coarseSearchMs, largeResult.findTiming.coarseSearchMs);
    printRow("  PyramidRefine (ms)", smallResult.findTiming.pyramidRefineMs, largeResult.findTiming.pyramidRefineMs);
    printRow("  SubpixelRefine (ms)", smallResult.findTiming.subpixelRefineMs, largeResult.findTiming.subpixelRefineMs);
    printRow("  NMS (ms)", smallResult.findTiming.nmsMs, largeResult.findTiming.nmsMs);

    std::cout << "\n--- AnglePyramid::Build (inside Find) ---\n";
    printRow("  Total (ms)", smallResult.findPyramidTiming.totalMs, largeResult.findPyramidTiming.totalMs);
    printRow("  GaussPyramid (ms)", smallResult.findPyramidTiming.gaussPyramidMs, largeResult.findPyramidTiming.gaussPyramidMs);
    printRow("  Sobel (ms)", smallResult.findPyramidTiming.sobelMs, largeResult.findPyramidTiming.sobelMs);
    printRow("  Sqrt/mag (ms)", smallResult.findPyramidTiming.sqrtMs, largeResult.findPyramidTiming.sqrtMs);
    printRow("  Atan2/dir (ms)", smallResult.findPyramidTiming.atan2Ms, largeResult.findPyramidTiming.atan2Ms);
    printRow("  Quantize (ms)", smallResult.findPyramidTiming.quantizeMs, largeResult.findPyramidTiming.quantizeMs);

    std::cout << "\n=== Benchmark Complete ===\n";

    return 0;
}
