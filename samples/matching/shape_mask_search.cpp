/**
 * @file shape_mask_search.cpp
 * @brief Test FindShapeModel with search mask (decompiled find_shape_model_2)
 *
 * Validates:
 *   1. No mask → normal matching (baseline)
 *   2. Full mask (all 255) → same results as no mask
 *   3. Mask covering target → target not found
 *   4. Mask excluding target → target still found
 *
 * Template: tests/data/matching/image5/m15.bmp
 * Search:   tests/data/matching/image5/rings_01.png
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Matching/ShapeModel.h>
#include <QiVision/IO/ImageIO.h>

#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;
using namespace Qi::Vision::IO;

// Create a uint8 mask image filled with a value
static QImage CreateMask(int32_t w, int32_t h, uint8_t fillValue) {
    QImage mask(w, h, PixelType::UInt8);
    uint8_t* data = static_cast<uint8_t*>(mask.Data());
    const int32_t stride = mask.Stride();
    for (int32_t y = 0; y < h; ++y) {
        std::memset(data + y * stride, fillValue, w);
    }
    return mask;
}

// Set a rectangular region in mask to a value
static void SetMaskRect(QImage& mask, int32_t x0, int32_t y0,
                         int32_t x1, int32_t y1, uint8_t value) {
    uint8_t* data = static_cast<uint8_t*>(mask.Data());
    const int32_t stride = mask.Stride();
    const int32_t w = mask.Width();
    const int32_t h = mask.Height();
    x0 = std::max(0, std::min(x0, w - 1));
    x1 = std::max(0, std::min(x1, w - 1));
    y0 = std::max(0, std::min(y0, h - 1));
    y1 = std::max(0, std::min(y1, h - 1));
    for (int32_t y = y0; y <= y1; ++y) {
        std::memset(data + y * stride + x0, value, x1 - x0 + 1);
    }
}

int main()
{
    std::printf("========================================\n");
    std::printf("  FindShapeModel Mask Test\n");
    std::printf("========================================\n\n");

    // Load template and search image
    const std::string dataFolder = "tests/data/matching/image5";
    QImage templateGray, searchGray;
    ReadImageGray(dataFolder + "/m15.bmp", templateGray);
    ReadImageGray(dataFolder + "/rings_01.png", searchGray);
    if (templateGray.Empty() || searchGray.Empty()) {
        std::fprintf(stderr, "ERROR: Failed to load test images\n");
        return 1;
    }
    std::printf("Template: %dx%d\n", templateGray.Width(), templateGray.Height());
    std::printf("Search:   %dx%d\n\n", searchGray.Width(), searchGray.Height());

    // Create model
    ShapeModel model;
    CreateShapeModel(templateGray, model,
                     4, 0, RAD(360), 0, "auto",
                     "use_polarity", "auto", 10);
    if (!model.IsValid()) {
        std::fprintf(stderr, "ERROR: Model creation failed\n");
        return 1;
    }
    std::printf("Model created.\n\n");

    // Search parameters
    const double minScore = 0.5;
    const int32_t maxMatches = 4;
    const double maxOverlap = 0.5;
    const double greediness = 0.7;

    std::vector<double> rows, cols, angles, scores;
    int passed = 0, total = 0;

    // -----------------------------------------------------------------
    // Test 1: No mask (baseline)
    // -----------------------------------------------------------------
    ++total;
    FindShapeModel(searchGray, model, 0, RAD(360),
                   minScore, maxMatches, maxOverlap, "none", 0, greediness,
                   rows, cols, angles, scores);
    std::printf("Test 1 - No mask:           %zu matches", scores.size());
    if (!scores.empty()) {
        std::printf(" (best=%.3f at row=%.1f col=%.1f)", scores[0], rows[0], cols[0]);
        ++passed;
        std::printf(" PASS\n");
    } else {
        std::printf(" FAIL (expected matches)\n");
    }

    // Save baseline result for comparison
    const size_t baselineCount = scores.size();
    const double baselineRow = scores.empty() ? 0 : rows[0];
    const double baselineCol = scores.empty() ? 0 : cols[0];
    const double baselineScore = scores.empty() ? 0 : scores[0];

    // -----------------------------------------------------------------
    // Test 2: Full mask (all 255) → same as no mask
    // -----------------------------------------------------------------
    ++total;
    QImage fullMask = CreateMask(searchGray.Width(), searchGray.Height(), 255);
    FindShapeModel(searchGray, model, 0, RAD(360),
                   minScore, maxMatches, maxOverlap, "none", 0, greediness,
                   rows, cols, angles, scores, fullMask);
    std::printf("Test 2 - Full mask (255):   %zu matches", scores.size());
    if (scores.size() == baselineCount && !scores.empty() &&
        std::abs(scores[0] - baselineScore) < 0.05) {
        ++passed;
        std::printf(" (score=%.3f) PASS\n", scores[0]);
    } else {
        std::printf(" FAIL (expected same as baseline)\n");
    }

    // -----------------------------------------------------------------
    // Test 3: Mask blocks target area → should NOT find
    // -----------------------------------------------------------------
    ++total;
    {
        // Create mask with 0 in target area (block matching there)
        QImage blockMask = CreateMask(searchGray.Width(), searchGray.Height(), 255);
        // Block a generous area around the found target
        int32_t cx = static_cast<int32_t>(baselineCol);
        int32_t cy = static_cast<int32_t>(baselineRow);
        int32_t margin = 80;  // generous margin
        SetMaskRect(blockMask, cx - margin, cy - margin,
                    cx + margin, cy + margin, 0);

        FindShapeModel(searchGray, model, 0, RAD(360),
                       minScore, maxMatches, maxOverlap, "none", 0, greediness,
                       rows, cols, angles, scores, blockMask);

        // Check: the target at (baselineRow, baselineCol) should not be found
        bool targetBlocked = true;
        for (size_t i = 0; i < scores.size(); ++i) {
            double dist = std::sqrt((rows[i] - baselineRow) * (rows[i] - baselineRow) +
                                    (cols[i] - baselineCol) * (cols[i] - baselineCol));
            if (dist < 30.0) {
                targetBlocked = false;
                break;
            }
        }
        std::printf("Test 3 - Block target:      %zu matches", scores.size());
        if (targetBlocked) {
            ++passed;
            std::printf(" (target blocked) PASS\n");
        } else {
            std::printf(" FAIL (target should be blocked)\n");
        }
    }

    // -----------------------------------------------------------------
    // Test 4: Mask allows only target area → should find it
    // -----------------------------------------------------------------
    ++total;
    {
        // Create mask with 0 everywhere, 255 only around target
        QImage allowMask = CreateMask(searchGray.Width(), searchGray.Height(), 0);
        int32_t cx = static_cast<int32_t>(baselineCol);
        int32_t cy = static_cast<int32_t>(baselineRow);
        int32_t margin = 80;
        SetMaskRect(allowMask, cx - margin, cy - margin,
                    cx + margin, cy + margin, 255);

        FindShapeModel(searchGray, model, 0, RAD(360),
                       minScore, maxMatches, maxOverlap, "none", 0, greediness,
                       rows, cols, angles, scores, allowMask);

        bool targetFound = false;
        for (size_t i = 0; i < scores.size(); ++i) {
            double dist = std::sqrt((rows[i] - baselineRow) * (rows[i] - baselineRow) +
                                    (cols[i] - baselineCol) * (cols[i] - baselineCol));
            if (dist < 30.0) {
                targetFound = true;
                break;
            }
        }
        std::printf("Test 4 - Allow only target: %zu matches", scores.size());
        if (targetFound) {
            ++passed;
            std::printf(" (target found) PASS\n");
        } else {
            std::printf(" FAIL (target should be found)\n");
        }
    }

    // -----------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------
    std::printf("\n========================================\n");
    std::printf("  Result: %d/%d passed\n", passed, total);
    std::printf("========================================\n");

    return (passed == total) ? 0 : 1;
}
