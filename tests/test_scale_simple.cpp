/**
 * @file test_scale_simple.cpp
 * @brief Simple test to diagnose scaled matching issues
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Matching/ShapeModel.h>
#include <QiVision/Transform/AffineTransform.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Filter/Filter.h>
#include <cstdio>
#include <cmath>
#include <cstring>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::Transform;

int main() {
    std::printf("=== Simple Scale Matching Test ===\n\n");

    // Load template image
    QImage templateImg;
    ReadImageGray("tests/data/halcon_images/rings/mixed_01.png", templateImg);
    if (templateImg.Empty()) {
        std::printf("Failed to load image!\n");
        return 1;
    }
    std::printf("Template image: %dx%d\n", templateImg.Width(), templateImg.Height());

    // Define ROI (same as test program)
    Rect2i roi{367, 213, 89, 87};
    QRegion roiRegion = QRegion::Rectangle(roi.x, roi.y, roi.width, roi.height);

    // Extract template region
    QImage templateRegion = templateImg.SubImage(roi.x, roi.y, roi.width, roi.height);
    std::printf("Template region: %dx%d\n", templateRegion.Width(), templateRegion.Height());

    // Create shape model (NOT scaled - regular model)
    ShapeModel model;
    SetShapeModelDebugCreateGlobal(true);

    QImage templateSmooth;
    Filter::GaussFilter(templateImg, templateSmooth, 0.7);

    CreateShapeModel(
        templateSmooth, roiRegion, model,
        4,                      // numLevels
        0, RAD(360), RAD(5),    // angle
        "point_reduction_high",
        "use_polarity",
        "auto_contrast_hyst", 10.0
    );

    if (!model.IsValid()) {
        std::printf("Failed to create model!\n");
        return 1;
    }
    std::printf("\nModel created successfully.\n\n");

    // Test 1: Search in original image with scale=1.0 target
    {
        std::printf("--- Test 1: Search in original image (should find at template location) ---\n");
        std::vector<double> rows, cols, angles, scores;
        FindShapeModel(
            templateSmooth, model,
            0, RAD(360),
            0.5, 1, 0.5,
            "least_squares", 0, 0.8,
            rows, cols, angles, scores
        );

        std::printf("Found %zu matches\n", rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            std::printf("  Match %zu: pos=(%.1f, %.1f) angle=%.2f° score=%.4f\n",
                        i, cols[i], rows[i], angles[i] * 180.0 / M_PI, scores[i]);
        }
        std::printf("\n");
    }

    // Test 2: Create synthetic search image with scale=1.0 target
    {
        std::printf("--- Test 2: Synthetic image with scale=1.0 target ---\n");

        QImage searchImg(templateImg.Width(), templateImg.Height(), PixelType::UInt8);
        uint8_t* data = static_cast<uint8_t*>(searchImg.Data());
        std::memset(data, 128, searchImg.Height() * searchImg.Stride());

        int placeX = (searchImg.Width() - roi.width) / 2;
        int placeY = (searchImg.Height() - roi.height) / 2;

        for (int y = 0; y < roi.height && placeY + y < searchImg.Height(); ++y) {
            const uint8_t* srcRow = static_cast<const uint8_t*>(templateRegion.RowPtr(y));
            uint8_t* dstRow = static_cast<uint8_t*>(searchImg.RowPtr(placeY + y));
            for (int x = 0; x < roi.width && placeX + x < searchImg.Width(); ++x) {
                dstRow[placeX + x] = srcRow[x];
            }
        }

        // Expected position (center of template)
        double expectedX = placeX + roi.width / 2.0;
        double expectedY = placeY + roi.height / 2.0;

        // Search with non-scaled FindShapeModel
        std::vector<double> rows, cols, angles, scores;
        FindShapeModel(
            searchImg, model,
            0, RAD(360),
            0.5, 1, 0.5,
            "least_squares", 0, 0.8,
            rows, cols, angles, scores
        );

        std::printf("Expected position: (%.1f, %.1f)\n", expectedX, expectedY);
        std::printf("Found %zu matches\n", rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            std::printf("  Match %zu: pos=(%.1f, %.1f) angle=%.2f° score=%.4f\n",
                        i, cols[i], rows[i], angles[i] * 180.0 / M_PI, scores[i]);
        }
        std::printf("\n");
    }

    // Test 3: Create synthetic search image with scale=0.8 target and search with runtime scale
    {
        std::printf("--- Test 3: Synthetic image with scale=0.8 target (runtime scale search) ---\n");

        double targetScale = 0.8;
        int scaledW = static_cast<int>(roi.width * targetScale);
        int scaledH = static_cast<int>(roi.height * targetScale);

        QImage scaledTemplate;
        ScaleImage(templateRegion, scaledTemplate, targetScale, targetScale, "bilinear");

        QImage searchImg(templateImg.Width(), templateImg.Height(), PixelType::UInt8);
        uint8_t* data = static_cast<uint8_t*>(searchImg.Data());
        std::memset(data, 128, searchImg.Height() * searchImg.Stride());

        int placeX = (searchImg.Width() - scaledW) / 2;
        int placeY = (searchImg.Height() - scaledH) / 2;

        for (int y = 0; y < scaledH && placeY + y < searchImg.Height(); ++y) {
            const uint8_t* srcRow = static_cast<const uint8_t*>(scaledTemplate.RowPtr(y));
            uint8_t* dstRow = static_cast<uint8_t*>(searchImg.RowPtr(placeY + y));
            for (int x = 0; x < scaledW && placeX + x < searchImg.Width(); ++x) {
                dstRow[placeX + x] = srcRow[x];
            }
        }

        double expectedX = placeX + scaledW / 2.0;
        double expectedY = placeY + scaledH / 2.0;

        // Search with FindScaledShapeModel - focus on scale=0.8
        std::vector<double> rows, cols, angles, scales, scores;
        FindScaledShapeModel(
            searchImg, model,
            0, RAD(360),        // angle range
            0.7, 0.9,           // narrow scale range around 0.8
            0.3, 5, 0.5,        // lower minScore, more matches
            "least_squares", 0, 0.5,  // lower greediness
            rows, cols, angles, scales, scores
        );

        std::printf("Expected position: (%.1f, %.1f) scale: %.2f\n", expectedX, expectedY, targetScale);
        std::printf("Found %zu matches\n", rows.size());
        for (size_t i = 0; i < std::min(size_t(5), rows.size()); ++i) {
            double posErr = std::sqrt(std::pow(cols[i] - expectedX, 2) + std::pow(rows[i] - expectedY, 2));
            std::printf("  Match %zu: pos=(%.1f, %.1f) scale=%.3f angle=%.2f° score=%.4f posErr=%.1f\n",
                        i, cols[i], rows[i], scales[i], angles[i] * 180.0 / M_PI, scores[i], posErr);
        }
        std::printf("\n");
    }

    // Test 4: Direct score computation at known location
    {
        std::printf("--- Test 4: Direct score computation for scale=0.8 target ---\n");

        double targetScale = 0.8;
        int scaledW = static_cast<int>(roi.width * targetScale);
        int scaledH = static_cast<int>(roi.height * targetScale);

        QImage scaledTemplate;
        ScaleImage(templateRegion, scaledTemplate, targetScale, targetScale, "bilinear");

        QImage searchImg(templateImg.Width(), templateImg.Height(), PixelType::UInt8);
        uint8_t* data = static_cast<uint8_t*>(searchImg.Data());
        std::memset(data, 128, searchImg.Height() * searchImg.Stride());

        int placeX = (searchImg.Width() - scaledW) / 2;
        int placeY = (searchImg.Height() - scaledH) / 2;

        for (int y = 0; y < scaledH && placeY + y < searchImg.Height(); ++y) {
            const uint8_t* srcRow = static_cast<const uint8_t*>(scaledTemplate.RowPtr(y));
            uint8_t* dstRow = static_cast<uint8_t*>(searchImg.RowPtr(placeY + y));
            for (int x = 0; x < scaledW && placeX + x < searchImg.Width(); ++x) {
                dstRow[placeX + x] = srcRow[x];
            }
        }

        double expectedX = placeX + scaledW / 2.0;
        double expectedY = placeY + scaledH / 2.0;

        // Try FindShapeModel with scale=1.0 - this should fail because target is scaled
        std::printf("Test 4a: FindShapeModel (scale=1.0) on scale=0.8 target:\n");
        {
            std::vector<double> rows, cols, angles, scores;
            FindShapeModel(searchImg, model, 0, RAD(360), 0.3, 5, 0.5, "least_squares", 0, 0.5,
                           rows, cols, angles, scores);
            std::printf("  Found %zu matches (expected 0 or low-score match)\n", rows.size());
            for (size_t i = 0; i < std::min(size_t(3), rows.size()); ++i) {
                std::printf("    Match %zu: pos=(%.1f, %.1f) score=%.4f\n", i, cols[i], rows[i], scores[i]);
            }
        }

        // Try very narrow scale range around 0.8
        std::printf("Test 4b: FindScaledShapeModel with scale=[0.79, 0.81]:\n");
        {
            std::vector<double> rows, cols, angles, scales, scores;
            FindScaledShapeModel(searchImg, model, 0, RAD(360), 0.79, 0.81, 0.3, 5, 0.5,
                                 "least_squares", 0, 0.5, rows, cols, angles, scales, scores);
            std::printf("  Expected position: (%.1f, %.1f) scale: %.2f\n", expectedX, expectedY, targetScale);
            std::printf("  Found %zu matches\n", rows.size());
            for (size_t i = 0; i < std::min(size_t(3), rows.size()); ++i) {
                std::printf("    Match %zu: pos=(%.1f, %.1f) scale=%.3f score=%.4f\n",
                            i, cols[i], rows[i], scales[i], scores[i]);
            }
        }
        std::printf("\n");
    }

    std::printf("=== Test Complete ===\n");
    return 0;
}
