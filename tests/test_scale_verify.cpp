/**
 * @file test_scale_verify.cpp
 * @brief Verify scaled shape matching with known scale targets
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
    std::printf("=== Scale Matching Verification Test ===\n\n");

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

    // Create scaled shape model
    ShapeModel model;
    SetShapeModelDebugCreateGlobal(true);

    QImage templateSmooth;
    Filter::GaussFilter(templateImg, templateSmooth, 0.7);

    CreateScaledShapeModel(
        templateSmooth, roiRegion, model,
        4,                      // numLevels
        0, RAD(360), RAD(5),    // angle
        0.5, 1.5, 0.1,          // scale range [0.5, 1.5], step=0.1
        "point_reduction_high",
        "use_polarity",
        "auto_contrast_hyst", 10.0
    );

    if (!model.IsValid()) {
        std::printf("Failed to create model!\n");
        return 1;
    }
    std::printf("\nModel created successfully.\n\n");

    // Test different scales
    double testScales[] = {0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};

    for (double targetScale : testScales) {
        std::printf("--- Testing target scale = %.2f ---\n", targetScale);

        // Create search image with scaled target
        // 1. Extract template region
        // 2. Scale it
        // 3. Place in a blank image

        int scaledW = static_cast<int>(roi.width * targetScale);
        int scaledH = static_cast<int>(roi.height * targetScale);

        // Scale the template region
        QImage scaledTemplate;
        ScaleImage(templateRegion, scaledTemplate, targetScale, targetScale, "bilinear");

        // Create search image (same size as original)
        QImage searchImg(templateImg.Width(), templateImg.Height(), PixelType::UInt8);
        // Fill with gray background
        uint8_t* data = static_cast<uint8_t*>(searchImg.Data());
        std::memset(data, 128, searchImg.Height() * searchImg.Stride());

        // Place scaled template at center of search image
        int placeX = (searchImg.Width() - scaledW) / 2;
        int placeY = (searchImg.Height() - scaledH) / 2;

        // Copy scaled template to search image
        for (int y = 0; y < scaledH && placeY + y < searchImg.Height(); ++y) {
            const uint8_t* srcRow = static_cast<const uint8_t*>(scaledTemplate.RowPtr(y));
            uint8_t* dstRow = static_cast<uint8_t*>(searchImg.RowPtr(placeY + y));
            for (int x = 0; x < scaledW && placeX + x < searchImg.Width(); ++x) {
                dstRow[placeX + x] = srcRow[x];
            }
        }

        // Search
        std::vector<double> rows, cols, angles, scales, scores;
        FindScaledShapeModel(
            searchImg, model,
            0, RAD(360),        // angle range
            0.5, 1.5,           // scale range
            0.5, 1, 0.5,        // minScore, numMatches, maxOverlap
            "least_squares", 0, 0.8,
            rows, cols, angles, scales, scores
        );

        // Expected position
        double expectedX = placeX + scaledW / 2.0;
        double expectedY = placeY + scaledH / 2.0;

        if (!rows.empty()) {
            double posErr = std::sqrt(std::pow(cols[0] - expectedX, 2) + std::pow(rows[0] - expectedY, 2));
            double scaleErr = std::abs(scales[0] - targetScale);
            std::printf("  FOUND: pos=(%.1f, %.1f) expected=(%.1f, %.1f) err=%.2f\n",
                        cols[0], rows[0], expectedX, expectedY, posErr);
            std::printf("         scale=%.3f expected=%.3f err=%.3f, score=%.4f\n",
                        scales[0], targetScale, scaleErr, scores[0]);
            if (scaleErr > 0.15) {
                std::printf("  WARNING: Scale mismatch!\n");
            }
        } else {
            std::printf("  NOT FOUND! (expected at %.1f, %.1f with scale %.2f)\n",
                        expectedX, expectedY, targetScale);
        }
        std::printf("\n");
    }

    std::printf("=== Test Complete ===\n");
    return 0;
}
