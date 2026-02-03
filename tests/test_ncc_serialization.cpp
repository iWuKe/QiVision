/**
 * @file test_ncc_serialization.cpp
 * @brief Test NCCModel serialization and deep copy functionality
 */

#include <QiVision/Matching/NCCModel.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Platform/FileIO.h>

#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::Color;

// Test result tracking
int testsPassed = 0;
int testsFailed = 0;

void AssertTrue(bool condition, const char* message) {
    if (condition) {
        std::cout << "  [PASS] " << message << "\n";
        testsPassed++;
    } else {
        std::cout << "  [FAIL] " << message << "\n";
        testsFailed++;
    }
}

void AssertNear(double a, double b, double eps, const char* message) {
    bool ok = std::abs(a - b) < eps;
    if (ok) {
        std::cout << "  [PASS] " << message << " (" << a << " == " << b << ")\n";
        testsPassed++;
    } else {
        std::cout << "  [FAIL] " << message << " (" << a << " != " << b << ")\n";
        testsFailed++;
    }
}

int main() {
    std::cout << "=== NCCModel Serialization & Deep Copy Test ===\n\n";

    // 1. Create a synthetic template image
    std::cout << "1. Creating synthetic template...\n";
    QImage templateImg(64, 64, PixelType::UInt8, ChannelType::Gray);
    {
        // Create a simple pattern: bright cross on dark background
        for (int y = 0; y < 64; y++) {
            uint8_t* row = static_cast<uint8_t*>(templateImg.RowPtr(y));
            for (int x = 0; x < 64; x++) {
                bool isCross = (std::abs(x - 32) < 5) || (std::abs(y - 32) < 5);
                row[x] = isCross ? 200 : 50;
            }
        }
    }
    std::cout << "  Template size: " << templateImg.Width() << "x" << templateImg.Height() << "\n";

    // 2. Create NCC model
    std::cout << "\n2. Creating NCCModel...\n";
    NCCModel model;
    CreateNCCModel(
        templateImg,
        model,
        3,                      // numLevels
        0.0,                    // angleStart
        DegToRad(180.0),        // angleExtent
        DegToRad(5.0),          // angleStep
        "use_polarity"
    );

    AssertTrue(model.IsValid(), "Model created successfully");

    // Get original model params
    int32_t origNumLevels;
    double origAngleStart, origAngleExtent, origAngleStep;
    std::string origMetric;
    GetNCCModelParams(model, origNumLevels, origAngleStart, origAngleExtent, origAngleStep, origMetric);

    double origOriginRow, origOriginCol;
    GetNCCModelOrigin(model, origOriginRow, origOriginCol);

    int32_t origWidth, origHeight;
    GetNCCModelSize(model, origWidth, origHeight);

    std::cout << "  Original params: levels=" << origNumLevels
              << ", angleStart=" << RadToDeg(origAngleStart)
              << ", angleExtent=" << RadToDeg(origAngleExtent)
              << ", metric=" << origMetric << "\n";
    std::cout << "  Original origin: (" << origOriginCol << ", " << origOriginRow << ")\n";
    std::cout << "  Original size: " << origWidth << "x" << origHeight << "\n";

    // 3. Test Deep Copy
    std::cout << "\n3. Testing Deep Copy...\n";
    {
        // Copy constructor
        NCCModel copiedModel(model);
        AssertTrue(copiedModel.IsValid(), "Copy constructor: model valid");

        int32_t copyNumLevels;
        double copyAngleStart, copyAngleExtent, copyAngleStep;
        std::string copyMetric;
        GetNCCModelParams(copiedModel, copyNumLevels, copyAngleStart, copyAngleExtent, copyAngleStep, copyMetric);

        AssertTrue(copyNumLevels == origNumLevels, "Copy constructor: numLevels matches");
        AssertNear(copyAngleStart, origAngleStart, 1e-10, "Copy constructor: angleStart matches");
        AssertNear(copyAngleExtent, origAngleExtent, 1e-10, "Copy constructor: angleExtent matches");
        AssertTrue(copyMetric == origMetric, "Copy constructor: metric matches");

        double copyOriginRow, copyOriginCol;
        GetNCCModelOrigin(copiedModel, copyOriginRow, copyOriginCol);
        AssertNear(copyOriginRow, origOriginRow, 1e-10, "Copy constructor: origin.row matches");
        AssertNear(copyOriginCol, origOriginCol, 1e-10, "Copy constructor: origin.col matches");

        // Modify original, verify copy is independent
        SetNCCModelOrigin(model, 10.0, 20.0);
        double modOriginRow, modOriginCol;
        GetNCCModelOrigin(model, modOriginRow, modOriginCol);
        GetNCCModelOrigin(copiedModel, copyOriginRow, copyOriginCol);
        AssertNear(modOriginRow, 10.0, 1e-10, "Original modified to row=10");
        AssertNear(copyOriginRow, origOriginRow, 1e-10, "Copy remains independent");

        // Restore original
        SetNCCModelOrigin(model, origOriginRow, origOriginCol);

        // Assignment operator
        NCCModel assignedModel;
        assignedModel = model;
        AssertTrue(assignedModel.IsValid(), "Assignment operator: model valid");

        int32_t assignNumLevels;
        double assignAngleStart, assignAngleExtent, assignAngleStep;
        std::string assignMetric;
        GetNCCModelParams(assignedModel, assignNumLevels, assignAngleStart, assignAngleExtent, assignAngleStep, assignMetric);
        AssertTrue(assignNumLevels == origNumLevels, "Assignment operator: numLevels matches");
    }

    // 4. Test Serialization (Write/Read)
    std::cout << "\n4. Testing Serialization (Write/Read)...\n";
    {
        std::string testFile = "/tmp/test_ncc_model.qincc";

        // Write model
        std::cout << "  Writing model to " << testFile << "...\n";
        WriteNCCModel(model, testFile);

        AssertTrue(Platform::FileExists(testFile), "Model file created");
        int64_t fileSize = Platform::GetFileSize(testFile);
        std::cout << "  File size: " << fileSize << " bytes\n";
        AssertTrue(fileSize > 100, "File has reasonable size");

        // Read model
        std::cout << "  Reading model from " << testFile << "...\n";
        NCCModel loadedModel;
        ReadNCCModel(testFile, loadedModel);

        AssertTrue(loadedModel.IsValid(), "Loaded model is valid");

        // Verify params
        int32_t loadNumLevels;
        double loadAngleStart, loadAngleExtent, loadAngleStep;
        std::string loadMetric;
        GetNCCModelParams(loadedModel, loadNumLevels, loadAngleStart, loadAngleExtent, loadAngleStep, loadMetric);

        AssertTrue(loadNumLevels == origNumLevels, "Loaded numLevels matches");
        AssertNear(loadAngleStart, origAngleStart, 1e-10, "Loaded angleStart matches");
        AssertNear(loadAngleExtent, origAngleExtent, 1e-10, "Loaded angleExtent matches");
        AssertTrue(loadMetric == origMetric, "Loaded metric matches");

        double loadOriginRow, loadOriginCol;
        GetNCCModelOrigin(loadedModel, loadOriginRow, loadOriginCol);
        AssertNear(loadOriginRow, origOriginRow, 1e-10, "Loaded origin.row matches");
        AssertNear(loadOriginCol, origOriginCol, 1e-10, "Loaded origin.col matches");

        int32_t loadWidth, loadHeight;
        GetNCCModelSize(loadedModel, loadWidth, loadHeight);
        AssertTrue(loadWidth == origWidth, "Loaded width matches");
        AssertTrue(loadHeight == origHeight, "Loaded height matches");

        // Clean up
        Platform::DeleteFile(testFile);
    }

    // 5. Test loaded model can Find matches
    std::cout << "\n5. Testing Loaded Model Matching...\n";
    {
        // Create search image with the template
        QImage searchImg(128, 128, PixelType::UInt8, ChannelType::Gray);
        {
            // Fill with gray background
            for (int y = 0; y < 128; y++) {
                uint8_t* row = static_cast<uint8_t*>(searchImg.RowPtr(y));
                for (int x = 0; x < 128; x++) {
                    row[x] = 100;
                }
            }
            // Copy template at offset (40, 40)
            for (int y = 0; y < 64; y++) {
                uint8_t* srcRow = static_cast<uint8_t*>(templateImg.RowPtr(y));
                uint8_t* dstRow = static_cast<uint8_t*>(searchImg.RowPtr(y + 40));
                for (int x = 0; x < 64; x++) {
                    dstRow[x + 40] = srcRow[x];
                }
            }
        }

        // Save and reload model
        std::string testFile = "/tmp/test_ncc_model_match.qincc";
        WriteNCCModel(model, testFile);

        NCCModel loadedModel;
        ReadNCCModel(testFile, loadedModel);

        // Find with original model
        std::vector<double> origRows, origCols, origAngles, origScores;
        FindNCCModel(searchImg, model, 0.0, DegToRad(360.0), 0.8, 1, 0.5,
                     "interpolation", 0, origRows, origCols, origAngles, origScores);

        // Find with loaded model
        std::vector<double> loadRows, loadCols, loadAngles, loadScores;
        FindNCCModel(searchImg, loadedModel, 0.0, DegToRad(360.0), 0.8, 1, 0.5,
                     "interpolation", 0, loadRows, loadCols, loadAngles, loadScores);

        AssertTrue(!origRows.empty(), "Original model found match");
        AssertTrue(!loadRows.empty(), "Loaded model found match");

        if (!origRows.empty() && !loadRows.empty()) {
            std::cout << "  Original: pos=(" << origCols[0] << ", " << origRows[0]
                      << "), score=" << origScores[0] << "\n";
            std::cout << "  Loaded:   pos=(" << loadCols[0] << ", " << loadRows[0]
                      << "), score=" << loadScores[0] << "\n";

            AssertNear(origCols[0], loadCols[0], 0.5, "Position X matches");
            AssertNear(origRows[0], loadRows[0], 0.5, "Position Y matches");
            AssertNear(origScores[0], loadScores[0], 0.01, "Score matches");
        }

        // Clean up
        Platform::DeleteFile(testFile);
    }

    // Summary
    std::cout << "\n=== Test Summary ===\n";
    std::cout << "Passed: " << testsPassed << "\n";
    std::cout << "Failed: " << testsFailed << "\n";

    return testsFailed > 0 ? 1 : 0;
}
