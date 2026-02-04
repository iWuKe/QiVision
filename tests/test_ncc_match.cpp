/**
 * @file test_ncc_match.cpp
 * @brief Basic NCCModel matching tests
 */

#include <QiVision/Matching/NCCModel.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Core/Exception.h>

#include <iostream>
#include <cmath>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;

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
        std::cout << "  [PASS] " << message << " (" << a << " ~= " << b << ")\n";
        testsPassed++;
    } else {
        std::cout << "  [FAIL] " << message << " (" << a << " != " << b << ")\n";
        testsFailed++;
    }
}

QImage CreateTemplateImage() {
    QImage img(64, 64, PixelType::UInt8, ChannelType::Gray);
    for (int y = 0; y < 64; y++) {
        uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int x = 0; x < 64; x++) {
            bool isCross = (std::abs(x - 32) < 4) || (std::abs(y - 32) < 4);
            row[x] = isCross ? 220 : 40;
        }
    }
    return img;
}

QImage CreateSearchImage(const QImage& templ, int offsetX, int offsetY) {
    QImage img(128, 128, PixelType::UInt8, ChannelType::Gray);
    for (int y = 0; y < img.Height(); y++) {
        uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int x = 0; x < img.Width(); x++) {
            row[x] = 100;
        }
    }

    for (int y = 0; y < templ.Height(); y++) {
        const uint8_t* srcRow = static_cast<const uint8_t*>(templ.RowPtr(y));
        uint8_t* dstRow = static_cast<uint8_t*>(img.RowPtr(offsetY + y));
        for (int x = 0; x < templ.Width(); x++) {
            dstRow[offsetX + x] = srcRow[x];
        }
    }
    return img;
}

int main() {
    std::cout << "=== NCCModel Match Test ===\n\n";

    // 1) Create template and model
    QImage templ = CreateTemplateImage();
    NCCModel model;
    CreateNCCModel(
        templ,
        model,
        3,          // numLevels
        0.0,        // angleStart
        0.0,        // angleExtent (0 = all)
        0.0,        // angleStep (0 = auto)
        "use_polarity"
    );
    AssertTrue(model.IsValid(), "Model created");

    // 2) Basic match on synthetic image
    int offsetX = 30;
    int offsetY = 40;
    QImage search = CreateSearchImage(templ, offsetX, offsetY);

    std::vector<double> rows, cols, angles, scores;
    FindNCCModel(search, model,
                 0.0, DegToRad(360.0),
                 0.8, 1, 0.5,
                 "interpolation", 0,
                 rows, cols, angles, scores);

    AssertTrue(!rows.empty(), "Found match");
    if (!rows.empty()) {
        double expectedCol = offsetX + templ.Width() * 0.5;
        double expectedRow = offsetY + templ.Height() * 0.5;
        AssertNear(cols[0], expectedCol, 1.0, "Match X near expected");
        AssertNear(rows[0], expectedRow, 1.0, "Match Y near expected");
        AssertTrue(scores[0] > 0.8, "Score above threshold");
    }

    // 3) Empty search image should return no matches (no exception)
    QImage empty;
    rows.clear(); cols.clear(); angles.clear(); scores.clear();
    try {
        FindNCCModel(empty, model,
                     0.0, DegToRad(360.0),
                     0.8, 1, 0.5,
                     "interpolation", 0,
                     rows, cols, angles, scores);
        AssertTrue(rows.empty(), "Empty image -> no matches");
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Empty image threw: " << e.what() << "\n";
        testsFailed++;
    }

    std::cout << "\n=== Test Summary ===\n";
    std::cout << "Passed: " << testsPassed << "\n";
    std::cout << "Failed: " << testsFailed << "\n";
    return testsFailed > 0 ? 1 : 0;
}
