/**
 * @file test_intensity_stats.cpp
 * @brief Test ROI intensity statistics
 */

#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>

#include <iostream>
#include <cmath>

using namespace Qi::Vision;
using namespace Qi::Vision::Color;

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

int main() {
    std::cout << "=== IntensityStats ROI Test ===\n\n";

    // 4x4 image with values 0..15
    QImage img(4, 4, PixelType::UInt8, ChannelType::Gray);
    for (int y = 0; y < 4; ++y) {
        uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int x = 0; x < 4; ++x) {
            row[x] = static_cast<uint8_t>(y * 4 + x);
        }
    }

    // ROI: 2x2 block at (1,1) -> values {5,6,9,10}
    QRegion roi = QRegion::Rectangle(1, 1, 2, 2);

    double minGray = 0.0, maxGray = 0.0, mean = 0.0, stddev = 0.0;
    IntensityStats(img, roi, minGray, maxGray, mean, stddev);

    AssertNear(minGray, 5.0, 1e-6, "MinGray in ROI");
    AssertNear(maxGray, 10.0, 1e-6, "MaxGray in ROI");
    AssertNear(mean, 7.5, 1e-6, "Mean in ROI");

    // Variance of {5,6,9,10} = 4.25, stddev ~ 2.061553
    AssertNear(stddev, std::sqrt(4.25), 1e-6, "StdDev in ROI");

    // MeanGray/StdDevGray wrappers
    AssertNear(MeanGray(img, roi), 7.5, 1e-6, "MeanGray wrapper");
    AssertNear(StdDevGray(img, roi), std::sqrt(4.25), 1e-6, "StdDevGray wrapper");

    // Empty region should return zeros
    QRegion empty;
    IntensityStats(img, empty, minGray, maxGray, mean, stddev);
    AssertNear(minGray, 0.0, 1e-6, "Empty region min=0");
    AssertNear(maxGray, 0.0, 1e-6, "Empty region max=0");
    AssertNear(mean, 0.0, 1e-6, "Empty region mean=0");
    AssertNear(stddev, 0.0, 1e-6, "Empty region stddev=0");

    std::cout << "\n=== Test Summary ===\n";
    std::cout << "Passed: " << testsPassed << "\n";
    std::cout << "Failed: " << testsFailed << "\n";
    return testsFailed > 0 ? 1 : 0;
}
