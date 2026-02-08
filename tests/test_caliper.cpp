/**
 * @file test_caliper.cpp
 * @brief Basic Caliper/CaliperArray tests
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Measure/Caliper.h>
#include <QiVision/Measure/MeasureHandle.h>
#include <QiVision/Measure/CaliperArray.h>

#include <cmath>
#include <iostream>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Measure;

namespace {

constexpr double kPi = 3.14159265358979323846;

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
    bool ok = std::abs(a - b) <= eps;
    if (ok) {
        std::cout << "  [PASS] " << message << " (" << a << " ~= " << b << ")\n";
        testsPassed++;
    } else {
        std::cout << "  [FAIL] " << message << " (" << a << " != " << b << ")\n";
        testsFailed++;
    }
}

QImage CreateStepImage(int width, int height, int edgeX, uint8_t leftVal, uint8_t rightVal) {
    QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
    for (int y = 0; y < height; ++y) {
        uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int x = 0; x < width; ++x) {
            row[x] = (x < edgeX) ? leftVal : rightVal;
        }
    }
    return img;
}

QImage CreateBarImage(int width, int height, int leftX, int rightX, uint8_t bgVal, uint8_t barVal) {
    QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
    for (int y = 0; y < height; ++y) {
        uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int x = 0; x < width; ++x) {
            row[x] = (x >= leftX && x <= rightX) ? barVal : bgVal;
        }
    }
    return img;
}

void TestMeasurePosOnStepEdge() {
    std::cout << "\n[Test] MeasurePos on step edge\n";

    QImage img = CreateStepImage(200, 120, 100, 30, 220);
    MeasureRectangle2 handle = GenMeasureRectangle2(
        60.0, 100.0,  // row, col
        -kPi * 0.5,   // phi: profile horizontal
        40.0, 6.0     // length1, length2
    );

    auto edges = MeasurePos(img, handle, 1.0, 20.0, "all", "strongest");

    AssertTrue(!edges.empty(), "Found at least one edge");
    if (!edges.empty()) {
        AssertNear(edges[0].column, 100.0, 2.0, "Edge X near expected");
        AssertNear(edges[0].row, 60.0, 1.0, "Edge Y near center row");
        AssertTrue(edges[0].amplitude > 20.0, "Edge amplitude above threshold");
    }
}

void TestMeasurePairsOnBar() {
    std::cout << "\n[Test] MeasurePairs on bright bar\n";

    QImage img = CreateBarImage(240, 120, 70, 130, 25, 230);
    MeasureRectangle2 handle = GenMeasureRectangle2(60.0, 100.0, -kPi * 0.5, 80.0, 6.0);

    auto pairs = MeasurePairs(img, handle, 1.0, 20.0, "all", "strongest");

    AssertTrue(!pairs.empty(), "Found at least one edge pair");
    if (!pairs.empty()) {
        const PairResult& p = pairs[0];
        AssertNear(p.intraDistance, 60.0, 3.0, "Pair width near expected");
        AssertTrue(p.first.column < p.second.column, "Pair edge order is left-to-right");
    }
}

void TestInvalidSelectThrows() {
    std::cout << "\n[Test] Invalid select string throws\n";

    QImage img = CreateStepImage(120, 80, 50, 20, 220);
    MeasureRectangle2 handle = GenMeasureRectangle2(40.0, 50.0, 0.0, 25.0, 4.0);

    bool threw = false;
    try {
        (void)MeasurePos(img, handle, 1.0, 20.0, "all", "bad_select_mode");
    } catch (const InvalidArgumentException&) {
        threw = true;
    }
    AssertTrue(threw, "Invalid select triggers InvalidArgumentException");
}

void TestCaliperArrayMeasurePos() {
    std::cout << "\n[Test] CaliperArray MeasurePos + stats\n";

    QImage img = CreateStepImage(220, 140, 100, 20, 220);

    CaliperArray array;
    bool ok = array.CreateAlongLine(
        Point2d{100.0, 20.0},
        Point2d{100.0, 120.0},
        8,
        30.0,
        5.0
    );
    AssertTrue(ok, "CreateAlongLine succeeded");
    AssertTrue(array.IsValid(), "CaliperArray is valid");

    CaliperArrayStats stats;
    CaliperArrayResult result = array.MeasurePos(img, 1.0, 20.0, "all", "strongest", &stats);

    AssertTrue(result.numCalipers == 8, "Caliper count matches request");
    AssertTrue(result.numValid >= 6, "Most calipers found valid edge");
    AssertTrue(stats.totalEdgesFound >= result.numValid, "Stats totalEdgesFound is reasonable");
    AssertTrue(stats.measurementTime >= 0.0, "Stats measurementTime populated");
    AssertTrue(stats.avgTimePerCaliper >= 0.0, "Stats avgTimePerCaliper populated");

    if (!result.firstEdgePoints.empty()) {
        double meanX = 0.0;
        for (const auto& p : result.firstEdgePoints) {
            meanX += p.x;
        }
        meanX /= static_cast<double>(result.firstEdgePoints.size());
        AssertNear(meanX, 100.0, 2.0, "Array edge X mean near step edge");
    }
}

void TestCaliperArrayMeasurePairsFieldConsistency() {
    std::cout << "\n[Test] CaliperArray MeasurePairs field consistency\n";

    QImage img = CreateBarImage(260, 140, 80, 160, 20, 220);

    CaliperArray array;
    bool ok = array.CreateAlongLine(
        Point2d{120.0, 20.0},
        Point2d{120.0, 120.0},
        6,
        50.0,
        5.0
    );
    AssertTrue(ok, "CreateAlongLine for pairs succeeded");

    CaliperArrayResult result = array.MeasurePairs(img, 1.0, 20.0, "positive", "first");
    AssertTrue(result.numValid >= 4, "Array pairs found on most calipers");

    size_t n = std::min(result.firstEdgePoints.size(), result.firstPairEdges.size());
    AssertTrue(n > 0, "Pair result arrays are non-empty");
    for (size_t i = 0; i < n; ++i) {
        AssertNear(result.firstEdgePoints[i].x, result.firstPairEdges[i].x, 1e-6,
                   "firstEdgePoints[i] equals firstPairEdges[i] (x)");
        AssertNear(result.firstEdgePoints[i].y, result.firstPairEdges[i].y, 1e-6,
                   "firstEdgePoints[i] equals firstPairEdges[i] (y)");
    }

    if (!result.widths.empty()) {
        AssertNear(result.meanWidth, 80.0, 4.0, "Array mean width near expected");
    }
}

void TestCaliperArrayRobustLineFit() {
    std::cout << "\n[Test] CaliperArray robust line fit\n";

    QImage img = CreateStepImage(220, 140, 100, 20, 220);

    CaliperRobustFitParams fitParams;
    fitParams.fitMethod = "ransac";
    fitParams.distanceThreshold = 2.0;
    fitParams.ignorePointCount = 1;
    fitParams.ignorePointPolicy = "residual";

    auto lineOpt = MeasureAndFitLineRobust(
        img,
        Point2d{100.0, 20.0},
        Point2d{100.0, 120.0},
        10,
        1.0,
        20.0,
        "all",
        "strongest",
        fitParams
    );

    AssertTrue(lineOpt.has_value(), "MeasureAndFitLineRobust returns valid line");
    if (lineOpt.has_value()) {
        double d = lineOpt->Distance(Point2d{100.0, 70.0});
        AssertNear(d, 0.0, 2.0, "Robust fitted line passes near expected edge");
    }
}

} // namespace

int main() {
    std::cout << "=== Caliper Test ===\n";

    TestMeasurePosOnStepEdge();
    TestMeasurePairsOnBar();
    TestInvalidSelectThrows();
    TestCaliperArrayMeasurePos();
    TestCaliperArrayMeasurePairsFieldConsistency();
    TestCaliperArrayRobustLineFit();

    std::cout << "\n=== Test Summary ===\n";
    std::cout << "Passed: " << testsPassed << "\n";
    std::cout << "Failed: " << testsFailed << "\n";
    return testsFailed > 0 ? 1 : 0;
}
