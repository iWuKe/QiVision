/**
 * @file test_caliper_regression.cpp
 * @brief Real-image regression tests for Caliper/Metrology
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Measure/Caliper.h>
#include <QiVision/Measure/MeasureHandle.h>
#include <QiVision/Measure/CaliperArray.h>
#include <QiVision/Measure/Metrology.h>

#include <cmath>
#include <iostream>

using namespace Qi::Vision;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::Measure;

namespace {

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

void AssertInRange(double value, double low, double high, const char* message) {
    bool ok = (value >= low && value <= high);
    if (ok) {
        std::cout << "  [PASS] " << message << " (" << value << " in [" << low << ", " << high << "])\n";
        testsPassed++;
    } else {
        std::cout << "  [FAIL] " << message << " (" << value << " not in [" << low << ", " << high << "])\n";
        testsFailed++;
    }
}

void TestSingleCaliperOnCirclePlate() {
    std::cout << "\n[Test] Single caliper pair on circle_plate\n";

    QImage gray;
    ReadImageGray("tests/data/halcon_images/circle_plate.png", gray);
    AssertTrue(!gray.Empty(), "Loaded circle_plate image");
    if (gray.Empty()) return;

    MeasureRectangle2 handle = GenMeasureRectangle2(420.0, 210.0, 0.0, 100.0, 10.0);
    auto pairs = MeasurePairs(gray, handle, 1.0, 20.0, "positive", "first");

    AssertTrue(!pairs.empty(), "Found at least one edge pair");
    if (!pairs.empty()) {
        const auto& p = pairs[0];
        AssertInRange(p.width, 120.0, 220.0, "Measured width in expected range");
        AssertInRange(p.centerColumn, 190.0, 230.0, "Pair center X in expected range");
    }
}

void TestCaliperArrayOnCirclePlate() {
    std::cout << "\n[Test] CaliperArray pairs on circle_plate\n";

    QImage gray;
    ReadImageGray("tests/data/halcon_images/circle_plate.png", gray);
    AssertTrue(!gray.Empty(), "Loaded circle_plate image for array");
    if (gray.Empty()) return;

    CaliperArray array;
    bool created = array.CreateAlongLine(
        Point2d{210.0, 380.0},
        Point2d{210.0, 460.0},
        8,
        90.0,
        8.0
    );
    AssertTrue(created, "CaliperArray created");

    CaliperArrayStats stats;
    CaliperArrayResult result = array.MeasurePairs(gray, 1.0, 20.0, "positive", "first", &stats);

    AssertTrue(result.numValid >= 4, "Array has enough valid pair results");
    if (result.numValid > 0) {
        AssertInRange(result.meanWidth, 110.0, 220.0, "Array mean width in expected range");
    }
    AssertTrue(stats.measurementTime >= 0.0, "Array stats measurementTime populated");
    AssertTrue(stats.avgTimePerCaliper >= 0.0, "Array stats avgTimePerCaliper populated");

    size_t n = std::min(result.firstEdgePoints.size(), result.firstPairEdges.size());
    AssertTrue(n > 0, "Pair edge arrays non-empty");
    bool aligned = true;
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(result.firstEdgePoints[i].x - result.firstPairEdges[i].x) > 1e-9 ||
            std::abs(result.firstEdgePoints[i].y - result.firstPairEdges[i].y) > 1e-9) {
            aligned = false;
            break;
        }
    }
    AssertTrue(aligned, "firstEdgePoints matches firstPairEdges");
}

void TestCircleMetrologyRegression() {
    std::cout << "\n[Test] Circle metrology regression on circle_plate\n";

    QImage gray;
    ReadImageGray("tests/data/halcon_images/circle_plate.png", gray);
    AssertTrue(!gray.Empty(), "Loaded circle_plate image for metrology");
    if (gray.Empty()) return;

    MetrologyModel model;
    MetrologyMeasureParams params;
    params.SetNumMeasures(24).SetThreshold("auto").SetMeasureLength(20.0, 5.0);

    int idx = model.AddCircleMeasure(
        420.0, 210.0, 63.0,
        20.0, 5.0,
        "all", "strongest",
        params
    );
    AssertTrue(idx >= 0, "Added circle metrology object");

    bool ok = model.Apply(gray);
    AssertTrue(ok, "Metrology apply succeeded");
    if (!ok) return;

    auto circle = model.GetCircleResult(idx);
    AssertTrue(circle.IsValid(), "Circle metrology result valid");
    if (circle.IsValid()) {
        AssertInRange(circle.column, 180.0, 240.0, "Circle center X in expected range");
        AssertInRange(circle.row, 390.0, 450.0, "Circle center Y in expected range");
        AssertInRange(circle.radius, 45.0, 85.0, "Circle radius in expected range");
        AssertTrue(circle.numUsed >= 8, "Circle fit used enough points");
    }
}

void TestIgnorePointCountRefit() {
    std::cout << "\n[Test] IgnorePointCount refit behavior\n";

    QImage gray;
    ReadImageGray("tests/data/halcon_images/circle_plate.png", gray);
    AssertTrue(!gray.Empty(), "Loaded circle_plate for ignore-point test");
    if (gray.Empty()) return;

    MetrologyMeasureParams baseParams;
    baseParams.SetNumMeasures(32)
              .SetMeasureLength(20.0, 6.0)
              .SetThreshold("auto")
              .SetFitMethod("huber")
              .SetMinScore(0.3)
              .SetMeasureTransition("all")
              .SetMeasureSelect("strongest");

    MetrologyModel modelBase;
    int idxBase = modelBase.AddCircleMeasure(420.0, 210.0, 63.0, 20.0, 6.0, "all", "strongest", baseParams);
    bool okBase = idxBase >= 0 && modelBase.Apply(gray);
    AssertTrue(okBase, "Baseline metrology apply succeeded");
    if (!okBase) return;
    auto base = modelBase.GetCircleResult(idxBase);
    AssertTrue(base.IsValid(), "Baseline circle valid");
    if (!base.IsValid()) return;

    MetrologyMeasureParams ignoreParams = baseParams;
    ignoreParams.SetIgnorePointCount(4).SetIgnorePointPolicy("residual");

    MetrologyModel modelIgnore;
    int idxIgnore = modelIgnore.AddCircleMeasure(420.0, 210.0, 63.0, 20.0, 6.0, "all", "strongest", ignoreParams);
    bool okIgnore = idxIgnore >= 0 && modelIgnore.Apply(gray);
    AssertTrue(okIgnore, "Ignore-point metrology apply succeeded");
    if (!okIgnore) return;
    auto withIgnore = modelIgnore.GetCircleResult(idxIgnore);
    AssertTrue(withIgnore.IsValid(), "Ignore-point circle valid");
    if (!withIgnore.IsValid()) return;

    AssertTrue(withIgnore.numUsed <= base.numUsed, "numUsed decreased or unchanged after ignoring points");
    AssertInRange(withIgnore.radius, 45.0, 85.0, "Ignore-point radius stays in expected range");
}

void TestRectangleIgnorePointCountRefit() {
    std::cout << "\n[Test] Rectangle2 IgnorePointCount refit behavior\n";

    QImage gray;
    ReadImageGray("tests/data/halcon_images/circle_plate.png", gray);
    AssertTrue(!gray.Empty(), "Loaded circle_plate for rectangle ignore-point test");
    if (gray.Empty()) return;

    MetrologyMeasureParams baseParams;
    baseParams.SetNumMeasures(32)
              .SetMeasureLength(20.0, 5.0)
              .SetThreshold("auto")
              .SetFitMethod("huber")
              .SetMinScore(0.3)
              .SetMeasureTransition("all")
              .SetMeasureSelect("strongest");

    MetrologyModel modelBase;
    int idxBase = modelBase.AddRectangle2Measure(
        705.0, 210.0, 0.0, 65.0, 70.0,
        20.0, 5.0, "all", "strongest", baseParams
    );
    bool okBase = idxBase >= 0 && modelBase.Apply(gray);
    AssertTrue(okBase, "Baseline rectangle metrology apply succeeded");
    if (!okBase) return;
    auto base = modelBase.GetRectangle2Result(idxBase);
    AssertTrue(base.IsValid(), "Baseline rectangle valid");
    if (!base.IsValid()) return;

    MetrologyMeasureParams ignoreParams = baseParams;
    ignoreParams.SetIgnorePointCount(6).SetIgnorePointPolicy("residual");

    MetrologyModel modelIgnore;
    int idxIgnore = modelIgnore.AddRectangle2Measure(
        705.0, 210.0, 0.0, 65.0, 70.0,
        20.0, 5.0, "all", "strongest", ignoreParams
    );
    bool okIgnore = idxIgnore >= 0 && modelIgnore.Apply(gray);
    AssertTrue(okIgnore, "Ignore-point rectangle metrology apply succeeded");
    if (!okIgnore) return;
    auto withIgnore = modelIgnore.GetRectangle2Result(idxIgnore);
    AssertTrue(withIgnore.IsValid(), "Ignore-point rectangle valid");
    if (!withIgnore.IsValid()) return;

    AssertTrue(withIgnore.numUsed <= base.numUsed, "Rectangle numUsed decreased or unchanged after ignoring points");
    AssertInRange(withIgnore.column, 120.0, 300.0, "Rectangle center X remains in expected range");
    AssertInRange(withIgnore.row, 620.0, 790.0, "Rectangle center Y remains in expected range");
    AssertInRange(withIgnore.length1, 35.0, 100.0, "Rectangle length1 remains in expected range");
    AssertInRange(withIgnore.length2, 40.0, 110.0, "Rectangle length2 remains in expected range");
}

void TestLineMetrologyRegression() {
    std::cout << "\n[Test] Line metrology regression on ic image\n";

    QImage gray;
    ReadImageGray("tests/data/halcon_images/ic.png", gray);
    AssertTrue(!gray.Empty(), "Loaded ic image");
    if (gray.Empty()) return;

    MetrologyModel model;
    MetrologyMeasureParams params;
    params.SetNumMeasures(20)
          .SetMeasureLength(30.0, 5.0)
          .SetThreshold("auto")
          .SetFitMethod("ransac")
          .SetMinScore(0.0)
          .SetDistanceThreshold(2.0);

    int idx = model.AddLineMeasure(
        120.0, 100.0,
        120.0, gray.Width() - 100.0,
        30.0, 5.0,
        "all", "strongest",
        params
    );
    AssertTrue(idx >= 0, "Added line metrology object");

    bool ok = model.Apply(gray);
    AssertTrue(ok, "Metrology line apply succeeded");
    if (!ok) return;

    auto line = model.GetLineResult(idx);
    AssertTrue(line.IsValid(), "Line metrology result valid");
    if (line.IsValid()) {
        double angleDeg = std::atan2(line.row2 - line.row1, line.col2 - line.col1) * 180.0 / 3.14159265358979323846;
        AssertInRange(std::abs(angleDeg), 0.0, 20.0, "Line angle near horizontal");
        AssertTrue(line.numUsed >= 6, "Line fit used enough points");
    }
}

} // namespace

int main() {
    std::cout << "=== Caliper Regression Test ===\n";

    try {
        TestSingleCaliperOnCirclePlate();
        TestCaliperArrayOnCirclePlate();
        TestCircleMetrologyRegression();
        TestIgnorePointCountRefit();
        TestRectangleIgnorePointCountRefit();
        TestLineMetrologyRegression();
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Unhandled exception: " << e.what() << "\n";
        testsFailed++;
    }

    std::cout << "\n=== Test Summary ===\n";
    std::cout << "Passed: " << testsPassed << "\n";
    std::cout << "Failed: " << testsFailed << "\n";
    return testsFailed > 0 ? 1 : 0;
}
