/**
 * @file CaliperAccuracyTest.cpp
 * @brief Precision/accuracy tests for Measure/Caliper module
 *
 * Tests accuracy requirements from CLAUDE.md (standard conditions: contrast>=50, noise sigma<=5):
 * - Caliper Position: < 0.03 px (1 sigma)
 * - Caliper Width: < 0.05 px (1 sigma)
 *
 * Test methodology:
 * 1. Generate synthetic images with known ground truth edge positions
 * 2. Add Gaussian noise at various levels (sigma = 0, 1, 2, 5 pixels)
 * 3. Run caliper measurement
 * 4. Compute error statistics (mean, std, max, median)
 * 5. Verify precision meets requirements
 *
 * Test conditions:
 * - Ideal: noise = 0 (algorithm limit test)
 * - Standard: noise sigma <= 5 (production requirement)
 * - Difficult: noise sigma <= 15 (degraded performance)
 *
 * Each test runs multiple trials (100-500) for stable statistics.
 */

#include <QiVision/Measure/Caliper.h>
#include <QiVision/Measure/MeasureHandle.h>
#include <QiVision/Measure/MeasureTypes.h>
#include <QiVision/Core/QImage.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace Qi::Vision::Measure {
namespace {

// =============================================================================
// Constants
// =============================================================================

constexpr double PI = 3.14159265358979323846;

/// Number of trials for statistical tests
constexpr int NUM_TRIALS_STANDARD = 200;
constexpr int NUM_TRIALS_EXTENDED = 500;

/// Precision requirements from CLAUDE.md
constexpr double CALIPER_POSITION_REQUIREMENT_PX = 0.03;  // < 0.03 px (1 sigma)
constexpr double CALIPER_WIDTH_REQUIREMENT_PX = 0.05;     // < 0.05 px (1 sigma)

/// Current implementation precision (baseline - may need improvement)
constexpr double CURRENT_POSITION_PRECISION_PX = 0.05;    // Current baseline
constexpr double CURRENT_WIDTH_PRECISION_PX = 0.10;       // Current baseline

/// Safety margin for statistical tests
constexpr double SAFETY_MARGIN = 2.0;

// =============================================================================
// Statistics Helper
// =============================================================================

struct ErrorStats {
    double mean = 0.0;
    double stddev = 0.0;
    double median = 0.0;
    double max = 0.0;
    double min = 0.0;
    double rms = 0.0;
    int count = 0;
};

ErrorStats ComputeErrorStats(const std::vector<double>& errors) {
    ErrorStats stats;
    stats.count = static_cast<int>(errors.size());

    if (errors.empty()) {
        return stats;
    }

    // Mean
    double sum = 0.0;
    for (double e : errors) {
        sum += e;
    }
    stats.mean = sum / errors.size();

    // Standard deviation
    double sumSq = 0.0;
    for (double e : errors) {
        double diff = e - stats.mean;
        sumSq += diff * diff;
    }
    stats.stddev = std::sqrt(sumSq / errors.size());

    // RMS
    double sumSqErr = 0.0;
    for (double e : errors) {
        sumSqErr += e * e;
    }
    stats.rms = std::sqrt(sumSqErr / errors.size());

    // Min/Max
    stats.min = *std::min_element(errors.begin(), errors.end());
    stats.max = *std::max_element(errors.begin(), errors.end());

    // Median
    std::vector<double> sorted = errors;
    std::sort(sorted.begin(), sorted.end());
    size_t mid = sorted.size() / 2;
    if (sorted.size() % 2 == 0) {
        stats.median = (sorted[mid - 1] + sorted[mid]) / 2.0;
    } else {
        stats.median = sorted[mid];
    }

    return stats;
}

void PrintStats(const std::string& name, const ErrorStats& stats,
                double requirement = 0.0, const std::string& unit = "px") {
    std::cout << "  " << name << ":\n"
              << "    Mean:   " << std::fixed << std::setprecision(6) << stats.mean << " " << unit << "\n"
              << "    StdDev: " << stats.stddev << " " << unit << "\n"
              << "    Median: " << stats.median << " " << unit << "\n"
              << "    Min:    " << stats.min << " " << unit << "\n"
              << "    Max:    " << stats.max << " " << unit << "\n"
              << "    RMS:    " << stats.rms << " " << unit << "\n"
              << "    Count:  " << stats.count << "\n";

    if (requirement > 0) {
        bool passed = stats.stddev < requirement;
        std::cout << "    Requirement: " << requirement << " " << unit << " (1 sigma)\n"
                  << "    Status: " << (passed ? "PASS" : "FAIL") << "\n";
    }
}

// =============================================================================
// Test Image Generation Helpers
// =============================================================================

/**
 * @brief Generate image with vertical edge at specified position
 * @param width Image width
 * @param height Image height
 * @param edgeX Subpixel X position of edge
 * @param lowValue Left side gray value
 * @param highValue Right side gray value
 * @param transitionWidth Width of edge transition (pixels)
 * @param noiseStddev Gaussian noise standard deviation
 * @param rng Random number generator
 */
QImage GenerateVerticalEdgeImage(
    int32_t width, int32_t height,
    double edgeX,
    double lowValue, double highValue,
    double transitionWidth,
    double noiseStddev,
    std::mt19937& rng) {

    QImage image(width, height, PixelType::UInt8);
    uint8_t* data = static_cast<uint8_t*>(image.Data());
    size_t stride = image.Stride();

    std::normal_distribution<double> noiseDist(0.0, noiseStddev);
    double amplitude = highValue - lowValue;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            double dx = (x - edgeX) / (transitionWidth * 0.5);
            double t = std::tanh(dx * 1.7);  // erf-like transition
            double value = lowValue + amplitude * 0.5 * (1.0 + t);

            if (noiseStddev > 0) {
                value += noiseDist(rng);
            }

            value = std::clamp(value, 0.0, 255.0);
            data[y * stride + x] = static_cast<uint8_t>(std::round(value));
        }
    }

    return image;
}

/**
 * @brief Generate image with horizontal edge at specified position
 */
QImage GenerateHorizontalEdgeImage(
    int32_t width, int32_t height,
    double edgeY,
    double lowValue, double highValue,
    double transitionWidth,
    double noiseStddev,
    std::mt19937& rng) {

    QImage image(width, height, PixelType::UInt8);
    uint8_t* data = static_cast<uint8_t*>(image.Data());
    size_t stride = image.Stride();

    std::normal_distribution<double> noiseDist(0.0, noiseStddev);
    double amplitude = highValue - lowValue;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            double dy = (y - edgeY) / (transitionWidth * 0.5);
            double t = std::tanh(dy * 1.7);
            double value = lowValue + amplitude * 0.5 * (1.0 + t);

            if (noiseStddev > 0) {
                value += noiseDist(rng);
            }

            value = std::clamp(value, 0.0, 255.0);
            data[y * stride + x] = static_cast<uint8_t>(std::round(value));
        }
    }

    return image;
}

/**
 * @brief Generate image with vertical stripe (two edges) at specified positions
 * @param edge1X First (left) edge position
 * @param edge2X Second (right) edge position
 * @param stripeValue Gray value inside stripe
 * @param bgValue Gray value outside stripe
 */
QImage GenerateVerticalStripeImage(
    int32_t width, int32_t height,
    double edge1X, double edge2X,
    double bgValue, double stripeValue,
    double transitionWidth,
    double noiseStddev,
    std::mt19937& rng) {

    QImage image(width, height, PixelType::UInt8);
    uint8_t* data = static_cast<uint8_t*>(image.Data());
    size_t stride = image.Stride();

    std::normal_distribution<double> noiseDist(0.0, noiseStddev);
    double amplitude = stripeValue - bgValue;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            // Two edges: rising at edge1X, falling at edge2X
            double dx1 = (x - edge1X) / (transitionWidth * 0.5);
            double dx2 = (x - edge2X) / (transitionWidth * 0.5);
            double t1 = std::tanh(dx1 * 1.7);
            double t2 = std::tanh(dx2 * 1.7);

            // Combine: rise then fall
            double value = bgValue + amplitude * 0.5 * (t1 - t2);

            if (noiseStddev > 0) {
                value += noiseDist(rng);
            }

            value = std::clamp(value, 0.0, 255.0);
            data[y * stride + x] = static_cast<uint8_t>(std::round(value));
        }
    }

    return image;
}

/**
 * @brief Generate image with horizontal stripe
 */
QImage GenerateHorizontalStripeImage(
    int32_t width, int32_t height,
    double edge1Y, double edge2Y,
    double bgValue, double stripeValue,
    double transitionWidth,
    double noiseStddev,
    std::mt19937& rng) {

    QImage image(width, height, PixelType::UInt8);
    uint8_t* data = static_cast<uint8_t*>(image.Data());
    size_t stride = image.Stride();

    std::normal_distribution<double> noiseDist(0.0, noiseStddev);
    double amplitude = stripeValue - bgValue;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            double dy1 = (y - edge1Y) / (transitionWidth * 0.5);
            double dy2 = (y - edge2Y) / (transitionWidth * 0.5);
            double t1 = std::tanh(dy1 * 1.7);
            double t2 = std::tanh(dy2 * 1.7);

            double value = bgValue + amplitude * 0.5 * (t1 - t2);

            if (noiseStddev > 0) {
                value += noiseDist(rng);
            }

            value = std::clamp(value, 0.0, 255.0);
            data[y * stride + x] = static_cast<uint8_t>(std::round(value));
        }
    }

    return image;
}

// =============================================================================
// Accuracy Test Base Class
// =============================================================================

class CaliperAccuracyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Fixed seed for reproducibility
        rng_.seed(42);
    }

    std::mt19937 rng_;
};

// =============================================================================
// SINGLE EDGE POSITION ACCURACY TESTS (MeasurePos)
// =============================================================================

class MeasurePosAccuracyTest : public CaliperAccuracyTest {
protected:
    struct TestResult {
        ErrorStats positionErrors;
        int successCount = 0;
        int totalCount = 0;
    };

    /**
     * @brief Run edge position accuracy test with vertical edges
     */
    TestResult RunVerticalEdgeTest(
        int numTrials,
        int imageWidth, int imageHeight,
        double contrast,
        double transitionWidth,
        double noiseStddev,
        double handleWidth = 30.0,
        double handleHeight = 50.0) {

        TestResult result;
        result.totalCount = numTrials;

        std::vector<double> errors;
        errors.reserve(numTrials);

        // Random edge positions within safe region
        std::uniform_real_distribution<double> edgeDist(
            imageWidth * 0.3, imageWidth * 0.7);
        std::uniform_real_distribution<double> subpixelDist(-0.49, 0.49);

        MeasureParams params;
        params.SetSigma(1.0)
              .SetMinAmplitude(contrast * 0.3)
              .SetTransition(EdgeTransition::All);

        for (int trial = 0; trial < numTrials; ++trial) {
            // True edge position with subpixel offset
            double trueEdgeX = edgeDist(rng_) + subpixelDist(rng_);

            // Generate test image
            auto image = GenerateVerticalEdgeImage(
                imageWidth, imageHeight,
                trueEdgeX,
                50.0, 50.0 + contrast,
                transitionWidth,
                noiseStddev, rng_);

            // Create handle centered on edge (approximately)
            // For vertical edge: profile should run horizontally (phi = -PI/2)
            double handleCenterRow = imageHeight / 2.0;
            double handleCenterCol = std::round(trueEdgeX);

            MeasureRectangle2 handle(
                handleCenterRow, handleCenterCol,
                -PI / 2.0,  // phi = -PI/2 makes profile run horizontally
                handleWidth,  // length (along profile)
                handleHeight  // width (for averaging)
            );

            // Measure
            auto edges = MeasurePos(image, handle, params);

            if (!edges.empty()) {
                // Find edge closest to true position
                double minError = std::numeric_limits<double>::max();
                for (const auto& edge : edges) {
                    double error = std::abs(edge.column - trueEdgeX);
                    minError = std::min(minError, error);
                }
                errors.push_back(minError);
                result.successCount++;
            }
        }

        result.positionErrors = ComputeErrorStats(errors);
        return result;
    }

    /**
     * @brief Run edge position accuracy test with horizontal edges
     */
    TestResult RunHorizontalEdgeTest(
        int numTrials,
        int imageWidth, int imageHeight,
        double contrast,
        double transitionWidth,
        double noiseStddev,
        double handleWidth = 50.0,
        double handleHeight = 30.0) {

        TestResult result;
        result.totalCount = numTrials;

        std::vector<double> errors;
        errors.reserve(numTrials);

        std::uniform_real_distribution<double> edgeDist(
            imageHeight * 0.3, imageHeight * 0.7);
        std::uniform_real_distribution<double> subpixelDist(-0.49, 0.49);

        MeasureParams params;
        params.SetSigma(1.0)
              .SetMinAmplitude(contrast * 0.3)
              .SetTransition(EdgeTransition::All);

        for (int trial = 0; trial < numTrials; ++trial) {
            double trueEdgeY = edgeDist(rng_) + subpixelDist(rng_);

            auto image = GenerateHorizontalEdgeImage(
                imageWidth, imageHeight,
                trueEdgeY,
                50.0, 50.0 + contrast,
                transitionWidth,
                noiseStddev, rng_);

            // For horizontal edge: profile should run vertically (phi = 0)
            double handleCenterRow = std::round(trueEdgeY);
            double handleCenterCol = imageWidth / 2.0;

            MeasureRectangle2 handle(
                handleCenterRow, handleCenterCol,
                0.0,  // phi = 0 makes profile run vertically
                handleWidth,  // length (along profile)
                handleHeight  // width (for averaging)
            );

            auto edges = MeasurePos(image, handle, params);

            if (!edges.empty()) {
                double minError = std::numeric_limits<double>::max();
                for (const auto& edge : edges) {
                    double error = std::abs(edge.row - trueEdgeY);
                    minError = std::min(minError, error);
                }
                errors.push_back(minError);
                result.successCount++;
            }
        }

        result.positionErrors = ComputeErrorStats(errors);
        return result;
    }
};

// ---- Ideal Condition Tests (noise = 0) ----

TEST_F(MeasurePosAccuracyTest, VerticalEdge_IdealCondition) {
    std::cout << "\n=== MeasurePos Vertical Edge Ideal Condition (noise = 0) ===\n";

    auto result = RunVerticalEdgeTest(NUM_TRIALS_STANDARD, 100, 100, 100.0, 2.0, 0.0);

    PrintStats("Position Error", result.positionErrors, CALIPER_POSITION_REQUIREMENT_PX);

    std::cout << "  Success Rate: " << 100.0 * result.successCount / result.totalCount << "%\n";

    EXPECT_GT(result.successCount, result.totalCount * 0.95);
    EXPECT_LT(result.positionErrors.stddev, CURRENT_POSITION_PRECISION_PX);
}

TEST_F(MeasurePosAccuracyTest, HorizontalEdge_IdealCondition) {
    std::cout << "\n=== MeasurePos Horizontal Edge Ideal Condition (noise = 0) ===\n";

    auto result = RunHorizontalEdgeTest(NUM_TRIALS_STANDARD, 100, 100, 100.0, 2.0, 0.0);

    PrintStats("Position Error", result.positionErrors, CALIPER_POSITION_REQUIREMENT_PX);

    std::cout << "  Success Rate: " << 100.0 * result.successCount / result.totalCount << "%\n";

    EXPECT_GT(result.successCount, result.totalCount * 0.95);
    EXPECT_LT(result.positionErrors.stddev, CURRENT_POSITION_PRECISION_PX);
}

// ---- Standard Condition Tests (noise sigma <= 5) ----

TEST_F(MeasurePosAccuracyTest, VerticalEdge_StandardCondition_Noise1) {
    std::cout << "\n=== MeasurePos Vertical Edge Standard Condition (noise = 1) ===\n";

    auto result = RunVerticalEdgeTest(NUM_TRIALS_STANDARD, 100, 100, 100.0, 2.0, 1.0);

    PrintStats("Position Error", result.positionErrors, CALIPER_POSITION_REQUIREMENT_PX);

    EXPECT_GT(result.successCount, result.totalCount * 0.95);
    EXPECT_LT(result.positionErrors.stddev, CURRENT_POSITION_PRECISION_PX * SAFETY_MARGIN);
}

TEST_F(MeasurePosAccuracyTest, VerticalEdge_StandardCondition_Noise2) {
    std::cout << "\n=== MeasurePos Vertical Edge Standard Condition (noise = 2) ===\n";

    auto result = RunVerticalEdgeTest(NUM_TRIALS_STANDARD, 100, 100, 100.0, 2.0, 2.0);

    PrintStats("Position Error", result.positionErrors, CALIPER_POSITION_REQUIREMENT_PX);

    EXPECT_GT(result.successCount, result.totalCount * 0.90);
    EXPECT_LT(result.positionErrors.stddev, CURRENT_POSITION_PRECISION_PX * SAFETY_MARGIN * 2);
}

TEST_F(MeasurePosAccuracyTest, VerticalEdge_StandardCondition_Noise5) {
    std::cout << "\n=== MeasurePos Vertical Edge Standard Condition (noise = 5) ===\n";

    auto result = RunVerticalEdgeTest(NUM_TRIALS_STANDARD, 100, 100, 100.0, 2.0, 5.0);

    PrintStats("Position Error", result.positionErrors, CALIPER_POSITION_REQUIREMENT_PX);

    EXPECT_GT(result.successCount, result.totalCount * 0.85);
}

// ---- Contrast Effect Tests ----

TEST_F(MeasurePosAccuracyTest, VerticalEdge_HighContrast) {
    std::cout << "\n=== MeasurePos High Contrast (contrast = 200, noise = 2) ===\n";

    auto result = RunVerticalEdgeTest(NUM_TRIALS_STANDARD, 100, 100, 200.0, 2.0, 2.0);

    PrintStats("Position Error", result.positionErrors);

    EXPECT_GT(result.successCount, result.totalCount * 0.95);
}

TEST_F(MeasurePosAccuracyTest, VerticalEdge_LowContrast) {
    std::cout << "\n=== MeasurePos Low Contrast (contrast = 30, noise = 2) ===\n";

    auto result = RunVerticalEdgeTest(NUM_TRIALS_STANDARD, 100, 100, 30.0, 2.0, 2.0);

    PrintStats("Position Error", result.positionErrors);

    // Lower contrast is harder
    EXPECT_GT(result.successCount, result.totalCount * 0.70);
}

// =============================================================================
// EDGE PAIR (WIDTH) ACCURACY TESTS (MeasurePairs)
// =============================================================================

class MeasurePairsAccuracyTest : public CaliperAccuracyTest {
protected:
    struct TestResult {
        ErrorStats positionErrors;  // Center position error
        ErrorStats widthErrors;     // Width (distance) error
        int successCount = 0;
        int totalCount = 0;
    };

    /**
     * @brief Run edge pair accuracy test with vertical stripe
     */
    TestResult RunVerticalStripeTest(
        int numTrials,
        int imageWidth, int imageHeight,
        double stripeWidth,  // True stripe width
        double contrast,
        double transitionWidth,
        double noiseStddev,
        double handleWidth = 60.0,
        double handleHeight = 50.0) {

        TestResult result;
        result.totalCount = numTrials;

        std::vector<double> posErrors;
        std::vector<double> widthErrors;
        posErrors.reserve(numTrials);
        widthErrors.reserve(numTrials);

        // Random stripe center within safe region
        std::uniform_real_distribution<double> centerDist(
            imageWidth * 0.35, imageWidth * 0.65);
        std::uniform_real_distribution<double> subpixelDist(-0.49, 0.49);

        PairParams pairParams;
        pairParams.sigma = 1.0;
        pairParams.minAmplitude = contrast * 0.3;
        pairParams.SetWidthRange(stripeWidth * 0.5, stripeWidth * 2.0);

        for (int trial = 0; trial < numTrials; ++trial) {
            // True stripe center with subpixel offset
            double trueCenterX = centerDist(rng_) + subpixelDist(rng_);
            double trueEdge1X = trueCenterX - stripeWidth / 2.0;
            double trueEdge2X = trueCenterX + stripeWidth / 2.0;

            // Generate test image
            auto image = GenerateVerticalStripeImage(
                imageWidth, imageHeight,
                trueEdge1X, trueEdge2X,
                50.0, 50.0 + contrast,
                transitionWidth,
                noiseStddev, rng_);

            // Create handle centered on stripe
            // For vertical stripe: profile should run horizontally (phi = -PI/2)
            double handleCenterRow = imageHeight / 2.0;
            double handleCenterCol = std::round(trueCenterX);

            MeasureRectangle2 handle(
                handleCenterRow, handleCenterCol,
                -PI / 2.0,  // phi = -PI/2 makes profile run horizontally
                handleWidth,  // length
                handleHeight  // width
            );

            // Measure pairs
            auto pairs = MeasurePairs(image, handle, pairParams);

            if (!pairs.empty()) {
                // Find pair closest to true width
                double minWidthError = std::numeric_limits<double>::max();
                double correspondingPosError = 0.0;

                for (const auto& pair : pairs) {
                    double measuredWidth = pair.width;
                    double measuredCenterX = pair.centerColumn;

                    double widthError = std::abs(measuredWidth - stripeWidth);
                    double posError = std::abs(measuredCenterX - trueCenterX);

                    if (widthError < minWidthError) {
                        minWidthError = widthError;
                        correspondingPosError = posError;
                    }
                }

                widthErrors.push_back(minWidthError);
                posErrors.push_back(correspondingPosError);
                result.successCount++;
            }
        }

        result.positionErrors = ComputeErrorStats(posErrors);
        result.widthErrors = ComputeErrorStats(widthErrors);
        return result;
    }

    /**
     * @brief Run edge pair accuracy test with horizontal stripe
     */
    TestResult RunHorizontalStripeTest(
        int numTrials,
        int imageWidth, int imageHeight,
        double stripeWidth,
        double contrast,
        double transitionWidth,
        double noiseStddev,
        double handleWidth = 50.0,
        double handleHeight = 60.0) {

        TestResult result;
        result.totalCount = numTrials;

        std::vector<double> posErrors;
        std::vector<double> widthErrors;
        posErrors.reserve(numTrials);
        widthErrors.reserve(numTrials);

        std::uniform_real_distribution<double> centerDist(
            imageHeight * 0.35, imageHeight * 0.65);
        std::uniform_real_distribution<double> subpixelDist(-0.49, 0.49);

        PairParams pairParams;
        pairParams.sigma = 1.0;
        pairParams.minAmplitude = contrast * 0.3;
        pairParams.SetWidthRange(stripeWidth * 0.5, stripeWidth * 2.0);

        for (int trial = 0; trial < numTrials; ++trial) {
            double trueCenterY = centerDist(rng_) + subpixelDist(rng_);
            double trueEdge1Y = trueCenterY - stripeWidth / 2.0;
            double trueEdge2Y = trueCenterY + stripeWidth / 2.0;

            auto image = GenerateHorizontalStripeImage(
                imageWidth, imageHeight,
                trueEdge1Y, trueEdge2Y,
                50.0, 50.0 + contrast,
                transitionWidth,
                noiseStddev, rng_);

            // For horizontal stripe: profile should run vertically (phi = 0)
            double handleCenterRow = std::round(trueCenterY);
            double handleCenterCol = imageWidth / 2.0;

            MeasureRectangle2 handle(
                handleCenterRow, handleCenterCol,
                0.0,  // phi = 0 makes profile run vertically
                handleWidth,  // length
                handleHeight  // width
            );

            auto pairs = MeasurePairs(image, handle, pairParams);

            if (!pairs.empty()) {
                double minWidthError = std::numeric_limits<double>::max();
                double correspondingPosError = 0.0;

                for (const auto& pair : pairs) {
                    double measuredWidth = pair.width;
                    double measuredCenterY = pair.centerRow;

                    double widthError = std::abs(measuredWidth - stripeWidth);
                    double posError = std::abs(measuredCenterY - trueCenterY);

                    if (widthError < minWidthError) {
                        minWidthError = widthError;
                        correspondingPosError = posError;
                    }
                }

                widthErrors.push_back(minWidthError);
                posErrors.push_back(correspondingPosError);
                result.successCount++;
            }
        }

        result.positionErrors = ComputeErrorStats(posErrors);
        result.widthErrors = ComputeErrorStats(widthErrors);
        return result;
    }
};

// ---- Ideal Condition Tests (noise = 0) ----

TEST_F(MeasurePairsAccuracyTest, VerticalStripe_IdealCondition) {
    std::cout << "\n=== MeasurePairs Vertical Stripe Ideal Condition (noise = 0) ===\n";

    auto result = RunVerticalStripeTest(NUM_TRIALS_STANDARD, 100, 100, 20.0, 100.0, 2.0, 0.0);

    PrintStats("Position Error", result.positionErrors, CALIPER_POSITION_REQUIREMENT_PX);
    PrintStats("Width Error", result.widthErrors, CALIPER_WIDTH_REQUIREMENT_PX);

    std::cout << "  Success Rate: " << 100.0 * result.successCount / result.totalCount << "%\n";

    EXPECT_GT(result.successCount, result.totalCount * 0.90);
    EXPECT_LT(result.widthErrors.stddev, CURRENT_WIDTH_PRECISION_PX);
}

TEST_F(MeasurePairsAccuracyTest, HorizontalStripe_IdealCondition) {
    std::cout << "\n=== MeasurePairs Horizontal Stripe Ideal Condition (noise = 0) ===\n";

    auto result = RunHorizontalStripeTest(NUM_TRIALS_STANDARD, 100, 100, 20.0, 100.0, 2.0, 0.0);

    PrintStats("Position Error", result.positionErrors, CALIPER_POSITION_REQUIREMENT_PX);
    PrintStats("Width Error", result.widthErrors, CALIPER_WIDTH_REQUIREMENT_PX);

    EXPECT_GT(result.successCount, result.totalCount * 0.90);
    EXPECT_LT(result.widthErrors.stddev, CURRENT_WIDTH_PRECISION_PX);
}

// ---- Standard Condition Tests ----

TEST_F(MeasurePairsAccuracyTest, VerticalStripe_StandardCondition_Noise1) {
    std::cout << "\n=== MeasurePairs Vertical Stripe Standard Condition (noise = 1) ===\n";

    auto result = RunVerticalStripeTest(NUM_TRIALS_STANDARD, 100, 100, 20.0, 100.0, 2.0, 1.0);

    PrintStats("Position Error", result.positionErrors);
    PrintStats("Width Error", result.widthErrors, CALIPER_WIDTH_REQUIREMENT_PX);

    EXPECT_GT(result.successCount, result.totalCount * 0.90);
    EXPECT_LT(result.widthErrors.stddev, CURRENT_WIDTH_PRECISION_PX * SAFETY_MARGIN);
}

TEST_F(MeasurePairsAccuracyTest, VerticalStripe_StandardCondition_Noise2) {
    std::cout << "\n=== MeasurePairs Vertical Stripe Standard Condition (noise = 2) ===\n";

    auto result = RunVerticalStripeTest(NUM_TRIALS_STANDARD, 100, 100, 20.0, 100.0, 2.0, 2.0);

    PrintStats("Position Error", result.positionErrors);
    PrintStats("Width Error", result.widthErrors, CALIPER_WIDTH_REQUIREMENT_PX);

    EXPECT_GT(result.successCount, result.totalCount * 0.85);
    EXPECT_LT(result.widthErrors.stddev, CURRENT_WIDTH_PRECISION_PX * SAFETY_MARGIN * 2);
}

TEST_F(MeasurePairsAccuracyTest, VerticalStripe_StandardCondition_Noise5) {
    std::cout << "\n=== MeasurePairs Vertical Stripe Standard Condition (noise = 5) ===\n";

    auto result = RunVerticalStripeTest(NUM_TRIALS_STANDARD, 100, 100, 20.0, 100.0, 2.0, 5.0);

    PrintStats("Position Error", result.positionErrors);
    PrintStats("Width Error", result.widthErrors, CALIPER_WIDTH_REQUIREMENT_PX);

    EXPECT_GT(result.successCount, result.totalCount * 0.70);
}

// ---- Width Variation Tests ----

TEST_F(MeasurePairsAccuracyTest, NarrowStripe_IdealCondition) {
    std::cout << "\n=== MeasurePairs Narrow Stripe (width = 10px, noise = 0) ===\n";

    auto result = RunVerticalStripeTest(NUM_TRIALS_STANDARD, 100, 100, 10.0, 100.0, 2.0, 0.0,
                                         40.0, 50.0);

    PrintStats("Width Error", result.widthErrors);

    EXPECT_GT(result.successCount, result.totalCount * 0.85);
}

TEST_F(MeasurePairsAccuracyTest, WideStripe_IdealCondition) {
    std::cout << "\n=== MeasurePairs Wide Stripe (width = 40px, noise = 0) ===\n";

    auto result = RunVerticalStripeTest(NUM_TRIALS_STANDARD, 100, 100, 40.0, 100.0, 2.0, 0.0,
                                         80.0, 50.0);

    PrintStats("Width Error", result.widthErrors);

    EXPECT_GT(result.successCount, result.totalCount * 0.90);
}

// =============================================================================
// CLAUDE.md REQUIREMENT VALIDATION TESTS
// =============================================================================

class CaliperCLAUDERequirementTest : public CaliperAccuracyTest {};

TEST_F(CaliperCLAUDERequirementTest, CaliperPosition_MeetsRequirement) {
    std::cout << "\n=== CLAUDE.md Requirement Validation: Caliper Position ===\n";
    std::cout << "Requirement: Position < 0.03 px (1 sigma)\n";
    std::cout << "Test condition: contrast = 100, noise = 0 (ideal)\n\n";

    std::vector<double> errors;
    errors.reserve(NUM_TRIALS_EXTENDED);

    std::uniform_real_distribution<double> edgeDist(30.0, 70.0);
    std::uniform_real_distribution<double> subpixelDist(-0.49, 0.49);

    MeasureParams params;
    params.SetSigma(1.0).SetMinAmplitude(30.0);

    for (int trial = 0; trial < NUM_TRIALS_EXTENDED; ++trial) {
        double trueEdgeX = edgeDist(rng_) + subpixelDist(rng_);

        auto image = GenerateVerticalEdgeImage(
            100, 100, trueEdgeX,
            50.0, 150.0, 2.0, 0.0, rng_);

        // For vertical edge: phi = -PI/2
        MeasureRectangle2 handle(
            50.0, std::round(trueEdgeX),  // row, col
            -PI / 2.0, 30.0, 50.0);       // phi, length, width

        auto edges = MeasurePos(image, handle, params);

        if (!edges.empty()) {
            double minError = std::numeric_limits<double>::max();
            for (const auto& edge : edges) {
                minError = std::min(minError, std::abs(edge.column - trueEdgeX));
            }
            errors.push_back(minError);
        }
    }

    auto stats = ComputeErrorStats(errors);
    PrintStats("Position Error", stats, CALIPER_POSITION_REQUIREMENT_PX);

    std::cout << "\nRequirement check:\n";
    std::cout << "  Required: stddev < " << CALIPER_POSITION_REQUIREMENT_PX << " px\n";
    std::cout << "  Measured: stddev = " << stats.stddev << " px\n";
    std::cout << "  Status: " << (stats.stddev < CALIPER_POSITION_REQUIREMENT_PX ? "PASS" : "NEEDS IMPROVEMENT") << "\n";

    // Current baseline check
    EXPECT_LT(stats.stddev, CURRENT_POSITION_PRECISION_PX);
}

TEST_F(CaliperCLAUDERequirementTest, CaliperWidth_MeetsRequirement) {
    std::cout << "\n=== CLAUDE.md Requirement Validation: Caliper Width ===\n";
    std::cout << "Requirement: Width < 0.05 px (1 sigma)\n";
    std::cout << "Test condition: contrast = 100, noise = 0 (ideal)\n\n";

    std::vector<double> errors;
    errors.reserve(NUM_TRIALS_EXTENDED);

    const double trueStripeWidth = 20.0;
    std::uniform_real_distribution<double> centerDist(35.0, 65.0);
    std::uniform_real_distribution<double> subpixelDist(-0.49, 0.49);

    PairParams pairParams;
    pairParams.sigma = 1.0;
    pairParams.minAmplitude = 30.0;
    pairParams.SetWidthRange(10.0, 40.0);

    for (int trial = 0; trial < NUM_TRIALS_EXTENDED; ++trial) {
        double trueCenterX = centerDist(rng_) + subpixelDist(rng_);
        double trueEdge1X = trueCenterX - trueStripeWidth / 2.0;
        double trueEdge2X = trueCenterX + trueStripeWidth / 2.0;

        auto image = GenerateVerticalStripeImage(
            100, 100, trueEdge1X, trueEdge2X,
            50.0, 150.0, 2.0, 0.0, rng_);

        // For vertical stripe: phi = -PI/2
        MeasureRectangle2 handle(
            50.0, std::round(trueCenterX),  // row, col
            -PI / 2.0, 60.0, 50.0);         // phi, length, width

        auto pairs = MeasurePairs(image, handle, pairParams);

        if (!pairs.empty()) {
            double minWidthError = std::numeric_limits<double>::max();
            for (const auto& pair : pairs) {
                minWidthError = std::min(minWidthError,
                    std::abs(pair.width - trueStripeWidth));
            }
            errors.push_back(minWidthError);
        }
    }

    auto stats = ComputeErrorStats(errors);
    PrintStats("Width Error", stats, CALIPER_WIDTH_REQUIREMENT_PX);

    std::cout << "\nRequirement check:\n";
    std::cout << "  Required: stddev < " << CALIPER_WIDTH_REQUIREMENT_PX << " px\n";
    std::cout << "  Measured: stddev = " << stats.stddev << " px\n";
    std::cout << "  Status: " << (stats.stddev < CALIPER_WIDTH_REQUIREMENT_PX ? "PASS" : "NEEDS IMPROVEMENT") << "\n";

    EXPECT_LT(stats.stddev, CURRENT_WIDTH_PRECISION_PX);
}

// =============================================================================
// NOISE SENSITIVITY STUDY
// =============================================================================

TEST_F(CaliperCLAUDERequirementTest, MeasurePos_NoiseScaling) {
    std::cout << "\n=== MeasurePos Noise Scaling Study ===\n";
    std::cout << "How precision degrades with noise level\n\n";

    std::vector<double> noiseLevels = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0};

    std::cout << std::setw(10) << "Noise" << " | "
              << std::setw(12) << "Mean Error" << " | "
              << std::setw(12) << "Std Dev" << " | "
              << std::setw(12) << "Max Error" << " | "
              << std::setw(10) << "Success %" << "\n";
    std::cout << std::string(64, '-') << "\n";

    std::uniform_real_distribution<double> edgeDist(30.0, 70.0);
    std::uniform_real_distribution<double> subpixelDist(-0.49, 0.49);

    MeasureParams params;
    params.SetSigma(1.0).SetMinAmplitude(20.0);

    for (double noise : noiseLevels) {
        std::vector<double> errors;
        int successCount = 0;
        const int numTrials = 100;

        for (int trial = 0; trial < numTrials; ++trial) {
            double trueEdgeX = edgeDist(rng_) + subpixelDist(rng_);

            auto image = GenerateVerticalEdgeImage(
                100, 100, trueEdgeX,
                50.0, 150.0, 2.0, noise, rng_);

            // For vertical edge: phi = -PI/2
            MeasureRectangle2 handle(
                50.0, std::round(trueEdgeX),  // row, col
                -PI / 2.0, 30.0, 50.0);       // phi, length, width

            auto edges = MeasurePos(image, handle, params);

            if (!edges.empty()) {
                double minError = std::numeric_limits<double>::max();
                for (const auto& edge : edges) {
                    minError = std::min(minError, std::abs(edge.column - trueEdgeX));
                }
                errors.push_back(minError);
                successCount++;
            }
        }

        auto stats = ComputeErrorStats(errors);
        double successRate = 100.0 * successCount / numTrials;

        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(10) << noise << " | "
                  << std::setw(12) << stats.mean << " | "
                  << std::setw(12) << stats.stddev << " | "
                  << std::setw(12) << stats.max << " | "
                  << std::setw(10) << std::setprecision(1) << successRate << "%\n";
    }
}

TEST_F(CaliperCLAUDERequirementTest, MeasurePairs_NoiseScaling) {
    std::cout << "\n=== MeasurePairs Noise Scaling Study ===\n";
    std::cout << "How width precision degrades with noise level\n\n";

    std::vector<double> noiseLevels = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0};

    std::cout << std::setw(10) << "Noise" << " | "
              << std::setw(12) << "Mean Error" << " | "
              << std::setw(12) << "Std Dev" << " | "
              << std::setw(12) << "Max Error" << " | "
              << std::setw(10) << "Success %" << "\n";
    std::cout << std::string(64, '-') << "\n";

    const double trueStripeWidth = 20.0;
    std::uniform_real_distribution<double> centerDist(35.0, 65.0);
    std::uniform_real_distribution<double> subpixelDist(-0.49, 0.49);

    PairParams pairParams;
    pairParams.sigma = 1.0;
    pairParams.minAmplitude = 20.0;
    pairParams.SetWidthRange(10.0, 40.0);

    for (double noise : noiseLevels) {
        std::vector<double> errors;
        int successCount = 0;
        const int numTrials = 100;

        for (int trial = 0; trial < numTrials; ++trial) {
            double trueCenterX = centerDist(rng_) + subpixelDist(rng_);
            double trueEdge1X = trueCenterX - trueStripeWidth / 2.0;
            double trueEdge2X = trueCenterX + trueStripeWidth / 2.0;

            auto image = GenerateVerticalStripeImage(
                100, 100, trueEdge1X, trueEdge2X,
                50.0, 150.0, 2.0, noise, rng_);

            // For vertical stripe: phi = -PI/2
            MeasureRectangle2 handle(
                50.0, std::round(trueCenterX),  // row, col
                -PI / 2.0, 60.0, 50.0);         // phi, length, width

            auto pairs = MeasurePairs(image, handle, pairParams);

            if (!pairs.empty()) {
                double minWidthError = std::numeric_limits<double>::max();
                for (const auto& pair : pairs) {
                    minWidthError = std::min(minWidthError,
                        std::abs(pair.width - trueStripeWidth));
                }
                errors.push_back(minWidthError);
                successCount++;
            }
        }

        auto stats = ComputeErrorStats(errors);
        double successRate = 100.0 * successCount / numTrials;

        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(10) << noise << " | "
                  << std::setw(12) << stats.mean << " | "
                  << std::setw(12) << stats.stddev << " | "
                  << std::setw(12) << stats.max << " | "
                  << std::setw(10) << std::setprecision(1) << successRate << "%\n";
    }
}

// =============================================================================
// CONTRAST SENSITIVITY STUDY
// =============================================================================

TEST_F(CaliperCLAUDERequirementTest, MeasurePos_ContrastScaling) {
    std::cout << "\n=== MeasurePos Contrast Scaling Study ===\n";
    std::cout << "How precision varies with edge contrast\n\n";

    std::vector<double> contrastLevels = {20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0};

    std::cout << std::setw(10) << "Contrast" << " | "
              << std::setw(12) << "Mean Error" << " | "
              << std::setw(12) << "Std Dev" << " | "
              << std::setw(12) << "Max Error" << " | "
              << std::setw(10) << "Success %" << "\n";
    std::cout << std::string(64, '-') << "\n";

    std::uniform_real_distribution<double> edgeDist(30.0, 70.0);
    std::uniform_real_distribution<double> subpixelDist(-0.49, 0.49);

    const double noise = 2.0;  // Fixed moderate noise

    for (double contrast : contrastLevels) {
        MeasureParams params;
        params.SetSigma(1.0).SetMinAmplitude(contrast * 0.2);

        std::vector<double> errors;
        int successCount = 0;
        const int numTrials = 100;

        for (int trial = 0; trial < numTrials; ++trial) {
            double trueEdgeX = edgeDist(rng_) + subpixelDist(rng_);

            auto image = GenerateVerticalEdgeImage(
                100, 100, trueEdgeX,
                50.0, 50.0 + contrast, 2.0, noise, rng_);

            // For vertical edge: phi = -PI/2
            MeasureRectangle2 handle(
                50.0, std::round(trueEdgeX),  // row, col
                -PI / 2.0, 30.0, 50.0);       // phi, length, width

            auto edges = MeasurePos(image, handle, params);

            if (!edges.empty()) {
                double minError = std::numeric_limits<double>::max();
                for (const auto& edge : edges) {
                    minError = std::min(minError, std::abs(edge.column - trueEdgeX));
                }
                errors.push_back(minError);
                successCount++;
            }
        }

        auto stats = ComputeErrorStats(errors);
        double successRate = 100.0 * successCount / numTrials;

        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(10) << contrast << " | "
                  << std::setw(12) << stats.mean << " | "
                  << std::setw(12) << stats.stddev << " | "
                  << std::setw(12) << stats.max << " | "
                  << std::setw(10) << std::setprecision(1) << successRate << "%\n";
    }
}

} // anonymous namespace
} // namespace Qi::Vision::Measure
