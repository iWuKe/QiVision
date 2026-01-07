/**
 * @file test_nms.cpp
 * @brief Unit tests for NonMaxSuppression module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/NonMaxSuppression.h>

#include <cmath>
#include <random>
#include <numeric>

using namespace Qi::Vision::Internal;

// ============================================================================
// 1D Peak Detection Tests
// ============================================================================

class NMS1DTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(NMS1DTest, FindLocalMaxima1D_SinglePeak) {
    std::vector<double> signal = {0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0};
    auto maxima = FindLocalMaxima1D(signal.data(), signal.size(), 0.0);

    ASSERT_EQ(maxima.size(), 1u);
    EXPECT_EQ(maxima[0], 5);  // Peak at index 5
}

TEST_F(NMS1DTest, FindLocalMaxima1D_MultiplePeaks) {
    std::vector<double> signal = {0, 3, 1, 5, 2, 4, 1, 6, 2, 3, 0};
    auto maxima = FindLocalMaxima1D(signal.data(), signal.size(), 0.0);

    EXPECT_GE(maxima.size(), 3u);  // At least 3 peaks
}

TEST_F(NMS1DTest, FindLocalMaxima1D_Threshold) {
    std::vector<double> signal = {0, 2, 1, 5, 2, 3, 1, 8, 2, 3, 0};
    auto maxima = FindLocalMaxima1D(signal.data(), signal.size(), 4.0);

    // Only peaks >= 4.0 should be found
    ASSERT_GE(maxima.size(), 2u);
    for (int32_t idx : maxima) {
        EXPECT_GE(signal[idx], 4.0);
    }
}

TEST_F(NMS1DTest, FindLocalMaxima1DRadius_LargerRadius) {
    std::vector<double> signal = {0, 1, 2, 3, 4, 5, 4, 3, 4, 5, 4, 3, 2, 1, 0};
    auto peaks = FindLocalMaxima1DRadius(signal.data(), signal.size(), 3, 0.0);

    // With radius=3, two peaks exist (indices 5 and 9, both with value 5)
    // They are 4 samples apart which is greater than radius=3, so both survive
    ASSERT_EQ(peaks.size(), 2u);
    EXPECT_EQ(peaks[0].value, 5.0);
    EXPECT_EQ(peaks[1].value, 5.0);
}

TEST_F(NMS1DTest, FindPeaks1D_SubpixelRefinement) {
    // Create a parabolic peak at x=5.3
    std::vector<double> signal(11);
    double peakX = 5.3;
    for (size_t i = 0; i < signal.size(); ++i) {
        double x = static_cast<double>(i);
        signal[i] = 10.0 - std::pow(x - peakX, 2);
    }

    auto peaks = FindPeaks1D(signal.data(), signal.size(), 1, 0.0, true);

    ASSERT_EQ(peaks.size(), 1u);
    EXPECT_NEAR(peaks[0].subpixelIndex, peakX, 0.05);
}

TEST_F(NMS1DTest, FindValleys1D_Basic) {
    std::vector<double> signal = {5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5};
    auto valleys = FindValleys1D(signal.data(), signal.size(), 1, 10.0, false);

    ASSERT_EQ(valleys.size(), 1u);
    EXPECT_EQ(valleys[0].index, 5);  // Valley at index 5
    EXPECT_NEAR(valleys[0].value, 0.0, 0.01);
}

TEST_F(NMS1DTest, SuppressPeaks1D_MaxCount) {
    std::vector<Peak1D> peaks = {
        {0, 5.0, 0.0},
        {2, 8.0, 2.0},
        {4, 3.0, 4.0},
        {6, 7.0, 6.0},
        {8, 4.0, 8.0}
    };

    auto result = SuppressPeaks1D(peaks, 2, 0.0);

    ASSERT_EQ(result.size(), 2u);
    // Should keep the top 2 by value: 8.0 and 7.0
    EXPECT_EQ(result[0].value, 8.0);
    EXPECT_EQ(result[1].value, 7.0);
}

TEST_F(NMS1DTest, SuppressPeaks1D_MinDistance) {
    std::vector<Peak1D> peaks = {
        {0, 5.0, 0.0},
        {1, 8.0, 1.0},
        {2, 6.0, 2.0},  // Too close to index 1
        {5, 7.0, 5.0},
        {6, 4.0, 6.0}   // Too close to index 5
    };

    auto result = SuppressPeaks1D(peaks, 10, 2.0);

    // With minDistance=2.0, we should get non-overlapping peaks
    ASSERT_GE(result.size(), 2u);
    for (size_t i = 1; i < result.size(); ++i) {
        EXPECT_GE(std::abs(result[i].subpixelIndex - result[i-1].subpixelIndex), 2.0);
    }
}

// ============================================================================
// 1D Subpixel Refinement Tests
// ============================================================================

TEST_F(NMS1DTest, RefineSubpixelParabolic_CenteredPeak) {
    // Symmetric parabola centered at 0
    double v0 = 8.0, v1 = 10.0, v2 = 8.0;
    double offset = RefineSubpixelParabolic(v0, v1, v2);

    EXPECT_NEAR(offset, 0.0, 0.01);
}

TEST_F(NMS1DTest, RefineSubpixelParabolic_OffsetPeak) {
    // Parabola with peak at 0.25
    // y = -4(x - 0.25)^2 + 10
    // y(-1) = -4*1.5625 + 10 = 3.75
    // y(0) = -4*0.0625 + 10 = 9.75
    // y(1) = -4*0.5625 + 10 = 7.75
    double v0 = 3.75, v1 = 9.75, v2 = 7.75;
    double offset = RefineSubpixelParabolic(v0, v1, v2);

    EXPECT_NEAR(offset, 0.25, 0.01);
}

TEST_F(NMS1DTest, InterpolatedPeakValue_AtMaximum) {
    double v0 = 8.0, v1 = 10.0, v2 = 8.0;
    double offset = 0.0;
    double value = InterpolatedPeakValue(v0, v1, v2, offset);

    EXPECT_NEAR(value, 10.0, 0.01);
}

// ============================================================================
// 2D Gradient NMS Tests (Canny)
// ============================================================================

class NMS2DGradientTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(NMS2DGradientTest, QuantizeDirection_Horizontal) {
    // 0° → horizontal (0)
    EXPECT_EQ(QuantizeDirection(0.0f), 0);
    // 180° → horizontal (0)
    EXPECT_EQ(QuantizeDirection(static_cast<float>(M_PI)), 0);
    // -π → horizontal (0)
    EXPECT_EQ(QuantizeDirection(static_cast<float>(-M_PI)), 0);
}

TEST_F(NMS2DGradientTest, QuantizeDirection_45Degrees) {
    EXPECT_EQ(QuantizeDirection(static_cast<float>(M_PI / 4)), 1);
}

TEST_F(NMS2DGradientTest, QuantizeDirection_Vertical) {
    EXPECT_EQ(QuantizeDirection(static_cast<float>(M_PI / 2)), 2);
}

TEST_F(NMS2DGradientTest, QuantizeDirection_135Degrees) {
    EXPECT_EQ(QuantizeDirection(static_cast<float>(3 * M_PI / 4)), 3);
}

TEST_F(NMS2DGradientTest, NMS2DGradientQuantized_HorizontalEdge) {
    // Create a simple 5x5 image with a horizontal edge
    const int32_t w = 5, h = 5;
    std::vector<float> magnitude(w * h, 0.0f);
    std::vector<float> direction(w * h, 0.0f);
    std::vector<float> output(w * h);

    // Set a horizontal line of edge pixels in the middle row
    // Direction should be vertical (90°) for horizontal edges
    for (int32_t x = 1; x < w - 1; ++x) {
        magnitude[2 * w + x] = 10.0f;
        direction[2 * w + x] = static_cast<float>(M_PI / 2);  // 90°
    }

    NMS2DGradientQuantized(magnitude.data(), direction.data(),
                           output.data(), w, h);

    // Middle row pixels should survive NMS
    for (int32_t x = 1; x < w - 1; ++x) {
        EXPECT_GT(output[2 * w + x], 0.0f);
    }
}

TEST_F(NMS2DGradientTest, NMS2DGradientQuantized_SuppressesWeakNeighbors) {
    // Create a 5x5 image with a ridge (peak in the middle)
    const int32_t w = 5, h = 5;
    std::vector<float> magnitude(w * h, 0.0f);
    std::vector<float> direction(w * h, 0.0f);
    std::vector<float> output(w * h);

    // Create vertical ridge in column 2
    // Gradient direction is horizontal (0°) for vertical edges
    for (int32_t y = 1; y < h - 1; ++y) {
        magnitude[y * w + 1] = 5.0f;
        magnitude[y * w + 2] = 10.0f;  // Peak
        magnitude[y * w + 3] = 5.0f;
        direction[y * w + 1] = 0.0f;
        direction[y * w + 2] = 0.0f;
        direction[y * w + 3] = 0.0f;
    }

    NMS2DGradientQuantized(magnitude.data(), direction.data(),
                           output.data(), w, h);

    // Only the peak column should survive
    for (int32_t y = 1; y < h - 1; ++y) {
        EXPECT_EQ(output[y * w + 1], 0.0f);  // Suppressed
        EXPECT_GT(output[y * w + 2], 0.0f);  // Peak survives
        EXPECT_EQ(output[y * w + 3], 0.0f);  // Suppressed
    }
}

// ============================================================================
// 2D Feature NMS Tests
// ============================================================================

class NMS2DFeatureTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(NMS2DFeatureTest, FindLocalMaxima2D_SinglePeak) {
    const int32_t w = 5, h = 5;
    std::vector<float> response(w * h, 0.0f);

    // Create single peak at (2, 2)
    response[2 * w + 2] = 10.0f;
    response[1 * w + 2] = 5.0f;
    response[3 * w + 2] = 5.0f;
    response[2 * w + 1] = 5.0f;
    response[2 * w + 3] = 5.0f;

    auto peaks = FindLocalMaxima2D(response.data(), w, h, 1, 0.0);

    ASSERT_EQ(peaks.size(), 1u);
    EXPECT_EQ(peaks[0].x, 2);
    EXPECT_EQ(peaks[0].y, 2);
    EXPECT_EQ(peaks[0].value, 10.0f);
}

TEST_F(NMS2DFeatureTest, FindLocalMaxima2D_MultiplePeaks) {
    const int32_t w = 7, h = 7;
    std::vector<float> response(w * h, 0.0f);

    // Create two peaks at (1, 1) and (5, 5)
    response[1 * w + 1] = 10.0f;
    response[5 * w + 5] = 8.0f;

    auto peaks = FindLocalMaxima2D(response.data(), w, h, 1, 0.0);

    ASSERT_EQ(peaks.size(), 2u);
}

TEST_F(NMS2DFeatureTest, FindPeaks2D_SubpixelRefinement) {
    const int32_t w = 5, h = 5;
    std::vector<float> response(w * h, 0.0f);

    // Create a 2D parabolic peak at (2.3, 2.4)
    double peakX = 2.3, peakY = 2.4;
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            double dx = x - peakX;
            double dy = y - peakY;
            response[y * w + x] = static_cast<float>(20.0 - (dx * dx + dy * dy));
        }
    }

    auto peaks = FindPeaks2D(response.data(), w, h, 1, 0.0, true);

    ASSERT_EQ(peaks.size(), 1u);
    EXPECT_NEAR(peaks[0].subpixelX, peakX, 0.1);
    EXPECT_NEAR(peaks[0].subpixelY, peakY, 0.1);
}

TEST_F(NMS2DFeatureTest, SuppressPeaks2D_MaxCount) {
    std::vector<Peak2D> peaks = {
        {0, 0, 5.0, 0.0, 0.0},
        {2, 2, 10.0, 2.0, 2.0},
        {4, 4, 7.0, 4.0, 4.0}
    };

    auto result = SuppressPeaks2D(peaks, 2, 0.0);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].value, 10.0);  // Highest
    EXPECT_EQ(result[1].value, 7.0);   // Second highest
}

TEST_F(NMS2DFeatureTest, SuppressPeaks2D_MinDistance) {
    std::vector<Peak2D> peaks = {
        {0, 0, 10.0, 0.0, 0.0},
        {1, 0, 8.0, 1.0, 0.0},   // Close to (0,0)
        {5, 5, 9.0, 5.0, 5.0},
        {6, 5, 7.0, 6.0, 5.0}    // Close to (5,5)
    };

    auto result = SuppressPeaks2D(peaks, 10, 2.0);

    // Should keep (0,0) and (5,5), suppress (1,0) and (6,5)
    EXPECT_EQ(result.size(), 2u);
}

TEST_F(NMS2DFeatureTest, SuppressPeaks2DGrid_Basic) {
    const int32_t w = 10, h = 10;
    std::vector<Peak2D> peaks = {
        {1, 1, 5.0, 1.0, 1.0},
        {2, 2, 8.0, 2.0, 2.0},   // Same cell as (1,1) with cellSize=4
        {6, 6, 10.0, 6.0, 6.0},
        {7, 7, 7.0, 7.0, 7.0}    // Same cell as (6,6) with cellSize=4
    };

    auto result = SuppressPeaks2DGrid(peaks, w, h, 4);

    // Should keep one peak per 4x4 cell
    EXPECT_EQ(result.size(), 2u);
    // Should keep the strongest in each cell
    bool has8 = false, has10 = false;
    for (const auto& p : result) {
        if (p.value == 8.0) has8 = true;
        if (p.value == 10.0) has10 = true;
    }
    EXPECT_TRUE(has8);
    EXPECT_TRUE(has10);
}

// ============================================================================
// 2D Subpixel Refinement Tests
// ============================================================================

TEST_F(NMS2DFeatureTest, RefineSubpixel2D_CenteredPeak) {
    const int32_t w = 3, h = 3;
    std::vector<float> response = {
        5, 8, 5,
        8, 10, 8,
        5, 8, 5
    };

    double subX, subY;
    double value = RefineSubpixel2D(response.data(), w, h, 1, 1, subX, subY);

    EXPECT_NEAR(subX, 1.0, 0.01);
    EXPECT_NEAR(subY, 1.0, 0.01);
    EXPECT_NEAR(value, 10.0, 0.1);
}

TEST_F(NMS2DFeatureTest, RefineSubpixel2DTaylor_CenteredPeak) {
    const int32_t w = 3, h = 3;
    std::vector<float> response = {
        5, 8, 5,
        8, 10, 8,
        5, 8, 5
    };

    double subX, subY;
    double value = RefineSubpixel2DTaylor(response.data(), w, h, 1, 1, subX, subY);

    EXPECT_NEAR(subX, 1.0, 0.01);
    EXPECT_NEAR(subY, 1.0, 0.01);
}

// ============================================================================
// Box NMS Tests
// ============================================================================

class BoxNMSTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(BoxNMSTest, ComputeIoU_NoOverlap) {
    BoundingBox a{0, 0, 10, 10, 1.0, 0};
    BoundingBox b{20, 20, 30, 30, 1.0, 0};

    EXPECT_NEAR(ComputeIoU(a, b), 0.0, 0.001);
}

TEST_F(BoxNMSTest, ComputeIoU_PartialOverlap) {
    BoundingBox a{0, 0, 10, 10, 1.0, 0};
    BoundingBox b{5, 5, 15, 15, 1.0, 0};

    // Intersection: 5x5 = 25
    // Union: 100 + 100 - 25 = 175
    // IoU = 25/175 ≈ 0.143
    EXPECT_NEAR(ComputeIoU(a, b), 25.0 / 175.0, 0.001);
}

TEST_F(BoxNMSTest, ComputeIoU_FullOverlap) {
    BoundingBox a{0, 0, 10, 10, 1.0, 0};
    BoundingBox b{0, 0, 10, 10, 1.0, 0};

    EXPECT_NEAR(ComputeIoU(a, b), 1.0, 0.001);
}

TEST_F(BoxNMSTest, NMSBoxes_SuppressOverlapping) {
    std::vector<BoundingBox> boxes = {
        {0, 0, 10, 10, 0.9, 0},
        {1, 1, 11, 11, 0.8, 0},  // Overlaps with first
        {20, 20, 30, 30, 0.7, 0} // No overlap
    };

    auto kept = NMSBoxes(boxes, 0.5);

    // First and third should be kept
    EXPECT_EQ(kept.size(), 2u);
    EXPECT_EQ(kept[0], 0);  // Highest score
    EXPECT_EQ(kept[1], 2);  // Non-overlapping
}

TEST_F(BoxNMSTest, NMSBoxes_KeepNonOverlapping) {
    std::vector<BoundingBox> boxes = {
        {0, 0, 10, 10, 0.9, 0},
        {20, 20, 30, 30, 0.8, 0},
        {40, 40, 50, 50, 0.7, 0}
    };

    auto kept = NMSBoxes(boxes, 0.5);

    // All should be kept
    EXPECT_EQ(kept.size(), 3u);
}

TEST_F(BoxNMSTest, NMSBoxesMultiClass_DifferentClasses) {
    std::vector<BoundingBox> boxes = {
        {0, 0, 10, 10, 0.9, 0},   // Class 0
        {1, 1, 11, 11, 0.85, 1},  // Class 1, overlaps but different class
        {0.5, 0.5, 10.5, 10.5, 0.8, 0}  // Class 0, high overlap (IoU>0.5) with first
    };

    auto kept = NMSBoxesMultiClass(boxes, 0.5);

    // Box 0 and 1 are different classes: both kept
    // Box 2 overlaps with box 0 (same class, IoU ≈ 0.82): suppressed
    EXPECT_EQ(kept.size(), 2u);
}

TEST_F(BoxNMSTest, SoftNMSBoxes_DecayScores) {
    std::vector<BoundingBox> boxes = {
        {0, 0, 10, 10, 0.9, 0},
        {1, 1, 11, 11, 0.8, 0},  // Overlaps with first
        {20, 20, 30, 30, 0.7, 0} // No overlap
    };

    auto kept = SoftNMSBoxes(boxes, 0.5, 0.001);

    // All boxes should be returned (just with decayed scores)
    EXPECT_EQ(kept.size(), 3u);
    // Second box's score should be decayed
    EXPECT_LT(boxes[1].score, 0.8);
}

// ============================================================================
// Hysteresis Thresholding Tests
// ============================================================================

class HysteresisTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(HysteresisTest, HysteresisThreshold_StrongEdges) {
    const int32_t w = 5, h = 5;
    std::vector<float> edges(w * h, 0.0f);
    std::vector<uint8_t> output(w * h);

    // Create strong edge
    edges[2 * w + 2] = 20.0f;

    HysteresisThreshold(edges.data(), output.data(), w, h, 5.0f, 10.0f);

    EXPECT_EQ(output[2 * w + 2], 255);
}

TEST_F(HysteresisTest, HysteresisThreshold_WeakEdgesConnected) {
    const int32_t w = 5, h = 1;
    std::vector<float> edges = {0.0f, 7.0f, 15.0f, 7.0f, 0.0f};
    std::vector<uint8_t> output(w * h);

    // Low=5, High=10
    // Center pixel (15) is strong
    // Adjacent pixels (7) are weak but connected to strong
    HysteresisThreshold(edges.data(), output.data(), w, h, 5.0f, 10.0f);

    EXPECT_EQ(output[2], 255);  // Strong
    EXPECT_EQ(output[1], 255);  // Weak connected to strong
    EXPECT_EQ(output[3], 255);  // Weak connected to strong
    EXPECT_EQ(output[0], 0);    // Below threshold
    EXPECT_EQ(output[4], 0);    // Below threshold
}

TEST_F(HysteresisTest, HysteresisThreshold_WeakEdgesNotConnected) {
    const int32_t w = 5, h = 1;
    std::vector<float> edges = {0.0f, 7.0f, 0.0f, 7.0f, 0.0f};
    std::vector<uint8_t> output(w * h);

    // Weak edges not connected to any strong edge
    HysteresisThreshold(edges.data(), output.data(), w, h, 5.0f, 10.0f);

    for (int32_t i = 0; i < w; ++i) {
        EXPECT_EQ(output[i], 0);
    }
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST(NMSUtilityTest, SortPeaks1DByPosition) {
    std::vector<Peak1D> peaks = {
        {5, 10.0, 5.0},
        {2, 8.0, 2.0},
        {8, 7.0, 8.0}
    };

    SortPeaks1DByPosition(peaks);

    EXPECT_EQ(peaks[0].index, 2);
    EXPECT_EQ(peaks[1].index, 5);
    EXPECT_EQ(peaks[2].index, 8);
}

TEST(NMSUtilityTest, SortPeaks2DByPosition) {
    std::vector<Peak2D> peaks = {
        {5, 3, 10.0, 5.0, 3.0},
        {2, 1, 8.0, 2.0, 1.0},
        {2, 3, 7.0, 2.0, 3.0}
    };

    SortPeaks2DByPosition(peaks);

    // Should be sorted by y then x
    EXPECT_EQ(peaks[0].y, 1);
    EXPECT_EQ(peaks[1].x, 2);
    EXPECT_EQ(peaks[1].y, 3);
    EXPECT_EQ(peaks[2].x, 5);
    EXPECT_EQ(peaks[2].y, 3);
}

TEST(NMSUtilityTest, PeakDistance) {
    Peak2D a{0, 0, 0.0, 0.0, 0.0};
    Peak2D b{0, 0, 0.0, 3.0, 4.0};

    EXPECT_NEAR(PeakDistance(a, b), 5.0, 0.001);
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

TEST(NMSEdgeCaseTest, EmptyInput_1D) {
    std::vector<double> signal;
    auto maxima = FindLocalMaxima1D(signal.data(), 0, 0.0);
    EXPECT_TRUE(maxima.empty());
}

TEST(NMSEdgeCaseTest, TooShortSignal_1D) {
    std::vector<double> signal = {1.0, 2.0};
    auto maxima = FindLocalMaxima1D(signal.data(), 2, 0.0);
    EXPECT_TRUE(maxima.empty());
}

TEST(NMSEdgeCaseTest, EmptyInput_2D) {
    auto peaks = FindLocalMaxima2D(nullptr, 0, 0, 1, 0.0);
    EXPECT_TRUE(peaks.empty());
}

TEST(NMSEdgeCaseTest, EmptyBoxList) {
    std::vector<BoundingBox> boxes;
    auto kept = NMSBoxes(boxes, 0.5);
    EXPECT_TRUE(kept.empty());
}

TEST(NMSEdgeCaseTest, SingleBox) {
    std::vector<BoundingBox> boxes = {{0, 0, 10, 10, 0.9, 0}};
    auto kept = NMSBoxes(boxes, 0.5);
    ASSERT_EQ(kept.size(), 1u);
    EXPECT_EQ(kept[0], 0);
}

// ============================================================================
// Performance-Related Tests
// ============================================================================

TEST(NMSPerformanceTest, LargeSignal_1D) {
    const size_t size = 10000;
    std::vector<double> signal(size);

    // Create signal with periodic peaks
    for (size_t i = 0; i < size; ++i) {
        signal[i] = std::sin(static_cast<double>(i) * 0.1) + 1.0;
    }

    auto peaks = FindPeaks1D(signal.data(), size, 3, 1.5, true);

    // Should find roughly size / (2π / 0.1) ≈ 160 peaks
    EXPECT_GT(peaks.size(), 100u);
    EXPECT_LT(peaks.size(), 200u);
}

TEST(NMSPerformanceTest, LargeImage_2D) {
    const int32_t w = 100, h = 100;
    std::vector<float> response(w * h, 0.0f);

    // Create sparse peaks
    response[10 * w + 10] = 10.0f;
    response[50 * w + 50] = 8.0f;
    response[90 * w + 90] = 9.0f;

    auto peaks = FindLocalMaxima2D(response.data(), w, h, 1, 5.0);

    EXPECT_EQ(peaks.size(), 3u);
}

TEST(NMSPerformanceTest, ManyBoxes) {
    const size_t numBoxes = 100;
    std::vector<BoundingBox> boxes;
    boxes.reserve(numBoxes);

    // Create non-overlapping boxes
    for (size_t i = 0; i < numBoxes; ++i) {
        double x = (i % 10) * 20.0;
        double y = (i / 10) * 20.0;
        boxes.push_back({x, y, x + 10, y + 10,
                         static_cast<double>(numBoxes - i) / numBoxes, 0});
    }

    auto kept = NMSBoxes(boxes, 0.5);

    // All boxes should be kept (no overlap)
    EXPECT_EQ(kept.size(), numBoxes);
}
