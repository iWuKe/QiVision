/**
 * @file test_edge1d.cpp
 * @brief Unit tests for Internal/Edge1D.h
 */

#include <QiVision/Internal/Edge1D.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

class Edge1DTest : public ::testing::Test {
protected:
    // Step edge profile: 0...0, 255...255
    std::vector<double> stepProfile_;

    // Gaussian edge profile (smooth transition)
    std::vector<double> gaussEdge_;

    // Double edge (object with width)
    std::vector<double> doubleEdge_;

    void SetUp() override {
        // Step profile: 0 for first half, 255 for second half
        stepProfile_.resize(100);
        for (size_t i = 0; i < 50; ++i) stepProfile_[i] = 0.0;
        for (size_t i = 50; i < 100; ++i) stepProfile_[i] = 255.0;

        // Gaussian edge: smooth transition around position 50
        gaussEdge_.resize(100);
        for (size_t i = 0; i < 100; ++i) {
            double t = static_cast<double>(i) - 50.0;
            // Sigmoid-like curve
            gaussEdge_[i] = 255.0 / (1.0 + std::exp(-0.5 * t));
        }

        // Double edge: object from 30 to 70 (dark-bright-dark)
        doubleEdge_.resize(100);
        for (size_t i = 0; i < 100; ++i) {
            if (i >= 30 && i < 70) {
                doubleEdge_[i] = 255.0;
            } else {
                doubleEdge_[i] = 0.0;
            }
        }
    }
};

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST_F(Edge1DTest, ClassifyPolarity) {
    EXPECT_EQ(ClassifyPolarity(10.0), EdgePolarity::Positive);
    EXPECT_EQ(ClassifyPolarity(-10.0), EdgePolarity::Negative);
    EXPECT_EQ(ClassifyPolarity(0.0), EdgePolarity::Negative); // 0 is not > 0
}

TEST_F(Edge1DTest, MatchesPolarity) {
    EXPECT_TRUE(MatchesPolarity(EdgePolarity::Positive, EdgePolarity::Both));
    EXPECT_TRUE(MatchesPolarity(EdgePolarity::Negative, EdgePolarity::Both));
    EXPECT_TRUE(MatchesPolarity(EdgePolarity::Positive, EdgePolarity::Positive));
    EXPECT_TRUE(MatchesPolarity(EdgePolarity::Negative, EdgePolarity::Negative));
    EXPECT_FALSE(MatchesPolarity(EdgePolarity::Positive, EdgePolarity::Negative));
    EXPECT_FALSE(MatchesPolarity(EdgePolarity::Negative, EdgePolarity::Positive));
}

// ============================================================================
// Profile Gradient Tests
// ============================================================================

TEST_F(Edge1DTest, ComputeProfileGradient) {
    std::vector<double> gradient(stepProfile_.size());
    ComputeProfileGradient(stepProfile_.data(), gradient.data(),
                           stepProfile_.size(), 0.0);

    // Maximum gradient should be near position 50
    double maxGrad = 0.0;
    size_t maxIdx = 0;
    for (size_t i = 0; i < gradient.size(); ++i) {
        if (std::abs(gradient[i]) > maxGrad) {
            maxGrad = std::abs(gradient[i]);
            maxIdx = i;
        }
    }

    // For a step edge at 49/50 boundary, both positions have identical gradient
    // (central difference: g[49]=(255-0)/2, g[50]=(255-0)/2)
    // Either 49 or 50 is mathematically correct; the true edge is at 49.5
    EXPECT_GE(maxIdx, 49u);
    EXPECT_LE(maxIdx, 50u);
    EXPECT_GT(maxGrad, 100.0);  // Significant gradient
}

TEST_F(Edge1DTest, ComputeProfileGradientSmooth) {
    std::vector<double> gradient(gaussEdge_.size());
    ComputeProfileGradientSmooth(gaussEdge_.data(), gradient.data(),
                                  gaussEdge_.size(), 2.0);

    // Maximum gradient should be near 50
    double maxGrad = 0.0;
    size_t maxIdx = 0;
    for (size_t i = 0; i < gradient.size(); ++i) {
        if (std::abs(gradient[i]) > maxGrad) {
            maxGrad = std::abs(gradient[i]);
            maxIdx = i;
        }
    }

    EXPECT_NEAR(maxIdx, 50, 3);  // Near position 50
}

TEST_F(Edge1DTest, GradientShortProfile) {
    std::vector<double> short1 = {100.0};
    std::vector<double> short2 = {0.0, 255.0};
    std::vector<double> grad1(1), grad2(2);

    ComputeProfileGradient(short1.data(), grad1.data(), 1, 0.0);
    ComputeProfileGradient(short2.data(), grad2.data(), 2, 0.0);

    EXPECT_DOUBLE_EQ(grad1[0], 0.0);
    EXPECT_DOUBLE_EQ(grad2[0], 255.0);
    EXPECT_DOUBLE_EQ(grad2[1], 255.0);
}

// ============================================================================
// Peak Finding Tests
// ============================================================================

TEST_F(Edge1DTest, FindGradientPeaks) {
    std::vector<double> gradient(stepProfile_.size());
    ComputeProfileGradient(stepProfile_.data(), gradient.data(),
                           stepProfile_.size(), 0.0);

    auto peaks = FindGradientPeaks(gradient.data(), gradient.size(), 10.0);

    // Should find one peak near position 50
    EXPECT_EQ(peaks.size(), 1u);
    if (!peaks.empty()) {
        EXPECT_EQ(peaks[0], 50u);
    }
}

TEST_F(Edge1DTest, FindGradientPeaksDoubleEdge) {
    std::vector<double> gradient(doubleEdge_.size());
    ComputeProfileGradient(doubleEdge_.data(), gradient.data(),
                           doubleEdge_.size(), 0.0);

    auto peaks = FindGradientPeaks(gradient.data(), gradient.size(), 10.0);

    // Should find two peaks (rising and falling edges)
    EXPECT_EQ(peaks.size(), 2u);
}

// ============================================================================
// Subpixel Refinement Tests
// ============================================================================

TEST_F(Edge1DTest, RefineEdgeSubpixel) {
    // Create gradient with clear peak
    std::vector<double> grad = {0.0, 5.0, 10.0, 8.0, 3.0, 0.0};

    double refined = RefineEdgeSubpixel(grad.data(), grad.size(), 2.0);

    // Peak is at index 2, but should be refined to slightly before
    EXPECT_GE(refined, 1.5);
    EXPECT_LE(refined, 2.5);
}

TEST_F(Edge1DTest, RefineEdgeSubpixelSymmetric) {
    // Symmetric peak - should stay at center
    std::vector<double> grad = {0.0, 5.0, 10.0, 5.0, 0.0};

    double refined = RefineEdgeSubpixel(grad.data(), grad.size(), 2.0);

    EXPECT_NEAR(refined, 2.0, 0.01);
}

TEST_F(Edge1DTest, RefineEdgeZeroCrossing) {
    double refined = RefineEdgeZeroCrossing(gaussEdge_.data(), gaussEdge_.size(),
                                             50.0, 2.0);

    // Should be near position 50
    EXPECT_NEAR(refined, 50.0, 1.0);
}

// ============================================================================
// Single Edge Detection Tests
// ============================================================================

TEST_F(Edge1DTest, DetectEdges1DStepEdge) {
    auto edges = DetectEdges1D(stepProfile_.data(), stepProfile_.size(),
                                10.0, EdgePolarity::Both, 0.0);

    EXPECT_EQ(edges.size(), 1u);
    if (!edges.empty()) {
        EXPECT_NEAR(edges[0].position, 50.0, 0.5);
        EXPECT_EQ(edges[0].polarity, EdgePolarity::Positive);
        EXPECT_GT(edges[0].amplitude, 100.0);
    }
}

TEST_F(Edge1DTest, DetectEdges1DGaussEdge) {
    auto edges = DetectEdges1D(gaussEdge_.data(), gaussEdge_.size(),
                                5.0, EdgePolarity::Both, 2.0);

    EXPECT_EQ(edges.size(), 1u);
    if (!edges.empty()) {
        EXPECT_NEAR(edges[0].position, 50.0, 2.0);
        EXPECT_EQ(edges[0].polarity, EdgePolarity::Positive);
    }
}

TEST_F(Edge1DTest, DetectEdges1DDoubleEdge) {
    auto edges = DetectEdges1D(doubleEdge_.data(), doubleEdge_.size(),
                                10.0, EdgePolarity::Both, 0.0);

    EXPECT_EQ(edges.size(), 2u);
    if (edges.size() >= 2) {
        // First edge (rising) at ~30
        EXPECT_NEAR(edges[0].position, 30.0, 0.5);
        EXPECT_EQ(edges[0].polarity, EdgePolarity::Positive);

        // Second edge (falling) at ~70
        EXPECT_NEAR(edges[1].position, 70.0, 0.5);
        EXPECT_EQ(edges[1].polarity, EdgePolarity::Negative);
    }
}

TEST_F(Edge1DTest, DetectEdges1DPolarityFilter) {
    // Detect only positive edges
    auto positive = DetectEdges1D(doubleEdge_.data(), doubleEdge_.size(),
                                   10.0, EdgePolarity::Positive, 0.0);
    EXPECT_EQ(positive.size(), 1u);
    if (!positive.empty()) {
        EXPECT_EQ(positive[0].polarity, EdgePolarity::Positive);
    }

    // Detect only negative edges
    auto negative = DetectEdges1D(doubleEdge_.data(), doubleEdge_.size(),
                                   10.0, EdgePolarity::Negative, 0.0);
    EXPECT_EQ(negative.size(), 1u);
    if (!negative.empty()) {
        EXPECT_EQ(negative[0].polarity, EdgePolarity::Negative);
    }
}

TEST_F(Edge1DTest, DetectSingleEdge1DFirst) {
    bool found;
    auto edge = DetectSingleEdge1D(doubleEdge_.data(), doubleEdge_.size(),
                                    10.0, EdgePolarity::Both, EdgeSelect::First,
                                    0.0, found);

    EXPECT_TRUE(found);
    EXPECT_NEAR(edge.position, 30.0, 0.5);
}

TEST_F(Edge1DTest, DetectSingleEdge1DLast) {
    bool found;
    auto edge = DetectSingleEdge1D(doubleEdge_.data(), doubleEdge_.size(),
                                    10.0, EdgePolarity::Both, EdgeSelect::Last,
                                    0.0, found);

    EXPECT_TRUE(found);
    EXPECT_NEAR(edge.position, 70.0, 0.5);
}

TEST_F(Edge1DTest, DetectSingleEdge1DStrongest) {
    bool found;
    auto edge = DetectSingleEdge1D(stepProfile_.data(), stepProfile_.size(),
                                    10.0, EdgePolarity::Both, EdgeSelect::Strongest,
                                    0.0, found);

    EXPECT_TRUE(found);
    EXPECT_NEAR(edge.position, 50.0, 0.5);
}

TEST_F(Edge1DTest, DetectSingleEdge1DNotFound) {
    // High threshold that won't match any edge
    bool found;
    auto edge = DetectSingleEdge1D(stepProfile_.data(), stepProfile_.size(),
                                    1000.0, EdgePolarity::Both, EdgeSelect::First,
                                    0.0, found);

    EXPECT_FALSE(found);
}

// ============================================================================
// Edge Pair Detection Tests
// ============================================================================

TEST_F(Edge1DTest, DetectEdgePairs1DAll) {
    auto pairs = DetectEdgePairs1D(doubleEdge_.data(), doubleEdge_.size(),
                                    10.0, EdgePairSelect::All, 0.0);

    EXPECT_EQ(pairs.size(), 1u);
    if (!pairs.empty()) {
        EXPECT_NEAR(pairs[0].first.position, 30.0, 0.5);
        EXPECT_NEAR(pairs[0].second.position, 70.0, 0.5);
        EXPECT_NEAR(pairs[0].distance, 40.0, 1.0);
    }
}

TEST_F(Edge1DTest, DetectEdgePairs1DFirstLast) {
    auto pairs = DetectEdgePairs1D(doubleEdge_.data(), doubleEdge_.size(),
                                    10.0, EdgePairSelect::FirstLast, 0.0);

    EXPECT_EQ(pairs.size(), 1u);
}

TEST_F(Edge1DTest, DetectEdgePairs1DBestPair) {
    auto pairs = DetectEdgePairs1D(doubleEdge_.data(), doubleEdge_.size(),
                                    10.0, EdgePairSelect::BestPair, 0.0);

    EXPECT_EQ(pairs.size(), 1u);
}

TEST_F(Edge1DTest, DetectEdgePairs1DClosest) {
    auto pairs = DetectEdgePairs1D(doubleEdge_.data(), doubleEdge_.size(),
                                    10.0, EdgePairSelect::Closest, 0.0);

    EXPECT_EQ(pairs.size(), 1u);
}

TEST_F(Edge1DTest, DetectSinglePair1D) {
    bool found;
    auto pair = DetectSinglePair1D(doubleEdge_.data(), doubleEdge_.size(),
                                    10.0, EdgePairSelect::BestPair, 0.0, found);

    EXPECT_TRUE(found);
    EXPECT_NEAR(pair.distance, 40.0, 1.0);
}

TEST_F(Edge1DTest, DetectEdgePairs1DNoNegative) {
    // Profile with only positive edge
    auto pairs = DetectEdgePairs1D(stepProfile_.data(), stepProfile_.size(),
                                    10.0, EdgePairSelect::All, 0.0);

    // Should find no pairs (need both positive and negative)
    EXPECT_TRUE(pairs.empty());
}

// ============================================================================
// Profile Extraction Tests
// ============================================================================

TEST_F(Edge1DTest, ExtractProfileHorizontal) {
    // Create horizontal gradient image
    std::vector<uint8_t> img(100 * 10);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 100; ++x) {
            img[y * 100 + x] = static_cast<uint8_t>(x * 2.5);  // 0-250
        }
    }

    std::vector<double> profile;
    ExtractProfile(img.data(), 100, 10, 0.0, 5.0, 99.0, 5.0, profile);

    EXPECT_EQ(profile.size(), 100u);
    EXPECT_NEAR(profile[0], 0.0, 1.0);
    EXPECT_NEAR(profile[99], 247.5, 2.0);
}

TEST_F(Edge1DTest, ExtractProfileDiagonal) {
    // Simple image for diagonal test
    std::vector<uint8_t> img(10 * 10, 128);

    std::vector<double> profile;
    ExtractProfile(img.data(), 10, 10, 0.0, 0.0, 9.0, 9.0, profile);

    // Should have ~14 samples (sqrt(81+81) ≈ 12.7 -> 13+1)
    EXPECT_GE(profile.size(), 13u);
    EXPECT_LE(profile.size(), 15u);

    // All values should be 128
    for (double val : profile) {
        EXPECT_NEAR(val, 128.0, 1.0);
    }
}

TEST_F(Edge1DTest, ExtractProfileFixedSamples) {
    std::vector<uint8_t> img(100 * 10, 100);

    std::vector<double> profile;
    ExtractProfile(img.data(), 100, 10, 0.0, 5.0, 50.0, 5.0, profile, 25);

    EXPECT_EQ(profile.size(), 25u);
}

TEST_F(Edge1DTest, ExtractPerpendicularProfile) {
    // Image with vertical edge at x=50
    std::vector<uint8_t> img(100 * 100);
    for (int y = 0; y < 100; ++y) {
        for (int x = 0; x < 100; ++x) {
            img[y * 100 + x] = (x < 50) ? 0 : 255;
        }
    }

    std::vector<double> profile;
    ExtractPerpendicularProfile(img.data(), 100, 100,
                                 50.0, 50.0,  // Center on edge
                                 M_PI / 2.0,  // Vertical line
                                 20.0,        // Profile length
                                 profile, 21);

    EXPECT_EQ(profile.size(), 21u);

    // Profile should cross from 0 to 255
    EXPECT_LT(profile[0], 50.0);
    EXPECT_GT(profile[20], 200.0);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Edge1DTest, EmptyProfile) {
    std::vector<double> empty;
    auto edges = DetectEdges1D(empty.data(), 0, 10.0);
    EXPECT_TRUE(edges.empty());
}

TEST_F(Edge1DTest, ShortProfile) {
    std::vector<double> shortProf = {0.0, 255.0};
    auto edges = DetectEdges1D(shortProf.data(), shortProf.size(), 10.0);
    // Too short for internal peaks
    EXPECT_TRUE(edges.empty());
}

TEST_F(Edge1DTest, ConstantProfile) {
    std::vector<double> constant(100, 128.0);
    auto edges = DetectEdges1D(constant.data(), constant.size(), 1.0);
    EXPECT_TRUE(edges.empty());
}

TEST_F(Edge1DTest, NoisyProfile) {
    // Add noise to step profile
    std::vector<double> noisy = stepProfile_;
    for (size_t i = 0; i < noisy.size(); ++i) {
        noisy[i] += (i % 5) * 2.0 - 5.0;  // Small noise
    }

    // With smoothing, should still detect main edge
    auto edges = DetectEdges1D(noisy.data(), noisy.size(), 10.0,
                                EdgePolarity::Both, 2.0);

    EXPECT_EQ(edges.size(), 1u);
    if (!edges.empty()) {
        EXPECT_NEAR(edges[0].position, 50.0, 2.0);
    }
}

// ============================================================================
// Precision Tests
// ============================================================================

TEST_F(Edge1DTest, SubpixelPrecisionGaussian) {
    // Create edge at subpixel position 50.3
    std::vector<double> subpixelEdge(100);
    double edgePos = 50.3;
    for (size_t i = 0; i < 100; ++i) {
        double t = static_cast<double>(i) - edgePos;
        subpixelEdge[i] = 255.0 / (1.0 + std::exp(-t));
    }

    auto edges = DetectEdges1D(subpixelEdge.data(), subpixelEdge.size(),
                                5.0, EdgePolarity::Both, 1.0);

    EXPECT_EQ(edges.size(), 1u);
    if (!edges.empty()) {
        // Should be accurate to within 0.2 pixels
        EXPECT_NEAR(edges[0].position, edgePos, 0.2);
    }
}

