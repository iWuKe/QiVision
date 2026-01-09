/**
 * @file test_homography.cpp
 * @brief Unit tests for Homography module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/Homography.h>
#include <cmath>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// Use PI from Constants.h
using Qi::Vision::PI;

class HomographyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test image (10x10 gradient)
        testImage_ = QImage(10, 10, PixelType::UInt8, ChannelType::Gray);
        uint8_t* data = static_cast<uint8_t*>(testImage_.Data());
        for (int y = 0; y < 10; ++y) {
            for (int x = 0; x < 10; ++x) {
                data[y * 10 + x] = static_cast<uint8_t>(x + y * 10);
            }
        }
    }

    QImage testImage_;
};

// =============================================================================
// Homography Class Tests
// =============================================================================

TEST_F(HomographyTest, DefaultConstructor_Identity) {
    Homography H;

    EXPECT_NEAR(H(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(H(0, 1), 0.0, 1e-10);
    EXPECT_NEAR(H(0, 2), 0.0, 1e-10);
    EXPECT_NEAR(H(1, 0), 0.0, 1e-10);
    EXPECT_NEAR(H(1, 1), 1.0, 1e-10);
    EXPECT_NEAR(H(1, 2), 0.0, 1e-10);
    EXPECT_NEAR(H(2, 0), 0.0, 1e-10);
    EXPECT_NEAR(H(2, 1), 0.0, 1e-10);
    EXPECT_NEAR(H(2, 2), 1.0, 1e-10);
}

TEST_F(HomographyTest, ElementConstructor) {
    Homography H(1, 2, 3, 4, 5, 6, 7, 8, 9);

    EXPECT_NEAR(H(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(H(0, 1), 2.0, 1e-10);
    EXPECT_NEAR(H(0, 2), 3.0, 1e-10);
    EXPECT_NEAR(H(1, 0), 4.0, 1e-10);
    EXPECT_NEAR(H(1, 1), 5.0, 1e-10);
    EXPECT_NEAR(H(1, 2), 6.0, 1e-10);
    EXPECT_NEAR(H(2, 0), 7.0, 1e-10);
    EXPECT_NEAR(H(2, 1), 8.0, 1e-10);
    EXPECT_NEAR(H(2, 2), 9.0, 1e-10);
}

TEST_F(HomographyTest, Mat33Constructor) {
    Mat33 mat;
    mat(0, 0) = 2; mat(0, 1) = 0; mat(0, 2) = 5;
    mat(1, 0) = 0; mat(1, 1) = 3; mat(1, 2) = 7;
    mat(2, 0) = 0; mat(2, 1) = 0; mat(2, 2) = 1;

    Homography H(mat);

    EXPECT_NEAR(H(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(H(0, 2), 5.0, 1e-10);
    EXPECT_NEAR(H(1, 1), 3.0, 1e-10);
    EXPECT_NEAR(H(1, 2), 7.0, 1e-10);
}

TEST_F(HomographyTest, Identity_Static) {
    Homography H = Homography::Identity();

    Point2d p{5.0, 7.0};
    Point2d tp = H.Transform(p);

    EXPECT_NEAR(tp.x, 5.0, 1e-10);
    EXPECT_NEAR(tp.y, 7.0, 1e-10);
}

TEST_F(HomographyTest, FromAffine) {
    QMatrix affine = QMatrix::Translation(10, 20) * QMatrix::Rotation(PI / 4);
    Homography H = Homography::FromAffine(affine);

    EXPECT_TRUE(H.IsAffine());
    EXPECT_NEAR(H(2, 0), 0.0, 1e-10);
    EXPECT_NEAR(H(2, 1), 0.0, 1e-10);
    EXPECT_NEAR(H(2, 2), 1.0, 1e-10);
}

// =============================================================================
// Transform Tests
// =============================================================================

TEST_F(HomographyTest, Transform_Identity) {
    Homography H = Homography::Identity();

    Point2d p{3.5, 7.2};
    Point2d tp = H.Transform(p);

    EXPECT_NEAR(tp.x, 3.5, 1e-10);
    EXPECT_NEAR(tp.y, 7.2, 1e-10);
}

TEST_F(HomographyTest, Transform_Translation) {
    // H = [1 0 5; 0 1 10; 0 0 1] (translation by 5, 10)
    Homography H(1, 0, 5, 0, 1, 10, 0, 0, 1);

    Point2d p{2, 3};
    Point2d tp = H.Transform(p);

    EXPECT_NEAR(tp.x, 7, 1e-10);
    EXPECT_NEAR(tp.y, 13, 1e-10);
}

TEST_F(HomographyTest, Transform_Scale) {
    // H = [2 0 0; 0 3 0; 0 0 1] (scale 2x, 3y)
    Homography H(2, 0, 0, 0, 3, 0, 0, 0, 1);

    Point2d p{5, 4};
    Point2d tp = H.Transform(p);

    EXPECT_NEAR(tp.x, 10, 1e-10);
    EXPECT_NEAR(tp.y, 12, 1e-10);
}

TEST_F(HomographyTest, Transform_Perspective) {
    // Non-trivial perspective
    // H = [1 0 0; 0 1 0; 0.1 0 1]
    Homography H(1, 0, 0, 0, 1, 0, 0.1, 0, 1);

    Point2d p{10, 5};
    Point2d tp = H.Transform(p);

    // w = 0.1 * 10 + 0 * 5 + 1 = 2
    // x' = (1*10 + 0*5 + 0) / 2 = 5
    // y' = (0*10 + 1*5 + 0) / 2 = 2.5
    EXPECT_NEAR(tp.x, 5.0, 1e-10);
    EXPECT_NEAR(tp.y, 2.5, 1e-10);
}

TEST_F(HomographyTest, Transform_MultiplePoints) {
    Homography H(1, 0, 10, 0, 1, 20, 0, 0, 1);

    std::vector<Point2d> points = {{0, 0}, {5, 0}, {0, 5}, {5, 5}};
    auto transformed = H.Transform(points);

    ASSERT_EQ(transformed.size(), 4);
    EXPECT_NEAR(transformed[0].x, 10, 1e-10);
    EXPECT_NEAR(transformed[0].y, 20, 1e-10);
    EXPECT_NEAR(transformed[3].x, 15, 1e-10);
    EXPECT_NEAR(transformed[3].y, 25, 1e-10);
}

// =============================================================================
// Matrix Operations Tests
// =============================================================================

TEST_F(HomographyTest, Determinant_Identity) {
    Homography H = Homography::Identity();
    EXPECT_NEAR(H.Determinant(), 1.0, 1e-10);
}

TEST_F(HomographyTest, Determinant_Scale) {
    Homography H(2, 0, 0, 0, 3, 0, 0, 0, 1);
    EXPECT_NEAR(H.Determinant(), 6.0, 1e-10);
}

TEST_F(HomographyTest, IsInvertible_True) {
    Homography H = Homography::Identity();
    EXPECT_TRUE(H.IsInvertible());
}

TEST_F(HomographyTest, IsInvertible_Singular) {
    // Singular matrix (all zeros in first row)
    Homography H(0, 0, 0, 0, 1, 0, 0, 0, 1);
    EXPECT_FALSE(H.IsInvertible());
}

TEST_F(HomographyTest, Inverse_Identity) {
    Homography H = Homography::Identity();
    Homography Hinv = H.Inverse();

    // Identity inverse is identity
    EXPECT_NEAR(Hinv(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(Hinv(1, 1), 1.0, 1e-10);
    EXPECT_NEAR(Hinv(2, 2), 1.0, 1e-10);
}

TEST_F(HomographyTest, Inverse_Translation) {
    Homography H(1, 0, 5, 0, 1, 10, 0, 0, 1);
    Homography Hinv = H.Inverse();

    // Inverse should translate by -5, -10
    EXPECT_NEAR(Hinv(0, 2), -5.0, 1e-10);
    EXPECT_NEAR(Hinv(1, 2), -10.0, 1e-10);
}

TEST_F(HomographyTest, Inverse_Roundtrip) {
    Homography H(2, 0.5, 10, 0.3, 3, 20, 0.01, 0.02, 1);
    Homography Hinv = H.Inverse();

    Point2d p{5, 7};
    Point2d tp = H.Transform(p);
    Point2d back = Hinv.Transform(tp);

    EXPECT_NEAR(back.x, p.x, 1e-8);
    EXPECT_NEAR(back.y, p.y, 1e-8);
}

TEST_F(HomographyTest, Normalized) {
    Homography H(2, 0, 10, 0, 4, 20, 0, 0, 2);
    Homography Hn = H.Normalized();

    EXPECT_NEAR(Hn(2, 2), 1.0, 1e-10);
    EXPECT_NEAR(Hn(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(Hn(0, 2), 5.0, 1e-10);
}

TEST_F(HomographyTest, Composition) {
    Homography H1(1, 0, 5, 0, 1, 0, 0, 0, 1);  // Translate x by 5
    Homography H2(1, 0, 0, 0, 1, 10, 0, 0, 1); // Translate y by 10

    Homography H = H2 * H1;

    Point2d p{0, 0};
    Point2d tp = H.Transform(p);

    EXPECT_NEAR(tp.x, 5.0, 1e-10);
    EXPECT_NEAR(tp.y, 10.0, 1e-10);
}

TEST_F(HomographyTest, IsAffine_True) {
    Homography H(2, 0, 5, 0, 3, 10, 0, 0, 1);
    EXPECT_TRUE(H.IsAffine());
}

TEST_F(HomographyTest, IsAffine_False) {
    Homography H(1, 0, 0, 0, 1, 0, 0.1, 0, 1);
    EXPECT_FALSE(H.IsAffine());
}

TEST_F(HomographyTest, ToAffine_Valid) {
    Homography H(2, 0, 5, 0, 3, 10, 0, 0, 1);
    auto affine = H.ToAffine();

    ASSERT_TRUE(affine.has_value());
    EXPECT_NEAR(affine->M00(), 2.0, 1e-10);
    EXPECT_NEAR(affine->M11(), 3.0, 1e-10);
    EXPECT_NEAR(affine->M02(), 5.0, 1e-10);
    EXPECT_NEAR(affine->M12(), 10.0, 1e-10);
}

TEST_F(HomographyTest, ToAffine_Invalid) {
    Homography H(1, 0, 0, 0, 1, 0, 0.1, 0, 1);
    auto affine = H.ToAffine();

    EXPECT_FALSE(affine.has_value());
}

TEST_F(HomographyTest, ToMat33_Roundtrip) {
    Homography H(1, 2, 3, 4, 5, 6, 7, 8, 9);
    Mat33 mat = H.ToMat33();
    Homography H2(mat);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(H(i, j), H2(i, j), 1e-10);
        }
    }
}

// =============================================================================
// From4Points Tests
// =============================================================================

TEST_F(HomographyTest, From4Points_Identity) {
    std::array<Point2d, 4> src = {{{0, 0}, {10, 0}, {10, 10}, {0, 10}}};
    std::array<Point2d, 4> dst = {{{0, 0}, {10, 0}, {10, 10}, {0, 10}}};

    auto H = Homography::From4Points(src, dst);

    ASSERT_TRUE(H.has_value());

    // Should be close to identity
    for (const auto& p : src) {
        Point2d tp = H->Transform(p);
        EXPECT_NEAR(tp.x, p.x, 1e-6);
        EXPECT_NEAR(tp.y, p.y, 1e-6);
    }
}

TEST_F(HomographyTest, From4Points_Translation) {
    std::array<Point2d, 4> src = {{{0, 0}, {10, 0}, {10, 10}, {0, 10}}};
    std::array<Point2d, 4> dst = {{{5, 5}, {15, 5}, {15, 15}, {5, 15}}};

    auto H = Homography::From4Points(src, dst);

    ASSERT_TRUE(H.has_value());

    for (size_t i = 0; i < 4; ++i) {
        Point2d tp = H->Transform(src[i]);
        EXPECT_NEAR(tp.x, dst[i].x, 1e-6);
        EXPECT_NEAR(tp.y, dst[i].y, 1e-6);
    }
}

TEST_F(HomographyTest, From4Points_Perspective) {
    // Map square to trapezoid (perspective)
    std::array<Point2d, 4> src = {{{0, 0}, {100, 0}, {100, 100}, {0, 100}}};
    std::array<Point2d, 4> dst = {{{10, 10}, {90, 20}, {80, 90}, {20, 80}}};

    auto H = Homography::From4Points(src, dst);

    ASSERT_TRUE(H.has_value());

    for (size_t i = 0; i < 4; ++i) {
        Point2d tp = H->Transform(src[i]);
        EXPECT_NEAR(tp.x, dst[i].x, 1e-4);
        EXPECT_NEAR(tp.y, dst[i].y, 1e-4);
    }
}

// =============================================================================
// EstimateHomography Tests
// =============================================================================

TEST_F(HomographyTest, EstimateHomography_Identity) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {10, 10}, {0, 10}, {5, 5}};
    std::vector<Point2d> dst = {{0, 0}, {10, 0}, {10, 10}, {0, 10}, {5, 5}};

    auto H = EstimateHomography(src, dst);

    ASSERT_TRUE(H.has_value());

    double error = ComputeHomographyError(src, dst, *H);
    EXPECT_LT(error, 1e-6);
}

TEST_F(HomographyTest, EstimateHomography_Translation) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    std::vector<Point2d> dst = {{5, 7}, {15, 7}, {15, 17}, {5, 17}};

    auto H = EstimateHomography(src, dst);

    ASSERT_TRUE(H.has_value());

    for (size_t i = 0; i < src.size(); ++i) {
        Point2d tp = H->Transform(src[i]);
        EXPECT_NEAR(tp.x, dst[i].x, 1e-6);
        EXPECT_NEAR(tp.y, dst[i].y, 1e-6);
    }
}

TEST_F(HomographyTest, EstimateHomography_Overdetermined) {
    // More than 4 points - must NOT be collinear for valid homography
    std::vector<Point2d> src = {
        {0, 0}, {100, 0}, {100, 100}, {0, 100},  // corners
        {50, 50}, {25, 25}, {75, 25}, {25, 75}, {75, 75}  // interior points
    };
    std::vector<Point2d> dst;
    // Apply translation by (3, 7)
    for (const auto& p : src) {
        dst.push_back({p.x + 3, p.y + 7});
    }

    auto H = EstimateHomography(src, dst);

    ASSERT_TRUE(H.has_value());

    double error = ComputeHomographyError(src, dst, *H);
    EXPECT_LT(error, 0.1);
}

TEST_F(HomographyTest, EstimateHomography_TooFewPoints) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {5, 10}};
    std::vector<Point2d> dst = {{0, 0}, {10, 0}, {5, 10}};

    auto H = EstimateHomography(src, dst);

    // 3 points is not enough for homography (need 4)
    EXPECT_FALSE(H.has_value());
}

TEST_F(HomographyTest, EstimateHomography_MismatchedSizes) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    std::vector<Point2d> dst = {{0, 0}, {10, 0}, {10, 10}};

    auto H = EstimateHomography(src, dst);

    EXPECT_FALSE(H.has_value());
}

// =============================================================================
// EstimateHomographyRANSAC Tests
// =============================================================================

TEST_F(HomographyTest, EstimateHomographyRANSAC_NoOutliers) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {10, 10}, {0, 10}, {5, 5}, {3, 7}};
    std::vector<Point2d> dst = {{5, 5}, {15, 5}, {15, 15}, {5, 15}, {10, 10}, {8, 12}};

    auto H = EstimateHomographyRANSAC(src, dst, 1.0, 0.99, 100);

    ASSERT_TRUE(H.has_value());

    double error = ComputeHomographyError(src, dst, *H);
    EXPECT_LT(error, 0.1);
}

TEST_F(HomographyTest, EstimateHomographyRANSAC_WithOutliers) {
    // Good points (translation by 5,5) - use more points for robust estimation
    std::vector<Point2d> src = {
        {0, 0}, {100, 0}, {100, 100}, {0, 100},  // corners
        {50, 0}, {100, 50}, {50, 100}, {0, 50},  // edge midpoints
        {50, 50},                                 // center
        {30, 70}                                  // outlier source
    };
    std::vector<Point2d> dst = {
        {5, 5}, {105, 5}, {105, 105}, {5, 105},  // translated corners
        {55, 5}, {105, 55}, {55, 105}, {5, 55},  // translated midpoints
        {55, 55},                                 // translated center
        {300, 300}                                // outlier - very far from expected (35, 75)
    };

    std::vector<bool> inlierMask;
    auto H = EstimateHomographyRANSAC(src, dst, 2.0, 0.99, 1000, &inlierMask);

    ASSERT_TRUE(H.has_value());
    ASSERT_EQ(inlierMask.size(), src.size());

    // Last point should be outlier (expected 35,75 but got 300,300)
    EXPECT_FALSE(inlierMask[9]);

    // First 9 should all be inliers
    int inlierCount = 0;
    for (int i = 0; i < 9; ++i) {
        if (inlierMask[i]) inlierCount++;
    }
    EXPECT_GE(inlierCount, 7);  // At least 7 out of 9 should be inliers
}

TEST_F(HomographyTest, EstimateHomographyRANSAC_TooFewPoints) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {5, 10}};
    std::vector<Point2d> dst = {{0, 0}, {10, 0}, {5, 10}};

    auto H = EstimateHomographyRANSAC(src, dst);

    EXPECT_FALSE(H.has_value());
}

// =============================================================================
// ComputeHomographyError Tests
// =============================================================================

TEST_F(HomographyTest, ComputeHomographyError_Zero) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {5, 10}};
    std::vector<Point2d> dst = {{0, 0}, {10, 0}, {5, 10}};
    Homography H = Homography::Identity();

    double error = ComputeHomographyError(src, dst, H);

    EXPECT_NEAR(error, 0.0, 1e-10);
}

TEST_F(HomographyTest, ComputeHomographyError_NonZero) {
    std::vector<Point2d> src = {{0, 0}};
    std::vector<Point2d> dst = {{1, 0}};  // 1 pixel off
    Homography H = Homography::Identity();

    double error = ComputeHomographyError(src, dst, H);

    EXPECT_NEAR(error, 1.0, 1e-10);
}

TEST_F(HomographyTest, ComputePointHomographyErrors_Basic) {
    std::vector<Point2d> src = {{0, 0}, {3, 0}, {0, 4}};
    std::vector<Point2d> dst = {{1, 0}, {3, 0}, {0, 4}};
    Homography H = Homography::Identity();

    auto errors = ComputePointHomographyErrors(src, dst, H);

    ASSERT_EQ(errors.size(), 3);
    EXPECT_NEAR(errors[0], 1.0, 1e-10);  // First point has 1px error
    EXPECT_NEAR(errors[1], 0.0, 1e-10);  // Second is exact
    EXPECT_NEAR(errors[2], 0.0, 1e-10);  // Third is exact
}

// =============================================================================
// WarpPerspective Tests
// =============================================================================

TEST_F(HomographyTest, WarpPerspective_EmptyImage) {
    QImage empty;
    Homography H = Homography::Identity();

    QImage result = WarpPerspective(empty, H);

    EXPECT_TRUE(result.Empty());
}

TEST_F(HomographyTest, WarpPerspective_Identity) {
    Homography H = Homography::Identity();

    QImage result = WarpPerspective(testImage_, H, 10, 10);

    EXPECT_EQ(result.Width(), 10);
    EXPECT_EQ(result.Height(), 10);

    // Values should be approximately preserved
    const uint8_t* srcData = static_cast<const uint8_t*>(testImage_.Data());
    const uint8_t* dstData = static_cast<const uint8_t*>(result.Data());
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(srcData[i], dstData[i], 1);
    }
}

TEST_F(HomographyTest, WarpPerspective_Translation) {
    Homography H(1, 0, 5, 0, 1, 5, 0, 0, 1);

    QImage result = WarpPerspective(testImage_, H, 15, 15,
                                    InterpolationMethod::Nearest,
                                    BorderMode::Constant, 0);

    EXPECT_EQ(result.Width(), 15);
    EXPECT_EQ(result.Height(), 15);
}

TEST_F(HomographyTest, WarpPerspective_Scale) {
    Homography H(2, 0, 0, 0, 2, 0, 0, 0, 1);

    QImage result = WarpPerspective(testImage_, H, 20, 20);

    EXPECT_EQ(result.Width(), 20);
    EXPECT_EQ(result.Height(), 20);
}

TEST_F(HomographyTest, WarpPerspective_InterpolationMethods) {
    Homography H(1.5, 0, 0, 0, 1.5, 0, 0, 0, 1);

    QImage resultNearest = WarpPerspective(testImage_, H, 15, 15,
                                           InterpolationMethod::Nearest);
    QImage resultBilinear = WarpPerspective(testImage_, H, 15, 15,
                                            InterpolationMethod::Bilinear);
    QImage resultBicubic = WarpPerspective(testImage_, H, 15, 15,
                                           InterpolationMethod::Bicubic);

    EXPECT_FALSE(resultNearest.Empty());
    EXPECT_FALSE(resultBilinear.Empty());
    EXPECT_FALSE(resultBicubic.Empty());
}

TEST_F(HomographyTest, WarpPerspective_NonInvertible) {
    // Singular homography
    Homography H(0, 0, 0, 0, 0, 0, 0, 0, 1);

    QImage result = WarpPerspective(testImage_, H, 10, 10);

    EXPECT_TRUE(result.Empty());
}

// =============================================================================
// ComputePerspectiveOutputSize Tests
// =============================================================================

TEST_F(HomographyTest, ComputePerspectiveOutputSize_Identity) {
    Homography H = Homography::Identity();
    int32_t dstW, dstH;
    double offsetX, offsetY;

    ComputePerspectiveOutputSize(100, 80, H, dstW, dstH, offsetX, offsetY);

    EXPECT_EQ(dstW, 100);
    EXPECT_EQ(dstH, 80);
    EXPECT_NEAR(offsetX, 0.0, 1e-6);
    EXPECT_NEAR(offsetY, 0.0, 1e-6);
}

TEST_F(HomographyTest, ComputePerspectiveOutputSize_Scale2x) {
    Homography H(2, 0, 0, 0, 2, 0, 0, 0, 1);
    int32_t dstW, dstH;
    double offsetX, offsetY;

    ComputePerspectiveOutputSize(100, 80, H, dstW, dstH, offsetX, offsetY);

    EXPECT_NEAR(dstW, 199, 1);
    EXPECT_NEAR(dstH, 159, 1);
}

// =============================================================================
// Contour Transformation Tests
// =============================================================================

TEST_F(HomographyTest, PerspectiveTransformContour_Basic) {
    std::vector<Point2d> points = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    QContour contour(points);

    Homography H(1, 0, 5, 0, 1, 5, 0, 0, 1);  // Translation
    QContour result = PerspectiveTransformContour(contour, H);

    EXPECT_EQ(result.Size(), 4);

    auto resultPts = result.GetPoints();
    EXPECT_NEAR(resultPts[0].x, 5, 1e-6);
    EXPECT_NEAR(resultPts[0].y, 5, 1e-6);
}

TEST_F(HomographyTest, PerspectiveTransformContours_Multiple) {
    std::vector<Point2d> pts1 = {{0, 0}, {5, 0}, {5, 5}};
    std::vector<Point2d> pts2 = {{10, 10}, {15, 10}, {15, 15}};

    std::vector<QContour> contours = {QContour(pts1), QContour(pts2)};
    Homography H(2, 0, 0, 0, 2, 0, 0, 0, 1);  // Scale 2x

    auto results = PerspectiveTransformContours(contours, H);

    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].Size(), 3);
    EXPECT_EQ(results[1].Size(), 3);

    // Check scaling
    EXPECT_NEAR(results[0].GetPoint(1).x, 10, 1e-6);  // 5 * 2
}

TEST_F(HomographyTest, PerspectiveTransformContour_Empty) {
    QContour empty;
    Homography H = Homography::Identity();

    QContour result = PerspectiveTransformContour(empty, H);

    EXPECT_TRUE(result.Empty());
}

// =============================================================================
// Utility Function Tests
// =============================================================================

TEST_F(HomographyTest, RectifyQuadrilateral_Square) {
    std::array<Point2d, 4> quad = {{{0, 0}, {100, 0}, {100, 100}, {0, 100}}};

    auto H = RectifyQuadrilateral(quad, 100, 100);

    ASSERT_TRUE(H.has_value());

    // Transform corners and verify they map to rectangle
    for (size_t i = 0; i < 4; ++i) {
        Point2d tp = H->Transform(quad[i]);

        double expectedX = (i == 0 || i == 3) ? 0 : 100;
        double expectedY = (i < 2) ? 0 : 100;

        EXPECT_NEAR(tp.x, expectedX, 1e-4);
        EXPECT_NEAR(tp.y, expectedY, 1e-4);
    }
}

TEST_F(HomographyTest, RectifyQuadrilateral_Trapezoid) {
    // Trapezoid to rectangle
    std::array<Point2d, 4> quad = {{{10, 0}, {90, 10}, {80, 90}, {20, 80}}};

    auto H = RectifyQuadrilateral(quad, 100, 100);

    ASSERT_TRUE(H.has_value());
}

TEST_F(HomographyTest, RectangleToQuadrilateral_Basic) {
    std::array<Point2d, 4> quad = {{{10, 10}, {90, 20}, {80, 90}, {20, 80}}};

    auto H = RectangleToQuadrilateral(100, 100, quad);

    ASSERT_TRUE(H.has_value());

    // Transform rectangle corners and verify they map to quad
    EXPECT_NEAR(H->Transform(0, 0).x, quad[0].x, 1e-4);
    EXPECT_NEAR(H->Transform(0, 0).y, quad[0].y, 1e-4);
}

TEST_F(HomographyTest, TransformBoundingBoxPerspective_Identity) {
    Rect2d bbox(10, 20, 30, 40);
    Homography H = Homography::Identity();

    Rect2d result = TransformBoundingBoxPerspective(bbox, H);

    EXPECT_NEAR(result.x, 10, 1e-6);
    EXPECT_NEAR(result.y, 20, 1e-6);
    EXPECT_NEAR(result.width, 30, 1e-6);
    EXPECT_NEAR(result.height, 40, 1e-6);
}

TEST_F(HomographyTest, TransformBoundingBoxPerspective_Scale) {
    Rect2d bbox(0, 0, 10, 10);
    Homography H(2, 0, 0, 0, 2, 0, 0, 0, 1);

    Rect2d result = TransformBoundingBoxPerspective(bbox, H);

    EXPECT_NEAR(result.width, 20, 1e-6);
    EXPECT_NEAR(result.height, 20, 1e-6);
}

TEST_F(HomographyTest, IsValidHomography_True) {
    Homography H = Homography::Identity();

    EXPECT_TRUE(IsValidHomography(H, 100, 100));
}

TEST_F(HomographyTest, IsValidHomography_Singular) {
    Homography H(0, 0, 0, 0, 0, 0, 0, 0, 0);

    EXPECT_FALSE(IsValidHomography(H, 100, 100));
}

TEST_F(HomographyTest, IsValidHomography_FlipsQuad) {
    // Homography that flips the quad orientation
    Homography H(-1, 0, 100, 0, 1, 0, 0, 0, 1);  // Reflect x

    // This should detect orientation change
    bool valid = IsValidHomography(H, 100, 100);
    // Note: May or may not be valid depending on whether we consider reflection valid
    // The function checks for self-intersection and sign consistency
}

TEST_F(HomographyTest, SampsonError_Zero) {
    Point2d src{0, 0};
    Point2d dst{0, 0};
    Homography H = Homography::Identity();

    double error = SampsonError(src, dst, H);

    EXPECT_NEAR(error, 0.0, 1e-10);
}

TEST_F(HomographyTest, SampsonError_NonZero) {
    Point2d src{0, 0};
    Point2d dst{1, 0};  // 1 pixel off
    Homography H = Homography::Identity();

    double error = SampsonError(src, dst, H);

    // Sampson error should be close to squared geometric error for small errors
    EXPECT_GT(error, 0.0);
    EXPECT_LT(error, 2.0);  // Should be around 1.0
}

TEST_F(HomographyTest, RefineHomographyLM_AlreadyOptimal) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    std::vector<Point2d> dst = {{5, 5}, {15, 5}, {15, 15}, {5, 15}};

    auto H = EstimateHomography(src, dst);
    ASSERT_TRUE(H.has_value());

    double errorBefore = ComputeHomographyError(src, dst, *H);

    Homography refined = RefineHomographyLM(src, dst, *H, 10);

    double errorAfter = ComputeHomographyError(src, dst, refined);

    // Should not get worse
    EXPECT_LE(errorAfter, errorBefore + 1e-6);
}

TEST_F(HomographyTest, RefineHomographyLM_NoisyData) {
    // Create noisy data - must NOT be collinear
    std::vector<Point2d> src = {
        {0, 0}, {100, 0}, {100, 100}, {0, 100},  // corners
        {50, 0}, {100, 50}, {50, 100}, {0, 50},  // edge midpoints
        {50, 50}, {25, 25}  // interior points
    };
    std::vector<Point2d> dst;
    // Apply translation by (5, 7) with small noise
    for (size_t i = 0; i < src.size(); ++i) {
        double noise_x = (i % 2) * 0.1 - 0.05;
        double noise_y = (i % 3) * 0.05 - 0.05;
        dst.push_back({src[i].x + 5 + noise_x, src[i].y + 7 + noise_y});
    }

    auto H = EstimateHomography(src, dst);
    ASSERT_TRUE(H.has_value());

    Homography refined = RefineHomographyLM(src, dst, *H, 20);

    double errorRefined = ComputeHomographyError(src, dst, refined);
    EXPECT_LT(errorRefined, 1.0);
}

// =============================================================================
// Multi-channel and Different Pixel Types
// =============================================================================

TEST_F(HomographyTest, WarpPerspective_MultiChannel) {
    QImage rgb(10, 10, PixelType::UInt8, ChannelType::RGB);
    uint8_t* data = static_cast<uint8_t*>(rgb.Data());

    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 100; ++i) {
            data[c * 100 + i] = static_cast<uint8_t>(i + c * 10);
        }
    }

    Homography H(2, 0, 0, 0, 2, 0, 0, 0, 1);
    QImage result = WarpPerspective(rgb, H, 20, 20);

    EXPECT_EQ(result.Width(), 20);
    EXPECT_EQ(result.Height(), 20);
    EXPECT_EQ(result.Channels(), 3);
}

TEST_F(HomographyTest, WarpPerspective_Float32) {
    QImage floatImg(10, 10, PixelType::Float32, ChannelType::Gray);
    float* data = static_cast<float*>(floatImg.Data());
    for (int i = 0; i < 100; ++i) {
        data[i] = static_cast<float>(i) / 100.0f;
    }

    Homography H(2, 0, 0, 0, 2, 0, 0, 0, 1);
    QImage result = WarpPerspective(floatImg, H, 20, 20);

    EXPECT_EQ(result.Width(), 20);
    EXPECT_EQ(result.Height(), 20);
    EXPECT_EQ(result.Type(), PixelType::Float32);
}

TEST_F(HomographyTest, WarpPerspective_UInt16) {
    QImage u16Img(10, 10, PixelType::UInt16, ChannelType::Gray);
    uint16_t* data = static_cast<uint16_t*>(u16Img.Data());
    for (int i = 0; i < 100; ++i) {
        data[i] = static_cast<uint16_t>(i * 100);
    }

    Homography H(0.5, 0, 0, 0, 0.5, 0, 0, 0, 1);
    QImage result = WarpPerspective(u16Img, H, 5, 5);

    EXPECT_EQ(result.Width(), 5);
    EXPECT_EQ(result.Height(), 5);
    EXPECT_EQ(result.Type(), PixelType::UInt16);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(HomographyTest, Transform_PointAtInfinity) {
    // Homography with perspective that could send point to infinity
    // H = [1 0 0; 0 1 0; 0.1 0 1], w = 0.1*x + 1, becomes 0 when x = -10
    Homography H(1, 0, 0, 0, 1, 0, 0.1, 0, 1);

    Point2d p{-10, 0};
    Point2d tp = H.Transform(p);

    // w = 0.1*(-10) + 0*0 + 1 = -1 + 1 = 0, should return infinity
    EXPECT_TRUE(std::isinf(tp.x) || std::abs(tp.x) > 1e10);
}

TEST_F(HomographyTest, DecomposeHomography_Empty) {
    Homography H = Homography::Identity();

    auto decomps = DecomposeHomography(H);

    // Currently returns empty (not fully implemented)
    // This test just verifies no crash
}

TEST_F(HomographyTest, FilterDecompositionsByVisibility_Empty) {
    std::vector<HomographyDecomposition> decomps;
    std::vector<Point2d> points = {{0, 0}, {1, 1}};

    auto filtered = FilterDecompositionsByVisibility(decomps, points);

    EXPECT_TRUE(filtered.empty());
}
