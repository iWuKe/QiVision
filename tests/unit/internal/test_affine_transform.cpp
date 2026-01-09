/**
 * @file test_affine_transform.cpp
 * @brief Unit tests for AffineTransform module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/AffineTransform.h>
#include <cmath>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// Use the PI from Constants.h
using Qi::Vision::PI;

class AffineTransformTest : public ::testing::Test {
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
// ComputeAffineOutputSize Tests
// =============================================================================

TEST_F(AffineTransformTest, ComputeOutputSize_Identity) {
    QMatrix identity = QMatrix::Identity();
    int32_t dstW, dstH;
    double offsetX, offsetY;

    ComputeAffineOutputSize(100, 80, identity, dstW, dstH, offsetX, offsetY);

    EXPECT_EQ(dstW, 100);
    EXPECT_EQ(dstH, 80);
    EXPECT_NEAR(offsetX, 0.0, 1e-6);
    EXPECT_NEAR(offsetY, 0.0, 1e-6);
}

TEST_F(AffineTransformTest, ComputeOutputSize_Scale2x) {
    QMatrix scale = QMatrix::Scaling(2.0, 2.0);
    int32_t dstW, dstH;
    double offsetX, offsetY;

    ComputeAffineOutputSize(100, 80, scale, dstW, dstH, offsetX, offsetY);

    EXPECT_NEAR(dstW, 199, 1);
    EXPECT_NEAR(dstH, 159, 1);
}

TEST_F(AffineTransformTest, ComputeOutputSize_Rotation90) {
    QMatrix rot = QMatrix::Rotation(PI / 2, 50, 40);
    int32_t dstW, dstH;
    double offsetX, offsetY;

    ComputeAffineOutputSize(100, 80, rot, dstW, dstH, offsetX, offsetY);

    // 90 degree rotation swaps dimensions approximately
    EXPECT_GT(dstW, 50);
    EXPECT_GT(dstH, 50);
}

// =============================================================================
// WarpAffine Tests
// =============================================================================

TEST_F(AffineTransformTest, WarpAffine_EmptyImage) {
    QImage empty;
    QMatrix identity = QMatrix::Identity();

    QImage result = WarpAffine(empty, identity);

    EXPECT_TRUE(result.Empty());
}

TEST_F(AffineTransformTest, WarpAffine_Identity) {
    QMatrix identity = QMatrix::Identity();

    QImage result = WarpAffine(testImage_, identity, 10, 10);

    EXPECT_EQ(result.Width(), 10);
    EXPECT_EQ(result.Height(), 10);

    // Values should be approximately preserved
    const uint8_t* srcData = static_cast<const uint8_t*>(testImage_.Data());
    const uint8_t* dstData = static_cast<const uint8_t*>(result.Data());
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(srcData[i], dstData[i], 1);
    }
}

TEST_F(AffineTransformTest, WarpAffine_Translation) {
    QMatrix trans = QMatrix::Translation(5, 5);

    QImage result = WarpAffine(testImage_, trans, 15, 15,
                               InterpolationMethod::Nearest,
                               BorderMode::Constant, 0);

    EXPECT_EQ(result.Width(), 15);
    EXPECT_EQ(result.Height(), 15);

    // Check that content is shifted
    const uint8_t* dstData = static_cast<const uint8_t*>(result.Data());
    // Original (0,0) should now be at (5,5)
    EXPECT_EQ(dstData[5 * 15 + 5], 0);  // Original value at (0,0)
}

TEST_F(AffineTransformTest, WarpAffine_Scale) {
    QMatrix scale = QMatrix::Scaling(2.0, 2.0);

    QImage result = WarpAffine(testImage_, scale, 20, 20);

    EXPECT_EQ(result.Width(), 20);
    EXPECT_EQ(result.Height(), 20);
}

TEST_F(AffineTransformTest, WarpAffine_BorderModes) {
    QMatrix trans = QMatrix::Translation(-2, -2);

    // Test different border modes
    QImage resultConst = WarpAffine(testImage_, trans, 10, 10,
                                    InterpolationMethod::Bilinear,
                                    BorderMode::Constant, 128);

    QImage resultRepl = WarpAffine(testImage_, trans, 10, 10,
                                   InterpolationMethod::Bilinear,
                                   BorderMode::Replicate, 0);

    // Both should produce valid images
    EXPECT_FALSE(resultConst.Empty());
    EXPECT_FALSE(resultRepl.Empty());
}

TEST_F(AffineTransformTest, WarpAffine_InterpolationMethods) {
    QMatrix scale = QMatrix::Scaling(1.5, 1.5);

    QImage resultNearest = WarpAffine(testImage_, scale, 15, 15,
                                      InterpolationMethod::Nearest);
    QImage resultBilinear = WarpAffine(testImage_, scale, 15, 15,
                                       InterpolationMethod::Bilinear);
    QImage resultBicubic = WarpAffine(testImage_, scale, 15, 15,
                                      InterpolationMethod::Bicubic);

    EXPECT_FALSE(resultNearest.Empty());
    EXPECT_FALSE(resultBilinear.Empty());
    EXPECT_FALSE(resultBicubic.Empty());
}

// =============================================================================
// RotateImage Tests
// =============================================================================

TEST_F(AffineTransformTest, RotateImage_0Degrees) {
    QImage result = RotateImage(testImage_, 0.0, false);

    EXPECT_EQ(result.Width(), testImage_.Width());
    EXPECT_EQ(result.Height(), testImage_.Height());
}

TEST_F(AffineTransformTest, RotateImage_90Degrees_NoResize) {
    QImage result = RotateImage(testImage_, PI / 2, false);

    EXPECT_EQ(result.Width(), 10);
    EXPECT_EQ(result.Height(), 10);
}

TEST_F(AffineTransformTest, RotateImage_90Degrees_Resize) {
    QImage result = RotateImage(testImage_, PI / 2, true);

    // With resize, output should be larger to fit rotated image
    EXPECT_GT(result.Width(), 0);
    EXPECT_GT(result.Height(), 0);
}

TEST_F(AffineTransformTest, RotateImage_180Degrees) {
    QImage result = RotateImage(testImage_, PI, false);

    EXPECT_FALSE(result.Empty());
}

TEST_F(AffineTransformTest, RotateImage_WithCenter) {
    double cx = 5.0, cy = 5.0;
    QImage result = RotateImage(testImage_, PI / 4, cx, cy, false);

    EXPECT_EQ(result.Width(), 10);
    EXPECT_EQ(result.Height(), 10);
}

TEST_F(AffineTransformTest, RotateImage_Empty) {
    QImage empty;
    QImage result = RotateImage(empty, PI / 2);

    EXPECT_TRUE(result.Empty());
}

// =============================================================================
// ScaleImage Tests
// =============================================================================

TEST_F(AffineTransformTest, ScaleImage_2x) {
    QImage result = ScaleImage(testImage_, 20, 20);

    EXPECT_EQ(result.Width(), 20);
    EXPECT_EQ(result.Height(), 20);
}

TEST_F(AffineTransformTest, ScaleImage_HalfSize) {
    QImage result = ScaleImage(testImage_, 5, 5);

    EXPECT_EQ(result.Width(), 5);
    EXPECT_EQ(result.Height(), 5);
}

TEST_F(AffineTransformTest, ScaleImage_NonUniform) {
    QImage result = ScaleImage(testImage_, 20, 5);

    EXPECT_EQ(result.Width(), 20);
    EXPECT_EQ(result.Height(), 5);
}

TEST_F(AffineTransformTest, ScaleImage_InvalidSize) {
    QImage result = ScaleImage(testImage_, 0, 10);
    EXPECT_TRUE(result.Empty());

    result = ScaleImage(testImage_, 10, -1);
    EXPECT_TRUE(result.Empty());
}

TEST_F(AffineTransformTest, ScaleImageFactor_2x) {
    QImage result = ScaleImageFactor(testImage_, 2.0, 2.0);

    EXPECT_EQ(result.Width(), 20);
    EXPECT_EQ(result.Height(), 20);
}

TEST_F(AffineTransformTest, ScaleImageFactor_Invalid) {
    QImage result = ScaleImageFactor(testImage_, 0.0, 1.0);
    EXPECT_TRUE(result.Empty());

    result = ScaleImageFactor(testImage_, 1.0, -1.0);
    EXPECT_TRUE(result.Empty());
}

// =============================================================================
// CropRotatedRect Tests
// =============================================================================

TEST_F(AffineTransformTest, CropRotatedRect_Basic) {
    RotatedRect2d rect;
    rect.center = {5.0, 5.0};
    rect.width = 6.0;
    rect.height = 4.0;
    rect.angle = 0.0;

    QImage result = CropRotatedRect(testImage_, rect);

    EXPECT_EQ(result.Width(), 6);
    EXPECT_EQ(result.Height(), 4);
}

TEST_F(AffineTransformTest, CropRotatedRect_WithRotation) {
    RotatedRect2d rect;
    rect.center = {5.0, 5.0};
    rect.width = 6.0;
    rect.height = 4.0;
    rect.angle = PI / 4;

    QImage result = CropRotatedRect(testImage_, rect);

    EXPECT_EQ(result.Width(), 6);
    EXPECT_EQ(result.Height(), 4);
}

TEST_F(AffineTransformTest, CropRotatedRect_Empty) {
    QImage empty;
    RotatedRect2d rect;
    rect.center = {5.0, 5.0};
    rect.width = 6.0;
    rect.height = 4.0;
    rect.angle = 0.0;

    QImage result = CropRotatedRect(empty, rect);

    EXPECT_TRUE(result.Empty());
}

// =============================================================================
// EstimateAffine Tests
// =============================================================================

TEST_F(AffineTransformTest, EstimateAffine_Identity) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {0, 10}, {10, 10}};
    std::vector<Point2d> dst = {{0, 0}, {10, 0}, {0, 10}, {10, 10}};

    auto result = EstimateAffine(src, dst);

    ASSERT_TRUE(result.has_value());

    // Should be close to identity
    for (const auto& pt : src) {
        Point2d transformed = result->Transform(pt);
        EXPECT_NEAR(transformed.x, pt.x, 1e-6);
        EXPECT_NEAR(transformed.y, pt.y, 1e-6);
    }
}

TEST_F(AffineTransformTest, EstimateAffine_Translation) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {5, 10}};
    std::vector<Point2d> dst = {{5, 3}, {15, 3}, {10, 13}};

    auto result = EstimateAffine(src, dst);

    ASSERT_TRUE(result.has_value());

    for (size_t i = 0; i < src.size(); ++i) {
        Point2d transformed = result->Transform(src[i]);
        EXPECT_NEAR(transformed.x, dst[i].x, 1e-4);
        EXPECT_NEAR(transformed.y, dst[i].y, 1e-4);
    }
}

TEST_F(AffineTransformTest, EstimateAffine_Scale) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {0, 10}};
    std::vector<Point2d> dst = {{0, 0}, {20, 0}, {0, 20}};

    auto result = EstimateAffine(src, dst);

    ASSERT_TRUE(result.has_value());

    for (size_t i = 0; i < src.size(); ++i) {
        Point2d transformed = result->Transform(src[i]);
        EXPECT_NEAR(transformed.x, dst[i].x, 1e-4);
        EXPECT_NEAR(transformed.y, dst[i].y, 1e-4);
    }
}

TEST_F(AffineTransformTest, EstimateAffine_Rotation) {
    // 90 degree rotation around origin
    std::vector<Point2d> src = {{1, 0}, {0, 1}, {-1, 0}};
    std::vector<Point2d> dst = {{0, 1}, {-1, 0}, {0, -1}};

    auto result = EstimateAffine(src, dst);

    ASSERT_TRUE(result.has_value());

    for (size_t i = 0; i < src.size(); ++i) {
        Point2d transformed = result->Transform(src[i]);
        EXPECT_NEAR(transformed.x, dst[i].x, 1e-4);
        EXPECT_NEAR(transformed.y, dst[i].y, 1e-4);
    }
}

TEST_F(AffineTransformTest, EstimateAffine_TooFewPoints) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}};
    std::vector<Point2d> dst = {{5, 3}, {15, 3}};

    auto result = EstimateAffine(src, dst);

    EXPECT_FALSE(result.has_value());
}

TEST_F(AffineTransformTest, EstimateAffine_MismatchedSizes) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {5, 10}};
    std::vector<Point2d> dst = {{5, 3}, {15, 3}};

    auto result = EstimateAffine(src, dst);

    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// EstimateRigid Tests
// =============================================================================

TEST_F(AffineTransformTest, EstimateRigid_Identity) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {5, 10}};
    std::vector<Point2d> dst = {{0, 0}, {10, 0}, {5, 10}};

    auto result = EstimateRigid(src, dst);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(IsRigidTransform(*result));
}

TEST_F(AffineTransformTest, EstimateRigid_Translation) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {5, 10}};
    std::vector<Point2d> dst = {{5, 3}, {15, 3}, {10, 13}};

    auto result = EstimateRigid(src, dst);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(IsRigidTransform(*result));

    for (size_t i = 0; i < src.size(); ++i) {
        Point2d transformed = result->Transform(src[i]);
        EXPECT_NEAR(transformed.x, dst[i].x, 1e-4);
        EXPECT_NEAR(transformed.y, dst[i].y, 1e-4);
    }
}

TEST_F(AffineTransformTest, EstimateRigid_Rotation) {
    // Pure 90 degree rotation
    std::vector<Point2d> src = {{0, 0}, {1, 0}, {0, 1}};
    std::vector<Point2d> dst = {{0, 0}, {0, 1}, {-1, 0}};

    auto result = EstimateRigid(src, dst);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(IsRigidTransform(*result));
}

TEST_F(AffineTransformTest, EstimateRigid_TooFewPoints) {
    std::vector<Point2d> src = {{0, 0}};
    std::vector<Point2d> dst = {{5, 3}};

    auto result = EstimateRigid(src, dst);

    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// EstimateSimilarity Tests
// =============================================================================

TEST_F(AffineTransformTest, EstimateSimilarity_Identity) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {5, 10}};
    std::vector<Point2d> dst = {{0, 0}, {10, 0}, {5, 10}};

    auto result = EstimateSimilarity(src, dst);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(IsSimilarityTransform(*result));
}

TEST_F(AffineTransformTest, EstimateSimilarity_UniformScale) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {0, 10}};
    std::vector<Point2d> dst = {{0, 0}, {20, 0}, {0, 20}};

    auto result = EstimateSimilarity(src, dst);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(IsSimilarityTransform(*result));

    // Verify scale is 2x
    Point2d p1 = result->Transform({1, 0});
    double scale = std::sqrt(p1.x * p1.x + p1.y * p1.y);
    EXPECT_NEAR(scale, 2.0, 1e-4);
}

TEST_F(AffineTransformTest, EstimateSimilarity_RotationAndScale) {
    // 90 degree rotation + 2x scale
    std::vector<Point2d> src = {{0, 0}, {1, 0}, {0, 1}};
    std::vector<Point2d> dst = {{0, 0}, {0, 2}, {-2, 0}};

    auto result = EstimateSimilarity(src, dst);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(IsSimilarityTransform(*result));
}

// =============================================================================
// RANSAC Tests
// =============================================================================

TEST_F(AffineTransformTest, EstimateAffineRANSAC_NoOutliers) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {0, 10}, {10, 10}, {5, 5}};
    std::vector<Point2d> dst = {{5, 5}, {15, 5}, {5, 15}, {15, 15}, {10, 10}};

    auto result = EstimateAffineRANSAC(src, dst, 1.0, 0.99, 100);

    ASSERT_TRUE(result.has_value());

    double error = ComputeTransformError(src, dst, *result);
    EXPECT_LT(error, 0.1);
}

TEST_F(AffineTransformTest, EstimateAffineRANSAC_WithOutliers) {
    // Good points (translation by 5,5)
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {0, 10}, {10, 10}, {5, 5},
                                {100, 100}};  // Outlier
    std::vector<Point2d> dst = {{5, 5}, {15, 5}, {5, 15}, {15, 15}, {10, 10},
                                {200, 200}};  // Outlier

    std::vector<bool> inlierMask;
    auto result = EstimateAffineRANSAC(src, dst, 1.0, 0.99, 500, &inlierMask);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(inlierMask.size(), src.size());

    // Last point should be outlier
    EXPECT_FALSE(inlierMask[5]);

    // First 5 should be inliers
    int inlierCount = 0;
    for (int i = 0; i < 5; ++i) {
        if (inlierMask[i]) inlierCount++;
    }
    EXPECT_GE(inlierCount, 4);
}

TEST_F(AffineTransformTest, EstimateRigidRANSAC_Basic) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {5, 10}};
    std::vector<Point2d> dst = {{5, 3}, {15, 3}, {10, 13}};

    auto result = EstimateRigidRANSAC(src, dst, 1.0, 0.99, 100);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(IsRigidTransform(*result, 0.1));
}

TEST_F(AffineTransformTest, EstimateSimilarityRANSAC_Basic) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {0, 10}};
    std::vector<Point2d> dst = {{0, 0}, {20, 0}, {0, 20}};

    auto result = EstimateSimilarityRANSAC(src, dst, 1.0, 0.99, 100);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(IsSimilarityTransform(*result, 0.1));
}

// =============================================================================
// ComputeTransformError Tests
// =============================================================================

TEST_F(AffineTransformTest, ComputeTransformError_Zero) {
    std::vector<Point2d> src = {{0, 0}, {10, 0}, {5, 10}};
    std::vector<Point2d> dst = {{0, 0}, {10, 0}, {5, 10}};
    QMatrix identity = QMatrix::Identity();

    double error = ComputeTransformError(src, dst, identity);

    EXPECT_NEAR(error, 0.0, 1e-10);
}

TEST_F(AffineTransformTest, ComputeTransformError_NonZero) {
    std::vector<Point2d> src = {{0, 0}};
    std::vector<Point2d> dst = {{1, 0}};  // 1 pixel off
    QMatrix identity = QMatrix::Identity();

    double error = ComputeTransformError(src, dst, identity);

    EXPECT_NEAR(error, 1.0, 1e-10);
}

TEST_F(AffineTransformTest, ComputePointErrors_Basic) {
    std::vector<Point2d> src = {{0, 0}, {3, 0}, {0, 4}};
    std::vector<Point2d> dst = {{1, 0}, {3, 0}, {0, 4}};
    QMatrix identity = QMatrix::Identity();

    auto errors = ComputePointErrors(src, dst, identity);

    ASSERT_EQ(errors.size(), 3);
    EXPECT_NEAR(errors[0], 1.0, 1e-10);  // First point has 1px error
    EXPECT_NEAR(errors[1], 0.0, 1e-10);  // Second is exact
    EXPECT_NEAR(errors[2], 0.0, 1e-10);  // Third is exact
}

// =============================================================================
// Region Transformation Tests
// =============================================================================

TEST_F(AffineTransformTest, AffineTransformRegion_Empty) {
    QRegion empty;
    QMatrix trans = QMatrix::Translation(10, 10);

    QRegion result = AffineTransformRegion(empty, trans);

    EXPECT_TRUE(result.Empty());
}

TEST_F(AffineTransformTest, AffineTransformRegion_Translation) {
    // Create a simple rectangular region
    std::vector<QRegion::Run> runs = {
        {0, 0, 5},
        {1, 0, 5},
        {2, 0, 5}
    };
    QRegion region(runs);

    QMatrix trans = QMatrix::Translation(10, 10);
    QRegion result = AffineTransformRegion(region, trans);

    EXPECT_FALSE(result.Empty());

    // Check that region is shifted
    Rect2i bbox = result.BoundingBox();
    EXPECT_EQ(bbox.x, 10);
    EXPECT_EQ(bbox.y, 10);
}

TEST_F(AffineTransformTest, AffineTransformRegion_Scale) {
    std::vector<QRegion::Run> runs = {
        {0, 0, 10},
        {1, 0, 10}
    };
    QRegion region(runs);

    QMatrix scale = QMatrix::Scaling(2.0, 2.0);
    QRegion result = AffineTransformRegion(region, scale);

    EXPECT_FALSE(result.Empty());

    // Area should be larger (discretization can cause variations)
    int64_t origArea = region.Area();
    int64_t newArea = result.Area();
    // With 2x scale, expect roughly 2-4x area due to discrete sampling
    EXPECT_GT(newArea, origArea);  // At minimum, should be larger
}

TEST_F(AffineTransformTest, AffineTransformRegions_Multiple) {
    std::vector<QRegion::Run> runs1 = {{0, 0, 5}, {1, 0, 5}};
    std::vector<QRegion::Run> runs2 = {{0, 10, 15}, {1, 10, 15}};

    std::vector<QRegion> regions = {QRegion(runs1), QRegion(runs2)};
    QMatrix trans = QMatrix::Translation(5, 5);

    auto results = AffineTransformRegions(regions, trans);

    EXPECT_EQ(results.size(), 2);
    EXPECT_FALSE(results[0].Empty());
    EXPECT_FALSE(results[1].Empty());
}

// =============================================================================
// Contour Transformation Tests
// =============================================================================

TEST_F(AffineTransformTest, AffineTransformContour_Basic) {
    std::vector<Point2d> points = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    QContour contour(points);

    QMatrix trans = QMatrix::Translation(5, 5);
    QContour result = AffineTransformContour(contour, trans);

    EXPECT_EQ(result.Size(), 4);

    auto resultPts = result.GetPoints();
    EXPECT_NEAR(resultPts[0].x, 5, 1e-6);
    EXPECT_NEAR(resultPts[0].y, 5, 1e-6);
}

TEST_F(AffineTransformTest, AffineTransformContours_Multiple) {
    std::vector<Point2d> pts1 = {{0, 0}, {5, 0}, {5, 5}};
    std::vector<Point2d> pts2 = {{10, 10}, {15, 10}, {15, 15}};

    std::vector<QContour> contours = {QContour(pts1), QContour(pts2)};
    QMatrix scale = QMatrix::Scaling(2.0, 2.0);

    auto results = AffineTransformContours(contours, scale);

    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].Size(), 3);
    EXPECT_EQ(results[1].Size(), 3);

    // Check scaling
    EXPECT_NEAR(results[0].GetPoint(1).x, 10, 1e-6);  // 5 * 2
}

// =============================================================================
// DecomposeAffine Tests
// =============================================================================

TEST_F(AffineTransformTest, DecomposeAffine_Identity) {
    QMatrix identity = QMatrix::Identity();
    double tx, ty, angle, scaleX, scaleY, shear;

    bool success = DecomposeAffine(identity, tx, ty, angle, scaleX, scaleY, shear);

    EXPECT_TRUE(success);
    EXPECT_NEAR(tx, 0, 1e-6);
    EXPECT_NEAR(ty, 0, 1e-6);
    EXPECT_NEAR(angle, 0, 1e-6);
    EXPECT_NEAR(scaleX, 1, 1e-6);
    EXPECT_NEAR(scaleY, 1, 1e-6);
    EXPECT_NEAR(shear, 0, 1e-6);
}

TEST_F(AffineTransformTest, DecomposeAffine_Translation) {
    QMatrix trans = QMatrix::Translation(10, 20);
    double tx, ty, angle, scaleX, scaleY, shear;

    bool success = DecomposeAffine(trans, tx, ty, angle, scaleX, scaleY, shear);

    EXPECT_TRUE(success);
    EXPECT_NEAR(tx, 10, 1e-6);
    EXPECT_NEAR(ty, 20, 1e-6);
}

TEST_F(AffineTransformTest, DecomposeAffine_Scale) {
    QMatrix scale = QMatrix::Scaling(2, 3);
    double tx, ty, angle, scaleX, scaleY, shear;

    bool success = DecomposeAffine(scale, tx, ty, angle, scaleX, scaleY, shear);

    EXPECT_TRUE(success);
    EXPECT_NEAR(scaleX, 2, 1e-6);
    EXPECT_NEAR(scaleY, 3, 1e-6);
}

TEST_F(AffineTransformTest, DecomposeAffine_Rotation) {
    double inputAngle = PI / 4;  // 45 degrees
    QMatrix rot = QMatrix::Rotation(inputAngle);
    double tx, ty, angle, scaleX, scaleY, shear;

    bool success = DecomposeAffine(rot, tx, ty, angle, scaleX, scaleY, shear);

    EXPECT_TRUE(success);
    EXPECT_NEAR(angle, inputAngle, 1e-6);
    EXPECT_NEAR(scaleX, 1, 1e-6);
    EXPECT_NEAR(std::abs(scaleY), 1, 1e-6);
}

// =============================================================================
// IsRigidTransform Tests
// =============================================================================

TEST_F(AffineTransformTest, IsRigidTransform_Identity) {
    EXPECT_TRUE(IsRigidTransform(QMatrix::Identity()));
}

TEST_F(AffineTransformTest, IsRigidTransform_Translation) {
    EXPECT_TRUE(IsRigidTransform(QMatrix::Translation(10, 20)));
}

TEST_F(AffineTransformTest, IsRigidTransform_Rotation) {
    EXPECT_TRUE(IsRigidTransform(QMatrix::Rotation(PI / 6)));
}

TEST_F(AffineTransformTest, IsRigidTransform_Scale_False) {
    EXPECT_FALSE(IsRigidTransform(QMatrix::Scaling(2, 2)));
}

TEST_F(AffineTransformTest, IsRigidTransform_NonUniformScale_False) {
    EXPECT_FALSE(IsRigidTransform(QMatrix::Scaling(2, 3)));
}

// =============================================================================
// IsSimilarityTransform Tests
// =============================================================================

TEST_F(AffineTransformTest, IsSimilarityTransform_Identity) {
    EXPECT_TRUE(IsSimilarityTransform(QMatrix::Identity()));
}

TEST_F(AffineTransformTest, IsSimilarityTransform_UniformScale) {
    EXPECT_TRUE(IsSimilarityTransform(QMatrix::Scaling(2, 2)));
}

TEST_F(AffineTransformTest, IsSimilarityTransform_NonUniformScale_False) {
    EXPECT_FALSE(IsSimilarityTransform(QMatrix::Scaling(2, 3)));
}

TEST_F(AffineTransformTest, IsSimilarityTransform_RotationAndScale) {
    QMatrix m = QMatrix::Rotation(PI / 4) * QMatrix::Scaling(2, 2);
    EXPECT_TRUE(IsSimilarityTransform(m));
}

// =============================================================================
// InterpolateTransform Tests
// =============================================================================

TEST_F(AffineTransformTest, InterpolateTransform_T0) {
    QMatrix m1 = QMatrix::Identity();
    QMatrix m2 = QMatrix::Translation(10, 10);

    QMatrix result = InterpolateTransform(m1, m2, 0.0);

    EXPECT_NEAR(result.M02(), 0, 1e-6);
    EXPECT_NEAR(result.M12(), 0, 1e-6);
}

TEST_F(AffineTransformTest, InterpolateTransform_T1) {
    QMatrix m1 = QMatrix::Identity();
    QMatrix m2 = QMatrix::Translation(10, 10);

    QMatrix result = InterpolateTransform(m1, m2, 1.0);

    EXPECT_NEAR(result.M02(), 10, 1e-6);
    EXPECT_NEAR(result.M12(), 10, 1e-6);
}

TEST_F(AffineTransformTest, InterpolateTransform_T05) {
    QMatrix m1 = QMatrix::Identity();
    QMatrix m2 = QMatrix::Translation(10, 10);

    QMatrix result = InterpolateTransform(m1, m2, 0.5);

    EXPECT_NEAR(result.M02(), 5, 1e-6);
    EXPECT_NEAR(result.M12(), 5, 1e-6);
}

// =============================================================================
// RectToRectTransform Tests
// =============================================================================

TEST_F(AffineTransformTest, RectToRectTransform_Same) {
    Rect2d src(0, 0, 10, 10);
    Rect2d dst(0, 0, 10, 10);

    QMatrix m = RectToRectTransform(src, dst);

    Point2d p = m.Transform({5, 5});
    EXPECT_NEAR(p.x, 5, 1e-6);
    EXPECT_NEAR(p.y, 5, 1e-6);
}

TEST_F(AffineTransformTest, RectToRectTransform_Scale) {
    Rect2d src(0, 0, 10, 10);
    Rect2d dst(0, 0, 20, 20);

    QMatrix m = RectToRectTransform(src, dst);

    Point2d corner = m.Transform({10, 10});
    EXPECT_NEAR(corner.x, 20, 1e-6);
    EXPECT_NEAR(corner.y, 20, 1e-6);
}

TEST_F(AffineTransformTest, RectToRectTransform_Translate) {
    Rect2d src(0, 0, 10, 10);
    Rect2d dst(5, 5, 10, 10);

    QMatrix m = RectToRectTransform(src, dst);

    Point2d origin = m.Transform({0, 0});
    EXPECT_NEAR(origin.x, 5, 1e-6);
    EXPECT_NEAR(origin.y, 5, 1e-6);
}

// =============================================================================
// RotatedRectToAxisAligned Tests
// =============================================================================

TEST_F(AffineTransformTest, RotatedRectToAxisAligned_NoRotation) {
    RotatedRect2d rect;
    rect.center = {5, 5};
    rect.width = 10;
    rect.height = 6;
    rect.angle = 0;

    QMatrix m = RotatedRectToAxisAligned(rect);

    // Center should map to output center
    Point2d center = m.Transform(rect.center);
    EXPECT_NEAR(center.x, 5, 1e-6);   // width/2
    EXPECT_NEAR(center.y, 3, 1e-6);   // height/2
}

TEST_F(AffineTransformTest, RotatedRectToAxisAligned_WithRotation) {
    RotatedRect2d rect;
    rect.center = {10, 10};
    rect.width = 8;
    rect.height = 4;
    rect.angle = PI / 4;

    QMatrix m = RotatedRectToAxisAligned(rect);

    // Center should still map to output center
    Point2d center = m.Transform(rect.center);
    EXPECT_NEAR(center.x, 4, 1e-6);   // width/2
    EXPECT_NEAR(center.y, 2, 1e-6);   // height/2
}

// =============================================================================
// TransformBoundingBox Tests
// =============================================================================

TEST_F(AffineTransformTest, TransformBoundingBox_Identity) {
    Rect2d bbox(10, 20, 30, 40);
    QMatrix identity = QMatrix::Identity();

    Rect2d result = TransformBoundingBox(bbox, identity);

    EXPECT_NEAR(result.x, 10, 1e-6);
    EXPECT_NEAR(result.y, 20, 1e-6);
    EXPECT_NEAR(result.width, 30, 1e-6);
    EXPECT_NEAR(result.height, 40, 1e-6);
}

TEST_F(AffineTransformTest, TransformBoundingBox_Scale) {
    Rect2d bbox(0, 0, 10, 10);
    QMatrix scale = QMatrix::Scaling(2, 2);

    Rect2d result = TransformBoundingBox(bbox, scale);

    EXPECT_NEAR(result.width, 20, 1e-6);
    EXPECT_NEAR(result.height, 20, 1e-6);
}

TEST_F(AffineTransformTest, TransformBoundingBox_Rotation) {
    Rect2d bbox(0, 0, 10, 10);
    QMatrix rot = QMatrix::Rotation(PI / 4, 5, 5);

    Rect2d result = TransformBoundingBox(bbox, rot);

    // Rotated square should have larger bbox
    double diagonal = 10 * std::sqrt(2.0);
    EXPECT_GT(result.width, 10);
    EXPECT_LT(result.width, diagonal + 1);
}

// =============================================================================
// TransformPointsBoundingBox Tests
// =============================================================================

TEST_F(AffineTransformTest, TransformPointsBoundingBox_Empty) {
    std::vector<Point2d> points;
    QMatrix identity = QMatrix::Identity();

    Rect2d result = TransformPointsBoundingBox(points, identity);

    EXPECT_EQ(result.width, 0);
    EXPECT_EQ(result.height, 0);
}

TEST_F(AffineTransformTest, TransformPointsBoundingBox_SinglePoint) {
    std::vector<Point2d> points = {{5, 10}};
    QMatrix identity = QMatrix::Identity();

    Rect2d result = TransformPointsBoundingBox(points, identity);

    EXPECT_NEAR(result.x, 5, 1e-6);
    EXPECT_NEAR(result.y, 10, 1e-6);
    EXPECT_NEAR(result.width, 0, 1e-6);
    EXPECT_NEAR(result.height, 0, 1e-6);
}

TEST_F(AffineTransformTest, TransformPointsBoundingBox_Multiple) {
    std::vector<Point2d> points = {{0, 0}, {10, 0}, {10, 5}, {0, 5}};
    QMatrix scale = QMatrix::Scaling(2, 2);

    Rect2d result = TransformPointsBoundingBox(points, scale);

    EXPECT_NEAR(result.x, 0, 1e-6);
    EXPECT_NEAR(result.y, 0, 1e-6);
    EXPECT_NEAR(result.width, 20, 1e-6);
    EXPECT_NEAR(result.height, 10, 1e-6);
}

// =============================================================================
// Multi-channel Image Tests
// =============================================================================

TEST_F(AffineTransformTest, WarpAffine_MultiChannel) {
    // Create 3-channel image
    QImage rgb(10, 10, PixelType::UInt8, ChannelType::RGB);
    uint8_t* data = static_cast<uint8_t*>(rgb.Data());

    // Fill each channel differently
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 100; ++i) {
            data[c * 100 + i] = static_cast<uint8_t>(i + c * 10);
        }
    }

    QMatrix scale = QMatrix::Scaling(2, 2);
    QImage result = WarpAffine(rgb, scale, 20, 20);

    EXPECT_EQ(result.Width(), 20);
    EXPECT_EQ(result.Height(), 20);
    EXPECT_EQ(result.Channels(), 3);
}

// =============================================================================
// Different Pixel Types Tests
// =============================================================================

TEST_F(AffineTransformTest, WarpAffine_Float32) {
    QImage floatImg(10, 10, PixelType::Float32, ChannelType::Gray);
    float* data = static_cast<float*>(floatImg.Data());
    for (int i = 0; i < 100; ++i) {
        data[i] = static_cast<float>(i) / 100.0f;
    }

    QMatrix scale = QMatrix::Scaling(2, 2);
    QImage result = WarpAffine(floatImg, scale, 20, 20);

    EXPECT_EQ(result.Width(), 20);
    EXPECT_EQ(result.Height(), 20);
    EXPECT_EQ(result.Type(), PixelType::Float32);
}

TEST_F(AffineTransformTest, WarpAffine_UInt16) {
    QImage u16Img(10, 10, PixelType::UInt16, ChannelType::Gray);
    uint16_t* data = static_cast<uint16_t*>(u16Img.Data());
    for (int i = 0; i < 100; ++i) {
        data[i] = static_cast<uint16_t>(i * 100);
    }

    QMatrix scale = QMatrix::Scaling(0.5, 0.5);
    QImage result = WarpAffine(u16Img, scale, 5, 5);

    EXPECT_EQ(result.Width(), 5);
    EXPECT_EQ(result.Height(), 5);
    EXPECT_EQ(result.Type(), PixelType::UInt16);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(AffineTransformTest, WarpAffine_SingularMatrix) {
    // Create a matrix that's not invertible
    QMatrix singular(0, 0, 0, 0, 0, 0);

    QImage result = WarpAffine(testImage_, singular, 10, 10);

    EXPECT_TRUE(result.Empty());
}

TEST_F(AffineTransformTest, EstimateAffine_CollinearPoints) {
    // All points on a line - should still work but might be less stable
    std::vector<Point2d> src = {{0, 0}, {5, 0}, {10, 0}};
    std::vector<Point2d> dst = {{0, 5}, {5, 5}, {10, 5}};

    auto result = EstimateAffine(src, dst);

    // May or may not succeed depending on numerical stability
    // Just verify no crash
}

TEST_F(AffineTransformTest, AffineTransformRegion_NonInvertible) {
    std::vector<QRegion::Run> runs = {{0, 0, 5}, {1, 0, 5}};
    QRegion region(runs);

    QMatrix singular(0, 0, 0, 0, 0, 0);
    QRegion result = AffineTransformRegion(region, singular);

    EXPECT_TRUE(result.Empty());
}
