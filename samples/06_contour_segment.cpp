/**
 * @file 06_contour_segment.cpp
 * @brief 示例：轮廓分割 / Example: Contour Segmentation
 *
 * 演示将轮廓分割为直线段和圆弧
 * Demonstrates segmenting contours into line segments and arcs
 */

#include <QiVision/QiVision.h>
#include <QiVision/Internal/ContourSegment.h>
#include <QiVision/Internal/ContourAnalysis.h>
#include <cstdio>
#include <cmath>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// 创建矩形轮廓（带点采样）/ Create rectangle contour with point sampling
QContour CreateRectangle(double width, double height, Point2d center, int pointsPerEdge = 20) {
    QContour contour;
    double hw = width / 2, hh = height / 2;

    Point2d corners[4] = {
        {center.x - hw, center.y - hh},
        {center.x + hw, center.y - hh},
        {center.x + hw, center.y + hh},
        {center.x - hw, center.y + hh}
    };

    for (int edge = 0; edge < 4; ++edge) {
        Point2d p1 = corners[edge];
        Point2d p2 = corners[(edge + 1) % 4];
        for (int i = 0; i < pointsPerEdge; ++i) {
            double t = static_cast<double>(i) / pointsPerEdge;
            contour.AddPoint(p1.x + t * (p2.x - p1.x),
                            p1.y + t * (p2.y - p1.y));
        }
    }
    contour.SetClosed(true);
    return contour;
}

// 创建圆角矩形 / Create rounded rectangle
QContour CreateRoundedRect(double width, double height, double radius,
                           Point2d center, int pointsPerEdge = 15, int pointsPerCorner = 10) {
    QContour contour;
    double hw = width / 2 - radius;
    double hh = height / 2 - radius;

    // 四个圆角的圆心 / Centers of four corner arcs
    Point2d arcCenters[4] = {
        {center.x + hw, center.y - hh},  // 右上
        {center.x + hw, center.y + hh},  // 右下
        {center.x - hw, center.y + hh},  // 左下
        {center.x - hw, center.y - hh}   // 左上
    };

    double startAngles[4] = {-M_PI/2, 0, M_PI/2, M_PI};

    for (int corner = 0; corner < 4; ++corner) {
        // 直线段 / Straight edge
        Point2d lineStart, lineEnd;
        if (corner == 0) {
            lineStart = {center.x - hw, center.y - height/2};
            lineEnd = {center.x + hw, center.y - height/2};
        } else if (corner == 1) {
            lineStart = {center.x + width/2, center.y - hh};
            lineEnd = {center.x + width/2, center.y + hh};
        } else if (corner == 2) {
            lineStart = {center.x + hw, center.y + height/2};
            lineEnd = {center.x - hw, center.y + height/2};
        } else {
            lineStart = {center.x - width/2, center.y + hh};
            lineEnd = {center.x - width/2, center.y - hh};
        }

        for (int i = 0; i < pointsPerEdge; ++i) {
            double t = static_cast<double>(i) / pointsPerEdge;
            contour.AddPoint(lineStart.x + t * (lineEnd.x - lineStart.x),
                            lineStart.y + t * (lineEnd.y - lineStart.y));
        }

        // 圆弧 / Arc
        for (int i = 0; i < pointsPerCorner; ++i) {
            double t = static_cast<double>(i) / pointsPerCorner;
            double angle = startAngles[corner] + t * (M_PI / 2);
            contour.AddPoint(arcCenters[corner].x + radius * std::cos(angle),
                            arcCenters[corner].y + radius * std::sin(angle));
        }
    }

    contour.SetClosed(true);
    return contour;
}

// 创建 L 形轮廓 / Create L-shaped contour
QContour CreateLShape(int pointsPerSegment = 30) {
    QContour contour;

    // 水平部分 / Horizontal part
    for (int i = 0; i <= pointsPerSegment; ++i) {
        contour.AddPoint(i * 3.0, 0);
    }

    // 垂直部分 / Vertical part
    for (int i = 1; i <= pointsPerSegment; ++i) {
        contour.AddPoint(pointsPerSegment * 3.0, i * 3.0);
    }

    contour.SetClosed(false);
    return contour;
}

void PrintPrimitive(const PrimitiveFitResult& p, int index) {
    if (p.type == PrimitiveType::Line) {
        printf("   [%d] LINE: (%.1f, %.1f) -> (%.1f, %.1f), length=%.1f, error=%.3f\n",
               index,
               p.segment.p1.x, p.segment.p1.y,
               p.segment.p2.x, p.segment.p2.y,
               p.Length(), p.error);
    } else if (p.type == PrimitiveType::Arc) {
        printf("   [%d] ARC: center=(%.1f, %.1f), r=%.1f, sweep=%.1f°, error=%.3f\n",
               index,
               p.arc.center.x, p.arc.center.y,
               p.arc.radius,
               p.arc.sweepAngle * 180.0 / M_PI,
               p.error);
    }
}

int main(int argc, char* argv[]) {
    printf("=== QiVision Sample: Contour Segmentation ===\n\n");

    // 1. L 形轮廓分割 / L-shape segmentation
    printf("1. L-shape contour segmentation:\n");
    {
        QContour lShape = CreateLShape(30);
        printf("   Input: %zu points, open contour\n", lShape.Size());

        SegmentParams params;
        params.mode = SegmentMode::LinesOnly;
        params.maxLineError = 1.0;

        SegmentationResult result = SegmentContour(lShape, params);

        printf("   Found %zu primitives (%zu lines, %zu arcs)\n",
               result.primitives.size(), result.LineCount(), result.ArcCount());

        for (size_t i = 0; i < result.primitives.size(); ++i) {
            PrintPrimitive(result.primitives[i], static_cast<int>(i));
        }
    }

    // 2. 矩形分割 / Rectangle segmentation
    printf("\n2. Rectangle contour segmentation:\n");
    {
        QContour rect = CreateRectangle(100, 60, {150, 100}, 25);
        printf("   Input: %zu points, closed contour\n", rect.Size());

        SegmentParams params;
        params.mode = SegmentMode::LinesOnly;
        params.maxLineError = 1.0;

        SegmentationResult result = SegmentContour(rect, params);

        printf("   Found %zu lines\n", result.LineCount());
        for (size_t i = 0; i < result.primitives.size(); ++i) {
            PrintPrimitive(result.primitives[i], static_cast<int>(i));
        }
    }

    // 3. 圆角矩形分割（线段+圆弧）/ Rounded rectangle (lines + arcs)
    printf("\n3. Rounded rectangle segmentation (lines and arcs):\n");
    {
        QContour roundRect = CreateRoundedRect(120, 80, 15, {200, 150}, 20, 12);
        printf("   Input: %zu points\n", roundRect.Size());

        SegmentParams params;
        params.mode = SegmentMode::LinesAndArcs;
        params.maxLineError = 1.0;
        params.maxArcError = 1.0;

        SegmentationResult result = SegmentContour(roundRect, params);

        printf("   Found %zu primitives (%zu lines, %zu arcs)\n",
               result.primitives.size(), result.LineCount(), result.ArcCount());

        for (size_t i = 0; i < result.primitives.size(); ++i) {
            PrintPrimitive(result.primitives[i], static_cast<int>(i));
        }
    }

    // 4. 角点检测 / Corner detection
    printf("\n4. Corner detection on rectangle:\n");
    {
        QContour rect = CreateRectangle(80, 60, {100, 100}, 20);

        std::vector<size_t> corners = DetectCorners(rect, 0.05, 5);

        printf("   Found %zu corners at indices: ", corners.size());
        for (size_t idx : corners) {
            Point2d pt = rect.GetPoint(idx);
            printf("%zu(%.0f,%.0f) ", idx, pt.x, pt.y);
        }
        printf("\n");
    }

    // 5. 不同分割算法对比 / Compare segmentation algorithms
    printf("\n5. Segmentation algorithms comparison:\n");
    {
        QContour roundRect = CreateRoundedRect(100, 70, 12, {150, 120}, 15, 10);

        printf("   Input: Rounded rectangle, %zu points\n", roundRect.Size());

        // Curvature-based
        SegmentParams curvParams;
        curvParams.algorithm = SegmentAlgorithm::Curvature;
        curvParams.mode = SegmentMode::LinesAndArcs;
        SegmentationResult curvResult = SegmentContour(roundRect, curvParams);

        // Error-based
        SegmentParams errParams;
        errParams.algorithm = SegmentAlgorithm::ErrorBased;
        errParams.mode = SegmentMode::LinesAndArcs;
        SegmentationResult errResult = SegmentContour(roundRect, errParams);

        // Hybrid
        SegmentParams hybridParams;
        hybridParams.algorithm = SegmentAlgorithm::Hybrid;
        hybridParams.mode = SegmentMode::LinesAndArcs;
        SegmentationResult hybridResult = SegmentContour(roundRect, hybridParams);

        printf("   Curvature-based: %zu primitives (L:%zu, A:%zu)\n",
               curvResult.primitives.size(), curvResult.LineCount(), curvResult.ArcCount());
        printf("   Error-based:     %zu primitives (L:%zu, A:%zu)\n",
               errResult.primitives.size(), errResult.LineCount(), errResult.ArcCount());
        printf("   Hybrid:          %zu primitives (L:%zu, A:%zu)\n",
               hybridResult.primitives.size(), hybridResult.LineCount(), hybridResult.ArcCount());
    }

    // 6. 线段拟合质量 / Line fitting quality
    printf("\n6. Line fitting to contour segment:\n");
    {
        QContour line;
        for (int i = 0; i < 50; ++i) {
            // 带噪声的直线 / Line with noise
            line.AddPoint(i * 2.0 + (rand() % 100 - 50) * 0.01,
                         i * 1.5 + 10.0 + (rand() % 100 - 50) * 0.01);
        }

        PrimitiveFitResult fit = FitLineToContour(line, 0, line.Size() - 1);

        printf("   Input: %zu points\n", line.Size());
        printf("   Fitted line: (%.2f, %.2f) -> (%.2f, %.2f)\n",
               fit.segment.p1.x, fit.segment.p1.y,
               fit.segment.p2.x, fit.segment.p2.y);
        printf("   RMS error: %.4f px\n", fit.error);
        printf("   Max error: %.4f px\n", fit.maxError);
    }

    // 7. 分割结果统计 / Segmentation statistics
    printf("\n7. Segmentation result statistics:\n");
    {
        QContour roundRect = CreateRoundedRect(100, 70, 15, {150, 120});

        SegmentParams params;
        params.mode = SegmentMode::LinesAndArcs;

        SegmentationResult result = SegmentContour(roundRect, params);

        printf("   Total primitives: %zu\n", result.primitives.size());
        printf("   Lines: %zu, Arcs: %zu\n", result.LineCount(), result.ArcCount());
        printf("   Total fitting error: %.4f\n", result.totalError);
        printf("   Coverage ratio: %.2f%%\n", result.coverageRatio * 100);

        // 提取所有线段和圆弧 / Extract all lines and arcs
        std::vector<Segment2d> lines = result.GetLines();
        std::vector<Arc2d> arcs = result.GetArcs();

        double totalLineLength = 0;
        for (const auto& seg : lines) {
            totalLineLength += seg.Length();
        }

        double totalArcLength = 0;
        for (const auto& arc : arcs) {
            totalArcLength += arc.Length();
        }

        printf("   Total line length: %.2f\n", totalLineLength);
        printf("   Total arc length: %.2f\n", totalArcLength);
    }

    printf("\n=== Done ===\n");
    return 0;
}
