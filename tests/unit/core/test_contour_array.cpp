#include <gtest/gtest.h>
#include <QiVision/Core/QContourArray.h>
#include <QiVision/Core/QMatrix.h>
#include <QiVision/Core/Constants.h>

using namespace Qi::Vision;

// =============================================================================
// Constructor Tests
// =============================================================================

TEST(QContourArrayTest, DefaultConstructor) {
    QContourArray arr;
    EXPECT_TRUE(arr.Empty());
    EXPECT_EQ(arr.Size(), 0u);
}

TEST(QContourArrayTest, ConstructFromSingleContour) {
    QContour c;
    c.AddPoint(0, 0);
    c.AddPoint(1, 1);

    QContourArray arr(c);
    EXPECT_EQ(arr.Size(), 1u);
    EXPECT_EQ(arr[0].Size(), 2u);
}

TEST(QContourArrayTest, ConstructFromVector) {
    std::vector<QContour> contours;
    contours.emplace_back();
    contours[0].AddPoint(0, 0);
    contours.emplace_back();
    contours[1].AddPoint(1, 1);

    QContourArray arr(contours);
    EXPECT_EQ(arr.Size(), 2u);
}

// =============================================================================
// Container Operations Tests
// =============================================================================

TEST(QContourArrayTest, AtAccess) {
    QContourArray arr;
    QContour c;
    c.AddPoint(0, 0);
    arr.Add(c);

    EXPECT_EQ(arr.At(0).Size(), 1u);
    EXPECT_THROW(arr.At(5), std::out_of_range);
}

TEST(QContourArrayTest, FrontBack) {
    QContourArray arr;
    QContour c1, c2;
    c1.AddPoint(1, 1);
    c2.AddPoint(2, 2);
    arr.Add(c1);
    arr.Add(c2);

    EXPECT_DOUBLE_EQ(arr.Front()[0].x, 1.0);
    EXPECT_DOUBLE_EQ(arr.Back()[0].x, 2.0);
}

TEST(QContourArrayTest, FrontBackEmpty) {
    QContourArray arr;
    EXPECT_THROW(arr.Front(), std::out_of_range);
    EXPECT_THROW(arr.Back(), std::out_of_range);
}

// =============================================================================
// Modification Tests
// =============================================================================

TEST(QContourArrayTest, AddContour) {
    QContourArray arr;
    QContour c;
    c.AddPoint(0, 0);
    arr.Add(c);

    EXPECT_EQ(arr.Size(), 1u);
}

TEST(QContourArrayTest, AddArray) {
    QContourArray arr1, arr2;
    QContour c1, c2;
    c1.AddPoint(0, 0);
    c2.AddPoint(1, 1);
    arr1.Add(c1);
    arr2.Add(c2);

    arr1.Add(arr2);
    EXPECT_EQ(arr1.Size(), 2u);
}

TEST(QContourArrayTest, InsertRemove) {
    QContourArray arr;
    QContour c1, c2, c3;
    c1.AddPoint(0, 0);
    c2.AddPoint(1, 1);
    c3.AddPoint(2, 2);

    arr.Add(c1);
    arr.Add(c3);
    arr.Insert(1, c2);

    EXPECT_EQ(arr.Size(), 3u);
    EXPECT_DOUBLE_EQ(arr[1][0].x, 1.0);

    arr.Remove(1);
    EXPECT_EQ(arr.Size(), 2u);
    EXPECT_DOUBLE_EQ(arr[1][0].x, 2.0);
}

TEST(QContourArrayTest, RemoveIf) {
    QContourArray arr;
    QContour c1, c2;
    c1.AddPoint(0, 0);
    c1.AddPoint(1, 1);  // Length ~1.41
    c2.AddPoint(0, 0);
    c2.AddPoint(10, 0);  // Length 10

    arr.Add(c1);
    arr.Add(c2);

    arr.RemoveIf([](const QContour& c) { return c.Length() < 5; });
    EXPECT_EQ(arr.Size(), 1u);
    EXPECT_DOUBLE_EQ(arr[0].Length(), 10.0);
}

TEST(QContourArrayTest, Clear) {
    QContourArray arr;
    QContour c;
    c.AddPoint(0, 0);
    arr.Add(c);
    arr.Add(c);

    arr.Clear();
    EXPECT_TRUE(arr.Empty());
}

// =============================================================================
// Hierarchy Tests
// =============================================================================

TEST(QContourArrayTest, BuildHierarchy) {
    // Create outer square and inner square
    // Note: inner's centroid (35, 35) must be clearly inside outer (0, 0, 100, 100)
    // but outer's centroid (50, 50) must NOT be inside inner (20, 20, 30, 30)
    QContour outer = QContour::FromRectangle(Rect2d(0, 0, 100, 100));
    QContour inner = QContour::FromRectangle(Rect2d(20, 20, 30, 30));

    QContourArray arr;
    arr.Add(outer);
    arr.Add(inner);

    arr.BuildHierarchy();

    // Inner should be child of outer
    EXPECT_EQ(arr.GetParent(0), -1);  // outer has no parent
    EXPECT_EQ(arr.GetParent(1), 0);   // inner's parent is outer

    auto roots = arr.GetRootContours();
    EXPECT_EQ(roots.size(), 1u);
    EXPECT_EQ(roots[0], 0u);

    auto children = arr.GetChildren(0);
    EXPECT_EQ(children.size(), 1u);
    EXPECT_EQ(children[0], 1u);
}

TEST(QContourArrayTest, GetDepth) {
    // Create 3-level hierarchy with non-overlapping centroids
    // outer: rect(0,0,100,100) center at (50, 50)
    // middle: rect(5,5,20,20) center at (15, 15) - inside outer, outer's center NOT inside it
    // inner: rect(8,8,8,8) center at (12, 12) - inside middle, middle's center NOT inside it
    QContour outer = QContour::FromRectangle(Rect2d(0, 0, 100, 100));   // center (50, 50)
    QContour middle = QContour::FromRectangle(Rect2d(5, 5, 20, 20));    // center (15, 15)
    QContour inner = QContour::FromRectangle(Rect2d(8, 8, 8, 8));       // center (12, 12)

    QContourArray arr;
    arr.Add(outer);
    arr.Add(middle);
    arr.Add(inner);
    arr.BuildHierarchy();

    EXPECT_EQ(arr.GetDepth(0), 0);  // outer is root
    EXPECT_EQ(arr.GetDepth(1), 1);  // middle is depth 1
    EXPECT_EQ(arr.GetDepth(2), 2);  // inner is depth 2
}

TEST(QContourArrayTest, FlattenHierarchy) {
    QContour outer = QContour::FromRectangle(Rect2d(0, 0, 100, 100));
    QContour inner = QContour::FromRectangle(Rect2d(20, 20, 30, 30));

    QContourArray arr;
    arr.Add(outer);
    arr.Add(inner);
    arr.BuildHierarchy();

    arr.FlattenHierarchy();

    EXPECT_EQ(arr.GetParent(0), -1);
    EXPECT_EQ(arr.GetParent(1), -1);
    EXPECT_TRUE(arr.GetChildren(0).empty());
}

// =============================================================================
// Selection Tests
// =============================================================================

TEST(QContourArrayTest, SelectByLength) {
    QContourArray arr;
    QContour c1, c2;
    c1.AddPoint(0, 0);
    c1.AddPoint(5, 0);  // Length 5
    c2.AddPoint(0, 0);
    c2.AddPoint(20, 0);  // Length 20

    arr.Add(c1);
    arr.Add(c2);

    auto selected = arr.SelectByLength(10, 100);
    EXPECT_EQ(selected.Size(), 1u);
    EXPECT_DOUBLE_EQ(selected[0].Length(), 20.0);
}

TEST(QContourArrayTest, SelectByArea) {
    QContourArray arr;
    arr.Add(QContour::FromRectangle(Rect2d(0, 0, 10, 10)));   // Area 100
    arr.Add(QContour::FromRectangle(Rect2d(0, 0, 20, 20)));   // Area 400

    auto selected = arr.SelectByArea(200, 500);
    EXPECT_EQ(selected.Size(), 1u);
    EXPECT_NEAR(selected[0].Area(), 400.0, 1e-10);
}

TEST(QContourArrayTest, SelectClosedOpen) {
    QContourArray arr;
    QContour closed = QContour::FromRectangle(Rect2d(0, 0, 10, 10));
    QContour open;
    open.AddPoint(0, 0);
    open.AddPoint(10, 0);

    arr.Add(closed);
    arr.Add(open);

    EXPECT_EQ(arr.SelectClosed().Size(), 1u);
    EXPECT_EQ(arr.SelectOpen().Size(), 1u);
}

TEST(QContourArrayTest, SelectByIndex) {
    QContourArray arr;
    for (int i = 0; i < 5; ++i) {
        QContour c;
        c.AddPoint(static_cast<double>(i), 0);
        arr.Add(c);
    }

    auto selected = arr.SelectByIndex({1, 3});
    EXPECT_EQ(selected.Size(), 2u);
    EXPECT_DOUBLE_EQ(selected[0][0].x, 1.0);
    EXPECT_DOUBLE_EQ(selected[1][0].x, 3.0);
}

// =============================================================================
// Geometric Properties Tests
// =============================================================================

TEST(QContourArrayTest, TotalLength) {
    QContourArray arr;
    QContour c1, c2;
    c1.AddPoint(0, 0);
    c1.AddPoint(10, 0);
    c2.AddPoint(0, 0);
    c2.AddPoint(20, 0);

    arr.Add(c1);
    arr.Add(c2);

    EXPECT_DOUBLE_EQ(arr.TotalLength(), 30.0);
}

TEST(QContourArrayTest, TotalArea) {
    QContourArray arr;
    arr.Add(QContour::FromRectangle(Rect2d(0, 0, 10, 10)));
    arr.Add(QContour::FromRectangle(Rect2d(0, 0, 20, 20)));

    EXPECT_DOUBLE_EQ(arr.TotalArea(), 500.0);
}

TEST(QContourArrayTest, BoundingBox) {
    QContourArray arr;
    QContour c1, c2;
    c1.AddPoint(0, 0);
    c1.AddPoint(10, 10);
    c2.AddPoint(5, 5);
    c2.AddPoint(20, 15);

    arr.Add(c1);
    arr.Add(c2);

    Rect2d bbox = arr.BoundingBox();
    EXPECT_DOUBLE_EQ(bbox.x, 0.0);
    EXPECT_DOUBLE_EQ(bbox.y, 0.0);
    EXPECT_DOUBLE_EQ(bbox.width, 20.0);
    EXPECT_DOUBLE_EQ(bbox.height, 15.0);
}

TEST(QContourArrayTest, Centroid) {
    QContourArray arr;
    QContour c;
    c.AddPoint(0, 0);
    c.AddPoint(10, 0);
    c.AddPoint(10, 10);
    c.AddPoint(0, 10);
    arr.Add(c);

    Point2d center = arr.Centroid();
    EXPECT_DOUBLE_EQ(center.x, 5.0);
    EXPECT_DOUBLE_EQ(center.y, 5.0);
}

// =============================================================================
// Transformation Tests
// =============================================================================

TEST(QContourArrayTest, Translate) {
    QContourArray arr;
    QContour c;
    c.AddPoint(0, 0);
    c.AddPoint(10, 0);
    arr.Add(c);

    auto translated = arr.Translate(5, 10);
    EXPECT_DOUBLE_EQ(translated[0][0].x, 5.0);
    EXPECT_DOUBLE_EQ(translated[0][0].y, 10.0);
}

TEST(QContourArrayTest, Scale) {
    QContourArray arr;
    arr.Add(QContour::FromRectangle(Rect2d(0, 0, 10, 10)));

    auto scaled = arr.Scale(2.0, 2.0, Point2d{0, 0});
    Rect2d bbox = scaled.BoundingBox();
    EXPECT_DOUBLE_EQ(bbox.width, 20.0);
    EXPECT_DOUBLE_EQ(bbox.height, 20.0);
}

TEST(QContourArrayTest, Rotate) {
    QContourArray arr;
    QContour c;
    c.AddPoint(10, 0);
    arr.Add(c);

    auto rotated = arr.Rotate(HALF_PI, Point2d{0, 0});
    EXPECT_NEAR(rotated[0][0].x, 0.0, 1e-10);
    EXPECT_NEAR(rotated[0][0].y, 10.0, 1e-10);
}

TEST(QContourArrayTest, Transform) {
    QContourArray arr;
    QContour c;
    c.AddPoint(0, 0);
    arr.Add(c);

    QMatrix m = QMatrix::Translation(100, 200);
    auto transformed = arr.Transform(m);
    EXPECT_DOUBLE_EQ(transformed[0][0].x, 100.0);
    EXPECT_DOUBLE_EQ(transformed[0][0].y, 200.0);
}

// =============================================================================
// Processing Tests
// =============================================================================

TEST(QContourArrayTest, Smooth) {
    QContourArray arr;
    QContour c;
    for (int i = 0; i < 20; ++i) {
        c.AddPoint(static_cast<double>(i), (i % 2 == 0) ? 1.0 : -1.0);
    }
    arr.Add(c);

    auto smoothed = arr.Smooth(1.0);
    EXPECT_EQ(smoothed.Size(), 1u);
    EXPECT_EQ(smoothed[0].Size(), c.Size());
}

TEST(QContourArrayTest, Simplify) {
    QContourArray arr;
    QContour c;
    for (int i = 0; i <= 100; ++i) {
        c.AddPoint(static_cast<double>(i), 0.0);
    }
    arr.Add(c);

    auto simplified = arr.Simplify(0.1);
    EXPECT_EQ(simplified.Size(), 1u);
    EXPECT_EQ(simplified[0].Size(), 2u);  // Only start and end
}

TEST(QContourArrayTest, CloseOpenAll) {
    QContourArray arr;
    QContour c1, c2;
    c1.AddPoint(0, 0);
    c1.AddPoint(1, 1);
    c2.AddPoint(2, 2);
    c2.AddPoint(3, 3);

    arr.Add(c1);
    arr.Add(c2);

    arr.CloseAll();
    EXPECT_TRUE(arr[0].IsClosed());
    EXPECT_TRUE(arr[1].IsClosed());

    arr.OpenAll();
    EXPECT_FALSE(arr[0].IsClosed());
    EXPECT_FALSE(arr[1].IsClosed());
}

TEST(QContourArrayTest, ReverseAll) {
    QContourArray arr;
    QContour c;
    c.AddPoint(0, 0);
    c.AddPoint(1, 0);
    c.AddPoint(2, 0);
    arr.Add(c);

    arr.ReverseAll();
    EXPECT_DOUBLE_EQ(arr[0][0].x, 2.0);
    EXPECT_DOUBLE_EQ(arr[0][2].x, 0.0);
}

// =============================================================================
// Merging / Splitting Tests
// =============================================================================

TEST(QContourArrayTest, Concatenate) {
    QContourArray arr;
    QContour c1, c2;
    c1.AddPoint(0, 0);
    c1.AddPoint(1, 0);
    c2.AddPoint(2, 0);
    c2.AddPoint(3, 0);

    arr.Add(c1);
    arr.Add(c2);

    QContour merged = arr.Concatenate();
    EXPECT_EQ(merged.Size(), 4u);
}

// =============================================================================
// Iterator Tests
// =============================================================================

TEST(QContourArrayTest, Iterators) {
    QContourArray arr;
    for (int i = 0; i < 3; ++i) {
        QContour c;
        c.AddPoint(static_cast<double>(i), 0);
        arr.Add(c);
    }

    int count = 0;
    for (const auto& c : arr) {
        EXPECT_DOUBLE_EQ(c[0].x, static_cast<double>(count));
        count++;
    }
    EXPECT_EQ(count, 3);
}

// =============================================================================
// Utility Tests
// =============================================================================

TEST(QContourArrayTest, Clone) {
    QContourArray arr;
    QContour c;
    c.AddPoint(0, 0);
    arr.Add(c);

    QContourArray clone = arr.Clone();
    clone.Add(c);

    EXPECT_EQ(arr.Size(), 1u);
    EXPECT_EQ(clone.Size(), 2u);
}

TEST(QContourArrayTest, SortByLength) {
    QContourArray arr;
    QContour c1, c2, c3;
    c1.AddPoint(0, 0);
    c1.AddPoint(5, 0);   // Length 5
    c2.AddPoint(0, 0);
    c2.AddPoint(20, 0);  // Length 20
    c3.AddPoint(0, 0);
    c3.AddPoint(10, 0);  // Length 10

    arr.Add(c1);
    arr.Add(c2);
    arr.Add(c3);

    arr.SortByLength(true);  // Descending
    EXPECT_DOUBLE_EQ(arr[0].Length(), 20.0);
    EXPECT_DOUBLE_EQ(arr[1].Length(), 10.0);
    EXPECT_DOUBLE_EQ(arr[2].Length(), 5.0);

    arr.SortByLength(false);  // Ascending
    EXPECT_DOUBLE_EQ(arr[0].Length(), 5.0);
    EXPECT_DOUBLE_EQ(arr[2].Length(), 20.0);
}

TEST(QContourArrayTest, SortByArea) {
    QContourArray arr;
    arr.Add(QContour::FromRectangle(Rect2d(0, 0, 5, 5)));    // Area 25
    arr.Add(QContour::FromRectangle(Rect2d(0, 0, 20, 20)));  // Area 400
    arr.Add(QContour::FromRectangle(Rect2d(0, 0, 10, 10)));  // Area 100

    arr.SortByArea(true);  // Descending
    EXPECT_DOUBLE_EQ(arr[0].Area(), 400.0);
    EXPECT_DOUBLE_EQ(arr[1].Area(), 100.0);
    EXPECT_DOUBLE_EQ(arr[2].Area(), 25.0);
}

// =============================================================================
// Alias Test
// =============================================================================

TEST(QContourArrayTest, QXldArrayAlias) {
    QXldArray xldArr;
    QXld xld;
    xld.AddPoint(0, 0);
    xldArr.Add(xld);

    EXPECT_EQ(xldArr.Size(), 1u);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(QContourArrayTest, EmptyArray) {
    QContourArray arr;

    EXPECT_DOUBLE_EQ(arr.TotalLength(), 0.0);
    EXPECT_DOUBLE_EQ(arr.TotalArea(), 0.0);

    Rect2d bbox = arr.BoundingBox();
    EXPECT_DOUBLE_EQ(bbox.width, 0.0);
    EXPECT_DOUBLE_EQ(bbox.height, 0.0);

    Point2d center = arr.Centroid();
    EXPECT_DOUBLE_EQ(center.x, 0.0);
    EXPECT_DOUBLE_EQ(center.y, 0.0);
}
