# Internal/Intersection Module Design Document

## Overview

The Intersection module provides geometric intersection calculations between various 2D primitives. It is a fundamental building block for many vision algorithms including:

- Measure: Finding edge intersections for dimensional measurement
- Metrology: Computing constraint-based geometric relationships
- Fitting: Clipping and validation operations
- Matching: Shape boundary analysis

## Dependencies

```
Intersection.h
    |
    +-- Core/Types.h (Point2d, Line2d, Segment2d, Circle2d, Ellipse2d, Arc2d, RotatedRect2d)
    +-- Core/Constants.h (EPSILON, PI, etc.)
    +-- Internal/Geometry2d.h (NormalizeAngle, RotatePoint, etc.)
    +-- Internal/Distance.h (DistancePointToLine, etc. for tangent cases)
```

## Design Principles

1. **Pure Functions**: All functions are stateless with no global state
2. **High Precision**: Use `double` for all calculations
3. **Numerical Stability**: Handle degenerate and near-degenerate cases robustly
4. **Complete Results**: Return all intersection points when multiple exist
5. **Parameter Information**: Optionally return intersection parameters (t for segments, angle for arcs)

## Intersection Result Structures

### Basic Intersection Result

```cpp
struct IntersectionResult {
    bool exists = false;           // True if intersection exists
    Point2d point;                 // Primary intersection point
    double param1 = 0.0;           // Parameter on first primitive
    double param2 = 0.0;           // Parameter on second primitive
};
```

### Multiple Intersection Result

```cpp
struct IntersectionResult2 {
    int count = 0;                 // Number of intersections (0, 1, or 2)
    Point2d point1;                // First intersection point
    Point2d point2;                // Second intersection point
    double param1_1 = 0.0;         // First point param on primitive 1
    double param1_2 = 0.0;         // First point param on primitive 2
    double param2_1 = 0.0;         // Second point param on primitive 1
    double param2_2 = 0.0;         // Second point param on primitive 2

    bool HasIntersection() const { return count > 0; }
    bool HasTwoIntersections() const { return count == 2; }
};
```

### Special Cases

| Case | Handling |
|------|----------|
| No intersection | count = 0, empty result |
| Tangent (1 point) | count = 1, single point |
| Two intersections | count = 2, both points returned |
| Coincident/Overlapping | Return endpoints of overlap, or special status |
| Degenerate input | Return empty result (no exception) |

## Supported Intersection Types

### Category 1: Line/Segment Intersections

| Function | Primitives | Max Intersections | Notes |
|----------|------------|-------------------|-------|
| `IntersectLineLine` | Line-Line | 1 | Parallel lines: no intersection |
| `IntersectLineSegment` | Line-Segment | 1 | Checks segment bounds |
| `IntersectSegmentSegment` | Segment-Segment | 1 | Checks both segment bounds |

### Category 2: Line/Segment with Circle

| Function | Primitives | Max Intersections | Notes |
|----------|------------|-------------------|-------|
| `IntersectLineCircle` | Line-Circle | 2 | Tangent case: 1 point |
| `IntersectSegmentCircle` | Segment-Circle | 2 | Filters by segment parameter |

### Category 3: Line/Segment with Ellipse

| Function | Primitives | Max Intersections | Notes |
|----------|------------|-------------------|-------|
| `IntersectLineEllipse` | Line-Ellipse | 2 | Uses quartic formula |
| `IntersectSegmentEllipse` | Segment-Ellipse | 2 | Filters by segment parameter |

### Category 4: Line/Segment with Arc

| Function | Primitives | Max Intersections | Notes |
|----------|------------|-------------------|-------|
| `IntersectLineArc` | Line-Arc | 2 | Filters by arc angle range |
| `IntersectSegmentArc` | Segment-Arc | 2 | Filters by both constraints |

### Category 5: Circle-Circle

| Function | Primitives | Max Intersections | Notes |
|----------|------------|-------------------|-------|
| `IntersectCircleCircle` | Circle-Circle | 2 | Concentric: no intersection |

### Category 6: Circle-Ellipse

| Function | Primitives | Max Intersections | Notes |
|----------|------------|-------------------|-------|
| `IntersectCircleEllipse` | Circle-Ellipse | 4 | Uses numerical root finding |

### Category 7: Ellipse-Ellipse (Optional)

| Function | Primitives | Max Intersections | Notes |
|----------|------------|-------------------|-------|
| `IntersectEllipseEllipse` | Ellipse-Ellipse | 4 | Complex, numerical method |

### Category 8: Line/Segment with RotatedRect

| Function | Primitives | Max Intersections | Notes |
|----------|------------|-------------------|-------|
| `IntersectLineRotatedRect` | Line-RotatedRect | 2 | Intersect with 4 edges |
| `IntersectSegmentRotatedRect` | Segment-RotatedRect | 2 | Intersect with 4 edges |

## Algorithm Details

### Line-Line Intersection

For two lines:
- Line 1: `a1*x + b1*y + c1 = 0`
- Line 2: `a2*x + b2*y + c2 = 0`

Solve:
```
det = a1*b2 - a2*b1
x = (b1*c2 - b2*c1) / det
y = (a2*c1 - a1*c2) / det
```

If `|det| < epsilon`, lines are parallel.

### Segment Parameter

For segment from P1 to P2, point P lies on segment if:
```
t = (P - P1) . (P2 - P1) / |P2 - P1|^2
valid if 0 <= t <= 1
```

### Line-Circle Intersection

For line `ax + by + c = 0` and circle (cx, cy, r):

1. Distance from center to line: `d = |a*cx + b*cy + c|` (assuming normalized line)
2. If `d > r`: no intersection
3. If `d == r`: tangent, 1 point
4. If `d < r`: 2 points

Intersection points along the line at distance `sqrt(r^2 - d^2)` from the foot of perpendicular.

### Line-Ellipse Intersection

Transform the problem:
1. Translate so ellipse center is at origin
2. Rotate so ellipse axes align with coordinate axes
3. Solve with axis-aligned ellipse
4. Transform intersection points back

For axis-aligned ellipse `x^2/a^2 + y^2/b^2 = 1` and line `px + qy + r = 0`:
Substitute and solve quadratic equation.

### Circle-Circle Intersection

For circles (x1,y1,r1) and (x2,y2,r2):

1. Distance `d = sqrt((x2-x1)^2 + (y2-y1)^2)`
2. If `d > r1 + r2` or `d < |r1 - r2|`: no intersection
3. If `d == r1 + r2` or `d == |r1 - r2|`: tangent (1 point)
4. Otherwise: 2 points

Use the radical line method:
```
a = (r1^2 - r2^2 + d^2) / (2*d)
h = sqrt(r1^2 - a^2)
```

### Arc Angle Filtering

For arc with center (cx, cy), startAngle, sweepAngle:
- Point P is on arc if angle to center is within [startAngle, startAngle + sweepAngle]
- Handle sweep direction (positive = CCW, negative = CW)

### RotatedRect Intersection

1. Get 4 edges as segments
2. Intersect line/segment with each edge
3. Collect valid intersection points

## API Specification

### Line-Line

```cpp
/**
 * @brief Compute intersection of two infinite lines
 * @param line1 First line (normalized: a^2 + b^2 = 1)
 * @param line2 Second line
 * @return Intersection result (exists=false if parallel)
 */
IntersectionResult IntersectLineLine(const Line2d& line1, const Line2d& line2);
```

### Line-Segment

```cpp
/**
 * @brief Compute intersection of infinite line and segment
 * @param line Infinite line
 * @param segment Line segment
 * @return Intersection result with param1=line param, param2=segment t in [0,1]
 */
IntersectionResult IntersectLineSegment(const Line2d& line, const Segment2d& segment);
```

### Segment-Segment

```cpp
/**
 * @brief Compute intersection of two line segments
 * @param seg1 First segment
 * @param seg2 Second segment
 * @return Intersection result with both t parameters in [0,1]
 */
IntersectionResult IntersectSegmentSegment(const Segment2d& seg1, const Segment2d& seg2);
```

### Line-Circle

```cpp
/**
 * @brief Compute intersection of infinite line and circle
 * @param line Infinite line
 * @param circle Circle
 * @return Up to 2 intersection points with angle parameters
 */
IntersectionResult2 IntersectLineCircle(const Line2d& line, const Circle2d& circle);
```

### Segment-Circle

```cpp
/**
 * @brief Compute intersection of segment and circle
 * @param segment Line segment
 * @param circle Circle
 * @return Up to 2 intersection points within segment
 */
IntersectionResult2 IntersectSegmentCircle(const Segment2d& segment, const Circle2d& circle);
```

### Line-Ellipse

```cpp
/**
 * @brief Compute intersection of infinite line and ellipse
 * @param line Infinite line
 * @param ellipse Ellipse (may be rotated)
 * @return Up to 2 intersection points with theta parameters
 */
IntersectionResult2 IntersectLineEllipse(const Line2d& line, const Ellipse2d& ellipse);
```

### Segment-Ellipse

```cpp
/**
 * @brief Compute intersection of segment and ellipse
 * @param segment Line segment
 * @param ellipse Ellipse (may be rotated)
 * @return Up to 2 intersection points within segment
 */
IntersectionResult2 IntersectSegmentEllipse(const Segment2d& segment, const Ellipse2d& ellipse);
```

### Line-Arc

```cpp
/**
 * @brief Compute intersection of infinite line and circular arc
 * @param line Infinite line
 * @param arc Circular arc
 * @return Up to 2 intersection points within arc range
 */
IntersectionResult2 IntersectLineArc(const Line2d& line, const Arc2d& arc);
```

### Segment-Arc

```cpp
/**
 * @brief Compute intersection of segment and circular arc
 * @param segment Line segment
 * @param arc Circular arc
 * @return Up to 2 intersection points within both segment and arc
 */
IntersectionResult2 IntersectSegmentArc(const Segment2d& segment, const Arc2d& arc);
```

### Circle-Circle

```cpp
/**
 * @brief Compute intersection of two circles
 * @param circle1 First circle
 * @param circle2 Second circle
 * @return Up to 2 intersection points with angle parameters on each circle
 */
IntersectionResult2 IntersectCircleCircle(const Circle2d& circle1, const Circle2d& circle2);
```

### Circle-Ellipse

```cpp
/**
 * @brief Compute intersection of circle and ellipse
 * @param circle Circle
 * @param ellipse Ellipse
 * @param maxIterations Maximum Newton iterations for numerical solving
 * @param tolerance Convergence tolerance
 * @return Vector of intersection points (up to 4)
 */
std::vector<Point2d> IntersectCircleEllipse(
    const Circle2d& circle,
    const Ellipse2d& ellipse,
    int maxIterations = 20,
    double tolerance = 1e-10);
```

### Ellipse-Ellipse (Optional)

```cpp
/**
 * @brief Compute intersection of two ellipses
 * @param ellipse1 First ellipse
 * @param ellipse2 Second ellipse
 * @param maxIterations Maximum iterations for numerical solving
 * @param tolerance Convergence tolerance
 * @return Vector of intersection points (up to 4)
 * @note Complex algorithm, may be slower than other intersections
 */
std::vector<Point2d> IntersectEllipseEllipse(
    const Ellipse2d& ellipse1,
    const Ellipse2d& ellipse2,
    int maxIterations = 50,
    double tolerance = 1e-10);
```

### Line-RotatedRect

```cpp
/**
 * @brief Compute intersection of infinite line and rotated rectangle
 * @param line Infinite line
 * @param rect Rotated rectangle
 * @return Up to 2 intersection points on rectangle boundary
 */
IntersectionResult2 IntersectLineRotatedRect(const Line2d& line, const RotatedRect2d& rect);
```

### Segment-RotatedRect

```cpp
/**
 * @brief Compute intersection of segment and rotated rectangle
 * @param segment Line segment
 * @param rect Rotated rectangle
 * @return Up to 2 intersection points within segment and on rectangle boundary
 */
IntersectionResult2 IntersectSegmentRotatedRect(const Segment2d& segment, const RotatedRect2d& rect);
```

## Batch Operations

```cpp
/**
 * @brief Compute intersections of a line with multiple segments
 */
std::vector<IntersectionResult> IntersectLineWithSegments(
    const Line2d& line,
    const std::vector<Segment2d>& segments);

/**
 * @brief Compute intersections of a line with a polyline/contour
 */
std::vector<IntersectionResult> IntersectLineWithContour(
    const Line2d& line,
    const std::vector<Point2d>& contourPoints,
    bool closed = false);
```

## Utility Functions

```cpp
/**
 * @brief Check if two segments overlap (collinear with shared portion)
 */
bool SegmentsOverlap(const Segment2d& seg1, const Segment2d& seg2,
                     double tolerance = GEOM_TOLERANCE);

/**
 * @brief Get the overlapping portion of two collinear segments
 */
std::optional<Segment2d> SegmentOverlap(const Segment2d& seg1, const Segment2d& seg2,
                                        double tolerance = GEOM_TOLERANCE);

/**
 * @brief Clip a segment to a rectangle
 */
std::optional<Segment2d> ClipSegmentToRect(const Segment2d& segment, const Rect2d& rect);

/**
 * @brief Clip a segment to a rotated rectangle
 */
std::optional<Segment2d> ClipSegmentToRotatedRect(const Segment2d& segment,
                                                   const RotatedRect2d& rect);

/**
 * @brief Check if line passes through circle (fast check without finding points)
 */
bool LineIntersectsCircle(const Line2d& line, const Circle2d& circle);

/**
 * @brief Check if segment intersects circle (fast check)
 */
bool SegmentIntersectsCircle(const Segment2d& segment, const Circle2d& circle);
```

## Numerical Considerations

### Tolerance Values

| Context | Tolerance | Constant |
|---------|-----------|----------|
| Parallel/perpendicular check | 1e-9 | `GEOM_TOLERANCE` |
| Point equality | 1e-9 | `GEOM_TOLERANCE` |
| Determinant for singular | 1e-12 | `SINGULAR_TOLERANCE` |
| Arc angle comparison | 1e-9 | `ANGLE_TOLERANCE` |

### Degenerate Case Handling

| Case | Behavior |
|------|----------|
| Zero-length segment | Treat as point, check point-primitive containment |
| Zero-radius circle | Treat as point |
| Zero-area ellipse | Degenerate to line or point |
| Identical primitives | Return special "coincident" status |

## Performance Considerations

1. **Early Rejection**: Use bounding box tests before expensive calculations
2. **Cached Values**: Precompute sin/cos for rotated primitives
3. **Avoid sqrt**: Use squared distances where possible
4. **Branch Prediction**: Order checks by probability

## Testing Strategy

### Unit Tests

1. **Basic Cases**: Standard non-degenerate intersections
2. **Boundary Cases**: Tangent, endpoints, zero parameters
3. **Degenerate Cases**: Parallel lines, concentric circles
4. **Numerical Edge Cases**: Near-parallel, near-tangent

### Precision Tests

| Test | Expected Precision |
|------|-------------------|
| Line-Line intersection point | < 1e-12 |
| Line-Circle tangent detection | < 1e-9 |
| Ellipse-Ellipse (numerical) | < 1e-8 |

### Test Cases Examples

```cpp
// Line-Line: Standard intersection
TEST(IntersectLineLine, Standard) {
    Line2d l1 = Line2d::FromPoints({0, 0}, {1, 0});  // horizontal
    Line2d l2 = Line2d::FromPoints({0, -1}, {0, 1}); // vertical
    auto result = IntersectLineLine(l1, l2);
    EXPECT_TRUE(result.exists);
    EXPECT_NEAR(result.point.x, 0.0, 1e-12);
    EXPECT_NEAR(result.point.y, 0.0, 1e-12);
}

// Line-Line: Parallel (no intersection)
TEST(IntersectLineLine, Parallel) {
    Line2d l1 = Line2d::FromPoints({0, 0}, {1, 0});
    Line2d l2 = Line2d::FromPoints({0, 1}, {1, 1});
    auto result = IntersectLineLine(l1, l2);
    EXPECT_FALSE(result.exists);
}

// Circle-Circle: Two points
TEST(IntersectCircleCircle, TwoPoints) {
    Circle2d c1({0, 0}, 1.0);
    Circle2d c2({1, 0}, 1.0);
    auto result = IntersectCircleCircle(c1, c2);
    EXPECT_EQ(result.count, 2);
    EXPECT_NEAR(result.point1.x, 0.5, 1e-12);
    EXPECT_NEAR(result.point2.x, 0.5, 1e-12);
}
```

## File Structure

```
include/QiVision/Internal/
    Intersection.h          # Header with all declarations

src/Internal/
    Intersection.cpp        # Implementation

tests/unit/Internal/
    IntersectionTest.cpp    # Unit tests

tests/accuracy/Internal/
    IntersectionAccuracyTest.cpp  # Precision tests
```

## Changelog

- 2026-01-06: Initial design document created
