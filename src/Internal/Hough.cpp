/**
 * @file Hough.cpp
 * @brief Hough Transform implementation
 */

#include <QiVision/Internal/Hough.h>
#include <QiVision/Internal/Gradient.h>
#include <QiVision/Internal/Fitting.h>
#include <QiVision/Platform/Random.h>

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

namespace Qi::Vision::Internal {

namespace {
    constexpr double PI = 3.14159265358979323846;
    constexpr double EPSILON = 1e-10;

    // Precompute sin/cos tables for efficiency
    struct SinCosTable {
        std::vector<double> sinTable;
        std::vector<double> cosTable;

        void Init(double thetaMin, double thetaMax, double thetaStep) {
            int numAngles = static_cast<int>((thetaMax - thetaMin) / thetaStep) + 1;
            sinTable.resize(numAngles);
            cosTable.resize(numAngles);

            for (int i = 0; i < numAngles; ++i) {
                double theta = thetaMin + i * thetaStep;
                sinTable[i] = std::sin(theta);
                cosTable[i] = std::cos(theta);
            }
        }
    };

    // Hash for point coordinates (for probabilistic Hough)
    struct PointHash {
        size_t operator()(const std::pair<int, int>& p) const {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 16);
        }
    };
}

// =============================================================================
// HoughLine Methods
// =============================================================================

Line2d HoughLine::ToLine2d() const {
    // Line equation: x*cos(theta) + y*sin(theta) = rho
    // Convert to ax + by + c = 0 form: cos(theta)*x + sin(theta)*y - rho = 0
    double a = std::cos(theta);
    double b = std::sin(theta);
    double c = -rho;
    return Line2d(a, b, c);
}

std::pair<Point2d, Point2d> HoughLine::GetTwoPoints(double length) const {
    // Get a point on the line
    double x0 = rho * std::cos(theta);
    double y0 = rho * std::sin(theta);

    // Direction perpendicular to normal (along the line)
    double dx = -std::sin(theta);
    double dy = std::cos(theta);

    Point2d p1(x0 - length * 0.5 * dx, y0 - length * 0.5 * dy);
    Point2d p2(x0 + length * 0.5 * dx, y0 + length * 0.5 * dy);

    return {p1, p2};
}

// =============================================================================
// HoughLineSegment Methods
// =============================================================================

double HoughLineSegment::Length() const {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return std::sqrt(dx * dx + dy * dy);
}

double HoughLineSegment::Angle() const {
    return std::atan2(p2.y - p1.y, p2.x - p1.x);
}

// =============================================================================
// HoughAccumulator Methods
// =============================================================================

int HoughAccumulator::GetRhoIndex(double rho) const {
    return static_cast<int>((rho - rhoMin) / rhoStep + 0.5);
}

int HoughAccumulator::GetThetaIndex(double theta) const {
    return static_cast<int>((theta - thetaMin) / thetaStep + 0.5);
}

// =============================================================================
// Standard Hough Line Transform
// =============================================================================

HoughAccumulator GetHoughAccumulator(const std::vector<Point2d>& points,
                                      int imageWidth, int imageHeight,
                                      const HoughLineParams& params) {
    HoughAccumulator acc;

    // Calculate rho range
    double diagonal = std::sqrt(static_cast<double>(imageWidth * imageWidth +
                                                     imageHeight * imageHeight));
    acc.rhoMin = -diagonal;
    acc.rhoMax = diagonal;
    acc.rhoStep = params.rhoResolution;
    acc.thetaMin = 0.0;
    acc.thetaMax = PI;
    acc.thetaStep = params.thetaResolution;

    int numRho = static_cast<int>((acc.rhoMax - acc.rhoMin) / acc.rhoStep) + 1;
    int numTheta = static_cast<int>((acc.thetaMax - acc.thetaMin) / acc.thetaStep) + 1;

    acc.data = MatX::Zero(numRho, numTheta);

    // Precompute sin/cos tables
    SinCosTable table;
    table.Init(acc.thetaMin, acc.thetaMax, acc.thetaStep);

    // Vote for each point
    for (const auto& pt : points) {
        for (int t = 0; t < numTheta; ++t) {
            double rho = pt.x * table.cosTable[t] + pt.y * table.sinTable[t];
            int rhoIdx = static_cast<int>((rho - acc.rhoMin) / acc.rhoStep + 0.5);

            if (rhoIdx >= 0 && rhoIdx < numRho) {
                acc.data(rhoIdx, t) += 1.0;
            }
        }
    }

    return acc;
}

HoughAccumulator GetHoughAccumulator(const QImage& edgeImage,
                                      const HoughLineParams& params) {
    // Extract edge points
    std::vector<Point2d> points;

    if (edgeImage.Empty()) {
        return HoughAccumulator();
    }

    int width = edgeImage.Width();
    int height = edgeImage.Height();

    for (int y = 0; y < height; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(edgeImage.RowPtr(y));
        for (int x = 0; x < width; ++x) {
            if (row[x] > 0) {
                points.emplace_back(static_cast<double>(x), static_cast<double>(y));
            }
        }
    }

    return GetHoughAccumulator(points, width, height, params);
}

std::vector<HoughLine> FindAccumulatorPeaks(const HoughAccumulator& accumulator,
                                             double threshold,
                                             int numPeaks,
                                             int minDistance) {
    std::vector<HoughLine> lines;

    if (accumulator.data.Rows() == 0 || accumulator.data.Cols() == 0) {
        return lines;
    }

    int numRho = accumulator.data.Rows();
    int numTheta = accumulator.data.Cols();

    // Find max value for relative threshold
    double maxVal = 0.0;
    for (int r = 0; r < numRho; ++r) {
        for (int t = 0; t < numTheta; ++t) {
            maxVal = std::max(maxVal, accumulator.data(r, t));
        }
    }

    if (maxVal < EPSILON) {
        return lines;
    }

    // Collect all peaks above threshold
    struct Peak {
        int rhoIdx;
        int thetaIdx;
        double score;

        bool operator<(const Peak& other) const {
            return score < other.score;  // For max-heap via std::greater
        }
    };

    std::vector<Peak> peaks;

    for (int r = 1; r < numRho - 1; ++r) {
        for (int t = 1; t < numTheta - 1; ++t) {
            double val = accumulator.data(r, t);

            if (val < threshold * maxVal) continue;

            // Check if local maximum (8-connected)
            bool isMax = true;
            for (int dr = -1; dr <= 1 && isMax; ++dr) {
                for (int dt = -1; dt <= 1 && isMax; ++dt) {
                    if (dr == 0 && dt == 0) continue;
                    if (accumulator.data(r + dr, t + dt) > val) {
                        isMax = false;
                    }
                }
            }

            if (isMax) {
                peaks.push_back({r, t, val});
            }
        }
    }

    // Sort by score (descending)
    std::sort(peaks.begin(), peaks.end(), [](const Peak& a, const Peak& b) {
        return a.score > b.score;
    });

    // Non-maximum suppression
    std::vector<bool> suppressed(peaks.size(), false);

    for (size_t i = 0; i < peaks.size(); ++i) {
        if (suppressed[i]) continue;

        HoughLine line;
        line.rho = accumulator.GetRho(peaks[i].rhoIdx);
        line.theta = accumulator.GetTheta(peaks[i].thetaIdx);
        line.score = peaks[i].score;
        lines.push_back(line);

        if (numPeaks > 0 && static_cast<int>(lines.size()) >= numPeaks) {
            break;
        }

        // Suppress nearby peaks
        for (size_t j = i + 1; j < peaks.size(); ++j) {
            if (suppressed[j]) continue;

            int dr = std::abs(peaks[i].rhoIdx - peaks[j].rhoIdx);
            int dt = std::abs(peaks[i].thetaIdx - peaks[j].thetaIdx);

            if (dr < minDistance && dt < minDistance) {
                suppressed[j] = true;
            }
        }
    }

    return lines;
}

std::vector<HoughLine> HoughLines(const std::vector<Point2d>& points,
                                   int imageWidth, int imageHeight,
                                   const HoughLineParams& params) {
    HoughAccumulator acc = GetHoughAccumulator(points, imageWidth, imageHeight, params);

    double threshold = params.thresholdIsRatio ? params.threshold : params.threshold / 100.0;

    std::vector<HoughLine> lines = FindAccumulatorPeaks(
        acc,
        threshold,
        params.maxLines,
        static_cast<int>(params.minDistance / params.rhoResolution)
    );

    if (params.suppressOverlapping) {
        lines = MergeHoughLines(lines, params.minDistance, params.thetaResolution * 5);
    }

    return lines;
}

std::vector<HoughLine> HoughLines(const QImage& edgeImage,
                                   const HoughLineParams& params) {
    if (edgeImage.Empty()) {
        return {};
    }

    // Extract edge points
    std::vector<Point2d> points;
    int width = edgeImage.Width();
    int height = edgeImage.Height();

    for (int y = 0; y < height; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(edgeImage.RowPtr(y));
        for (int x = 0; x < width; ++x) {
            if (row[x] > 0) {
                points.emplace_back(static_cast<double>(x), static_cast<double>(y));
            }
        }
    }

    return HoughLines(points, width, height, params);
}

// =============================================================================
// Probabilistic Hough Transform
// =============================================================================

std::vector<HoughLineSegment> HoughLinesP(const std::vector<Point2d>& points,
                                           int imageWidth, int imageHeight,
                                           const HoughLineProbParams& params) {
    std::vector<HoughLineSegment> segments;

    if (points.empty()) {
        return segments;
    }

    // Calculate rho range
    double diagonal = std::sqrt(static_cast<double>(imageWidth * imageWidth +
                                                     imageHeight * imageHeight));
    double rhoMin = -diagonal;
    double rhoMax = diagonal;

    int numRho = static_cast<int>((rhoMax - rhoMin) / params.rhoResolution) + 1;
    int numTheta = static_cast<int>(PI / params.thetaResolution) + 1;

    // Accumulator
    MatX acc = MatX::Zero(numRho, numTheta);

    // Sin/cos tables
    SinCosTable table;
    table.Init(0.0, PI, params.thetaResolution);

    // Track which points have been used
    std::vector<bool> used(points.size(), false);

    // Random order for point processing
    std::vector<int> indices(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        indices[i] = static_cast<int>(i);
    }

    // Simple shuffle using Platform::Random
    for (size_t i = points.size() - 1; i > 0; --i) {
        int j = Platform::RandomInt(0, static_cast<int>(i));
        std::swap(indices[i], indices[j]);
    }

    for (int idx : indices) {
        if (used[idx]) continue;

        const Point2d& pt = points[idx];

        // Vote
        for (int t = 0; t < numTheta; ++t) {
            double rho = pt.x * table.cosTable[t] + pt.y * table.sinTable[t];
            int rhoIdx = static_cast<int>((rho - rhoMin) / params.rhoResolution + 0.5);

            if (rhoIdx >= 0 && rhoIdx < numRho) {
                acc(rhoIdx, t) += 1.0;
            }
        }

        // Check for line
        for (int t = 0; t < numTheta; ++t) {
            double rho = pt.x * table.cosTable[t] + pt.y * table.sinTable[t];
            int rhoIdx = static_cast<int>((rho - rhoMin) / params.rhoResolution + 0.5);

            if (rhoIdx < 0 || rhoIdx >= numRho) continue;
            if (acc(rhoIdx, t) < params.threshold) continue;

            // Found potential line - extract segment
            std::vector<int> linePoints;
            double cosT = table.cosTable[t];
            double sinT = table.sinTable[t];

            // Find all points on this line
            for (size_t i = 0; i < points.size(); ++i) {
                if (used[i]) continue;

                double r = points[i].x * cosT + points[i].y * sinT;
                if (std::abs(r - rho) < params.rhoResolution * 2) {
                    linePoints.push_back(static_cast<int>(i));
                }
            }

            if (linePoints.size() < 2) continue;

            // Sort points along line direction
            double dx = -sinT;
            double dy = cosT;

            std::sort(linePoints.begin(), linePoints.end(), [&](int a, int b) {
                double projA = points[a].x * dx + points[a].y * dy;
                double projB = points[b].x * dx + points[b].y * dy;
                return projA < projB;
            });

            // Find continuous segments
            size_t segStart = 0;
            for (size_t i = 1; i <= linePoints.size(); ++i) {
                bool endSegment = (i == linePoints.size());

                if (!endSegment) {
                    double dist = std::abs(
                        (points[linePoints[i]].x - points[linePoints[i-1]].x) * dx +
                        (points[linePoints[i]].y - points[linePoints[i-1]].y) * dy
                    );
                    if (dist > params.maxLineGap) {
                        endSegment = true;
                    }
                }

                if (endSegment && i > segStart) {
                    // Check segment length
                    const Point2d& p1 = points[linePoints[segStart]];
                    const Point2d& p2 = points[linePoints[i-1]];
                    double len = std::sqrt((p2.x - p1.x) * (p2.x - p1.x) +
                                           (p2.y - p1.y) * (p2.y - p1.y));

                    if (len >= params.minLineLength) {
                        segments.emplace_back(p1, p2, static_cast<double>(i - segStart));

                        // Mark points as used
                        for (size_t j = segStart; j < i; ++j) {
                            used[linePoints[j]] = true;
                        }

                        // Remove votes
                        for (size_t j = segStart; j < i; ++j) {
                            for (int tt = 0; tt < numTheta; ++tt) {
                                double r = points[linePoints[j]].x * table.cosTable[tt] +
                                           points[linePoints[j]].y * table.sinTable[tt];
                                int ri = static_cast<int>((r - rhoMin) / params.rhoResolution + 0.5);
                                if (ri >= 0 && ri < numRho) {
                                    acc(ri, tt) = std::max(0.0, acc(ri, tt) - 1.0);
                                }
                            }
                        }
                    }

                    segStart = i;
                }
            }
        }

        used[idx] = true;

        if (params.maxLines > 0 && static_cast<int>(segments.size()) >= params.maxLines) {
            break;
        }
    }

    // Sort by length
    std::sort(segments.begin(), segments.end(), [](const HoughLineSegment& a, const HoughLineSegment& b) {
        return a.Length() > b.Length();
    });

    return segments;
}

std::vector<HoughLineSegment> HoughLinesP(const QImage& edgeImage,
                                           const HoughLineProbParams& params) {
    if (edgeImage.Empty()) {
        return {};
    }

    std::vector<Point2d> points;
    int width = edgeImage.Width();
    int height = edgeImage.Height();

    for (int y = 0; y < height; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(edgeImage.RowPtr(y));
        for (int x = 0; x < width; ++x) {
            if (row[x] > 0) {
                points.emplace_back(static_cast<double>(x), static_cast<double>(y));
            }
        }
    }

    return HoughLinesP(points, width, height, params);
}

// =============================================================================
// Hough Circle Transform
// =============================================================================

std::vector<HoughCircle> HoughCircles(const std::vector<Point2d>& points,
                                       int minRadius, int maxRadius,
                                       double threshold,
                                       int maxCircles) {
    std::vector<HoughCircle> circles;

    if (points.empty() || minRadius >= maxRadius) {
        return circles;
    }

    // Determine accumulator size
    double minX = points[0].x, maxX = points[0].x;
    double minY = points[0].y, maxY = points[0].y;

    for (const auto& pt : points) {
        minX = std::min(minX, pt.x);
        maxX = std::max(maxX, pt.x);
        minY = std::min(minY, pt.y);
        maxY = std::max(maxY, pt.y);
    }

    int width = static_cast<int>(maxX - minX) + 1;
    int height = static_cast<int>(maxY - minY) + 1;

    // 3D accumulator: (x, y, r)
    // To save memory, process one radius at a time
    for (int r = minRadius; r <= maxRadius; ++r) {
        MatX acc = MatX::Zero(height, width);

        // Vote for each point
        for (const auto& pt : points) {
            // Draw circle of radius r around point
            int cx = static_cast<int>(pt.x - minX);
            int cy = static_cast<int>(pt.y - minY);

            // Use Bresenham's circle algorithm for efficiency
            int x = 0;
            int y = r;
            int d = 3 - 2 * r;

            while (x <= y) {
                // 8 symmetric points
                int pts[8][2] = {
                    {cx + x, cy + y}, {cx - x, cy + y},
                    {cx + x, cy - y}, {cx - x, cy - y},
                    {cx + y, cy + x}, {cx - y, cy + x},
                    {cx + y, cy - x}, {cx - y, cy - x}
                };

                for (auto& p : pts) {
                    if (p[0] >= 0 && p[0] < width && p[1] >= 0 && p[1] < height) {
                        acc(p[1], p[0]) += 1.0;
                    }
                }

                if (d < 0) {
                    d += 4 * x + 6;
                } else {
                    d += 4 * (x - y) + 10;
                    --y;
                }
                ++x;
            }
        }

        // Find peaks in accumulator
        double maxVal = 0.0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                maxVal = std::max(maxVal, acc(i, j));
            }
        }

        if (maxVal < EPSILON) continue;

        double thresholdVal = threshold * maxVal;

        for (int cy = 1; cy < height - 1; ++cy) {
            for (int cx = 1; cx < width - 1; ++cx) {
                double val = acc(cy, cx);
                if (val < thresholdVal) continue;

                // Check local maximum
                bool isMax = true;
                for (int dy = -1; dy <= 1 && isMax; ++dy) {
                    for (int dx = -1; dx <= 1 && isMax; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        if (acc(cy + dy, cx + dx) > val) {
                            isMax = false;
                        }
                    }
                }

                if (isMax) {
                    HoughCircle circle;
                    circle.center = Point2d(cx + minX, cy + minY);
                    circle.radius = r;
                    circle.score = val;
                    circles.push_back(circle);
                }
            }
        }
    }

    // Sort by score
    std::sort(circles.begin(), circles.end(), [](const HoughCircle& a, const HoughCircle& b) {
        return a.score > b.score;
    });

    // Non-maximum suppression
    circles = MergeHoughCircles(circles, 10.0, 5.0);

    if (maxCircles > 0 && static_cast<int>(circles.size()) > maxCircles) {
        circles.resize(maxCircles);
    }

    return circles;
}

std::vector<HoughCircle> HoughCirclesStandard(const QImage& edgeImage,
                                               int minRadius, int maxRadius,
                                               double threshold,
                                               int maxCircles) {
    if (edgeImage.Empty()) {
        return {};
    }

    std::vector<Point2d> points;
    int width = edgeImage.Width();
    int height = edgeImage.Height();

    for (int y = 0; y < height; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(edgeImage.RowPtr(y));
        for (int x = 0; x < width; ++x) {
            if (row[x] > 0) {
                points.emplace_back(static_cast<double>(x), static_cast<double>(y));
            }
        }
    }

    return HoughCircles(points, minRadius, maxRadius, threshold, maxCircles);
}

std::vector<HoughCircle> HoughCircles(const QImage& edgeImage,
                                       const QImage& gradientX,
                                       const QImage& gradientY,
                                       const HoughCircleParams& params) {
    std::vector<HoughCircle> circles;

    if (edgeImage.Empty()) {
        return circles;
    }

    int width = edgeImage.Width();
    int height = edgeImage.Height();

    int accWidth = static_cast<int>(width / params.dp);
    int accHeight = static_cast<int>(height / params.dp);

    int maxRadius = params.maxRadius;
    if (maxRadius <= 0) {
        maxRadius = static_cast<int>(std::sqrt(width * width + height * height) / 2);
    }

    // Center accumulator
    MatX centerAcc = MatX::Zero(accHeight, accWidth);

    // Vote for centers using gradient direction
    for (int y = 0; y < height; ++y) {
        const uint8_t* edgeRow = static_cast<const uint8_t*>(edgeImage.RowPtr(y));
        const float* gxRow = nullptr;
        const float* gyRow = nullptr;

        if (!gradientX.Empty() && !gradientY.Empty()) {
            gxRow = static_cast<const float*>(gradientX.RowPtr(y));
            gyRow = static_cast<const float*>(gradientY.RowPtr(y));
        }

        for (int x = 0; x < width; ++x) {
            if (edgeRow[x] == 0) continue;

            double gx = 0, gy = 0;
            if (gxRow && gyRow) {
                gx = gxRow[x];
                gy = gyRow[x];
            }

            double mag = std::sqrt(gx * gx + gy * gy);
            if (mag < EPSILON) continue;

            // Normalize gradient
            gx /= mag;
            gy /= mag;

            // Vote along gradient direction for different radii
            for (int r = params.minRadius; r <= maxRadius; ++r) {
                // Two possible center positions along gradient
                int cx1 = static_cast<int>((x + r * gx) / params.dp);
                int cy1 = static_cast<int>((y + r * gy) / params.dp);
                int cx2 = static_cast<int>((x - r * gx) / params.dp);
                int cy2 = static_cast<int>((y - r * gy) / params.dp);

                if (cx1 >= 0 && cx1 < accWidth && cy1 >= 0 && cy1 < accHeight) {
                    centerAcc(cy1, cx1) += 1.0;
                }
                if (cx2 >= 0 && cx2 < accWidth && cy2 >= 0 && cy2 < accHeight) {
                    centerAcc(cy2, cx2) += 1.0;
                }
            }
        }
    }

    // Find center candidates
    double maxVal = 0.0;
    for (int i = 0; i < accHeight; ++i) {
        for (int j = 0; j < accWidth; ++j) {
            maxVal = std::max(maxVal, centerAcc(i, j));
        }
    }

    if (maxVal < EPSILON) {
        return circles;
    }

    std::vector<Point2d> centers;
    double thresholdVal = params.param2;

    for (int cy = 1; cy < accHeight - 1; ++cy) {
        for (int cx = 1; cx < accWidth - 1; ++cx) {
            double val = centerAcc(cy, cx);
            if (val < thresholdVal) continue;

            // Local maximum check
            bool isMax = true;
            for (int dy = -1; dy <= 1 && isMax; ++dy) {
                for (int dx = -1; dx <= 1 && isMax; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (centerAcc(cy + dy, cx + dx) > val) {
                        isMax = false;
                    }
                }
            }

            if (isMax) {
                centers.emplace_back(cx * params.dp, cy * params.dp);
            }
        }
    }

    // For each center candidate, find best radius
    for (const auto& center : centers) {
        // Count edge points at each radius
        std::vector<int> radiusHist(maxRadius + 1, 0);

        for (int y = 0; y < height; ++y) {
            const uint8_t* edgeRow = static_cast<const uint8_t*>(edgeImage.RowPtr(y));
            for (int x = 0; x < width; ++x) {
                if (edgeRow[x] == 0) continue;

                double dx = x - center.x;
                double dy = y - center.y;
                int r = static_cast<int>(std::sqrt(dx * dx + dy * dy) + 0.5);

                if (r >= params.minRadius && r <= maxRadius) {
                    radiusHist[r]++;
                }
            }
        }

        // Find best radius
        int bestRadius = 0;
        int bestCount = 0;

        for (int r = params.minRadius; r <= maxRadius; ++r) {
            if (radiusHist[r] > bestCount) {
                bestCount = radiusHist[r];
                bestRadius = r;
            }
        }

        if (bestRadius > 0 && bestCount > params.param2) {
            HoughCircle circle;
            circle.center = center;
            circle.radius = bestRadius;
            circle.score = bestCount;
            circles.push_back(circle);
        }
    }

    // Non-maximum suppression
    circles = MergeHoughCircles(circles, params.minDist, 5.0);

    // Sort by score
    std::sort(circles.begin(), circles.end(), [](const HoughCircle& a, const HoughCircle& b) {
        return a.score > b.score;
    });

    if (params.maxCircles > 0 && static_cast<int>(circles.size()) > params.maxCircles) {
        circles.resize(params.maxCircles);
    }

    return circles;
}

std::vector<HoughCircle> HoughCircles(const QImage& image,
                                       const HoughCircleParams& params) {
    if (image.Empty()) {
        return {};
    }

    int width = image.Width();
    int height = image.Height();

    // Extract edge points using simple threshold on gradient magnitude
    // Compute gradients using central differences
    std::vector<Point2d> edgePoints;

    for (int y = 1; y < height - 1; ++y) {
        const uint8_t* rowPrev = static_cast<const uint8_t*>(image.RowPtr(y - 1));
        const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
        const uint8_t* rowNext = static_cast<const uint8_t*>(image.RowPtr(y + 1));

        for (int x = 1; x < width - 1; ++x) {
            // Central difference gradient
            double gx = (row[x + 1] - row[x - 1]) * 0.5;
            double gy = (rowNext[x] - rowPrev[x]) * 0.5;
            double mag = std::sqrt(gx * gx + gy * gy);

            if (mag > params.param1 / 2.0) {
                edgePoints.emplace_back(static_cast<double>(x), static_cast<double>(y));
            }
        }
    }

    // Use standard 3D Hough circle detection
    int maxRadius = params.maxRadius;
    if (maxRadius <= 0) {
        maxRadius = static_cast<int>(std::sqrt(width * width + height * height) / 2);
    }

    auto circles = HoughCircles(edgePoints, params.minRadius, maxRadius,
                                params.param2 / 100.0, params.maxCircles);

    // Apply min distance suppression
    circles = MergeHoughCircles(circles, params.minDist, 5.0);

    return circles;
}

// =============================================================================
// Refinement Functions
// =============================================================================

HoughLine RefineHoughLine(const HoughLine& line,
                          const std::vector<Point2d>& points,
                          double searchWidth) {
    // Collect points near the line
    std::vector<Point2d> nearPoints;
    double cosT = std::cos(line.theta);
    double sinT = std::sin(line.theta);

    for (const auto& pt : points) {
        double d = std::abs(pt.x * cosT + pt.y * sinT - line.rho);
        if (d <= searchWidth) {
            nearPoints.push_back(pt);
        }
    }

    if (nearPoints.size() < 2) {
        return line;
    }

    // Fit line to near points
    LineFitResult fit = FitLine(nearPoints, FitMethod::LeastSquares);

    if (!fit.success) {
        return line;
    }

    // Convert back to Hough form
    return CartesianToHoughLine(fit.line);
}

HoughLine RefineHoughLine(const HoughLine& line,
                          const QImage& edgeImage,
                          double searchWidth) {
    if (edgeImage.Empty()) {
        return line;
    }

    std::vector<Point2d> points;
    int width = edgeImage.Width();
    int height = edgeImage.Height();

    for (int y = 0; y < height; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(edgeImage.RowPtr(y));
        for (int x = 0; x < width; ++x) {
            if (row[x] > 0) {
                points.emplace_back(static_cast<double>(x), static_cast<double>(y));
            }
        }
    }

    return RefineHoughLine(line, points, searchWidth);
}

HoughCircle RefineHoughCircle(const HoughCircle& circle,
                               const std::vector<Point2d>& points,
                               double searchWidth) {
    // Collect points near the circle
    std::vector<Point2d> nearPoints;

    for (const auto& pt : points) {
        double dx = pt.x - circle.center.x;
        double dy = pt.y - circle.center.y;
        double d = std::abs(std::sqrt(dx * dx + dy * dy) - circle.radius);

        if (d <= searchWidth) {
            nearPoints.push_back(pt);
        }
    }

    if (nearPoints.size() < 3) {
        return circle;
    }

    // Fit circle to near points
    CircleFitResult fit = FitCircleAlgebraic(nearPoints);

    if (!fit.success) {
        return circle;
    }

    HoughCircle refined;
    refined.center = fit.circle.center;
    refined.radius = fit.circle.radius;
    refined.score = circle.score;

    return refined;
}

HoughCircle RefineHoughCircle(const HoughCircle& circle,
                               const QImage& edgeImage,
                               double searchWidth) {
    if (edgeImage.Empty()) {
        return circle;
    }

    std::vector<Point2d> points;
    int width = edgeImage.Width();
    int height = edgeImage.Height();

    for (int y = 0; y < height; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(edgeImage.RowPtr(y));
        for (int x = 0; x < width; ++x) {
            if (row[x] > 0) {
                points.emplace_back(static_cast<double>(x), static_cast<double>(y));
            }
        }
    }

    return RefineHoughCircle(circle, points, searchWidth);
}

// =============================================================================
// Utility Functions
// =============================================================================

Line2d HoughLineToCartesian(const HoughLine& line) {
    return line.ToLine2d();
}

HoughLine CartesianToHoughLine(const Line2d& line) {
    // Line: ax + by + c = 0
    // Normalize so that a² + b² = 1
    double norm = std::sqrt(line.a * line.a + line.b * line.b);

    if (norm < EPSILON) {
        return HoughLine(0, 0, 0);
    }

    double a = line.a / norm;
    double b = line.b / norm;
    double c = line.c / norm;

    // theta = atan2(b, a), rho = -c
    double theta = std::atan2(b, a);
    double rho = -c;

    // Ensure theta is in [0, PI)
    if (theta < 0) {
        theta += PI;
        rho = -rho;
    }

    return HoughLine(rho, theta);
}

Segment2d ClipHoughLineToImage(const HoughLine& line, int width, int height) {
    auto [p1, p2] = line.GetTwoPoints(std::max(width, height) * 2.0);

    // Clip to image bounds using Liang-Barsky algorithm
    double x1 = p1.x, y1 = p1.y;
    double x2 = p2.x, y2 = p2.y;
    double dx = x2 - x1;
    double dy = y2 - y1;

    double tMin = 0.0, tMax = 1.0;

    auto clip = [&](double p, double q) -> bool {
        if (std::abs(p) < EPSILON) {
            return q >= 0;
        }
        double t = q / p;
        if (p < 0) {
            if (t > tMax) return false;
            if (t > tMin) tMin = t;
        } else {
            if (t < tMin) return false;
            if (t < tMax) tMax = t;
        }
        return true;
    };

    if (!clip(-dx, x1 - 0)) return Segment2d();
    if (!clip(dx, width - x1)) return Segment2d();
    if (!clip(-dy, y1 - 0)) return Segment2d();
    if (!clip(dy, height - y1)) return Segment2d();

    Point2d clipped1(x1 + tMin * dx, y1 + tMin * dy);
    Point2d clipped2(x1 + tMax * dx, y1 + tMax * dy);

    return Segment2d(clipped1, clipped2);
}

std::vector<HoughLine> MergeHoughLines(const std::vector<HoughLine>& lines,
                                        double rhoThreshold,
                                        double thetaThreshold) {
    if (lines.empty()) {
        return lines;
    }

    std::vector<HoughLine> merged;
    std::vector<bool> used(lines.size(), false);

    for (size_t i = 0; i < lines.size(); ++i) {
        if (used[i]) continue;

        // Find all similar lines
        std::vector<size_t> group;
        group.push_back(i);

        for (size_t j = i + 1; j < lines.size(); ++j) {
            if (used[j]) continue;

            double dRho = std::abs(lines[i].rho - lines[j].rho);
            double dTheta = std::abs(lines[i].theta - lines[j].theta);

            // Handle theta wraparound at PI
            if (dTheta > PI / 2) {
                dTheta = PI - dTheta;
                dRho = std::abs(lines[i].rho + lines[j].rho);
            }

            if (dRho < rhoThreshold && dTheta < thetaThreshold) {
                group.push_back(j);
                used[j] = true;
            }
        }

        // Weighted average
        double totalWeight = 0;
        double avgRho = 0;
        double avgSinTheta = 0;
        double avgCosTheta = 0;
        double maxScore = 0;

        for (size_t idx : group) {
            double w = lines[idx].score;
            totalWeight += w;
            avgRho += w * lines[idx].rho;
            avgSinTheta += w * std::sin(lines[idx].theta);
            avgCosTheta += w * std::cos(lines[idx].theta);
            maxScore = std::max(maxScore, lines[idx].score);
        }

        if (totalWeight > 0) {
            avgRho /= totalWeight;
            double avgTheta = std::atan2(avgSinTheta, avgCosTheta);
            if (avgTheta < 0) avgTheta += PI;

            merged.emplace_back(avgRho, avgTheta, maxScore);
        }
    }

    return merged;
}

std::vector<HoughCircle> MergeHoughCircles(const std::vector<HoughCircle>& circles,
                                            double centerThreshold,
                                            double radiusThreshold) {
    if (circles.empty()) {
        return circles;
    }

    std::vector<HoughCircle> merged;
    std::vector<bool> used(circles.size(), false);

    for (size_t i = 0; i < circles.size(); ++i) {
        if (used[i]) continue;

        std::vector<size_t> group;
        group.push_back(i);

        for (size_t j = i + 1; j < circles.size(); ++j) {
            if (used[j]) continue;

            double dx = circles[i].center.x - circles[j].center.x;
            double dy = circles[i].center.y - circles[j].center.y;
            double centerDist = std::sqrt(dx * dx + dy * dy);
            double radiusDiff = std::abs(circles[i].radius - circles[j].radius);

            if (centerDist < centerThreshold && radiusDiff < radiusThreshold) {
                group.push_back(j);
                used[j] = true;
            }
        }

        // Weighted average
        double totalWeight = 0;
        double avgX = 0, avgY = 0, avgR = 0;
        double maxScore = 0;

        for (size_t idx : group) {
            double w = circles[idx].score;
            totalWeight += w;
            avgX += w * circles[idx].center.x;
            avgY += w * circles[idx].center.y;
            avgR += w * circles[idx].radius;
            maxScore = std::max(maxScore, circles[idx].score);
        }

        if (totalWeight > 0) {
            avgX /= totalWeight;
            avgY /= totalWeight;
            avgR /= totalWeight;

            merged.emplace_back(Point2d(avgX, avgY), avgR, maxScore);
        }
    }

    return merged;
}

double PointToHoughLineDistance(const Point2d& point, const HoughLine& line) {
    return point.x * std::cos(line.theta) + point.y * std::sin(line.theta) - line.rho;
}

bool AreHoughLinesParallel(const HoughLine& line1, const HoughLine& line2,
                           double angleTolerance) {
    double dTheta = std::abs(line1.theta - line2.theta);

    // Handle wraparound at PI
    if (dTheta > PI / 2) {
        dTheta = PI - dTheta;
    }

    return dTheta < angleTolerance;
}

bool AreHoughLinesPerpendicular(const HoughLine& line1, const HoughLine& line2,
                                double angleTolerance) {
    double dTheta = std::abs(line1.theta - line2.theta);

    // Handle wraparound
    if (dTheta > PI / 2) {
        dTheta = PI - dTheta;
    }

    return std::abs(dTheta - PI / 2) < angleTolerance;
}

bool HoughLinesIntersection(const HoughLine& line1, const HoughLine& line2,
                            Point2d& intersection) {
    // Convert to Cartesian form and solve
    double cos1 = std::cos(line1.theta), sin1 = std::sin(line1.theta);
    double cos2 = std::cos(line2.theta), sin2 = std::sin(line2.theta);

    double det = cos1 * sin2 - cos2 * sin1;

    if (std::abs(det) < EPSILON) {
        return false;  // Lines are parallel
    }

    intersection.x = (line1.rho * sin2 - line2.rho * sin1) / det;
    intersection.y = (line2.rho * cos1 - line1.rho * cos2) / det;

    return true;
}

} // namespace Qi::Vision::Internal
