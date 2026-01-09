#include <QiVision/Internal/RLEOps.h>
#include <QiVision/Internal/Histogram.h>
#include <QiVision/Internal/Threshold.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <unordered_set>
#include <unordered_map>

namespace Qi::Vision::Internal {

// =============================================================================
// RLE Utilities
// =============================================================================

void SortRuns(RunVector& runs) {
    std::sort(runs.begin(), runs.end(),
        [](const Run& a, const Run& b) {
            if (a.row != b.row) return a.row < b.row;
            return a.colBegin < b.colBegin;
        });
}

void MergeRuns(RunVector& runs) {
    if (runs.size() < 2) return;

    std::vector<Run> merged;
    merged.reserve(runs.size());
    merged.push_back(runs[0]);

    for (size_t i = 1; i < runs.size(); ++i) {
        Run& last = merged.back();
        const Run& curr = runs[i];

        if (curr.row == last.row && curr.colBegin <= last.colEnd) {
            // Merge overlapping/adjacent runs
            last.colEnd = std::max(last.colEnd, curr.colEnd);
        } else {
            merged.push_back(curr);
        }
    }

    runs = std::move(merged);
}

void NormalizeRuns(RunVector& runs) {
    if (runs.empty()) return;
    SortRuns(runs);
    MergeRuns(runs);
}

bool ValidateRuns(const RunVector& runs) {
    for (size_t i = 0; i < runs.size(); ++i) {
        // Check valid run
        if (runs[i].colEnd <= runs[i].colBegin) return false;

        // Check sorted order
        if (i > 0) {
            const Run& prev = runs[i - 1];
            const Run& curr = runs[i];
            if (curr.row < prev.row) return false;
            if (curr.row == prev.row && curr.colBegin < prev.colEnd) return false;
        }
    }
    return true;
}

RunVector TranslateRuns(const RunVector& runs, int32_t dx, int32_t dy) {
    RunVector result;
    result.reserve(runs.size());

    for (const auto& run : runs) {
        result.push_back(Run(run.row + dy, run.colBegin + dx, run.colEnd + dx));
    }

    return result;
}

RunVector ClipRuns(const RunVector& runs, const Rect2i& bounds) {
    RunVector result;
    result.reserve(runs.size());

    int32_t minRow = bounds.y;
    int32_t maxRow = bounds.y + bounds.height;
    int32_t minCol = bounds.x;
    int32_t maxCol = bounds.x + bounds.width;

    for (const auto& run : runs) {
        if (run.row < minRow || run.row >= maxRow) continue;
        if (run.colEnd <= minCol || run.colBegin >= maxCol) continue;

        int32_t newBegin = std::max(run.colBegin, minCol);
        int32_t newEnd = std::min(run.colEnd, maxCol);

        if (newEnd > newBegin) {
            result.push_back(Run(run.row, newBegin, newEnd));
        }
    }

    return result;
}

RunVector GetRunsForRow(const RunVector& runs, int32_t row) {
    RunVector result;

    // Binary search for first run on row
    auto it = std::lower_bound(runs.begin(), runs.end(), row,
        [](const Run& run, int32_t r) { return run.row < r; });

    while (it != runs.end() && it->row == row) {
        result.push_back(*it);
        ++it;
    }

    return result;
}

void GetRowRange(const RunVector& runs, int32_t& minRow, int32_t& maxRow) {
    if (runs.empty()) {
        minRow = maxRow = 0;
        return;
    }
    minRow = runs[0].row;
    maxRow = runs[0].row;
    for (const auto& run : runs) {
        if (run.row < minRow) minRow = run.row;
        if (run.row > maxRow) maxRow = run.row;
    }
}

// =============================================================================
// Analysis Operations
// =============================================================================

int64_t ComputeArea(const RunVector& runs) {
    int64_t area = 0;
    for (const auto& run : runs) {
        area += run.Length();
    }
    return area;
}

Rect2i ComputeBoundingBox(const RunVector& runs) {
    if (runs.empty()) {
        return Rect2i();
    }

    int32_t minRow = runs.front().row;
    int32_t maxRow = runs.back().row;
    int32_t minCol = std::numeric_limits<int32_t>::max();
    int32_t maxCol = std::numeric_limits<int32_t>::min();

    for (const auto& run : runs) {
        minCol = std::min(minCol, run.colBegin);
        maxCol = std::max(maxCol, run.colEnd);
    }

    return Rect2i(minCol, minRow, maxCol - minCol, maxRow - minRow + 1);
}

Point2d ComputeCentroid(const RunVector& runs) {
    if (runs.empty()) {
        return Point2d();
    }

    double sumX = 0.0;
    double sumY = 0.0;
    int64_t count = 0;

    for (const auto& run : runs) {
        int32_t len = run.Length();
        // Sum of x coordinates in run: sum from colBegin to colEnd-1
        // = (colBegin + colEnd - 1) * len / 2
        sumX += (run.colBegin + run.colEnd - 1) * 0.5 * len;
        sumY += run.row * len;
        count += len;
    }

    if (count == 0) return Point2d();

    return Point2d(sumX / count, sumY / count);
}

double ComputePerimeter(const QRegion& region, Connectivity connectivity) {
    QRegion boundary = ExtractBoundary(region, connectivity);
    return static_cast<double>(boundary.Area());
}

double ComputeCircularity(const QRegion& region) {
    if (region.Empty()) return 0.0;

    double area = static_cast<double>(region.Area());
    double perimeter = ComputePerimeter(region, Connectivity::Eight);

    if (perimeter < 1e-10) return 0.0;

    // 4 * pi * area / perimeter^2
    return (4.0 * 3.14159265358979323846 * area) / (perimeter * perimeter);
}

double ComputeCompactness(const QRegion& region) {
    if (region.Empty()) return 0.0;

    double area = static_cast<double>(region.Area());
    double perimeter = ComputePerimeter(region, Connectivity::Eight);

    if (area < 1e-10) return 0.0;

    return (perimeter * perimeter) / area;
}

double ComputeRectangularity(const QRegion& region) {
    if (region.Empty()) return 0.0;

    Rect2i bbox = region.BoundingBox();
    int64_t bboxArea = static_cast<int64_t>(bbox.width) * bbox.height;

    if (bboxArea == 0) return 0.0;

    return static_cast<double>(region.Area()) / bboxArea;
}

// =============================================================================
// Set Operations on Run Vectors
// =============================================================================

RunVector UnionRuns(const RunVector& runs1, const RunVector& runs2) {
    RunVector result;
    result.reserve(runs1.size() + runs2.size());

    result.insert(result.end(), runs1.begin(), runs1.end());
    result.insert(result.end(), runs2.begin(), runs2.end());

    NormalizeRuns(result);
    return result;
}

RunVector IntersectRuns(const RunVector& runs1, const RunVector& runs2) {
    RunVector result;

    if (runs1.empty() || runs2.empty()) return result;

    // Process row by row
    size_t i1 = 0, i2 = 0;

    while (i1 < runs1.size() && i2 < runs2.size()) {
        const Run& r1 = runs1[i1];
        const Run& r2 = runs2[i2];

        if (r1.row < r2.row) {
            ++i1;
        } else if (r1.row > r2.row) {
            ++i2;
        } else {
            // Same row - check for intersection
            int32_t overlapBegin = std::max(r1.colBegin, r2.colBegin);
            int32_t overlapEnd = std::min(r1.colEnd, r2.colEnd);

            if (overlapEnd > overlapBegin) {
                result.push_back(Run(r1.row, overlapBegin, overlapEnd));
            }

            // Advance the run that ends first
            if (r1.colEnd <= r2.colEnd) {
                ++i1;
            } else {
                ++i2;
            }
        }
    }

    return result;
}

RunVector DifferenceRuns(const RunVector& runs1, const RunVector& runs2) {
    if (runs1.empty()) return RunVector();
    if (runs2.empty()) return runs1;

    RunVector result;
    result.reserve(runs1.size());

    size_t i2 = 0;

    for (const auto& r1 : runs1) {
        // Advance i2 to first run on same or later row
        while (i2 < runs2.size() && runs2[i2].row < r1.row) {
            ++i2;
        }

        // Collect all runs from runs2 on this row
        std::vector<Run> subtractRuns;
        for (size_t j = i2; j < runs2.size() && runs2[j].row == r1.row; ++j) {
            subtractRuns.push_back(runs2[j]);
        }

        if (subtractRuns.empty()) {
            result.push_back(r1);
            continue;
        }

        // Subtract runs from r1
        int32_t currentBegin = r1.colBegin;

        for (const auto& sub : subtractRuns) {
            if (sub.colEnd <= currentBegin) continue;
            if (sub.colBegin >= r1.colEnd) break;

            if (sub.colBegin > currentBegin) {
                result.push_back(Run(r1.row, currentBegin, sub.colBegin));
            }
            currentBegin = std::max(currentBegin, sub.colEnd);
        }

        if (currentBegin < r1.colEnd) {
            result.push_back(Run(r1.row, currentBegin, r1.colEnd));
        }
    }

    return result;
}

RunVector ComplementRuns(const RunVector& runs, const Rect2i& bounds) {
    RunVector fullRegion;
    for (int32_t row = bounds.y; row < bounds.y + bounds.height; ++row) {
        fullRegion.push_back(Run(row, bounds.x, bounds.x + bounds.width));
    }
    return DifferenceRuns(fullRegion, runs);
}

RunVector SymmetricDifferenceRuns(const RunVector& runs1, const RunVector& runs2) {
    // XOR = (A - B) union (B - A)
    RunVector diff1 = DifferenceRuns(runs1, runs2);
    RunVector diff2 = DifferenceRuns(runs2, runs1);
    return UnionRuns(diff1, diff2);
}

// =============================================================================
// Image to Region Conversion
// =============================================================================

QRegion ThresholdToRegion(const QImage& image,
                          double minVal,
                          double maxVal,
                          ThresholdMode mode) {
    if (image.Empty() || image.Channels() != 1) {
        return QRegion();
    }

    std::vector<Run> runs;
    int32_t width = image.Width();
    int32_t height = image.Height();

    uint8_t minV = static_cast<uint8_t>(std::max(0.0, std::min(255.0, minVal)));
    uint8_t maxV = static_cast<uint8_t>(std::max(0.0, std::min(255.0, maxVal)));

    for (int32_t y = 0; y < height; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
        int32_t runStart = -1;

        for (int32_t x = 0; x < width; ++x) {
            uint8_t val = row[x];
            bool inRegion = false;

            switch (mode) {
                case ThresholdMode::Binary:
                    inRegion = (val >= minV);
                    break;
                case ThresholdMode::BinaryInv:
                    inRegion = (val < minV);
                    break;
                case ThresholdMode::Range:
                    inRegion = (val >= minV && val <= maxV);
                    break;
                case ThresholdMode::RangeInv:
                    inRegion = (val < minV || val > maxV);
                    break;
            }

            if (inRegion) {
                if (runStart < 0) runStart = x;
            } else {
                if (runStart >= 0) {
                    runs.push_back(Run(y, runStart, x));
                    runStart = -1;
                }
            }
        }

        // Handle run extending to end of row
        if (runStart >= 0) {
            runs.push_back(Run(y, runStart, width));
        }
    }

    return QRegion(runs);
}

QRegion DynamicThreshold(const QImage& image, int maskSize, double offset) {
    if (image.Empty() || image.Channels() != 1) {
        return QRegion();
    }

    // Use box filter for local mean
    int32_t width = image.Width();
    int32_t height = image.Height();
    int32_t halfSize = maskSize / 2;

    std::vector<Run> runs;

    // Compute integral image for fast local mean
    std::vector<int64_t> integral((width + 1) * (height + 1), 0);

    for (int32_t y = 0; y < height; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
        int64_t rowSum = 0;
        for (int32_t x = 0; x < width; ++x) {
            rowSum += row[x];
            integral[(y + 1) * (width + 1) + (x + 1)] =
                rowSum + integral[y * (width + 1) + (x + 1)];
        }
    }

    // Apply dynamic threshold
    for (int32_t y = 0; y < height; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
        int32_t runStart = -1;

        for (int32_t x = 0; x < width; ++x) {
            // Compute local mean using integral image
            int32_t x1 = std::max(0, x - halfSize);
            int32_t y1 = std::max(0, y - halfSize);
            int32_t x2 = std::min(width - 1, x + halfSize);
            int32_t y2 = std::min(height - 1, y + halfSize);

            int64_t sum = integral[(y2 + 1) * (width + 1) + (x2 + 1)]
                        - integral[y1 * (width + 1) + (x2 + 1)]
                        - integral[(y2 + 1) * (width + 1) + x1]
                        + integral[y1 * (width + 1) + x1];

            int32_t count = (x2 - x1 + 1) * (y2 - y1 + 1);
            double localMean = static_cast<double>(sum) / count;

            bool inRegion = (row[x] > localMean + offset);

            if (inRegion) {
                if (runStart < 0) runStart = x;
            } else {
                if (runStart >= 0) {
                    runs.push_back(Run(y, runStart, x));
                    runStart = -1;
                }
            }
        }

        if (runStart >= 0) {
            runs.push_back(Run(y, runStart, width));
        }
    }

    return QRegion(runs);
}

QRegion AutoThreshold(const QImage& image, const std::string& method) {
    if (image.Empty() || image.Channels() != 1) {
        return QRegion();
    }

    double threshold = 128.0;

    if (method == "otsu") {
        threshold = ComputeOtsuThreshold(image);
    } else if (method == "mean") {
        // Compute mean
        int64_t sum = 0;
        int64_t count = 0;
        for (int32_t y = 0; y < image.Height(); ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < image.Width(); ++x) {
                sum += row[x];
                ++count;
            }
        }
        threshold = static_cast<double>(sum) / count;
    } else if (method == "median") {
        std::vector<int> histogram(256, 0);
        int64_t count = 0;
        for (int32_t y = 0; y < image.Height(); ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < image.Width(); ++x) {
                histogram[row[x]]++;
                ++count;
            }
        }
        int64_t halfCount = count / 2;
        int64_t cumSum = 0;
        for (int i = 0; i < 256; ++i) {
            cumSum += histogram[i];
            if (cumSum >= halfCount) {
                threshold = i;
                break;
            }
        }
    }

    return ThresholdToRegion(image, threshold, 255.0, ThresholdMode::Binary);
}

QRegion NonZeroToRegion(const QImage& image) {
    return ThresholdToRegion(image, 1.0, 255.0, ThresholdMode::Range);
}

// =============================================================================
// Region to Image Conversion
// =============================================================================

void PaintRegion(const QRegion& region, QImage& image, double value) {
    if (region.Empty() || image.Empty()) return;

    uint8_t val = static_cast<uint8_t>(std::max(0.0, std::min(255.0, value)));

    for (const auto& run : region.Runs()) {
        if (run.row < 0 || run.row >= image.Height()) continue;

        uint8_t* row = static_cast<uint8_t*>(image.RowPtr(run.row));
        int32_t colBegin = std::max<int32_t>(0, run.colBegin);
        int32_t colEnd = std::min<int32_t>(image.Width(), run.colEnd);

        for (int32_t x = colBegin; x < colEnd; ++x) {
            row[x] = val;
        }
    }
}

QImage RegionToMask(const QRegion& region, int32_t width, int32_t height) {
    Rect2i bbox = region.BoundingBox();

    if (width <= 0) width = bbox.x + bbox.width;
    if (height <= 0) height = bbox.y + bbox.height;

    if (width <= 0 || height <= 0) {
        return QImage();
    }

    QImage mask(width, height, PixelType::UInt8, ChannelType::Gray);

    // Clear to zero
    for (int32_t y = 0; y < height; ++y) {
        uint8_t* row = static_cast<uint8_t*>(mask.RowPtr(y));
        std::memset(row, 0, width);
    }

    PaintRegion(region, mask, 255.0);

    return mask;
}

QImage RegionsToLabels(const std::vector<QRegion>& regions,
                       int32_t width, int32_t height) {
    if (width <= 0 || height <= 0) {
        return QImage();
    }

    // Use 16-bit for labels to support more regions
    QImage labels(width, height, PixelType::UInt16, ChannelType::Gray);

    // Clear to zero
    for (int32_t y = 0; y < height; ++y) {
        uint16_t* row = static_cast<uint16_t*>(labels.RowPtr(y));
        std::memset(row, 0, width * sizeof(uint16_t));
    }

    // Paint each region with its label (1-based)
    for (size_t i = 0; i < regions.size(); ++i) {
        uint16_t label = static_cast<uint16_t>(i + 1);

        for (const auto& run : regions[i].Runs()) {
            if (run.row < 0 || run.row >= height) continue;

            uint16_t* row = static_cast<uint16_t*>(labels.RowPtr(run.row));
            int32_t colBegin = std::max<int32_t>(0, run.colBegin);
            int32_t colEnd = std::min<int32_t>(width, run.colEnd);

            for (int32_t x = colBegin; x < colEnd; ++x) {
                row[x] = label;
            }
        }
    }

    return labels;
}

// =============================================================================
// Boundary Operations
// =============================================================================

QRegion ExtractBoundary(const QRegion& region, Connectivity connectivity) {
    if (region.Empty()) return QRegion();

    const auto& runs = region.Runs();

    // Build a set of all pixels in region for fast lookup
    std::unordered_set<int64_t> pixelSet;
    for (const auto& run : runs) {
        for (int32_t col = run.colBegin; col < run.colEnd; ++col) {
            int64_t key = (static_cast<int64_t>(run.row) << 32) | static_cast<uint32_t>(col);
            pixelSet.insert(key);
        }
    }

    auto isInRegion = [&pixelSet](int32_t row, int32_t col) -> bool {
        int64_t key = (static_cast<int64_t>(row) << 32) | static_cast<uint32_t>(col);
        return pixelSet.count(key) > 0;
    };

    // Extract boundary pixels
    std::vector<Run> boundaryRuns;
    for (const auto& run : runs) {
        for (int32_t col = run.colBegin; col < run.colEnd; ++col) {
            bool isBoundary = false;

            if (connectivity == Connectivity::Four) {
                // 4-connected: check 4 neighbors
                isBoundary = !isInRegion(run.row - 1, col) ||
                             !isInRegion(run.row + 1, col) ||
                             !isInRegion(run.row, col - 1) ||
                             !isInRegion(run.row, col + 1);
            } else {
                // 8-connected: check 8 neighbors
                isBoundary = !isInRegion(run.row - 1, col - 1) ||
                             !isInRegion(run.row - 1, col) ||
                             !isInRegion(run.row - 1, col + 1) ||
                             !isInRegion(run.row, col - 1) ||
                             !isInRegion(run.row, col + 1) ||
                             !isInRegion(run.row + 1, col - 1) ||
                             !isInRegion(run.row + 1, col) ||
                             !isInRegion(run.row + 1, col + 1);
            }

            if (isBoundary) {
                boundaryRuns.emplace_back(run.row, col, col + 1);
            }
        }
    }

    NormalizeRuns(boundaryRuns);
    return QRegion(boundaryRuns);
}

QRegion InnerBoundary(const QRegion& region) {
    return ExtractBoundary(region, Connectivity::Eight);
}

QRegion OuterBoundary(const QRegion& region) {
    if (region.Empty()) return QRegion();

    const auto& runs = region.Runs();

    // Build a set of all pixels in region for fast lookup
    std::unordered_set<int64_t> pixelSet;
    for (const auto& run : runs) {
        for (int32_t col = run.colBegin; col < run.colEnd; ++col) {
            int64_t key = (static_cast<int64_t>(run.row) << 32) | static_cast<uint32_t>(col);
            pixelSet.insert(key);
        }
    }

    auto isInRegion = [&pixelSet](int32_t row, int32_t col) -> bool {
        int64_t key = (static_cast<int64_t>(row) << 32) | static_cast<uint32_t>(col);
        return pixelSet.count(key) > 0;
    };

    // Find all outer boundary pixels (outside region but adjacent to region)
    std::unordered_set<int64_t> outerPixels;
    for (const auto& run : runs) {
        for (int32_t col = run.colBegin; col < run.colEnd; ++col) {
            // Check 8 neighbors
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int32_t nr = run.row + dy;
                    int32_t nc = col + dx;
                    if (!isInRegion(nr, nc)) {
                        int64_t key = (static_cast<int64_t>(nr) << 32) | static_cast<uint32_t>(nc);
                        outerPixels.insert(key);
                    }
                }
            }
        }
    }

    // Convert to runs
    std::vector<Run> outerRuns;
    for (int64_t key : outerPixels) {
        int32_t row = static_cast<int32_t>(key >> 32);
        int32_t col = static_cast<int32_t>(key & 0xFFFFFFFF);
        outerRuns.emplace_back(row, col, col + 1);
    }

    NormalizeRuns(outerRuns);
    return QRegion(outerRuns);
}

// =============================================================================
// Fill Operations
// =============================================================================

QRegion FillHorizontalGaps(const QRegion& region, int32_t maxGap) {
    if (region.Empty() || maxGap <= 0) return region;

    const auto& runs = region.Runs();
    std::vector<Run> result;
    result.reserve(runs.size());

    size_t i = 0;
    while (i < runs.size()) {
        Run current = runs[i];
        ++i;

        // Try to merge with following runs on same row
        while (i < runs.size() && runs[i].row == current.row) {
            int32_t gap = runs[i].colBegin - current.colEnd;
            if (gap <= maxGap) {
                // Merge
                current.colEnd = runs[i].colEnd;
                ++i;
            } else {
                break;
            }
        }

        result.push_back(current);
    }

    return QRegion(result);
}

QRegion FillVerticalGaps(const QRegion& region, int32_t maxGap) {
    if (region.Empty() || maxGap <= 0) return region;

    // This is more complex - need to track columns across rows
    Rect2i bbox = region.BoundingBox();

    // For each column, find vertical gaps and fill them
    std::vector<Run> result = region.Runs();

    // Group runs by row for efficient lookup
    std::unordered_map<int32_t, std::vector<Run>> rowRuns;
    for (const auto& run : result) {
        rowRuns[run.row].push_back(run);
    }

    // Check each pair of adjacent rows
    for (int32_t row = bbox.y; row < bbox.y + bbox.height - 1; ++row) {
        int32_t gap = 1;
        while (gap <= maxGap && row + gap < bbox.y + bbox.height) {
            auto it1 = rowRuns.find(row);
            auto it2 = rowRuns.find(row + gap + 1);

            if (it1 != rowRuns.end() && it2 != rowRuns.end()) {
                // Find overlapping column ranges
                for (const auto& r1 : it1->second) {
                    for (const auto& r2 : it2->second) {
                        int32_t overlapBegin = std::max(r1.colBegin, r2.colBegin);
                        int32_t overlapEnd = std::min(r1.colEnd, r2.colEnd);

                        if (overlapEnd > overlapBegin) {
                            // Fill gap rows
                            for (int32_t fillRow = row + 1; fillRow <= row + gap; ++fillRow) {
                                result.push_back(Run(fillRow, overlapBegin, overlapEnd));
                            }
                        }
                    }
                }
            }
            ++gap;
        }
    }

    NormalizeRuns(result);
    return QRegion(result);
}

QRegion FillHoles(const QRegion& region) {
    if (region.Empty()) return region;

    Rect2i bbox = region.BoundingBox();

    // Create mask and flood fill from border
    int32_t width = bbox.width + 2;
    int32_t height = bbox.height + 2;

    // Translate region to (1,1)
    auto runs = TranslateRuns(region.Runs(), 1 - bbox.x, 1 - bbox.y);

    // Create full rectangle and subtract region
    std::vector<Run> fullRuns;
    for (int32_t y = 0; y < height; ++y) {
        fullRuns.push_back(Run(y, 0, width));
    }

    auto background = DifferenceRuns(fullRuns, runs);

    // Flood fill background from (0,0) to find exterior
    std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));
    std::queue<std::pair<int32_t, int32_t>> queue;

    // Mark background pixels
    std::vector<std::vector<bool>> isBg(height, std::vector<bool>(width, false));
    for (const auto& run : background) {
        for (int32_t x = run.colBegin; x < run.colEnd; ++x) {
            isBg[run.row][x] = true;
        }
    }

    // BFS from corners
    auto tryAdd = [&](int32_t x, int32_t y) {
        if (x >= 0 && x < width && y >= 0 && y < height &&
            !visited[y][x] && isBg[y][x]) {
            visited[y][x] = true;
            queue.push({x, y});
        }
    };

    tryAdd(0, 0);

    while (!queue.empty()) {
        auto [x, y] = queue.front();
        queue.pop();

        tryAdd(x - 1, y);
        tryAdd(x + 1, y);
        tryAdd(x, y - 1);
        tryAdd(x, y + 1);
    }

    // Everything not visited and not in region is a hole
    std::vector<Run> holes;
    for (int32_t y = 1; y < height - 1; ++y) {
        int32_t runStart = -1;
        for (int32_t x = 1; x < width - 1; ++x) {
            bool isHole = isBg[y][x] && !visited[y][x];
            if (isHole) {
                if (runStart < 0) runStart = x;
            } else {
                if (runStart >= 0) {
                    holes.push_back(Run(y, runStart, x));
                    runStart = -1;
                }
            }
        }
        if (runStart >= 0) {
            holes.push_back(Run(y, runStart, width - 1));
        }
    }

    // Translate holes back
    holes = TranslateRuns(holes, bbox.x - 1, bbox.y - 1);

    // Union with original region
    return QRegion(UnionRuns(region.Runs(), holes));
}

QRegion FillConvex(const QRegion& region) {
    if (region.Empty()) return region;

    // Get all boundary points
    QRegion boundary = ExtractBoundary(region, Connectivity::Eight);

    // Compute convex hull of boundary points
    std::vector<Point2i> points;
    for (const auto& run : boundary.Runs()) {
        for (int32_t x = run.colBegin; x < run.colEnd; ++x) {
            points.push_back(Point2i(x, run.row));
        }
    }

    if (points.size() < 3) return region;

    // Simple convex hull using Graham scan
    // Sort by y, then x
    std::sort(points.begin(), points.end(), [](const Point2i& a, const Point2i& b) {
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    // Remove duplicates
    points.erase(std::unique(points.begin(), points.end(),
        [](const Point2i& a, const Point2i& b) {
            return a.x == b.x && a.y == b.y;
        }), points.end());

    if (points.size() < 3) return region;

    // Build convex hull
    auto cross = [](const Point2i& O, const Point2i& A, const Point2i& B) -> int64_t {
        return static_cast<int64_t>(A.x - O.x) * (B.y - O.y) -
               static_cast<int64_t>(A.y - O.y) * (B.x - O.x);
    };

    std::vector<Point2i> hull;

    // Build lower hull
    for (const auto& p : points) {
        while (hull.size() >= 2 && cross(hull[hull.size()-2], hull[hull.size()-1], p) <= 0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    // Build upper hull
    size_t lower = hull.size();
    for (int i = static_cast<int>(points.size()) - 2; i >= 0; --i) {
        while (hull.size() > lower && cross(hull[hull.size()-2], hull[hull.size()-1], points[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }

    hull.pop_back(); // Remove last point (duplicate of first)

    if (hull.size() < 3) return region;

    // Fill convex hull using scanline
    Rect2i bbox = region.BoundingBox();
    std::vector<Run> runs;

    for (int32_t y = bbox.y; y < bbox.y + bbox.height; ++y) {
        // Find intersections with hull edges
        std::vector<int32_t> intersections;

        for (size_t i = 0; i < hull.size(); ++i) {
            const Point2i& p1 = hull[i];
            const Point2i& p2 = hull[(i + 1) % hull.size()];

            if ((p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y)) {
                double t = static_cast<double>(y - p1.y) / (p2.y - p1.y);
                int32_t x = p1.x + static_cast<int32_t>(t * (p2.x - p1.x));
                intersections.push_back(x);
            }
        }

        if (intersections.size() >= 2) {
            std::sort(intersections.begin(), intersections.end());
            for (size_t i = 0; i + 1 < intersections.size(); i += 2) {
                runs.push_back(Run(y, intersections[i], intersections[i + 1] + 1));
            }
        }
    }

    NormalizeRuns(runs);
    return QRegion(runs);
}

// =============================================================================
// Connection Operations
// =============================================================================

std::vector<QRegion> SplitConnectedComponents(const QRegion& region,
                                               Connectivity connectivity) {
    std::vector<QRegion> components;

    if (region.Empty()) return components;

    Rect2i bbox = region.BoundingBox();

    // Create label image
    int32_t width = bbox.width;
    int32_t height = bbox.height;
    int32_t offsetX = bbox.x;
    int32_t offsetY = bbox.y;

    std::vector<std::vector<int32_t>> labels(height, std::vector<int32_t>(width, 0));

    // Mark region pixels as -1 (unlabeled foreground)
    for (const auto& run : region.Runs()) {
        int32_t y = run.row - offsetY;
        for (int32_t x = run.colBegin - offsetX; x < run.colEnd - offsetX; ++x) {
            if (y >= 0 && y < height && x >= 0 && x < width) {
                labels[y][x] = -1;
            }
        }
    }

    int32_t currentLabel = 0;
    std::vector<std::vector<Run>> componentRuns;

    // BFS to label connected components
    std::vector<std::pair<int, int>> neighbors;
    if (connectivity == Connectivity::Four) {
        neighbors = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    } else {
        neighbors = {{-1, 0}, {1, 0}, {0, -1}, {0, 1},
                     {-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    }

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            if (labels[y][x] == -1) {
                // Start new component
                ++currentLabel;
                componentRuns.push_back({});

                std::queue<std::pair<int32_t, int32_t>> queue;
                queue.push({x, y});
                labels[y][x] = currentLabel;

                while (!queue.empty()) {
                    auto [cx, cy] = queue.front();
                    queue.pop();

                    for (const auto& [dx, dy] : neighbors) {
                        int32_t nx = cx + dx;
                        int32_t ny = cy + dy;

                        if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                            labels[ny][nx] == -1) {
                            labels[ny][nx] = currentLabel;
                            queue.push({nx, ny});
                        }
                    }
                }
            }
        }
    }

    // Extract runs for each component
    componentRuns.resize(currentLabel);

    for (int32_t y = 0; y < height; ++y) {
        int32_t runStart = -1;
        int32_t runLabel = 0;

        for (int32_t x = 0; x <= width; ++x) {
            int32_t label = (x < width) ? labels[y][x] : 0;

            if (label > 0 && label == runLabel) {
                // Continue run
            } else {
                if (runStart >= 0 && runLabel > 0) {
                    // End run
                    componentRuns[runLabel - 1].push_back(
                        Run(y + offsetY, runStart + offsetX, x + offsetX));
                }
                runStart = (label > 0) ? x : -1;
                runLabel = label;
            }
        }
    }

    // Create regions
    for (auto& runs : componentRuns) {
        if (!runs.empty()) {
            components.push_back(QRegion(runs));
        }
    }

    return components;
}

bool IsConnected(const QRegion& region, Connectivity connectivity) {
    if (region.Empty()) return true;
    if (region.Area() <= 1) return true;

    auto components = SplitConnectedComponents(region, connectivity);
    return components.size() <= 1;
}

size_t CountConnectedComponents(const QRegion& region, Connectivity connectivity) {
    return SplitConnectedComponents(region, connectivity).size();
}

} // namespace Qi::Vision::Internal
