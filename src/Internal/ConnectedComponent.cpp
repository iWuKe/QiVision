#include <QiVision/Internal/ConnectedComponent.h>
#include <QiVision/Internal/RLEOps.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cstring>

namespace Qi::Vision::Internal {

// =============================================================================
// Union-Find Data Structure
// =============================================================================

namespace {

class UnionFind {
public:
    explicit UnionFind(int32_t size) : parent_(size), rank_(size, 0) {
        for (int32_t i = 0; i < size; ++i) {
            parent_[i] = i;
        }
    }

    int32_t Find(int32_t x) {
        if (parent_[x] != x) {
            parent_[x] = Find(parent_[x]);  // Path compression
        }
        return parent_[x];
    }

    void Union(int32_t x, int32_t y) {
        int32_t px = Find(x);
        int32_t py = Find(y);
        if (px == py) return;

        // Union by rank
        if (rank_[px] < rank_[py]) {
            parent_[px] = py;
        } else if (rank_[px] > rank_[py]) {
            parent_[py] = px;
        } else {
            parent_[py] = px;
            rank_[px]++;
        }
    }

private:
    std::vector<int32_t> parent_;
    std::vector<int32_t> rank_;
};

} // anonymous namespace

// =============================================================================
// Image-Based Connected Component Labeling
// =============================================================================

QImage LabelConnectedComponents(const QImage& binary,
                                 Connectivity connectivity,
                                 int32_t& numLabels) {
    numLabels = 0;
    if (binary.Empty()) return QImage();

    int32_t width = binary.Width();
    int32_t height = binary.Height();
    int32_t srcStride = static_cast<int32_t>(binary.Stride());

    const uint8_t* srcData = static_cast<const uint8_t*>(binary.Data());

    // Create label image (32-bit for large images)
    QImage labels(width, height, PixelType::UInt8, ChannelType::Gray);
    std::vector<int32_t> labelMap(width * height, 0);

    // First pass: assign provisional labels
    int32_t nextLabel = 1;
    UnionFind uf(width * height / 4 + 1);  // Estimate max labels

    for (int32_t r = 0; r < height; ++r) {
        const uint8_t* srcRow = srcData + r * srcStride;

        for (int32_t c = 0; c < width; ++c) {
            if (srcRow[c] == 0) continue;  // Background

            int32_t idx = r * width + c;
            std::vector<int32_t> neighbors;

            // Check neighbors (depending on connectivity)
            // North
            if (r > 0 && srcData[(r - 1) * srcStride + c] != 0) {
                neighbors.push_back(labelMap[(r - 1) * width + c]);
            }
            // West
            if (c > 0 && srcRow[c - 1] != 0) {
                neighbors.push_back(labelMap[r * width + c - 1]);
            }

            if (connectivity == Connectivity::Eight) {
                // Northwest
                if (r > 0 && c > 0 && srcData[(r - 1) * srcStride + c - 1] != 0) {
                    neighbors.push_back(labelMap[(r - 1) * width + c - 1]);
                }
                // Northeast
                if (r > 0 && c < width - 1 && srcData[(r - 1) * srcStride + c + 1] != 0) {
                    neighbors.push_back(labelMap[(r - 1) * width + c + 1]);
                }
            }

            if (neighbors.empty()) {
                // New label
                labelMap[idx] = nextLabel++;
            } else {
                // Find minimum label
                int32_t minLabel = *std::min_element(neighbors.begin(), neighbors.end());
                labelMap[idx] = minLabel;

                // Union with all neighbor labels
                for (int32_t lbl : neighbors) {
                    if (lbl != minLabel) {
                        uf.Union(minLabel, lbl);
                    }
                }
            }
        }
    }

    // Second pass: flatten labels
    std::unordered_map<int32_t, int32_t> labelRemap;
    int32_t finalLabel = 0;

    uint8_t* dstData = static_cast<uint8_t*>(labels.Data());
    int32_t dstStride = static_cast<int32_t>(labels.Stride());

    for (int32_t r = 0; r < height; ++r) {
        for (int32_t c = 0; c < width; ++c) {
            int32_t idx = r * width + c;
            if (labelMap[idx] == 0) {
                dstData[r * dstStride + c] = 0;
            } else {
                int32_t root = uf.Find(labelMap[idx]);
                if (labelRemap.find(root) == labelRemap.end()) {
                    labelRemap[root] = ++finalLabel;
                }
                // Clamp to 255 for 8-bit output
                dstData[r * dstStride + c] = static_cast<uint8_t>(
                    std::min(labelRemap[root], 255));
            }
        }
    }

    numLabels = finalLabel;
    return labels;
}

QImage LabelConnectedComponents(const QImage& binary, Connectivity connectivity) {
    int32_t numLabels;
    return LabelConnectedComponents(binary, connectivity, numLabels);
}

std::vector<ComponentStats> GetComponentStats(const QImage& labels, int32_t numLabels) {
    if (labels.Empty() || numLabels <= 0) return {};

    std::vector<ComponentStats> stats(numLabels);
    for (int32_t i = 0; i < numLabels; ++i) {
        stats[i].label = i + 1;
        stats[i].minRow = labels.Height();
        stats[i].maxRow = 0;
        stats[i].minCol = labels.Width();
        stats[i].maxCol = 0;
    }

    int32_t width = labels.Width();
    int32_t height = labels.Height();
    int32_t stride = static_cast<int32_t>(labels.Stride());
    const uint8_t* data = static_cast<const uint8_t*>(labels.Data());

    // Accumulate statistics
    for (int32_t r = 0; r < height; ++r) {
        const uint8_t* row = data + r * stride;
        for (int32_t c = 0; c < width; ++c) {
            int32_t lbl = row[c];
            if (lbl > 0 && lbl <= numLabels) {
                auto& s = stats[lbl - 1];
                s.area++;
                s.centroidX += c;
                s.centroidY += r;
                s.minRow = std::min(s.minRow, r);
                s.maxRow = std::max(s.maxRow, r);
                s.minCol = std::min(s.minCol, c);
                s.maxCol = std::max(s.maxCol, c);
            }
        }
    }

    // Finalize statistics
    for (auto& s : stats) {
        if (s.area > 0) {
            s.centroidX /= s.area;
            s.centroidY /= s.area;
            s.boundingBox = Rect2i(s.minCol, s.minRow,
                                   s.maxCol - s.minCol + 1,
                                   s.maxRow - s.minRow + 1);
        }
    }

    return stats;
}

QImage ExtractComponent(const QImage& labels, int32_t label) {
    if (labels.Empty() || label <= 0) return QImage();

    int32_t width = labels.Width();
    int32_t height = labels.Height();

    QImage result(width, height, PixelType::UInt8, ChannelType::Gray);

    const uint8_t* srcData = static_cast<const uint8_t*>(labels.Data());
    uint8_t* dstData = static_cast<uint8_t*>(result.Data());
    int32_t srcStride = static_cast<int32_t>(labels.Stride());
    int32_t dstStride = static_cast<int32_t>(result.Stride());

    for (int32_t r = 0; r < height; ++r) {
        const uint8_t* srcRow = srcData + r * srcStride;
        uint8_t* dstRow = dstData + r * dstStride;
        for (int32_t c = 0; c < width; ++c) {
            dstRow[c] = (srcRow[c] == label) ? 255 : 0;
        }
    }

    return result;
}

std::vector<QImage> ExtractAllComponents(const QImage& labels, int32_t numLabels) {
    std::vector<QImage> components;
    components.reserve(numLabels);

    for (int32_t i = 1; i <= numLabels; ++i) {
        components.push_back(ExtractComponent(labels, i));
    }

    return components;
}

// =============================================================================
// Region-Based Connected Component Labeling (RLE)
// =============================================================================

// Note: SplitConnectedComponents and CountConnectedComponents are in RLEOps.cpp

QRegion GetLargestComponent(const QRegion& region, Connectivity connectivity) {
    auto components = SplitConnectedComponents(region, connectivity);
    if (components.empty()) return QRegion();

    auto it = std::max_element(components.begin(), components.end(),
        [](const QRegion& a, const QRegion& b) {
            return a.Area() < b.Area();
        });

    return *it;
}

std::vector<QRegion> GetLargestComponents(const QRegion& region,
                                           int32_t n,
                                           Connectivity connectivity) {
    auto components = SplitConnectedComponents(region, connectivity);
    if (components.empty()) return {};

    // Sort by area descending
    std::sort(components.begin(), components.end(),
        [](const QRegion& a, const QRegion& b) {
            return a.Area() > b.Area();
        });

    // Take top n
    if (static_cast<int32_t>(components.size()) > n) {
        components.resize(n);
    }

    return components;
}

// =============================================================================
// Component Filtering
// =============================================================================

std::vector<QRegion> FilterByArea(const std::vector<QRegion>& components,
                                   int64_t minArea,
                                   int64_t maxArea) {
    std::vector<QRegion> result;
    result.reserve(components.size());

    for (const auto& comp : components) {
        int64_t area = comp.Area();
        if (area >= minArea && (maxArea == 0 || area <= maxArea)) {
            result.push_back(comp);
        }
    }

    return result;
}

std::vector<QRegion> FilterBySize(const std::vector<QRegion>& components,
                                   int32_t minWidth, int32_t maxWidth,
                                   int32_t minHeight, int32_t maxHeight) {
    std::vector<QRegion> result;
    result.reserve(components.size());

    for (const auto& comp : components) {
        Rect2i bbox = comp.BoundingBox();
        if (bbox.width >= minWidth && (maxWidth == 0 || bbox.width <= maxWidth) &&
            bbox.height >= minHeight && (maxHeight == 0 || bbox.height <= maxHeight)) {
            result.push_back(comp);
        }
    }

    return result;
}

std::vector<QRegion> FilterByAspectRatio(const std::vector<QRegion>& components,
                                          double minRatio,
                                          double maxRatio) {
    std::vector<QRegion> result;
    result.reserve(components.size());

    for (const auto& comp : components) {
        Rect2i bbox = comp.BoundingBox();
        if (bbox.height == 0) continue;

        double ratio = static_cast<double>(bbox.width) / bbox.height;
        if (ratio >= minRatio && ratio <= maxRatio) {
            result.push_back(comp);
        }
    }

    return result;
}

std::vector<QRegion> FilterByPredicate(const std::vector<QRegion>& components,
                                        std::function<bool(const QRegion&)> predicate) {
    std::vector<QRegion> result;
    result.reserve(components.size());

    for (const auto& comp : components) {
        if (predicate(comp)) {
            result.push_back(comp);
        }
    }

    return result;
}

std::vector<QRegion> SelectBorderComponents(const std::vector<QRegion>& components,
                                             const Rect2i& bounds) {
    std::vector<QRegion> result;

    for (const auto& comp : components) {
        bool touchesBorder = false;

        for (const auto& run : comp.Runs()) {
            // Check top/bottom border
            if (run.row == bounds.y || run.row == bounds.y + bounds.height - 1) {
                touchesBorder = true;
                break;
            }
            // Check left border
            if (run.colBegin <= bounds.x) {
                touchesBorder = true;
                break;
            }
            // Check right border
            if (run.colEnd >= bounds.x + bounds.width) {
                touchesBorder = true;
                break;
            }
        }

        if (touchesBorder) {
            result.push_back(comp);
        }
    }

    return result;
}

std::vector<QRegion> RemoveBorderComponents(const std::vector<QRegion>& components,
                                             const Rect2i& bounds) {
    std::vector<QRegion> result;

    for (const auto& comp : components) {
        bool touchesBorder = false;

        for (const auto& run : comp.Runs()) {
            if (run.row == bounds.y || run.row == bounds.y + bounds.height - 1) {
                touchesBorder = true;
                break;
            }
            if (run.colBegin <= bounds.x) {
                touchesBorder = true;
                break;
            }
            if (run.colEnd >= bounds.x + bounds.width) {
                touchesBorder = true;
                break;
            }
        }

        if (!touchesBorder) {
            result.push_back(comp);
        }
    }

    return result;
}

// =============================================================================
// Component Merging
// =============================================================================

QRegion MergeComponents(const std::vector<QRegion>& components) {
    if (components.empty()) return QRegion();
    if (components.size() == 1) return components[0];

    // Collect all runs
    std::vector<QRegion::Run> allRuns;
    for (const auto& comp : components) {
        const auto& runs = comp.Runs();
        allRuns.insert(allRuns.end(), runs.begin(), runs.end());
    }

    // Sort and normalize (merge overlapping runs)
    NormalizeRuns(allRuns);
    return QRegion(allRuns);
}

std::vector<QRegion> MergeNearbyComponents(const std::vector<QRegion>& components,
                                            double maxDistance) {
    if (components.empty()) return {};
    if (components.size() == 1) return components;

    int32_t n = static_cast<int32_t>(components.size());
    UnionFind uf(n);

    // Compute centroids and bounding boxes
    std::vector<Point2d> centroids(n);
    std::vector<Rect2i> bboxes(n);

    for (int32_t i = 0; i < n; ++i) {
        auto centroid = ComputeCentroid(components[i].Runs());
        centroids[i] = Point2d(centroid.x, centroid.y);
        bboxes[i] = components[i].BoundingBox();
    }

    // Find nearby pairs (simple O(n^2) for now)
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = i + 1; j < n; ++j) {
            // Quick check using bounding boxes
            int32_t dx = std::max(0, std::max(bboxes[i].x - bboxes[j].x - bboxes[j].width,
                                               bboxes[j].x - bboxes[i].x - bboxes[i].width));
            int32_t dy = std::max(0, std::max(bboxes[i].y - bboxes[j].y - bboxes[j].height,
                                               bboxes[j].y - bboxes[i].y - bboxes[i].height));

            double dist = std::sqrt(dx * dx + dy * dy);
            if (dist <= maxDistance) {
                uf.Union(i, j);
            }
        }
    }

    // Group components by root
    std::unordered_map<int32_t, std::vector<int32_t>> groups;
    for (int32_t i = 0; i < n; ++i) {
        groups[uf.Find(i)].push_back(i);
    }

    // Merge each group
    std::vector<QRegion> result;
    result.reserve(groups.size());

    for (const auto& [root, indices] : groups) {
        std::vector<QRegion> groupComponents;
        for (int32_t idx : indices) {
            groupComponents.push_back(components[idx]);
        }
        result.push_back(MergeComponents(groupComponents));
    }

    return result;
}

// =============================================================================
// Hole Detection
// =============================================================================

std::vector<QRegion> FindHoles(const QRegion& region, const Rect2i& bounds) {
    if (region.Empty()) return {};

    // Complement the region within bounds
    auto complement = ComplementRuns(region.Runs(), bounds);
    if (complement.empty()) return {};

    // Split complement into connected components
    auto components = Internal::SplitConnectedComponents(complement, Connectivity::Four);

    // Remove the background component (touches border)
    std::vector<QRegion> holes;
    for (const auto& comp : components) {
        bool touchesBorder = false;

        for (const auto& run : comp.Runs()) {
            if (run.row == bounds.y || run.row == bounds.y + bounds.height - 1) {
                touchesBorder = true;
                break;
            }
            if (run.colBegin <= bounds.x || run.colEnd >= bounds.x + bounds.width) {
                touchesBorder = true;
                break;
            }
        }

        if (!touchesBorder) {
            holes.push_back(comp);
        }
    }

    return holes;
}

bool HasHoles(const QRegion& region, const Rect2i& bounds) {
    auto holes = FindHoles(region, bounds);
    return !holes.empty();
}

int32_t CountHoles(const QRegion& region, const Rect2i& bounds) {
    auto holes = FindHoles(region, bounds);
    return static_cast<int32_t>(holes.size());
}

} // namespace Qi::Vision::Internal
