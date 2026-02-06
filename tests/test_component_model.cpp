/**
 * @file test_component_model.cpp
 * @brief Basic ComponentModel matching tests
 */

#include <QiVision/Matching/ComponentModel.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Exception.h>

#include <cmath>
#include <iostream>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;

int testsPassed = 0;
int testsFailed = 0;

void AssertTrue(bool condition, const char* message) {
    if (condition) {
        std::cout << "  [PASS] " << message << "\n";
        testsPassed++;
    } else {
        std::cout << "  [FAIL] " << message << "\n";
        testsFailed++;
    }
}

void AssertNear(double a, double b, double eps, const char* message) {
    bool ok = std::abs(a - b) < eps;
    if (ok) {
        std::cout << "  [PASS] " << message << " (" << a << " ~= " << b << ")\n";
        testsPassed++;
    } else {
        std::cout << "  [FAIL] " << message << " (" << a << " != " << b << ")\n";
        testsFailed++;
    }
}

QImage CreateRootTemplate() {
    QImage img(32, 32, PixelType::UInt8, ChannelType::Gray);
    for (int y = 0; y < img.Height(); ++y) {
        uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int x = 0; x < img.Width(); ++x) {
            bool isCross = (std::abs(x - 16) <= 2) || (std::abs(y - 16) <= 2);
            row[x] = isCross ? 220 : 30;
        }
    }
    return img;
}

QImage CreateChildTemplate() {
    QImage img(16, 16, PixelType::UInt8, ChannelType::Gray);
    for (int y = 0; y < img.Height(); ++y) {
        uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int x = 0; x < img.Width(); ++x) {
            bool isFilled = (x >= 3 && x <= 12 && y >= 3 && y <= 12);
            row[x] = isFilled ? 200 : 50;
        }
    }
    return img;
}

void Blit(const QImage& src, QImage& dst, int offsetX, int offsetY) {
    for (int y = 0; y < src.Height(); ++y) {
        const uint8_t* srcRow = static_cast<const uint8_t*>(src.RowPtr(y));
        uint8_t* dstRow = static_cast<uint8_t*>(dst.RowPtr(offsetY + y));
        for (int x = 0; x < src.Width(); ++x) {
            dstRow[offsetX + x] = srcRow[x];
        }
    }
}

SearchParams MakeParams(double minScore) {
    SearchParams params;
    params.minScore = minScore;
    params.maxMatches = 5;
    params.angleMode = AngleSearchMode::Range;
    params.angleStart = 0.0;
    params.angleExtent = 0.0; // no rotation
    params.subpixelMethod = SubpixelMethod::Parabolic;
    params.numLevels = 3;
    return params;
}

int main() {
    std::cout << "=== ComponentModel Test ===\n\n";

    QImage rootTemplate = CreateRootTemplate();
    QImage childTemplate = CreateChildTemplate();

    NCCModel rootModel;
    NCCModel childModel;
    CreateNCCModel(rootTemplate, rootModel, 3, 0.0, 0.0, 0.0, "use_polarity");
    CreateNCCModel(childTemplate, childModel, 3, 0.0, 0.0, 0.0, "use_polarity");

    AssertTrue(rootModel.IsValid(), "Root model created");
    AssertTrue(childModel.IsValid(), "Child model created");

    // Create a search image with root + child at known offset
    QImage search(160, 160, PixelType::UInt8, ChannelType::Gray);
    for (int y = 0; y < search.Height(); ++y) {
        uint8_t* row = static_cast<uint8_t*>(search.RowPtr(y));
        for (int x = 0; x < search.Width(); ++x) {
            row[x] = 100;
        }
    }

    const int rootX = 30;
    const int rootY = 40;
    const int childX = 86;
    const int childY = 70;

    Blit(rootTemplate, search, rootX, rootY);
    Blit(childTemplate, search, childX, childY);

    // Build component model
    ComponentModel comp;
    CreateComponentModel(comp);
    int32_t rootIdx = AddComponent(comp, rootModel, MakeParams(0.85));
    int32_t childIdx = AddComponent(comp, childModel, MakeParams(0.80));
    SetComponentRoot(comp, rootIdx);

    const Point2d rootCenter{rootX + rootTemplate.Width() * 0.5,
                             rootY + rootTemplate.Height() * 0.5};
    const Point2d childCenter{childX + childTemplate.Width() * 0.5,
                              childY + childTemplate.Height() * 0.5};

    ComponentConstraint constraint;
    constraint.offset = Point2d{childCenter.x - rootCenter.x, childCenter.y - rootCenter.y};
    constraint.positionTolerance = 3.0;
    constraint.angleTolerance = 0.1;
    constraint.scale = 1.0;
    constraint.scaleTolerance = 0.2;
    constraint.weight = 1.0;

    SetComponentRelation(comp, childIdx, rootIdx, constraint);

    std::vector<ComponentMatch> matches;
    FindComponentModel(search, comp, 0.75, 0, matches);

    AssertTrue(!matches.empty(), "Found component match");
    if (!matches.empty()) {
        const auto& group = matches.front();
        AssertTrue(group.Size() == 2, "Component count == 2");
        AssertTrue(group.score >= 0.75, "Group score above threshold");

        const auto& rootMatch = group.components[rootIdx];
        const auto& childMatch = group.components[childIdx];

        AssertNear(rootMatch.x, rootCenter.x, 1.0, "Root X near expected");
        AssertNear(rootMatch.y, rootCenter.y, 1.0, "Root Y near expected");
        AssertNear(childMatch.x, childCenter.x, 1.0, "Child X near expected");
        AssertNear(childMatch.y, childCenter.y, 1.0, "Child Y near expected");
    }

    // Constraint rejection: move child far away
    QImage searchBad(160, 160, PixelType::UInt8, ChannelType::Gray);
    for (int y = 0; y < searchBad.Height(); ++y) {
        uint8_t* row = static_cast<uint8_t*>(searchBad.RowPtr(y));
        for (int x = 0; x < searchBad.Width(); ++x) {
            row[x] = 100;
        }
    }
    Blit(rootTemplate, searchBad, rootX, rootY);
    Blit(childTemplate, searchBad, 120, 10); // far away

    matches.clear();
    FindComponentModel(searchBad, comp, 0.75, 0, matches);
    AssertTrue(matches.empty(), "Constraint rejected distant child");

    // Multiple roots: only one has valid child
    QImage searchMulti(200, 200, PixelType::UInt8, ChannelType::Gray);
    for (int y = 0; y < searchMulti.Height(); ++y) {
        uint8_t* row = static_cast<uint8_t*>(searchMulti.RowPtr(y));
        for (int x = 0; x < searchMulti.Width(); ++x) {
            row[x] = 100;
        }
    }
    Blit(rootTemplate, searchMulti, rootX, rootY);
    Blit(childTemplate, searchMulti, childX, childY);

    const int root2X = 120;
    const int root2Y = 130;
    Blit(rootTemplate, searchMulti, root2X, root2Y);

    matches.clear();
    FindComponentModel(searchMulti, comp, 0.75, 0, matches);
    AssertTrue(matches.size() == 1, "Only one group passes constraints");

    std::cout << "\n=== Test Summary ===\n";
    std::cout << "Passed: " << testsPassed << "\n";
    std::cout << "Failed: " << testsFailed << "\n";
    return testsFailed > 0 ? 1 : 0;
}
