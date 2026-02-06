/**
 * @file component_model_demo.cpp
 * @brief ComponentModel demo (interactive-style console output)
 */

#include <QiVision/Matching/ComponentModel.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/GUI/Window.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/Color/ColorConvert.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;

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

bool ParseInt(const char* s, int& out) {
    try {
        out = std::stoi(s);
        return true;
    } catch (...) {
        return false;
    }
}

void PrintUsage(const char* exe) {
    std::cout << "Usage:\n";
    std::cout << "  " << exe << "\n";
    std::cout << "  " << exe << " <image_path>\n";
    std::cout << "  " << exe << " <image_dir>\n";
    std::cout << "  " << exe << " <image_path> <rx> <ry> <rw> <rh> <cx> <cy> <cw> <ch>\n";
    std::cout << "  " << exe << " <image_path> <rx> <ry> <rw> <rh> <cx> <cy> <cw> <ch>"
              << " <gx> <gy> <gw> <gh>\n";
    std::cout << "\nIf no args are given, a synthetic image is used.\n";
}

bool IsImageFile(const std::filesystem::path& path) {
    if (!path.has_extension()) {
        return false;
    }
    std::string ext = path.extension().string();
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext == ".bmp" || ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tif" || ext == ".tiff";
}

int main(int argc, char** argv) {
    std::cout << "=== ComponentModel Demo ===\n\n";

    QImage rootTemplate;
    QImage childTemplate;
    QImage grandTemplate;
    QImage templateImage;
    QImage search;

    std::vector<std::filesystem::path> imageList;
    bool batchMode = false;

    int rx = 0, ry = 0, rw = 0, rh = 0, cx = 0, cy = 0, cw = 0, ch = 0;
    int gx = 0, gy = 0, gw = 0, gh = 0;
    bool useGrandChild = false;

    if (argc == 1) {
        // Synthetic mode
        rootTemplate = CreateRootTemplate();
        childTemplate = CreateChildTemplate();

        templateImage = QImage(160, 160, PixelType::UInt8, ChannelType::Gray);
        for (int y = 0; y < templateImage.Height(); ++y) {
            uint8_t* row = static_cast<uint8_t*>(templateImage.RowPtr(y));
            for (int x = 0; x < templateImage.Width(); ++x) {
                row[x] = 100;
            }
        }
        search = templateImage.Clone();
    } else if (argc == 2 || argc == 10 || argc == 14) {
        // Real image or directory mode with GUI ROI or explicit ROI
        std::filesystem::path inputPath(argv[1]);
        if (std::filesystem::is_directory(inputPath)) {
            for (const auto& entry : std::filesystem::directory_iterator(inputPath)) {
                if (entry.is_regular_file() && IsImageFile(entry.path())) {
                    imageList.push_back(entry.path());
                }
            }
            std::sort(imageList.begin(), imageList.end());
            if (imageList.empty()) {
                std::cout << "No images found in directory: " << inputPath.string() << "\n";
                return 1;
            }
            batchMode = true;
            IO::ReadImageGray(imageList.front().string(), templateImage);
        } else {
            imageList.push_back(inputPath);
            IO::ReadImageGray(inputPath.string(), templateImage);
        }

        if (!templateImage.IsValid() || templateImage.Empty()) {
            std::cout << "Failed to read image: " << inputPath.string() << "\n";
            return 1;
        }

        if (argc == 2) {
            using namespace Qi::Vision::GUI;
            std::cout << "Draw ROOT ROI (drag rectangle, release to confirm).\n";
            Window winRoot("Draw ROOT ROI");
            winRoot.SetAutoResize(true, 1280, 960);
            winRoot.DispImage(templateImage, ScaleMode::Fit);
            ROIResult roiRoot = winRoot.DrawRectangle();
            if (!roiRoot.valid) {
                std::cout << "ROI selection cancelled.\n";
                return 1;
            }

            std::cout << "Draw CHILD ROI (drag rectangle, release to confirm).\n";
            Window winChild("Draw CHILD ROI");
            winChild.SetAutoResize(true, 1280, 960);
            winChild.DispImage(templateImage, ScaleMode::Fit);
            ROIResult roiChild = winChild.DrawRectangle();
            if (!roiChild.valid) {
                std::cout << "ROI selection cancelled.\n";
                return 1;
            }

            rx = static_cast<int>(std::min(roiRoot.col1, roiRoot.col2));
            ry = static_cast<int>(std::min(roiRoot.row1, roiRoot.row2));
            rw = static_cast<int>(std::abs(roiRoot.col2 - roiRoot.col1));
            rh = static_cast<int>(std::abs(roiRoot.row2 - roiRoot.row1));

            cx = static_cast<int>(std::min(roiChild.col1, roiChild.col2));
            cy = static_cast<int>(std::min(roiChild.row1, roiChild.row2));
            cw = static_cast<int>(std::abs(roiChild.col2 - roiChild.col1));
            ch = static_cast<int>(std::abs(roiChild.row2 - roiChild.row1));

            std::cout << "Add GRANDCHILD ROI? (y/N): ";
            std::string answer;
            std::getline(std::cin, answer);
            if (!answer.empty() && (answer[0] == 'y' || answer[0] == 'Y')) {
                std::cout << "Draw GRANDCHILD ROI (drag rectangle, release to confirm).\n";
                Window winGrand("Draw GRANDCHILD ROI");
                winGrand.SetAutoResize(true, 1280, 960);
                winGrand.DispImage(templateImage, ScaleMode::Fit);
                ROIResult roiGrand = winGrand.DrawRectangle();
                if (!roiGrand.valid) {
                    std::cout << "ROI selection cancelled.\n";
                    return 1;
                }
                gx = static_cast<int>(std::min(roiGrand.col1, roiGrand.col2));
                gy = static_cast<int>(std::min(roiGrand.row1, roiGrand.row2));
                gw = static_cast<int>(std::abs(roiGrand.col2 - roiGrand.col1));
                gh = static_cast<int>(std::abs(roiGrand.row2 - roiGrand.row1));
                useGrandChild = true;
            }
        } else {
            bool ok = ParseInt(argv[2], rx) && ParseInt(argv[3], ry) &&
                      ParseInt(argv[4], rw) && ParseInt(argv[5], rh) &&
                      ParseInt(argv[6], cx) && ParseInt(argv[7], cy) &&
                      ParseInt(argv[8], cw) && ParseInt(argv[9], ch);
            if (!ok) {
                PrintUsage(argv[0]);
                return 1;
            }
            if (argc == 14) {
                bool ok2 = ParseInt(argv[10], gx) && ParseInt(argv[11], gy) &&
                           ParseInt(argv[12], gw) && ParseInt(argv[13], gh);
                if (!ok2) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                useGrandChild = true;
            }
        }

        rootTemplate = templateImage.SubImage(rx, ry, rw, rh).Clone();
        childTemplate = templateImage.SubImage(cx, cy, cw, ch).Clone();
        if (useGrandChild) {
            grandTemplate = templateImage.SubImage(gx, gy, gw, gh).Clone();
        }
    } else {
        PrintUsage(argv[0]);
        return 1;
    }

    NCCModel rootModel;
    NCCModel childModel;
    NCCModel grandModel;
    CreateNCCModel(rootTemplate, rootModel, 3, 0.0, 0.0, 0.0, "use_polarity");
    CreateNCCModel(childTemplate, childModel, 3, 0.0, 0.0, 0.0, "use_polarity");
    if (useGrandChild) {
        CreateNCCModel(grandTemplate, grandModel, 3, 0.0, 0.0, 0.0, "use_polarity");
    }

    if (!rootModel.IsValid() || !childModel.IsValid() || (useGrandChild && !grandModel.IsValid())) {
        std::cout << "Model creation failed.\n";
        return 1;
    }

    int rootX = 30;
    int rootY = 40;
    int childX = 86;
    int childY = 70;

    if (argc == 1) {
        Blit(rootTemplate, search, rootX, rootY);
        Blit(childTemplate, search, childX, childY);
    }

    // Build component model
    ComponentModel comp;
    CreateComponentModel(comp);
    int32_t rootIdx = AddComponent(comp, rootModel, MakeParams(0.85));
    int32_t childIdx = AddComponent(comp, childModel, MakeParams(0.80));
    int32_t grandIdx = -1;
    if (useGrandChild) {
        grandIdx = AddComponent(comp, grandModel, MakeParams(0.80));
    }
    SetComponentRoot(comp, rootIdx);

    Point2d rootCenter;
    Point2d childCenter;
    Point2d grandCenter;
    if (argc == 1) {
        rootCenter = Point2d{rootX + rootTemplate.Width() * 0.5,
                             rootY + rootTemplate.Height() * 0.5};
        childCenter = Point2d{childX + childTemplate.Width() * 0.5,
                              childY + childTemplate.Height() * 0.5};
    } else {
        rootCenter = Point2d{rx + rw * 0.5, ry + rh * 0.5};
        childCenter = Point2d{cx + cw * 0.5, cy + ch * 0.5};
        if (useGrandChild) {
            grandCenter = Point2d{gx + gw * 0.5, gy + gh * 0.5};
        }
    }

    ComponentConstraint constraint;
    constraint.offset = Point2d{childCenter.x - rootCenter.x, childCenter.y - rootCenter.y};
    constraint.positionTolerance = 3.0;
    constraint.angleTolerance = 0.1;
    constraint.scale = 1.0;
    constraint.scaleTolerance = 0.2;
    constraint.weight = 1.0;

    SetComponentRelation(comp, childIdx, rootIdx, constraint);
    if (useGrandChild) {
        ComponentConstraint constraint2;
        constraint2.offset = Point2d{grandCenter.x - childCenter.x, grandCenter.y - childCenter.y};
        constraint2.positionTolerance = 3.0;
        constraint2.angleTolerance = 0.1;
        constraint2.scale = 1.0;
        constraint2.scaleTolerance = 0.2;
        constraint2.weight = 1.0;
        SetComponentRelation(comp, grandIdx, childIdx, constraint2);
    }

    using namespace Qi::Vision::GUI;
    Window win("ComponentModel Result");
    win.SetAutoResize(true, 1280, 960);

    auto processImage = [&](const QImage& img, const std::string& label) -> bool {
        std::vector<ComponentMatch> matches;
        FindComponentModel(img, comp, 0.75, 0, matches);

        std::cout << "Image: " << label << "\n";
        std::cout << "Matches: " << matches.size() << "\n";
        for (size_t i = 0; i < matches.size(); ++i) {
            const auto& group = matches[i];
            std::cout << "  Group " << i << ": score=" << group.score << "\n";
            for (size_t j = 0; j < group.components.size(); ++j) {
                const auto& m = group.components[j];
                std::cout << "    Component " << j
                          << " at (" << m.x << ", " << m.y << ")"
                          << " angle=" << m.angle
                          << " score=" << m.score << "\n";
            }
        }

        if (!img.Empty()) {
            QImage vis;
            if (img.Channels() == 1) {
                Color::GrayToRgb(img, vis);
            } else {
                vis = img.Clone();
            }

            for (const auto& group : matches) {
                if (group.components.size() < 2) {
                    continue;
                }
                const auto& rootMatch = group.components[rootIdx];
                const auto& childMatch = group.components[childIdx];

                Rect2i rootRect(
                    static_cast<int32_t>(rootMatch.x - rootTemplate.Width() * 0.5),
                    static_cast<int32_t>(rootMatch.y - rootTemplate.Height() * 0.5),
                    rootTemplate.Width(), rootTemplate.Height());
                Rect2i childRect(
                    static_cast<int32_t>(childMatch.x - childTemplate.Width() * 0.5),
                    static_cast<int32_t>(childMatch.y - childTemplate.Height() * 0.5),
                    childTemplate.Width(), childTemplate.Height());

                Draw::Rectangle(vis, rootRect, Scalar::Green(), 2);
                Draw::Cross(vis, rootMatch.x, rootMatch.y, 10, Scalar::Green(), 2);
                Draw::Rectangle(vis, childRect, Scalar::Red(), 2);
                Draw::Cross(vis, childMatch.x, childMatch.y, 10, Scalar::Red(), 2);

                if (useGrandChild && grandIdx >= 0 && grandIdx < static_cast<int32_t>(group.components.size())) {
                    const auto& grandMatch = group.components[grandIdx];
                    Rect2i grandRect(
                        static_cast<int32_t>(grandMatch.x - gw * 0.5),
                        static_cast<int32_t>(grandMatch.y - gh * 0.5),
                        gw, gh);
                    Draw::Rectangle(vis, grandRect, Scalar::Cyan(), 2);
                    Draw::Cross(vis, grandMatch.x, grandMatch.y, 10, Scalar::Cyan(), 2);
                }
            }

            win.SetTitle("ComponentModel Result - " + label);
            win.DispImage(vis, ScaleMode::Fit);
            int32_t key = win.WaitKey();
            if (key == 27 || key == 'q' || key == 'Q') {
                return false;
            }
        }

        return true;
    };

    if (argc == 1) {
        processImage(search, "synthetic");
    } else if (batchMode) {
        for (const auto& path : imageList) {
            QImage img;
            IO::ReadImageGray(path.string(), img);
            if (!img.IsValid() || img.Empty()) {
                std::cout << "Failed to read image: " << path.string() << "\n";
                continue;
            }
            if (!processImage(img, path.filename().string())) {
                break;
            }
        }
    } else {
        QImage img;
        IO::ReadImageGray(imageList.front().string(), img);
        processImage(img, imageList.front().filename().string());
    }

    std::cout << "\nTip: edit tolerances in samples/matching/component_model_demo.cpp to explore behavior.\n";
    return 0;
}
