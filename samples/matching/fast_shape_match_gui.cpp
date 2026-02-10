/**
 * @file fast_shape_match_gui.cpp
 * @brief Interactive fast shape matching demo.
 *
 * Default behavior:
 * - Load tests/data/halcon_images/clip.png
 * - Draw ROI interactively to create template
 * - Match in same image (maxMatches=20, minScore=0.6)
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Display/Draw.h>
#include <QiVision/GUI/Window.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Matching/FastShapeModel.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::GUI;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::Matching;

namespace {
namespace fs = std::filesystem;

std::string NormalizePath(std::string path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    const std::string prefix = "//wsl.localhost/Ubuntu";
    if (path.rfind(prefix, 0) == 0) {
        path = path.substr(prefix.size());
        if (path.empty() || path[0] != '/') {
            path = "/" + path;
        }
    }
    return path;
}

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool IsImageFile(const fs::path& p) {
    if (!fs::is_regular_file(p)) {
        return false;
    }
    const std::string ext = ToLower(p.extension().string());
    static const char* kExts[] = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm", ".ppm"
    };
    for (const char* e : kExts) {
        if (ext == e) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> CollectInputImages(const std::string& inputPath) {
    std::vector<std::string> paths;
    fs::path p(inputPath);
    if (!fs::exists(p)) {
        return paths;
    }

    if (fs::is_regular_file(p) && IsImageFile(p)) {
        paths.push_back(p.string());
        return paths;
    }

    if (fs::is_directory(p)) {
        for (const auto& entry : fs::directory_iterator(p)) {
            if (IsImageFile(entry.path())) {
                paths.push_back(entry.path().string());
            }
        }
        std::sort(paths.begin(), paths.end());
    }

    return paths;
}

std::string BuildOutputPath(const std::string& imagePath, size_t index) {
    fs::path p(imagePath);
    std::ostringstream oss;
    oss << "tests/output/fast_shape_match_gui_result_" << index << "_" << p.stem().string() << ".bmp";
    return oss.str();
}

Rect2i ToRect(const ROIResult& roi) {
    Rect2i r;
    r.x = static_cast<int32_t>(std::floor(std::min(roi.col1, roi.col2)));
    r.y = static_cast<int32_t>(std::floor(std::min(roi.row1, roi.row2)));
    r.width = static_cast<int32_t>(std::ceil(std::abs(roi.col2 - roi.col1)));
    r.height = static_cast<int32_t>(std::ceil(std::abs(roi.row2 - roi.row1)));
    return r;
}

} // namespace

int main(int argc, char* argv[]) {
    constexpr double kPi = 3.14159265358979323846;
    auto rad = [](double deg) { return deg * kPi / 180.0; };
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](const std::chrono::steady_clock::time_point& t0,
                 const std::chrono::steady_clock::time_point& t1) {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };
    const auto appStart = now();

    std::string inputPath = "tests/data/halcon_images/clip.png";
    if (argc > 1) {
        inputPath = argv[1];
    }
    inputPath = NormalizePath(inputPath);

    std::vector<std::string> imagePaths = CollectInputImages(inputPath);
    if (imagePaths.empty()) {
        std::cerr << "No valid image found in: " << inputPath << std::endl;
        std::cerr << "Usage: " << argv[0] << " [image_file_or_directory]" << std::endl;
        return 1;
    }

    fs::create_directories("tests/output");

    const std::string templatePath = imagePaths.front();

    QImage templateGray;
    ReadImageGray(templatePath, templateGray);
    if (templateGray.Empty()) {
        std::cerr << "Failed to load template image: " << templatePath << std::endl;
        return 1;
    }

    Window win("Fast Shape Match - Draw ROI", 1200, 860);
    win.SetAutoResize(true, 1600, 1000);
    win.SetResizable(true);
    win.Move(60, 60);
    win.EnablePixelInfo(true);
    win.DispImage(templateGray, ScaleMode::Fit);
    for (int i = 0; i < 20 && win.IsOpen(); ++i) {
        win.WaitKey(10);
    }

    if (!win.IsOpen()) {
        std::cerr << "Window failed to open (check DISPLAY/X11)." << std::endl;
        return 1;
    }

    std::cout << "Template image: " << templatePath << "\n";
    if (imagePaths.size() > 1) {
        std::cout << "Batch mode: use first image as template, match next "
                  << (imagePaths.size() - 1) << " images.\n";
    }
    std::cout << "Step1: Drag mouse to draw ROI. ESC to cancel.\n";
    ROIResult roiRes = win.DrawRectangle();
    if (!roiRes.valid) {
        std::cerr << "ROI drawing cancelled." << std::endl;
        return 1;
    }

    Rect2i roi = ToRect(roiRes);
    if (roi.width < 8 || roi.height < 8) {
        std::cerr << "ROI too small. Need at least 8x8." << std::endl;
        return 1;
    }

    FastShapeModel model;
    FastShapeModelStrategy strategy;
    strategy.tAtLevel = {5, 8};
    strategy.weakThreshold = 20.0;
    strategy.strongThreshold = 60.0;
    strategy.numFeatures = 63;
    const auto tModel0 = now();
    CreateFastShapeModel(
        templateGray, roi, model,
        0,             // auto levels
        0.0, rad(360), // full rotation
        0.0,           // auto angle step
        strategy
    );
    const auto tModel1 = now();
    if (!model.IsValid()) {
        std::cerr << "CreateFastShapeModel failed." << std::endl;
        return 1;
    }
    std::cout << "Model creation time: " << ms(tModel0, tModel1) << " ms\n";

    std::vector<Point2d> modelFeat;
    GetFastShapeModelFeaturePoints(model, modelFeat);

    bool uiExitRequested = false;

    std::vector<std::string> targets;
    if (imagePaths.size() == 1) {
        targets = imagePaths;
    } else {
        targets.assign(imagePaths.begin() + 1, imagePaths.end());
    }

    double totalLoadMs = 0.0;
    double totalFindMs = 0.0;
    double totalDrawMs = 0.0;
    double totalSaveMs = 0.0;
    int32_t processed = 0;

    for (size_t ti = 0; ti < targets.size(); ++ti) {
        const auto tLoad0 = now();
        QImage gray;
        ReadImageGray(targets[ti], gray);
        const auto tLoad1 = now();
        if (gray.Empty()) {
            std::cerr << "Skip unreadable image: " << targets[ti] << "\n";
            continue;
        }
        totalLoadMs += ms(tLoad0, tLoad1);
        processed++;

        std::vector<double> rows, cols, angles, scores, scales;
        const auto tFind0 = now();
        FindFastShapeModel(
            gray, model,
            0.8, // default min score
            20,  // max matches
            0.35,
            0.9,
            rows, cols, angles, scores, &scales
        );
        const auto tFind1 = now();
        totalFindMs += ms(tFind0, tFind1);

        const auto tDraw0 = now();
        QImage vis;
        Color::GrayToRgb(gray, vis);
        for (size_t i = 0; i < rows.size(); ++i) {
            const double scale = (i < scales.size()) ? scales[i] : 1.0;
            Draw::RotatedRectangle(vis, Point2d(cols[i], rows[i]),
                                   static_cast<double>(roi.width) * scale,
                                   static_cast<double>(roi.height) * scale,
                                   angles[i], Scalar::Green(), 1);

            double c = std::cos(angles[i]);
            double s = std::sin(angles[i]);
            for (const auto& p : modelFeat) {
                Point2d pt{
                    c * p.x - s * p.y + cols[i],
                    s * p.x + c * p.y + rows[i]
                };
                Draw::Cross(vis, pt, 3, Scalar::Green(), 1);
            }
        }

        Draw::Text(vis, 12, 12, "Keys: Enter=Next, Q/Esc=Exit", Scalar::Yellow(), 1);
        const auto tDraw1 = now();
        totalDrawMs += ms(tDraw0, tDraw1);

        std::cout << "[" << (ti + 1) << "/" << targets.size() << "] "
                  << targets[ti] << " => Matches: " << rows.size()
                  << " (max 20, minScore 0.8)\n";
        if (!scores.empty()) {
            std::cout << "Top score: " << scores[0] << "\n";
            if (!scales.empty()) {
                std::cout << "Top scale: " << scales[0] << "\n";
            }
        }

        const std::string outPath = BuildOutputPath(targets[ti], ti + 1);
        const auto tSave0 = now();
        WriteImage(vis, outPath);
        const auto tSave1 = now();
        totalSaveMs += ms(tSave0, tSave1);
        std::cout << "Saved: " << outPath << "\n";
        std::cout << "Timing(ms): load=" << ms(tLoad0, tLoad1)
                  << ", match=" << ms(tFind0, tFind1)
                  << ", draw=" << ms(tDraw0, tDraw1)
                  << ", save=" << ms(tSave0, tSave1) << "\n";

        fs::path namePath(targets[ti]);
        win.SetTitle("Fast Shape Match Result - " + namePath.filename().string());
        uiExitRequested = false;
        win.DispImage(vis, ScaleMode::Fit);

        while (win.IsOpen()) {
            int32_t key = win.WaitKey(30);
            if (key == 27 || key == 'q' || key == 'Q') {
                uiExitRequested = true;
                break;
            }
            // Cross-platform Enter handling:
            // - Windows: VK_RETURN (13)
            // - X11 main Enter: XK_Return (65293)
            // - X11 keypad Enter: XK_KP_Enter (65421)
            if (key == 13 || key == 10 || key == '\r' || key == '\n' ||
                key == 65293 || key == 65421) {
                break;
            }
        }

        if (!win.IsOpen() || uiExitRequested) {
            break;
        }
    }

    if (win.IsOpen()) {
        const auto appEnd = now();
        if (processed > 0) {
            std::cout << "Summary timing(ms): avg_load=" << (totalLoadMs / processed)
                      << ", avg_match=" << (totalFindMs / processed)
                      << ", avg_draw=" << (totalDrawMs / processed)
                      << ", avg_save=" << (totalSaveMs / processed) << "\n";
        }
        std::cout << "Total elapsed: " << ms(appStart, appEnd) << " ms\n";
        std::cout << "Press any key to close..." << std::endl;
        win.WaitKey(0);
    }
    return 0;
}
