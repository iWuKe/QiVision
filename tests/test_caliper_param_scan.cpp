/**
 * @file test_caliper_param_scan.cpp
 * @brief Parameter sweep for Caliper/Metrology on real images
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Measure/Caliper.h>
#include <QiVision/Measure/CaliperArray.h>
#include <QiVision/Measure/MeasureHandle.h>
#include <QiVision/Measure/Metrology.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::Measure;

namespace {

struct ScanResult {
    double score = -std::numeric_limits<double>::infinity();
    std::string desc;
};

double Abs(double x) {
    return (x < 0.0) ? -x : x;
}

ScanResult ScanSingleCaliper(const QImage& gray) {
    const std::vector<double> sigmaList{0.8, 1.0, 1.2, 1.5};
    const std::vector<double> thrList{10.0, 15.0, 20.0, 25.0, 30.0};
    const std::vector<std::string> transitionList{"positive", "all"};
    const std::vector<std::string> selectList{"first", "strongest"};

    MeasureRectangle2 handle = GenMeasureRectangle2(420.0, 210.0, 0.0, 100.0, 10.0);

    ScanResult best;
    int total = 0;
    int valid = 0;
    for (double sigma : sigmaList) {
        for (double thr : thrList) {
            for (const auto& trans : transitionList) {
                for (const auto& sel : selectList) {
                    total++;
                    auto pairs = MeasurePairs(gray, handle, sigma, thr, trans, sel);
                    if (pairs.empty()) {
                        continue;
                    }
                    valid++;
                    const auto& p = pairs[0];
                    const double widthTarget = 160.0;
                    const double widthErr = Abs(p.width - widthTarget);
                    const double centerErr = Abs(p.centerColumn - 210.0);
                    double s = 0.0;
                    s += p.score * 100.0;
                    s += std::min(p.first.amplitude, 255.0) * 0.1;
                    s += std::min(p.second.amplitude, 255.0) * 0.1;
                    s -= widthErr * 0.8;
                    s -= centerErr * 0.5;
                    if (s > best.score) {
                        best.score = s;
                        std::ostringstream oss;
                        oss << "sigma=" << sigma
                            << ", threshold=" << thr
                            << ", transition=" << trans
                            << ", select=" << sel
                            << ", width=" << p.width
                            << ", centerX=" << p.centerColumn
                            << ", pairScore=" << p.score;
                        best.desc = oss.str();
                    }
                }
            }
        }
    }

    std::cout << "[SingleCaliper] valid/total = " << valid << "/" << total << "\n";
    return best;
}

ScanResult ScanCaliperArray(const QImage& gray) {
    const std::vector<int> countList{6, 8, 10, 12};
    const std::vector<double> lenList{70.0, 90.0, 110.0};
    const std::vector<double> widthList{6.0, 8.0, 10.0};
    const std::vector<double> sigmaList{0.8, 1.0, 1.2};
    const std::vector<double> thrList{15.0, 20.0, 25.0};

    ScanResult best;
    int total = 0;
    int valid = 0;
    for (int count : countList) {
        for (double len : lenList) {
            for (double w : widthList) {
                CaliperArray array;
                if (!array.CreateAlongLine(Point2d{210.0, 380.0}, Point2d{210.0, 460.0}, count, len, w)) {
                    continue;
                }
                for (double sigma : sigmaList) {
                    for (double thr : thrList) {
                        total++;
                        CaliperArrayStats stats;
                        auto r = array.MeasurePairs(gray, sigma, thr, "positive", "first", &stats);
                        if (r.numValid <= 0 || r.widths.empty()) {
                            continue;
                        }
                        valid++;
                        double validRatio = static_cast<double>(r.numValid) / std::max(1, r.numCalipers);
                        double s = 0.0;
                        s += validRatio * 120.0;
                        s += r.meanScore * 80.0;
                        s -= r.stdWidth * 0.4;
                        s -= stats.avgTimePerCaliper * 0.05;
                        if (s > best.score) {
                            best.score = s;
                            std::ostringstream oss;
                            oss << "count=" << count
                                << ", profileHalfLen=" << len
                                << ", handleWidth=" << w
                                << ", sigma=" << sigma
                                << ", threshold=" << thr
                                << ", numValid=" << r.numValid << "/" << r.numCalipers
                                << ", meanWidth=" << r.meanWidth
                                << ", stdWidth=" << r.stdWidth
                                << ", meanScore=" << r.meanScore
                                << ", avgTimePerCaliperMs=" << stats.avgTimePerCaliper;
                            best.desc = oss.str();
                        }
                    }
                }
            }
        }
    }
    std::cout << "[CaliperArray] valid/total = " << valid << "/" << total << "\n";
    return best;
}

ScanResult ScanCircleMetrology(const QImage& gray) {
    const std::vector<int> numMeasuresList{16, 24, 32, 40};
    const std::vector<double> len1List{15.0, 20.0, 25.0};
    const std::vector<double> len2List{4.0, 5.0, 6.0, 8.0};
    const std::vector<std::string> thresholdModeList{"auto", "manual20", "manual30"};
    const std::vector<std::string> fitMethodList{"ransac", "huber"};

    ScanResult best;
    int total = 0;
    int valid = 0;
    for (int n : numMeasuresList) {
        for (double l1 : len1List) {
            for (double l2 : len2List) {
                for (const auto& tm : thresholdModeList) {
                    for (const auto& fm : fitMethodList) {
                        total++;
                        MetrologyModel model;
                        MetrologyMeasureParams params;
                        params.SetNumMeasures(n)
                              .SetMeasureLength(l1, l2)
                              .SetFitMethod(fm)
                              .SetMeasureTransition("all")
                              .SetMeasureSelect("strongest");
                        if (tm == "auto") {
                            params.SetThreshold("auto");
                        } else if (tm == "manual20") {
                            params.SetThreshold(20.0);
                        } else {
                            params.SetThreshold(30.0);
                        }

                        int idx = model.AddCircleMeasure(420.0, 210.0, 63.0, l1, l2, "all", "strongest", params);
                        if (idx < 0 || !model.Apply(gray)) {
                            continue;
                        }
                        auto c = model.GetCircleResult(idx);
                        if (!c.IsValid()) {
                            continue;
                        }
                        valid++;

                        const double cxTarget = 208.4;
                        const double cyTarget = 420.6;
                        const double rTarget = 63.0;
                        const double centerErr = std::hypot(c.column - cxTarget, c.row - cyTarget);
                        const double radiusErr = Abs(c.radius - rTarget);
                        double s = 0.0;
                        s += c.score * 100.0;
                        s += c.numUsed * 1.5;
                        s -= c.rmsError * 2.0;
                        s -= centerErr * 2.5;
                        s -= radiusErr * 2.0;

                        if (s > best.score) {
                            best.score = s;
                            std::ostringstream oss;
                            oss << "numMeasures=" << n
                                << ", measureLength1=" << l1
                                << ", measureLength2=" << l2
                                << ", thresholdMode=" << tm
                                << ", fitMethod=" << fm
                                << ", center=(" << c.column << ", " << c.row << ")"
                                << ", radius=" << c.radius
                                << ", numUsed=" << c.numUsed
                                << ", score=" << c.score
                                << ", rms=" << c.rmsError;
                            best.desc = oss.str();
                        }
                    }
                }
            }
        }
    }
    std::cout << "[CircleMetrology] valid/total = " << valid << "/" << total << "\n";
    return best;
}

ScanResult ScanLineMetrology(const QImage& gray) {
    const std::vector<int> numMeasuresList{16, 20, 24, 28};
    const std::vector<double> len1List{20.0, 30.0, 40.0};
    const std::vector<double> len2List{4.0, 5.0, 6.0};
    const std::vector<std::string> thresholdModeList{"auto", "manual20", "manual30"};
    const std::vector<std::string> fitMethodList{"ransac", "huber"};

    ScanResult best;
    int total = 0;
    int valid = 0;
    for (int n : numMeasuresList) {
        for (double l1 : len1List) {
            for (double l2 : len2List) {
                for (const auto& tm : thresholdModeList) {
                    for (const auto& fm : fitMethodList) {
                        total++;
                        MetrologyModel model;
                        MetrologyMeasureParams params;
                        params.SetNumMeasures(n)
                              .SetMeasureLength(l1, l2)
                              .SetFitMethod(fm)
                              .SetMeasureTransition("all")
                              .SetMeasureSelect("strongest")
                              .SetDistanceThreshold(2.0);
                        if (tm == "auto") {
                            params.SetThreshold("auto");
                        } else if (tm == "manual20") {
                            params.SetThreshold(20.0);
                        } else {
                            params.SetThreshold(30.0);
                        }

                        int idx = model.AddLineMeasure(120.0, 100.0, 120.0, gray.Width() - 100.0,
                                                       l1, l2, "all", "strongest", params);
                        if (idx < 0 || !model.Apply(gray)) {
                            continue;
                        }
                        auto line = model.GetLineResult(idx);
                        if (!line.IsValid()) {
                            continue;
                        }
                        valid++;

                        double angleDeg = std::atan2(line.row2 - line.row1, line.col2 - line.col1) * 180.0 / 3.14159265358979323846;
                        double angleErr = Abs(angleDeg);
                        double s = 0.0;
                        s += line.score * 100.0;
                        s += line.numUsed * 1.5;
                        s -= line.rmsError * 1.5;
                        s -= angleErr * 1.0;

                        if (s > best.score) {
                            best.score = s;
                            std::ostringstream oss;
                            oss << "numMeasures=" << n
                                << ", measureLength1=" << l1
                                << ", measureLength2=" << l2
                                << ", thresholdMode=" << tm
                                << ", fitMethod=" << fm
                                << ", angleDeg=" << angleDeg
                                << ", numUsed=" << line.numUsed
                                << ", score=" << line.score
                                << ", rms=" << line.rmsError;
                            best.desc = oss.str();
                        }
                    }
                }
            }
        }
    }
    std::cout << "[LineMetrology] valid/total = " << valid << "/" << total << "\n";
    return best;
}

} // namespace

int main() {
    std::cout << "=== Caliper Parameter Scan ===\n\n";

    QImage circlePlate;
    QImage ic;
    ReadImageGray("tests/data/halcon_images/circle_plate.png", circlePlate);
    ReadImageGray("tests/data/halcon_images/ic.png", ic);
    if (circlePlate.Empty() || ic.Empty()) {
        std::cerr << "Failed to load required images.\n";
        return 1;
    }

    auto bestSingle = ScanSingleCaliper(circlePlate);
    auto bestArray = ScanCaliperArray(circlePlate);
    auto bestCircle = ScanCircleMetrology(circlePlate);
    auto bestLine = ScanLineMetrology(ic);

    std::cout << "\n=== Best Parameters ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[SingleCaliper] score=" << bestSingle.score << "\n  " << bestSingle.desc << "\n";
    std::cout << "[CaliperArray] score=" << bestArray.score << "\n  " << bestArray.desc << "\n";
    std::cout << "[CircleMetrology] score=" << bestCircle.score << "\n  " << bestCircle.desc << "\n";
    std::cout << "[LineMetrology] score=" << bestLine.score << "\n  " << bestLine.desc << "\n";

    return 0;
}

