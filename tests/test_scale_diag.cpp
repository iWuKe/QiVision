/**
 * @file test_scale_diag.cpp
 * @brief Diagnostic test: 4 switches x 5 runs for scaled matching accuracy
 */
#include <QiVision/Core/QImage.h>
#include <QiVision/Matching/ShapeModel.h>
#include <QiVision/Transform/AffineTransform.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Filter/Filter.h>
#include "../src/Matching/DiagnosticFlags.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;
using namespace Qi::Vision::IO;
using namespace Qi::Vision::Transform;
using namespace Qi::Vision::Internal;

// Ground truth entry
struct GroundTruth {
    double row, col, angle, scale;
};

// Per-run metrics
struct RunMetrics {
    int tp = 0, fp = 0, fn = 0;
    double totalScaleErr = 0.0;
    int top1Correct = 0;
    int numCases = 0;
};

static bool IsMatch(double predRow, double predCol, double predScale,
                    const GroundTruth& gt, double posTol, double scaleTol) {
    double dr = predRow - gt.row;
    double dc = predCol - gt.col;
    double dist = std::sqrt(dr * dr + dc * dc);
    double scaleErr = std::abs(predScale - gt.scale);
    return dist < posTol && scaleErr < scaleTol;
}

static RunMetrics EvaluateConfig(const std::string& name,
                                  ShapeModel& model,
                                  const QImage& templateImg,
                                  const Rect2i& roi) {
    RunMetrics metrics;

    // Test scales: 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3
    double testScales[] = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};

    QImage templateRegion = templateImg.SubImage(roi.x, roi.y, roi.width, roi.height);

    for (double targetScale : testScales) {
        metrics.numCases++;

        int scaledW = static_cast<int>(roi.width * targetScale);
        int scaledH = static_cast<int>(roi.height * targetScale);

        QImage scaledTemplate;
        ScaleImage(templateRegion, scaledTemplate, targetScale, targetScale, "bilinear");

        QImage searchImg(templateImg.Width(), templateImg.Height(), PixelType::UInt8);
        uint8_t* data = static_cast<uint8_t*>(searchImg.Data());
        std::memset(data, 128, searchImg.Height() * searchImg.Stride());

        int placeX = (searchImg.Width() - scaledW) / 2;
        int placeY = (searchImg.Height() - scaledH) / 2;

        for (int y = 0; y < scaledH && placeY + y < searchImg.Height(); ++y) {
            const uint8_t* srcRow = static_cast<const uint8_t*>(scaledTemplate.RowPtr(y));
            uint8_t* dstRow = static_cast<uint8_t*>(searchImg.RowPtr(placeY + y));
            for (int x = 0; x < scaledW && placeX + x < searchImg.Width(); ++x) {
                dstRow[placeX + x] = srcRow[x];
            }
        }

        // Ground truth: center of placed region, angle 0, target scale
        double gtCol = placeX + scaledW / 2.0;
        double gtRow = placeY + scaledH / 2.0;
        GroundTruth gt{gtRow, gtCol, 0.0, targetScale};

        std::vector<double> rows, cols, angles, scales, scores;
        FindScaledShapeModel(
            searchImg, model,
            0, RAD(360),
            0.5, 1.5,
            0.3, 5, 0.5,   // minScore=0.3, numMatches=5, maxOverlap=0.5
            "least_squares", 0, 0, 0.7,
            rows, cols, angles, scales, scores
        );

        // Check top-1
        bool top1Hit = false;
        if (!rows.empty()) {
            top1Hit = IsMatch(rows[0], cols[0], scales[0], gt, 10.0, 0.15);
        }
        if (top1Hit) metrics.top1Correct++;

        // Count TP/FP among all results
        bool gtMatched = false;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (IsMatch(rows[i], cols[i], scales[i], gt, 10.0, 0.15)) {
                if (!gtMatched) {
                    metrics.tp++;
                    gtMatched = true;
                    metrics.totalScaleErr += std::abs(scales[i] - targetScale);
                }
                // additional matches to same GT are duplicates, not FP
            } else {
                metrics.fp++;
            }
        }
        if (!gtMatched) {
            metrics.fn++;
        }

        // Print per-case result
        const char* status = gtMatched ? (top1Hit ? "OK" : "ok(not top1)") : "MISS";
        std::printf("  scale=%.2f: %s", targetScale, status);
        if (!rows.empty()) {
            std::printf(" [%zu hits, top1: pos=(%.1f,%.1f) scale=%.3f score=%.3f]",
                        rows.size(), cols[0], rows[0], scales[0], scores[0]);
        }
        std::printf("\n");
    }

    return metrics;
}

int main() {
    std::printf("=== Scale Matching Diagnostic Matrix ===\n\n");

    // Load template
    QImage templateImg;
    ReadImageGray("tests/data/halcon_images/rings/mixed_01.png", templateImg);
    if (templateImg.Empty()) {
        // Fallback to m1.bmp
        ReadImageGray("tests/data/imgs/m1.bmp", templateImg);
    }
    if (templateImg.Empty()) {
        std::printf("Failed to load image!\n");
        return 1;
    }
    std::printf("Template: %dx%d\n\n", templateImg.Width(), templateImg.Height());

    // Define ROI -- use full image if m1.bmp, or specific ROI if rings
    Rect2i roi;
    bool isRings = (templateImg.Width() == 640 && templateImg.Height() == 480);
    if (isRings) {
        roi = {367, 213, 89, 87};
    } else {
        roi = {0, 0, templateImg.Width(), templateImg.Height()};
    }

    // Create model
    ShapeModel model;
    QImage templateSmooth;
    Filter::GaussFilter(templateImg, templateSmooth, 0.7);

    QRegion roiRegion = QRegion::Rectangle(roi.x, roi.y, roi.width, roi.height);
    CreateScaledShapeModel(
        templateSmooth, roiRegion, model,
        3,
        0, RAD(360), RAD(5),
        0.5, 1.5, 0.1,
        "point_reduction_high",
        "use_polarity",
        "auto_contrast_hyst", 10.0
    );

    if (!model.IsValid()) {
        std::printf("Failed to create model!\n");
        return 1;
    }
    std::printf("Model created.\n\n");

    // ==================== v2: upstream diagnostic matrix ====================
    struct RunConfig {
        const char* name;
        bool disableAngleSup;   // B1
        bool disableSpatialNMS; // B2
        bool clampScale;        // B4
    };
    RunConfig configs[] = {
        {"B0: baseline",                     false, false, false},
        {"B1: no_angle_suppress",            true,  false, false},
        {"B2: no_spatial_NMS",               false, true,  false},
        {"B3: no_angle+no_spatial",          true,  true,  false},
        {"B4: clamp_refine_scale",           false, false, true },
    };

    std::printf("%-35s  Recall  Precis  MeanSE  Top1%%\n", "Config");
    std::printf("%-35s  ------  ------  ------  -----\n", "------");

    for (const auto& cfg : configs) {
        std::printf("\n--- %s ---\n", cfg.name);

        // Reset all flags
        g_scaleDiag = ScaleDiagFlags{};
        g_scaleDiag.disableCoarseAngleSuppress = cfg.disableAngleSup;
        g_scaleDiag.disableCoarseSpatialNMS = cfg.disableSpatialNMS;
        g_scaleDiag.clampRefineScale = cfg.clampScale;

        RunMetrics m = EvaluateConfig(cfg.name, model, templateImg, roi);

        double recall = (m.tp + m.fn > 0) ? 100.0 * m.tp / (m.tp + m.fn) : 0.0;
        double precision = (m.tp + m.fp > 0) ? 100.0 * m.tp / (m.tp + m.fp) : 0.0;
        double meanScaleErr = (m.tp > 0) ? m.totalScaleErr / m.tp : -1.0;
        double top1Pct = (m.numCases > 0) ? 100.0 * m.top1Correct / m.numCases : 0.0;

        std::printf("  >> TP=%d FP=%d FN=%d | Recall=%.1f%% Precision=%.1f%% MeanScaleErr=%.4f Top1=%.1f%%\n",
                    m.tp, m.fp, m.fn, recall, precision, meanScaleErr, top1Pct);
    }

    // Reset
    g_scaleDiag = ScaleDiagFlags{};

    std::printf("\n=== Diagnostic Complete ===\n");
    return 0;
}
