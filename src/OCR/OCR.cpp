/**
 * @file OCR.cpp
 * @brief OCR implementation using ONNXRuntime (PaddleOCR models)
 *
 * No OpenCV dependency - uses QiVision's own image processing APIs.
 * Models: PaddleOCR v4 converted to ONNX format
 */

#include <QiVision/OCR/OCR.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>
#include <QiVision/IO/ImageIO.h>
#include <QiVision/Filter/Filter.h>
#include <QiVision/Color/ColorConvert.h>
#include <QiVision/Segment/Segment.h>
#include <QiVision/Blob/Blob.h>
#include <QiVision/Morphology/Morphology.h>
#include <QiVision/Platform/Timer.h>
#include <QiVision/Platform/FileIO.h>
#include <QiVision/Internal/ContourConvert.h>
#include <QiVision/Internal/ContourAnalysis.h>
#include <QiVision/Internal/Geometry2d.h>
#include <QiVision/Internal/Homography.h>
#include <QiVision/Internal/AffineTransform.h>

#ifdef QIVISION_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <mutex>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdlib>

namespace Qi::Vision::OCR {

namespace {

// OCR requires UInt8 image (gray, RGB, or RGBA)
inline bool RequireOCRImage(const QImage& image, const char* funcName) {
    return Validate::RequireImageU8Channels(image, funcName, true, true, true);
}

inline void ValidateOCRParams(const OCRParams& params, const char* funcName) {
    Validate::RequireNonNegative(params.padding, "padding", funcName);
    Validate::RequirePositive(params.maxSideLen, "maxSideLen", funcName);
    Validate::RequireRange(params.boxScoreThresh, 0.0, 1.0, "boxScoreThresh", funcName);
    Validate::RequireRange(params.boxThresh, 0.0, 1.0, "boxThresh", funcName);
    Validate::RequirePositive(params.unClipRatio, "unClipRatio", funcName);
    Validate::RequireThreadCount(params.numThread, funcName);
    Validate::RequireGpuIndex(params.gpuIndex, funcName);
}

} // namespace

// =============================================================================
// TextBlock Implementation
// =============================================================================

Rect2i TextBlock::BoundingRect() const {
    if (corners.size() < 4) return Rect2i();

    double minX = corners[0].x, maxX = corners[0].x;
    double minY = corners[0].y, maxY = corners[0].y;

    for (const auto& pt : corners) {
        minX = std::min(minX, pt.x);
        maxX = std::max(maxX, pt.x);
        minY = std::min(minY, pt.y);
        maxY = std::max(maxY, pt.y);
    }

    return Rect2i(static_cast<int>(minX), static_cast<int>(minY),
                  static_cast<int>(maxX - minX + 1),
                  static_cast<int>(maxY - minY + 1));
}

Point2d TextBlock::Center() const {
    if (corners.size() < 4) return Point2d();

    double cx = 0, cy = 0;
    for (const auto& pt : corners) {
        cx += pt.x;
        cy += pt.y;
    }
    return Point2d(cx / corners.size(), cy / corners.size());
}

// =============================================================================
// OCRModel Implementation
// =============================================================================

#ifdef QIVISION_HAS_ONNXRUNTIME

class OCRModel::Impl {
public:
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "QiVisionOCR"};
    Ort::SessionOptions sessionOptions;

    std::unique_ptr<Ort::Session> detSession;
    std::unique_ptr<Ort::Session> clsSession;
    std::unique_ptr<Ort::Session> recSession;

    std::vector<std::string> keys;  // Character dictionary
    bool initialized = false;
    int numThread = 4;
    int gpuIndex = -1;  // -1 = CPU, >= 0 = GPU index
    bool useGpu = false;

    // Model input/output names
    std::vector<const char*> detInputNames = {"x"};
    std::vector<const char*> detOutputNames = {"sigmoid_0.tmp_0"};
    std::vector<const char*> clsInputNames = {"x"};
    std::vector<const char*> clsOutputNames = {"softmax_0.tmp_0"};
    std::vector<const char*> recInputNames = {"x"};
    std::vector<const char*> recOutputNames = {"softmax_11.tmp_0"};

    bool Init(const std::string& modelDir,
              const std::string& detModel,
              const std::string& clsModel,
              const std::string& recModel,
              const std::string& keysFile) {
        try {
            sessionOptions.SetIntraOpNumThreads(numThread);
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // Enable CUDA if requested
            if (useGpu && gpuIndex >= 0) {
                OrtCUDAProviderOptions cudaOptions;
                cudaOptions.device_id = gpuIndex;
                sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            }

            std::string detPath = modelDir + "/" + detModel;
            std::string clsPath = modelDir + "/" + clsModel;
            std::string recPath = modelDir + "/" + recModel;
            std::string keysPath = modelDir + "/" + keysFile;

            // Load models
#ifdef _WIN32
            std::wstring wDetPath(detPath.begin(), detPath.end());
            std::wstring wRecPath(recPath.begin(), recPath.end());
            detSession = std::make_unique<Ort::Session>(env, wDetPath.c_str(), sessionOptions);
            recSession = std::make_unique<Ort::Session>(env, wRecPath.c_str(), sessionOptions);
            // cls model is optional
            try {
                std::wstring wClsPath(clsPath.begin(), clsPath.end());
                clsSession = std::make_unique<Ort::Session>(env, wClsPath.c_str(), sessionOptions);
            } catch (...) {
                clsSession = nullptr;  // cls is optional
            }
#else
            detSession = std::make_unique<Ort::Session>(env, detPath.c_str(), sessionOptions);
            recSession = std::make_unique<Ort::Session>(env, recPath.c_str(), sessionOptions);
            // cls model is optional
            try {
                clsSession = std::make_unique<Ort::Session>(env, clsPath.c_str(), sessionOptions);
            } catch (...) {
                clsSession = nullptr;  // cls is optional
            }
#endif

            // Load character dictionary
            if (!LoadKeys(keysPath)) {
                return false;
            }

            initialized = true;
            return true;
        } catch (const Ort::Exception& e) {
            return false;
        }
    }

    bool LoadKeys(const std::string& keysPath) {
        std::ifstream file(keysPath);
        if (!file.is_open()) return false;

        keys.clear();
        keys.push_back("");  // Index 0 is blank for CTC

        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                keys.push_back(line);
            }
        }
        return !keys.empty();
    }

    // Preprocess image for detection model
    std::vector<float> PreprocessForDetection(const QImage& image, int targetWidth, int targetHeight) {
        // Resize image to target size using nearest neighbor
        QImage resized;
        int channels = image.Channels();

        if (image.Width() != targetWidth || image.Height() != targetHeight) {
            ChannelType chType = (channels == 1) ? ChannelType::Gray :
                                 (channels == 3) ? ChannelType::RGB : ChannelType::RGBA;
            resized = QImage(targetWidth, targetHeight, PixelType::UInt8, chType);

            float scaleX = static_cast<float>(image.Width()) / targetWidth;
            float scaleY = static_cast<float>(image.Height()) / targetHeight;

            for (int y = 0; y < targetHeight; ++y) {
                uint8_t* dstRow = static_cast<uint8_t*>(resized.RowPtr(y));
                int srcY = std::min(static_cast<int>(y * scaleY), image.Height() - 1);
                const uint8_t* srcRow = static_cast<const uint8_t*>(image.RowPtr(srcY));

                for (int x = 0; x < targetWidth; ++x) {
                    int srcX = std::min(static_cast<int>(x * scaleX), image.Width() - 1);

                    for (int c = 0; c < channels; ++c) {
                        dstRow[x * channels + c] = srcRow[srcX * channels + c];
                    }
                }
            }
        } else {
            resized = image;
        }

        // Convert to float and normalize
        // PaddleOCR uses: (img / 255.0 - 0.5) / 0.5 = img / 127.5 - 1.0
        std::vector<float> input(3 * targetHeight * targetWidth);

        int resizedChannels = resized.Channels();

        for (int y = 0; y < targetHeight; ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(resized.RowPtr(y));

            for (int x = 0; x < targetWidth; ++x) {
                for (int c = 0; c < 3; ++c) {
                    float pixel;
                    if (resizedChannels == 1) {
                        pixel = row[x];
                    } else {
                        // PaddleOCR uses RGB (same as QiVision)
                        pixel = row[x * resizedChannels + c];
                    }
                    // CHW format: channel first
                    // Normalize: (pixel / 255.0 - 0.5) / 0.5 = pixel / 127.5 - 1.0
                    input[c * targetHeight * targetWidth + y * targetWidth + x] =
                        pixel / 127.5f - 1.0f;
                }
            }
        }

        return input;
    }

    // Compute box score: mean probability within the region (not bounding box!)
    double ComputeBoxScore(const float* probData, int probWidth, int probHeight,
                           const QRegion& region) {
        if (region.Empty()) return 0.0;

        double sum = 0.0;
        int count = 0;

        // Sample probability values inside region runs (precise, no background dilution)
        const auto& runs = region.Runs();
        for (const auto& run : runs) {
            int y = run.row;
            if (y < 0 || y >= probHeight) continue;
            int x1 = std::max(0, run.colBegin);
            int x2 = std::min(probWidth - 1, run.colEnd - 1);  // colEnd is exclusive
            for (int x = x1; x <= x2; ++x) {
                sum += probData[y * probWidth + x];
                count++;
            }
        }

        return count > 0 ? sum / count : 0.0;
    }

    // Unclip polygon: expand by ratio based on perimeter/area
    std::vector<Point2d> UnclipPolygon(const std::vector<Point2d>& polygon, double unclipRatio) {
        if (polygon.size() < 3) return polygon;

        // Compute perimeter and area
        double perimeter = 0.0;
        double area = 0.0;
        size_t n = polygon.size();

        for (size_t i = 0; i < n; ++i) {
            size_t j = (i + 1) % n;
            double dx = polygon[j].x - polygon[i].x;
            double dy = polygon[j].y - polygon[i].y;
            perimeter += std::sqrt(dx * dx + dy * dy);
            area += polygon[i].x * polygon[j].y - polygon[j].x * polygon[i].y;
        }
        area = std::abs(area) * 0.5;

        if (perimeter < 1e-6 || area < 1e-6) return polygon;

        // Expand distance
        double distance = area * unclipRatio / perimeter;

        // Expand each vertex outward
        std::vector<Point2d> expanded;
        expanded.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            size_t prev = (i + n - 1) % n;
            size_t next = (i + 1) % n;

            // Compute edge normals (outward for clockwise polygon)
            double dx1 = polygon[i].x - polygon[prev].x;
            double dy1 = polygon[i].y - polygon[prev].y;
            double len1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
            if (len1 < 1e-6) { expanded.push_back(polygon[i]); continue; }
            double nx1 = dy1 / len1, ny1 = -dx1 / len1;  // (dy, -dx) for outward normal

            double dx2 = polygon[next].x - polygon[i].x;
            double dy2 = polygon[next].y - polygon[i].y;
            double len2 = std::sqrt(dx2 * dx2 + dy2 * dy2);
            if (len2 < 1e-6) { expanded.push_back(polygon[i]); continue; }
            double nx2 = dy2 / len2, ny2 = -dx2 / len2;  // (dy, -dx) for outward normal

            // Average normal direction
            double nx = (nx1 + nx2) * 0.5;
            double ny = (ny1 + ny2) * 0.5;
            double nlen = std::sqrt(nx * nx + ny * ny);
            if (nlen < 1e-6) { expanded.push_back(polygon[i]); continue; }
            nx /= nlen;
            ny /= nlen;

            expanded.push_back(Point2d(
                polygon[i].x + nx * distance,
                polygon[i].y + ny * distance
            ));
        }

        return expanded;
    }

    // Run text detection with improved DB post-processing
    std::vector<std::vector<Point2d>> DetectTextBoxes(const QImage& image, const OCRParams& params) {
        std::vector<std::vector<Point2d>> boxes;

        // Calculate target size (multiple of 32)
        int targetWidth = std::min(image.Width(), params.maxSideLen);
        int targetHeight = std::min(image.Height(), params.maxSideLen);

        float ratio = std::min(
            static_cast<float>(params.maxSideLen) / image.Width(),
            static_cast<float>(params.maxSideLen) / image.Height()
        );
        if (ratio < 1.0f) {
            targetWidth = static_cast<int>(image.Width() * ratio);
            targetHeight = static_cast<int>(image.Height() * ratio);
        }

        // Round to multiple of 32
        targetWidth = ((targetWidth + 31) / 32) * 32;
        targetHeight = ((targetHeight + 31) / 32) * 32;

        // Preprocess
        auto input = PreprocessForDetection(image, targetWidth, targetHeight);

        // Create input tensor
        std::vector<int64_t> inputShape = {1, 3, targetHeight, targetWidth};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, input.data(), input.size(), inputShape.data(), inputShape.size());

        // Run detection
        auto outputTensors = detSession->Run(Ort::RunOptions{nullptr},
            detInputNames.data(), &inputTensor, 1,
            detOutputNames.data(), 1);

        // Process detection output (probability map)
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        int outHeight = static_cast<int>(outputShape[2]);
        int outWidth = static_cast<int>(outputShape[3]);

        float scaleX = static_cast<float>(image.Width()) / outWidth;
        float scaleY = static_cast<float>(image.Height()) / outHeight;

        // Check if output needs sigmoid (if values are outside [0,1] range)
        float minP = outputData[0], maxP = outputData[0];
        for (int i = 1; i < outWidth * outHeight; ++i) {
            minP = std::min(minP, outputData[i]);
            maxP = std::max(maxP, outputData[i]);
        }

        // Apply sigmoid if output appears to be logits (not probabilities)
        if (minP < -0.1f || maxP > 1.1f) {
            for (int i = 0; i < outWidth * outHeight; ++i) {
                outputData[i] = 1.0f / (1.0f + std::exp(-outputData[i]));
            }
        }

        // Create binary mask from probability map
        QImage probMap(outWidth, outHeight, PixelType::UInt8, ChannelType::Gray);
        for (int y = 0; y < outHeight; ++y) {
            uint8_t* row = static_cast<uint8_t*>(probMap.RowPtr(y));
            for (int x = 0; x < outWidth; ++x) {
                float prob = outputData[y * outWidth + x];
                row[x] = (prob > params.boxThresh) ? 255 : 0;
            }
        }

        // Dilate to expand and connect text regions (critical for DB)
        QImage dilated;
        Morphology::GrayDilationRectangle(probMap, dilated, 3, 3);

        // Find connected regions
        QRegion binaryRegion = Segment::ThresholdToRegion(dilated, 128.0, 255.0);

        std::vector<QRegion> regions;
        Blob::Connection(binaryRegion, regions);

        // Minimum area threshold
        constexpr int minArea = 50;
        constexpr double minShortEdge = 3.0;

        for (const auto& region : regions) {
            if (region.Area() < minArea) continue;

            // Convert region to contour for shape analysis
            QContour contour = Internal::RegionToContour(region);
            if (contour.Size() < 4) continue;

            // Compute box score (mean probability in region - precise, no background dilution)
            double boxScore = ComputeBoxScore(outputData, outWidth, outHeight, region);
            if (boxScore < params.boxScoreThresh) continue;

            // Get minimum area rectangle
            auto minRectOpt = Internal::ContourMinAreaRect(contour);
            if (!minRectOpt) continue;

            RotatedRect2d minRect = *minRectOpt;

            // Filter small boxes
            double shortEdge = std::min(minRect.width, minRect.height);
            if (shortEdge < minShortEdge) continue;

            // Get corners and apply unclip expansion
            auto corners = Internal::RotatedRectCorners(minRect);
            std::vector<Point2d> polygon(corners.begin(), corners.end());
            polygon = UnclipPolygon(polygon, params.unClipRatio);

            // Scale corners back to original image coordinates
            std::vector<Point2d> scaledCorners;
            scaledCorners.reserve(4);
            for (const auto& pt : polygon) {
                double x = std::clamp(pt.x * scaleX, 0.0, static_cast<double>(image.Width() - 1));
                double y = std::clamp(pt.y * scaleY, 0.0, static_cast<double>(image.Height() - 1));
                scaledCorners.push_back(Point2d(x, y));
            }

            if (scaledCorners.size() >= 4) {
                boxes.push_back(std::move(scaledCorners));
            }
        }

        return boxes;
    }

    // Recognize text in a cropped region
    std::string RecognizeText(const QImage& cropImage, double& confidence) {
        // Resize to recognition model input size (height=48, variable width)
        int targetHeight = 48;
        int targetWidth = static_cast<int>(cropImage.Width() * targetHeight / static_cast<float>(cropImage.Height()));
        targetWidth = std::max(targetWidth, 10);

        auto input = PreprocessForDetection(cropImage, targetWidth, targetHeight);

        // Create input tensor
        std::vector<int64_t> inputShape = {1, 3, targetHeight, targetWidth};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, input.data(), input.size(), inputShape.data(), inputShape.size());

        // Run recognition
        auto outputTensors = recSession->Run(Ort::RunOptions{nullptr},
            recInputNames.data(), &inputTensor, 1,
            recOutputNames.data(), 1);

        // CTC decode
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        int seqLen = static_cast<int>(outputShape[1]);
        int numClasses = static_cast<int>(outputShape[2]);

        std::string text;
        std::vector<double> scores;
        int lastIdx = 0;

        for (int t = 0; t < seqLen; ++t) {
            // Find max probability class
            int maxIdx = 0;
            float maxProb = outputData[t * numClasses];
            for (int c = 1; c < numClasses; ++c) {
                if (outputData[t * numClasses + c] > maxProb) {
                    maxProb = outputData[t * numClasses + c];
                    maxIdx = c;
                }
            }

            // CTC: skip blank (0) and repeated characters
            if (maxIdx != 0 && maxIdx != lastIdx) {
                if (maxIdx < static_cast<int>(keys.size())) {
                    text += keys[maxIdx];
                    scores.push_back(maxProb);
                }
            }
            lastIdx = maxIdx;
        }

        confidence = scores.empty() ? 0.0 :
            std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();

        return text;
    }

    // Directly recognize a single line (skip detection)
    std::string RecognizeLineImpl(const QImage& image, double& confidence) {
        if (!initialized) {
            confidence = 0.0;
            return "";
        }

        // Ensure RGB image
        QImage rgbImage;
        if (image.Channels() == 1) {
            Color::GrayToRgb(image, rgbImage);
        } else {
            rgbImage = image;
        }

        return RecognizeText(rgbImage, confidence);
    }

    OCRResult Recognize(const QImage& image, const OCRParams& params) {
        OCRResult result;

        if (!initialized) {
            return result;
        }

        Platform::Timer timer;
        timer.Start();

        // Ensure RGB image
        QImage rgbImage;
        if (image.Channels() == 1) {
            Color::GrayToRgb(image, rgbImage);
        } else {
            rgbImage = image;
        }

        // Step 1: Detect text boxes
        auto boxes = DetectTextBoxes(rgbImage, params);
        result.detectTime = timer.ElapsedMs();

        // Step 2: Recognize text in each box with perspective correction
        timer.Start();
        for (const auto& corners : boxes) {
            if (corners.size() < 4) continue;

            TextBlock tb;
            tb.corners = corners;

            // Compute target width and height from corner distances
            // corners: [top-left, top-right, bottom-right, bottom-left]
            double topWidth = std::sqrt(
                std::pow(corners[1].x - corners[0].x, 2) +
                std::pow(corners[1].y - corners[0].y, 2));
            double bottomWidth = std::sqrt(
                std::pow(corners[2].x - corners[3].x, 2) +
                std::pow(corners[2].y - corners[3].y, 2));
            double leftHeight = std::sqrt(
                std::pow(corners[3].x - corners[0].x, 2) +
                std::pow(corners[3].y - corners[0].y, 2));
            double rightHeight = std::sqrt(
                std::pow(corners[2].x - corners[1].x, 2) +
                std::pow(corners[2].y - corners[1].y, 2));

            int dstWidth = static_cast<int>(std::max(topWidth, bottomWidth));
            int dstHeight = static_cast<int>(std::max(leftHeight, rightHeight));

            if (dstWidth < 8 || dstHeight < 8) continue;

            // Check if box is significantly rotated (use perspective transform)
            // Otherwise use simple crop (faster)
            double angle = std::atan2(corners[1].y - corners[0].y, corners[1].x - corners[0].x);
            bool needPerspective = std::abs(angle) > 0.05;  // ~3 degrees

            QImage crop;

            if (needPerspective) {
                // Use perspective transform for rotated text
                std::array<Point2d, 4> srcPts = {corners[0], corners[1], corners[2], corners[3]};
                std::array<Point2d, 4> dstPts = {
                    Point2d(0, 0),
                    Point2d(dstWidth - 1, 0),
                    Point2d(dstWidth - 1, dstHeight - 1),
                    Point2d(0, dstHeight - 1)
                };

                auto H = Internal::Homography::From4Points(srcPts, dstPts);
                if (H) {
                    crop = Internal::WarpPerspective(rgbImage, *H, dstWidth, dstHeight,
                                                      Internal::InterpolationMethod::Bilinear,
                                                      Internal::BorderMode::Constant, 0.0);
                }
            }

            // Fallback to simple crop if perspective transform failed or not needed
            if (crop.Empty()) {
                auto rect = tb.BoundingRect();
                if (rect.width <= 0 || rect.height <= 0) continue;

                rect.x = std::max(0, rect.x);
                rect.y = std::max(0, rect.y);
                rect.width = std::min(rect.width, rgbImage.Width() - rect.x);
                rect.height = std::min(rect.height, rgbImage.Height() - rect.y);

                if (rect.width <= 0 || rect.height <= 0) continue;

                crop = rgbImage.SubImage(rect.x, rect.y, rect.width, rect.height).Clone();
            }

            if (crop.Empty()) continue;

            // Recognize
            tb.text = RecognizeText(crop, tb.confidence);
            tb.boxScore = 1.0;

            if (!tb.text.empty()) {
                result.textBlocks.push_back(std::move(tb));

                if (!result.fullText.empty()) {
                    result.fullText += "\n";
                }
                result.fullText += result.textBlocks.back().text;
            }
        }
        result.recognizeTime = timer.ElapsedMs();
        result.totalTime = result.detectTime + result.recognizeTime;

        return result;
    }
};

#else  // !QIVISION_HAS_ONNXRUNTIME

class OCRModel::Impl {
public:
    bool initialized = false;
    int numThread = 4;

    bool Init(const std::string&, const std::string&, const std::string&,
              const std::string&, const std::string&) {
        return false;
    }

    OCRResult Recognize(const QImage&, const OCRParams&) {
        return OCRResult();
    }

    std::string RecognizeLineImpl(const QImage&, double& confidence) {
        confidence = 0.0;
        return "";
    }
};

#endif  // QIVISION_HAS_ONNXRUNTIME

OCRModel::OCRModel() : impl_(std::make_unique<Impl>()) {}

OCRModel::~OCRModel() = default;

OCRModel::OCRModel(OCRModel&& other) noexcept = default;

OCRModel& OCRModel::operator=(OCRModel&& other) noexcept = default;

bool OCRModel::Init(const std::string& modelDir,
                    const std::string& detModel,
                    const std::string& clsModel,
                    const std::string& recModel,
                    const std::string& keysFile) {
    if (modelDir.empty()) {
        throw InvalidArgumentException("OCRModel::Init: modelDir is empty. "
            "Use GetDefaultModelDir() to get the default path, or set QIVISION_OCR_MODELS environment variable.");
    }
    if (detModel.empty() || recModel.empty() || keysFile.empty()) {
        throw InvalidArgumentException("OCRModel::Init: model filenames must not be empty");
    }

    // Check if directory exists
    if (!Platform::DirectoryExists(modelDir)) {
        std::string msg = "OCRModel::Init: Model directory not found: " + modelDir + "\n\n";
        msg += "To install OCR models, use one of these methods:\n";
        msg += "1. Call OCR::DownloadModels(\"" + modelDir + "\")\n";
        msg += "2. Download manually from: " + GetModelDownloadUrl() + "\n";
        msg += "3. Set QIVISION_OCR_MODELS environment variable\n\n";
        msg += "Required files: ";
        auto requiredFiles = GetRequiredModelFiles();
        for (size_t i = 0; i < requiredFiles.size(); ++i) {
            if (i > 0) msg += ", ";
            msg += requiredFiles[i];
        }
        throw IOException(msg);
    }

    // Check for required model files
    std::vector<std::string> filesToCheck = {detModel, recModel, keysFile};
    std::vector<std::string> missingFiles;

    for (const auto& file : filesToCheck) {
        std::string path = modelDir + "/" + file;
        if (!Platform::FileExists(path)) {
            missingFiles.push_back(file);
        }
    }

    if (!missingFiles.empty()) {
        std::string msg = "OCRModel::Init: Required model files not found in " + modelDir + ":\n";
        for (const auto& f : missingFiles) {
            msg += "  - " + f + "\n";
        }
        msg += "\nTo install, run: OCR::DownloadModels(\"" + modelDir + "\")\n";
        msg += "Or download from: " + GetModelDownloadUrl();
        throw IOException(msg);
    }

    bool success = impl_->Init(modelDir, detModel, clsModel, recModel, keysFile);

    if (!success) {
        std::string msg = "OCRModel::Init: Failed to load ONNX models.\n\n";
        msg += "Possible causes:\n";
        msg += "1. Model files may be corrupted - try re-downloading\n";
        msg += "2. ONNXRuntime version mismatch - ensure compatible version\n";
        msg += "3. Insufficient memory for model loading\n\n";
        msg += "Model directory: " + modelDir;
        throw IOException(msg);
    }

    return true;
}

bool OCRModel::InitDefault() {
    std::string modelDir = GetDefaultModelDir();

    // Check models before attempting to init
    ModelStatus status = CheckModels(modelDir);
    if (!status.IsReady()) {
        std::string msg = "OCRModel::InitDefault: " + status.GetMessage();
        throw IOException(msg);
    }

    return Init(modelDir);
}

bool OCRModel::IsValid() const {
    return impl_->initialized;
}

void OCRModel::SetNumThread(int numThread) {
    if (numThread <= 0) {
        throw InvalidArgumentException("OCRModel::SetNumThread: numThread must be >= 1");
    }
#ifdef QIVISION_HAS_ONNXRUNTIME
    impl_->numThread = numThread;
    impl_->sessionOptions.SetIntraOpNumThreads(numThread);
#else
    (void)numThread;
#endif
}

void OCRModel::SetGpuIndex(int gpuIndex) {
    if (gpuIndex < -1) {
        throw InvalidArgumentException("OCRModel::SetGpuIndex: gpuIndex must be >= -1");
    }
#ifdef QIVISION_HAS_ONNXRUNTIME
    impl_->gpuIndex = gpuIndex;
    impl_->useGpu = (gpuIndex >= 0);
#else
    (void)gpuIndex;
#endif
}

OCRResult OCRModel::Recognize(const QImage& image, const OCRParams& params) const {
    if (!IsValid()) {
        throw InvalidArgumentException("OCRModel::Recognize: Model not initialized");
    }
    ValidateOCRParams(params, "OCRModel::Recognize");
    if (!RequireOCRImage(image, "OCRModel::Recognize")) {
        return OCRResult();
    }
    return impl_->Recognize(image, params);
}

OCRResult OCRModel::Recognize(const std::string& imagePath, const OCRParams& params) const {
    if (!IsValid()) {
        throw InvalidArgumentException("OCRModel::Recognize: Model not initialized");
    }
    ValidateOCRParams(params, "OCRModel::Recognize");

    QImage image;
    IO::ReadImage(imagePath, image);
    if (!RequireOCRImage(image, "OCRModel::Recognize")) {
        return OCRResult();
    }
    return impl_->Recognize(image, params);
}

std::string OCRModel::RecognizeLine(const QImage& image, double& confidence) const {
    if (!IsValid()) {
        throw InvalidArgumentException("OCRModel::RecognizeLine: Model not initialized");
    }
    if (!RequireOCRImage(image, "OCRModel::RecognizeLine")) {
        confidence = 0.0;
        return "";
    }
    return impl_->RecognizeLineImpl(image, confidence);
}

// =============================================================================
// Global Model
// =============================================================================

namespace {
    std::unique_ptr<OCRModel> g_ocrModel;
    std::mutex g_ocrMutex;
}

bool InitOCR(const std::string& modelDir, int gpuIndex) {
    if (modelDir.empty()) {
        std::string defaultDir = GetDefaultModelDir();
        throw InvalidArgumentException(
            "InitOCR: modelDir is empty.\n"
            "Use InitOCRDefault() for automatic model discovery, or specify a path.\n"
            "Default model directory: " + defaultDir);
    }

    // Check models before initializing
    ModelStatus status = CheckModels(modelDir);
    if (!status.IsReady()) {
        throw IOException("InitOCR: " + status.GetMessage());
    }

    std::lock_guard<std::mutex> lock(g_ocrMutex);
    g_ocrModel = std::make_unique<OCRModel>();
    if (gpuIndex >= 0) {
        g_ocrModel->SetGpuIndex(gpuIndex);
    }
    return g_ocrModel->Init(modelDir);
}

bool InitOCRDefault(int gpuIndex) {
    std::string modelDir = GetDefaultModelDir();

    // Check models before initializing
    ModelStatus status = CheckModels(modelDir);
    if (!status.IsReady()) {
        throw IOException("InitOCRDefault: " + status.GetMessage());
    }

    std::lock_guard<std::mutex> lock(g_ocrMutex);
    g_ocrModel = std::make_unique<OCRModel>();
    if (gpuIndex >= 0) {
        g_ocrModel->SetGpuIndex(gpuIndex);
    }
    return g_ocrModel->Init(modelDir);
}

void ReleaseOCR() {
    std::lock_guard<std::mutex> lock(g_ocrMutex);
    g_ocrModel.reset();
}

bool IsOCRReady() {
    std::lock_guard<std::mutex> lock(g_ocrMutex);
    return g_ocrModel && g_ocrModel->IsValid();
}

OCRResult RecognizeText(const QImage& image, const OCRParams& params) {
    std::lock_guard<std::mutex> lock(g_ocrMutex);
    if (!g_ocrModel || !g_ocrModel->IsValid()) {
        throw InvalidArgumentException("RecognizeText: OCR not initialized. Call InitOCR() first.");
    }
    return g_ocrModel->Recognize(image, params);
}

OCRResult RecognizeText(const std::string& imagePath, const OCRParams& params) {
    std::lock_guard<std::mutex> lock(g_ocrMutex);
    if (!g_ocrModel || !g_ocrModel->IsValid()) {
        throw InvalidArgumentException("RecognizeText: OCR not initialized. Call InitOCR() first.");
    }
    return g_ocrModel->Recognize(imagePath, params);
}

std::string ReadText(const QImage& image) {
    try {
        auto result = RecognizeText(image);
        return result.fullText;
    } catch (...) {
        return "";
    }
}

std::string ReadText(const std::string& imagePath) {
    try {
        auto result = RecognizeText(imagePath);
        return result.fullText;
    } catch (...) {
        return "";
    }
}

std::string RecognizeLine(const QImage& image, double& confidence) {
    std::lock_guard<std::mutex> lock(g_ocrMutex);
    if (!g_ocrModel || !g_ocrModel->IsValid()) {
        throw InvalidArgumentException("RecognizeLine: OCR not initialized. Call InitOCR() first.");
    }
    return g_ocrModel->RecognizeLine(image, confidence);
}

// =============================================================================
// Model Management
// =============================================================================

namespace {

// Required model files (must be present)
const std::vector<std::string> g_requiredModelFiles = {
    "ch_PP-OCRv4_det_infer.onnx",   // Text detection model
    "ch_PP-OCRv4_rec_infer.onnx",   // Text recognition model
    "ppocr_keys_v1.txt"             // Character dictionary
};

// Optional model files (enhance functionality)
const std::vector<std::string> g_optionalModelFiles = {
    "ch_ppocr_mobile_v2.0_cls_infer.onnx"  // Angle classification (optional)
};

// Model download base URL
const std::string g_modelDownloadUrl =
    "https://github.com/PaddlePaddle/PaddleOCR/releases/download/v2.8.0/";

// Alternative download URLs
const std::vector<std::string> g_alternativeUrls = {
    "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/",
    "https://huggingface.co/QiVision/ocr-models/resolve/main/"
};

// Validate path for shell command safety (prevent command injection)
// Returns true if path is safe to use in shell commands
bool IsPathSafeForShell(const std::string& path) {
    // Reject empty paths
    if (path.empty()) return false;

    // Reject paths with shell metacharacters
    const std::string dangerous = "`$;|&<>(){}[]!\\'\"\n\r\t";
    for (char c : path) {
        if (dangerous.find(c) != std::string::npos) {
            return false;
        }
    }

    // Reject paths starting with -
    if (path[0] == '-') return false;

    return true;
}

// Validate URL for safety
bool IsUrlSafe(const std::string& url) {
    // Must start with https://
    if (url.find("https://") != 0) return false;

    // No shell metacharacters
    return IsPathSafeForShell(url);
}

} // anonymous namespace

std::string ModelStatus::GetMessage() const {
    std::string msg;

    if (allRequired) {
        msg = "OCR models ready";
        if (!allOptional) {
            msg += " (optional models missing: angle classification)";
        }
    } else {
        msg = "OCR models not found\n";
        msg += "Directory: " + modelDir + "\n";
        msg += "Missing required files:\n";
        for (const auto& f : missing) {
            msg += "  - " + f + "\n";
        }
        msg += "\nTo install OCR models:\n";
        msg += "1. Download from: " + GetModelDownloadUrl() + "\n";
        msg += "2. Place files in: " + modelDir + "\n";
        msg += "\nOr run: OCR::DownloadModels(\"" + modelDir + "\")";
    }

    return msg;
}

std::vector<std::string> GetRequiredModelFiles() {
    return g_requiredModelFiles;
}

std::vector<std::string> GetOptionalModelFiles() {
    return g_optionalModelFiles;
}

std::string GetModelDownloadUrl() {
    return g_modelDownloadUrl;
}

ModelStatus CheckModels(const std::string& modelDir) {
    ModelStatus status;
    status.modelDir = modelDir.empty() ? GetDefaultModelDir() : modelDir;

    // Check required files
    for (const auto& file : g_requiredModelFiles) {
        std::string path = status.modelDir + "/" + file;
        if (Platform::FileExists(path)) {
            status.found.push_back(file);
        } else {
            status.missing.push_back(file);
        }
    }

    // Check optional files
    bool allOptionalFound = true;
    for (const auto& file : g_optionalModelFiles) {
        std::string path = status.modelDir + "/" + file;
        if (Platform::FileExists(path)) {
            status.found.push_back(file);
        } else {
            allOptionalFound = false;
        }
    }

    status.allRequired = status.missing.empty();
    status.allOptional = allOptionalFound;

    return status;
}

bool DownloadModels(const std::string& modelDir, bool verbose) {
    std::string targetDir = modelDir.empty() ? GetDefaultModelDir() : modelDir;

    // Create directory if needed
    if (!Platform::DirectoryExists(targetDir)) {
        if (!Platform::CreateDirectory(targetDir)) {
            if (verbose) {
                std::cerr << "Error: Cannot create directory: " << targetDir << "\n";
            }
            return false;
        }
    }

    // Check which files are missing
    ModelStatus status = CheckModels(targetDir);
    if (status.allRequired) {
        if (verbose) {
            std::cout << "All required OCR models already present in: " << targetDir << "\n";
        }
        return true;
    }

    if (verbose) {
        std::cout << "Downloading OCR models to: " << targetDir << "\n";
        std::cout << "Missing files: " << status.missing.size() << "\n";
    }

    // Try to download using curl or wget
    bool hasCurl = false;
    bool hasWget = false;

#ifndef _WIN32
    hasCurl = (std::system("which curl > /dev/null 2>&1") == 0);
    hasWget = (std::system("which wget > /dev/null 2>&1") == 0);
#else
    hasCurl = (std::system("where curl > nul 2>&1") == 0);
    hasWget = (std::system("where wget > nul 2>&1") == 0);
#endif

    if (!hasCurl && !hasWget) {
        if (verbose) {
            std::cerr << "\nError: Neither curl nor wget found on this system.\n";
            std::cerr << "Please download models manually:\n\n";
            PrintModelInstallInstructions();
        }
        return false;
    }

    // Download each missing file
    bool allSuccess = true;
    for (const auto& file : status.missing) {
        std::string url = g_modelDownloadUrl + file;
        std::string destPath = targetDir + "/" + file;

        // Security: validate paths before shell execution
        if (!IsPathSafeForShell(destPath) || !IsUrlSafe(url)) {
            if (verbose) {
                std::cerr << "Error: Invalid path or URL detected, skipping: " << file << "\n";
            }
            allSuccess = false;
            continue;
        }

        if (verbose) {
            std::cout << "Downloading: " << file << "...\n";
        }

        std::string cmd;
        if (hasCurl) {
            cmd = "curl -L -o \"" + destPath + "\" \"" + url + "\" 2>/dev/null";
        } else {
            cmd = "wget -q -O \"" + destPath + "\" \"" + url + "\"";
        }

        int result = std::system(cmd.c_str());
        if (result != 0 || !Platform::FileExists(destPath) || Platform::GetFileSize(destPath) < 1000) {
            // Try alternative URLs
            bool downloaded = false;
            for (const auto& altUrl : g_alternativeUrls) {
                std::string altFullUrl = altUrl + file;

                // Security: validate alternative URL
                if (!IsUrlSafe(altFullUrl)) continue;

                if (verbose) {
                    std::cout << "  Trying alternative source...\n";
                }

                if (hasCurl) {
                    cmd = "curl -L -o \"" + destPath + "\" \"" + altFullUrl + "\" 2>/dev/null";
                } else {
                    cmd = "wget -q -O \"" + destPath + "\" \"" + altFullUrl + "\"";
                }

                result = std::system(cmd.c_str());
                if (result == 0 && Platform::FileExists(destPath) && Platform::GetFileSize(destPath) >= 1000) {
                    downloaded = true;
                    break;
                }
            }

            if (!downloaded) {
                if (verbose) {
                    std::cerr << "  Failed to download: " << file << "\n";
                }
                // Clean up partial file
                Platform::DeleteFile(destPath);
                allSuccess = false;
            }
        }

        if (verbose && Platform::FileExists(destPath)) {
            int64_t size = Platform::GetFileSize(destPath);
            std::cout << "  Done (" << (size / 1024 / 1024) << " MB)\n";
        }
    }

    if (allSuccess) {
        if (verbose) {
            std::cout << "\nAll OCR models downloaded successfully!\n";
            std::cout << "You can now use: OCR::InitOCR(\"" << targetDir << "\")\n";
        }
    } else {
        if (verbose) {
            std::cerr << "\nSome downloads failed. Please download manually:\n";
            PrintModelInstallInstructions();
        }
    }

    return allSuccess;
}

std::string GetDefaultModelDir() {
    // Check environment variable first
    const char* envDir = std::getenv("QIVISION_OCR_MODELS");
    if (envDir && Platform::DirectoryExists(envDir)) {
        return envDir;
    }

    // Check local project directory first (for development)
    if (Platform::FileExists("models/ocr/ch_PP-OCRv4_det_infer.onnx")) {
        return "models/ocr";
    }

    // Check relative to executable
    if (Platform::FileExists("./ocr_models/ch_PP-OCRv4_det_infer.onnx")) {
        return "./ocr_models";
    }

    // Default system paths
#ifdef _WIN32
    const char* appData = std::getenv("LOCALAPPDATA");
    if (appData) {
        return std::string(appData) + "/QiVision/ocr_models";
    }
    return "C:/ProgramData/QiVision/ocr_models";
#elif defined(__APPLE__)
    const char* home = std::getenv("HOME");
    if (home) {
        return std::string(home) + "/Library/Application Support/QiVision/ocr_models";
    }
    return "/usr/local/share/qivision/ocr_models";
#else
    // Linux/Unix
    const char* xdgData = std::getenv("XDG_DATA_HOME");
    if (xdgData) {
        return std::string(xdgData) + "/qivision/ocr_models";
    }
    const char* home = std::getenv("HOME");
    if (home) {
        return std::string(home) + "/.local/share/qivision/ocr_models";
    }
    return "/usr/share/qivision/ocr_models";
#endif
}

// =============================================================================
// Utility Functions
// =============================================================================

bool IsAvailable() {
#ifdef QIVISION_HAS_ONNXRUNTIME
    return true;
#else
    return false;
#endif
}

std::string GetVersion() {
#ifdef QIVISION_HAS_ONNXRUNTIME
    return "QiVision OCR v1.0 (ONNXRuntime + PaddleOCR v4)";
#else
    return "OCR backend not available (compile with -DQIVISION_BUILD_OCR=ON and ONNXRuntime)";
#endif
}

void PrintModelInstallInstructions() {
    std::string defaultDir = GetDefaultModelDir();

    std::cout << "\n";
    std::cout << "=== QiVision OCR Model Installation ===\n";
    std::cout << "\n";
    std::cout << "Required model files:\n";
    for (const auto& f : g_requiredModelFiles) {
        std::cout << "  - " << f << "\n";
    }
    std::cout << "\n";
    std::cout << "Optional model files:\n";
    for (const auto& f : g_optionalModelFiles) {
        std::cout << "  - " << f << " (angle classification)\n";
    }
    std::cout << "\n";
    std::cout << "Download from:\n";
    std::cout << "  " << g_modelDownloadUrl << "\n";
    std::cout << "\n";
    std::cout << "Alternative sources:\n";
    for (const auto& url : g_alternativeUrls) {
        std::cout << "  " << url << "\n";
    }
    std::cout << "\n";
    std::cout << "Installation:\n";
    std::cout << "  1. Download all required files from one of the sources above\n";
    std::cout << "  2. Create directory: " << defaultDir << "\n";
    std::cout << "  3. Copy all model files to that directory\n";
    std::cout << "  4. Or set QIVISION_OCR_MODELS environment variable to your model path\n";
    std::cout << "\n";
    std::cout << "Quick install (Linux/macOS):\n";
    std::cout << "  mkdir -p " << defaultDir << "\n";
    std::cout << "  cd " << defaultDir << "\n";
    for (const auto& f : g_requiredModelFiles) {
        std::cout << "  curl -LO \"" << g_modelDownloadUrl << f << "\"\n";
    }
    std::cout << "\n";
    std::cout << "Or use programmatic download:\n";
    std::cout << "  OCR::DownloadModels(\"" << defaultDir << "\");\n";
    std::cout << "\n";
}

} // namespace Qi::Vision::OCR
