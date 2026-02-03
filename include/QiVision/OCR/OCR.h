#pragma once

/**
 * @file OCR.h
 * @brief Optical Character Recognition (ONNXRuntime + PaddleOCR models)
 *
 * Supports:
 * - Chinese and English text recognition
 * - Text detection (localization)
 * - Text angle classification
 * - High accuracy with PaddleOCR v4 models
 *
 * Dependencies:
 * - ONNXRuntime only (no OpenCV required)
 * - All image preprocessing uses QiVision's native APIs
 *
 * Models:
 * - ch_PP-OCRv4_det_infer.onnx (text detection)
 * - ch_ppocr_mobile_v2.0_cls_infer.onnx (angle classification)
 * - ch_PP-OCRv4_rec_infer.onnx (text recognition)
 * - ppocr_keys_v1.txt (character dictionary)
 *
 * Enable: cmake -DQIVISION_BUILD_OCR=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime
 */

#include <QiVision/Core/Export.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Types.h>

#include <memory>
#include <string>
#include <vector>

namespace Qi::Vision::OCR {

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Single text block detection and recognition result
 */
struct QIVISION_API TextBlock {
    std::string text;                   ///< Recognized text
    double confidence = 0.0;            ///< Recognition confidence (0-1)
    std::vector<Point2d> corners;       ///< Four corner points (clockwise)
    double boxScore = 0.0;              ///< Detection box score
    int angleIndex = 0;                 ///< Angle class (0=0°, 1=180°)
    double angleScore = 0.0;            ///< Angle classification score

    /// Get bounding rectangle
    Rect2i BoundingRect() const;

    /// Get center position
    Point2d Center() const;
};

/**
 * @brief Complete OCR result for an image
 */
struct QIVISION_API OCRResult {
    std::vector<TextBlock> textBlocks;  ///< All detected text blocks
    std::string fullText;               ///< Concatenated text (all blocks)
    double detectTime = 0.0;            ///< Detection time (ms)
    double recognizeTime = 0.0;         ///< Recognition time (ms)
    double totalTime = 0.0;             ///< Total processing time (ms)

    /// Check if any text was found
    bool Empty() const { return textBlocks.empty(); }

    /// Get number of text blocks
    size_t Size() const { return textBlocks.size(); }
};

/**
 * @brief OCR parameters
 */
struct QIVISION_API OCRParams {
    // Detection parameters
    int padding = 50;                   ///< Image padding for detection
    int maxSideLen = 1024;              ///< Max image side length (resize if larger)
    double boxScoreThresh = 0.5;        ///< Box score threshold
    double boxThresh = 0.3;             ///< Box threshold
    double unClipRatio = 1.6;           ///< Unclip ratio for text box expansion

    // Recognition parameters
    bool doAngleClassify = true;        ///< Enable angle classification
    bool mostAngle = true;              ///< Use majority angle

    // Performance
    int numThread = 4;                  ///< Number of threads
    int gpuIndex = -1;                  ///< GPU index (-1 for CPU)

    // Debug
    bool debug = false;                 ///< Print debug statistics

    /// Default parameters
    static OCRParams Default() { return OCRParams(); }

    /// Fast mode (lower accuracy, faster speed)
    static OCRParams Fast() {
        OCRParams p;
        p.maxSideLen = 640;
        p.doAngleClassify = false;
        return p;
    }

    /// Accurate mode (higher accuracy, slower)
    static OCRParams Accurate() {
        OCRParams p;
        p.maxSideLen = 2048;
        p.boxScoreThresh = 0.3;
        p.boxThresh = 0.2;
        return p;
    }
};

// =============================================================================
// OCR Model Class
// =============================================================================

/**
 * @brief OCR model handle (manages loaded models)
 */
class QIVISION_API OCRModel {
public:
    OCRModel();
    ~OCRModel();

    // Non-copyable, movable
    OCRModel(const OCRModel&) = delete;
    OCRModel& operator=(const OCRModel&) = delete;
    OCRModel(OCRModel&& other) noexcept;
    OCRModel& operator=(OCRModel&& other) noexcept;

    /**
     * @brief Initialize OCR models
     *
     * @param modelDir Directory containing model files
     * @param detModel Detection model filename (default: ch_PP-OCRv4_det_infer.onnx)
     * @param clsModel Classification model filename (default: ch_ppocr_mobile_v2.0_cls_infer.onnx)
     * @param recModel Recognition model filename (default: ch_PP-OCRv4_rec_infer.onnx)
     * @param keysFile Keys file for recognition (default: ppocr_keys_v1.txt)
     * @return true if initialization successful
     */
    bool Init(const std::string& modelDir,
              const std::string& detModel = "ch_PP-OCRv4_det_infer.onnx",
              const std::string& clsModel = "ch_ppocr_mobile_v2.0_cls_infer.onnx",
              const std::string& recModel = "ch_PP-OCRv4_rec_infer.onnx",
              const std::string& keysFile = "ppocr_keys_v1.txt");

    /**
     * @brief Initialize with default models (auto-download if needed)
     * @return true if initialization successful
     */
    bool InitDefault();

    /**
     * @brief Check if model is initialized
     */
    bool IsValid() const;

    /**
     * @brief Set number of threads
     */
    void SetNumThread(int numThread);

    /**
     * @brief Set GPU index (-1 for CPU)
     */
    void SetGpuIndex(int gpuIndex);

    /**
     * @brief Recognize text in image
     *
     * @param image Input image (grayscale or color)
     * @param params OCR parameters
     * @return OCR result with detected text blocks
     */
    OCRResult Recognize(const QImage& image, const OCRParams& params = OCRParams::Default()) const;

    /**
     * @brief Recognize text in image file
     *
     * @param imagePath Path to image file
     * @param params OCR parameters
     * @return OCR result
     */
    OCRResult Recognize(const std::string& imagePath, const OCRParams& params = OCRParams::Default()) const;

    /**
     * @brief Recognize single text line (skip detection)
     *
     * @param image Pre-cropped text line image
     * @param confidence Output confidence score
     * @return Recognized text
     */
    std::string RecognizeLine(const QImage& image, double& confidence) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Convenience Functions (using global model)
// =============================================================================

/**
 * @brief Initialize global OCR model
 *
 * @param modelDir Model directory path
 * @param gpuIndex GPU device index (-1 for CPU, >=0 for GPU)
 * @return true if successful
 */
QIVISION_API bool InitOCR(const std::string& modelDir, int gpuIndex = -1);

/**
 * @brief Initialize global OCR model with default models
 * @param gpuIndex GPU device index (-1 for CPU, >=0 for GPU)
 * @return true if successful
 */
QIVISION_API bool InitOCRDefault(int gpuIndex = -1);

/**
 * @brief Release global OCR model
 */
QIVISION_API void ReleaseOCR();

/**
 * @brief Check if global OCR model is initialized
 */
QIVISION_API bool IsOCRReady();

/**
 * @brief Recognize text using global model
 *
 * @param image Input image
 * @param params OCR parameters
 * @return OCR result
 */
QIVISION_API OCRResult RecognizeText(const QImage& image,
                                      const OCRParams& params = OCRParams::Default());

/**
 * @brief Recognize text from image file using global model
 */
QIVISION_API OCRResult RecognizeText(const std::string& imagePath,
                                      const OCRParams& params = OCRParams::Default());

/**
 * @brief Simple text recognition (returns string only)
 *
 * @param image Input image
 * @return Recognized text (empty if failed)
 */
QIVISION_API std::string ReadText(const QImage& image);

/**
 * @brief Simple text recognition from file
 */
QIVISION_API std::string ReadText(const std::string& imagePath);

/**
 * @brief Recognize single text line (skip detection)
 *
 * Use this for pre-cropped text line images. Skips the detection step
 * and directly runs recognition on the entire image.
 *
 * @param image Input image (should contain a single line of text)
 * @param confidence Output confidence score
 * @return Recognized text
 */
QIVISION_API std::string RecognizeLine(const QImage& image, double& confidence);

// =============================================================================
// Model Management
// =============================================================================

/**
 * @brief Model status information
 */
struct QIVISION_API ModelStatus {
    bool allRequired = false;          ///< All required models present
    bool allOptional = false;          ///< All optional models present
    std::vector<std::string> missing;  ///< Missing required files
    std::vector<std::string> found;    ///< Found files
    std::string modelDir;              ///< Checked directory

    /// Check if models are ready for use
    bool IsReady() const { return allRequired; }

    /// Get human-readable status message
    std::string GetMessage() const;
};

/**
 * @brief Check if OCR models are present in directory
 *
 * @param modelDir Directory to check (empty = default)
 * @return Model status with missing/found files
 */
QIVISION_API ModelStatus CheckModels(const std::string& modelDir = "");

/**
 * @brief Get list of required model files
 */
QIVISION_API std::vector<std::string> GetRequiredModelFiles();

/**
 * @brief Get list of optional model files
 */
QIVISION_API std::vector<std::string> GetOptionalModelFiles();

/**
 * @brief Get model download URL for manual download
 */
QIVISION_API std::string GetModelDownloadUrl();

/**
 * @brief Download models to specified directory
 *
 * Uses curl/wget if available on the system.
 * If download tools are not available, prints instructions for manual download.
 *
 * @param modelDir Target directory (created if not exists)
 * @param verbose Print progress messages
 * @return true if all required models downloaded successfully
 */
QIVISION_API bool DownloadModels(const std::string& modelDir, bool verbose = true);

/**
 * @brief Get default model directory path
 *
 * Search order:
 * 1. QIVISION_OCR_MODELS environment variable
 * 2. ./models/ocr (development)
 * 3. Platform-specific system path
 */
QIVISION_API std::string GetDefaultModelDir();

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Check if ONNXRuntime backend is available
 * @return true if OCR backend is available
 */
QIVISION_API bool IsAvailable();

/**
 * @brief Get OCR backend version
 */
QIVISION_API std::string GetVersion();

/**
 * @brief Print installation instructions for OCR models
 *
 * Outputs detailed instructions on how to download and install OCR models.
 */
QIVISION_API void PrintModelInstallInstructions();

} // namespace Qi::Vision::OCR
