#pragma once

/**
 * @file Validate.h
 * @brief Unified validation utilities for QiVision SDK
 *
 * Provides consistent input validation across all modules.
 * Design goals:
 * - Uniform error messages
 * - Reusable validation logic
 * - SDK-style consistency
 *
 * Usage:
 * @code
 * void MyFunction(const QImage& image, double threshold) {
 *     // Returns false for empty image (caller handles), throws for invalid
 *     if (!Validate::RequireImage(image, "MyFunction")) return;
 *
 *     // Throws if invalid
 *     Validate::RequireRange(threshold, 0.0, 1.0, "threshold", "MyFunction");
 * }
 * @endcode
 */

#include <QiVision/Core/Export.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/QImage.h>

#include <string>

namespace Qi::Vision::Validate {

// =============================================================================
// Image Validation
// =============================================================================

/**
 * @brief Validate image for processing
 *
 * @param image Input image to validate
 * @param funcName Function name for error messages
 * @return false if image is empty (caller should return empty result)
 * @throws InvalidArgumentException if image is invalid
 * @throws UnsupportedException if pixel type is not UInt8
 */
inline bool RequireImage(const QImage& image, const char* funcName) {
    if (image.Empty()) {
        return false;  // Empty = no-op, not an error
    }
    if (!image.IsValid()) {
        throw InvalidArgumentException(std::string(funcName) + ": image is invalid");
    }
    if (image.Type() != PixelType::UInt8) {
        throw UnsupportedException(std::string(funcName) + ": only UInt8 images are supported");
    }
    return true;
}

/**
 * @brief Validate image with channel count check
 *
 * @param image Input image to validate
 * @param funcName Function name for error messages
 * @param allowGray Allow 1-channel grayscale
 * @param allowRgb Allow 3-channel RGB
 * @param allowRgba Allow 4-channel RGBA
 * @return false if image is empty
 * @throws UnsupportedException if channel count not allowed
 */
inline bool RequireImageChannels(const QImage& image, const char* funcName,
                                  bool allowGray = true, bool allowRgb = true,
                                  bool allowRgba = true) {
    if (!RequireImage(image, funcName)) {
        return false;
    }

    int channels = image.Channels();
    bool valid = (allowGray && channels == 1) ||
                 (allowRgb && channels == 3) ||
                 (allowRgba && channels == 4);

    if (!valid) {
        std::string allowed;
        if (allowGray) allowed += "1";
        if (allowRgb) allowed += (allowed.empty() ? "3" : ", 3");
        if (allowRgba) allowed += (allowed.empty() ? "4" : ", 4");
        throw UnsupportedException(std::string(funcName) + ": expected " + allowed +
                                   " channel(s), got " + std::to_string(channels));
    }
    return true;
}

/**
 * @brief Validate grayscale image
 */
inline bool RequireGrayImage(const QImage& image, const char* funcName) {
    return RequireImageChannels(image, funcName, true, false, false);
}

/**
 * @brief Validate RGB image (3 channels)
 */
inline bool RequireRgbImage(const QImage& image, const char* funcName) {
    return RequireImageChannels(image, funcName, false, true, false);
}

/**
 * @brief Validate color image (3 or 4 channels)
 */
inline bool RequireColorImage(const QImage& image, const char* funcName) {
    return RequireImageChannels(image, funcName, false, true, true);
}

// =============================================================================
// Value Range Validation
// =============================================================================

/**
 * @brief Validate value is in range [min, max]
 * @throws InvalidArgumentException if out of range
 */
template<typename T>
inline void RequireRange(T value, T minVal, T maxVal,
                         const char* paramName, const char* funcName) {
    if (value < minVal || value > maxVal) {
        throw InvalidArgumentException(
            std::string(funcName) + ": " + paramName + " must be in [" +
            std::to_string(minVal) + ", " + std::to_string(maxVal) + "], got " +
            std::to_string(value));
    }
}

/**
 * @brief Validate value is positive (> 0)
 * @throws InvalidArgumentException if not positive
 */
template<typename T>
inline void RequirePositive(T value, const char* paramName, const char* funcName) {
    if (value <= T(0)) {
        throw InvalidArgumentException(
            std::string(funcName) + ": " + paramName + " must be > 0, got " +
            std::to_string(value));
    }
}

/**
 * @brief Validate value is non-negative (>= 0)
 * @throws InvalidArgumentException if negative
 */
template<typename T>
inline void RequireNonNegative(T value, const char* paramName, const char* funcName) {
    if (value < T(0)) {
        throw InvalidArgumentException(
            std::string(funcName) + ": " + paramName + " must be >= 0, got " +
            std::to_string(value));
    }
}

/**
 * @brief Validate value is at least minimum (>= min)
 * @throws InvalidArgumentException if below minimum
 */
template<typename T>
inline void RequireMin(T value, T minVal, const char* paramName, const char* funcName) {
    if (value < minVal) {
        throw InvalidArgumentException(
            std::string(funcName) + ": " + paramName + " must be >= " +
            std::to_string(minVal) + ", got " + std::to_string(value));
    }
}

// =============================================================================
// GPU/Thread Validation
// =============================================================================

/**
 * @brief Validate thread count (>= 1)
 */
inline void RequireThreadCount(int numThread, const char* funcName) {
    if (numThread < 1) {
        throw InvalidArgumentException(
            std::string(funcName) + ": numThread must be >= 1, got " +
            std::to_string(numThread));
    }
}

/**
 * @brief Validate GPU index (-1 for CPU, >= 0 for GPU)
 */
inline void RequireGpuIndex(int gpuIndex, const char* funcName) {
    if (gpuIndex < -1) {
        throw InvalidArgumentException(
            std::string(funcName) + ": gpuIndex must be >= -1, got " +
            std::to_string(gpuIndex));
    }
}

// =============================================================================
// Convenience Macros
// =============================================================================

/**
 * Macro for quick image validation with early return
 * Usage: QIVISION_REQUIRE_IMAGE(image); // uses __func__
 */
#define QIVISION_REQUIRE_IMAGE(img) \
    if (!::Qi::Vision::Validate::RequireImage(img, __func__)) return {}

/**
 * Macro for image validation with channels
 */
#define QIVISION_REQUIRE_IMAGE_CHANNELS(img, gray, rgb, rgba) \
    if (!::Qi::Vision::Validate::RequireImageChannels(img, __func__, gray, rgb, rgba)) return {}

/**
 * Macro for range validation
 */
#define QIVISION_REQUIRE_RANGE(val, min, max) \
    ::Qi::Vision::Validate::RequireRange(val, min, max, #val, __func__)

/**
 * Macro for positive value validation
 */
#define QIVISION_REQUIRE_POSITIVE(val) \
    ::Qi::Vision::Validate::RequirePositive(val, #val, __func__)

/**
 * Macro for non-negative value validation
 */
#define QIVISION_REQUIRE_NON_NEGATIVE(val) \
    ::Qi::Vision::Validate::RequireNonNegative(val, #val, __func__)

} // namespace Qi::Vision::Validate
