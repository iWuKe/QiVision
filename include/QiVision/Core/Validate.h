#pragma once

/**
 * @file Validate.h
 * @brief Unified validation utilities for QiVision SDK
 *
 * Design principles:
 * - Type restriction and channel restriction are INDEPENDENT
 * - Empty image returns false (not an error), invalid throws
 * - Consistent error message format
 *
 * Layered API:
 * - RequireImageValid(): only checks empty/valid (no type restriction)
 * - RequireImageType(): checks pixel type (UInt8, Float32, etc.)
 * - RequireImageChannels(): checks channel count (independent of type)
 * - Combined: RequireImageU8(), RequireImageU8Gray(), etc.
 */

#include <QiVision/Core/Export.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/QImage.h>

#include <string>
#include <cstdio>

namespace Qi::Vision::Validate {

// =============================================================================
// Internal Formatting
// =============================================================================

namespace Detail {

// Format double with limited precision (avoid long tails)
inline std::string FormatValue(double val) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4g", val);
    return buf;
}

inline std::string FormatValue(float val) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4g", static_cast<double>(val));
    return buf;
}

inline std::string FormatValue(int val) {
    return std::to_string(val);
}

inline std::string FormatValue(int64_t val) {
    return std::to_string(val);
}

inline std::string FormatValue(uint32_t val) {
    return std::to_string(val);
}

inline std::string FormatValue(size_t val) {
    return std::to_string(val);
}

inline const char* PixelTypeName(PixelType type) {
    switch (type) {
        case PixelType::UInt8:   return "UInt8";
        case PixelType::UInt16:  return "UInt16";
        case PixelType::Int16:   return "Int16";
        case PixelType::Float32: return "Float32";
        default:                 return "Unknown";
    }
}

} // namespace Detail

// =============================================================================
// Image Validation - Layer 1: Basic validity
// =============================================================================

/**
 * @brief Check image is allocated and valid (no type restriction)
 *
 * Use this when empty image should be a silent no-op (e.g., filtering).
 *
 * @param image Input image
 * @param funcName Function name for error messages
 * @return false if empty (caller should return empty result)
 * @throws InvalidArgumentException if image is invalid (corrupted)
 */
inline bool RequireImageValid(const QImage& image, const char* funcName) {
    if (image.Empty()) {
        return false;  // Empty = no-op, not an error
    }
    if (!image.IsValid()) {
        throw InvalidArgumentException(std::string(funcName) + ": image is invalid");
    }
    return true;
}

/**
 * @brief Check image is non-empty and valid (throws on empty)
 *
 * Use this when empty image is an error (e.g., model creation, fitting).
 *
 * @param image Input image
 * @param funcName Function name for error messages
 * @throws InvalidArgumentException if image is empty or invalid
 */
inline void RequireImageNonEmpty(const QImage& image, const char* funcName) {
    if (image.Empty()) {
        throw InvalidArgumentException(std::string(funcName) + ": image is empty");
    }
    if (!image.IsValid()) {
        throw InvalidArgumentException(std::string(funcName) + ": image is invalid");
    }
}

// =============================================================================
// Image Validation - Layer 2: Type restriction (independent)
// =============================================================================

/**
 * @brief Check image has specific pixel type
 *
 * @param image Input image (must be non-empty)
 * @param expected Expected pixel type
 * @param funcName Function name for error messages
 * @throws UnsupportedException if type mismatch
 */
inline void RequireImageType(const QImage& image, PixelType expected, const char* funcName) {
    if (image.Type() != expected) {
        throw UnsupportedException(
            std::string(funcName) + ": expected " + Detail::PixelTypeName(expected) +
            " image, got " + Detail::PixelTypeName(image.Type()));
    }
}

/**
 * @brief Check image type is one of allowed types
 *
 * @param image Input image
 * @param types Array of allowed types
 * @param count Number of types in array
 * @param funcName Function name
 * @throws UnsupportedException if type not in list
 */
inline void RequireImageTypeOneOf(const QImage& image, const PixelType* types,
                                   size_t count, const char* funcName) {
    PixelType actual = image.Type();
    for (size_t i = 0; i < count; ++i) {
        if (actual == types[i]) return;
    }

    std::string allowed;
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) allowed += ", ";
        allowed += Detail::PixelTypeName(types[i]);
    }
    throw UnsupportedException(
        std::string(funcName) + ": expected " + allowed +
        " image, got " + Detail::PixelTypeName(actual));
}

// =============================================================================
// Image Validation - Layer 2: Channel restriction (independent)
// =============================================================================

/**
 * @brief Check image channel count
 *
 * @param image Input image (must be non-empty)
 * @param allowGray Allow 1 channel
 * @param allowRgb Allow 3 channels
 * @param allowRgba Allow 4 channels
 * @param funcName Function name
 * @throws UnsupportedException if channel count not allowed
 */
inline void RequireChannelCount(const QImage& image,
                                 bool allowGray, bool allowRgb, bool allowRgba,
                                 const char* funcName) {
    int channels = image.Channels();
    bool valid = (allowGray && channels == 1) ||
                 (allowRgb && channels == 3) ||
                 (allowRgba && channels == 4);

    if (!valid) {
        std::string allowed;
        if (allowGray) allowed += "1";
        if (allowRgb) allowed += (allowed.empty() ? "3" : ", 3");
        if (allowRgba) allowed += (allowed.empty() ? "4" : ", 4");
        throw UnsupportedException(
            std::string(funcName) + ": expected " + allowed +
            " channel(s), got " + std::to_string(channels));
    }
}

/**
 * @brief Check image has exactly N channels
 */
inline void RequireChannelCountExact(const QImage& image, int expected, const char* funcName) {
    int actual = image.Channels();
    if (actual != expected) {
        throw UnsupportedException(
            std::string(funcName) + ": expected " + std::to_string(expected) +
            " channel(s), got " + std::to_string(actual));
    }
}

// =============================================================================
// Image Validation - Layer 3: Combined convenience functions
// =============================================================================

// --- Non-empty variants (throw on empty, for model creation/fitting) ---

/**
 * @brief Check image is non-empty, valid, and has specific type (throws on empty)
 */
inline void RequireImageNonEmptyType(const QImage& image, PixelType expected, const char* funcName) {
    RequireImageNonEmpty(image, funcName);
    RequireImageType(image, expected, funcName);
}

/**
 * @brief Check image is non-empty, valid, UInt8 (throws on empty)
 */
inline void RequireImageNonEmptyU8(const QImage& image, const char* funcName) {
    RequireImageNonEmpty(image, funcName);
    RequireImageType(image, PixelType::UInt8, funcName);
}

/**
 * @brief Check image is non-empty with channel count (throws on empty)
 */
inline void RequireImageNonEmptyChannels(const QImage& image, const char* funcName,
                                          bool allowGray = true, bool allowRgb = true,
                                          bool allowRgba = true) {
    RequireImageNonEmpty(image, funcName);
    RequireChannelCount(image, allowGray, allowRgb, allowRgba, funcName);
}

/**
 * @brief Check image is non-empty, UInt8, with channel count (throws on empty)
 */
inline void RequireImageNonEmptyU8Channels(const QImage& image, const char* funcName,
                                            bool allowGray = true, bool allowRgb = true,
                                            bool allowRgba = true) {
    RequireImageNonEmpty(image, funcName);
    RequireImageType(image, PixelType::UInt8, funcName);
    RequireChannelCount(image, allowGray, allowRgb, allowRgba, funcName);
}

// --- Optional empty variants (return false on empty, for filtering/search) ---

/**
 * @brief Validate UInt8 image (common case)
 * @return false if empty
 */
inline bool RequireImageU8(const QImage& image, const char* funcName) {
    if (!RequireImageValid(image, funcName)) return false;
    RequireImageType(image, PixelType::UInt8, funcName);
    return true;
}

/**
 * @brief Validate UInt8 grayscale image
 */
inline bool RequireImageU8Gray(const QImage& image, const char* funcName) {
    if (!RequireImageValid(image, funcName)) return false;
    RequireImageType(image, PixelType::UInt8, funcName);
    RequireChannelCountExact(image, 1, funcName);
    return true;
}

/**
 * @brief Validate Float32 image
 */
inline bool RequireImageFloat(const QImage& image, const char* funcName) {
    if (!RequireImageValid(image, funcName)) return false;
    RequireImageType(image, PixelType::Float32, funcName);
    return true;
}

/**
 * @brief Validate Float32 grayscale image
 */
inline bool RequireImageFloatGray(const QImage& image, const char* funcName) {
    if (!RequireImageValid(image, funcName)) return false;
    RequireImageType(image, PixelType::Float32, funcName);
    RequireChannelCountExact(image, 1, funcName);
    return true;
}

/**
 * @brief Validate grayscale image (any supported type)
 */
inline bool RequireGrayImage(const QImage& image, const char* funcName) {
    if (!RequireImageValid(image, funcName)) return false;
    RequireChannelCountExact(image, 1, funcName);
    return true;
}

/**
 * @brief Validate image with flexible channel check (any supported type)
 */
inline bool RequireImageChannels(const QImage& image, const char* funcName,
                                  bool allowGray = true, bool allowRgb = true,
                                  bool allowRgba = true) {
    if (!RequireImageValid(image, funcName)) return false;
    RequireChannelCount(image, allowGray, allowRgb, allowRgba, funcName);
    return true;
}

/**
 * @brief Validate UInt8 image with channel check
 */
inline bool RequireImageU8Channels(const QImage& image, const char* funcName,
                                    bool allowGray = true, bool allowRgb = true,
                                    bool allowRgba = true) {
    if (!RequireImageValid(image, funcName)) return false;
    RequireImageType(image, PixelType::UInt8, funcName);
    RequireChannelCount(image, allowGray, allowRgb, allowRgba, funcName);
    return true;
}

// --- Multi-type variants ---

/**
 * @brief Validate image is UInt8 or Float32 (common for algorithms supporting both)
 * @return false if empty
 */
inline bool RequireImageU8OrFloat(const QImage& image, const char* funcName) {
    if (!RequireImageValid(image, funcName)) return false;
    static const PixelType allowed[] = { PixelType::UInt8, PixelType::Float32 };
    RequireImageTypeOneOf(image, allowed, 2, funcName);
    return true;
}

/**
 * @brief Validate grayscale image is UInt8 or Float32
 */
inline bool RequireImageU8OrFloatGray(const QImage& image, const char* funcName) {
    if (!RequireImageValid(image, funcName)) return false;
    static const PixelType allowed[] = { PixelType::UInt8, PixelType::Float32 };
    RequireImageTypeOneOf(image, allowed, 2, funcName);
    RequireChannelCountExact(image, 1, funcName);
    return true;
}

// =============================================================================
// Value Range Validation
// =============================================================================

/**
 * @brief Validate value is in range [min, max]
 */
template<typename T>
inline void RequireRange(T value, T minVal, T maxVal,
                         const char* paramName, const char* funcName) {
    if (value < minVal || value > maxVal) {
        throw InvalidArgumentException(
            std::string(funcName) + ": " + paramName + " must be in [" +
            Detail::FormatValue(minVal) + ", " + Detail::FormatValue(maxVal) +
            "], got " + Detail::FormatValue(value));
    }
}

/**
 * @brief Validate value is positive (> 0)
 */
template<typename T>
inline void RequirePositive(T value, const char* paramName, const char* funcName) {
    if (value <= T(0)) {
        throw InvalidArgumentException(
            std::string(funcName) + ": " + paramName + " must be > 0, got " +
            Detail::FormatValue(value));
    }
}

/**
 * @brief Validate value is non-negative (>= 0)
 */
template<typename T>
inline void RequireNonNegative(T value, const char* paramName, const char* funcName) {
    if (value < T(0)) {
        throw InvalidArgumentException(
            std::string(funcName) + ": " + paramName + " must be >= 0, got " +
            Detail::FormatValue(value));
    }
}

/**
 * @brief Validate value is at least minimum (>= min)
 */
template<typename T>
inline void RequireMin(T value, T minVal, const char* paramName, const char* funcName) {
    if (value < minVal) {
        throw InvalidArgumentException(
            std::string(funcName) + ": " + paramName + " must be >= " +
            Detail::FormatValue(minVal) + ", got " + Detail::FormatValue(value));
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
 * Macro usage convention by return type:
 *
 * | Return type      | Macro to use                          |
 * |------------------|---------------------------------------|
 * | void             | QIVISION_REQUIRE_IMAGE_VOID(img)      |
 * | bool / int       | QIVISION_REQUIRE_IMAGE_OR(img, false) |
 * | struct / vector  | QIVISION_REQUIRE_IMAGE(img)           |
 *
 * For algorithms where empty input is an ERROR (not no-op),
 * use RequireImageNonEmpty() directly instead of macros.
 */

/**
 * Early return with custom return value (for bool/int/pointer returns)
 */
#define QIVISION_REQUIRE_IMAGE_OR(img, retval) \
    if (!::Qi::Vision::Validate::RequireImageValid(img, __func__)) return retval

/**
 * Early return {} (for struct/vector/container returns)
 */
#define QIVISION_REQUIRE_IMAGE(img) \
    QIVISION_REQUIRE_IMAGE_OR(img, {})

/**
 * Early return for void functions
 */
#define QIVISION_REQUIRE_IMAGE_VOID(img) \
    if (!::Qi::Vision::Validate::RequireImageValid(img, __func__)) return

/**
 * UInt8 variants
 */
#define QIVISION_REQUIRE_IMAGE_U8(img) \
    if (!::Qi::Vision::Validate::RequireImageU8(img, __func__)) return {}

#define QIVISION_REQUIRE_IMAGE_U8_VOID(img) \
    if (!::Qi::Vision::Validate::RequireImageU8(img, __func__)) return

#define QIVISION_REQUIRE_IMAGE_U8_OR(img, retval) \
    if (!::Qi::Vision::Validate::RequireImageU8(img, __func__)) return retval

/**
 * Float32 variants
 */
#define QIVISION_REQUIRE_IMAGE_FLOAT(img) \
    if (!::Qi::Vision::Validate::RequireImageFloat(img, __func__)) return {}

#define QIVISION_REQUIRE_IMAGE_FLOAT_VOID(img) \
    if (!::Qi::Vision::Validate::RequireImageFloat(img, __func__)) return

#define QIVISION_REQUIRE_IMAGE_FLOAT_OR(img, retval) \
    if (!::Qi::Vision::Validate::RequireImageFloat(img, __func__)) return retval

/**
 * UInt8 or Float32 variants (common for algorithms supporting both)
 */
#define QIVISION_REQUIRE_IMAGE_U8_OR_FLOAT(img) \
    if (!::Qi::Vision::Validate::RequireImageU8OrFloat(img, __func__)) return {}

#define QIVISION_REQUIRE_IMAGE_U8_OR_FLOAT_VOID(img) \
    if (!::Qi::Vision::Validate::RequireImageU8OrFloat(img, __func__)) return

/**
 * Range validation macro
 */
#define QIVISION_REQUIRE_RANGE(val, min, max) \
    ::Qi::Vision::Validate::RequireRange(val, min, max, #val, __func__)

#define QIVISION_REQUIRE_POSITIVE(val) \
    ::Qi::Vision::Validate::RequirePositive(val, #val, __func__)

#define QIVISION_REQUIRE_NON_NEGATIVE(val) \
    ::Qi::Vision::Validate::RequireNonNegative(val, #val, __func__)

} // namespace Qi::Vision::Validate
