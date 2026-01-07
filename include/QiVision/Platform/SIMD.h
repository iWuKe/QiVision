#pragma once

/**
 * @file SIMD.h
 * @brief SIMD capability detection and utilities
 */

#include <cstddef>
#include <cstdint>

namespace Qi::Vision::Platform {

/**
 * @brief SIMD instruction set levels
 */
enum class SIMDLevel {
    None,       ///< No SIMD
    SSE2,       ///< SSE2 (baseline x64)
    SSE4,       ///< SSE4.1/4.2
    AVX,        ///< AVX
    AVX2,       ///< AVX2 + FMA
    AVX512,     ///< AVX-512
    NEON        ///< ARM NEON
};

/**
 * @brief Get the highest supported SIMD level
 */
SIMDLevel GetSIMDLevel();

/**
 * @brief Check if SSE4.1 is supported
 */
bool HasSSE4();

/**
 * @brief Check if AVX2 is supported
 */
bool HasAVX2();

/**
 * @brief Check if AVX-512 is supported
 */
bool HasAVX512();

/**
 * @brief Check if ARM NEON is supported
 */
bool HasNEON();

/**
 * @brief Get SIMD level name as string
 */
const char* GetSIMDLevelName(SIMDLevel level);

/**
 * @brief Get current SIMD level name
 */
inline const char* GetSIMDLevelName() {
    return GetSIMDLevelName(GetSIMDLevel());
}

/**
 * @brief SIMD vector width in bytes for current architecture
 */
size_t GetSIMDWidth();

} // namespace Qi::Vision::Platform
