#pragma once

/**
 * @file Memory.h
 * @brief Memory management utilities
 */

#include <QiVision/Core/Constants.h>

#include <cstddef>
#include <memory>

namespace Qi::Vision::Platform {

/**
 * @brief Allocate aligned memory
 * @param size Size in bytes
 * @param alignment Alignment in bytes (default: 64 for AVX512)
 * @return Pointer to aligned memory, or nullptr on failure
 */
void* AlignedAlloc(size_t size, size_t alignment = MEMORY_ALIGNMENT);

/**
 * @brief Free aligned memory
 * @param ptr Pointer previously returned by AlignedAlloc
 */
void AlignedFree(void* ptr);

/**
 * @brief Check if pointer is aligned
 * @param ptr Pointer to check
 * @param alignment Required alignment
 * @return true if aligned
 */
inline bool IsAligned(const void* ptr, size_t alignment = MEMORY_ALIGNMENT) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

/**
 * @brief Calculate aligned size (round up to alignment boundary)
 */
inline size_t AlignedSize(size_t size, size_t alignment = MEMORY_ALIGNMENT) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Custom deleter for aligned memory
 */
struct AlignedDeleter {
    void operator()(void* ptr) const {
        AlignedFree(ptr);
    }
};

/**
 * @brief Smart pointer type for aligned memory
 */
template<typename T>
using AlignedPtr = std::unique_ptr<T[], AlignedDeleter>;

/**
 * @brief Allocate aligned array
 */
template<typename T>
AlignedPtr<T> AllocateAligned(size_t count, size_t alignment = MEMORY_ALIGNMENT) {
    T* ptr = static_cast<T*>(AlignedAlloc(count * sizeof(T), alignment));
    return AlignedPtr<T>(ptr);
}

} // namespace Qi::Vision::Platform
