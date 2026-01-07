#include <gtest/gtest.h>
#include <QiVision/Platform/SIMD.h>
#include <QiVision/Platform/Memory.h>

using namespace Qi::Vision::Platform;

// =============================================================================
// SIMD Detection Tests
// =============================================================================

TEST(SIMDTest, GetLevel) {
    SIMDLevel level = GetSIMDLevel();
    // Should return a valid level
    EXPECT_GE(static_cast<int>(level), static_cast<int>(SIMDLevel::None));
    EXPECT_LE(static_cast<int>(level), static_cast<int>(SIMDLevel::NEON));
}

TEST(SIMDTest, GetLevelName) {
    const char* name = GetSIMDLevelName();
    EXPECT_NE(name, nullptr);
    EXPECT_GT(strlen(name), 0u);

    // Print for info
    std::cout << "Detected SIMD level: " << name << std::endl;
}

TEST(SIMDTest, GetSIMDWidth) {
    size_t width = GetSIMDWidth();
    // Width should be 1, 16, 32, or 64
    EXPECT_TRUE(width == 1 || width == 16 || width == 32 || width == 64);

    std::cout << "SIMD width: " << width << " bytes" << std::endl;
}

TEST(SIMDTest, ConsistentFlags) {
    SIMDLevel level = GetSIMDLevel();

    // Higher levels imply lower level support (on x86)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (level >= SIMDLevel::AVX2) {
        EXPECT_TRUE(HasAVX2());
    }
    if (level >= SIMDLevel::SSE4) {
        EXPECT_TRUE(HasSSE4());
    }
#endif
}

// =============================================================================
// Memory Alignment Tests
// =============================================================================

TEST(MemoryTest, AlignedAlloc) {
    void* ptr = AlignedAlloc(1024, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(IsAligned(ptr, 64));
    AlignedFree(ptr);
}

TEST(MemoryTest, AlignedAllocZeroSize) {
    void* ptr = AlignedAlloc(0);
    EXPECT_EQ(ptr, nullptr);
}

TEST(MemoryTest, AlignedSize) {
    EXPECT_EQ(AlignedSize(1, 64), 64u);
    EXPECT_EQ(AlignedSize(64, 64), 64u);
    EXPECT_EQ(AlignedSize(65, 64), 128u);
    EXPECT_EQ(AlignedSize(100, 64), 128u);
}

TEST(MemoryTest, IsAligned) {
    void* ptr = AlignedAlloc(256, 64);
    ASSERT_NE(ptr, nullptr);

    EXPECT_TRUE(IsAligned(ptr, 64));
    EXPECT_TRUE(IsAligned(ptr, 32));
    EXPECT_TRUE(IsAligned(ptr, 16));
    EXPECT_TRUE(IsAligned(ptr, 8));

    AlignedFree(ptr);
}

TEST(MemoryTest, AllocateAligned) {
    auto ptr = AllocateAligned<float>(100);
    ASSERT_NE(ptr.get(), nullptr);
    EXPECT_TRUE(IsAligned(ptr.get(), 64));

    // Can write to it
    for (size_t i = 0; i < 100; ++i) {
        ptr[i] = static_cast<float>(i);
    }

    EXPECT_FLOAT_EQ(ptr[50], 50.0f);
}

TEST(MemoryTest, LargeAllocation) {
    // Allocate 100MB aligned
    size_t size = 100 * 1024 * 1024;
    void* ptr = AlignedAlloc(size, 64);

    if (ptr != nullptr) {  // May fail on low memory systems
        EXPECT_TRUE(IsAligned(ptr, 64));
        AlignedFree(ptr);
    }
}
