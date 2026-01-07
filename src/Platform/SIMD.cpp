#include <QiVision/Platform/SIMD.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace Qi::Vision::Platform {

namespace {

// CPU feature bits
struct CPUFeatures {
    bool sse2 = false;
    bool sse4_1 = false;
    bool avx = false;
    bool avx2 = false;
    bool avx512f = false;
    bool neon = false;
    bool initialized = false;

    void Detect() {
        if (initialized) return;

#if defined(__ARM_NEON) || defined(__aarch64__)
        neon = true;
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        int cpuInfo[4] = {0};

#ifdef _MSC_VER
        __cpuid(cpuInfo, 1);
#else
        __cpuid(1, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
#endif

        sse2 = (cpuInfo[3] & (1 << 26)) != 0;
        sse4_1 = (cpuInfo[2] & (1 << 19)) != 0;
        avx = (cpuInfo[2] & (1 << 28)) != 0;

        // Check for AVX2
#ifdef _MSC_VER
        __cpuidex(cpuInfo, 7, 0);
#else
        __cpuid_count(7, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
#endif

        avx2 = (cpuInfo[1] & (1 << 5)) != 0;
        avx512f = (cpuInfo[1] & (1 << 16)) != 0;
#endif

        initialized = true;
    }
};

CPUFeatures& GetCPUFeatures() {
    static CPUFeatures features;
    features.Detect();
    return features;
}

} // anonymous namespace

SIMDLevel GetSIMDLevel() {
    const auto& f = GetCPUFeatures();

    if (f.avx512f) return SIMDLevel::AVX512;
    if (f.avx2) return SIMDLevel::AVX2;
    if (f.avx) return SIMDLevel::AVX;
    if (f.sse4_1) return SIMDLevel::SSE4;
    if (f.sse2) return SIMDLevel::SSE2;
    if (f.neon) return SIMDLevel::NEON;

    return SIMDLevel::None;
}

bool HasSSE4() {
    return GetCPUFeatures().sse4_1;
}

bool HasAVX2() {
    return GetCPUFeatures().avx2;
}

bool HasAVX512() {
    return GetCPUFeatures().avx512f;
}

bool HasNEON() {
#if defined(__ARM_NEON) || defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

const char* GetSIMDLevelName(SIMDLevel level) {
    switch (level) {
        case SIMDLevel::None: return "None";
        case SIMDLevel::SSE2: return "SSE2";
        case SIMDLevel::SSE4: return "SSE4";
        case SIMDLevel::AVX: return "AVX";
        case SIMDLevel::AVX2: return "AVX2";
        case SIMDLevel::AVX512: return "AVX-512";
        case SIMDLevel::NEON: return "NEON";
    }
    return "Unknown";
}

size_t GetSIMDWidth() {
    switch (GetSIMDLevel()) {
        case SIMDLevel::AVX512: return 64;
        case SIMDLevel::AVX2:
        case SIMDLevel::AVX: return 32;
        case SIMDLevel::SSE4:
        case SIMDLevel::SSE2: return 16;
        case SIMDLevel::NEON: return 16;
        default: return 1;
    }
}

} // namespace Qi::Vision::Platform
